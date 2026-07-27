# ==============================================================================
# S1 Feature Engine — Compute S1 derived features from S0 parquets.
#
# INPUT:  /data_storage/s0_features/s0_features_{asset}_{date}_{hh}.parquet
# OUTPUT: /data_storage/s1_features/s1_features_{asset}_{date}_{hh}.parquet
#
# All S1 features are computed from S0 feature columns.
# No raw data access needed — only the S0 feature parquet.
#
# COLUMN RETENTION:
#   The output retains ALL S0 columns and adds S1 columns on top.
#   Each stage only extends the DataFrame; previous-stage columns are kept.
#   The S1 output file supersedes the S0 input (same data + S1 extensions).
#
# CONTEXT WINDOW:
#   Hourly files are 3600 rows (1 per second). Features with large windows
#   (e.g. 3600s rolling, 3600s forward shift) need data from adjacent hours.
#   The engine loads up to 1 hour before (lookback) and 1 hour after
#   (lookahead) as context. S1 computation runs on the full concatenated
#   DataFrame, then slices back to the target hour rows before saving.
#   This eliminates warmup NaN for all window sizes ≤ 3600s.
#
# DISPATCH:
#   _dispatch_operator reads spec.operator, collects dep columns from df,
#   and delegates to the appropriate computation function.
#
# TOPOLOGICAL SORT:
#   Kahn's BFS algorithm. Intra-S1 dependencies are detected by checking
#   whether a dep name matches another S1 feature name. Cycles raise
#   ValueError with the list of cycle members.
#
# DEP-ORDER VALIDATION:
#   Before computing operators that assume bid-first / ask-second ordering
#   (queue_imbalance, queue_pressure, queue_pressure_log, book_asymmetry),
#   the engine validates that dep[0] contains "bid" and dep[1] contains "ask".
#
# REGISTRY VALIDATION:
#   Before the compute loop, every spec.operator is checked against the
#   S1_OPERATORS registry. Unknown operators raise ValueError. Arity
#   mismatches raise ValueError.
#
# ATOMIC WRITE:
#   Output parquet is written to a temporary file, then atomically replaced
#   via os.replace to prevent partial writes on crash.
#
# CONTEXT JOINS:
#   Four context-join functions inject pre-computed columns onto every 1s
#   bucket of the combined S0 DataFrame:
#     _join_ohlc              — daily high/low/open/close + prev-day levels
#     _join_weekly            — ISO-week open/high/low + Monday + prev-week
#     _join_monthly           — calendar month open/high/low + prev-month
#     _join_volume_profile    — POC/VAH/VAL + poc_migration for 60m/240m/1d
#
#   Each join uses the appropriate key:
#     OHLC    → merged on date_str (broadcast one row/day)
#     Weekly  → merged on bucket_dt_utc (1s-grid join)
#     Monthly → merged on bucket_dt_utc
#     VP      → merged on bucket_dt_utc
#
# ==============================================================================

from __future__ import annotations
from common.paths import DATA_ROOT

import argparse
import os
import tempfile
import time
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from etl.operators.s1_operators import S1_OPERATORS
from etl.spec import FeatureSpec, Dep
from etl.spec.s1.s1_absorption import S1_ABSORPTION_FEATURES
from etl.spec.s1.s1_activity import S1_ACTIVITY_FEATURES
from etl.spec.s1.s1_aggression import S1_AGGRESSION_FEATURES
from etl.spec.s1.s1_bookshape import S1_BOOKSHAPE_FEATURES
from etl.spec.s1.s1_cross_market import S1_CROSS_MARKET_FEATURES
from etl.spec.s1.s1_imbalance import S1_IMBALANCE_FEATURES
from etl.spec.s1.s1_liquidity_events import S1_LIQUIDITY_EVENTS_FEATURES
from etl.spec.s1.s1_meta import S1_META_FEATURES
from etl.spec.s1.s1_normalization import S1_NORMALIZATION_FEATURES
from etl.spec.s1.s1_pressure import S1_PRESSURE_FEATURES
from etl.spec.s1.s1_price import S1_PRICE_FEATURES
from etl.spec.s1.s1_range import S1_RANGE_FEATURES
from etl.spec.s1.s1_returns import S1_RETURNS_FEATURES

# Phase 4 additions — new spec modules.
from etl.spec.s1.s1_forward_excursion import S1_FORWARD_EXCURSION_FEATURES
from etl.spec.s1.s1_forward_rv import S1_FORWARD_RV_FEATURES
from etl.spec.s1.s1_trend import S1_TREND_FEATURES
from etl.spec.s1.s1_session_levels import S1_SESSION_LEVELS_FEATURES
from etl.spec.s1.s1_level_events import S1_LEVEL_EVENTS_FEATURES
from etl.spec.s1.s1_volume_profile import S1_VOLUME_PROFILE_FEATURES


PARQUET_COMPRESSION = "zstd"
EPS = 1e-12

_ENGINE_DIR = Path(__file__).resolve().parent
_DEFAULT_S0_DIR = DATA_ROOT / "s0_features"
_DEFAULT_OUT_DIR = DATA_ROOT / "s1_features"


# =============================================================================
# Feature Assembly
# =============================================================================

ALL_S1_FEATURES: List[FeatureSpec] = (
    list(S1_ABSORPTION_FEATURES) +
    list(S1_ACTIVITY_FEATURES) +
    list(S1_AGGRESSION_FEATURES) +
    list(S1_BOOKSHAPE_FEATURES) +
    list(S1_CROSS_MARKET_FEATURES) +
    list(S1_IMBALANCE_FEATURES) +
    list(S1_LIQUIDITY_EVENTS_FEATURES) +
    list(S1_META_FEATURES) +
    list(S1_NORMALIZATION_FEATURES) +
    list(S1_PRESSURE_FEATURES) +
    list(S1_PRICE_FEATURES) +
    list(S1_RANGE_FEATURES) +
    list(S1_RETURNS_FEATURES) +
    list(S1_FORWARD_EXCURSION_FEATURES) +
    list(S1_FORWARD_RV_FEATURES) +
    list(S1_TREND_FEATURES) +
    list(S1_SESSION_LEVELS_FEATURES) +
    list(S1_LEVEL_EVENTS_FEATURES) +
    list(S1_VOLUME_PROFILE_FEATURES)
)


# =============================================================================
# Utilities
# =============================================================================

def _log(enabled: bool, msg: str) -> None:
    if enabled:
        print(f"[{pd.Timestamp.utcnow().isoformat()}] [S1_FEATURE_ENGINE] {msg}")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _read_parquet(path: str) -> pd.DataFrame:
    return pq.read_table(path).to_pandas()


def _safe_int(val: Any, default: int = 1) -> int:
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


# =============================================================================
# Topological Sort — Kahn's BFS
# =============================================================================

def _toposort_specs(specs: List[FeatureSpec]) -> List[FeatureSpec]:
    """
    Topologically sort S1 feature specs so that intra-S1 dependencies are
    computed before the features that depend on them.

    Node = FeatureSpec.name
    Edge = dep.name -> spec.name, if dep.name is also an S1 feature name.

    Deterministic: ready queue sorted by (feature_id, name).
    Raises ValueError with cycle member names if a cycle is detected.
    """
    name_to_idx: Dict[str, int] = {s.name: i for i, s in enumerate(specs)}

    in_degree = [0] * len(specs)
    dependents: Dict[int, List[int]] = defaultdict(list)

    for i, s in enumerate(specs):
        for dep in s.depends_on:
            if dep.name in name_to_idx and dep.name != s.name:
                dep_idx = name_to_idx[dep.name]
                in_degree[i] += 1
                dependents[dep_idx].append(i)

    queue = [i for i in range(len(specs)) if in_degree[i] == 0]
    sorted_indices: List[int] = []

    while queue:
        queue.sort(key=lambda idx: (specs[idx].feature_id or 0, specs[idx].name))
        current = queue.pop(0)
        sorted_indices.append(current)
        for dep_idx in dependents[current]:
            in_degree[dep_idx] -= 1
            if in_degree[dep_idx] == 0:
                queue.append(dep_idx)

    if len(sorted_indices) != len(specs):
        remaining = [specs[i].name for i in range(len(specs))
                     if i not in set(sorted_indices)]
        raise ValueError(
            f"Topological sort failed: cycle detected among {len(remaining)} "
            f"specs: {remaining}"
        )

    return [specs[i] for i in sorted_indices]


# =============================================================================
# Validation
# =============================================================================

# Operators that require bid-first, ask-second dep ordering.
_BID_ASK_ORDER_OPS = frozenset({
    "l2.queue_imbalance_1s",
    "l2.queue_pressure",
    "l2.queue_pressure_log_1s",
    "l2.book_asymmetry",
})


def _validate_registry(specs: List[FeatureSpec], registry: dict) -> None:
    """
    Pre-compute-loop validation:
      - spec.operator must exist in registry
      - arity (len(depends_on)) must match registry.n_input_cols (if > 0)
    Raises ValueError on first mismatch.
    """
    for spec in specs:
        op = spec.operator
        if op not in registry:
            raise ValueError(
                f"Registry validation failed: unknown operator '{op}' "
                f"used by feature '{spec.name}' (id={spec.feature_id})"
            )
        reg_entry = registry[op]
        expected_arity = reg_entry.n_input_cols
        actual_arity = len(spec.depends_on)
        if expected_arity > 0 and actual_arity != expected_arity:
            raise ValueError(
                f"Registry validation failed: arity mismatch for '{spec.name}' "
                f"(id={spec.feature_id}): operator '{op}' expects "
                f"{expected_arity} inputs, spec has {actual_arity} deps"
            )


def _validate_dep_order(spec: FeatureSpec) -> None:
    """
    For operators that assume bid-first / ask-second dep ordering,
    verify that dep[0] contains 'bid' and dep[1] contains 'ask'.
    Raises ValueError on violation.
    """
    if spec.operator not in _BID_ASK_ORDER_OPS:
        return
    deps = spec.depends_on
    if len(deps) < 2:
        return
    d0 = deps[0].name.lower()
    d1 = deps[1].name.lower()
    if "bid" not in d0 or "ask" not in d1:
        raise ValueError(
            f"Dep-order validation failed for '{spec.name}' "
            f"(operator={spec.operator}): expected dep[0] to contain 'bid' "
            f"and dep[1] to contain 'ask', got dep[0]='{deps[0].name}', "
            f"dep[1]='{deps[1].name}'"
        )


# =============================================================================
# S1 Feature Engine
# =============================================================================

class S1FeatureEngine:
    """
    Compute S1 features from an S0 feature DataFrame.

    The engine reads the S0 feature parquet (wide format, 1s buckets),
    applies each S1 operator using the appropriate S0 columns as inputs,
    and produces a wide S1 feature DataFrame.

    Operator dispatch uses S1_OPERATORS registry for validation.
    Actual computation is in the _dispatch_operator method.

    [STATEFUL-EMA]
    The engine optionally accepts a dict of ema_history mapping
    {feature_name -> pd.Series} where the Series is indexed by bucket_dt_utc
    and contains EMA values for the last MAX_LOOKBACK seconds (4 hours).

    Usage by `derived.ema` operator:
      - For pre-target rows (= prev hour context), EMA values are HYDRATED
        from history rather than recomputed (which would drift). The recursive
        EMA loop starts at the target_slice boundary using the last history
        value as init.
    Usage by `derived.ema_slope_bps` operator:
      - For shifts that exceed the in-frame lookback, ema_prev[t] is looked up
        in the *extended* series (history + current target) by bucket_dt_utc.

    The full target-slice EMA values plus prior history are exposed via
    `get_ema_target_history()` and persisted to disk by
    `build_s1_features_for_hour` for the next hour to consume.
    """

    def __init__(self, verbose: bool = True,
                 ema_history: Optional[dict] = None):
        self.verbose = verbose
        self._op_registry = S1_OPERATORS
        # ema_history[name] -> pd.Series indexed by bucket_dt_utc with EMA values
        # for the last MAX_LOOKBACK seconds (loaded from prev hour's state file).
        self._ema_history: dict = dict(ema_history) if ema_history else {}
        # Captured during compute: feature_name -> pd.Series of target-hour EMA
        # values (indexed by bucket_dt_utc). Saved to disk after compute.
        self._ema_target_values: dict = {}
        # Working buffer: feature_name -> extended pd.Series (history + new
        # target slice), used by the slope operator for back-shift lookups.
        self._ema_extended: dict = {}
        # Will be set externally before compute_all if engine should know
        # which row range corresponds to the target hour for state capture.
        self._target_slice: Optional[Tuple[int, int]] = None

    def get_ema_target_history(self) -> dict:
        """Return {feature_name -> pd.Series} of target-hour EMA values
        (indexed by bucket_dt_utc). Used by the persistence layer to extend
        the rolling history for the next hour's run."""
        return dict(self._ema_target_values)

    def compute_all(
        self,
        s0_df: pd.DataFrame,
        specs: Optional[List[FeatureSpec]] = None,
        features_filter: Optional[List[str]] = None,
        context_slice: Optional[Tuple[int, int]] = None,
    ) -> pd.DataFrame:
        """
        Compute all S1 features from S0 DataFrame.

        Args:
            s0_df:           S0 features DataFrame (wide, 1s buckets).
                             May include context rows from adjacent hours.
            specs:           Feature specs to compute (default: ALL_S1_FEATURES).
            features_filter: Optional list of feature names to compute.
            context_slice:   Optional (start_idx, end_idx) tuple indicating the
                             target hour's rows within s0_df. If provided, the
                             returned DataFrame is sliced to only these rows
                             after computation completes. Context rows are used
                             for warmup / lookahead but not included in output.

        Returns:
            DataFrame with S1 feature columns added to S0 base.
            If context_slice is provided, only the target rows are returned.
        """
        if specs is None:
            specs = ALL_S1_FEATURES

        if features_filter:
            keep = set(features_filter)
            specs = [s for s in specs if s.name in keep]

        # [STATEFUL-EMA 2026-04-26] Tell engine which row range = target hour
        # so the EMA operator captures its final state at the END of the
        # target hour (not the end of the next-hour context window).
        self._target_slice = context_slice

        # --- Registry validation (before compute loop) ---
        # Comment: This guarantees we fail fast if a spec references an operator
        # that doesn't exist, or if the spec's dependency arity is wrong.
        _validate_registry(specs, self._op_registry)

        # --- Topological sort ---
        # Comment: S1 features can depend on other S1 features. We sort the specs
        # so upstream features are computed before downstream ones.
        sorted_specs = _toposort_specs(specs)

        _log(self.verbose,
             f"Computing {len(sorted_specs)} S1 features (toposorted)")

        t0 = time.time()
        result = s0_df.copy()
        computed, errors = 0, 0

        for spec in sorted_specs:
            try:
                # --- Dep-order validation ---
                # Comment: Certain L2 operators assume [bid, ask] dep ordering.
                # This prevents silently swapping sides (which would invert signals).
                _validate_dep_order(spec)

                col = self._dispatch_operator(spec, result)
                result[spec.name] = col
                computed += 1
            except Exception as e:
                errors += 1
                _log(self.verbose, f"WARN: {spec.name} failed: {e}")
                result[spec.name] = np.nan

        elapsed = time.time() - t0

        # --- Slice to target hour if context was used ---
        # ---------------------------------------------------------------------
        # IMPORTANT CONTEXT-WINDOW MECHANISM (the "contiguous hours" solution):
        #
        # We compute features on a concatenated DataFrame:
        #   [prev_hour] + [target_hour] + [next_hour]
        #
        # This avoids warmup NaNs for large rolling windows (e.g., 3600s) and
        # also enables forward-shift features (e.g., fwd return) that need
        # future rows beyond the current hour.
        #
        # After computing on the full combined frame, we cut back to the exact
        # row-range that corresponds to the target hour so the saved parquet
        # remains "one hour per file" while still benefiting from context.
        # ---------------------------------------------------------------------
        if context_slice is not None:
            start, end = context_slice
            _log(self.verbose,
                 f"Slicing context: rows [{start}:{end}] "
                 f"({end - start} target rows from {len(result)} total)")
            result = result.iloc[start:end].reset_index(drop=True)

        _log(self.verbose,
             f"S1 compute done: computed={computed} errors={errors} "
             f"in {elapsed:.2f}s | total cols={len(result.columns)}")
        return result

    # =====================================================================
    # Dispatch
    # =====================================================================

    def _dispatch_operator(
        self, spec: FeatureSpec, df: pd.DataFrame
    ) -> pd.Series:
        """
        Dispatch a single feature computation.

        Reads spec.operator, collects dep columns from df,
        and returns the computed pd.Series.
        """
        op = spec.operator
        deps = [d.name for d in spec.depends_on]
        params = spec.params
        name = spec.name

        # Validate deps exist
        missing = [c for c in deps if c not in df.columns]
        if missing:
            raise ValueError(f"{name}: missing deps: {missing}")

        inputs = [df[c].astype("float64") for c in deps]
        window_s = _safe_int(params.get("window_s", 0))

        # ═══════════════════════════════════════════════════════════════
        # DERIVED — Generic arithmetic
        # ═══════════════════════════════════════════════════════════════

        if op == "derived.add":
            return inputs[0] + inputs[1]

        if op == "derived.sub":
            return inputs[0] - inputs[1]

        if op == "derived.ratio":
            denom = inputs[1].where(inputs[1].abs() > EPS, np.nan)
            return (inputs[0] / denom).clip(-1e6, 1e6)

        if op == "derived.count_ratio":
            # [CROSS-DIV-FIX 2026-04-27]
            # Like derived.ratio, but tailored for COUNT/VOLUME inputs that are
            # legitimately zero in seconds without market activity.
            #
            # Semantics:
            #   0 / 0   -> 0    (defined: no activity in either market)
            #   0 / x   -> 0    (defined: only the other market traded —
            #                    "this market has 0x relative activity")
            #   x / 0   -> NaN  (genuinely undefined: cannot express
            #                    "x times zero" as a finite ratio without
            #                    losing information)
            #   x / y   -> clip(x/y, 0, 1e6)
            #
            # Why this matters: the previous derived.ratio returned NaN at
            # every 0/0 second. Inputs like trade_count_fut_1s have ~3-6%
            # zero-seconds. After 300s rolling_mean with strict
            # min_periods=window_s, the probability of having 300 consecutive
            # non-NaN seconds is (1-0.06)^300 ~ 1e-8, so the rolled feature
            # was 100% NaN. Defining 0/0 = 0 eliminates the NaN propagation
            # without semantic loss: a 300s window with no trade activity
            # has a defined ratio of zero, not "undefined".
            num = inputs[0]
            denom = inputs[1]
            num_pos   = num.abs()   > EPS
            denom_pos = denom.abs() > EPS

            # x/0 stays NaN; otherwise compute the safe ratio
            denom_safe = denom.where(denom_pos, np.nan)
            raw = (num / denom_safe).clip(0.0, 1e6)

            # 0/0 = 0 and 0/x = 0  (any case with num ~= 0 is defined as 0)
            zero_num = ~num_pos
            raw = raw.mask(zero_num, 0.0)
            return raw

        if op == "derived.share":
            # share = a / (a + b + eps). Range [0, 1].
            return inputs[0] / (inputs[0] + inputs[1] + EPS)

        # ═══════════════════════════════════════════════════════════════
        # DERIVED — Rolling aggregation
        # ═══════════════════════════════════════════════════════════════

        if op == "derived.roll_mean":
            col = inputs[0]
            return col.rolling(window=window_s, min_periods=window_s).mean()

        if op == "derived.roll_sum":
            col = inputs[0]
            return col.rolling(window=window_s, min_periods=window_s).sum()

        # ═══════════════════════════════════════════════════════════════
        # DERIVED — Statistical normalization
        # ═══════════════════════════════════════════════════════════════

        if op == "derived.robust_zscore":
            # Single-column robust z-score: (x - median) / (1.4826 * MAD + eps)
            col = inputs[0]
            if window_s > 0:
                med = col.rolling(window=window_s, min_periods=window_s).median()
                abs_dev = (col - med).abs()
                mad = abs_dev.rolling(window=window_s, min_periods=window_s).median()
            else:
                # No window_s provided: use expanding window (full history)
                med = col.expanding(min_periods=1).median()
                abs_dev = (col - med).abs()
                mad = abs_dev.expanding(min_periods=1).median()
            return ((col - med) / (1.4826 * mad + EPS)).clip(-100, 100)

        if op == "derived.zscore_diff":
            # 2-dep z-score: z-score of (col_a - col_b) over rolling window.
            diff = inputs[0] - inputs[1]
            med = diff.rolling(window=window_s, min_periods=window_s).median()
            abs_dev = (diff - med).abs()
            mad = abs_dev.rolling(window=window_s, min_periods=window_s).median()
            return ((diff - med) / (1.4826 * mad + EPS)).clip(-100, 100)

        # ═══════════════════════════════════════════════════════════════
        # DERIVED — Returns / Price
        # ═══════════════════════════════════════════════════════════════

        if op == "derived.log_return":
            col = inputs[0]
            return np.log(col / col.shift(1))

        if op == "derived.ret_fwd":
            col = inputs[0]
            shift = _safe_int(params.get("window_s", params.get("shift_s", 1)))
            return np.log(col.shift(-shift) / col)

        if op == "derived.range_pct":
            col = inputs[0]
            hi = col.rolling(window=window_s, min_periods=1).max()
            lo = col.rolling(window=window_s, min_periods=1).min()
            mid = (hi + lo) / 2
            return (hi - lo) / (mid + EPS)

        if op == "derived.range_pos":
            col = inputs[0]
            hi = col.rolling(window=window_s, min_periods=1).max()
            lo = col.rolling(window=window_s, min_periods=1).min()
            span = hi - lo
            return (col - lo) / (span + EPS)

        # ═══════════════════════════════════════════════════════════════
        # RANGE — External OHLC operators (daily range context)
        # These operators receive pre-computed day_high/day_low columns
        # that were injected by _join_ohlc() before feature computation.
        # If OHLC context is unavailable, inputs will be NaN → output NaN.
        # ═══════════════════════════════════════════════════════════════

        if op == "range.dist_to_high_bps":
            # (high - mid) / mid * 10000. Deps: [mid_col, high_col]
            mid  = inputs[0]
            high = inputs[1]
            return ((high - mid) / (mid + EPS) * 10000.0).clip(-1e5, 1e5)

        if op == "range.dist_to_low_bps":
            # (mid - low) / mid * 10000. Deps: [mid_col, low_col]
            mid = inputs[0]
            low = inputs[1]
            return ((mid - low) / (mid + EPS) * 10000.0).clip(-1e5, 1e5)

        if op == "range.ext_position":
            # (mid - low) / (high - low + eps). Deps: [mid_col, low_col, high_col]
            mid  = inputs[0]
            low  = inputs[1]
            high = inputs[2]
            span = high - low
            return ((mid - low) / (span + EPS)).clip(0.0, 1.0)

        if op == "range.ext_range_bps":
            # (high - low) / mid * 10000. Deps: [high_col, low_col]
            high = inputs[0]
            low  = inputs[1]
            mid  = (high + low) / 2.0
            return ((high - low) / (mid + EPS) * 10000.0).clip(0.0, 1e5)

        # ═══════════════════════════════════════════════════════════════
        # FORWARD-LOOKING (Cold-Path only)
        # Uses reverse-rolling trick: O(N log W) instead of O(N*W).
        # Not computable in real time: they read future buckets.
        # ═══════════════════════════════════════════════════════════════

        if op == "derived.mae_fwd":
            # MAE = (mid[t] - min(mid[t..t+w])) / mid[t] * 10000
            window_s = _safe_int(params["window_s"])
            mid = inputs[0]
            fwd_min = mid.iloc[::-1].rolling(window_s, min_periods=1).min().iloc[::-1]
            return ((mid - fwd_min) / (mid + EPS) * 10000.0).clip(0, 1e5)

        if op == "derived.mfe_fwd":
            window_s = _safe_int(params["window_s"])
            mid = inputs[0]
            fwd_max = mid.iloc[::-1].rolling(window_s, min_periods=1).max().iloc[::-1]
            return ((fwd_max - mid) / (mid + EPS) * 10000.0).clip(0, 1e5)

        if op == "derived.rv_fwd":
            # sqrt(sum(r_1s^2, forward window_s))
            # Input is mid; we compute r_1s inline.
            window_s = _safe_int(params["window_s"])
            mid = inputs[0]
            r_1s = np.log(mid / mid.shift(1))
            sq = r_1s ** 2
            # Forward-sum via reverse-rolling-sum shifted by 1 (we want t+1..t+w).
            rev_sum = sq.iloc[::-1].rolling(window_s, min_periods=window_s).sum().iloc[::-1]
            rv = np.sqrt(rev_sum.shift(-1))
            return rv

        # ═══════════════════════════════════════════════════════════════
        # EMA / TREND
        # ═══════════════════════════════════════════════════════════════

        if op == "derived.ema":
            # [STATEFUL-EMA]
            # Recursive EMA: ema[t] = alpha * x[t] + (1-alpha) * ema[t-1]
            # where alpha = 2 / (span + 1).
            #
            # To keep EMA continuity across the hour boundary, the recursion
            # must not be re-seeded from a single end-of-hour value applied at
            # the start of the prev-hour context (that produces drifted values
            # and large boundary jumps). Instead the engine receives a
            # per-second history
            # of EMA values for the last MAX_LOOKBACK seconds via
            # self._ema_history[name] (a pd.Series indexed by bucket_dt_utc).
            #
            #   - Pre-target rows (the prev-hour context): we do NOT recompute
            #     EMA. Instead we hydrate those rows from history aligned by
            #     bucket_dt_utc. This preserves correct EMA values without the
            #     drift bug.
            #   - Target+next rows: we run the recursive EMA seeded with the
            #     last available history value. The recursion is mathematically
            #     identical to a continuous EMA over the entire dataset.
            #   - First hour (no history): bootstrap with first valid sample,
            #     same as pandas ewm(adjust=False).
            #
            # We also build an "extended" EMA series (history + new target
            # values) and store it in self._ema_extended[name] so the slope
            # operator can perform back-shift lookups beyond the combined
            # frame's lookback.
            span_s = _safe_int(params["span_s"])
            col = inputs[0].astype("float64")
            alpha = 2.0 / (span_s + 1.0)

            history = self._ema_history.get(name)  # pd.Series or None
            history_has_data = (history is not None) and (len(history) > 0) \
                               and (not history.dropna().empty)
            if history_has_data:
                # Last (latest) non-NaN value in history = EMA at end of prev hour
                history_clean = history.dropna()
                init_val = float(history_clean.iloc[-1])
            else:
                init_val = float("nan")

            n = len(col)
            out = np.full(n, np.nan, dtype="float64")
            arr = col.values

            # Determine target slice; if not set, treat full frame as target.
            if self._target_slice is not None:
                target_start, target_end = self._target_slice
            else:
                target_start, target_end = 0, n

            if not np.isnan(init_val):
                # ── Path A: history available ────────────────────────────
                # Pre-target rows: hydrate from history (aligned on bucket_dt_utc)
                if target_start > 0:
                    if "bucket_dt_utc" in df.columns:
                        try:
                            pre_dt = pd.to_datetime(
                                df["bucket_dt_utc"].iloc[:target_start],
                                utc=True,
                            )
                            # Reindex history at the pre-target timestamps.
                            # Missing alignments will be NaN (acceptable: those
                            # rows are sliced off in the final output anyway,
                            # and the slope op uses _ema_extended for lookups).
                            hist_aligned = history.reindex(pre_dt.values)
                            out[:target_start] = hist_aligned.values
                        except Exception as e:
                            _log(self.verbose,
                                 f"  WARN: {name}: history hydration failed ({e}); "
                                 f"pre-target rows will be NaN")

                # Target+next rows: recursive EMA seeded from history end
                last = init_val
                for i in range(target_start, n):
                    x = arr[i]
                    if np.isnan(x):
                        out[i] = last
                    else:
                        last = alpha * x + (1.0 - alpha) * last
                        out[i] = last
            else:
                # ── Path B: no history (cold start, e.g. first hour ever) ──
                # Bootstrap with first valid sample, matching pandas
                # ewm(adjust=False) semantics.
                last = float("nan")
                for i in range(n):
                    x = arr[i]
                    if np.isnan(x):
                        out[i] = last
                    else:
                        if np.isnan(last):
                            last = x
                        else:
                            last = alpha * x + (1.0 - alpha) * last
                        out[i] = last

            # ── Capture target-hour values for persistence ──────────────────
            # Index by bucket_dt_utc so future hours can align via timestamp.
            if "bucket_dt_utc" in df.columns:
                try:
                    dt_index = pd.to_datetime(
                        df["bucket_dt_utc"].iloc[target_start:target_end].values,
                        utc=True,
                    )
                    target_vals = pd.Series(
                        out[target_start:target_end],
                        index=dt_index,
                        name=name,
                    )
                    self._ema_target_values[name] = target_vals

                    # Build extended series: history + new target slice
                    # (deduplicated on timestamp, keeping new values for any
                    # overlap so we don't accidentally use stale data).
                    if history_has_data:
                        ext = pd.concat([history, target_vals]).sort_index()
                        ext = ext[~ext.index.duplicated(keep="last")]
                    else:
                        ext = target_vals
                    self._ema_extended[name] = ext
                except Exception as e:
                    _log(self.verbose,
                         f"  WARN: {name}: history capture failed ({e})")

            return pd.Series(out, index=col.index)

        if op == "derived.price_vs_ema_bps":
            # Deps: [price, ema]
            price = inputs[0]
            ema = inputs[1]
            return ((price - ema) / (ema + EPS) * 10000.0).clip(-1e5, 1e5)

        if op == "derived.ema_slope_bps":
            # [STATEFUL-EMA]
            # slope = (ema[t] - ema[t - shift_s]) / ema[t - shift_s] * 10000
            #
            # Previously: ema_prev = ema.shift(shift_s)
            # This only looked back within the in-memory combined frame
            # (~10800 rows). For shift_s = 14400 (240m), the lookup always
            # fell off the start -> 100% NaN. For shift_s = 3600 (60m),
            # it fell into the prev-hour rows whose values were either
            # the (drifted) replay or constants -> wrong values.
            #
            # Now: we look up ema_prev[t] in the EXTENDED EMA series
            # (history + new target slice) by bucket_dt_utc - shift_s.
            # The extended series has up to MAX_LOOKBACK seconds of
            # history loaded from prev-hour state files, so all
            # configured shift_s values (up to MAX_LOOKBACK) work.
            #
            # Falls back to in-frame shift if extended series is unavailable
            # (e.g. very first hour with no history yet).
            shift_s = _safe_int(params["shift_s"])
            ema = inputs[0]
            ema_dep_name = spec.depends_on[0].name

            ext = self._ema_extended.get(ema_dep_name)

            if ext is not None and "bucket_dt_utc" in df.columns:
                try:
                    dt_now = pd.to_datetime(df["bucket_dt_utc"].values, utc=True)
                    dt_back = dt_now - pd.Timedelta(seconds=shift_s)
                    # Reindex extended series at the back-shifted timestamps.
                    # Missing alignments (timestamps before history start)
                    # will be NaN -> slope is NaN there. That is the correct
                    # behaviour: for the very first hours of the dataset,
                    # 240m slope should be NaN until enough history accumulated.
                    ema_prev_vals = ext.reindex(dt_back).values
                    ema_prev = pd.Series(ema_prev_vals, index=ema.index)
                except Exception as e:
                    _log(self.verbose,
                         f"  WARN: {name}: extended-shift failed ({e}); "
                         f"falling back to in-frame shift")
                    ema_prev = ema.shift(shift_s)
            else:
                # Cold start: no extended series yet. Best-effort shift.
                ema_prev = ema.shift(shift_s)

            return ((ema - ema_prev) / (ema_prev + EPS) * 10000.0).clip(-1e5, 1e5)

        if op == "derived.trend_align":
            # Deps: [price, ema_short, ema_long]
            price, ema_short, ema_long = inputs[0], inputs[1], inputs[2]
            up = (price > ema_short) & (ema_short > ema_long)
            down = (price < ema_short) & (ema_short < ema_long)
            out = pd.Series(0.0, index=price.index)
            out[up] = 1.0
            out[down] = -1.0
            return out

        # ═══════════════════════════════════════════════════════════════
        # LEVEL-DISTANCE (generic, reused for prev_day / week / month / POC)
        # ═══════════════════════════════════════════════════════════════

        if op == "range.dist_to_level_bps":
            # (price - level) / price * 10000
            # Symmetric: sign follows (price - level)
            # Deps: [price, level]
            price = inputs[0]
            level = inputs[1]
            return ((price - level) / (price + EPS) * 10000.0).clip(-1e5, 1e5)

        # ═══════════════════════════════════════════════════════════════
        # RECLAIM / BREAK EVENTS (DEBOUNCED)
        # Simple above/below are instantaneous; reclaim/break_* stay active
        # for window_s seconds after the cross event as long as the
        # post-cross price condition holds.
        # ═══════════════════════════════════════════════════════════════

        if op == "derived.above_level":
            price, level = inputs[0], inputs[1]
            return (price > level).astype("float64")

        if op == "derived.below_level":
            price, level = inputs[0], inputs[1]
            return (price < level).astype("float64")

        if op == "derived.reclaim_flag":
            # 1 for window_s after crossing UP through level, while still above.
            window_s = _safe_int(params["window_s"])
            price, level = inputs[0], inputs[1]
            was_below = (price.shift(1) < level.shift(1)).astype("float64")
            is_above = (price >= level).astype("float64")
            cross_up = was_below * is_above  # 1 exactly on cross tick
            # debounce: event stays active for window_s seconds
            debounced = cross_up.rolling(window_s, min_periods=1).max()
            return debounced * is_above  # condition out when price drops back below

        if op == "derived.break_flag_high":
            # identical to reclaim_flag but semantically for resistance level
            window_s = _safe_int(params["window_s"])
            price, level = inputs[0], inputs[1]
            was_below = (price.shift(1) < level.shift(1)).astype("float64")
            is_above = (price >= level).astype("float64")
            cross_up = was_below * is_above
            debounced = cross_up.rolling(window_s, min_periods=1).max()
            return debounced * is_above

        if op == "derived.break_flag_low":
            # 1 for window_s after crossing DOWN through support, while still below.
            window_s = _safe_int(params["window_s"])
            price, level = inputs[0], inputs[1]
            was_above = (price.shift(1) > level.shift(1)).astype("float64")
            is_below = (price <= level).astype("float64")
            cross_dn = was_above * is_below
            debounced = cross_dn.rolling(window_s, min_periods=1).max()
            return debounced * is_below

        # ═══════════════════════════════════════════════════════════════
        # FIBONACCI
        # ═══════════════════════════════════════════════════════════════

        if op == "derived.fib_dist_bps":
            # fib_price = low + fib_level * (high - low)
            # (price - fib_price) / price * 10000
            # Deps: [price, low, high]
            fib_level = float(params["fib_level"])
            price, low, high = inputs[0], inputs[1], inputs[2]
            fib_price = low + fib_level * (high - low)
            return ((price - fib_price) / (price + EPS) * 10000.0).clip(-1e5, 1e5)

        # ═══════════════════════════════════════════════════════════════
        # VOLUME PROFILE
        # POC/VAH/VAL levels come from vp_{asset}_{date}.parquet injected
        # via a _join_volume_profile() context join (Phase 3).
        # ═══════════════════════════════════════════════════════════════

        if op == "derived.price_vs_va":
            # Deps: [price, vah, val]
            price, vah, val = inputs[0], inputs[1], inputs[2]
            out = pd.Series(1.0, index=price.index)  # default: inside VA
            out[price > vah] = 2.0
            out[price < val] = 0.0
            return out

        if op == "derived.poc_migration_bps":
            shift_s = _safe_int(params["shift_s"])
            poc = inputs[0]
            poc_prev = poc.shift(shift_s)
            return ((poc - poc_prev) / (poc_prev + EPS) * 10000.0).clip(-1e5, 1e5)

        # ═══════════════════════════════════════════════════════════════
        # DERIVED — Cross-market / basis
        # ═══════════════════════════════════════════════════════════════

        if op == "derived.basis":
            return inputs[0] - inputs[1]

        if op == "derived.basis_bps":
            # (dep0 - dep1) / (dep1 + eps) * 10000
            raw = (inputs[0] - inputs[1]) / (inputs[1] + EPS) * 10_000
            if window_s > 1:
                return raw.rolling(
                    window=window_s, min_periods=1
                ).mean()
            return raw

        if op == "deriv.basis_mid":
            raw = inputs[0] - inputs[1]
            if window_s > 1:
                return raw.rolling(window=window_s, min_periods=1).mean()
            return raw

        # ═══════════════════════════════════════════════════════════════
        # DERIVED — Activity / Meta
        # ═══════════════════════════════════════════════════════════════

        if op == "derived.participation_rate_1s":
            # volume / EWMA(volume, halflife)
            col = inputs[0]
            halflife = _safe_int(params.get("ewma_halflife_s", 3600))
            ewma = col.ewm(halflife=halflife, min_periods=1).mean()
            return col / (ewma + EPS)

        if op == "deriv.spot_fut_taker_activity_share_1s":
            # spot_act / (spot_act + fut_act + eps)
            # deps: [spot_buy, spot_sell, fut_buy, fut_sell]
            spot_act = inputs[0] + inputs[1]
            fut_act = inputs[2] + inputs[3]
            return spot_act / (spot_act + fut_act + EPS)

        # ═══════════════════════════════════════════════════════════════
        # TRADES — Aggregation
        # ═══════════════════════════════════════════════════════════════

        if op == "trades.avg_trade_size":
            denom = inputs[1].where(inputs[1].abs() > EPS, np.nan)
            return inputs[0] / denom

        if op == "trades.taker_imbalance":
            # (buy - sell) / (buy + sell + eps)
            return (inputs[0] - inputs[1]) / (inputs[0] + inputs[1] + EPS)

        if op == "trades.vwap":
            denom = inputs[1].where(inputs[1].abs() > EPS, np.nan)
            return inputs[0] / denom

        # ═══════════════════════════════════════════════════════════════
        # L2 — Bookshape
        # ═══════════════════════════════════════════════════════════════

        if op == "l2.book_asymmetry":
            # (bid - ask) / (bid + ask + eps)
            return (inputs[0] - inputs[1]) / (inputs[0] + inputs[1] + EPS)

        if op == "l2.depth_gradient_ask" or op == "l2.depth_gradient_bid":
            # Depth gradient: fraction of depth beyond inner band.
            # 2 deps: [inner_depth, outer_depth]. Result ∈ [0, 1].
            # 0 = all depth within inner band, 1 = all depth in outer band.
            # Clip handles snapshot jitter where inner > outer momentarily.
            inner, outer = inputs[0], inputs[1]
            return ((outer - inner) / (outer.abs() + EPS)).clip(0.0, 1.0)

        if op == "l2.liq_cluster_asymmetry":
            # Rolling coefficient-of-variation difference between ask/bid depth.
            # CV = std/mean measures variability; high CV = dispersed liquidity.
            # Result: (CV_bid - CV_ask) / (CV_bid + CV_ask + eps)
            # Positive → bid more variable (less clustered) than ask.
            ask_depth, bid_depth = inputs[0], inputs[1]
            w = max(window_s, 2)
            ask_mean = ask_depth.rolling(window=w, min_periods=1).mean()
            ask_std = ask_depth.rolling(window=w, min_periods=1).std(ddof=0)
            bid_mean = bid_depth.rolling(window=w, min_periods=1).mean()
            bid_std = bid_depth.rolling(window=w, min_periods=1).std(ddof=0)
            cv_ask = ask_std / (ask_mean.abs() + EPS)
            cv_bid = bid_std / (bid_mean.abs() + EPS)
            return (cv_bid - cv_ask) / (cv_bid + cv_ask + EPS)

        if op == "l2.liq_concentration_ask" or op == "l2.liq_concentration_bid":
            # Liquidity concentration: fraction of depth within inner band.
            # 2 deps: [inner_depth, outer_depth]. By definition inner ≤ outer,
            # so result ∈ [0, 1]. Clip handles edge cases (snapshot jitter).
            inner, outer = inputs[0], inputs[1]
            return (inner / (outer.abs() + EPS)).clip(0.0, 1.0)

        if op == "l2.liq_sum":
            return inputs[0] + inputs[1]

        # NOTE: l2.max_liq_distance removed from specs — requires per-level data.

        # ═══════════════════════════════════════════════════════════════
        # L2 — Price
        # ═══════════════════════════════════════════════════════════════

        # NOTE: l2.lwp removed from specs — True LWP requires best-level quantities.

        if op == "l2.mid_touch":
            # Mid-touch: (bid + ask) / 2
            return (inputs[0] + inputs[1]) / 2

        # ═══════════════════════════════════════════════════════════════
        # L2 — Pressure
        # ═══════════════════════════════════════════════════════════════

        if op == "l2.queue_imbalance_1s":
            # (bid - ask) / (bid + ask + eps)
            return (inputs[0] - inputs[1]) / (inputs[0] + inputs[1] + EPS)

        if op == "l2.queue_pressure":
            # (bid - ask) / (bid + ask + eps), with optional rolling mean
            raw = (inputs[0] - inputs[1]) / (inputs[0] + inputs[1] + EPS)
            if window_s > 1:
                return raw.rolling(window=window_s, min_periods=1).mean()
            return raw

        if op == "l2.queue_pressure_log_1s":
            # log((bid + eps) / (ask + eps))
            return np.log((inputs[0] + EPS) / (inputs[1] + EPS))

        if op == "l2.net_pressure":
            # 6 deps: (add_bid, cancel_bid, add_ask, cancel_ask, depth_ask, depth_bid)
            # net = (add_bid - cancel_bid) - (add_ask - cancel_ask)
            # normalized by total depth, with optional rolling mean
            add_bid, cancel_bid = inputs[0], inputs[1]
            add_ask, cancel_ask = inputs[2], inputs[3]
            depth_ask, depth_bid = inputs[4], inputs[5]
            net = (add_bid - cancel_bid) - (add_ask - cancel_ask)
            total_depth = depth_ask + depth_bid
            denom = total_depth.where(total_depth.abs() > EPS, np.nan)
            raw = net / denom
            if window_s > 1:
                return raw.rolling(window=window_s, min_periods=1).mean()
            return raw

        # ═══════════════════════════════════════════════════════════════
        # L2 — Absorption / Liquidity Events
        # ═══════════════════════════════════════════════════════════════

        if op == "l2.absorb_refill_ask" or op == "l2.absorb_refill_bid":
            return inputs[0] * inputs[1]

        if op == "l2.aggr_absorp_ratio_ask" or op == "l2.aggr_absorp_ratio_bid":
            # Aggressor / depth ratio. Both are non-negative, so result ≥ 0.
            # Clip upper bound prevents extreme spikes when depth ≈ 0.
            denom = inputs[1].where(inputs[1].abs() > EPS, np.nan)
            return (inputs[0] / denom).clip(0, 1e6)

        if op == "l2.add_rate_ask" or op == "l2.add_rate_bid":
            # Liquidity added = positive depth change: max(0, depth(t) - depth(t-1))
            return inputs[0].diff().clip(lower=0).fillna(0)

        if op == "l2.cancel_rate_ask" or op == "l2.cancel_rate_bid":
            # Liquidity removed = negative depth change: max(0, depth(t-1) - depth(t))
            return (-inputs[0].diff()).clip(lower=0).fillna(0)

        if op == "l2.fill_rate_ahead":
            # 4 deps: directional fill rate, with rolling mean + clip
            # Fills are non-negative events, so result ≥ 0.
            if len(inputs) == 4:
                raw = (inputs[0] + inputs[1]) / (inputs[2] + inputs[3] + EPS)
            else:
                raw = inputs[0]
            raw = raw.clip(0, 1e6)
            if window_s > 1:
                return raw.rolling(window=window_s, min_periods=1).mean()
            return raw

        if op == "l2.pull_rate":
            # Normalized cancel rate: total cancels / (total depth + eps)
            # 4 deps: [cancel_ask, cancel_bid, depth_ask, depth_bid]
            # Cancels are non-negative, so result ≥ 0.
            if len(inputs) == 4:
                raw = (inputs[0] + inputs[1]) / (inputs[2] + inputs[3] + EPS)
            else:
                denom = inputs[1].where(inputs[1].abs() > EPS, np.nan)
                raw = inputs[0] / denom
            raw = raw.clip(0, 1e6)
            if window_s > 1:
                return raw.rolling(window=window_s, min_periods=1).mean()
            return raw

        if op == "l2.refill_rate":
            # Normalized add rate: total adds / (total depth + eps)
            # 4 deps: [add_ask, add_bid, depth_ask, depth_bid]
            # Adds are non-negative, so result ≥ 0.
            if len(inputs) == 4:
                raw = (inputs[0] + inputs[1]) / (inputs[2] + inputs[3] + EPS)
            else:
                denom = inputs[1].where(inputs[1].abs() > EPS, np.nan)
                raw = inputs[0] / denom
            raw = raw.clip(0, 1e6)
            if window_s > 1:
                return raw.rolling(window=window_s, min_periods=1).mean()
            return raw

        # ═══════════════════════════════════════════════════════════════
        # L2 — Activity / Meta
        # ═══════════════════════════════════════════════════════════════

        if op == "l2.l2_update_count":
            # Count of depth changes per bucket: any ask or bid depth moved
            ask_changed = inputs[0].diff().abs() > EPS
            bid_changed = inputs[1].diff().abs() > EPS
            return (ask_changed | bid_changed).astype("float64")

        # ═══════════════════════════════════════════════════════════════
        # Fallback
        # ═══════════════════════════════════════════════════════════════

        raise ValueError(f"{name}: unknown S1 operator '{op}'")


# =============================================================================
# Path helpers
# =============================================================================

def _paths_for_hour(
    s0_dir: str, out_dir: str, asset: str, date_str: str, hour: int
) -> tuple:
    hh = f"{int(hour):02d}"
    suffix = f"{date_str}_{hh}.parquet"
    a = asset.lower()
    s0_path = Path(s0_dir) / f"s0_features_{a}_{suffix}"
    out_path = Path(out_dir) / f"s1_features_{a}_{suffix}"
    return s0_path, out_path


def _adjacent_hour(date_str: str, hour: int, delta: int) -> Tuple[str, int]:
    """
    Resolve an adjacent hour, handling midnight crossing.

    Comment: This is a small but important detail for the context-window logic.
    If you request hour=-1 from 00:00, you must roll back to the previous day 23:00.
    Likewise hour=+1 from 23:00 must roll forward to next day 00:00.

    Args:
        date_str: Current date as 'YYYY-MM-DD'.
        hour:     Current hour (0-23).
        delta:    Hour offset (-1 = previous, +1 = next).

    Returns:
        (new_date_str, new_hour) tuple.
    """
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(
        hour=hour, tzinfo=timezone.utc
    )
    dt2 = dt + timedelta(hours=delta)
    return dt2.strftime("%Y-%m-%d"), dt2.hour


def _try_load_s0(s0_dir: str, asset: str, date_str: str, hour: int) -> Optional[pd.DataFrame]:
    """
    Try to load an S0 parquet file. Returns None if the file doesn't exist.

    Comment: Context hours are optional. We attempt to load them and silently
    proceed without them if the file isn't present (e.g., at dataset boundaries
    or when backfilling incomplete days).
    """
    hh = f"{int(hour):02d}"
    a = asset.lower()
    path = Path(s0_dir) / f"s0_features_{a}_{date_str}_{hh}.parquet"
    if path.exists():
        return _read_parquet(str(path))
    return None


def _load_with_context(
    s0_dir: str,
    asset: str,
    date_str: str,
    hour: int,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, int, int]:
    """
    Load the target hour S0 file plus adjacent hours for context.

    Returns:
        (combined_df, start_idx, end_idx)
        where combined_df[start_idx:end_idx] is the target hour's rows.

    -------------------------------------------------------------------------
    THIS IS THE CORE "CONTIGUOUS HOURS" SOLUTION USED BY S1:
      1) Load target hour (required).
      2) Try to load previous hour (optional lookback context).
      3) Try to load next hour (optional lookahead context).
      4) Concatenate everything into ONE continuous timeline.
      5) Return the slice indices so we can later cut back to just the
         target hour rows after rolling/forward features were computed.
    Why both sides?
      - lookback (prev hour) fixes warmup NaNs for large rolling windows.
      - lookahead (next hour) enables forward-shift labels/features like
        ret_fwd_3600s without becoming NaN for the last N seconds of the hour.
    -------------------------------------------------------------------------
    """
    # --- Load target hour (required) ---
    target_df = _try_load_s0(s0_dir, asset, date_str, hour)
    if target_df is None:
        s0_path = Path(s0_dir) / f"s0_features_{asset.lower()}_{date_str}_{hour:02d}.parquet"
        raise FileNotFoundError(f"Missing S0 features: {s0_path}")

    n_target = len(target_df)

    # --- Load previous hour (optional — lookback context) ---
    prev_date, prev_hour = _adjacent_hour(date_str, hour, -1)
    prev_df = _try_load_s0(s0_dir, asset, prev_date, prev_hour)

    # --- Load next hour (optional — lookahead context) ---
    next_date, next_hour = _adjacent_hour(date_str, hour, +1)
    next_df = _try_load_s0(s0_dir, asset, next_date, next_hour)

    # --- Concatenate ---
    parts = []
    n_before = 0

    if prev_df is not None:
        n_before = len(prev_df)
        parts.append(prev_df)
        _log(verbose, f"  Context: loaded prev hour ({prev_date}_{prev_hour:02d}): {n_before} rows")

    parts.append(target_df)

    n_after = 0
    if next_df is not None:
        n_after = len(next_df)
        parts.append(next_df)
        _log(verbose, f"  Context: loaded next hour ({next_date}_{next_hour:02d}): {n_after} rows")

    if len(parts) == 1:
        # No context available — fall back to target-only
        _log(verbose, "  Context: no adjacent hours found, proceeding without context")
        return target_df, 0, n_target

    combined = pd.concat(parts, ignore_index=True)

    # Comment: The indices of target rows inside the combined frame.
    # If prev exists, target starts at len(prev). If not, start at 0.
    start_idx = n_before
    end_idx = n_before + n_target

    _log(verbose,
         f"  Context window: {len(combined)} rows "
         f"(prev={n_before} + target={n_target} + next={n_after})")

    return combined, start_idx, end_idx


# =============================================================================
# OHLC Context Join
# =============================================================================

# All OHLC columns that we must guarantee exist on the output.
# Updated 2026-04-26 for Running-OHLC architecture:
#   - day_close_* removed (was redundant with current mid_*; meaningful only at
#     23:59:59 UTC, useless during the day, and broken in hot-path)
#   - day_high/low/open_* now per-second running values from
#     ohlc_running_{asset}_{date}_{hh}.parquet, joined on bucket_dt_utc
#   - prev_day_high/low_{spot,fut} loaded from the final state file of the
#     PREVIOUS UTC day (ohlc_state_{asset}_{prev_date}_23.parquet)
_OHLC_COLS = [
    "day_high_spot", "day_low_spot", "day_open_spot",
    "day_high_fut",  "day_low_fut",  "day_open_fut",
    "prev_day_high_spot", "prev_day_low_spot",
    "prev_day_high_fut",  "prev_day_low_fut",
]


def _join_ohlc(
    df: pd.DataFrame,
    ohlc_dir: str,
    asset: str,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Join Running-OHLC columns onto a combined S0 DataFrame.

    The df may span up to 3 calendar dates (prev/target/next hour at midnight
    crossings) and up to ~3 UTC hours. For each unique (date, hour) present
    in df, we load:
      ohlc_running_{asset}_{date}_{hh}.parquet
        - per-second day_high/low/open values, joined on bucket_dt_utc

    For each unique date in df, we load:
      ohlc_state_{asset}_{prev_date}_23.parquet
        - 1-row final state of the previous UTC day; provides prev_day_*

    [RUNNING-OHLC 2026-04-26]
      Old design: one ohlc_{asset}_{date}.parquet per day, joined on date_str,
      computed only after all 24 hours of the day were available. Hot-path
      could not reproduce this and got always-NaN day_high/low/open values.
      day_close was a separate column.

      New design: per-hour running cummax/cummin/open with state passed
      from prev hour. Always available, hot-path tractable, no day_close.

    Adds columns:
      day_high/low/open_{spot,fut}        — 6 per-second running
      prev_day_high/low_{spot,fut}        — 4 from prev-day final state

    Rows whose corresponding ohlc_running file is missing receive NaN.
    Rows whose prev day has no state file receive NaN for prev_day_*.
    """
    if "bucket_dt_utc" not in df.columns:
        _log(verbose, "WARN: bucket_dt_utc not in df -- skipping OHLC join")
        return df

    # Derive (date, hour) per row
    dt_col = pd.to_datetime(df["bucket_dt_utc"], utc=True)
    date_strs = dt_col.dt.strftime("%Y-%m-%d")
    hours     = dt_col.dt.hour
    unique_pairs = pd.DataFrame({
        "date_str": date_strs.values,
        "hour":     hours.values,
    }).drop_duplicates().sort_values(["date_str", "hour"])

    ohlc_dir_p = Path(ohlc_dir)
    asset_l    = asset.lower()

    # ─────────────────────────────────────────────────────────────────
    # 1) Load per-second running OHLC for each (date, hour) in df
    # ─────────────────────────────────────────────────────────────────
    running_frames: List[pd.DataFrame] = []
    for _, row in unique_pairs.iterrows():
        date_str = row["date_str"]
        hour     = int(row["hour"])
        p = ohlc_dir_p / f"ohlc_running_{asset_l}_{date_str}_{hour:02d}.parquet"
        if p.exists():
            try:
                running_frames.append(pq.read_table(str(p)).to_pandas())
            except Exception as e:
                _log(verbose, f"WARN: failed to load OHLC running {p.name}: {e}")
        else:
            _log(verbose,
                 f"  OHLC running not found for {date_str} h{hour:02d} -- "
                 f"per-second high/low/open will be NaN for those rows")

    if running_frames:
        running_df = pd.concat(running_frames, ignore_index=True)
        running_df["bucket_dt_utc"] = pd.to_datetime(
            running_df["bucket_dt_utc"], utc=True
        )
        # Deduplicate on bucket_dt_utc (defensive: same second from overlap)
        running_df = running_df.drop_duplicates(
            subset=["bucket_dt_utc"], keep="first"
        )
        df = df.copy()
        df["bucket_dt_utc"] = pd.to_datetime(df["bucket_dt_utc"], utc=True)
        df = df.merge(running_df, on="bucket_dt_utc", how="left")
    else:
        # No running files -- add NaN columns
        df = df.copy()
        for col in ["day_high_spot", "day_low_spot", "day_open_spot",
                    "day_high_fut",  "day_low_fut",  "day_open_fut"]:
            df[col] = float("nan")

    # ─────────────────────────────────────────────────────────────────
    # 2) Load prev_day_* from final state of previous UTC day
    #    For each unique date in df, we need the state from {date - 1d, h=23}.
    # ─────────────────────────────────────────────────────────────────
    prev_state_rows = []
    unique_dates = sorted(set(date_strs.values))
    for date_str in unique_dates:
        prev_dt = datetime.strptime(date_str, "%Y-%m-%d") - timedelta(days=1)
        prev_date = prev_dt.strftime("%Y-%m-%d")
        prev_path = (ohlc_dir_p
                     / f"ohlc_state_{asset_l}_{prev_date}_23.parquet")
        if prev_path.exists():
            try:
                state_df = pq.read_table(str(prev_path)).to_pandas()
                if not state_df.empty:
                    last = state_df.iloc[0]
                    prev_state_rows.append({
                        "_date_str":         date_str,
                        "prev_day_high_spot": float(last.get("day_high_spot", float("nan"))),
                        "prev_day_low_spot":  float(last.get("day_low_spot",  float("nan"))),
                        "prev_day_high_fut":  float(last.get("day_high_fut",  float("nan"))),
                        "prev_day_low_fut":   float(last.get("day_low_fut",   float("nan"))),
                    })
            except Exception as e:
                _log(verbose, f"WARN: failed to load prev-day state {prev_path.name}: {e}")

    if prev_state_rows:
        prev_df = pd.DataFrame(prev_state_rows)
        df = df.copy()
        df["_date_str"] = date_strs.values
        df = df.merge(prev_df, on="_date_str", how="left")
        df = df.drop(columns=["_date_str"], errors="ignore")
    else:
        df = df.copy()
        for col in ["prev_day_high_spot", "prev_day_low_spot",
                    "prev_day_high_fut",  "prev_day_low_fut"]:
            df[col] = float("nan")

    # ─────────────────────────────────────────────────────────────────
    # 3) Defensive: ensure all expected columns exist
    # ─────────────────────────────────────────────────────────────────
    for col in _OHLC_COLS:
        if col not in df.columns:
            df[col] = float("nan")

    _log(verbose,
         f"  OHLC join: running={len(running_frames)} hour-files, "
         f"prev-day state={len(prev_state_rows)} dates")
    return df


# =============================================================================
# =============================================================================

_WEEKLY_COLS = [
    "week_open_fut", "week_high_fut", "week_low_fut",
    "monday_high_fut", "monday_low_fut",
    "prev_week_high_fut", "prev_week_low_fut",
]


def _join_weekly(
    df: pd.DataFrame,
    weekly_dir: str,
    asset: str,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Join weekly-level columns (1s-grid) onto the combined S0 DataFrame.

    The df may span up to 3 calendar dates at hour boundaries, which may
    touch up to 2 ISO-weeks (e.g. a Sunday 23:00 → Monday 00:00 boundary).
    For each unique (iso_year, iso_week) present in df we load
    weekly_{asset}_{iso_year}_{iso_week:02d}.parquet and left-join it on
    bucket_dt_utc.

    Added columns: week_open_fut, week_high_fut, week_low_fut,
                   monday_high_fut, monday_low_fut,
                   prev_week_high_fut, prev_week_low_fut.

    Missing weekly parquets are tolerated — rows in that week receive NaN,
    and downstream features degrade gracefully.

    Args:
        df:         Combined DataFrame with 'bucket_dt_utc' column.
        weekly_dir: Directory containing weekly_{asset}_{iy}_{iw:02d}.parquet files.
        asset:      'btc' or 'eth'.
        verbose:    Print progress.

    Returns:
        df with weekly columns appended (same row count and index).
    """
    if "bucket_dt_utc" not in df.columns:
        _log(verbose, "WARN: bucket_dt_utc not in df — skipping weekly join")
        return df

    dt_col = pd.to_datetime(df["bucket_dt_utc"], utc=True)
    iso_cal = dt_col.dt.isocalendar()
    # Build a set of (iso_year, iso_week) pairs actually present in df.
    iso_pairs = set(zip(iso_cal["year"].astype(int), iso_cal["week"].astype(int)))

    weekly_rows = []
    loaded_pairs = []
    missing_pairs = []
    for iy, iw in sorted(iso_pairs):
        p = Path(weekly_dir) / f"weekly_{asset.lower()}_{iy}_{iw:02d}.parquet"
        if p.exists():
            try:
                wdf = pq.read_table(str(p)).to_pandas()
                weekly_rows.append(wdf)
                loaded_pairs.append((iy, iw))
            except Exception as e:
                _log(verbose, f"WARN: failed to load weekly {p.name}: {e}")
                missing_pairs.append((iy, iw))
        else:
            _log(verbose,
                 f"  Weekly not found for ISO {iy}-W{iw:02d} — "
                 f"week-level features will be NaN in that region")
            missing_pairs.append((iy, iw))

    if not weekly_rows:
        _log(verbose, "  No weekly files found — week-level features will be NaN")
        for col in _WEEKLY_COLS:
            df[col] = float("nan")
        return df

    weekly_df = pd.concat(weekly_rows, ignore_index=True)
    weekly_df["bucket_dt_utc"] = pd.to_datetime(weekly_df["bucket_dt_utc"], utc=True)

    # Retain only the weekly-level columns we care about plus the key.
    keep = ["bucket_dt_utc"] + [c for c in _WEEKLY_COLS if c in weekly_df.columns]
    weekly_df = weekly_df[keep]
    # Drop duplicate keys defensively (e.g. from overlapping files).
    weekly_df = weekly_df.drop_duplicates(subset=["bucket_dt_utc"], keep="last")

    # Normalise df's join key to the same dtype (tz-aware datetime).
    df = df.copy()
    df["bucket_dt_utc"] = dt_col
    merged = df.merge(weekly_df, on="bucket_dt_utc", how="left")

    # Guarantee all expected columns exist even if the loaded parquets
    # lacked any (defensive against schema drift).
    for col in _WEEKLY_COLS:
        if col not in merged.columns:
            merged[col] = float("nan")

    _log(verbose,
         f"  Weekly join: {len(loaded_pairs)} week(s) joined, "
         f"{len(missing_pairs)} missing — iso_pairs {sorted(iso_pairs)}")
    return merged


# =============================================================================
# =============================================================================

_MONTHLY_COLS = [
    "month_open_fut", "month_high_fut", "month_low_fut",
    "prev_month_high_fut", "prev_month_low_fut",
]


def _join_monthly(
    df: pd.DataFrame,
    monthly_dir: str,
    asset: str,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Join monthly-level columns (1s-grid) onto the combined S0 DataFrame.

    The df may span up to 2 calendar months at hour boundaries (e.g. the
    last hour of March and first hour of April). For each unique (year, month)
    present in df we load monthly_{asset}_{year}_{month:02d}.parquet and
    left-join it on bucket_dt_utc.

    Added columns: month_open_fut, month_high_fut, month_low_fut,
                   prev_month_high_fut, prev_month_low_fut.

    Missing monthly parquets are tolerated (NaN fallback).

    Args:
        df:          Combined DataFrame with 'bucket_dt_utc' column.
        monthly_dir: Directory with monthly_{asset}_{year}_{month:02d}.parquet.
        asset:       'btc' or 'eth'.
        verbose:     Print progress.

    Returns:
        df with monthly columns appended (same row count and index).
    """
    if "bucket_dt_utc" not in df.columns:
        _log(verbose, "WARN: bucket_dt_utc not in df — skipping monthly join")
        return df

    dt_col = pd.to_datetime(df["bucket_dt_utc"], utc=True)
    year_month = set(zip(dt_col.dt.year.astype(int), dt_col.dt.month.astype(int)))

    monthly_rows = []
    loaded_pairs = []
    missing_pairs = []
    for y, m in sorted(year_month):
        p = Path(monthly_dir) / f"monthly_{asset.lower()}_{y}_{m:02d}.parquet"
        if p.exists():
            try:
                mdf = pq.read_table(str(p)).to_pandas()
                monthly_rows.append(mdf)
                loaded_pairs.append((y, m))
            except Exception as e:
                _log(verbose, f"WARN: failed to load monthly {p.name}: {e}")
                missing_pairs.append((y, m))
        else:
            _log(verbose,
                 f"  Monthly not found for {y}-{m:02d} — "
                 f"month-level features will be NaN in that region")
            missing_pairs.append((y, m))

    if not monthly_rows:
        _log(verbose, "  No monthly files found — month-level features will be NaN")
        for col in _MONTHLY_COLS:
            df[col] = float("nan")
        return df

    monthly_df = pd.concat(monthly_rows, ignore_index=True)
    monthly_df["bucket_dt_utc"] = pd.to_datetime(monthly_df["bucket_dt_utc"], utc=True)

    keep = ["bucket_dt_utc"] + [c for c in _MONTHLY_COLS if c in monthly_df.columns]
    monthly_df = monthly_df[keep]
    monthly_df = monthly_df.drop_duplicates(subset=["bucket_dt_utc"], keep="last")

    df = df.copy()
    df["bucket_dt_utc"] = dt_col
    merged = df.merge(monthly_df, on="bucket_dt_utc", how="left")

    for col in _MONTHLY_COLS:
        if col not in merged.columns:
            merged[col] = float("nan")

    _log(verbose,
         f"  Monthly join: {len(loaded_pairs)} month(s) joined, "
         f"{len(missing_pairs)} missing — pairs {sorted(year_month)}")
    return merged


# =============================================================================
# =============================================================================

_VP_COLS = [
    "poc_60m_fut",  "poc_240m_fut",  "poc_1d_fut",
    "vah_60m_fut",  "vah_240m_fut",  "vah_1d_fut",
    "val_60m_fut",  "val_240m_fut",  "val_1d_fut",
    "poc_migration_60m_bps_fut",
    "poc_migration_240m_bps_fut",
    "poc_migration_1d_bps_fut",
]


def _join_volume_profile(
    df: pd.DataFrame,
    vp_dir: str,
    asset: str,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Join volume-profile columns (1s-grid) onto the combined S0 DataFrame.

    The df may span up to 3 calendar dates at hour boundaries. For each
    unique date present in df we load vp_{asset}_{date}.parquet and left-
    join it on bucket_dt_utc.

    Added columns (12 total):
      poc_{60m,240m,1d}_fut          — Point-of-Control
      vah_{60m,240m,1d}_fut          — Value-Area High
      val_{60m,240m,1d}_fut          — Value-Area Low
      poc_migration_{60m,240m,1d}_bps_fut — precomputed POC migration in bps

    Missing VP parquets are tolerated (NaN fallback).

    Args:
        df:      Combined DataFrame with 'bucket_dt_utc' column.
        vp_dir:  Directory containing vp_{asset}_{date}.parquet files.
        asset:   'btc' or 'eth'.
        verbose: Print progress.

    Returns:
        df with VP columns appended (same row count and index).
    """
    if "bucket_dt_utc" not in df.columns:
        _log(verbose, "WARN: bucket_dt_utc not in df — skipping VP join")
        return df

    dt_col = pd.to_datetime(df["bucket_dt_utc"], utc=True)
    unique_dates = dt_col.dt.strftime("%Y-%m-%d").unique()

    vp_rows = []
    loaded_dates = []
    missing_dates = []
    for date_str in sorted(unique_dates):
        p = Path(vp_dir) / f"vp_{asset.lower()}_{date_str}.parquet"
        if p.exists():
            try:
                vpdf = pq.read_table(str(p)).to_pandas()
                vp_rows.append(vpdf)
                loaded_dates.append(date_str)
            except Exception as e:
                _log(verbose, f"WARN: failed to load VP {p.name}: {e}")
                missing_dates.append(date_str)
        else:
            _log(verbose,
                 f"  VP not found for {date_str} — "
                 f"volume-profile features will be NaN in that region")
            missing_dates.append(date_str)

    if not vp_rows:
        _log(verbose, "  No VP files found — volume-profile features will be NaN")
        for col in _VP_COLS:
            df[col] = float("nan")
        return df

    vp_df = pd.concat(vp_rows, ignore_index=True)
    vp_df["bucket_dt_utc"] = pd.to_datetime(vp_df["bucket_dt_utc"], utc=True)

    keep = ["bucket_dt_utc"] + [c for c in _VP_COLS if c in vp_df.columns]
    vp_df = vp_df[keep]
    vp_df = vp_df.drop_duplicates(subset=["bucket_dt_utc"], keep="last")

    df = df.copy()
    df["bucket_dt_utc"] = dt_col
    merged = df.merge(vp_df, on="bucket_dt_utc", how="left")

    for col in _VP_COLS:
        if col not in merged.columns:
            merged[col] = float("nan")

    _log(verbose,
         f"  VP join: {len(loaded_dates)} date(s) joined, "
         f"{len(missing_dates)} missing — dates {sorted(unique_dates)}")
    return merged


# =============================================================================

def _atomic_write_parquet(
    df: pd.DataFrame, out_path: Path, compression: str = PARQUET_COMPRESSION
) -> None:
    """
    Write DataFrame to parquet atomically via tmp file + os.replace.
    Prevents partial writes on crash.

    Comment: Atomic replace is critical for pipelines where downstream jobs
    might pick up output files immediately; we avoid partially-written parquets.
    """
    table = pa.Table.from_pandas(df, preserve_index=False)
    _ensure_dir(out_path.parent)

    # Write to a temp file in the same directory (same filesystem for rename)
    fd, tmp_path = tempfile.mkstemp(
        suffix=".parquet.tmp",
        dir=str(out_path.parent),
    )
    try:
        os.close(fd)
        pq.write_table(table, tmp_path, compression=compression)
        os.replace(tmp_path, str(out_path))
    except BaseException:
        # Clean up temp file on any error
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# =============================================================================
# EMA State Persistence  [STATEFUL-EMA]
# =============================================================================
#
# Stateful EMAs persist their values across hour boundaries so that
# `derived.ema` operators can produce correct values for spans that exceed
# the prev+target+next 3-hour S1 context window (e.g. 60min, 240min EMAs).
#
# State schema — per asset-date-hour:
#   ema_state_{asset}_{date}_{hour:02d}.parquet
#     N rows where N <= MAX_LOOKBACK_S; one row per UTC second.
#     Columns:
#       bucket_dt_utc (datetime64[ns, UTC])  — per-second timestamp
#       <feature_name>: float64              — one column per EMA feature
#     The latest row has bucket_dt_utc == end-of-target-hour. Earlier rows
#     contain the rolling history needed by ema_slope_bps for back-shift
#     lookups beyond the in-frame combined-frame size.
#
# Why N rows instead of 1?
#   A single last-EMA value seeds the recursion but makes `ema_slope_240m_bps`
#   impossible to compute (shift(14400) on a 10800-row frame is always NaN)
#   and makes `ema_slope_60m_bps` constant in the early target-hour rows.
#   Saving the per-second history eliminates both issues.
#
# Lookup logic:
#   - For target hour H, we attempt to load the state of hour H-1 of the
#     same date, OR hour 23 of the previous date if H == 0.
#   - If absent: empty history dict; EMAs bootstrap with the first valid
#     sample (matches pandas ewm(adjust=False) behavior on cold start).
# =============================================================================

# Maximum EMA history retained per state file (in seconds = rows).
# Must be >= the largest configured ema_slope shift_s. 14400s = 4h supports
# all currently configured slopes (5m / 15m / 60m / 240m). Bumping this
# value is cheap (state files grow linearly, well-compressed by zstd).
MAX_EMA_HISTORY_S = 14400


def _ema_state_path(state_dir: Path, asset: str, date_str: str, hour: int) -> Path:
    return state_dir / f"ema_state_{asset.lower()}_{date_str}_{hour:02d}.parquet"


def _load_prev_ema_history(
    state_dir: str,
    asset: str,
    date_str: str,
    hour: int,
    verbose: bool,
) -> dict:
    """
    Load the EMA history of the immediately preceding hour.

    Returns {feature_name -> pd.Series indexed by bucket_dt_utc}, or empty
    dict if no prior state exists (cold start).

    Backwards compatibility: if a 1-row scalar state file (with
    columns `date_str`, `asset`, `hour`, then EMA cols) is encountered, we
    convert each scalar value into a single-element Series so the engine still
    receives meaningful init values. Slope features that need >1 second of
    history will fall back to NaN until the new schema accumulates.
    """
    if not state_dir:
        return {}

    state_dir_p = Path(state_dir)

    # Find prev (date, hour) -- handles midnight crossing
    prev_date_str, prev_hour = _adjacent_hour(date_str, hour, -1)
    prev_path = _ema_state_path(state_dir_p, asset, prev_date_str, prev_hour)

    if not prev_path.exists():
        _log(verbose, f"  EMA state: no prev-hour file ({prev_path.name}); "
                      f"EMAs will bootstrap from current hour")
        return {}

    try:
        df = pq.read_table(str(prev_path)).to_pandas()
        if df.empty:
            return {}

        # ── Detect schema (1-row scalar vs per-second history) ──────────
        if "bucket_dt_utc" in df.columns:
            # History schema: per-second history, key = bucket_dt_utc
            df["bucket_dt_utc"] = pd.to_datetime(df["bucket_dt_utc"], utc=True)
            df = df.sort_values("bucket_dt_utc")
            df = df.drop_duplicates(subset=["bucket_dt_utc"], keep="last")
            df = df.set_index("bucket_dt_utc")
            out = {}
            for col in df.columns:
                s = df[col].astype("float64")
                # Drop columns that are entirely NaN (no signal)
                if s.dropna().empty:
                    continue
                out[col] = s
            _log(verbose,
                 f"  EMA history: loaded {len(out)} EMA series, "
                 f"{len(df)} rows from {prev_path.name}")
            return out

        # ── 1-row scalar schema (one value per EMA) ──────────────────────
        # We synthesise a single-row Series at an artificial bucket_dt_utc
        # = (target_hour_start - 1s) so the new operator can use it as the
        # init value. Slopes will be NaN until this is replaced by a
        # per-second-history state file from a fresh run.
        row = df.iloc[0]
        meta_cols = {"date_str", "asset", "hour"}

        # Anchor the single scalar value at one second before the target
        # hour starts. The new EMA op only reads `.iloc[-1]` of the history
        # so an arbitrary timestamp before target_start is fine.
        anchor_dt = pd.Timestamp(date_str, tz="UTC") + pd.Timedelta(hours=hour) \
                    - pd.Timedelta(seconds=1)
        out = {}
        for col in df.columns:
            if col in meta_cols:
                continue
            v = row[col]
            if pd.notna(v):
                out[col] = pd.Series([float(v)], index=[anchor_dt], name=col)
        _log(verbose,
             f"  EMA state: compat 1-row schema detected; loaded "
             f"{len(out)} scalar values from {prev_path.name} "
             f"(slopes will be NaN until history schema accumulates)")
        return out

    except Exception as e:
        _log(verbose, f"  WARN: failed to load EMA history {prev_path.name}: {e}")
        return {}


def _save_ema_history(
    state_dir: str,
    asset: str,
    date_str: str,
    hour: int,
    prev_history: dict,
    target_history: dict,
    verbose: bool,
    max_lookback_s: int = MAX_EMA_HISTORY_S,
) -> None:
    """
    Save the rolling EMA history for this hour.

    The on-disk file contains, for each EMA feature, the last `max_lookback_s`
    seconds of EMA values, indexed by bucket_dt_utc. We compose it from:
      - prev_history: dict {name -> Series} loaded at the start of this hour
      - target_history: dict {name -> Series} captured during this hour's
        compute_all (target slice only, to avoid stale next-hour values)

    The two are concatenated by index (timestamp), deduplicated keeping the
    new target values for any overlap, and truncated to the trailing
    max_lookback_s seconds.
    """
    if not state_dir:
        return
    if not target_history and not prev_history:
        return

    state_dir_p = Path(state_dir)
    state_dir_p.mkdir(parents=True, exist_ok=True)

    # Union of feature names across prev + target
    all_names = set(prev_history.keys()) | set(target_history.keys())

    columns: dict = {}
    for name in all_names:
        prev_s   = prev_history.get(name)
        target_s = target_history.get(name)

        parts = []
        if prev_s is not None and len(prev_s) > 0:
            parts.append(prev_s)
        if target_s is not None and len(target_s) > 0:
            parts.append(target_s)

        if not parts:
            continue

        merged = pd.concat(parts).sort_index()
        # On overlap (e.g. compat single-row prev + new target), keep the
        # newer target value
        merged = merged[~merged.index.duplicated(keep="last")]

        # Truncate to last max_lookback_s rows
        if len(merged) > max_lookback_s:
            merged = merged.iloc[-max_lookback_s:]

        columns[name] = merged

    if not columns:
        return

    # Combine into a wide DataFrame indexed by bucket_dt_utc.
    # Outer-join so every column gets all timestamps; missing values stay NaN.
    wide = pd.concat(columns, axis=1)
    wide = wide.sort_index()

    # Truncate the merged frame too (in case some columns had longer history)
    if len(wide) > max_lookback_s:
        wide = wide.iloc[-max_lookback_s:]

    wide = wide.reset_index().rename(columns={"index": "bucket_dt_utc"})
    if "bucket_dt_utc" not in wide.columns:
        # `index` may have been preserved with a different name; rename
        # the first column if so.
        wide = wide.rename(columns={wide.columns[0]: "bucket_dt_utc"})

    out_path = _ema_state_path(state_dir_p, asset, date_str, hour)
    _atomic_write_parquet(wide, out_path)
    n_rows = len(wide)
    n_cols = len(wide.columns) - 1  # minus bucket_dt_utc
    size_kb = out_path.stat().st_size / 1024.0
    _log(verbose, f"  EMA history: saved {n_cols} EMAs x {n_rows} rows "
                  f"({size_kb:.1f} KB) to {out_path.name}")


# =============================================================================
# Build
# =============================================================================

def build_s1_features_for_hour(
    s0_dir: str,
    out_dir: str,
    asset: str,
    date_str: str,
    hour: int,
    features_filter: Optional[List[str]] = None,
    verbose: bool = True,
    use_context: bool = True,
    ohlc_dir: Optional[str] = None,
    weekly_dir: Optional[str] = None,
    monthly_dir: Optional[str] = None,
    vp_dir: Optional[str] = None,
    ema_state_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Main entry point: compute S1 features for one asset-hour.

    Reads S0 features (with optional adjacent-hour context), optionally joins
    daily OHLC, weekly, monthly, and volume-profile context columns, computes
    S1 derived features, writes parquet atomically. The output retains all S0
    columns plus the new S1 columns.

    [STATEFUL-EMA 2026-04-26] If ema_state_dir is provided, EMA state is
    loaded from the previous hour's state file and seeded into the engine
    so EMAs span multiple hours/days. After computation the final EMA
    values for the target hour are written to a new state file for the
    next hour to consume.

    Args:
        s0_dir:     Directory containing S0 feature parquets.
        out_dir:    Directory to write S1 feature parquets.
        asset:      'btc' or 'eth'.
        date_str:   Target date 'YYYY-MM-DD'.
        hour:       Target hour (0-23).
        features_filter: Optional subset of spec names to compute.
        verbose:    Print progress.
        use_context: If True, load prev/next hour context for warmup / lookahead.
        ohlc_dir:   If provided, daily OHLC parquets are loaded and joined
                    (enables dist_to_day_high_bps, range_pos_day, prev_day_*).
        weekly_dir: If provided, weekly-level parquets are loaded and joined
                    (enables dist_to_week_*, range_pos_week, Monday levels,
                    prev_week_*, fibonacci weekly levels).
        monthly_dir: If provided, monthly-level parquets are loaded and joined
                    (enables dist_to_month_*, range_pos_month, prev_month_*,
                    fibonacci monthly levels).
        vp_dir:     If provided, volume-profile parquets are loaded and joined
                    (enables dist_to_poc/vah/val, price_vs_va, poc_migration_*).
    """
    _, out_path = _paths_for_hour(s0_dir, out_dir, asset, date_str, hour)
    _ensure_dir(out_path.parent)
    _log(verbose, f"Building S1 features: {asset} {date_str} hour={hour:02d}")

    # --- Load with context window ---
    context_slice = None
    if use_context:
        combined_df, start_idx, end_idx = _load_with_context(
            s0_dir, asset, date_str, hour, verbose=verbose
        )
        if start_idx > 0 or end_idx < len(combined_df):
            context_slice = (start_idx, end_idx)
    else:
        s0_path, _ = _paths_for_hour(s0_dir, out_dir, asset, date_str, hour)
        if not s0_path.exists():
            raise FileNotFoundError(f"Missing S0 features: {s0_path}")
        combined_df = _read_parquet(str(s0_path))
        _log(verbose, f"Loaded S0 features (no context): {s0_path}")

    # --- Join daily OHLC context (optional) ---
    if ohlc_dir is not None:
        combined_df = _join_ohlc(combined_df, ohlc_dir, asset, verbose=verbose)

    # --- Join weekly levels (optional, Phase 3) ---
    if weekly_dir is not None:
        combined_df = _join_weekly(combined_df, weekly_dir, asset, verbose=verbose)

    # --- Join monthly levels (optional, Phase 3) ---
    if monthly_dir is not None:
        combined_df = _join_monthly(combined_df, monthly_dir, asset, verbose=verbose)

    # --- Join volume profile (optional, Phase 3) ---
    if vp_dir is not None:
        combined_df = _join_volume_profile(combined_df, vp_dir, asset, verbose=verbose)

    # --- Load EMA state (if state dir provided) ---
    # [STATEFUL-EMA] If caller did not pass an ema_state_dir,
    # default to a sibling of out_dir (..../ema_state). This enables
    # stateful EMAs without requiring callers/CLI runners to thread the
    # parameter through. Existing scripts continue to work.
    if ema_state_dir is None:
        ema_state_dir = str(Path(out_dir).parent / "ema_state")

    ema_history = _load_prev_ema_history(
        ema_state_dir, asset, date_str, hour, verbose
    )

    # --- Compute ---
    engine = S1FeatureEngine(verbose=verbose, ema_history=ema_history)
    df = engine.compute_all(
        combined_df,
        features_filter=features_filter,
        context_slice=context_slice,
    )

    # --- Save EMA history (for next hour to consume) ---
    _save_ema_history(
        ema_state_dir, asset, date_str, hour,
        prev_history=ema_history,
        target_history=engine.get_ema_target_history(),
        verbose=verbose,
    )

    # --- Atomic parquet write ---
    _log(verbose, f"Saving S1 features to: {out_path}")
    _atomic_write_parquet(df, out_path)

    mb = out_path.stat().st_size / (1024 * 1024)
    _log(verbose, f"Saved: {mb:.2f} MB | rows={len(df)} cols={len(df.columns)}")
    return df


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="S1 feature engine: compute S1 features from S0 parquets."
    )
    ap.add_argument("--s0-dir", type=str, default=str(_DEFAULT_S0_DIR))
    ap.add_argument("--out-dir", type=str, default=str(_DEFAULT_OUT_DIR))
    ap.add_argument("--asset", type=str, required=True, choices=["btc", "eth", "bnb"])
    ap.add_argument("--date", type=str, required=True)
    ap.add_argument("--hour", type=int, required=True)
    ap.add_argument("--features", type=str, nargs="+")
    ap.add_argument("--quiet", "-q", action="store_true")
    ap.add_argument(
        "--ohlc-dir", type=str, default=None,
        help=(
            "Directory containing daily OHLC parquets generated by "
            "generate_ohlc.py. If provided, daily range features "
            "(dist_to_day_high_bps, prev_day_*) are computed. "
            "If absent, those features degrade to NaN."
        ),
    )
    ap.add_argument(
        "--weekly-dir", type=str, default=None,
        help=(
            "Directory containing weekly-level parquets generated by "
            "generate_weekly_levels.py. If provided, week-level features "
            "(dist_to_week_*, range_pos_week, monday_*, prev_week_*, "
            "weekly fibonacci levels) are computed. If absent, NaN."
        ),
    )
    ap.add_argument(
        "--monthly-dir", type=str, default=None,
        help=(
            "Directory containing monthly-level parquets generated by "
            "generate_monthly_levels.py. If provided, month-level features "
            "(dist_to_month_*, range_pos_month, prev_month_*, "
            "monthly fibonacci levels) are computed. If absent, NaN."
        ),
    )
    ap.add_argument(
        "--vp-dir", type=str, default=None,
        help=(
            "Directory containing volume-profile parquets generated by "
            "generate_volume_profile.py. If provided, VP features "
            "(dist_to_poc/vah/val, price_vs_va) are computed and "
            "poc_migration_* passthroughs are injected. If absent, NaN."
        ),
    )
    ap.add_argument(
        "--ema-state-dir", type=str, default=None,
        help=(
            "Directory for EMA state persistence (stateful EMAs across "
            "hour boundaries). Reads ema_state_{asset}_{date}_{hour-1}.parquet "
            "as init state, writes new state file after computation. "
            "If absent, EMAs bootstrap fresh each hour-build (fine for "
            "short spans; long-span EMAs will be NaN)."
        ),
    )
    ap.add_argument(
        "--no-context", action="store_true",
        help="Disable context window (do not load adjacent hours). "
             "Faster but 3600s features will have warmup NaN.",
    )

    args = ap.parse_args()
    verbose = not args.quiet

    build_s1_features_for_hour(
        s0_dir=args.s0_dir,
        out_dir=args.out_dir,
        asset=args.asset,
        date_str=args.date,
        hour=args.hour,
        features_filter=args.features,
        verbose=verbose,
        use_context=not args.no_context,
        ohlc_dir=args.ohlc_dir,
        weekly_dir=args.weekly_dir,
        monthly_dir=args.monthly_dir,
        vp_dir=args.vp_dir,
        ema_state_dir=args.ema_state_dir,
    )


if __name__ == "__main__":
    main()