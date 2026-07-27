# ==============================================================================
# S2 Feature Engine — Binance-only, Multi-Asset (BTC + ETH + BNB)
#
# PURPOSE:
#   Compute S2 derived features from S1 feature parquets. S2 features are
#   temporal aggregations, statistical transforms, and cross-market metrics
#   built on top of the S1 feature columns.
#
# CONTRACT:
#   - Input:  S1 feature parquets from /data_storage/s1_features/
#             (these contain bucket_dt_utc + S0 columns + S1 columns)
#   - Output: S2 feature parquets to /data_storage/s2_features/
#             (these contain bucket_dt_utc + S0 + S1 + S2 columns)
#   - Each stage EXTENDS the DataFrame by adding new columns.
#     Previous-stage columns (S0, S1) are RETAINED in the output.
#   - The S2 output file supersedes the S1 input file. After successful
#     S2 computation the S1 file is archived (it's now redundant because
#     the S2 file contains all S1 data plus the new S2 features).
#   - The output file contains ALL columns: bucket_dt_utc + S0 + S1 + S2.
#
# TOPOLOGICAL SORT:
#   Some S2 features depend on other S2 features (intra-stage dependencies):
#     - d2_* depends on d1_* (second temporal difference)
#     - shock_* depends on median_* and mad_* (outlier detection)
#   The engine topologically sorts all specs before computation so that
#   dependencies are always available when needed. Max depth is 1 level.
#
#   of S2 feature names — regardless of dep.kind label. This mirrors the S3
#   engine and fixes silent ordering failures when specs label their intra-
#   stage deps as kind="col" instead of kind="s2".
#
# FILL CONTRACT:
#   - Rolling operators: first (window_s - 1) rows → NaN (insufficient history)
#   - Temporal diffs (d1, d2): first row → NaN
#   - Ratio/division operators: denom == 0 → NaN
#   - All operators propagate NaN from inputs (NaN in → NaN out).
#
# CONTEXT WINDOW (added to match S1 behavior):
#   Hourly files are 3600 rows (1 per second). S2 has rolling windows and
#   sometimes "needs" continuity across hour boundaries to avoid edge NaNs
#   (especially for large windows like 900s/1800s/3600s and any features that
#   rely on stable rolling statistics).
#
#   The engine can load up to 1 hour before (lookback) and 1 hour after
#   (lookahead) as context. Computation runs on the concatenated DataFrame,
#   then the result is sliced back to the target hour before writing.
#
#   Note: S2 does not currently contain explicit forward-shift operators in
#   this file, but we keep lookahead parity with S1 for consistency and to
#   support any future S2 forward-label features without redesign.
#
# POST-BUILD ARCHIVE:
#   After successful S2 feature computation the engine moves consumed S1
#   feature files into a date-partitioned archive directory:
#       data_storage/data_archive/{date_str}/s1_features/
#   This keeps s1_features/ clean for the next hour's pipeline.
#
# FIXES APPLIED vs. original:
#   [FIX-PRODUCT] Added derived.product operator: element-wise col_a * col_b.
#                 Required for flow_depth_align features (taker_imbalance ×
#                 depth_imbalance). NaN propagates from either input.
#           detection changed from dep.kind=="s2" to name-matching against
#           the full spec-name set. Robust to specs that use kind="col"
#           uniformly (which most S2 specs do).
#   [FIX-2] _op_roll_mean: handles multiple deps by element-wise averaging
#           before rolling. Fixes aggressor_absorption_ratio_* specs that
#           pass 2 deps (ask side + bid side) to derived.roll_mean.
#   [FIX-3] _op_impact_per_liquidity: column detection by name prefix
#           (ret_*, volume_/vol_*, depth_*ask, depth_*bid) instead of
#           positional. Fixes s2_impact.py dep ordering (depth_ask, depth_bid,
#           ret, vol) which is opposite to what positional code assumed.
#   [FIX-4] _op_impact_per_signed: column detection by name prefix
#           (ret_*, signed_vol*/taker_imbalance*). Handles both 2-dep and
#           3-dep specs without crashing or producing wrong values.
#   [FIX-5] _op_ret_vwap: column detection by name prefix (vwap*, mid*).
#           Fixes specs that supply (mid, vwap) while old code assumed (vwap, mid).
#   [FIX-6] _op_taker_imbalance_bucket: changed min_periods=1 → window_s
#           to enforce FILL CONTRACT (first window_s-1 rows → NaN).
#   [FIX-7] _op_dir_consistency: replaced same_sign.iloc[0] = np.nan with
#           .where() to avoid SettingWithCopyWarning in Pandas >= 2.0.
#   [FIX-8] _op_robust_zscore: zero-MAD protection — when rolling MAD = 0
#           (flat/sparse inputs like binary flags, pull_rate=0 in quiet
#           seconds, near-zero returns), return NaN instead of dividing by
#           EPS=1e-12 which produced extreme values (~1e9) failing the
#           zscore_extreme audit check. Also added multi-dep averaging so
#           specs like z_participation_rate_* (fut+spot deps) are handled
#           correctly.
# ==============================================================================

from __future__ import annotations
from common.paths import DATA_ROOT

import argparse
import math
import os
import shutil
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
#              loop — prevents stale carryover on silent compute failures.
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from etl.spec import FeatureSpec, Dep
from etl.operators.s2_operators import S2_OPERATORS

# ── S2 Spec Imports ──────────────────────────────────────────────────
from etl.spec.s2.s2_absorption import S2_ABSORPTION_FEATURES
from etl.spec.s2.s2_activity import S2_ACTIVITY_FEATURES
from etl.spec.s2.s2_aggression import S2_AGGRESSION_FEATURES
from etl.spec.s2.s2_bookshape import S2_BOOKSHAPE_FEATURES
from etl.spec.s2.s2_cross_market import S2_CROSS_MARKET_FEATURES
from etl.spec.s2.s2_dynamics import S2_DYNAMICS_FEATURES
from etl.spec.s2.s2_impact import S2_IMPACT_FEATURES
from etl.spec.s2.s2_liquidity_events import S2_LIQUIDITY_EVENTS_FEATURES
from etl.spec.s2.s2_meta import S2_META_FEATURES
from etl.spec.s2.s2_normalization import S2_NORMALIZATION_FEATURES
from etl.spec.s2.s2_pressure import S2_PRESSURE_FEATURES
from etl.spec.s2.s2_price import S2_PRICE_FEATURES
from etl.spec.s2.s2_returns import S2_RETURNS_FEATURES

PARQUET_COMPRESSION = "zstd"
EPS = 1e-12

_ENGINE_DIR = Path(__file__).resolve().parent
_DEFAULT_S1_DIR = DATA_ROOT / "s1_features"
_DEFAULT_OUT_DIR = DATA_ROOT / "s2_features"
_DEFAULT_ARCHIVE_DIR = DATA_ROOT / "data_archive"


# =============================================================================
# Utilities
# =============================================================================

def _log(enabled: bool, msg: str) -> None:
    if enabled:
        print(f"[{pd.Timestamp.utcnow().isoformat()}] [S2_FEATURE_ENGINE] {msg}")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _require_cols(df: pd.DataFrame, cols: Iterable[str], ctx: str) -> None:
    """Verify required columns exist in DataFrame, raise descriptive error if not."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{ctx}: missing required columns: {missing}. "
                         f"Have {len(df.columns)} cols, first 20: {list(df.columns)[:20]}")


def _safe_int(val: Any, default: int = 1) -> int:
    """Safely convert value to int."""
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def _safe_float(val: Any, default: float = 0.0) -> float:
    """Safely convert value to float."""
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


# =============================================================================
# Topological Sort
# =============================================================================

def _toposort_specs(specs: List[FeatureSpec]) -> List[FeatureSpec]:
    """
    Topologically sort feature specs so that intra-S2 dependencies are
    computed before the features that depend on them.

    [FIX-1] Intra-S2 deps are detected by NAME MATCHING against the full set
    of S2 feature names — regardless of the dep.kind label. This mirrors the
    S3 engine's approach and fixes silent ordering failures when specs label
    their intra-stage deps as kind="col" instead of kind="s2".

    Algorithm: Kahn's algorithm (BFS-based topological sort).
    """
    # Build name -> spec index mapping
    name_to_idx: Dict[str, int] = {}
    for i, s in enumerate(specs):
        name_to_idx[s.name] = i

    # Build adjacency: for each spec, which other specs must come first?
    # in_degree[i] = number of S2 deps spec[i] has that are in this spec list
    in_degree = [0] * len(specs)
    # dependents[i] = list of spec indices that depend on spec[i]
    dependents: Dict[int, List[int]] = defaultdict(list)

    for i, s in enumerate(specs):
        for dep in s.depends_on:
            # [FIX-1] Name-matching: robust to any dep.kind label.
            # Old code: if dep.kind == "s2" and dep.name in name_to_idx
            if dep.name in name_to_idx and dep.name != s.name:
                dep_idx = name_to_idx[dep.name]
                in_degree[i] += 1
                dependents[dep_idx].append(i)

    # BFS: start with specs that have no intra-S2 dependencies
    queue = [i for i in range(len(specs)) if in_degree[i] == 0]
    sorted_indices: List[int] = []

    while queue:
        # Sort the queue by feature_id for deterministic ordering
        queue.sort(key=lambda idx: specs[idx].feature_id or 0)
        current = queue.pop(0)
        sorted_indices.append(current)

        for dep_idx in dependents[current]:
            in_degree[dep_idx] -= 1
            if in_degree[dep_idx] == 0:
                queue.append(dep_idx)

    # Cycle detection
    if len(sorted_indices) != len(specs):
        remaining = [specs[i].name for i in range(len(specs)) if i not in sorted_indices]
        raise ValueError(
            f"Topological sort failed: cycle detected among {len(remaining)} specs. "
            f"First 10: {remaining[:10]}"
        )

    return [specs[i] for i in sorted_indices]


# =============================================================================
# S2 Feature Engine
# =============================================================================

class S2FeatureEngine:
    """
    Compute S2 features from S1 feature columns.

    The engine loads the S1 feature parquet into a wide DataFrame, then
    iterates through topologically-sorted specs, computing each feature
    and adding it as a new column. Intra-S2 dependencies (d2→d1, shock→mad)
    are resolved by the topological ordering.
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self._op_registry = S2_OPERATORS

    # =========================================================================
    # Registry Validation
    # =========================================================================

    def _validate_registry(self, specs: List[FeatureSpec]) -> None:
        """
        Pre-compute validation: every spec.operator must exist in registry.
        Arity mismatch raises ValueError.
        """
        for spec in specs:
            op = spec.operator
            if op not in self._op_registry:
                raise ValueError(
                    f"S2 registry validation failed: unknown operator '{op}' "
                    f"used by feature '{spec.name}' (id={spec.feature_id})"
                )
            reg = self._op_registry[op]
            actual = len(spec.depends_on)
            expected = reg.n_input_cols
            if expected > 0 and actual != expected:
                raise ValueError(
                    f"S2 arity mismatch for '{spec.name}': operator '{op}' "
                    f"expects {expected} inputs, spec has {actual} deps"
                )

    # =========================================================================
    # Main Entry: Compute All
    # =========================================================================

    def compute_all(
        self,
        s1_df: pd.DataFrame,
        specs: List[FeatureSpec],
        features_filter: Optional[List[str]] = None,
        context_slice: Optional[Tuple[int, int]] = None,
    ) -> pd.DataFrame:
        """
        Compute all S2 features on top of the S1 feature DataFrame.

        Args:
            s1_df:            DataFrame with bucket_dt_utc + S0 + S1 feature columns.
                              May include context rows from adjacent hours.
            specs:            List of S2 FeatureSpec objects.
            features_filter:  Optional subset of feature names to compute.
            context_slice:    Optional (start_idx, end_idx) tuple indicating the
                              target hour's rows within s1_df. If provided, the
                              returned DataFrame is sliced to only these rows
                              after computation completes. Context rows are used
                              for rolling warmup but not included in output.

        Returns:
            Wide DataFrame with bucket_dt_utc + S0 + S1 + S2 columns.
            If context_slice is provided, only the target rows are returned.
        """
        _require_cols(s1_df, ["bucket_dt_utc"], "s1_df")

        df = s1_df.copy()
        df = df.sort_values("bucket_dt_utc").reset_index(drop=True)
        df["bucket_dt_utc"] = pd.to_datetime(df["bucket_dt_utc"], utc=True)
        # --- Stale column guard ---
        # Drop any output columns that already exist in the input DataFrame.
        # Prevents stale values from a previous run from persisting when a
        # compute silently fails (WARN path) — without this, the old value
        # stays in df and is written to output as if freshly computed.
        _output_names = {s.name for s in specs}
        _stale = [c for c in _output_names if c in df.columns]
        if _stale:
            _log(self.verbose,
                 f"Stale column guard: dropping {len(_stale)} pre-existing "
                 f"output col(s) from input df to prevent carryover.")
            df = df.drop(columns=_stale)


        # Filter specs if requested
        if features_filter:
            wanted = set(features_filter)
            specs = [s for s in specs if s.name in wanted]

        # --- Registry validation (before compute loop) ---
        # Comment: Fail fast on unknown operators or wrong dependency arity.
        self._validate_registry(specs)

        # Topological sort for intra-S2 dependency resolution
        # Comment: Ensures d1_* exists before d2_*, median_/mad_ exist before shock_*.
        # [FIX-1] toposort now uses name-matching instead of dep.kind=="s2".
        sorted_specs = _toposort_specs(specs)
        _log(self.verbose, f"Computing S2 features: {len(sorted_specs)} specs "
             f"(toposorted from {len(specs)} input specs)")

        t0 = time.time()
        computed, errors = 0, 0
        s2_feature_names: List[str] = []

        for spec in sorted_specs:
            try:
                result = self._compute_one(spec, df)
                df[spec.name] = result
                s2_feature_names.append(spec.name)
                computed += 1
            except Exception as e:
                errors += 1
                if self.verbose:
                    print(f"  [WARN] {spec.name}: {e}")

        elapsed = time.time() - t0
        _log(self.verbose, f"Done. computed={computed} errors={errors} in {elapsed:.2f}s "
             f"| total cols={len(df.columns)} (S0/S1 retained + {len(s2_feature_names)} new S2)")

        # --- Slice back to the target hour if a context window was used ---
        # ---------------------------------------------------------------------
        # CONTEXT-WINDOW MECHANISM (same idea as S1):
        #
        # We compute S2 features on a concatenated timeline:
        #   [prev_hour] + [target_hour] + [next_hour]
        #
        # This stabilizes rolling features (mean/std/median/MAD/autocorr/etc.)
        # near hour boundaries. After computation finishes, we cut back to the
        # exact row-range for the target hour to keep "one hour per file".
        # ---------------------------------------------------------------------
        if context_slice is not None:
            start, end = context_slice
            _log(self.verbose,
                 f"Slicing context: rows [{start}:{end}] "
                 f"({end - start} target rows from {len(df)} total)")
            df = df.iloc[start:end].reset_index(drop=True)

        # Return full DataFrame: bucket_dt_utc + S0 + S1 + S2 columns (target rows only).
        return df

    # =========================================================================
    # Compute One Feature
    # =========================================================================

    def _compute_one(self, spec: FeatureSpec, df: pd.DataFrame) -> pd.Series:
        """
        Dispatch to the appropriate operator implementation.

        Args:
            spec: The feature specification.
            df:   The working DataFrame (S1 columns + already-computed S2 columns).

        Returns:
            A pd.Series with the computed feature values, aligned to df's index.
        """
        op = spec.operator
        params = spec.params
        deps = spec.depends_on
        name = spec.name
        window_s = _safe_int(params.get("window_s", 0))

        # Resolve dependency column names
        dep_names = [d.name for d in deps]

        # Validate dependencies exist in the DataFrame
        _require_cols(df, dep_names, name)

        # ── Dispatch by operator ─────────────────────────────────────

        # === ROLLING AGGREGATION ===
        if op == "derived.roll_mean":
            min_p = _safe_int(params.get("min_periods", window_s))
            return self._op_roll_mean(df, dep_names, window_s, name, min_p)

        if op == "derived.roll_sum":
            return self._op_roll_sum(df, dep_names, window_s, name)

        if op == "derived.roll_median":
            return self._op_roll_median(df, dep_names, window_s, name)

        # === ROLLING STATISTICS ===
        if op == "derived.mad":
            return self._op_mad(df, dep_names, window_s, name)

        if op == "derived.robust_zscore":
            # accept an optional min_periods override from spec params.
            # Needed for sparse signals like avg_trade_size_*_60s where the
            # 1s input is NaN ~5% of the time (seconds without trades) and
            # the strict default min_periods=window_s drives downstream to
            # 100% NaN.
            min_p = _safe_int(params.get("min_periods", window_s))
            return self._op_robust_zscore(df, dep_names, window_s, name, min_periods=min_p)

        if op == "derived.shock_detect":
            return self._op_shock_detect(df, dep_names, window_s, name)

        if op == "derived.shock":
            return self._op_shock(df, dep_names, window_s, name)

        if op == "derived.ofi_shock":
            return self._op_ofi_shock(df, dep_names, window_s, name)

        # === TEMPORAL DERIVATIVES ===
        if op == "derived.d1":
            return self._op_d1(df, dep_names, name)

        if op == "derived.d2":
            return self._op_d2(df, dep_names, name)

        # === ARITHMETIC ===
        if op == "derived.sub":
            return self._op_sub(df, dep_names, name)

        if op == "derived.product":
            return self._op_product(df, dep_names, name)

        if op == "derived.ratio":
            return self._op_ratio(df, dep_names, name)

        if op == "derived.asymmetry":
            return self._op_asymmetry(df, dep_names, name)

        # === ABSORPTION ===
        if op == "l2.absorb_refill_ask":
            return self._op_absorb_refill(df, dep_names, name)

        if op == "l2.absorb_refill_bid":
            return self._op_absorb_refill(df, dep_names, name)

        # === LIQUIDITY EVENTS ===
        if op == "l2.churn_ask" or op == "l2.churn_bid":
            return self._op_churn(df, dep_names, name)

        if op == "l2.net_add_pressure":
            return self._op_net_add_pressure(df, dep_names, name)

        if op == "l2.net_cancel_pressure":
            return self._op_net_cancel_pressure(df, dep_names, name)

        if op == "l2.pull_rate":
            return self._op_pull_rate(df, dep_names, window_s, name)

        if op == "l2.refill_rate":
            return self._op_refill_rate(df, dep_names, window_s, name)

        if op == "l2.refill_rate_behind":
            return self._op_refill_rate_behind(df, dep_names, name)

        if op == "l2.cancel_rate_ahead":
            return self._op_cancel_rate_directional(df, dep_names, name)

        if op == "l2.cancel_rate_behind":
            return self._op_cancel_rate_directional(df, dep_names, name)

        # === CROSS-MARKET ===
        if op == "derived.basis_vwap":
            return self._op_basis_vwap(df, dep_names, name)

        if op == "deriv.queue_pressure_log_div":
            return self._op_queue_pressure_log_div(df, dep_names, name)

        if op == "derived.z_volume_asym":
            return self._op_z_volume_asym(df, dep_names, window_s, name)

        # === TRADE OPERATORS ===
        if op == "trades.trade_absorption_ratio_1s":
            return self._op_trade_absorption_ratio(df, dep_names, name)

        if op == "trades.taker_imbalance_bucket":
            return self._op_taker_imbalance_bucket(df, dep_names, window_s, name)

        # === PERSISTENCE / AUTOCORRELATION ===
        if op == "derived.autocorr":
            return self._op_autocorr(df, dep_names, window_s, name)

        # === IMPACT ===
        if op == "derived.impact_per_liquidity":
            return self._op_impact_per_liquidity(df, dep_names, window_s, name)

        if op == "derived.impact_per_signed":
            return self._op_impact_per_signed(df, dep_names, window_s, name)

        # === META / REGIME ===
        if op == "derived.breakout_regime_flag":
            return self._op_breakout_regime_flag(df, dep_names, window_s, name)

        if op == "derived.dir_consistency":
            return self._op_dir_consistency(df, dep_names, window_s, name)

        if op == "derived.unidir_ratio":
            return self._op_unidir_ratio(df, dep_names, window_s, name)

        if op == "derived.depth_coherence":
            return self._op_depth_coherence(df, dep_names, window_s, name)

        if op == "derived.depth_slope":
            return self._op_depth_slope(df, dep_names, name)

        if op == "derived.depth_curvature":
            return self._op_depth_curvature(df, dep_names, name)

        # === PRICE ===
        if op == "derived.mid_touch_dev":
            return self._op_mid_touch_dev(df, dep_names, name)

        if op == "derived.price_acceleration":
            return self._op_price_acceleration(df, dep_names, window_s, name)

        if op == "derived.price_deviation_bps":
            return self._op_price_deviation_bps(df, dep_names, name)

        if op == "derived.ret_vwap":
            return self._op_ret_vwap(df, dep_names, window_s, name)

        if op == "derived.z_rv":
            return self._op_z_rv(df, dep_names, window_s, name)

        raise ValueError(f"{name}: unknown S2 operator '{op}'")

    # =====================================================================
    # OPERATOR IMPLEMENTATIONS
    # =====================================================================

    # ── Rolling Aggregation ──────────────────────────────────────────

    def _op_roll_mean(self, df: pd.DataFrame, deps: List[str],
                      window_s: int, name: str,
                      min_periods: Optional[int] = None) -> pd.Series:
        """
        Rolling mean over window_s rows.

        [FIX-2] Handles multiple deps: if more than one dep column is supplied,
        they are averaged element-wise first, then a rolling mean is applied.
        This supports specs like aggressor_absorption_ratio_* that supply both
        the ask-side and bid-side columns to be averaged before smoothing.
        For the common single-dep case behaviour is identical to before.

        [FIX-MINP] min_periods: optional override from FeatureSpec params.
                   Default = window_s (strict: require full window).
                   Use min_periods < window_s for sparse signals like
                   avg_trade_size where NaN input seconds are expected —
                   the rolling mean is computed over however many non-NaN
                   rows are available within the window.
        """
        min_p = min_periods if min_periods is not None else window_s
        if len(deps) == 1:
            col = df[deps[0]].astype("float64")
        else:
            # Element-wise mean across all dep columns, then roll.
            col = pd.concat(
                [df[d].astype("float64") for d in deps], axis=1
            ).mean(axis=1)
        return col.rolling(window=window_s, min_periods=min_p).mean()

    def _op_roll_sum(self, df: pd.DataFrame, deps: List[str],
                     window_s: int, name: str) -> pd.Series:
        """Rolling sum of first dependency column over window_s rows."""
        col = df[deps[0]].astype("float64")
        return col.rolling(window=window_s, min_periods=window_s).sum()

    def _op_roll_median(self, df: pd.DataFrame, deps: List[str],
                        window_s: int, name: str) -> pd.Series:
        """Rolling median of first dependency column over window_s rows."""
        col = df[deps[0]].astype("float64")
        return col.rolling(window=window_s, min_periods=window_s).median()

    # ── Rolling Statistics ───────────────────────────────────────────

    def _op_mad(self, df: pd.DataFrame, deps: List[str],
                window_s: int, name: str) -> pd.Series:
        """
        Median Absolute Deviation: median(|x - median(x)|) over window.

        The deps may contain the raw column and/or the pre-computed median.
        We always compute MAD from the raw column (first dep that is NOT
        a median_ column, or the first dep if ambiguous).
        """
        # Find the raw column (not the median column)
        raw_col_name = deps[0]
        for d in deps:
            if not d.startswith("median_"):
                raw_col_name = d
                break

        col = df[raw_col_name].astype("float64")
        rolling_med = col.rolling(window=window_s, min_periods=window_s).median()
        abs_dev = (col - rolling_med).abs()
        return abs_dev.rolling(window=window_s, min_periods=window_s).median()

    def _op_robust_zscore(self, df: pd.DataFrame, deps: List[str],
                          window_s: int, name: str,
                          min_periods: Optional[int] = None) -> pd.Series:
        """
        Robust z-score: (x - rolling_median) / (1.4826 * rolling_mad).

        [FIX-8] Zero-MAD protection: when rolling MAD = 0 (flat or sparse
        inputs — e.g. binary participation flags, pull_rate=0 in quiet
        seconds, near-zero returns in most 1s buckets), the z-score is
        mathematically undefined. We return NaN instead of dividing by
        EPS=1e-12, which previously produced extreme values (~1e9) that
        failed the zscore_extreme audit check in every file.

        [FIX-8] Multi-dep averaging: when more than one raw dep column is
        supplied (e.g. z_participation_rate_* with fut+spot deps), the
        columns are averaged element-wise before computing the z-score.
        This matches the intent of the spec (cross-market combined signal).

        [FIX-8b] Output clip to [-20, 20]: eliminates residual extreme values
        that arise from tiny-but-nonzero MAD. Example: a window like
        [0, 0, 0, 0.001, 0.1] has MAD=0.001 (non-zero, so passes the
        zero-MAD guard), but yields z ≈ 66 for the spike. For ML features
        |z| > 20 is never informative and is a pure artefact of small
        rolling windows (5s, 15s) over sparse signals like pull_rate,
        queue_pressure, and participation_rate.

        [FIX-MINP-S2 2026-04-27] min_periods override: for sparse signals
        like avg_trade_size_*_60s (driven by 1s inputs whose ~5% of seconds
        without trades produce NaN), the default min_periods=window_s
        produces 100% NaN downstream. Set min_periods=3 (or similar small
        value) in the spec params to compute z-scores from the available
        non-NaN samples in the window. Mirrors the same parameter in S3.
        """
        min_p = min_periods if min_periods is not None else window_s
        # Identify raw (non-auxiliary) dep columns — skip any pre-computed
        # median_ or mad_ columns if they happen to be listed as deps.
        raw_deps = [d for d in deps
                    if not d.startswith("median_") and not d.startswith("mad_")]
        if not raw_deps:
            raw_deps = [deps[0]]

        # [FIX-8] Average multiple raw dep columns element-wise, or use single
        # column directly (common single-dep case: identical to before).
        if len(raw_deps) == 1:
            col = df[raw_deps[0]].astype("float64")
        else:
            col = pd.concat(
                [df[d].astype("float64") for d in raw_deps], axis=1
            ).mean(axis=1)

        rolling_med = col.rolling(window=window_s, min_periods=min_p).median()
        abs_dev = (col - rolling_med).abs()
        rolling_mad = abs_dev.rolling(window=window_s, min_periods=min_p).median()

        # [FIX-8] Zero-MAD → NaN: MAD=0 means the window is completely flat
        # (no dispersion), so the z-score is undefined. Using EPS here would
        # amplify even tiny floating-point noise by ~1e12, producing garbage.
        scale = 1.4826 * rolling_mad
        scale = scale.where(scale > 0, np.nan)

        result = (col - rolling_med) / scale

        # [FIX-8b] Clip to [-20, 20]: guards against tiny-but-nonzero MAD
        # producing extreme values (e.g. one spike in a mostly-zero 5s window).
        return result.clip(-20, 20)

    def _op_shock_detect(self, df: pd.DataFrame, deps: List[str],
                         window_s: int, name: str) -> pd.Series:
        """
        Shock detector: (x - median) / (mad + eps).

        Deps contain the raw column, the median column, and the MAD column
        (all pre-computed via intra-S2 toposort). We resolve which dep is
        which by checking column name prefixes.

        [FIX-SHOCK] Zero-MAD protection: MAD=0 when the window is flat
                    (e.g. pull_rate=0 in all rows of a quiet 1bps band).
                    Dividing by EPS=1e-12 previously produced ~1e9 values
                    triggering extreme_shock warnings in every file.
                    Return NaN instead — undefined, not a near-zero divisor.
        [FIX-SHOCK] Output clipped to [-50, 50]: genuine liquidity shocks
                    can score 20–50σ; values beyond ±50 are artefacts of
                    near-zero MAD in sparse signals (narrow bps bands at 1s).
        """
        raw_col = None
        median_col = None
        mad_col = None

        for d in deps:
            if d.startswith("mad_"):
                mad_col = d
            elif d.startswith("median_"):
                median_col = d
            else:
                raw_col = d

        if raw_col is None or median_col is None or mad_col is None:
            raise ValueError(f"{name}: shock_detect requires raw, median_, and mad_ deps. "
                             f"Got: {deps}")

        x   = df[raw_col].astype("float64")
        med = df[median_col].astype("float64")
        mad = df[mad_col].astype("float64")

        # [FIX-SHOCK] Zero-MAD → NaN
        scale = 1.4826 * mad
        scale = scale.where(scale > 0, np.nan)

        # [FIX-SHOCK] Clip to [-50, 50]
        return ((x - med) / scale).clip(-50, 50)

    def _op_shock(self, df: pd.DataFrame, deps: List[str],
                  window_s: int, name: str) -> pd.Series:
        """
        Joint shock: combined return shock measure across markets.
        Deps: [ret_a, ret_b]. Compute: sqrt(ret_a^2 + ret_b^2).
        """
        a = df[deps[0]].astype("float64")
        b = df[deps[1]].astype("float64")
        return np.sqrt(a ** 2 + b ** 2)

    def _op_ofi_shock(self, df: pd.DataFrame, deps: List[str],
                      window_s: int, name: str) -> pd.Series:
        """
        Order Flow Imbalance shock: (x - rolling_mean) / rolling_std.
        Detects sudden spikes in taker imbalance relative to recent history.

        [FIX-OFI-STD-DENOM 2026-04-25] Earlier formulation was
            (col - rolling_mean) / (rolling_std + EPS)
        When rolling_std = 0 (constant col, e.g. flatlined feed) the EPS
        produced huge spikes if (col - rolling_mean) was nonzero. Now
        treats zero std as undefined (NaN). With 0-std, col == mean by
        definition, so a true shock score is undefined.
        """
        col = df[deps[0]].astype("float64")
        rolling_mean = col.rolling(window=window_s, min_periods=window_s).mean()
        rolling_std = col.rolling(window=window_s, min_periods=window_s).std()
        denom = rolling_std.where(rolling_std.abs() > EPS, np.nan)
        return ((col - rolling_mean) / denom).clip(-100, 100)

    # ── Temporal Derivatives ─────────────────────────────────────────

    def _op_d1(self, df: pd.DataFrame, deps: List[str],
               name: str) -> pd.Series:
        """First temporal difference: x_t - x_{t-1}. First row → NaN."""
        col = df[deps[0]].astype("float64")
        return col.diff(periods=1)

    def _op_d2(self, df: pd.DataFrame, deps: List[str],
               name: str) -> pd.Series:
        """
        Second temporal difference: d1_t - d1_{t-1}.

        Deps: [d1_col, raw_col]. The d1 column is an intra-S2 dependency
        that has already been computed thanks to topological sort.
        We take the diff of the d1 column.
        """
        # Find the d1 dependency
        d1_col = None
        for d in deps:
            if d.startswith("d1_"):
                d1_col = d
                break

        if d1_col is None:
            raise ValueError(f"{name}: d2 operator requires a d1_ dependency. Got: {deps}")

        return df[d1_col].astype("float64").diff(periods=1)

    # ── Arithmetic ───────────────────────────────────────────────────

    def _op_sub(self, df: pd.DataFrame, deps: List[str],
                name: str) -> pd.Series:
        """Difference: col_a - col_b."""
        a = df[deps[0]].astype("float64")
        b = df[deps[1]].astype("float64")
        return a - b

    def _op_product(self, df: pd.DataFrame, deps: List[str],
                    name: str) -> pd.Series:
        """
        Element-wise product: col_a * col_b.

        Used for flow-depth alignment features:
            flow_depth_align = taker_imbalance * depth_imbalance
        Result range [-1, 1]:
            > 0 → aggressive flow and passive book agree (reinforcing signal)
            < 0 → flow contradicts book structure (absorption / reversal signal)
            NaN propagated from either input.
        """
        a = df[deps[0]].astype("float64")
        b = df[deps[1]].astype("float64")
        return a * b

    def _op_ratio(self, df: pd.DataFrame, deps: List[str],
                  name: str) -> pd.Series:
        """
        Ratio: col_a / col_b (with NaN where denom ≈ 0).
        If more than 2 deps, interpret as pairs or use first two.
        """
        a = df[deps[0]].astype("float64")
        b = df[deps[1]].astype("float64")
        denom = b.where(b.abs() > EPS, np.nan)
        return a / denom

    def _op_asymmetry(self, df: pd.DataFrame, deps: List[str],
                      name: str) -> pd.Series:
        """(ask - bid) / (ask + bid + eps). Range [-1, +1]."""
        a = df[deps[0]].astype("float64")
        b = df[deps[1]].astype("float64")
        return (a - b) / (a + b + EPS)

    # ── Absorption ───────────────────────────────────────────────────

    def _op_absorb_refill(self, df: pd.DataFrame, deps: List[str],
                          name: str) -> pd.Series:
        """
        Absorption * refill: add_rate * taker_vol.
        Deps: [add_rate_col, taker_vol_col].
        """
        a = df[deps[0]].astype("float64")
        b = df[deps[1]].astype("float64")
        return a * b

    # ── Liquidity Events ─────────────────────────────────────────────

    def _op_churn(self, df: pd.DataFrame, deps: List[str],
                  name: str) -> pd.Series:
        """
        Churn: (add_rate + cancel_rate) / depth.
        Deps: [add_rate, cancel_rate, depth_notional].
        Measures order turnover intensity relative to standing depth.

        [FIX-CHURN-CLIP-REV 2026-04-25] Three cases distinguished:
          depth ~= 0 AND (add+cancel) ~= 0 -> 0.0  (defined: empty book,
                                                  no activity = quiet state)
          depth ~= 0 AND (add+cancel) > 0  -> NaN  (impossible: orderbook
                                                  activity with zero depth)
          depth > 0                        -> (add+cancel)/depth, clipped
                                                  to [0, 1e6]
        Earlier "+EPS" formulation produced ~1e19 spikes when depth was 0.
        """
        add = df[deps[0]].astype("float64")
        cancel = df[deps[1]].astype("float64")
        if len(deps) >= 3:
            depth = df[deps[2]].astype("float64")
            num = add + cancel
            den_pos = depth.abs() > EPS
            num_pos = num.abs() > EPS
            denom = depth.where(den_pos, np.nan)
            raw = (num / denom).clip(0.0, 1e6)
            empty_quiet = (~den_pos) & (~num_pos)
            raw = raw.mask(empty_quiet, 0.0)
            return raw
        return add + cancel

    def _op_net_add_pressure(self, df: pd.DataFrame, deps: List[str],
                             name: str) -> pd.Series:
        """
        Net add-side pressure:
        (add_bid - cancel_bid) - (add_ask - cancel_ask).
        Deps: [add_bid, add_ask, cancel_bid, cancel_ask] or
              [add_rate_bid, cancel_rate_bid, add_rate_ask, cancel_rate_ask].
        """
        if len(deps) >= 4:
            a = df[deps[0]].astype("float64")
            b = df[deps[1]].astype("float64")
            c = df[deps[2]].astype("float64")
            d = df[deps[3]].astype("float64")
            return (a - b) - (c - d)
        else:
            # Fallback: simple difference of first two
            a = df[deps[0]].astype("float64")
            b = df[deps[1]].astype("float64")
            return a - b

    def _op_net_cancel_pressure(self, df: pd.DataFrame, deps: List[str],
                                name: str) -> pd.Series:
        """Net cancel pressure: cancel_a - cancel_b."""
        a = df[deps[0]].astype("float64")
        b = df[deps[1]].astype("float64")
        return a - b

    def _op_pull_rate(self, df: pd.DataFrame, deps: List[str],
                      window_s: int, name: str) -> pd.Series:
        """
        Normalized cancel rate.

        4-dep variant:  (cancel_ask + cancel_bid) / (depth_ask + depth_bid + eps)
        2-dep variant:  cancel_rate / depth  (fallback for compat specs)

        Deps (4-dep): [cancel_ask, cancel_bid, depth_ask, depth_bid]
        Deps (2-dep): [cancel_col, depth_col]

        Post-processing:
          - Clip to [0, 1e6] (same bounds as S1 engine)
          - Rolling mean over window_s seconds if window_s > 1

        [FIX 2026-04-23] Previously used only deps[0] and deps[1], ignoring
        depth columns -- producing meaningless cancel_ask/cancel_bid ratios.
        Now matches the S1 engine implementation and the operator's docstring.

        [FIX-S2-PULL-RATE-DENOM 2026-04-25] When depth_ask + depth_bid ~= 0,
        EPS=1e-12 amplification produced spikes up to 6.7e4 (ratio 600x
        normal). Now: empty book + zero cancels = 0.0 (quiet state),
        empty book + nonzero cancels = NaN (impossible).
        """
        if len(deps) == 4:
            ca = df[deps[0]].astype("float64")   # cancel_ask
            cb = df[deps[1]].astype("float64")   # cancel_bid
            da = df[deps[2]].astype("float64")   # depth_ask
            db = df[deps[3]].astype("float64")   # depth_bid
            num = ca + cb
            den = da + db
            den_pos = den.abs() > EPS
            num_pos = num.abs() > EPS
            denom = den.where(den_pos, np.nan)
            raw = (num / denom).clip(0, 1e6)
            empty_quiet = (~den_pos) & (~num_pos)
            raw = raw.mask(empty_quiet, 0.0)
        else:
            # Compat 2-dep fallback
            a = df[deps[0]].astype("float64")
            b = df[deps[1]].astype("float64")
            denom = b.where(b.abs() > EPS, np.nan)
            raw = a / denom
        raw = raw.clip(0, 1e6)
        if window_s and window_s > 1:
            return raw.rolling(window=window_s, min_periods=1).mean()
        return raw

    def _op_refill_rate(self, df: pd.DataFrame, deps: List[str],
                        window_s: int, name: str) -> pd.Series:
        """
        Normalized add rate.

        4-dep variant:  (add_ask + add_bid) / (depth_ask + depth_bid + eps)
        2-dep variant:  add_rate / depth  (fallback for compat specs)

        Deps (4-dep): [add_ask, add_bid, depth_ask, depth_bid]
        Deps (2-dep): [add_col, depth_col]

        Post-processing:
          - Clip to [0, 1e6] (same bounds as S1 engine)
          - Rolling mean over window_s seconds if window_s > 1

        [FIX 2026-04-23] Previously used only deps[0] and deps[1], ignoring
        depth columns -- producing meaningless add_ask/add_bid ratios.
        Now matches the S1 engine implementation and the operator's docstring.

        [FIX-S2-REFILL-RATE-DENOM 2026-04-25] Same fix as pull_rate above.
        """
        if len(deps) == 4:
            aa = df[deps[0]].astype("float64")   # add_ask
            ab = df[deps[1]].astype("float64")   # add_bid
            da = df[deps[2]].astype("float64")   # depth_ask
            db = df[deps[3]].astype("float64")   # depth_bid
            num = aa + ab
            den = da + db
            den_pos = den.abs() > EPS
            num_pos = num.abs() > EPS
            denom = den.where(den_pos, np.nan)
            raw = (num / denom).clip(0, 1e6)
            empty_quiet = (~den_pos) & (~num_pos)
            raw = raw.mask(empty_quiet, 0.0)
        else:
            # Compat 2-dep fallback
            a = df[deps[0]].astype("float64")
            b = df[deps[1]].astype("float64")
            denom = b.where(b.abs() > EPS, np.nan)
            raw = a / denom
        raw = raw.clip(0, 1e6)
        if window_s and window_s > 1:
            return raw.rolling(window=window_s, min_periods=1).mean()
        return raw

    def _op_refill_rate_behind(self, df: pd.DataFrame, deps: List[str],
                               name: str) -> pd.Series:
        """Refill rate on the far side of the trade. Deps: [add_rate_far, depth_far]."""
        a = df[deps[0]].astype("float64")
        b = df[deps[1]].astype("float64")
        denom = b.where(b.abs() > EPS, np.nan)
        return a / denom

    def _op_cancel_rate_directional(self, df: pd.DataFrame, deps: List[str],
                                    name: str) -> pd.Series:
        """Directional cancel rate. Deps: [cancel_rate, depth]."""
        a = df[deps[0]].astype("float64")
        b = df[deps[1]].astype("float64")
        denom = b.where(b.abs() > EPS, np.nan)
        return a / denom

    # ── Cross-Market ─────────────────────────────────────────────────

    def _op_basis_vwap(self, df: pd.DataFrame, deps: List[str],
                       name: str) -> pd.Series:
        """Basis via VWAP: vwap_fut - vwap_spot."""
        fut = df[deps[0]].astype("float64")
        spot = df[deps[1]].astype("float64")
        return fut - spot

    def _op_queue_pressure_log_div(self, df: pd.DataFrame, deps: List[str],
                                   name: str) -> pd.Series:
        """Log queue pressure divergence: log_qp_fut - log_qp_spot."""
        fut = df[deps[0]].astype("float64")
        spot = df[deps[1]].astype("float64")
        return fut - spot

    def _op_z_volume_asym(self, df: pd.DataFrame, deps: List[str],
                          window_s: int, name: str) -> pd.Series:
        """
        Z-scored volume asymmetry:
        (vol_fut - vol_spot) / rolling_std(vol_fut - vol_spot).

        [FIX-Z-VOL-ASYM-DENOM 2026-04-25] Earlier "+EPS" produced spikes
        when rolling_std was 0 (constant flow). Treat as NaN since
        z-score is undefined for zero variance.
        """
        a = df[deps[0]].astype("float64")
        b = df[deps[1]].astype("float64")
        diff = a - b
        rolling_std = diff.rolling(window=window_s, min_periods=window_s).std()
        rolling_mean = diff.rolling(window=window_s, min_periods=window_s).mean()
        denom = rolling_std.where(rolling_std.abs() > EPS, np.nan)
        return ((diff - rolling_mean) / denom).clip(-100, 100)

    # ── Trade Operators ──────────────────────────────────────────────

    def _op_trade_absorption_ratio(self, df: pd.DataFrame, deps: List[str],
                                   name: str) -> pd.Series:
        """
        Trade absorption ratio: |ret| / volume.
        Deps: [ret_col, volume_col]. Measures price efficiency.

        [FIX-TRADE-ABSORP-DENOM 2026-04-25]
          vol = 0, ret = 0  -> 0.0  (no trades, no price move = quiet)
          vol = 0, ret > 0  -> NaN  (price moved without trades = exchange
                                     issue / illiquid book)
          vol > 0           -> |ret| / vol
        """
        ret = df[deps[0]].astype("float64")
        vol = df[deps[1]].astype("float64")
        vol_pos = vol.abs() > EPS
        ret_pos = ret.abs() > EPS
        denom = vol.where(vol_pos, np.nan)
        raw = (ret.abs() / denom)
        quiet_no_trade = (~vol_pos) & (~ret_pos)
        return raw.mask(quiet_no_trade, 0.0)

    def _op_taker_imbalance_bucket(self, df: pd.DataFrame, deps: List[str],
                                   window_s: int, name: str) -> pd.Series:
        """
        Taker imbalance bucketed: rolling mean of taker imbalance over window.

        [FIX-6] Changed min_periods from 1 to window_s to enforce the FILL
        CONTRACT: first (window_s - 1) rows → NaN (insufficient history).
        This is consistent with all other rolling operators in this engine.
        """
        col = df[deps[0]].astype("float64")
        return col.rolling(window=window_s, min_periods=window_s).mean()

    # ── Persistence / Autocorrelation ────────────────────────────────

    def _op_autocorr(self, df: pd.DataFrame, deps: List[str],
                     window_s: int, name: str) -> pd.Series:
        """
        Rolling lag-1 autocorrelation over window_s rows.

        Measures persistence: high autocorrelation = trending,
        low/negative = mean-reverting.
        """
        col = df[deps[0]].astype("float64")

        def _autocorr_1(x: pd.Series) -> float:
            if len(x) < 3:
                return np.nan
            return x.autocorr(lag=1)

        return col.rolling(window=window_s, min_periods=max(10, window_s // 2)).apply(
            _autocorr_1, raw=False
        )

    # ── Impact ───────────────────────────────────────────────────────

    def _op_impact_per_liquidity(self, df: pd.DataFrame, deps: List[str],
                                 window_s: int, name: str) -> pd.Series:
        """
        Market impact per unit liquidity: |ret| * volume / depth.

        [FIX-3] Column roles are now detected by NAME PREFIX instead of
        positional order:
          - ret_* / *_ret_*           → return column
          - volume_* / vol_* / *_volume_* → volume column
          - depth_*ask* / *ask*depth*  → ask-side depth
          - depth_*bid* / *bid*depth*  → bid-side depth

        This fixes s2_impact.py specs that supply deps in the order
        (depth_ask, depth_bid, ret, vol), which is opposite to the
        positional assumption in the original code.

        Positional fallback (ret=deps[-2], vol=deps[-1]) is used only when
        name-prefix detection fails entirely.
        """
        ret_col = vol_col = depth_ask_col = depth_bid_col = None

        for d in deps:
            dl = d.lower()
            if dl.startswith("ret_") or "_ret_" in dl:
                ret_col = d
            elif dl.startswith("volume_") or dl.startswith("vol_") or "_volume_" in dl:
                vol_col = d
            elif "depth" in dl and "ask" in dl:
                depth_ask_col = d
            elif "depth" in dl and "bid" in dl:
                depth_bid_col = d

        # Positional fallback when name-prefix detection fails
        if ret_col is None:
            ret_col = deps[-2] if len(deps) >= 2 else deps[0]
        if vol_col is None:
            vol_col = deps[-1] if len(deps) >= 1 else deps[0]

        ret = df[ret_col].astype("float64")
        vol = df[vol_col].astype("float64")

        if depth_ask_col and depth_bid_col:
            depth = df[depth_ask_col].astype("float64") + df[depth_bid_col].astype("float64")
        elif depth_ask_col or depth_bid_col:
            depth = df[depth_ask_col or depth_bid_col].astype("float64")
        else:
            # Fallback: no depth columns found — return plain impact
            return (ret.abs() * vol).rolling(window=window_s, min_periods=1).mean()

        denom = depth.where(depth.abs() > EPS, np.nan)
        raw_impact = ret.abs() * vol / denom
        return raw_impact.rolling(window=window_s, min_periods=1).mean()

    def _op_impact_per_signed(self, df: pd.DataFrame, deps: List[str],
                              window_s: int, name: str) -> pd.Series:
        """
        Signed market impact: ret / (signed_volume + eps).

        [FIX-4] Column roles are now detected by NAME PREFIX instead of
        positional order:
          - ret_* / *_ret_*                     → return column
          - signed_vol* / taker_imbalance*       → signed volume proxy

        This handles both the standard 2-dep case (ret, signed_vol) and
        gracefully skips extra deps (e.g. best_ask/best_bid) that appear
        in compat 3-dep specs, without crashing or producing wrong values.

        Positional fallback (deps[0]=ret, deps[1]=signed_vol) is used only
        when name-prefix detection yields no result.
        """
        ret_col = signed_vol_col = None

        for d in deps:
            dl = d.lower()
            if dl.startswith("ret_") or "_ret_" in dl:
                ret_col = d
            elif dl.startswith("signed_vol") or dl.startswith("taker_imbalance"):
                signed_vol_col = d

        # Positional fallback
        if ret_col is None:
            ret_col = deps[0]
        if signed_vol_col is None:
            signed_vol_col = deps[1] if len(deps) > 1 else deps[0]

        ret = df[ret_col].astype("float64")
        signed_vol = df[signed_vol_col].astype("float64")
        denom = signed_vol.where(signed_vol.abs() > EPS, np.nan)
        raw_impact = ret / denom
        return raw_impact.rolling(window=window_s, min_periods=1).mean()

    # ── Meta / Regime ────────────────────────────────────────────────

    def _op_breakout_regime_flag(self, df: pd.DataFrame, deps: List[str],
                                 window_s: int, name: str) -> pd.Series:
        """
        Binary breakout flag: 1.0 when BOTH rolling |ret| and volume exceed
        2 standard deviations above their rolling means.

        Deps: [ret_col, volume_col].
        """
        ret = df[deps[0]].astype("float64").abs()
        vol = df[deps[1]].astype("float64")

        ret_mean = ret.rolling(window=window_s, min_periods=window_s).mean()
        ret_std = ret.rolling(window=window_s, min_periods=window_s).std()
        vol_mean = vol.rolling(window=window_s, min_periods=window_s).mean()
        vol_std = vol.rolling(window=window_s, min_periods=window_s).std()

        ret_flag = ret > (ret_mean + 2 * ret_std)
        vol_flag = vol > (vol_mean + 2 * vol_std)

        result = (ret_flag & vol_flag).astype("float64")
        # NaN where rolling stats are NaN
        result = result.where(ret_mean.notna() & vol_mean.notna(), np.nan)
        return result

    def _op_dir_consistency(self, df: pd.DataFrame, deps: List[str],
                            window_s: int, name: str) -> pd.Series:
        """
        Directional consistency: fraction of same-sign returns in window.
        Ranges from 0.0 (all alternating) to 1.0 (all same direction).

        [FIX-7] Replaced same_sign.iloc[0] = np.nan with a .where() mask to
        avoid SettingWithCopyWarning in Pandas >= 2.0. Behaviour is identical:
        the first row (which has no predecessor) is forced to NaN.
        """
        col = df[deps[0]].astype("float64")
        sign = np.sign(col)
        # Same-sign as previous row
        same_sign = (sign == sign.shift(1)).astype("float64")
        # [FIX-7] Force NaN for first row via boolean mask instead of iloc assignment.
        same_sign = same_sign.where(same_sign.index > same_sign.index[0], np.nan)
        return same_sign.rolling(window=window_s, min_periods=window_s).mean()

    def _op_unidir_ratio(self, df: pd.DataFrame, deps: List[str],
                         window_s: int, name: str) -> pd.Series:
        """
        Unidirectional ratio: max(count_positive, count_negative) / total
        in the rolling window. Measures how one-sided the moves are.
        """
        col = df[deps[0]].astype("float64")
        pos = (col > 0).astype("float64")
        neg = (col < 0).astype("float64")

        pos_count = pos.rolling(window=window_s, min_periods=window_s).sum()
        neg_count = neg.rolling(window=window_s, min_periods=window_s).sum()
        total = pos_count + neg_count

        total_safe = total.where(total > 0, np.nan)
        max_dir = np.maximum(pos_count, neg_count)
        return max_dir / total_safe

    def _op_depth_coherence(self, df: pd.DataFrame, deps: List[str],
                            window_s: int, name: str) -> pd.Series:
        """
        Cross-depth coherence: average pairwise correlation of queue_pressure
        across BPS bands (1bps, 2bps, 5bps, 10bps) over rolling window.

        High coherence = uniform pressure across depth levels.
        Low coherence = divergent behavior at different depths.

        Implementation uses numpy arrays for efficient rolling pairwise
        correlation computation.
        """
        if len(deps) < 2:
            raise ValueError(f"{name}: depth_coherence needs >= 2 deps, got {deps}")

        n_bands = len(deps)
        arrays = np.column_stack([df[d].astype("float64").values for d in deps])
        n_rows = len(arrays)
        result = np.full(n_rows, np.nan, dtype=np.float64)

        for i in range(window_s - 1, n_rows):
            window = arrays[i - window_s + 1: i + 1]
            # Skip if any column is all-NaN
            if np.any(np.all(np.isnan(window), axis=0)):
                continue
            # Compute pairwise correlations (upper triangle)
            corr_vals = []
            for a in range(n_bands):
                for b in range(a + 1, n_bands):
                    col_a = window[:, a]
                    col_b = window[:, b]
                    valid = ~(np.isnan(col_a) | np.isnan(col_b))
                    if valid.sum() < 3:
                        continue
                    ca, cb = col_a[valid], col_b[valid]
                    std_a, std_b = ca.std(), cb.std()
                    if std_a < EPS or std_b < EPS:
                        continue
                    r = np.corrcoef(ca, cb)[0, 1]
                    if math.isfinite(r):
                        corr_vals.append(r)
            if corr_vals:
                result[i] = np.mean(corr_vals)

        return pd.Series(result, index=df.index, dtype="float64")

    def _op_depth_slope(self, df: pd.DataFrame, deps: List[str],
                        name: str) -> pd.Series:
        """
        Depth slope: first derivative of queue_pressure across BPS bands.
        Linear regression slope of [qp_1bps, qp_2bps, qp_5bps, qp_10bps]
        against band indices [1, 2, 5, 10].

        Positive slope = pressure increasing with depth.
        """
        bps_bands = np.array([1.0, 2.0, 5.0, 10.0], dtype=np.float64)
        n_bands = min(len(deps), len(bps_bands))
        bands = bps_bands[:n_bands]
        bands_mean = bands.mean()
        bands_var = ((bands - bands_mean) ** 2).sum()

        cols = [df[deps[i]].astype("float64") for i in range(n_bands)]
        vals_df = pd.concat(cols, axis=1)

        def _slope(row: np.ndarray) -> float:
            if np.any(np.isnan(row)):
                return np.nan
            y_mean = row.mean()
            cov = ((bands[:len(row)] - bands_mean) * (row - y_mean)).sum()
            return cov / bands_var if bands_var > 0 else np.nan

        return vals_df.apply(_slope, axis=1, raw=True)

    def _op_depth_curvature(self, df: pd.DataFrame, deps: List[str],
                            name: str) -> pd.Series:
        """
        Depth curvature: second derivative of queue_pressure across BPS bands.
        Approximated as: (qp_10bps - 2*qp_5bps + qp_2bps) normalized.

        Positive curvature = concave (pressure accelerating with depth).
        Negative curvature = convex (pressure decelerating).
        """
        if len(deps) < 3:
            raise ValueError(f"{name}: depth_curvature needs >= 3 deps, got {deps}")

        # NOTE: This implementation assumes deps arrive already in a consistent band order.
        # If you want to be strict, parse band bps from column names and sort explicitly.
        cols = [df[d].astype("float64") for d in deps]

        if len(cols) >= 4:
            c1, c2, c3, c4 = cols[0], cols[1], cols[2], cols[3]
            curv_a = c3 - 2 * c2 + c1
            curv_b = c4 - 2 * c3 + c2
            return (curv_a + curv_b) / 2.0
        else:
            c1, c2, c3 = cols[0], cols[1], cols[2]
            return c3 - 2 * c2 + c1

    # ── Price ────────────────────────────────────────────────────────

    def _op_mid_touch_dev(self, df: pd.DataFrame, deps: List[str],
                           name: str) -> pd.Series:
        """
        Microprice deviation from mid in BPS.
        Deps: [mid_touch_col, mid_col] or [imb_col, spread_col].

        mid_touch_dev = (mid_touch - mid) / mid * 10000.
        """
        a = df[deps[0]].astype("float64")
        b = df[deps[1]].astype("float64")
        denom = b.where(b.abs() > EPS, np.nan)
        return (a - b) / denom * 10_000

    def _op_price_acceleration(self, df: pd.DataFrame, deps: List[str],
                               window_s: int, name: str) -> pd.Series:
        """
        Price acceleration: second derivative of price ≈ diff(diff(col)).
        Smoothed with rolling mean over window_s.
        """
        col = df[deps[0]].astype("float64")
        d1 = col.diff(1)
        d2 = d1.diff(1)
        return d2.rolling(window=window_s, min_periods=1).mean()

    def _op_price_deviation_bps(self, df: pd.DataFrame, deps: List[str],
                                name: str) -> pd.Series:
        """
        Mid-to-VWAP deviation in basis points: (mid - vwap) / mid * 10000.
        Deps: [mid_col, vwap_col].
        """
        mid = df[deps[0]].astype("float64")
        vwap = df[deps[1]].astype("float64")
        denom = mid.where(mid.abs() > EPS, np.nan)
        return (mid - vwap) / denom * 10_000

    def _op_ret_vwap(self, df: pd.DataFrame, deps: List[str],
                     window_s: int, name: str) -> pd.Series:
        """
        VWAP return: log(vwap / mid). Smoothed over window.

        [FIX-5] Column roles are now detected by NAME PREFIX (vwap*, mid*)
        instead of positional order. This fixes specs in s2_returns.py that
        supply deps as (mid, vwap) while the original code assumed (vwap, mid),
        which caused a sign-flip in the output.

        Positional fallback (deps[0]=vwap, deps[1]=mid) is kept for any spec
        that does not have "vwap"/"mid" in its column names.
        """
        vwap_col = mid_col = None
        for d in deps:
            dl = d.lower()
            if "vwap" in dl:
                vwap_col = d
            elif "mid" in dl:
                mid_col = d

        # Positional fallback (original assumption: deps[0]=vwap, deps[1]=mid)
        if vwap_col is None:
            vwap_col = deps[0]
        if mid_col is None:
            mid_col = deps[1] if len(deps) > 1 else deps[0]

        vwap = df[vwap_col].astype("float64")
        mid = df[mid_col].astype("float64")
        ratio = vwap / mid.where(mid.abs() > EPS, np.nan)
        log_ret = np.log(ratio.where(ratio > 0, np.nan))
        return log_ret.rolling(window=window_s, min_periods=1).mean()

    def _op_z_rv(self, df: pd.DataFrame, deps: List[str],
                 window_s: int, name: str) -> pd.Series:
        """
        Z-scored realized volatility:
        rv = rolling_sum(ret^2, window_s)
        z_rv = (rv - rolling_mean(rv)) / (rolling_std(rv) + eps).

        [FIX-ZRV] Output clipped to [-20, 20]: identical rationale to
        FIX-8b in _op_robust_zscore — extreme volatility spikes (e.g.
        during flash crashes) can yield |z| >> 20 for z_basis and z_rv
        columns. These values are audit errors and not informative for ML.
        """
        col = df[deps[0]].astype("float64")
        rv = (col ** 2).rolling(window=window_s, min_periods=window_s).sum()
        long_window = min(window_s * 6, len(df))
        rv_mean = rv.rolling(window=long_window, min_periods=window_s).mean()
        rv_std = rv.rolling(window=long_window, min_periods=window_s).std()
        return ((rv - rv_mean) / (rv_std + EPS)).clip(-20, 20)  # [FIX-ZRV]


# =============================================================================
# Feature Registry
# =============================================================================

ALL_S2_FEATURES: List[FeatureSpec] = (
    list(S2_ABSORPTION_FEATURES)
    + list(S2_ACTIVITY_FEATURES)
    + list(S2_AGGRESSION_FEATURES)
    + list(S2_BOOKSHAPE_FEATURES)
    + list(S2_CROSS_MARKET_FEATURES)
    + list(S2_DYNAMICS_FEATURES)
    + list(S2_IMPACT_FEATURES)
    + list(S2_LIQUIDITY_EVENTS_FEATURES)
    + list(S2_META_FEATURES)
    + list(S2_NORMALIZATION_FEATURES)
    + list(S2_PRESSURE_FEATURES)
    + list(S2_PRICE_FEATURES)
    + list(S2_RETURNS_FEATURES)
)


def _find_feature_by_name(features: Iterable[FeatureSpec], name: str) -> FeatureSpec:
    for f in features:
        if f.name == name:
            return f
    raise KeyError(f"Feature not found: {name}")


# =============================================================================
# I/O Helpers
# =============================================================================

def _paths_for_hour(
    s1_dir: str,
    out_dir: str,
    asset: str,
    date_str: str,
    hour: int,
) -> Tuple[Path, Path]:
    """Derive S1 input path and S2 output path for one asset-hour."""
    hh = f"{int(hour):02d}"
    suffix = f"{date_str}_{hh}.parquet"
    a = asset.lower()

    s1_path = Path(s1_dir) / f"s1_features_{a}_{suffix}"
    out_path = Path(out_dir) / f"s2_features_{a}_{suffix}"
    return s1_path, out_path


def _adjacent_hour(date_str: str, hour: int, delta: int) -> Tuple[str, int]:
    """
    Resolve an adjacent hour, handling midnight crossing.

    Comment: This mirrors S1's helper to correctly locate prev/next hour files
    even when crossing day boundaries (00:00 -> previous day 23:00, etc.).
    """
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(hour=hour, tzinfo=timezone.utc)
    dt2 = dt + timedelta(hours=delta)
    return dt2.strftime("%Y-%m-%d"), dt2.hour


def _try_load_s1(s1_dir: str, asset: str, date_str: str, hour: int) -> Optional[pd.DataFrame]:
    """
    Try to load an S1 parquet file. Returns None if the file doesn't exist.

    Comment: Context hours are optional. We attempt to load them and proceed
    without them if missing (dataset boundaries / incomplete backfills).
    """
    hh = f"{int(hour):02d}"
    a = asset.lower()
    path = Path(s1_dir) / f"s1_features_{a}_{date_str}_{hh}.parquet"
    if path.exists():
        return pq.read_table(str(path)).to_pandas()
    return None


def _load_with_context(
    s1_dir: str,
    asset: str,
    date_str: str,
    hour: int,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, int, int, List[Path]]:
    """
    Load the target hour S1 file plus adjacent hours for context.

    Returns:
        (combined_df, start_idx, end_idx, files_used)
        where combined_df[start_idx:end_idx] is the target hour's rows
        and files_used contains ONLY the S1 paths actually loaded.

    -------------------------------------------------------------------------
    THIS IS THE CORE "CONTIGUOUS HOURS" SOLUTION USED BY S2 (copied from S1 idea):
      1) Load target hour (required).
      2) Try to load previous hour (optional lookback context).
      3) Try to load next hour (optional lookahead context).
      4) Concatenate into ONE continuous timeline.
      5) Return slice indices so we later cut back to target hour rows.
    -------------------------------------------------------------------------
    """
    files_used: List[Path] = []

    # --- Load target hour (required) ---
    target_path = Path(s1_dir) / f"s1_features_{asset.lower()}_{date_str}_{hour:02d}.parquet"
    target_df = _try_load_s1(s1_dir, asset, date_str, hour)
    if target_df is None:
        raise FileNotFoundError(f"Missing S1 feature file: {target_path}")
    files_used.append(target_path)

    n_target = len(target_df)

    # --- Load previous hour (optional — lookback context) ---
    prev_date, prev_hour = _adjacent_hour(date_str, hour, -1)
    prev_df = _try_load_s1(s1_dir, asset, prev_date, prev_hour)
    prev_path = Path(s1_dir) / f"s1_features_{asset.lower()}_{prev_date}_{prev_hour:02d}.parquet"

    # --- Load next hour (optional — lookahead context) ---
    next_date, next_hour = _adjacent_hour(date_str, hour, +1)
    next_df = _try_load_s1(s1_dir, asset, next_date, next_hour)
    next_path = Path(s1_dir) / f"s1_features_{asset.lower()}_{next_date}_{next_hour:02d}.parquet"

    parts: List[pd.DataFrame] = []
    n_before = 0

    if prev_df is not None:
        n_before = len(prev_df)
        parts.append(prev_df)
        files_used.append(prev_path)
        _log(verbose, f"  Context: loaded prev hour ({prev_date}_{prev_hour:02d}): {n_before} rows")

    parts.append(target_df)

    n_after = 0
    if next_df is not None:
        n_after = len(next_df)
        parts.append(next_df)
        files_used.append(next_path)
        _log(verbose, f"  Context: loaded next hour ({next_date}_{next_hour:02d}): {n_after} rows")

    if len(parts) == 1:
        _log(verbose, "  Context: no adjacent hours found, proceeding without context")
        return target_df, 0, n_target, files_used

    combined = pd.concat(parts, ignore_index=True)

    # [FIX-5] Remove duplicate timestamps that can arise at hour boundaries
    # when the last second of prev_hour == first second of target_hour, or
    # first second of next_hour == last second of target_hour.
    # keep="first" preserves the original ordering (prev → target → next).
    n_before_dedup = len(combined)
    combined = combined.drop_duplicates(subset=["bucket_dt_utc"], keep="first")
    combined = combined.reset_index(drop=True)
    n_deduped = n_before_dedup - len(combined)

    # Recalculate slice indices after dedup (timestamps are still sorted)
    target_ts = set(target_df["bucket_dt_utc"])
    mask = combined["bucket_dt_utc"].isin(target_ts)
    target_positions = combined.index[mask]
    start_idx = int(target_positions[0])
    end_idx   = int(target_positions[-1]) + 1

    _log(verbose,
         f"  Context window: {len(combined)} rows "
         f"(prev={n_before} + target={n_target} + next={n_after}"
         + (f", deduped={n_deduped}" if n_deduped else "") + ")")

    return combined, start_idx, end_idx, files_used


# =============================================================================
# Post-Build Archive
# =============================================================================

def _archive_files(
    files_to_move: List[Path],
    archive_dir: Path,
    date_str: str,
    sub_dir: str = "",
    verbose: bool = True,
) -> None:
    """
    Move consumed S1 feature files into a date-partitioned archive folder.

    Target layout:
        data_archive/{date_str}/s1_features/s1_features_btc_2026-02-16_03.parquet

    Files that don't exist (e.g. already archived) are silently skipped.
    """
    dest_dir = archive_dir / date_str
    if sub_dir:
        dest_dir = dest_dir / sub_dir
    _ensure_dir(dest_dir)

    for src in files_to_move:
        src_path = Path(src)
        if not src_path.exists():
            _log(verbose, f"Archive skip (not found): {src_path}")
            continue

        dest_path = dest_dir / src_path.name

        if dest_path.exists():
            _log(verbose, f"Archive skip (already exists): {dest_path}")
            continue

        shutil.move(str(src_path), str(dest_path))
        label = f"{date_str}/{sub_dir}" if sub_dir else date_str
        _log(verbose, f"Archived: {src_path.name} -> {label}/")


# =============================================================================
# Atomic Parquet Write
# =============================================================================

def _atomic_write_parquet(
    df: pd.DataFrame, out_path: Path, compression: str = PARQUET_COMPRESSION
) -> None:
    """Write DataFrame to parquet atomically via tmp file + os.replace."""
    table = pa.Table.from_pandas(df, preserve_index=False)
    _ensure_dir(out_path.parent)
    fd, tmp_path = tempfile.mkstemp(
        suffix=".parquet.tmp", dir=str(out_path.parent),
    )
    try:
        os.close(fd)
        pq.write_table(table, tmp_path, compression=compression)
        os.replace(tmp_path, str(out_path))
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# =============================================================================
# Build + Archive
# =============================================================================

def build_s2_features_for_hour(
    s1_dir: str,
    out_dir: str,
    asset: str,
    date_str: str,
    hour: int,
    features_filter: Optional[List[str]] = None,
    archive_dir: Optional[str] = None,
    verbose: bool = True,
    use_context: bool = True,
) -> pd.DataFrame:
    """
    Main entry point: compute S2 features for one asset-hour, write parquet,
    then archive the consumed S1 feature file (now superseded by S2).

    The output S2 parquet retains ALL previous-stage columns (S0, S1) plus
    the newly computed S2 features. The S1 input file is archived because
    the S2 file now contains everything the S1 file had, plus more.

    Context-window support (added):
      - If use_context=True, load prev/target/next hour S1 files (if available),
        compute S2 on the combined DataFrame, then slice back to target hour
        rows only before writing the S2 parquet.
      - This avoids edge effects for large rolling windows around hour boundaries.

    Args:
        s1_dir:           Directory containing S1 feature parquets.
        out_dir:          Directory to write S2 feature parquets.
        asset:            "btc", "eth", or "bnb".
        date_str:         Date string, e.g. "2026-02-16".
        hour:             Hour (0–23).
        features_filter:  Optional list of feature names to compute (None = all).
        archive_dir:      If set, move S1 files here after success.
                          Files land in {archive_dir}/{date_str}/s1_features/.
        verbose:          Print progress logs.
        use_context:      If True, attempt to load adjacent hours for seamless
                          rolling-window continuity. Disable for debugging.

    Returns:
        The computed feature DataFrame (S0 + S1 + S2 columns, written to disk).
        Returned DataFrame contains ONLY the target hour rows (even if context used).
    """
    s1_path, out_path = _paths_for_hour(s1_dir, out_dir, asset, date_str, hour)

    _ensure_dir(out_path.parent)

    context_slice = None
    files_used: List[Path] = []

    if use_context:
        _log(verbose, f"Loading S1 features with context: {asset} {date_str} hour={hour:02d}")
        combined_df, start_idx, end_idx, files_used = _load_with_context(
            s1_dir=s1_dir, asset=asset, date_str=date_str, hour=hour, verbose=verbose
        )
        if start_idx > 0 or end_idx < len(combined_df):
            context_slice = (start_idx, end_idx)
        _log(verbose, f"S1 data loaded (with context): {len(combined_df)} rows, {len(combined_df.columns)} cols")
    else:
        # Compat behavior: strict single-hour input.
        if not s1_path.exists():
            raise FileNotFoundError(f"Missing S1 feature file: {s1_path}")
        _log(verbose, f"Loading S1 features (no context): {s1_path}")
        combined_df = pq.read_table(str(s1_path)).to_pandas()
        files_used = [s1_path]
        _log(verbose, f"S1 data loaded: {len(combined_df)} rows, {len(combined_df.columns)} cols")

    engine = S2FeatureEngine(verbose=verbose)
    df = engine.compute_all(
        combined_df,
        specs=ALL_S2_FEATURES,
        features_filter=features_filter,
        context_slice=context_slice,
    )

    _log(verbose, f"Saving S2 features to: {out_path}")
    _atomic_write_parquet(df, out_path)

    mb = out_path.stat().st_size / (1024 * 1024)
    _log(verbose, f"Saved: {mb:.2f} MB | rows={len(df)} cols={len(df.columns)}")

    # ── Archive consumed S1 feature files ──
    # -------------------------------------------------------------------------
    # IMPORTANT: When context is enabled, we may have loaded prev/next hour S1
    # files for computation. We should NOT archive those, because they belong
    # to their own pipeline runs and may be needed by their own S2 build.
    #
    # Therefore we only archive the *target* hour S1 file (s1_path).
    # This matches the original contract: "consume S1 for this hour".
    # -------------------------------------------------------------------------
    if archive_dir is not None:
        _archive_files(
            files_to_move=[s1_path],  # archive only the target-hour S1 input
            archive_dir=Path(archive_dir),
            date_str=date_str,
            sub_dir="s1_features",
            verbose=verbose,
        )

    return df


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="S2 feature engine: compute S2 derived features from S1 feature parquets."
    )
    ap.add_argument("--s1-dir", type=str, default=str(_DEFAULT_S1_DIR),
                    help="Directory containing S1 feature parquets.")
    ap.add_argument("--out-dir", type=str, default=str(_DEFAULT_OUT_DIR),
                    help="Directory to write S2 feature parquets.")
    ap.add_argument("--archive-dir", type=str, default=str(_DEFAULT_ARCHIVE_DIR),
                    help="Archive directory for consumed S1 files. "
                         "Files are moved into {archive-dir}/{date}/s1_features/.")
    ap.add_argument("--no-archive", action="store_true",
                    help="Skip archiving (keep S1 files in place).")
    ap.add_argument("--asset", type=str, required=True, choices=["btc", "eth", "bnb"])
    ap.add_argument("--date", type=str, required=True)
    ap.add_argument("--hour", type=int, required=True)

    ap.add_argument("--features", type=str, nargs="+",
                    help="Optional: compute only these named features.")
    ap.add_argument("--feature", type=str,
                    help="Single-feature debug mode: compute one feature and print tail.")
    ap.add_argument("--tail", type=int, default=10)
    ap.add_argument("--quiet", "-q", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--format", choices=["table", "csv"], default="table")
    ap.add_argument(
        "--no-context", action="store_true",
        help="Disable context window (do not load adjacent hours). "
             "Faster but large rolling windows will have edge NaNs.",
    )

    args = ap.parse_args()
    verbose = not args.quiet
    use_context = not args.no_context

    if args.dry_run:
        s1_path, out_path = _paths_for_hour(args.s1_dir, args.out_dir, args.asset, args.date, args.hour)
        archive_label = "disabled" if args.no_archive else args.archive_dir
        print(f"Would read S1:       {s1_path}")
        print(f"Would write S2:      {out_path}")
        print(f"Archive dir:         {archive_label}")
        print(f"Total specs:         {len(ALL_S2_FEATURES)}")
        print(f"Context window:      {'enabled' if use_context else 'disabled'}")
        return

    # Single-feature debug mode (no archive, no write)
    if args.feature:
        # Comment: Debug mode stays strict single-hour by default.
        # If you want context in debug mode too, you can refactor similarly,
        # but most of the time you want the exact hour you're inspecting.
        s1_path, _ = _paths_for_hour(args.s1_dir, args.out_dir, args.asset, args.date, args.hour)
        if not s1_path.exists():
            raise FileNotFoundError(f"Missing S1 feature file: {s1_path}")

        s1_df = pq.read_table(str(s1_path)).to_pandas()
        s1_df = s1_df.sort_values("bucket_dt_utc").reset_index(drop=True)
        s1_df["bucket_dt_utc"] = pd.to_datetime(s1_df["bucket_dt_utc"], utc=True)

        spec = _find_feature_by_name(ALL_S2_FEATURES, args.feature)

        # For features with intra-S2 deps, compute the dependency chain
        engine = S2FeatureEngine(verbose=verbose)
        all_needed = _resolve_dependency_chain(spec, ALL_S2_FEATURES)
        sorted_needed = _toposort_specs(all_needed)

        for s in sorted_needed:
            s1_df[s.name] = engine._compute_one(s, s1_df)

        out = s1_df[["bucket_dt_utc", spec.name]].tail(args.tail)

        try:
            print(out.to_csv(index=False) if args.format == "csv" else out.to_string(index=False))
        except BrokenPipeError:
            pass
        return

    # Full build
    build_s2_features_for_hour(
        s1_dir=args.s1_dir,
        out_dir=args.out_dir,
        asset=args.asset,
        date_str=args.date,
        hour=args.hour,
        features_filter=args.features,
        archive_dir=None if args.no_archive else args.archive_dir,
        verbose=verbose,
        use_context=use_context,
    )


def _resolve_dependency_chain(
    spec: FeatureSpec,
    all_specs: List[FeatureSpec],
) -> List[FeatureSpec]:
    """
    Recursively resolve the full chain of intra-S2 dependencies for a spec.
    Returns a list containing the target spec plus all S2 specs it depends on.

    [FIX-1] Changed dep.kind=="s2" to name-matching for consistency with
    the updated _toposort_specs. A dep is treated as intra-S2 if its name
    appears in the known S2 feature set, regardless of the kind label.
    """
    name_to_spec = {s.name: s for s in all_specs}
    result: Dict[str, FeatureSpec] = {}

    def _resolve(s: FeatureSpec) -> None:
        if s.name in result:
            return
        result[s.name] = s
        for dep in s.depends_on:
            # [FIX-1] Name-matching instead of dep.kind == "s2"
            if dep.name in name_to_spec:
                _resolve(name_to_spec[dep.name])

    _resolve(spec)
    return list(result.values())


if __name__ == "__main__":
    main()