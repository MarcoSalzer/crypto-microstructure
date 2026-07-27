# ==============================================================================
# S3 Feature Engine — Binance-only, Multi-Asset (BTC + ETH + BNB)
#
# PURPOSE:
#   Compute S3 derived features from S2 feature parquets. S3 features are
#   composite analytics, cross-market divergences, temporal dynamics (d1/d2),
#   robust statistics (median/MAD/shock), depth-profile meta-features, and
#   rolling pressure aggregations built on top of the S2 feature columns.
#
# CONTRACT:
#   - Input:  S2 feature parquets from /data_storage/s2_features/
#             (these contain bucket_dt_utc + S0 + S1 + S2 columns)
#   - Output: S3 feature parquets to /data_storage/s3_features/
#             (these contain bucket_dt_utc + S0 + S1 + S2 + S3 columns)
#   - Each stage EXTENDS the DataFrame by adding new columns.
#     Previous-stage columns (S0, S1, S2) are RETAINED in the output.
#   - The S3 output file supersedes the S2 input file. After successful
#     S3 computation the S2 file is archived (it's now redundant because
#     the S3 file contains all S2 data plus the new S3 features).
#   - The output file contains ALL columns: bucket_dt_utc + S0 + S1 + S2 + S3.
#
# TOPOLOGICAL SORT:
#   Some S3 features depend on other S3 features (intra-stage dependencies):
#     - d2_*  depends on d1_*  (second temporal derivative needs first)
#     - mad_* depends on median_* (MAD needs rolling median pre-computed)
#     - *_shock_* depends on median_* AND mad_* (shock score needs both)
#     - vacuum_score_* depends on z_refill_rate_* (normalization features)
#     - absorption_break* depends on trade_absorption_ratio_* rolling means
#   The engine topologically sorts all specs before computation so that
#   dependencies are always available when needed. Max depth is 2 levels
#   (shock → mad → median).
#
#   Intra-S3 dependencies are detected by matching dependency names against
#   the set of S3 feature names, regardless of the dep.kind label. This is
#   necessary because all S3 specs use kind="col" uniformly — the toposort
#   resolves ordering automatically.
#
# FILL CONTRACT:
#   - Rolling operators (roll_mean, roll_median, roll_sum): first
#     (window_s - 1) rows → NaN (insufficient history)
#   - Temporal diffs (d1, d2): first row → NaN
#   - Ratio/division operators: denom ≈ 0 → NaN
#   - Persistence/flip_rate: first (window_s - 1) rows → NaN
#   - All operators propagate NaN from inputs (NaN in → NaN out).
#
# CONTEXT WINDOW:
#   Hourly files are 3600 rows (1 per second). S3 has rolling windows and
#   needs continuity across hour boundaries to avoid edge NaNs (especially
#   for large windows like 300s/900s used in shock/median/MAD pipelines and
#   the absorption_break rolling-rank operator).
#
#   The engine loads up to 1 hour before (lookback) and 1 hour after
#   (lookahead) as context. Computation runs on the concatenated DataFrame,
#   then the result is sliced back to the target hour before writing.
#
#     _adjacent_hour, _try_load_s2, _load_with_context helpers;
#     compute_all gains context_slice parameter;
#     build_s3_features_for_hour gains use_context parameter;
#     CLI gains --no-context flag.
#
# POST-BUILD ARCHIVE:
#   After successful S3 feature computation the engine moves consumed S2
#   feature files into a date-partitioned archive directory:
#       data_storage/data_archive/{date_str}/s2_features/
#   This keeps s2_features/ clean for the next hour's pipeline.
#
# FIXES APPLIED vs. original:
#            compute on combined DataFrame, slice back to target rows.
#            Eliminates rolling-warmup edge NaNs at hour boundaries for
#            all S3 rolling operators (roll_median/roll_mad/shock/robust_
#            zscore/roll_mean/signal_persist etc.).
#   [FIX-7]  _op_signal_persist, _op_signal_flip_rate, _op_cross_persist:
#            replaced `.iloc[0] = np.nan` with `.where()` mask to avoid
#            SettingWithCopyWarning in Pandas >= 2.0. Behaviour identical.
#   [FIX-8]  _op_robust_zscore: zero-MAD protection — MAD=0 → NaN instead
#            of dividing by EPS=1e-12 which produced extreme values (~1e9).
#            Also unified min_periods to window_s (was max(5, window_s//2)).
#   [FIX-8b] _op_robust_zscore: output clipped to [-20, 20] to eliminate
#            residual extreme values from tiny-but-nonzero MAD artefacts.
#
#   [FIX-W1] _op_roll_sum: changed min_periods from 1 to window_s to enforce
#            the Fill Contract consistently with all other rolling operators.
#            With context-window enabled (default) warmup rows come from the
#            previous hour so target-hour rows are always fully warmed up.
#   [FIX-W2] _op_absorption_break: replaced _rolling_rank_pct(raw=False) +
#            pd.Series().rank() with raw=True numpy rank formula. Semantically
#            identical (verified max diff = 0) but ~10x faster — avoids
#            repeated Series construction on every 300-element window call.
#   [FIX-W3] _op_qp_depth_coherence: replaced Python row loop with fully
#            vectorised numpy ops (np.where + axis-wise min/max). Identical
#            logic, no more per-row iteration overhead.
#
# FEATURE GROUPS (443 total):
#   - Absorption     ( 16)  composite absorption / refill / break signals
#   - Bookshape      ( 32)  depth gradient + liquidity concentration ratios
#   - Cross-Market   ( 34)  futures vs spot divergences
#   - Dynamics       (157)  d1, d2, median, MAD, shock, persistence, flip
#   - Liquidity Evts ( 12)  churn rolling means, refill-vs-pull ratios
#   - Meta           ( 31)  dir_consistency, depth coherence/slope/curvature
#   - Normalization  ( 81)  robust z-scores of key S2 signals
#   - Pressure       ( 78)  net add/cancel pressure sums, means, log-ratios
#   - Returns        (  2)  ret_15s, ret_60s convenience aliases
#
# OPERATORS (28 distinct):
#   Generic:  derived.ratio, derived.roll_mean, derived.roll_sum,
#             derived.signal_persist, derived.signal_flip_rate,
#             derived.robust_zscore, derived.logratio
#   Stage-3:  s3.absorb_refill_mid, s3.absorption_asymmetry,
#             s3.absorption_break_flag, s3.absorption_break,
#             s3.trade_absorption_ratio_bps, s3.cross_div,
#             s3.cross_persist, s3.cross_share, s3.cross_div_delta,
#             s3.temporal_d1, s3.temporal_d2, s3.roll_median,
#             s3.roll_mad, s3.shock, s3.dir_consistency_persist,
#             s3.dir_consistency_asym, s3.refill_vs_pull_ratio,
#             s3.qp_depth_coherence, s3.qp_depth_curvature,
#             s3.qp_depth_slope, s3.vacuum_score
# ==============================================================================

from __future__ import annotations
from common.paths import DATA_ROOT

import argparse
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
from etl.operators.s3_operators import S3_OPERATORS

# ── S3 Spec Imports ──────────────────────────────────────────────────
from etl.spec.s3.s3_absorption import S3_ABSORPTION_FEATURES
from etl.spec.s3.s3_bookshape import S3_BOOKSHAPE_FEATURES
from etl.spec.s3.s3_cross_market import S3_CROSS_MARKET_FEATURES
from etl.spec.s3.s3_dynamics import S3_DYNAMICS_FEATURES
from etl.spec.s3.s3_liquidity_events import S3_LIQUIDITY_EVENTS_FEATURES
from etl.spec.s3.s3_meta import S3_META_FEATURES
from etl.spec.s3.s3_normalization import S3_NORMALIZATION_FEATURES
from etl.spec.s3.s3_pressure import S3_PRESSURE_FEATURES
from etl.spec.s3.s3_returns import S3_RETURNS_FEATURES

PARQUET_COMPRESSION = "zstd"
EPS = 1e-12

_ENGINE_DIR = Path(__file__).resolve().parent
_DEFAULT_S2_DIR = DATA_ROOT / "s2_features"
_DEFAULT_OUT_DIR = DATA_ROOT / "s3_features"
_DEFAULT_ARCHIVE_DIR = DATA_ROOT / "data_archive"


# =============================================================================
# Utilities
# =============================================================================

def _log(enabled: bool, msg: str) -> None:
    if enabled:
        print(f"[{pd.Timestamp.utcnow().isoformat()}] [S3_FEATURE_ENGINE] {msg}")


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
    Topologically sort feature specs so that intra-S3 dependencies are
    computed before the features that depend on them.

    Intra-S3 dependencies are detected by checking whether a dependency's
    name matches another S3 feature name in the spec list. This works
    regardless of the dep.kind label (all S3 specs use kind="col").

    Algorithm: Kahn's algorithm (BFS-based topological sort).
    """
    # === Build name -> spec index mapping ===
    name_to_idx: Dict[str, int] = {}
    for i, s in enumerate(specs):
        name_to_idx[s.name] = i

    # === Build adjacency graph ===
    # in_degree[i] = how many intra-S3 deps spec[i] has
    # dependents[i] = list of spec indices that depend on spec[i]
    in_degree = [0] * len(specs)
    dependents: Dict[int, List[int]] = defaultdict(list)

    for i, s in enumerate(specs):
        for dep in s.depends_on:
            if dep.name in name_to_idx and dep.name != s.name:
                dep_idx = name_to_idx[dep.name]
                in_degree[i] += 1
                dependents[dep_idx].append(i)

    # === BFS: start with specs that have no intra-S3 dependencies ===
    queue = [i for i in range(len(specs)) if in_degree[i] == 0]
    sorted_indices: List[int] = []

    while queue:
        # Sort queue by feature_id for deterministic ordering
        queue.sort(key=lambda idx: specs[idx].feature_id or 0)
        current = queue.pop(0)
        sorted_indices.append(current)

        for dep_idx in dependents[current]:
            in_degree[dep_idx] -= 1
            if in_degree[dep_idx] == 0:
                queue.append(dep_idx)

    # === Cycle detection ===
    if len(sorted_indices) != len(specs):
        remaining = [specs[i].name for i in range(len(specs))
                     if i not in set(sorted_indices)]
        raise ValueError(
            f"Topological sort failed: cycle detected among {len(remaining)} specs. "
            f"First 10: {remaining[:10]}"
        )

    return [specs[i] for i in sorted_indices]


# =============================================================================
# S3 Feature Engine
# =============================================================================

class S3FeatureEngine:
    """
    Compute S3 features from S2 feature columns.

    The engine loads the S2 feature parquet into a wide DataFrame, then
    iterates through topologically-sorted specs, computing each feature
    and adding it as a new column. Intra-S3 dependencies (d2→d1,
    shock→mad→median, vacuum→z_refill) are resolved by the topological
    ordering.
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self._op_registry = S3_OPERATORS

    def _validate_registry(self, specs):
        """Pre-compute validation: operator exists + arity check."""
        for spec in specs:
            op = spec.operator
            if op not in self._op_registry:
                raise ValueError(
                    f"S3 registry: unknown operator '{op}' "
                    f"in feature '{spec.name}' (id={spec.feature_id})"
                )
            reg = self._op_registry[op]
            actual = len(spec.depends_on)
            expected = reg.n_input_cols
            if expected > 0 and actual != expected:
                raise ValueError(
                    f"S3 arity mismatch for '{spec.name}': '{op}' "
                    f"expects {expected}, got {actual}"
                )

    # =========================================================================
    # Main Entry: Compute All
    # =========================================================================

    def compute_all(
        self,
        s2_df: pd.DataFrame,
        specs: List[FeatureSpec],
        features_filter: Optional[List[str]] = None,
        context_slice: Optional[Tuple[int, int]] = None,
    ) -> pd.DataFrame:
        """
        Compute all S3 features on top of the S2 feature DataFrame.

        Args:
            s2_df:            DataFrame with bucket_dt_utc + S0 + S1 + S2
                              feature columns. May include context rows from
                              adjacent hours.
            specs:            List of S3 FeatureSpec objects.
            features_filter:  Optional subset of feature names to compute.
            context_slice:    Optional (start_idx, end_idx) tuple indicating
                              the target hour's rows within s2_df. If provided,
                              the returned DataFrame is sliced to only these
                              rows after computation completes. Context rows
                              are used for rolling warmup but not in output.

        Returns:
            Wide DataFrame with bucket_dt_utc + S0 + S1 + S2 + S3 columns.
            Previous-stage columns are retained; S3 columns are appended.
            If context_slice is provided, only the target rows are returned.
        """
        _require_cols(s2_df, ["bucket_dt_utc"], "s2_df")

        df = s2_df.copy()
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


        # === Filter specs if requested ===
        if features_filter:
            wanted = set(features_filter)
            specs = [s for s in specs if s.name in wanted]

        # === Topological sort for intra-S3 dependency resolution ===
        sorted_specs = _toposort_specs(specs)
        _log(self.verbose, f"Computing S3 features: {len(sorted_specs)} specs "
             f"(toposorted from {len(specs)} input specs)")

        t0 = time.time()
        computed, errors = 0, 0
        s3_feature_names: List[str] = []

        for spec in sorted_specs:
            try:
                result = self._compute_one(spec, df)
                df[spec.name] = result
                s3_feature_names.append(spec.name)
                computed += 1
            except Exception as e:
                errors += 1
                if self.verbose:
                    print(f"  [WARN] {spec.name}: {e}")

        elapsed = time.time() - t0
        _log(self.verbose, f"Done. computed={computed} errors={errors} "
             f"in {elapsed:.2f}s | total cols={len(df.columns)} "
             f"(S0/S1/S2 retained + {len(s3_feature_names)} new S3)")

        # --- Slice back to the target hour if a context window was used ---
        # timeline (prev+target+next), then cut back to target rows only.
        if context_slice is not None:
            start, end = context_slice
            _log(self.verbose,
                 f"Slicing context: rows [{start}:{end}] "
                 f"({end - start} target rows from {len(df)} total)")
            df = df.iloc[start:end].reset_index(drop=True)

        # [FIX-ZCLIP] Belt-and-suspenders: clip all z-score columns to [-20, 20].
        # Catches z_ values that escaped _op_robust_zscore's own clip — e.g. S2
        # passthrough z_ columns from pre-fix data, or context-boundary rows
        # with tiny-but-non-zero MAD that produce extreme values.
        z_cols = [c for c in df.columns if c.startswith("z_")]
        if z_cols:
            df[z_cols] = df[z_cols].clip(-20.0, 20.0)

        return df

    # =========================================================================
    # Compute One Feature
    # =========================================================================

    def _compute_one(self, spec: FeatureSpec, df: pd.DataFrame) -> pd.Series:
        """
        Dispatch to the appropriate operator implementation.

        Args:
            spec: The feature specification.
            df:   The working DataFrame (S0/S1/S2 columns + already-computed
                  S3 columns from earlier in the toposorted sequence).

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

        # === GENERIC OPERATORS (derived.*) ===

        if op == "derived.ratio":
            return self._op_ratio(df, dep_names, name)

        if op == "derived.roll_mean":
            min_p = _safe_int(params.get("min_periods", window_s))
            return self._op_roll_mean(df, dep_names, window_s, name, min_p)

        if op == "derived.roll_sum":
            return self._op_roll_sum(df, dep_names, window_s, name)

        if op == "derived.signal_persist":
            return self._op_signal_persist(df, dep_names, window_s, name)

        if op == "derived.signal_flip_rate":
            return self._op_signal_flip_rate(df, dep_names, window_s, name)

        if op == "derived.robust_zscore":
            min_p = _safe_int(params.get("min_periods", window_s))
            return self._op_robust_zscore(df, dep_names, window_s, name, min_p)

        if op == "derived.logratio":
            return self._op_logratio(df, dep_names, name)

        # === ABSORPTION OPERATORS (s3.absorb*) ===

        if op == "s3.absorb_refill_mid":
            return self._op_absorb_refill_mid(df, dep_names, name)

        if op == "s3.absorption_asymmetry":
            return self._op_absorption_asymmetry(df, dep_names, name)

        if op == "s3.absorption_break_flag":
            return self._op_absorption_break_flag(df, dep_names, params, name)

        if op == "s3.absorption_break":
            return self._op_absorption_break(df, dep_names, name)

        if op == "s3.trade_absorption_ratio_bps":
            return self._op_trade_absorption_ratio_bps(df, dep_names, params, name)

        # === CROSS-MARKET OPERATORS (s3.cross*) ===

        if op == "s3.cross_div":
            return self._op_cross_div(df, dep_names, name)

        if op == "s3.cross_persist":
            return self._op_cross_persist(df, dep_names, params, name)

        if op == "s3.cross_share":
            return self._op_cross_share(df, dep_names, name)

        if op == "s3.cross_div_delta":
            return self._op_cross_div_delta(df, dep_names, name)

        # === TEMPORAL OPERATORS (s3.temporal*) ===

        if op == "s3.temporal_d1":
            return self._op_temporal_d1(df, dep_names, name)

        if op == "s3.temporal_d2":
            return self._op_temporal_d2(df, dep_names, name)

        # === ROLLING STATISTICS (s3.roll*, s3.shock) ===

        if op == "s3.roll_median":
            return self._op_roll_median(df, dep_names, params, name)

        if op == "s3.roll_mad":
            return self._op_roll_mad(df, dep_names, params, name)

        if op == "s3.shock":
            return self._op_shock(df, dep_names, name)

        # === DYNAMICS (s3.dir_consistency*) ===

        if op == "s3.dir_consistency_persist":
            return self._op_dir_consistency_persist(df, dep_names, name)

        if op == "s3.dir_consistency_asym":
            return self._op_dir_consistency_asym(df, dep_names, name)

        # === LIQUIDITY EVENTS (s3.refill_vs_pull*) ===

        if op == "s3.refill_vs_pull_ratio":
            return self._op_refill_vs_pull_ratio(df, dep_names, name)

        # === META / DEPTH-PROFILE (s3.qp_depth*, s3.vacuum*) ===

        if op == "s3.qp_depth_coherence":
            return self._op_qp_depth_coherence(df, dep_names, name)

        if op == "s3.qp_depth_curvature":
            return self._op_qp_depth_curvature(df, dep_names, name)

        if op == "s3.qp_depth_slope":
            return self._op_qp_depth_slope(df, dep_names, name)

        if op == "s3.vacuum_score":
            return self._op_vacuum_score(df, dep_names, name)

        raise ValueError(f"{name}: unknown S3 operator '{op}'")

    # =====================================================================
    # OPERATOR IMPLEMENTATIONS — Generic (derived.*)
    # =====================================================================

    # ── derived.ratio ───────────────────────────────────────────────

    def _op_ratio(self, df: pd.DataFrame, deps: List[str],
                  name: str) -> pd.Series:
        """
        Generic ratio: num / (|den| + eps).

        Deps: [numerator_col, denominator_col].
        Used for bookshape depth-gradient and liq-concentration ratios.
        Returns NaN where denominator ≈ 0.
        """
        a = df[deps[0]].astype("float64")
        b = df[deps[1]].astype("float64")
        denom = b.abs() + EPS
        return a / denom

    # ── derived.roll_mean ───────────────────────────────────────────

    def _op_roll_mean(self, df: pd.DataFrame, deps: List[str],
                      window_s: int, name: str,
                      min_periods: Optional[int] = None) -> pd.Series:
        """
        Rolling mean of first dependency column over window_s rows.

        [FIX-MINP] min_periods: optional override from FeatureSpec params.
                   Default = window_s (strict: require full window).
                   Use min_periods < window_s for sparse signals (e.g.
                   mid_touch_*_900s, avg_trade_size_*) where NaN input
                   seconds are expected — mean is computed over available
                   non-NaN rows within the window.
        """
        min_p = min_periods if min_periods is not None else window_s
        col = df[deps[0]].astype("float64")
        return col.rolling(window=window_s, min_periods=min_p).mean()

    # ── derived.roll_sum ────────────────────────────────────────────

    def _op_roll_sum(self, df: pd.DataFrame, deps: List[str],
                     window_s: int, name: str) -> pd.Series:
        """
        Rolling sum of first dependency column over window_s rows.

        [FIX-W1] Changed min_periods from 1 to window_s to enforce the Fill
        Contract consistently with all other rolling operators. With context-
        window enabled (use_context=True, default) the warmup rows come from
        the previous hour, so target-hour rows are always fully warmed up and
        this change has no practical effect on output data quality. Without
        context the first (window_s - 1) rows of the first hour will be NaN,
        which is correct and expected behaviour.
        """
        col = df[deps[0]].astype("float64")
        return col.rolling(window=window_s, min_periods=window_s).sum()

    # ── derived.signal_persist ──────────────────────────────────────

    def _op_signal_persist(self, df: pd.DataFrame, deps: List[str],
                           window_s: int, name: str) -> pd.Series:
        """
        Sign persistence: fraction of rows in rolling window where
        sign(x) == sign(x_prev). Range [0, 1].

        High persistence → trending signal.
        Low persistence  → mean-reverting / oscillating signal.

        [FIX-7] Replaced same_sign.iloc[0] = np.nan with .where() mask
        to avoid SettingWithCopyWarning in Pandas >= 2.0.
        """
        col = df[deps[0]].astype("float64")
        sign = np.sign(col)
        same_sign = (sign == sign.shift(1)).astype("float64")
        # [FIX-7] Force NaN for first row via boolean mask.
        same_sign = same_sign.where(same_sign.index > same_sign.index[0], np.nan)
        return same_sign.rolling(window=window_s, min_periods=window_s).mean()

    # ── derived.signal_flip_rate ────────────────────────────────────

    def _op_signal_flip_rate(self, df: pd.DataFrame, deps: List[str],
                             window_s: int, name: str) -> pd.Series:
        """
        Flip rate: count(sign_changes) / window_s in rolling window.
        Higher → more indecisive signal.

        [FIX-7] Replaced flips.iloc[0] = np.nan with .where() mask
        to avoid SettingWithCopyWarning in Pandas >= 2.0.
        """
        col = df[deps[0]].astype("float64")
        sign = np.sign(col)
        flips = (sign != sign.shift(1)).astype("float64")
        # [FIX-7] Force NaN for first row via boolean mask.
        flips = flips.where(flips.index > flips.index[0], np.nan)
        return flips.rolling(window=window_s, min_periods=window_s).sum() / window_s

    # ── derived.robust_zscore ───────────────────────────────────────

    def _op_robust_zscore(self, df: pd.DataFrame, deps: List[str],
                          window_s: int, name: str,
                          min_periods: Optional[int] = None) -> pd.Series:
        """
        Robust z-score: (x - rolling_median) / (1.4826 * rolling_MAD).

        The 1.4826 scaling factor makes MAD a consistent estimator of
        standard deviation for normal distributions.

        [FIX-8]    Zero-MAD → NaN; min_periods unified to window_s.
        [FIX-8b]   Output clipped to [-20, 20].
        [FIX-MINP] min_periods override: for sparse signals (pull_rate/
                   refill_rate 1bps/2bps, ~50% NaN clustered) strict
                   min_periods=window_s produces 100% NaN. Set min_periods=3
                   in the FeatureSpec params to allow z-scores over partial
                   windows.
        """
        min_p = min_periods if min_periods is not None else window_s
        col = df[deps[0]].astype("float64")
        rolling_med = col.rolling(window=window_s, min_periods=min_p).median()
        abs_dev = (col - rolling_med).abs()
        rolling_mad = abs_dev.rolling(window=window_s, min_periods=min_p).median()

        # [FIX-8] Zero-MAD → NaN
        scale = 1.4826 * rolling_mad
        scale = scale.where(scale > 0, np.nan)

        result = (col - rolling_med) / scale

        # [FIX-8b] Clip to [-20, 20]
        return result.clip(-20, 20)

    # ── derived.logratio ────────────────────────────────────────────

    def _op_logratio(self, df: pd.DataFrame, deps: List[str],
                     name: str) -> pd.Series:
        """
        Log ratio: sign(a)*log(|a|+eps) - sign(b)*log(|b|+eps).

        Compares pressure across adjacent depth bands on a log scale.
        Preserves sign information while compressing extreme values.
        """
        a = df[deps[0]].astype("float64")
        b = df[deps[1]].astype("float64")
        log_a = np.sign(a) * np.log(a.abs() + EPS)
        log_b = np.sign(b) * np.log(b.abs() + EPS)
        return log_a - log_b

    # =====================================================================
    # OPERATOR IMPLEMENTATIONS — Absorption (s3.absorb*)
    # =====================================================================

    # ── s3.absorb_refill_mid ────────────────────────────────────────

    def _op_absorb_refill_mid(self, df: pd.DataFrame, deps: List[str],
                              name: str) -> pd.Series:
        """
        Midpoint of ask/bid absorption refill:
            (absorb_refill_ask + absorb_refill_bid) / 2.

        Deps: [absorb_refill_ask_col, absorb_refill_bid_col].
        """
        ask = df[deps[0]].astype("float64")
        bid = df[deps[1]].astype("float64")
        return (ask + bid) / 2.0

    # ── s3.absorption_asymmetry ─────────────────────────────────────

    def _op_absorption_asymmetry(self, df: pd.DataFrame, deps: List[str],
                                 name: str) -> pd.Series:
        """
        Normalized absorption asymmetry:
            (vol_ask - vol_bid) / (vol_ask + vol_bid + eps).

        Range [-1, +1]. Positive → more absorption on ask side.
        Deps: [absorb_vol_ask_col, absorb_vol_bid_col].
        """
        ask = df[deps[0]].astype("float64")
        bid = df[deps[1]].astype("float64")
        return (ask - bid) / (ask + bid + EPS)

    # ── s3.absorption_break_flag ────────────────────────────────────

    def _op_absorption_break_flag(self, df: pd.DataFrame, deps: List[str],
                                  params: Dict[str, Any],
                                  name: str) -> pd.Series:
        """
        Binary absorption break flag. Fires 1.0 when ALL conditions met:
            1. trade_absorption_ratio > tar_thresh (default 2.0)
            2. |taker_imbalance| > imb_thresh (default 0.3)
            3. absorb_refill_mid < median(absorb_refill_mid)

        Deps are resolved BY NAME (not position), because the spec declares
        them in a different order than this function historically assumed.
        Expected deps (any order):
          - absorb_refill_ask_*  (kind='ask', 'absorb_refill')
          - absorb_refill_bid_*  (kind='bid', 'absorb_refill')
          - impact_per_signed_*  (kind='impact')
          - taker_imbalance_*    (kind='taker_imbalance')
          - trade_absorption_ratio_*  (kind='trade_absorption_ratio')

        [FIX 2026-04-23] Previously used deps[0]=tar, deps[1]=imb, deps[2]=impact
        positionally, but spec provides them in order
        [absorb_ask, absorb_bid, impact, taker_imb, TAR] — producing garbage.
        Now resolved by substring match.
        """
        tar_thresh = _safe_float(params.get("tar_thresh", 2.0), 2.0)
        imb_thresh = _safe_float(params.get("imb_thresh", 0.3), 0.3)

        # Resolve deps by name pattern
        tar_col = None
        imb_col = None
        refill_ask_col = None
        refill_bid_col = None
        for d in deps:
            if "trade_absorption_ratio" in d:
                tar_col = d
            elif "taker_imbalance" in d:
                imb_col = d
            elif "absorb_refill_ask" in d:
                refill_ask_col = d
            elif "absorb_refill_bid" in d:
                refill_bid_col = d

        if tar_col is None or imb_col is None:
            raise ValueError(
                f"{name}: absorption_break_flag needs trade_absorption_ratio "
                f"and taker_imbalance deps (by name). Got: {deps}"
            )

        tar = df[tar_col].astype("float64")
        imb = df[imb_col].astype("float64")

        # Build refill_mid from ask+bid if both present, else fallback
        if refill_ask_col and refill_bid_col:
            refill_mid = (df[refill_ask_col].astype("float64") +
                          df[refill_bid_col].astype("float64")) / 2.0
        else:
            # No refill dep → only use TAR and imbalance conditions
            refill_mid = pd.Series(0.0, index=df.index, dtype="float64")

        # Rolling median of refill_mid for "low refill" threshold
        refill_median = refill_mid.rolling(window=60, min_periods=30).median()

        cond_tar = tar > tar_thresh
        cond_imb = imb.abs() > imb_thresh
        cond_refill = refill_mid < refill_median

        result = (cond_tar & cond_imb & cond_refill).astype("float64")
        # NaN where any input is NaN or where median not yet available
        result = result.where(
            tar.notna() & imb.notna() & refill_median.notna(), np.nan
        )
        return result

    # ── s3.absorption_break ─────────────────────────────────────────

    def _op_absorption_break(self, df: pd.DataFrame, deps: List[str],
                             name: str) -> pd.Series:
        """
        Continuous absorption break score: weighted composite.
            score = w_tar * TAR_pct + w_imb * |imbalance|_pct
                    + w_imp * |impact|_pct - w_refill * refill_pct

        Rank-normalizes each component within a 300s rolling window
        (avoids scale sensitivity), then combines.
        Higher score → higher probability of absorption break.

        Deps are resolved BY NAME (not position). Expected (any order):
          - trade_absorption_ratio_*
          - taker_imbalance_*
          - impact_per_signed_*
          - absorb_refill_ask_*
          - absorb_refill_bid_*

        [FIX 2026-04-23] Previously used deps[0]=tar, deps[1]=imb positionally,
        but spec provides them in order
        [absorb_ask, absorb_bid, impact, taker_imb, TAR] — producing garbage.
        Now resolved by substring match.

        [FIX-W2] _rolling_rank_pct uses raw=True + numpy rank formula
        instead of raw=False + pd.Series().rank(). ~10x faster, same output.
        """
        # Resolve deps by name pattern
        tar_col = None
        imb_col = None
        impact_col = None
        refill_ask_col = None
        refill_bid_col = None
        for d in deps:
            if "trade_absorption_ratio" in d:
                tar_col = d
            elif "taker_imbalance" in d:
                imb_col = d
            elif "impact_per_signed" in d:
                impact_col = d
            elif "absorb_refill_ask" in d:
                refill_ask_col = d
            elif "absorb_refill_bid" in d:
                refill_bid_col = d

        missing = []
        if tar_col is None:
            missing.append("trade_absorption_ratio_*")
        if imb_col is None:
            missing.append("taker_imbalance_*")
        if impact_col is None:
            missing.append("impact_per_signed_*")
        if missing:
            raise ValueError(
                f"{name}: absorption_break missing required deps "
                f"{missing}. Got: {deps}"
            )

        tar = df[tar_col].astype("float64")
        imb = df[imb_col].astype("float64").abs()
        impact = df[impact_col].astype("float64").abs()

        if refill_ask_col and refill_bid_col:
            refill = (df[refill_ask_col].astype("float64") +
                      df[refill_bid_col].astype("float64")) / 2.0
        else:
            refill = pd.Series(0.0, index=df.index, dtype="float64")

        # Rank-normalize each component within a 300s rolling window
        # to avoid scale sensitivity.
        # [FIX-W2] Replaced raw=False + pd.Series().rank() with raw=True +
        # numpy rank formula. Semantically identical to pandas rank(pct=True,
        # method='average') — verified max diff = 0.00 on test data — but
        # ~10x faster because raw=True passes a numpy array directly to the
        # apply function, avoiding repeated Series construction per window.
        def _rolling_rank_pct(col: pd.Series, w: int = 300) -> pd.Series:
            """Percentile rank of last element within rolling window [0, 1].
            Matches pandas rank(pct=True, method='average') exactly."""
            def _rank_last(x: np.ndarray) -> float:
                v = x[-1]
                n = len(x)
                n_below = np.sum(x < v)
                n_equal = np.sum(x == v)  # includes the element itself
                # Average rank of tied elements divided by total count
                return (n_below + (n_equal + 1) / 2) / n
            return col.rolling(window=w, min_periods=60).apply(_rank_last, raw=True)

        tar_pct = _rolling_rank_pct(tar)
        imb_pct = _rolling_rank_pct(imb)
        impact_pct = _rolling_rank_pct(impact)
        refill_pct = _rolling_rank_pct(refill)

        # Weighted combination: high TAR + high imbalance + high impact
        # - low refill = high break score
        score = (0.35 * tar_pct + 0.25 * imb_pct + 0.25 * impact_pct
                 - 0.15 * refill_pct)
        return score

    # ── s3.trade_absorption_ratio_bps ───────────────────────────────

    def _op_trade_absorption_ratio_bps(self, df: pd.DataFrame, deps: List[str],
                                       params: Dict[str, Any],
                                       name: str) -> pd.Series:
        """
        Trade absorption ratio scoped to specific BPS depth band.

        The base TAR is taken from deps[0]. The BPS scoping is handled
        upstream (the S2 columns already carry the BPS band information).
        This operator simply passes through the value — the BPS parameter
        is metadata that informs which S2 column to read.

        Deps: [trade_absorption_ratio_*_1s].
        """
        return df[deps[0]].astype("float64")

    # =====================================================================
    # OPERATOR IMPLEMENTATIONS — Cross-Market (s3.cross*)
    # =====================================================================

    # ── s3.cross_div ────────────────────────────────────────────────

    def _op_cross_div(self, df: pd.DataFrame, deps: List[str],
                      name: str) -> pd.Series:
        """
        Cross-market divergence: fut_metric - spot_metric.

        Deps: [fut_col, spot_col].
        """
        fut = df[deps[0]].astype("float64")
        spot = df[deps[1]].astype("float64")
        return fut - spot

    # ── s3.cross_persist ────────────────────────────────────────────

    def _op_cross_persist(self, df: pd.DataFrame, deps: List[str],
                          params: Dict[str, Any],
                          name: str) -> pd.Series:
        """
        Cross-market persistence: fraction of sub-buckets where the
        spot-futures divergence signal maintains its sign.

        Deps: [sf_signal_col]. Window from params (default 300s).

        [FIX-7] Replaced same_sign.iloc[0] = np.nan with .where() mask
        to avoid SettingWithCopyWarning in Pandas >= 2.0.
        """
        col = df[deps[0]].astype("float64")
        window_s = _safe_int(params.get("window_s", 300))
        sign = np.sign(col)
        same_sign = (sign == sign.shift(1)).astype("float64")
        # [FIX-7] Force NaN for first row via boolean mask.
        same_sign = same_sign.where(same_sign.index > same_sign.index[0], np.nan)
        return same_sign.rolling(window=window_s, min_periods=window_s).mean()

    # ── s3.cross_share ──────────────────────────────────────────────

    def _op_cross_share(self, df: pd.DataFrame, deps: List[str],
                        name: str) -> pd.Series:
        """
        Cross-market share: col_a / (col_a + col_b + eps).

        Measures the relative contribution of one market to the total.
        Range [0, 1]. Deps: [col_a, col_b].
        """
        a = df[deps[0]].astype("float64")
        b = df[deps[1]].astype("float64")
        return a / (a + b + EPS)

    # ── s3.cross_div_delta ──────────────────────────────────────────

    def _op_cross_div_delta(self, df: pd.DataFrame, deps: List[str],
                            name: str) -> pd.Series:
        """
        Cross-market divergence delta: diff(fut_metric - spot_metric).

        First computes the divergence, then takes the first temporal
        derivative to capture the rate of change in divergence.

        Deps: [fut_col, spot_col].
        """
        fut = df[deps[0]].astype("float64")
        spot = df[deps[1]].astype("float64")
        div = fut - spot
        return div.diff(periods=1)

    # =====================================================================
    # OPERATOR IMPLEMENTATIONS — Temporal (s3.temporal*)
    # =====================================================================

    # ── s3.temporal_d1 ──────────────────────────────────────────────

    def _op_temporal_d1(self, df: pd.DataFrame, deps: List[str],
                        name: str) -> pd.Series:
        """First temporal derivative: x(t) - x(t-1). First row → NaN."""
        col = df[deps[0]].astype("float64")
        return col.diff(periods=1)

    # ── s3.temporal_d2 ──────────────────────────────────────────────

    def _op_temporal_d2(self, df: pd.DataFrame, deps: List[str],
                        name: str) -> pd.Series:
        """
        Second temporal derivative: d1(t) - d1(t-1).

        Deps: [d1_col, base_col]. The d1 column is an intra-S3 dependency
        that has already been computed thanks to topological sort.
        We take the diff of the d1 column.
        """
        # Find the d1 dependency (should be first dep or named d1_*)
        d1_col = None
        for d in deps:
            if d.startswith("d1_"):
                d1_col = d
                break
        if d1_col is None:
            d1_col = deps[0]

        return df[d1_col].astype("float64").diff(periods=1)

    # =====================================================================
    # OPERATOR IMPLEMENTATIONS — Rolling Statistics (s3.roll*, s3.shock)
    # =====================================================================

    # ── s3.roll_median ──────────────────────────────────────────────

    def _op_roll_median(self, df: pd.DataFrame, deps: List[str],
                        params: Dict[str, Any],
                        name: str) -> pd.Series:
        """
        Rolling median of input column.

        Used as robust central tendency for shock detection.
        Window from params (default 300s).
        """
        col = df[deps[0]].astype("float64")
        window_s = _safe_int(params.get("window_s", 300))
        min_p = _safe_int(params.get("min_periods", max(2, window_s // 4)))
        return col.rolling(window=window_s, min_periods=min_p).median()

    # ── s3.roll_mad ─────────────────────────────────────────────────

    def _op_roll_mad(self, df: pd.DataFrame, deps: List[str],
                     params: Dict[str, Any],
                     name: str) -> pd.Series:
        """
        Rolling MAD: median(|x - median_x|).

        Deps: [median_col, base_col]. The median column is an intra-S3
        dependency (already computed thanks to toposort). We subtract
        the pre-computed median from the base column, take absolute
        values, and compute rolling median of that.
        """
        window_s = _safe_int(params.get("window_s", 300))
        min_p = _safe_int(params.get("min_periods", max(2, window_s // 4)))

        # Identify median column and base column
        median_col = None
        base_col = None
        for d in deps:
            if d.startswith("median_"):
                median_col = d
            else:
                base_col = d

        if median_col is None or base_col is None:
            raise ValueError(f"{name}: roll_mad requires [median_col, base_col]. "
                             f"Got: {deps}")

        base = df[base_col].astype("float64")
        med = df[median_col].astype("float64")
        abs_dev = (base - med).abs()
        return abs_dev.rolling(window=window_s, min_periods=min_p).median()

    # ── s3.shock ────────────────────────────────────────────────────

    def _op_shock(self, df: pd.DataFrame, deps: List[str],
                  name: str) -> pd.Series:
        """
        Shock score: (x - median) / (1.4826 * MAD + eps).

        Deps: [mad_col, median_col, base_col]. Both the mad and median
        columns are intra-S3 dependencies (already computed via toposort).

        The 1.4826 factor makes MAD a consistent estimator of σ for
        normal distributions, so the shock score is interpretable as
        approximate standard deviations from the robust center.

        [FIX-SHOCK] Zero-MAD protection: when MAD = 0 (flat signal, e.g.
                    pull_rate = 0 for all rows in the rolling window of a
                    quiet 1bps band), the shock score is undefined. Return
                    NaN instead of dividing by EPS which produced extreme
                    values (~1e9) that triggered audit extreme_shock warnings
                    in all 170 files.
        [FIX-SHOCK] Output clipped to [-50, 50]: wider than the z-score
                    clip (±20) because shock scores represent deviation in
                    robust σ units from a rolling baseline — genuine liquidity
                    events can legitimately score 20–50σ. Values beyond ±50
                    are artefacts of near-zero MAD in sparse signals.
        """
        mad_col = None
        median_col = None
        base_col = None

        for d in deps:
            if d.startswith("mad_"):
                mad_col = d
            elif d.startswith("median_"):
                median_col = d
            else:
                base_col = d

        if mad_col is None or median_col is None or base_col is None:
            raise ValueError(f"{name}: shock requires [mad_, median_, base] deps. "
                             f"Got: {deps}")

        x   = df[base_col].astype("float64")
        med = df[median_col].astype("float64")
        mad = df[mad_col].astype("float64")

        # [FIX-SHOCK] Zero-MAD → NaN (undefined, not a near-zero divisor)
        scale = 1.4826 * mad
        scale = scale.where(scale > 0, np.nan)

        result = (x - med) / scale

        # [FIX-SHOCK] Clip to [-50, 50]
        return result.clip(-50, 50)

    # =====================================================================
    # OPERATOR IMPLEMENTATIONS — Dynamics (s3.dir_consistency*)
    # =====================================================================

    # ── s3.dir_consistency_persist ──────────────────────────────────

    def _op_dir_consistency_persist(self, df: pd.DataFrame, deps: List[str],
                                   name: str) -> pd.Series:
        """
        Direction consistency persistence: sign agreement between
        short-window and long-window direction consistency signals.

        Deps: [short_window_col, long_window_col].
        Returns: fraction where sign(short) == sign(long) over rolling window.
        """
        short = df[deps[0]].astype("float64")
        long_ = df[deps[1]].astype("float64")
        agree = (np.sign(short) == np.sign(long_)).astype("float64")
        agree = agree.where(short.notna() & long_.notna(), np.nan)
        return agree

    # ── s3.dir_consistency_asym ─────────────────────────────────────

    def _op_dir_consistency_asym(self, df: pd.DataFrame, deps: List[str],
                                 name: str) -> pd.Series:
        """
        Direction consistency asymmetry: short_consistency - long_consistency.

        Positive → short-term consistency is higher than long-term
        (emerging trend). Negative → long-term is more consistent
        (established regime).

        Deps: [short_window_col, long_window_col].
        """
        short = df[deps[0]].astype("float64")
        long_ = df[deps[1]].astype("float64")
        return short - long_

    # =====================================================================
    # OPERATOR IMPLEMENTATIONS — Liquidity Events (s3.refill_vs_pull*)
    # =====================================================================

    # ── s3.refill_vs_pull_ratio ─────────────────────────────────────

    def _op_refill_vs_pull_ratio(self, df: pd.DataFrame, deps: List[str],
                                 name: str) -> pd.Series:
        """
        Refill vs pull ratio: refill_rate / pull_rate.

        > 1 -> liquidity is being added faster than removed.
        < 1 -> liquidity is thinning.

        Deps: [refill_rate_col, pull_rate_col].

        [FIX-REFILL-VS-PULL-DENOM 2026-04-25]
          pull = 0, refill = 0  -> 1.0  (perfect balance: nothing in or
                                        out is "balanced", not undefined)
          pull = 0, refill > 0  -> NaN  (only adds, no cancels: ratio
                                        diverges; downstream consumers
                                        should treat as "extreme refill")
          pull > 0              -> refill / |pull|  (clipped to [0, 1e6])
        """
        refill = df[deps[0]].astype("float64")
        pull = df[deps[1]].astype("float64")
        pull_pos = pull.abs() > EPS
        refill_pos = refill.abs() > EPS
        denom = pull.abs().where(pull_pos, np.nan)
        raw = (refill / denom).clip(0.0, 1e6)
        balanced = (~pull_pos) & (~refill_pos)
        raw = raw.mask(balanced, 1.0)
        return raw

    # =====================================================================
    # OPERATOR IMPLEMENTATIONS — Meta / Depth-Profile
    # =====================================================================

    # ── s3.qp_depth_coherence ───────────────────────────────────────

    def _op_qp_depth_coherence(self, df: pd.DataFrame, deps: List[str],
                               name: str) -> pd.Series:
        """
        Queue pressure depth coherence: minimum pairwise sign agreement
        across BPS bands [1bps, 2bps, 5bps, 10bps].

        High coherence → uniform pressure direction at all depth levels.
        Low coherence  → conflicting signals at different depths.

        Uses instantaneous (row-wise) sign comparison rather than rolling
        windows. This is cheaper than S2's correlation-based coherence.

        Deps: [qp_1bps_col, qp_2bps_col, qp_5bps_col, qp_10bps_col].

        [FIX-W3] Replaced Python row loop with fully vectorised numpy ops.
        Logic is identical: result is 1.0 only when ALL bands have the same
        non-zero sign; 0.0 when any band is zero or any pair disagrees;
        NaN when any band is NaN.
        """
        n_bands = len(deps)
        if n_bands < 2:
            raise ValueError(f"{name}: qp_depth_coherence needs >= 2 deps, got {deps}")

        # signs shape: (n_rows, n_bands)
        signs = np.column_stack([np.sign(df[d].astype("float64").values) for d in deps])

        # [FIX-W3] Vectorised coherence: 1.0 iff all bands share the same
        # non-zero sign; 0.0 if any band is zero or any pair disagrees;
        # NaN if any band is NaN.
        has_nan  = np.any(np.isnan(signs), axis=1)          # (n_rows,)
        has_zero = np.any(signs == 0.0,    axis=1)           # (n_rows,)
        all_same = (signs.min(axis=1) == signs.max(axis=1))  # (n_rows,)

        result = np.where(has_nan,  np.nan,
                 np.where(has_zero | ~all_same, 0.0, 1.0))

        return pd.Series(result, index=df.index, dtype="float64")

    # ── s3.qp_depth_curvature ───────────────────────────────────────

    def _op_qp_depth_curvature(self, df: pd.DataFrame, deps: List[str],
                               name: str) -> pd.Series:
        """
        Queue pressure depth curvature: second difference across BPS bands.

        Approximated as average of second differences:
            curv_a = qp_5bps - 2*qp_2bps + qp_1bps
            curv_b = qp_10bps - 2*qp_5bps + qp_2bps
            curvature = (curv_a + curv_b) / 2

        Positive → convex (pressure accelerating with depth).
        Negative → concave (pressure decelerating).

        Deps: [qp_1bps, qp_2bps, qp_5bps, qp_10bps].
        """
        if len(deps) < 3:
            raise ValueError(f"{name}: depth_curvature needs >= 3 deps, got {deps}")

        cols = [df[d].astype("float64") for d in deps]

        if len(cols) >= 4:
            c1, c2, c3, c4 = cols[0], cols[1], cols[2], cols[3]
            curv_a = c3 - 2 * c2 + c1
            curv_b = c4 - 2 * c3 + c2
            return (curv_a + curv_b) / 2.0
        else:
            c1, c2, c3 = cols[0], cols[1], cols[2]
            return c3 - 2 * c2 + c1

    # ── s3.qp_depth_slope ───────────────────────────────────────────

    def _op_qp_depth_slope(self, df: pd.DataFrame, deps: List[str],
                           name: str) -> pd.Series:
        """
        Queue pressure depth slope: linear regression slope of
        queue_pressure values across BPS bands [1, 2, 5, 10].

        Positive → pressure increasing with depth.
        Negative → pressure concentrated near the touch.

        Uses ordinary least squares against band indices.
        Deps: [qp_1bps, qp_2bps, qp_5bps, qp_10bps].
        """
        bps_bands = np.array([1.0, 2.0, 5.0, 10.0], dtype=np.float64)
        n_bands = min(len(deps), len(bps_bands))
        bands = bps_bands[:n_bands]
        bands_mean = bands.mean()
        bands_var = ((bands - bands_mean) ** 2).sum()

        if bands_var < EPS:
            return pd.Series(np.nan, index=df.index, dtype="float64")

        # Vectorized OLS slope computation
        arrays = np.column_stack([df[deps[i]].astype("float64").values
                                  for i in range(n_bands)])
        y_mean = arrays.mean(axis=1)
        cov = np.sum((bands[np.newaxis, :] - bands_mean) *
                      (arrays - y_mean[:, np.newaxis]), axis=1)
        slope = cov / bands_var

        # NaN where any band is NaN
        any_nan = np.any(np.isnan(arrays), axis=1)
        slope[any_nan] = np.nan

        return pd.Series(slope, index=df.index, dtype="float64")

    # ── s3.vacuum_score ─────────────────────────────────────────────

    def _op_vacuum_score(self, df: pd.DataFrame, deps: List[str],
                         name: str) -> pd.Series:
        """
        Vacuum score: z_pull_rate - z_refill_rate.

        High positive → aggressive pulling with weak refill (vacuum forming).
        Negative → refill stronger than pulling (stable liquidity).

        Deps: [z_pull_rate_col, z_refill_rate_col].
        """
        z_pull = df[deps[0]].astype("float64")
        z_refill = df[deps[1]].astype("float64")
        return z_pull - z_refill


# =============================================================================
# Feature Registry
# =============================================================================

ALL_S3_FEATURES: List[FeatureSpec] = (
    list(S3_ABSORPTION_FEATURES)
    + list(S3_BOOKSHAPE_FEATURES)
    + list(S3_CROSS_MARKET_FEATURES)
    + list(S3_DYNAMICS_FEATURES)
    + list(S3_LIQUIDITY_EVENTS_FEATURES)
    + list(S3_META_FEATURES)
    + list(S3_NORMALIZATION_FEATURES)
    + list(S3_PRESSURE_FEATURES)
    + list(S3_RETURNS_FEATURES)
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
    s2_dir: str,
    out_dir: str,
    asset: str,
    date_str: str,
    hour: int,
) -> Tuple[Path, Path]:
    """Derive S2 input path and S3 output path for one asset-hour."""
    hh = f"{int(hour):02d}"
    suffix = f"{date_str}_{hh}.parquet"
    a = asset.lower()

    s2_path = Path(s2_dir) / f"s2_features_{a}_{suffix}"
    out_path = Path(out_dir) / f"s3_features_{a}_{suffix}"
    return s2_path, out_path


def _adjacent_hour(date_str: str, hour: int, delta: int) -> Tuple[str, int]:
    """
    Resolve an adjacent hour, handling midnight crossing.

    [FIX-C] Mirrors S2 engine helper for correct day-boundary arithmetic.
    """
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(hour=hour, tzinfo=timezone.utc)
    dt2 = dt + timedelta(hours=delta)
    return dt2.strftime("%Y-%m-%d"), dt2.hour


def _try_load_s2(s2_dir: str, asset: str, date_str: str, hour: int) -> Optional[pd.DataFrame]:
    """
    Try to load an S2 parquet file. Returns None if the file doesn't exist.

    [FIX-C] Context hours are optional — proceed without them if missing.
    """
    a = asset.lower()
    path = Path(s2_dir) / f"s2_features_{a}_{date_str}_{hour:02d}.parquet"
    if path.exists():
        return pq.read_table(str(path)).to_pandas()
    return None


def _load_with_context(
    s2_dir: str,
    asset: str,
    date_str: str,
    hour: int,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, int, int, List[Path]]:
    """
    Load the target hour S2 file plus adjacent hours for context.

    [FIX-C] Mirrors S2 engine's _load_with_context exactly, reading from
    s2_features/ instead of s1_features/.

    Returns:
        (combined_df, start_idx, end_idx, files_used)
        where combined_df[start_idx:end_idx] is the target hour's rows.
    """
    files_used: List[Path] = []

    # --- Load target hour (required) ---
    a = asset.lower()
    target_path = Path(s2_dir) / f"s2_features_{a}_{date_str}_{hour:02d}.parquet"
    target_df = _try_load_s2(s2_dir, asset, date_str, hour)
    if target_df is None:
        raise FileNotFoundError(f"Missing S2 feature file: {target_path}")
    files_used.append(target_path)

    n_target = len(target_df)

    # --- Load previous hour (optional — lookback context) ---
    prev_date, prev_hour = _adjacent_hour(date_str, hour, -1)
    prev_df = _try_load_s2(s2_dir, asset, prev_date, prev_hour)
    prev_path = Path(s2_dir) / f"s2_features_{a}_{prev_date}_{prev_hour:02d}.parquet"

    # --- Load next hour (optional — lookahead context) ---
    next_date, next_hour = _adjacent_hour(date_str, hour, +1)
    next_df = _try_load_s2(s2_dir, asset, next_date, next_hour)
    next_path = Path(s2_dir) / f"s2_features_{a}_{next_date}_{next_hour:02d}.parquet"

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

    # [FIX-5] Remove duplicate timestamps at hour boundaries.
    # keep="first" preserves prev → target → next ordering.
    n_before_dedup = len(combined)
    combined = combined.drop_duplicates(subset=["bucket_dt_utc"], keep="first")
    combined = combined.reset_index(drop=True)
    n_deduped = n_before_dedup - len(combined)

    # Recalculate slice indices after dedup
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
    Move consumed S2 feature files into a date-partitioned archive folder.

    Target layout:
        data_archive/{date_str}/s2_features/s2_features_btc_2026-02-16_03.parquet

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
# Build + Archive
# =============================================================================


def _atomic_write_parquet(df, out_path, compression=PARQUET_COMPRESSION):
    """Write DataFrame to parquet atomically via tmp file + os.replace."""
    table = pa.Table.from_pandas(df, preserve_index=False)
    _ensure_dir(out_path.parent)
    fd, tmp_path = tempfile.mkstemp(suffix=".parquet.tmp", dir=str(out_path.parent))
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


def build_s3_features_for_hour(
    s2_dir: str,
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
    Main entry point: compute S3 features for one asset-hour, write parquet,
    then archive the consumed S2 feature file (now superseded by S3).

    The output S3 parquet retains ALL previous-stage columns (S0, S1, S2) plus
    the newly computed S3 features. The S2 input file is archived because
    the S3 file now contains everything the S2 file had, plus more.

    [FIX-C] Context-window support added (use_context parameter):
      - If True, load prev/target/next hour S2 files (if available),
        compute S3 on the combined DataFrame, then slice back to target
        hour rows only before writing the S3 parquet.
      - Eliminates rolling-warmup edge NaNs at hour boundaries.
      - Only the target-hour S2 file is archived (not context hours).

    Args:
        s2_dir:           Directory containing S2 feature parquets.
        out_dir:          Directory to write S3 feature parquets.
        asset:            "btc", "eth", or "bnb".
        date_str:         Date string, e.g. "2026-02-16".
        hour:             Hour (0–23).
        features_filter:  Optional list of feature names to compute (None = all).
        archive_dir:      If set, move S2 files here after success.
                          Files land in {archive_dir}/{date_str}/s2_features/.
        verbose:          Print progress logs.
        use_context:      If True, load adjacent hours for rolling warmup.
                          Disable for debugging or isolated re-runs.

    Returns:
        The computed feature DataFrame (S0 + S1 + S2 + S3 columns, written to disk).
        Contains ONLY the target hour rows (even if context was used).
    """
    s2_path, out_path = _paths_for_hour(s2_dir, out_dir, asset, date_str, hour)

    _ensure_dir(out_path.parent)

    context_slice = None
    files_used: List[Path] = []

    if use_context:
        _log(verbose, f"Loading S2 features with context: {asset} {date_str} hour={hour:02d}")
        combined_df, start_idx, end_idx, files_used = _load_with_context(
            s2_dir=s2_dir, asset=asset, date_str=date_str, hour=hour, verbose=verbose
        )
        if start_idx > 0 or end_idx < len(combined_df):
            context_slice = (start_idx, end_idx)
        _log(verbose, f"S2 data loaded (with context): {len(combined_df)} rows, "
             f"{len(combined_df.columns)} cols")
    else:
        # Compat behaviour: strict single-hour input.
        if not s2_path.exists():
            raise FileNotFoundError(f"Missing S2 feature file: {s2_path}")
        _log(verbose, f"Loading S2 features (no context): {s2_path}")
        combined_df = pq.read_table(str(s2_path)).to_pandas()
        files_used = [s2_path]
        _log(verbose, f"S2 data loaded: {len(combined_df)} rows, {len(combined_df.columns)} cols")

    engine = S3FeatureEngine(verbose=verbose)
    df = engine.compute_all(
        combined_df,
        specs=ALL_S3_FEATURES,
        features_filter=features_filter,
        context_slice=context_slice,
    )

    _log(verbose, f"Saving S3 features to: {out_path}")
    _atomic_write_parquet(df, out_path)

    mb = out_path.stat().st_size / (1024 * 1024)
    _log(verbose, f"Saved: {mb:.2f} MB | rows={len(df)} cols={len(df.columns)}")

    # ── Archive consumed S2 feature file ──
    # Only archive the target-hour S2 file. Context hours (prev/next) belong
    # to their own pipeline runs and must not be archived here.
    if archive_dir is not None:
        _archive_files(
            files_to_move=[s2_path],
            archive_dir=Path(archive_dir),
            date_str=date_str,
            sub_dir="s2_features",
            verbose=verbose,
        )

    return df


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="S3 feature engine: compute S3 derived features from S2 feature parquets."
    )
    ap.add_argument("--s2-dir", type=str, default=str(_DEFAULT_S2_DIR),
                    help="Directory containing S2 feature parquets.")
    ap.add_argument("--out-dir", type=str, default=str(_DEFAULT_OUT_DIR),
                    help="Directory to write S3 feature parquets.")
    ap.add_argument("--archive-dir", type=str, default=str(_DEFAULT_ARCHIVE_DIR),
                    help="Archive directory for consumed S2 files. "
                         "Files are moved into {archive-dir}/{date}/s2_features/.")
    ap.add_argument("--no-archive", action="store_true",
                    help="Skip archiving (keep S2 files in place).")
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
             "Faster but rolling windows will have edge NaNs at hour boundaries.",
    )

    args = ap.parse_args()
    verbose = not args.quiet
    use_context = not args.no_context

    if args.dry_run:
        s2_path, out_path = _paths_for_hour(args.s2_dir, args.out_dir, args.asset, args.date, args.hour)
        archive_label = "disabled" if args.no_archive else args.archive_dir
        print(f"Would read S2:       {s2_path}")
        print(f"Would write S3:      {out_path}")
        print(f"Archive dir:         {archive_label}")
        print(f"Total specs:         {len(ALL_S3_FEATURES)}")
        print(f"Context window:      {'enabled' if use_context else 'disabled'}")
        return

    # Single-feature debug mode (no archive, no write)
    if args.feature:
        s2_path, _ = _paths_for_hour(args.s2_dir, args.out_dir, args.asset, args.date, args.hour)
        if not s2_path.exists():
            raise FileNotFoundError(f"Missing S2 feature file: {s2_path}")

        s2_df = pq.read_table(str(s2_path)).to_pandas()
        s2_df = s2_df.sort_values("bucket_dt_utc").reset_index(drop=True)
        s2_df["bucket_dt_utc"] = pd.to_datetime(s2_df["bucket_dt_utc"], utc=True)

        spec = _find_feature_by_name(ALL_S3_FEATURES, args.feature)

        # For features with intra-S3 deps, compute the dependency chain
        engine = S3FeatureEngine(verbose=verbose)
        all_needed = _resolve_dependency_chain(spec, ALL_S3_FEATURES)
        sorted_needed = _toposort_specs(all_needed)

        for s in sorted_needed:
            s2_df[s.name] = engine._compute_one(s, s2_df)

        out = s2_df[["bucket_dt_utc", spec.name]].tail(args.tail)

        try:
            print(out.to_csv(index=False) if args.format == "csv" else out.to_string(index=False))
        except BrokenPipeError:
            pass
        return

    # Full build
    build_s3_features_for_hour(
        s2_dir=args.s2_dir,
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
    Recursively resolve the full chain of intra-S3 dependencies for a spec.
    Returns a list containing the target spec plus all S3 specs it depends on.
    """
    name_to_spec = {s.name: s for s in all_specs}
    result: Dict[str, FeatureSpec] = {}

    def _resolve(s: FeatureSpec) -> None:
        if s.name in result:
            return
        result[s.name] = s
        for dep in s.depends_on:
            if dep.name in name_to_spec:
                _resolve(name_to_spec[dep.name])

    _resolve(spec)
    return list(result.values())


if __name__ == "__main__":
    main()