# ==============================================================================
# S4 Feature Engine — Binance-only, Multi-Asset (BTC + ETH + BNB)
#
# PURPOSE:
#   Compute S4 derived features from S3 feature parquets. S4 features are
#   higher-order analytics built on S3 rolling aggregates: temporal dynamics
#   (d1/d2), robust statistics (median/MAD/shock), cross-market divergences,
#   depth-structure analysis (coherence/slope/curvature), signal regime
#   detection (persistence/flip/pct_rank), normalization (robust_zscore),
#   and rolling aggregation (roll_sum).
#
# CONTRACT:
#   - Input:  S3 feature parquets from /data_storage/s3_features/
#             (these contain bucket_dt_utc + S0 + S1 + S2 + S3 columns)
#   - Output: S4 feature parquets to /data_storage/s4_features/
#             (these contain bucket_dt_utc + S0 + S1 + S2 + S3 + S4 columns)
#   - Each stage EXTENDS the DataFrame by adding new columns.
#     Previous-stage columns (S0–S3) are RETAINED in the output.
#   - The output file contains ALL columns: bucket_dt_utc + S0–S3 + S4.
#
# TOPOLOGICAL SORT:
#   Some S4 features depend on other S4 features (intra-stage dependencies):
#     - d2_* depends on d1_*       (acceleration needs velocity first)
#     - mad_* depends on median_*  (MAD needs rolling median pre-computed)
#     - *_shock_* depends on median_* AND mad_* (3-level chain)
#     - absorption_break_shock depends on absorption_break (S3 output) and
#       median_/mad_absorption_break computed in S4
#   The engine topologically sorts all specs before computation (Kahn's BFS)
#   so that all intra-S4 dependencies are always available when needed.
#
# FILL CONTRACT:
#   - Rolling operators (rolling_median, robust_zscore, roll_sum, pct_rank):
#     first (window_s - 1) rows → NaN (insufficient history)
#   - Temporal diffs (d1, d2): first row → NaN
#   - Ratio/division operators (cross_market_div, ratio): denom ≈ 0 → NaN
#   - Persistence/flip_rate: first (window_s - 1) rows → NaN
#   - All operators propagate NaN from inputs (NaN in → NaN out).
#
# POST-BUILD ARCHIVE:
#   After successful S4 feature computation the engine optionally moves
#   consumed S3 feature files into a date-partitioned archive directory:
#       data_storage/data_archive/{date_str}/s3_features/
#
# FIXES APPLIED (ported from S2/S3 engines):
#   [FIX-8]    _op_robust_zscore: zero-MAD → NaN (was dividing by eps=1e-9
#              which produced extreme values ~1e9 on flat/sparse windows).
#   [FIX-8b]   _op_robust_zscore: output clipped to [-20, 20].
#   [FIX-SHOCK] _op_robust_shock: zero-MAD → NaN + clip output to [-50, 50].
#   [FIX-C]    _op_rolling_median: default min_periods max(5,w//2)→max(2,w//4).
#   [FIX-D]    _op_rolling_mad:    default min_periods max(5,w//2)→max(2,w//4).
#   [FIX-ZCLIP] compute_all: post-clip all z_ columns to [-20, 20].
#   [CLEANUP]  Removed derived.absorption_break and derived.vacuum_score operators.
#              absorption_break_{15s,60s} features moved to / already in S3.
#              vacuum_score features were duplicate S3 outputs → removed from S4.
#
# S4 OPERATORS (17 distinct):
#   Temporal:  derived.d1, derived.d2, derived.rolling_median,
#              derived.rolling_mad, derived.robust_shock
#   Cross-Mkt: derived.cross_market_div, derived.ratio
#   Depth:     derived.depth_coherence, derived.depth_slope,
#              derived.depth_curvature
#   Regime:    derived.signal_persist, derived.signal_flip_rate,
#              derived.pct_rank
#   Normal:    derived.robust_zscore
#   Aggregate: derived.roll_sum, derived.logratio
#   Utility:   derived.passthrough
# ==============================================================================

from __future__ import annotations
from common.paths import DATA_ROOT

import argparse
import os
import shutil
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
#              loop — prevents stale carryover on silent compute failures.
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from etl.spec import FeatureSpec, Dep
from etl.operators.s4_operators import S4_OPERATORS

# ── S4 Spec Imports ──────────────────────────────────────────────────
from etl.spec.s4.s4_cross_market import S4_CROSS_MARKET_FEATURES
from etl.spec.s4.s4_dynamics import S4_DYNAMICS_FEATURES
from etl.spec.s4.s4_meta import S4_META_FEATURES
from etl.spec.s4.s4_normalization import S4_NORMALIZATION_FEATURES
from etl.spec.s4.s4_pressure import S4_PRESSURE_FEATURES

PARQUET_COMPRESSION = "zstd"
EPS = 1e-12

_ENGINE_DIR = Path(__file__).resolve().parent
_DEFAULT_S3_DIR = DATA_ROOT / "s3_features"
_DEFAULT_OUT_DIR = DATA_ROOT / "s4_features"
_DEFAULT_ARCHIVE_DIR = DATA_ROOT / "data_archive"


# =============================================================================
# Utilities
# =============================================================================

def _log(enabled: bool, msg: str) -> None:
    if enabled:
        print(f"[{pd.Timestamp.utcnow().isoformat()}] [S4_FEATURE_ENGINE] {msg}")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _require_cols(df: pd.DataFrame, cols: Iterable[str], ctx: str) -> None:
    """Verify required columns exist in DataFrame, raise descriptive error if not."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"{ctx}: missing required columns: {missing}. "
            f"Have {len(df.columns)} cols, first 20: {list(df.columns)[:20]}"
        )


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
    Topologically sort feature specs so that intra-S4 dependencies are
    computed before the features that depend on them.

    Intra-S4 dependencies are detected by checking whether a dependency's
    name matches another S4 feature name in the spec list. This works
    regardless of dep.kind label (all S4 specs use kind="col" uniformly).

    Algorithm: Kahn's algorithm (BFS-based topological sort).
    """
    # === Build name -> spec index mapping ===
    name_to_idx: Dict[str, int] = {}
    for i, s in enumerate(specs):
        name_to_idx[s.name] = i

    # === Build adjacency graph ===
    # in_degree[i] = number of intra-S4 deps spec[i] has
    # dependents[i] = list of spec indices that depend on spec[i]
    in_degree = [0] * len(specs)
    dependents: Dict[int, List[int]] = defaultdict(list)

    for i, s in enumerate(specs):
        for dep in s.depends_on:
            if dep.name in name_to_idx and dep.name != s.name:
                dep_idx = name_to_idx[dep.name]
                in_degree[i] += 1
                dependents[dep_idx].append(i)

    # === BFS: start with specs that have no intra-S4 dependencies ===
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
# S4 Feature Engine
# =============================================================================

class S4FeatureEngine:
    """
    Compute S4 features from S3 feature columns.

    The engine loads the S3 feature parquet into a wide DataFrame, then
    iterates through topologically-sorted specs, computing each feature
    and appending it as a new column. Intra-S4 dependencies (d2→d1,
    shock→mad→median, absorption_break_shock→absorption_break (S3 output))
    are resolved by the topological ordering.
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self._op_registry = S4_OPERATORS

    def _validate_registry(self, specs):
        """Pre-compute validation: operator exists + arity check."""
        for spec in specs:
            op = spec.operator
            if op not in self._op_registry:
                raise ValueError(
                    f"S4 registry: unknown operator '{op}' "
                    f"in feature '{spec.name}' (id={spec.feature_id})"
                )
            reg = self._op_registry[op]
            actual = len(spec.depends_on)
            expected = reg.n_input_cols
            if expected > 0 and actual != expected:
                raise ValueError(
                    f"S4 arity mismatch for '{spec.name}': '{op}' "
                    f"expects {expected}, got {actual}"
                )

    # =========================================================================
    # Main Entry: Compute All
    # =========================================================================

    def compute_all(
        self,
        s3_df: pd.DataFrame,
        specs: List[FeatureSpec],
        features_filter: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Compute all S4 features on top of the S3 feature DataFrame.

        Args:
            s3_df:            DataFrame with bucket_dt_utc + S0–S3 feature columns.
            specs:            List of S4 FeatureSpec objects.
            features_filter:  Optional subset of feature names to compute.

        Returns:
            Wide DataFrame with bucket_dt_utc + S0–S3 + S4 columns.
            Previous-stage columns are retained; S4 columns are appended.
        """
        _require_cols(s3_df, ["bucket_dt_utc"], "s3_df")

        df = s3_df.copy()
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

        # === Topological sort for intra-S4 dependency resolution ===
        sorted_specs = _toposort_specs(specs)
        _log(self.verbose, f"Computing S4 features: {len(sorted_specs)} specs "
             f"(toposorted from {len(specs)} input specs)")

        t0 = time.time()
        computed, errors = 0, 0
        s4_feature_names: List[str] = []

        for spec in sorted_specs:
            try:
                result = self._compute_one(spec, df)
                df[spec.name] = result
                s4_feature_names.append(spec.name)
                computed += 1
            except Exception as e:
                errors += 1
                if self.verbose:
                    print(f"  [WARN] {spec.name}: {e}")

        elapsed = time.time() - t0
        _log(self.verbose,
             f"Done. computed={computed} errors={errors} "
             f"in {elapsed:.2f}s | total cols={len(df.columns)} "
             f"(S0–S3 retained + {len(s4_feature_names)} new S4)")

        # [FIX-ZCLIP] Belt-and-suspenders: clip all z-score output columns to
        # [-20, 20] after the full computation pass. Catches any z_ value that
        # escaped _op_robust_zscore's own clip (e.g. S3 passthrough z_columns
        # from pre-fix data or context-boundary rows with tiny-but-nonzero MAD).
        z_cols = [c for c in df.columns if c.startswith("z_")]
        if z_cols:
            df[z_cols] = df[z_cols].clip(-20.0, 20.0)

        return df

    # =========================================================================
    # Dispatch
    # =========================================================================

    def _compute_one(self, spec: FeatureSpec, df: pd.DataFrame) -> pd.Series:
        """
        Dispatch to the appropriate operator implementation.

        Args:
            spec: The S4 FeatureSpec.
            df:   Working DataFrame (S0–S3 columns + already-computed S4
                  columns from earlier in the toposorted sequence).

        Returns:
            pd.Series aligned to df's index.
        """
        op = spec.operator
        params = spec.params
        deps = spec.depends_on
        name = spec.name
        window_s = _safe_int(params.get("window_s", 0))

        # Resolve dependency column names
        dep_names = [d.name for d in deps]

        # Validate all dependency columns are present
        _require_cols(df, dep_names, name)

        # ── Dispatch by operator ─────────────────────────────────────

        # === TEMPORAL DYNAMICS ===

        if op == "derived.d1":
            return self._op_d1(df, dep_names, name)

        if op == "derived.d2":
            return self._op_d2(df, dep_names, name)

        if op == "derived.rolling_median":
            return self._op_rolling_median(df, dep_names, params, name)

        if op == "derived.rolling_mad":
            return self._op_rolling_mad(df, dep_names, params, name)

        if op == "derived.robust_shock":
            return self._op_robust_shock(df, dep_names, params, name)

        # === CROSS-MARKET ===

        if op == "derived.sub":
            # Simple subtraction: fut_col - spot_col.
            # Used by depth_gradient_div_* and liq_concentration_div_* after bugfix.
            a = df[dep_names[0]].astype("float64")
            b = df[dep_names[1]].astype("float64")
            return a - b

        if op == "derived.cross_market_div":
            return self._op_cross_market_div(df, dep_names, params, name)

        if op == "derived.ratio":
            return self._op_ratio(df, dep_names, params, name)

        # === DEPTH-STRUCTURE ANALYSIS ===

        if op == "derived.depth_coherence":
            return self._op_depth_coherence(df, dep_names, name)

        if op == "derived.depth_slope":
            return self._op_depth_slope(df, dep_names, params, name)

        if op == "derived.depth_curvature":
            return self._op_depth_curvature(df, dep_names, name)

        # === SIGNAL REGIME DETECTION ===

        if op == "derived.signal_persist":
            return self._op_signal_persist(df, dep_names, window_s, name)

        if op == "derived.signal_flip_rate":
            return self._op_signal_flip_rate(df, dep_names, window_s, name)

        if op == "derived.pct_rank":
            return self._op_pct_rank(df, dep_names, params, name)

        # === NORMALIZATION ===

        if op == "derived.robust_zscore":
            return self._op_robust_zscore(df, dep_names, params, name)

        # === AGGREGATION ===

        if op == "derived.roll_sum":
            return self._op_roll_sum(df, dep_names, window_s, name)

        if op == "derived.logratio":
            return self._op_logratio(df, dep_names, params, name)

        # === UTILITY ===

        if op == "derived.passthrough":
            return self._op_passthrough(df, dep_names, name)

        raise ValueError(f"{name}: unknown S4 operator '{op}'")

    # =========================================================================
    # OPERATOR IMPLEMENTATIONS — Temporal Dynamics
    # =========================================================================

    def _op_d1(self, df: pd.DataFrame, deps: List[str], name: str) -> pd.Series:
        """
        First temporal difference (velocity): x[t] - x[t-1].
        First row → NaN.
        """
        col = deps[0]  # d1 takes no params; the single dependency is the input column
        return df[col].astype("float64").diff(periods=1)

    def _op_d2(self, df: pd.DataFrame, deps: List[str], name: str) -> pd.Series:
        """
        Second temporal difference (acceleration): d1[t] - d1[t-1].

        Deps: [d1_col, base_col]. Detects the d1 column by "d1_" prefix;
        falls back to deps[0] if not found.
        """
        d1_col = None
        for d in deps:
            if d.startswith("d1_"):
                d1_col = d
                break
        if d1_col is None:
            d1_col = deps[0]

        return df[d1_col].astype("float64").diff(periods=1)

    def _op_rolling_median(
        self,
        df: pd.DataFrame,
        deps: List[str],
        params: Dict[str, Any],
        name: str,
    ) -> pd.Series:
        """
        Rolling median of input column.

        Window from params["window_s"]. min_periods from params or
        max(5, window_s // 2).
        """
        col = params.get("input_col", deps[0])
        if col not in df.columns:
            col = deps[0]

        window_s = _safe_int(params.get("window_s", 1))
        min_p = _safe_int(params.get("min_periods", max(2, window_s // 4)))
        return df[col].astype("float64").rolling(window=window_s, min_periods=min_p).median()

    def _op_rolling_mad(
        self,
        df: pd.DataFrame,
        deps: List[str],
        params: Dict[str, Any],
        name: str,
    ) -> pd.Series:
        """
        Rolling MAD: median(|x - median(x)|).

        Deps: [base_col, median_col] (detected by "median_" prefix).
        Inputs: base_col (raw series) and median_col (pre-computed rolling median).
        """
        window_s = _safe_int(params.get("window_s", 1))
        min_p = _safe_int(params.get("min_periods", max(2, window_s // 4)))

        median_col = base_col = None
        for d in deps:
            if d.startswith("median_"):
                median_col = d
            else:
                base_col = d

        if median_col is None or base_col is None:
            raise ValueError(
                f"{name}: rolling_mad requires [base_col, median_col] deps. "
                f"Got: {deps}"
            )

        base = df[base_col].astype("float64")
        med  = df[median_col].astype("float64")
        abs_dev = (base - med).abs()
        return abs_dev.rolling(window=window_s, min_periods=min_p).median()

    def _op_robust_shock(
        self,
        df: pd.DataFrame,
        deps: List[str],
        params: Dict[str, Any],
        name: str,
    ) -> pd.Series:
        """
        Robust shock detector: |x - median| / (scale * MAD).

        Deps detected by name prefix:
          - dep starting with "mad_"    → MAD column
          - dep starting with "median_" → median column
          - all others                  → base value column

        Falls back to positional order if prefix detection fails.
        scale taken from params (default: 1.4826).

        [FIX-SHOCK] Zero-MAD → NaN (was dividing by eps → extreme values).
        [FIX-SHOCK] Output clipped to [-50, 50] (near-zero MAD artefact guard).
        """
        eps   = _safe_float(params.get("eps", 1e-9), 1e-9)
        scale = _safe_float(params.get("scale", 1.4826), 1.4826)

        mad_col = median_col = base_col = None
        for d in deps:
            if d.startswith("mad_"):
                mad_col = d
            elif d.startswith("median_"):
                median_col = d
            else:
                base_col = d

        if None in (mad_col, median_col, base_col):
            # Positional fallback: (base, mad, median) or (base, median, mad)
            if len(deps) >= 3:
                # Identify by sorting: the non-prefixed dep is base
                named = {d: d for d in deps}
                for d in deps:
                    if d.startswith("mad_") and mad_col is None:
                        mad_col = d
                    elif d.startswith("median_") and median_col is None:
                        median_col = d
                    elif base_col is None and not d.startswith("mad_") \
                            and not d.startswith("median_"):
                        base_col = d
            if None in (mad_col, median_col, base_col):
                raise ValueError(
                    f"{name}: robust_shock requires [mad_, median_, base] deps. "
                    f"Got: {deps}"
                )

        x   = df[base_col].astype("float64")
        med = df[median_col].astype("float64")
        mad = df[mad_col].astype("float64")
        # [FIX-SHOCK] Zero-MAD → NaN. eps divisor produced extreme values
        #             when MAD = 0 (quiet sparse bands).
        denom = scale * mad
        denom = denom.where(denom > 0, np.nan)

        # [FIX-SHOCK] Clip to [-50, 50]. Wider than zscore (±20) because
        #             genuine liquidity events can score 20-50σ. Beyond ±50
        #             is artefact of near-zero MAD.
        return ((x - med).abs() / denom).clip(-50.0, 50.0)

    # =========================================================================
    # OPERATOR IMPLEMENTATIONS — Cross-Market
    # =========================================================================

    def _op_cross_market_div(
        self,
        df: pd.DataFrame,
        deps: List[str],
        params: Dict[str, Any],
        name: str,
    ) -> pd.Series:
        """
        Cross-market ratio: fut_col / (spot_col + eps).

        Column names from params["fut_col"] / params["spot_col"] with
        positional fallback to deps[0] / deps[1].
        """
        eps = _safe_float(params.get("eps", 1e-12), 1e-12)

        fut_col  = params.get("fut_col",  deps[0] if len(deps) > 0 else None)
        spot_col = params.get("spot_col", deps[1] if len(deps) > 1 else None)

        if fut_col not in df.columns or spot_col not in df.columns:
            # Fallback to positional
            fut_col, spot_col = deps[0], deps[1]

        fut  = df[fut_col].astype("float64")
        spot = df[spot_col].astype("float64")
        return fut / (spot + eps)

    def _op_ratio(
        self,
        df: pd.DataFrame,
        deps: List[str],
        params: Dict[str, Any],
        name: str,
    ) -> pd.Series:
        """
        Generic ratio: num / (|den| + eps)  or  num / (den + eps).

        abs_den (default True) controls whether denominator is abs'd.
        Deps: [num_col, den_col].
        """
        eps        = _safe_float(params.get("eps", 1e-12), 1e-12)
        abs_den    = str(params.get("abs_den", "true")).lower() not in ("false", "0", "no")
        # NaN-guard: when |den| < nan_thresh the ratio is undefined
        # (e.g. net_add_spot=0 → Fut/Spot ratio meaningless, not ±1e16).
        nan_thresh = _safe_float(params.get("nan_thresh", 1e-6), 1e-6)

        num   = df[deps[0]].astype("float64")
        den   = df[deps[1]].astype("float64")
        denom = den.abs() + eps if abs_den else den + eps
        result = num / denom
        # Mask rows where raw denominator was too close to zero
        return result.where(den.abs() >= nan_thresh)

    # =========================================================================
    # OPERATOR IMPLEMENTATIONS — Depth-Structure Analysis
    # =========================================================================

    def _op_depth_coherence(
        self, df: pd.DataFrame, deps: List[str], name: str
    ) -> pd.Series:
        """
        Cross-depth sign coherence: fraction of depth-band pairs with matching sign.

        For 4 bands → 6 pairs; coherence = fraction of pairs that agree.
        Range [0, 1]. 1 = all bands agree, 0 = all bands disagree.

        Deps: 4 net_pressure columns for bands {1,2,5,10}bps (any order via
        input_col_0..3 in params, but we use deps for column resolution).
        """
        n_bands = len(deps)
        if n_bands < 2:
            raise ValueError(f"{name}: depth_coherence needs >= 2 deps, got {deps}")

        signs = np.column_stack([np.sign(df[d].astype("float64").values) for d in deps])
        n_rows = len(signs)
        result = np.full(n_rows, np.nan, dtype=np.float64)

        # Total number of unique pairs
        n_pairs = n_bands * (n_bands - 1) // 2

        for i in range(n_rows):
            row = signs[i]
            if np.any(np.isnan(row)):
                continue
            agree_count = 0
            for a in range(n_bands):
                for b in range(a + 1, n_bands):
                    # Both non-zero and same sign → agreement
                    if row[a] != 0.0 and row[b] != 0.0 and row[a] == row[b]:
                        agree_count += 1
            result[i] = agree_count / n_pairs

        return pd.Series(result, index=df.index, dtype="float64")

    def _op_depth_slope(
        self,
        df: pd.DataFrame,
        deps: List[str],
        params: Dict[str, Any],
        name: str,
    ) -> pd.Series:
        """
        OLS slope of net_pressure vs log(depth_bps) across {1,2,5,10}bps.

        Uses log-scale x-axis (log of bps band values) as specified in the
        S4 operator description. This gives positive slope when pressure
        increases moving deeper into the book.

        Deps: 4 net_pressure columns for bands {1,2,5,10}bps.
        """
        # Parse depth bands from params (default: "1,2,5,10")
        bands_str = params.get("depth_bands_bps", "1,2,5,10")
        try:
            bps_bands = np.array([float(x.strip()) for x in bands_str.split(",")],
                                 dtype=np.float64)
        except ValueError:
            bps_bands = np.array([1.0, 2.0, 5.0, 10.0], dtype=np.float64)

        n_bands = min(len(deps), len(bps_bands))
        x = np.log(bps_bands[:n_bands])  # log-scale x-axis
        x_mean = x.mean()
        x_var = ((x - x_mean) ** 2).sum()

        if x_var < EPS:
            return pd.Series(np.nan, index=df.index, dtype="float64")

        # Vectorized OLS slope: cov(x, y) / var(x)
        arrays = np.column_stack([df[deps[i]].astype("float64").values
                                  for i in range(n_bands)])
        y_mean = arrays.mean(axis=1)
        cov = np.sum((x[np.newaxis, :] - x_mean) *
                     (arrays - y_mean[:, np.newaxis]), axis=1)
        slope = cov / x_var

        # NaN where any band is NaN
        any_nan = np.any(np.isnan(arrays), axis=1)
        slope[any_nan] = np.nan

        return pd.Series(slope, index=df.index, dtype="float64")

    def _op_depth_curvature(
        self, df: pd.DataFrame, deps: List[str], name: str
    ) -> pd.Series:
        """
        Curvature of net_pressure across depth bands.

        Approximated as the average of second finite differences:
            curv_a = y[5bps] - 2*y[2bps] + y[1bps]
            curv_b = y[10bps] - 2*y[5bps] + y[2bps]
            curvature = (curv_a + curv_b) / 2

        Positive → convex (pressure accelerates away from best).
        Negative → concave (pressure peaks at intermediate depth).

        Deps: 4 columns for bands {1,2,5,10}bps (or fewer).
        """
        if len(deps) < 3:
            raise ValueError(f"{name}: depth_curvature needs >= 3 deps, got {deps}")

        cols = [df[d].astype("float64") for d in deps]

        if len(cols) >= 4:
            c1, c2, c3, c4 = cols[0], cols[1], cols[2], cols[3]
            curv_a = c3 - 2.0 * c2 + c1
            curv_b = c4 - 2.0 * c3 + c2
            return (curv_a + curv_b) / 2.0
        else:
            c1, c2, c3 = cols[0], cols[1], cols[2]
            return c3 - 2.0 * c2 + c1

    # =========================================================================
    # OPERATOR IMPLEMENTATIONS — Signal Regime Detection
    # =========================================================================

    def _op_signal_persist(
        self, df: pd.DataFrame, deps: List[str], window_s: int, name: str
    ) -> pd.Series:
        """
        Sign persistence: fraction of rows in rolling window where
        sign(x[t]) == sign(x[t-1]). Range [0, 1].

        High persistence → trending directional signal.
        Low persistence  → oscillating / mean-reverting signal.
        """
        col = df[deps[0]].astype("float64")
        sign = np.sign(col)
        same_sign = (sign == sign.shift(1)).astype("float64")
        same_sign.iloc[0] = np.nan
        return same_sign.rolling(window=window_s, min_periods=window_s).mean()

    def _op_signal_flip_rate(
        self, df: pd.DataFrame, deps: List[str], window_s: int, name: str
    ) -> pd.Series:
        """
        Flip rate: count(sign changes) / window_s in rolling window.
        Higher → more indecisive order flow.
        """
        col = df[deps[0]].astype("float64")
        sign = np.sign(col)
        flips = (sign != sign.shift(1)).astype("float64")
        flips.iloc[0] = np.nan
        return flips.rolling(window=window_s, min_periods=window_s).sum() / window_s

    def _op_pct_rank(
        self,
        df: pd.DataFrame,
        deps: List[str],
        params: Dict[str, Any],
        name: str,
    ) -> pd.Series:
        """
        Rolling percentile rank: rank(x_t) / count within rolling window.
        Range [0, 1]. Non-parametric measure of signal strength.

        Avoids assumptions about distribution — robust to heavy tails.
        """
        col = df[deps[0]].astype("float64")
        window_s = _safe_int(params.get("window_s", 1))
        min_p = _safe_int(params.get("min_periods", 5))

        return col.rolling(window=window_s, min_periods=min_p).apply(
            lambda x: float(pd.Series(x).rank(pct=True).iloc[-1]),
            raw=False,
        )

    # =========================================================================
    # OPERATOR IMPLEMENTATIONS — Normalization
    # =========================================================================

    def _op_robust_zscore(
        self,
        df: pd.DataFrame,
        deps: List[str],
        params: Dict[str, Any],
        name: str,
    ) -> pd.Series:
        """
        Robust z-score: (x - rolling_median) / (scale * rolling_MAD).

        Preferred over (x - mean) / std for heavy-tailed microstructure
        distributions. scale=1.4826 makes MAD consistent with σ under
        normality.

        Both the rolling median and MAD are computed internally here;
        unlike robust_shock, this operator does NOT expect pre-computed
        median/MAD dependency columns.

        [FIX-8]  Zero-MAD → NaN (was dividing by eps → extreme values ~1e9).
        [FIX-8b] Output clipped to [-20, 20] (tiny-but-nonzero MAD guard).
        """
        col_name = params.get("input_col", deps[0])
        if col_name not in df.columns:
            col_name = deps[0]

        window_s = _safe_int(params.get("window_s", 1))
        min_p    = _safe_int(params.get("min_periods", 5))
        eps      = _safe_float(params.get("eps", 1e-9), 1e-9)
        scale    = _safe_float(params.get("scale", 1.4826), 1.4826)

        col = df[col_name].astype("float64")
        rolling_med = col.rolling(window=window_s, min_periods=min_p).median()
        abs_dev     = (col - rolling_med).abs()
        rolling_mad = abs_dev.rolling(window=window_s, min_periods=min_p).median()

        # [FIX-8]  Zero-MAD → NaN. Dividing by eps when MAD=0 (flat/sparse
        #          window) produced extreme values (~1e9). Return NaN instead.
        denom = scale * rolling_mad
        denom = denom.where(denom > 0, np.nan)

        # [FIX-8b] Clip to [-20, 20]. Guards against tiny-but-nonzero MAD
        #          (one spike in a mostly-zero window) producing extreme values.
        return ((col - rolling_med) / denom).clip(-20.0, 20.0)

    # =========================================================================
    # OPERATOR IMPLEMENTATIONS — Aggregation
    # =========================================================================

    def _op_roll_sum(
        self, df: pd.DataFrame, deps: List[str], window_s: int, name: str
    ) -> pd.Series:
        """Rolling sum of first dependency column over window_s rows."""
        col_name = deps[0]
        col = df[col_name].astype("float64")
        return col.rolling(window=window_s, min_periods=1).sum()

    def _op_logratio(
        self,
        df: pd.DataFrame,
        deps: List[str],
        params: Dict[str, Any],
        name: str,
    ) -> pd.Series:
        """
        S4 logratio: rolling mean of a pre-computed 1s log-ratio column.

        In S3, log-ratio was computed from two raw columns as
        sign(a)*log(|a|+eps) - sign(b)*log(|b|+eps).
        In S4, n_input_cols=1: the input is already a 1s log-ratio column
        (e.g. net_pressure_logratio_*_1s), and this operator applies a
        short rolling mean over window_s (default 5) to smooth it.
        """
        col = df[deps[0]].astype("float64")
        window_s = _safe_int(params.get("window_s", 5))
        return col.rolling(window=window_s, min_periods=1).mean()

    # =========================================================================
    # OPERATOR IMPLEMENTATIONS — Utility
    # =========================================================================

    def _op_passthrough(
        self, df: pd.DataFrame, deps: List[str], name: str
    ) -> pd.Series:
        """
        Identity / alias: output = input unchanged.
        Used for compat name mappings (e.g. ofi_shock → taker_imbalance_shock).
        """
        return df[deps[0]].astype("float64")


# =============================================================================
# Feature Registry
# =============================================================================

ALL_S4_FEATURES: List[FeatureSpec] = (
    list(S4_CROSS_MARKET_FEATURES)
    + list(S4_DYNAMICS_FEATURES)
    + list(S4_META_FEATURES)
    + list(S4_NORMALIZATION_FEATURES)
    + list(S4_PRESSURE_FEATURES)
)


def _find_feature_by_name(features: Iterable[FeatureSpec], name: str) -> FeatureSpec:
    for f in features:
        if f.name == name:
            return f
    raise KeyError(f"Feature not found: {name}")


# =============================================================================
# I/O Helpers
# =============================================================================

def _adjacent_hour(date_str: str, hour: int, delta: int) -> Tuple[str, int]:
    """
    Compute the date_str and hour for (hour + delta), crossing day boundaries.
    [FIX-S4-CONTEXT 2026-04-23] Imported from S5 engine.
    """
    dt = datetime.strptime(date_str, "%Y-%m-%d") + timedelta(hours=delta + hour)
    return dt.strftime("%Y-%m-%d"), dt.hour


def _s3_path_for(s3_dir: str, asset: str, date_str: str, hour: int) -> Path:
    """Construct the S3 parquet path for a given asset/date/hour."""
    hh = f"{int(hour):02d}"
    return Path(s3_dir) / f"s3_features_{asset.lower()}_{date_str}_{hh}.parquet"


def _load_context_block(
    s3_dir: str,
    asset: str,
    date_str: str,
    hour: int,
    verbose: bool,
) -> Tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    """
    [FIX-S4-CONTEXT 2026-04-23] Load prev_hour + target_hour + next_hour
    into one DataFrame.

    Previously S4 loaded ONLY the target hour, which meant rolling operators
    (median, mad, robust_zscore, robust_shock, depth_slope, etc.) had no
    history at the start of each hour and produced discontinuous values
    at hour boundaries. Now mirrors the S5 strategy: load prev + target +
    next so rolling windows have a continuous timeline, then trim back to
    target-hour rows after computation.

    The target hour file MUST exist (raises FileNotFoundError otherwise).
    Prev and next hour files are loaded when available; missing files are
    skipped silently.

    Returns:
        combined_df:    Concatenated DataFrame sorted by bucket_dt_utc.
        target_start:   First timestamp of the target hour (inclusive).
        target_end:     Last timestamp of the target hour (inclusive).
    """
    target_path = _s3_path_for(s3_dir, asset, date_str, hour)
    if not target_path.exists():
        raise FileNotFoundError(f"Missing S3 feature file: {target_path}")

    frames: List[pd.DataFrame] = []

    # ── Previous hour ────────────────────────────────────────────────────
    prev_date, prev_hour = _adjacent_hour(date_str, hour, -1)
    prev_path = _s3_path_for(s3_dir, asset, prev_date, prev_hour)
    if prev_path.exists():
        _log(verbose, f"Loading prev-hour context: {prev_path.name}")
        frames.append(pq.read_table(str(prev_path)).to_pandas())
    else:
        _log(verbose, f"Prev-hour context not found (skip): {prev_path.name}")

    # ── Target hour ──────────────────────────────────────────────────────
    _log(verbose, f"Loading target S3 features: {target_path.name}")
    target_df = pq.read_table(str(target_path)).to_pandas()
    frames.append(target_df)

    # ── Next hour ────────────────────────────────────────────────────────
    next_date, next_hour = _adjacent_hour(date_str, hour, +1)
    next_path = _s3_path_for(s3_dir, asset, next_date, next_hour)
    if next_path.exists():
        _log(verbose, f"Loading next-hour context: {next_path.name}")
        frames.append(pq.read_table(str(next_path)).to_pandas())
    else:
        _log(verbose, f"Next-hour context not found (skip): {next_path.name}")

    combined = pd.concat(frames, ignore_index=True)
    combined["bucket_dt_utc"] = pd.to_datetime(combined["bucket_dt_utc"], utc=True)

    # boundaries (last sec of prev_hour == first sec of target_hour).
    n_before_dedup = len(combined)
    combined = combined.drop_duplicates(subset=["bucket_dt_utc"], keep="first")
    n_deduped = n_before_dedup - len(combined)

    combined = combined.sort_values("bucket_dt_utc").reset_index(drop=True)

    # Determine target hour time range for later trimming
    target_df["bucket_dt_utc"] = pd.to_datetime(target_df["bucket_dt_utc"], utc=True)
    target_start = target_df["bucket_dt_utc"].min()
    target_end   = target_df["bucket_dt_utc"].max()

    _log(
        verbose,
        f"Context block: {len(combined)} rows across "
        f"{len([f for f in frames])} hours"
        + (f", deduped={n_deduped}" if n_deduped else ""),
    )

    return combined, target_start, target_end


def _paths_for_hour(
    s3_dir: str,
    out_dir: str,
    asset: str,
    date_str: str,
    hour: int,
) -> Tuple[Path, Path]:
    """Derive S3 input path and S4 output path for one asset-hour."""
    hh = f"{int(hour):02d}"
    suffix = f"{date_str}_{hh}.parquet"
    a = asset.lower()

    s3_path = Path(s3_dir) / f"s3_features_{a}_{suffix}"
    out_path = Path(out_dir) / f"s4_features_{a}_{suffix}"
    return s3_path, out_path


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
    Move consumed S3 feature files into a date-partitioned archive folder.

    Target layout:
        data_archive/{date_str}/s3_features/s3_features_btc_2026-02-16_03.parquet

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

def build_s4_features_for_hour(
    s3_dir: str,
    out_dir: str,
    asset: str,
    date_str: str,
    hour: int,
    features_filter: Optional[List[str]] = None,
    archive_dir: Optional[str] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Main entry point: compute S4 features for one asset-hour, write parquet,
    then optionally archive the consumed S3 feature file.

    The output S4 parquet retains ALL previous-stage columns (S0–S3) plus
    the newly computed S4 features. The S3 input file is superseded.

    [FIX-S4-CONTEXT 2026-04-23] Now loads prev_hour + target_hour + next_hour
    as a context block (mirrors S5 strategy). Without this fix, S4 rolling
    operators (median, mad, robust_zscore, robust_shock, depth_slope,
    cross_market features) produced discontinuous values at hour boundaries
    because they had no history at hour starts. After computation the output
    is trimmed back to target-hour rows before writing.

    Args:
        s3_dir:           Directory containing S3 feature parquets.
        out_dir:          Directory to write S4 feature parquets.
        asset:            "btc", "eth", or "bnb".
        date_str:         Date string, e.g. "2026-02-16".
        hour:             Hour (0–23).
        features_filter:  Optional list of feature names to compute (None = all).
        archive_dir:      If set, move TARGET S3 file here after success
                          (prev/next context files are NOT archived).
                          Files land in {archive_dir}/{date_str}/s3_features/.
        verbose:          Print progress logs.

    Returns:
        The target-hour feature DataFrame (S0–S3 + S4 columns, also written
        to disk).
    """
    s3_path, out_path = _paths_for_hour(s3_dir, out_dir, asset, date_str, hour)
    _ensure_dir(out_path.parent)

    combined_df, target_start, target_end = _load_context_block(
        s3_dir, asset, date_str, hour, verbose
    )
    _log(verbose, f"Context block loaded: {len(combined_df)} rows, "
                  f"{len(combined_df.columns)} cols")

    # Compute S4 features across the full context block
    engine = S4FeatureEngine(verbose=verbose)
    full_df = engine.compute_all(
        combined_df, specs=ALL_S4_FEATURES, features_filter=features_filter
    )

    target_mask = (
        (full_df["bucket_dt_utc"] >= target_start)
        & (full_df["bucket_dt_utc"] <= target_end)
    )
    df = full_df.loc[target_mask].reset_index(drop=True)
    _log(verbose, f"Trimmed to target hour: {len(df)} rows "
                  f"(from {len(full_df)} context rows)")

    _log(verbose, f"Saving S4 features to: {out_path}")
    _atomic_write_parquet(df, out_path)

    mb = out_path.stat().st_size / (1024 * 1024)
    _log(verbose, f"Saved: {mb:.2f} MB | rows={len(df)} cols={len(df.columns)}")

    # ── Archive consumed S3 feature files ──
    # IMPORTANT: only archive the TARGET S3 file. Prev/next files are still
    # needed as context for the next hour build.
    if archive_dir is not None:
        _archive_files(
            files_to_move=[s3_path],
            archive_dir=Path(archive_dir),
            date_str=date_str,
            sub_dir="s3_features",
            verbose=verbose,
        )

    return df


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="S4 feature engine: compute S4 derived features from S3 feature parquets."
    )
    ap.add_argument("--s3-dir", type=str, default=str(_DEFAULT_S3_DIR),
                    help="Directory containing S3 feature parquets.")
    ap.add_argument("--out-dir", type=str, default=str(_DEFAULT_OUT_DIR),
                    help="Directory to write S4 feature parquets.")
    ap.add_argument("--archive-dir", type=str, default=str(_DEFAULT_ARCHIVE_DIR),
                    help="Archive directory for consumed S3 files. "
                         "Files are moved into {archive-dir}/{date}/s3_features/.")
    ap.add_argument("--no-archive", action="store_true",
                    help="Skip archiving (keep S3 files in place).")
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

    args = ap.parse_args()
    verbose = not args.quiet

    if args.dry_run:
        s3_path, out_path = _paths_for_hour(
            args.s3_dir, args.out_dir, args.asset, args.date, args.hour
        )
        archive_label = "disabled" if args.no_archive else args.archive_dir
        print(f"Would read S3:       {s3_path}")
        print(f"Would write S4:      {out_path}")
        print(f"Archive dir:         {archive_label}")
        print(f"Total specs:         {len(ALL_S4_FEATURES)}")
        return

    # Single-feature debug mode (no archive, no write)
    if args.feature:
        s3_path, _ = _paths_for_hour(
            args.s3_dir, args.out_dir, args.asset, args.date, args.hour
        )
        if not s3_path.exists():
            raise FileNotFoundError(f"Missing S3 feature file: {s3_path}")

        s3_df = pq.read_table(str(s3_path)).to_pandas()
        s3_df = s3_df.sort_values("bucket_dt_utc").reset_index(drop=True)
        s3_df["bucket_dt_utc"] = pd.to_datetime(s3_df["bucket_dt_utc"], utc=True)

        spec = _find_feature_by_name(ALL_S4_FEATURES, args.feature)

        # For features with intra-S4 deps, compute the full dependency chain
        engine = S4FeatureEngine(verbose=verbose)
        all_needed = _resolve_dependency_chain(spec, ALL_S4_FEATURES)
        sorted_needed = _toposort_specs(all_needed)

        for s in sorted_needed:
            s3_df[s.name] = engine._compute_one(s, s3_df)

        out = s3_df[["bucket_dt_utc", spec.name]].tail(args.tail)

        try:
            print(
                out.to_csv(index=False) if args.format == "csv"
                else out.to_string(index=False)
            )
        except BrokenPipeError:
            pass
        return

    # Full build
    build_s4_features_for_hour(
        s3_dir=args.s3_dir,
        out_dir=args.out_dir,
        asset=args.asset,
        date_str=args.date,
        hour=args.hour,
        features_filter=args.features,
        archive_dir=None if args.no_archive else args.archive_dir,
        verbose=verbose,
    )


def _resolve_dependency_chain(
    spec: FeatureSpec,
    all_specs: List[FeatureSpec],
) -> List[FeatureSpec]:
    """
    Recursively resolve the full chain of intra-S4 dependencies for a spec.
    Returns a list containing the target spec plus all S4 specs it transitively
    depends on.
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