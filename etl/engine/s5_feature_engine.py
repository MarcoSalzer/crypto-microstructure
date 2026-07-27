# ==============================================================================
# S5 Feature Engine — Binance-only, Multi-Asset (BTC + ETH + BNB)
#
# PURPOSE:
#   Compute S5 derived features from S4 feature parquets. S5 features are
#   signal-quality analytics built on top of S4 rolling aggregates:
#     - Robust shock pipelines (median → MAD → shock) for absorption_break
#       and vacuum_score composite signals.
#     - Self-contained robust z-scores for cross-market sf ratios and
#       per-market net-add/cancel pressure series.
#     - Directional persistence (S5 formula) for sf ratios and per-market
#       net-add/cancel rolling sums.
#
# CONTRACT:
#   - Input:  S4 feature parquets from data_storage/s4_features/
#             (contain bucket_dt_utc + S0–S4 columns)
#   - Output: S5 feature parquets to data_storage/s5_features/
#             (contain bucket_dt_utc + S0–S4 + S5 columns)
#   - Each stage EXTENDS the DataFrame by adding new columns.
#     All previous-stage columns (S0–S4) are RETAINED in the output.
#
# CONTEXT WINDOW LOADING:
#   Rolling operators with large windows (up to window_s=3600) need warmup
#   rows from adjacent hours to avoid NaN at every hour boundary.
#   The build function loads prev_hour + target_hour + next_hour into a single
#   3-hour DataFrame, computes all S5 features on it, then trims the output
#   back to the target hour before writing. Context files (prev/next) are
#   loaded silently when available and skipped without error when missing.
#   Context files are NEVER archived — only the target file is archived.
#
# TOPOLOGICAL SORT:
#   Intra-S5 dependency chains (max depth 2):
#     rolling_median  → no intra-S5 deps           (Level 0)
#     rolling_mad     → depends on rolling_median   (Level 1)
#     robust_shock    → depends on median + MAD     (Level 2)
#   Self-contained operators (no intra-S5 deps):
#     robust_zscore, signal_persist
#   The engine topologically sorts all specs (Kahn's BFS) before computation
#   so dependencies are always satisfied when needed.
#
# OPERATOR SEMANTICS (S5-specific):
#   derived.rolling_median:
#     output = rolling median(x, window_s)
#
#   derived.rolling_mad:
#     output = rolling median(|x - median_x|, window_s)
#     Requires pre-computed median column (intra-S5 dep).
#
#   derived.robust_shock:
#     output = |x - median_x| / (MAD_x + eps)
#     Requires both median and MAD columns (intra-S5 deps).
#     NOTE: returns the absolute shock magnitude (not signed).
#
#   derived.robust_zscore  [SELF-CONTAINED, no intra-S5 deps]:
#     output = (x - rolling_median(x)) / (scale * rolling_MAD(x) + eps)
#     Both median and MAD are computed inline from the raw input column.
#     scale = 1.4826 (MAD → consistent σ estimate under normality).
#
#   derived.signal_persist  [S5 FORMULA — differs from S4]:
#     output = abs(roll_mean(x)) / (roll_mean(abs(x)) + eps)
#     Range [0, 1]. Threshold-free, magnitude-weighted persistence.
#     1 = fully consistent directional signal, 0 = perfectly balanced.
#
# FILL CONTRACT:
#   - rolling_median / rolling_mad: first (window_s - 1) rows → NaN.
#   - robust_shock: NaN where any of base / median / MAD is NaN.
#   - robust_zscore: first (window_s - 1) rows → NaN.
#   - signal_persist: first window_s rows → NaN (min_periods = window_s).
#   - All operators propagate NaN from inputs (NaN in → NaN out).
#
# POST-BUILD ARCHIVE:
#   After successful S5 computation the engine optionally moves consumed
#   S4 feature files into a date-partitioned archive directory:
#       data_storage/data_archive/{date_str}/s4_features/
#   Only the target hour file is archived; prev/next context files are not.
#
# FIXES APPLIED:
#               3h block, trim to target before writing. Eliminates NaN at
#               every hour boundary for all rolling operators.
#               which produced extreme values ~1e9 on flat/sparse windows).
#               to [-20, 20] after full compute pass.
#               NaN (not window_s - 1) — matches min_periods = window_s.
#               context window for accurate boundary diagnostics.
#
# FEATURE GROUPS (80 total):
#   Cross-Market (8):  persist + robust_zscore for sf net-add/cancel ratios
#   Dynamics    (48):  shock pipelines (absorption_break, vacuum_score) +
#                      per-market net-add/cancel signal_persist
#   Pressure    (24):  robust_zscore for per-market net-add/cancel rolling sums
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
from etl.operators.s5_operators import S5_OPERATORS

# ── S5 Spec Imports ──────────────────────────────────────────────────
from etl.spec.s5.s5_cross_market import S5_CROSS_MARKET_FEATURES
from etl.spec.s5.s5_dynamics import S5_DYNAMICS_FEATURES
from etl.spec.s5.s5_pressure import S5_PRESSURE_FEATURES

PARQUET_COMPRESSION = "zstd"
EPS = 1e-9  # S5 default eps (matches s5_operators.py defaults)

_ENGINE_DIR = Path(__file__).resolve().parent
_DEFAULT_S4_DIR = DATA_ROOT / "s4_features"
_DEFAULT_OUT_DIR = DATA_ROOT / "s5_features"
_DEFAULT_ARCHIVE_DIR = DATA_ROOT / "data_archive"


# =============================================================================
# Utilities
# =============================================================================

def _log(enabled: bool, msg: str) -> None:
    if enabled:
        print(f"[{pd.Timestamp.utcnow().isoformat()}] [S5_FEATURE_ENGINE] {msg}")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _require_cols(df: pd.DataFrame, cols: Iterable[str], ctx: str) -> None:
    """Raise a descriptive error if any of cols are missing from df."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"{ctx}: missing required columns: {missing}. "
            f"Have {len(df.columns)} cols, first 20: {list(df.columns)[:20]}"
        )


def _safe_int(val: Any, default: int = 1) -> int:
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


# =============================================================================
# Topological Sort  (Kahn's BFS algorithm)
# =============================================================================

def _toposort_specs(specs: List[FeatureSpec]) -> List[FeatureSpec]:
    """
    Topologically sort S5 feature specs so that intra-S5 dependencies are
    computed before the features that depend on them.

    Intra-S5 dependencies are detected by checking whether a dep name matches
    any other S5 feature name in the spec list (regardless of dep.kind).
    This resolves the three-level shock chain:
        rolling_median → rolling_mad → robust_shock

    Algorithm: Kahn's BFS.  Ties broken by feature_id for determinism.
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
        queue.sort(key=lambda idx: specs[idx].feature_id or 0)
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
            f"Topological sort failed — cycle detected among "
            f"{len(remaining)} specs. First 10: {remaining[:10]}"
        )

    return [specs[i] for i in sorted_indices]


# =============================================================================
# S5 Feature Engine
# =============================================================================

class S5FeatureEngine:
    """
    Compute S5 features from S4 feature columns.

    Iterates through topologically-sorted specs, computes each feature,
    and appends it as a new column to the working DataFrame.
    Intra-S5 dependencies (mad → median, shock → mad + median) are satisfied
    by the topological ordering — no explicit chain management required.
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self._op_registry = S5_OPERATORS

    def _validate_registry(self, specs: List[FeatureSpec]) -> None:
        """Pre-compute validation: operator exists + arity check."""
        for spec in specs:
            op = spec.operator
            if op not in self._op_registry:
                raise ValueError(
                    f"S5 registry: unknown operator '{op}' "
                    f"in feature '{spec.name}' (id={spec.feature_id})"
                )
            reg = self._op_registry[op]
            actual = len(spec.depends_on)
            expected = reg.n_input_cols
            if expected > 0 and actual != expected:
                raise ValueError(
                    f"S5 arity mismatch for '{spec.name}': '{op}' "
                    f"expects {expected}, got {actual}"
                )

    # =========================================================================
    # Main Entry
    # =========================================================================

    def compute_all(
        self,
        s4_df: pd.DataFrame,
        specs: List[FeatureSpec],
        features_filter: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Compute all S5 features on top of the S4 feature DataFrame.

        The caller is responsible for passing a DataFrame that already includes
        context rows from adjacent hours (prev + target + next). The engine
        computes features across the full block; the caller trims back to the
        target hour after this call returns.

        Args:
            s4_df:            DataFrame with bucket_dt_utc + S0–S4 columns.
                              May span multiple hours (context window included).
            specs:            List of S5 FeatureSpec objects.
            features_filter:  Optional subset of feature names to compute.

        Returns:
            Wide DataFrame with bucket_dt_utc + S0–S4 + S5 columns.
            All previous-stage columns are retained; S5 columns are appended.
        """
        _require_cols(s4_df, ["bucket_dt_utc"], "s4_df")

        df = s4_df.copy()
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


        # Deduplicate timestamps: keep last occurrence (most recently computed row).
        # Guards against upstream S4 files that were built before a dedup fix,
        # and provides a permanent safety net for any future duplicate rows.
        n_before = len(df)
        df = df.drop_duplicates(subset=["bucket_dt_utc"], keep="last").reset_index(drop=True)
        if len(df) < n_before:
            _log(self.verbose, f"Deduped {n_before - len(df)} duplicate timestamp(s) from input")

        if features_filter:
            wanted = set(features_filter)
            specs = [s for s in specs if s.name in wanted]

        # --- Registry validation ---
        self._validate_registry(specs)

        sorted_specs = _toposort_specs(specs)
        _log(self.verbose,
             f"Computing S5 features: {len(sorted_specs)} specs "
             f"(toposorted from {len(specs)} input specs) "
             f"on {len(df)} rows (incl. context)")

        t0 = time.time()
        computed, errors = 0, 0
        s5_names: List[str] = []

        for spec in sorted_specs:
            try:
                result = self._compute_one(spec, df)
                df[spec.name] = result
                s5_names.append(spec.name)
                computed += 1
            except Exception as e:
                errors += 1
                if self.verbose:
                    print(f"  [WARN] {spec.name}: {e}")

        elapsed = time.time() - t0
        _log(self.verbose,
             f"Done. computed={computed} errors={errors} "
             f"in {elapsed:.2f}s | total cols={len(df.columns)} "
             f"(S0–S4 retained + {len(s5_names)} new S5)")

        # Catches any z_ or *_robust_z_* value that escaped operator-level
        # clipping (e.g. near-zero-but-nonzero MAD producing extreme values).
        # Note: Pressure features use *_robust_z_* naming, not z_* prefix,
        # so both patterns must be matched.
        z_cols = [
            c for c in s5_names
            if c.startswith("z_") or "_robust_z_" in c
        ]
        if z_cols:
            df[z_cols] = df[z_cols].clip(-20.0, 20.0)

        return df

    # =========================================================================
    # Dispatch
    # =========================================================================

    def _compute_one(self, spec: FeatureSpec, df: pd.DataFrame) -> pd.Series:
        """
        Dispatch to the appropriate S5 operator implementation.

        Column resolution order for all operators:
          1. Explicit param name (e.g. params["input_col"], params["base_col"])
          2. Positional fallback to deps[i]
        This makes specs robust: params encode intent, deps encode the
        dependency graph for toposort.
        """
        op     = spec.operator
        params = spec.params
        deps   = spec.depends_on
        name   = spec.name

        dep_names = [d.name for d in deps]
        _require_cols(df, dep_names, name)

        if op == "derived.rolling_median":
            return self._op_rolling_median(df, dep_names, params, name)

        if op == "derived.rolling_mad":
            return self._op_rolling_mad(df, dep_names, params, name)

        if op == "derived.robust_shock":
            return self._op_robust_shock(df, dep_names, params, name)

        if op == "derived.robust_zscore":
            return self._op_robust_zscore(df, dep_names, params, name)

        if op == "derived.signal_persist":
            return self._op_signal_persist(df, dep_names, params, name)

        raise ValueError(f"{name}: unknown S5 operator '{op}'")

    # =========================================================================
    # OPERATOR IMPLEMENTATIONS
    # =========================================================================

    # ── derived.rolling_median ───────────────────────────────────────────────

    def _op_rolling_median(
        self,
        df: pd.DataFrame,
        deps: List[str],
        params: Dict[str, Any],
        name: str,
    ) -> pd.Series:
        """
        Rolling median of a single input column.

        Column: params["input_col"] → deps[0] fallback.
        Window: params["window_s"].
        min_periods: params["min_periods"] → max(2, window_s // 4) default.
        [FIX-S5-6] Lowered from max(5, w//2) to max(2, w//4) — consistent
                   with S4 [FIX-C]. Reduces unnecessary NaN rows at window
                   boundaries even after context loading.

        First (window_s - 1) rows → NaN (insufficient history).
        """
        col_name = params.get("input_col", deps[0])
        if col_name not in df.columns:
            col_name = deps[0]

        window_s = _safe_int(params.get("window_s", 1))
        min_p    = _safe_int(params.get("min_periods", max(2, window_s // 4)))

        col = df[col_name].astype("float64")
        return col.rolling(window=window_s, min_periods=min_p).median()

    # ── derived.rolling_mad ─────────────────────────────────────────────────

    def _op_rolling_mad(
        self,
        df: pd.DataFrame,
        deps: List[str],
        params: Dict[str, Any],
        name: str,
    ) -> pd.Series:
        """
        Rolling MAD: rolling median of |x - median_x|.

        Requires a pre-computed rolling median column (intra-S5 dependency,
        guaranteed present by toposort ordering).

        Column resolution:
          base_col   → params["base_col"]   → dep NOT starting with "median_"
          median_col → params["median_col"] → dep starting with "median_"
        Positional fallback: deps[0] = base, deps[1] = median (or whichever
        is not "median_"-prefixed vs is).

        Window: params["window_s"].
        min_periods: params["min_periods"] → max(2, window_s // 4).
        [FIX-S5-6] Lowered from max(5, w//2) to max(2, w//4).
        """
        window_s = _safe_int(params.get("window_s", 1))
        min_p    = _safe_int(params.get("min_periods", max(2, window_s // 4)))

        # Prefer explicit param names
        base_col   = params.get("base_col")
        median_col = params.get("median_col")

        # Fallback: detect by "median_" prefix
        if base_col is None or median_col is None:
            for d in deps:
                if d.startswith("median_"):
                    median_col = median_col or d
                else:
                    base_col = base_col or d

        if base_col is None or median_col is None:
            raise ValueError(
                f"{name}: rolling_mad requires base_col and median_col. "
                f"Got deps={deps}, params keys={list(params.keys())}"
            )

        _require_cols(df, [base_col, median_col], name)

        base = df[base_col].astype("float64")
        med  = df[median_col].astype("float64")
        abs_dev = (base - med).abs()
        return abs_dev.rolling(window=window_s, min_periods=min_p).median()

    # ── derived.robust_shock ─────────────────────────────────────────────────

    def _op_robust_shock(
        self,
        df: pd.DataFrame,
        deps: List[str],
        params: Dict[str, Any],
        name: str,
    ) -> pd.Series:
        """
        Robust shock / event-strength score:
            shock = |x - median_x| / (MAD_x + eps)

        Both median_col and mad_col must already be present in df (computed
        earlier in the toposorted sequence as intra-S5 deps).

        Column resolution:
          base_col   → params["base_col"]   → dep not starting with mad_/median_
          median_col → params["median_col"] → dep starting with "median_"
          mad_col    → params["mad_col"]    → dep starting with "mad_"

        eps: params["eps"] → EPS default (1e-9).

        [FIX-S5-4] Zero-MAD → NaN. Dividing by eps when MAD=0 (quiet/flat
                   window) produced extreme shock values (~1e9). Return NaN
                   instead to signal that the shock is undefined.
        [FIX-S5-4] Output clipped to [0, 50]. Shock is always ≥ 0 (absolute
                   value). Cap at 50 to guard against near-zero MAD artefacts.

        Returns: absolute shock magnitude ≥ 0. NaN where any input is NaN
                 or where MAD = 0.
        """
        eps = _safe_float(params.get("eps", EPS), EPS)

        # Prefer explicit param names
        base_col   = params.get("base_col")
        median_col = params.get("median_col")
        mad_col    = params.get("mad_col")

        # Fallback: detect by name prefix
        if None in (base_col, median_col, mad_col):
            for d in deps:
                if d.startswith("mad_") and mad_col is None:
                    mad_col = d
                elif d.startswith("median_") and median_col is None:
                    median_col = d
                elif base_col is None \
                        and not d.startswith("mad_") \
                        and not d.startswith("median_"):
                    base_col = d

        if None in (base_col, median_col, mad_col):
            raise ValueError(
                f"{name}: robust_shock requires base_col, median_col, mad_col. "
                f"Got deps={deps}, params keys={list(params.keys())}"
            )

        _require_cols(df, [base_col, median_col, mad_col], name)

        x   = df[base_col].astype("float64")
        med = df[median_col].astype("float64")
        mad = df[mad_col].astype("float64")

        denom = mad.where(mad > 0, np.nan)

        # upper cap guards against near-zero MAD artefacts.
        return ((x - med).abs() / denom).clip(0.0, 50.0)

    # ── derived.robust_zscore ────────────────────────────────────────────────

    def _op_robust_zscore(
        self,
        df: pd.DataFrame,
        deps: List[str],
        params: Dict[str, Any],
        name: str,
    ) -> pd.Series:
        """
        Self-contained robust z-score (no intra-S5 deps required):
            z = (x - rolling_median(x)) / (scale * rolling_MAD(x) + eps)

        Both rolling median and MAD are computed inline from the single
        input column. This differs from robust_shock which uses pre-computed
        intra-S5 median/MAD columns.

        Column: params["input_col"] → deps[0] fallback.
        Window: params["window_s"].
        scale: params["scale"] → 1.4826 default.
        eps:   params["eps"]   → EPS default.
        min_periods: params["min_periods"] → max(2, window_s // 4).
        [FIX-S5-7] min_periods brought in line with FIX-S5-6.

        [FIX-S5-2] Zero-MAD → NaN. Dividing by eps when MAD=0 produced
                   extreme values (~1e9) on flat/sparse windows. Return NaN
                   instead to signal that the z-score is undefined.
        [FIX-S5-3] Output clipped to [-20, 20]. Guards against tiny-but-
                   nonzero MAD (one spike in a mostly-zero window) producing
                   extreme values.
        """
        col_name = params.get("input_col", deps[0])
        if col_name not in df.columns:
            col_name = deps[0]

        window_s = _safe_int(params.get("window_s", 1))
        min_p    = _safe_int(params.get("min_periods", max(2, window_s // 4)))
        scale    = _safe_float(params.get("scale", 1.4826), 1.4826)
        eps      = _safe_float(params.get("eps", EPS), EPS)

        col = df[col_name].astype("float64")
        rolling_med = col.rolling(window=window_s, min_periods=min_p).median()
        abs_dev     = (col - rolling_med).abs()
        rolling_mad = abs_dev.rolling(window=window_s, min_periods=min_p).median()

        denom = scale * rolling_mad
        denom = denom.where(denom > 0, np.nan)

        return ((col - rolling_med) / denom).clip(-20.0, 20.0)

    # ── derived.signal_persist ───────────────────────────────────────────────

    def _op_signal_persist(
        self,
        df: pd.DataFrame,
        deps: List[str],
        params: Dict[str, Any],
        name: str,
    ) -> pd.Series:
        """
        S5 directional persistence formula (threshold-free, magnitude-weighted):
            persist = abs(roll_mean(x)) / (roll_mean(abs(x)) + eps)

        Range [0, 1].
          1 → signal is fully consistent in one direction (perfectly persistent).
          0 → signal oscillates symmetrically around zero (no net direction).

        This differs from the S4 signal_persist (sign-consistency fraction).
        The S5 formula is sensitive to signal *magnitude* as well as sign:
        a strong one-sided signal scores higher than a weak one-sided signal
        of equal sign consistency.

        Column: params["input_col"] → deps[0] fallback.
        Window: params["window_s"].
        eps:    params["eps"] → EPS default.
        min_periods: window_s (require full window for reliable persistence).
        [FIX-S5-8] Header corrected: first window_s rows → NaN (not
                   window_s - 1), because min_periods = window_s (full window
                   required — a single large value in a partial window would
                   dominate the persistence ratio).
        """
        col_name = params.get("input_col", deps[0])
        if col_name not in df.columns:
            col_name = deps[0]

        window_s = _safe_int(params.get("window_s", 1))
        eps      = _safe_float(params.get("eps", EPS), EPS)
        # Full window required: partial windows give unreliable persistence
        # estimates (a single large value can dominate the ratio).
        min_p    = _safe_int(params.get("min_periods", window_s))

        col = df[col_name].astype("float64")

        roll_mean_x   = col.rolling(window=window_s, min_periods=min_p).mean()
        roll_mean_abs = col.abs().rolling(window=window_s, min_periods=min_p).mean()

        # Jensen's inequality guarantees |mean(x)| ≤ mean(|x|), so the ratio
        # should never exceed 1.0.  The clip guards against rare floating-point
        # or inf-propagation pathologies in the upstream S4 column.
        return (roll_mean_x.abs() / (roll_mean_abs + eps)).clip(0.0, 1.0)


# =============================================================================
# Feature Registry
# =============================================================================

ALL_S5_FEATURES: List[FeatureSpec] = (
    list(S5_CROSS_MARKET_FEATURES)
    + list(S5_DYNAMICS_FEATURES)
    + list(S5_PRESSURE_FEATURES)
)


def _find_feature_by_name(features: Iterable[FeatureSpec], name: str) -> FeatureSpec:
    for f in features:
        if f.name == name:
            return f
    raise KeyError(f"S5 feature not found: '{name}'")


def _resolve_dependency_chain(
    spec: FeatureSpec,
    all_specs: List[FeatureSpec],
) -> List[FeatureSpec]:
    """
    Recursively collect the target spec plus all S5 specs it transitively
    depends on. Used for single-feature debug mode.
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


# =============================================================================
# =============================================================================

def _adjacent_hour(date_str: str, hour: int, delta: int) -> Tuple[str, int]:
    """
    Compute the date_str and hour for (hour + delta), crossing day boundaries.

    Args:
        date_str: "YYYY-MM-DD"
        hour:     0–23
        delta:    +1 or -1

    Returns:
        (new_date_str, new_hour)
    """
    dt = datetime.strptime(date_str, "%Y-%m-%d") + timedelta(hours=delta + hour)
    return dt.strftime("%Y-%m-%d"), dt.hour


def _s4_path_for(s4_dir: str, asset: str, date_str: str, hour: int) -> Path:
    """Construct the S4 parquet path for a given asset/date/hour."""
    hh = f"{int(hour):02d}"
    return Path(s4_dir) / f"s4_features_{asset.lower()}_{date_str}_{hh}.parquet"


def _load_context_block(
    s4_dir: str,
    asset: str,
    date_str: str,
    hour: int,
    verbose: bool,
) -> Tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    """
    [FIX-S5-1] Load prev_hour + target_hour + next_hour into one DataFrame.

    The target hour file MUST exist (raises FileNotFoundError otherwise).
    Prev and next hour files are loaded when available; missing files are
    skipped silently (logs when verbose=True).

    Returns:
        combined_df:    Concatenated DataFrame sorted by bucket_dt_utc,
                        spanning up to 3 hours of data.
        target_start:   First timestamp of the target hour (inclusive).
        target_end:     Last timestamp of the target hour (inclusive).
    """
    target_path = _s4_path_for(s4_dir, asset, date_str, hour)
    if not target_path.exists():
        raise FileNotFoundError(f"Missing S4 feature file: {target_path}")

    frames: List[pd.DataFrame] = []

    # ── Previous hour ────────────────────────────────────────────────────────
    prev_date, prev_hour = _adjacent_hour(date_str, hour, -1)
    prev_path = _s4_path_for(s4_dir, asset, prev_date, prev_hour)
    if prev_path.exists():
        _log(verbose, f"Loading prev-hour context: {prev_path.name}")
        frames.append(pq.read_table(str(prev_path)).to_pandas())
    else:
        _log(verbose, f"Prev-hour context not found (skip): {prev_path.name}")

    # ── Target hour ──────────────────────────────────────────────────────────
    _log(verbose, f"Loading target S4 features: {target_path.name}")
    target_df = pq.read_table(str(target_path)).to_pandas()
    frames.append(target_df)

    # ── Next hour ────────────────────────────────────────────────────────────
    next_date, next_hour = _adjacent_hour(date_str, hour, +1)
    next_path = _s4_path_for(s4_dir, asset, next_date, next_hour)
    if next_path.exists():
        _log(verbose, f"Loading next-hour context: {next_path.name}")
        frames.append(pq.read_table(str(next_path)).to_pandas())
    else:
        _log(verbose, f"Next-hour context not found (skip): {next_path.name}")

    combined = pd.concat(frames, ignore_index=True)
    combined["bucket_dt_utc"] = pd.to_datetime(combined["bucket_dt_utc"], utc=True)
    combined = combined.sort_values("bucket_dt_utc").reset_index(drop=True)

    # Determine target hour time range for later trimming
    target_df["bucket_dt_utc"] = pd.to_datetime(target_df["bucket_dt_utc"], utc=True)
    target_start = target_df["bucket_dt_utc"].min()
    target_end   = target_df["bucket_dt_utc"].max()

    _log(
        verbose,
        f"Context block: {len(combined)} rows across "
        f"{combined['bucket_dt_utc'].min()} → {combined['bucket_dt_utc'].max()} "
        f"| target window: {target_start} → {target_end}",
    )

    return combined, target_start, target_end


# =============================================================================
# I/O Helpers
# =============================================================================

def _paths_for_hour(
    s4_dir: str,
    out_dir: str,
    asset: str,
    date_str: str,
    hour: int,
) -> Tuple[Path, Path]:
    """Derive S4 input path and S5 output path for one asset-hour."""
    hh     = f"{int(hour):02d}"
    suffix = f"{date_str}_{hh}.parquet"
    a      = asset.lower()
    s4_path  = Path(s4_dir)  / f"s4_features_{a}_{suffix}"
    out_path = Path(out_dir) / f"s5_features_{a}_{suffix}"
    return s4_path, out_path


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
    Move consumed S4 feature files into a date-partitioned archive folder.

    Layout: data_archive/{date_str}/s4_features/{filename}.parquet
    Files that don't exist or are already archived are silently skipped.
    Only the target hour file is archived; prev/next context files are not.
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
        _log(verbose, f"Archived: {src_path.name} → {label}/")


# =============================================================================
# Build + Archive
# =============================================================================

def _atomic_write_parquet(df: pd.DataFrame, out_path: Path,
                           compression: str = PARQUET_COMPRESSION) -> None:
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


def build_s5_features_for_hour(
    s4_dir: str,
    out_dir: str,
    asset: str,
    date_str: str,
    hour: int,
    features_filter: Optional[List[str]] = None,
    archive_dir: Optional[str] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Main entry point: compute S5 features for one asset-hour, write parquet,
    then optionally archive the consumed S4 feature file.

    [FIX-S5-1] Loads prev_hour + target_hour + next_hour into a 3-hour
    context block to avoid NaN gaps at hour boundaries from rolling operators
    with large windows (up to window_s=3600). After computation the output is
    trimmed back to the target hour before writing.

    The output S5 parquet retains ALL previous-stage columns (S0–S4) plus
    the newly computed S5 features.

    Input dir:  data_storage/s4_features/
    Output dir: data_storage/s5_features/

    Args:
        s4_dir:           Directory containing S4 feature parquets.
        out_dir:          Directory to write S5 feature parquets.
        asset:            "btc", "eth", or "bnb".
        date_str:         Date string, e.g. "2026-02-24".
        hour:             Hour (0–23).
        features_filter:  Optional list of feature names to compute (None = all).
        archive_dir:      If set, move the TARGET S4 file here after success.
                          Files land in {archive_dir}/{date_str}/s4_features/.
                          Prev/next context files are NOT archived.
        verbose:          Print progress logs.

    Returns:
        The computed feature DataFrame for the target hour only
        (S0–S4 + S5 columns, also written to disk).
    """
    _ensure_dir(Path(out_dir))
    _, out_path = _paths_for_hour(s4_dir, out_dir, asset, date_str, hour)

    combined_df, target_start, target_end = _load_context_block(
        s4_dir, asset, date_str, hour, verbose
    )
    _log(verbose, f"Context block loaded: {len(combined_df)} rows, "
                  f"{len(combined_df.columns)} cols")

    # Compute S5 features across the full context block
    engine = S5FeatureEngine(verbose=verbose)
    full_df = engine.compute_all(
        combined_df,
        specs=ALL_S5_FEATURES,
        features_filter=features_filter,
    )

    # Trim back to target hour only before writing
    target_mask = (
        (full_df["bucket_dt_utc"] >= target_start) &
        (full_df["bucket_dt_utc"] <= target_end)
    )
    out_df = full_df.loc[target_mask].reset_index(drop=True)
    _log(verbose, f"Trimmed to target hour: {len(out_df)} rows "
                  f"({target_start} → {target_end})")

    _log(verbose, f"Saving S5 features to: {out_path}")
    _atomic_write_parquet(out_df, out_path)

    mb = out_path.stat().st_size / (1024 * 1024)
    _log(verbose, f"Saved: {mb:.2f} MB | rows={len(out_df)} cols={len(out_df.columns)}")

    # Archive only the target hour S4 file, never the context files
    if archive_dir is not None:
        target_s4_path = _s4_path_for(s4_dir, asset, date_str, hour)
        _archive_files(
            files_to_move=[target_s4_path],
            archive_dir=Path(archive_dir),
            date_str=date_str,
            sub_dir="s4_features",
            verbose=verbose,
        )

    return out_df


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "S5 feature engine: compute S5 derived features "
            "from S4 feature parquets.\n\n"
            "Input:  data_storage/s4_features/\n"
            "Output: data_storage/s5_features/"
        )
    )
    ap.add_argument("--s4-dir", type=str, default=str(_DEFAULT_S4_DIR),
                    help="Directory containing S4 feature parquets.")
    ap.add_argument("--out-dir", type=str, default=str(_DEFAULT_OUT_DIR),
                    help="Directory to write S5 feature parquets.")
    ap.add_argument("--archive-dir", type=str, default=str(_DEFAULT_ARCHIVE_DIR),
                    help="Archive directory for consumed S4 files. "
                         "Files land in {archive-dir}/{date}/s4_features/.")
    ap.add_argument("--no-archive", action="store_true",
                    help="Skip archiving (keep S4 files in place).")
    ap.add_argument("--asset", type=str, required=True, choices=["btc", "eth", "bnb"])
    ap.add_argument("--date", type=str, required=True,
                    help="Date string, e.g. 2026-02-24.")
    ap.add_argument("--hour", type=int, required=True,
                    help="Hour (0–23).")
    ap.add_argument("--features", type=str, nargs="+",
                    help="Compute only these named features (default: all).")
    ap.add_argument("--feature", type=str,
                    help="Single-feature debug mode: compute one feature and "
                         "print its tail (no write, no archive).")
    ap.add_argument("--tail", type=int, default=10,
                    help="Rows to print in --feature debug mode.")
    ap.add_argument("--with-context", action="store_true",
                    help="[FIX-S5-10] In --feature debug mode, load prev/next "
                         "hour context files for accurate boundary diagnostics. "
                         "Matches production behaviour.")
    ap.add_argument("--quiet", "-q", action="store_true",
                    help="Suppress progress logs.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print paths and spec count; do not read or write files.")
    ap.add_argument("--format", choices=["table", "csv"], default="table",
                    help="Output format for --feature debug mode.")

    args = ap.parse_args()
    verbose = not args.quiet

    # ── Dry run ──────────────────────────────────────────────────────────────
    if args.dry_run:
        s4_path, out_path = _paths_for_hour(
            args.s4_dir, args.out_dir, args.asset, args.date, args.hour
        )
        prev_date, prev_hour = _adjacent_hour(args.date, args.hour, -1)
        next_date, next_hour = _adjacent_hour(args.date, args.hour, +1)
        prev_path = _s4_path_for(args.s4_dir, args.asset, prev_date, prev_hour)
        next_path = _s4_path_for(args.s4_dir, args.asset, next_date, next_hour)
        archive_label = "disabled" if args.no_archive else args.archive_dir
        print(f"Would read prev S4:  {prev_path}  (exists={prev_path.exists()})")
        print(f"Would read S4:       {s4_path}")
        print(f"Would read next S4:  {next_path}  (exists={next_path.exists()})")
        print(f"Would write S5:      {out_path}")
        print(f"Archive dir:         {archive_label}")
        print(f"Total S5 specs:      {len(ALL_S5_FEATURES)}")
        return

    # ── Single-feature debug mode ────────────────────────────────────────────
    if args.feature:
        if args.with_context:
            combined_df, target_start, target_end = _load_context_block(
                args.s4_dir, args.asset, args.date, args.hour, verbose
            )
            work_df = combined_df
        else:
            s4_path = _s4_path_for(args.s4_dir, args.asset, args.date, args.hour)
            if not s4_path.exists():
                raise FileNotFoundError(f"Missing S4 feature file: {s4_path}")
            work_df = pq.read_table(str(s4_path)).to_pandas()
            work_df["bucket_dt_utc"] = pd.to_datetime(
                work_df["bucket_dt_utc"], utc=True
            )
            work_df = work_df.sort_values("bucket_dt_utc").reset_index(drop=True)
            target_start = work_df["bucket_dt_utc"].min()
            target_end   = work_df["bucket_dt_utc"].max()

        spec       = _find_feature_by_name(ALL_S5_FEATURES, args.feature)
        engine     = S5FeatureEngine(verbose=verbose)
        all_needed = _resolve_dependency_chain(spec, ALL_S5_FEATURES)
        sorted_needed = _toposort_specs(all_needed)

        for s in sorted_needed:
            work_df[s.name] = engine._compute_one(s, work_df)

        # If context was loaded, trim to target hour before showing tail
        if args.with_context:
            mask = (
                (work_df["bucket_dt_utc"] >= target_start) &
                (work_df["bucket_dt_utc"] <= target_end)
            )
            work_df = work_df.loc[mask].reset_index(drop=True)

        out = work_df[["bucket_dt_utc", spec.name]].tail(args.tail)

        try:
            print(
                out.to_csv(index=False) if args.format == "csv"
                else out.to_string(index=False)
            )
        except BrokenPipeError:
            pass
        return

    # ── Full build ───────────────────────────────────────────────────────────
    build_s5_features_for_hour(
        s4_dir=args.s4_dir,
        out_dir=args.out_dir,
        asset=args.asset,
        date_str=args.date,
        hour=args.hour,
        features_filter=args.features,
        archive_dir=None if args.no_archive else args.archive_dir,
        verbose=verbose,
    )


if __name__ == "__main__":
    main()