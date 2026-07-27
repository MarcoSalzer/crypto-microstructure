# ==============================================================================
# S0 Context Batch — Health + Calendar + Usability on Continuous Bucket Grid
#
# PURPOSE:
#   Build the S0 "context" table on a continuous bucket grid for one asset:
#     - Health flags (from s0_health.py; L2-centric global semantics)
#     - Calendar/session flags (deterministic from bucket_dt_utc)
#     - Soft health (B1a: allows limited missing L2 combos)
#     - Usability flag (rolling rule on the soft health flag)
#     - Continuous time grid (explicit missing buckets)
#
#   This module does NOT compute "market" features (aggression/activity/
#   price/volume/bookshape). Those are computed later by the feature engine
#   and joined with this context table.
#
# ARCHITECTURE CONTEXT:
#   Pipeline: Binance-only, multi-asset (BTC + ETH + BNB).
#   Each asset gets its own context file per hour.
#
#   I/O convention (current layout):
#     Input (per asset, e.g. btc):
#       {data_dir}/trades_btc_spot_YYYY-MM-DD_HH.parquet
#       {data_dir}/trades_btc_fut_YYYY-MM-DD_HH.parquet
#       {data_dir}/lobdeep_btc_spot_YYYY-MM-DD_HH.parquet
#       {data_dir}/lobdeep_btc_fut_YYYY-MM-DD_HH.parquet
#
#     Output:
#       {out_dir}/s0_context_btc_YYYY-MM-DD_HH.parquet
#       {out_dir}/s0_context_eth_YYYY-MM-DD_HH.parquet
#
#   NOTE: lob20 sources have been removed from the pipeline entirely.
#   The old 4-venue pipeline had lob20_spot / lob20_fut from each venue.
#   The current Binance-only pipeline produces only lobdeep.
#
# SOFT HEALTH (B1a):
#   With only 2 L2 combos (lobdeep_spot + lobdeep_fut), the soft missing
#   budget defaults to 1. This means at most one of spot/fut can be missing
#   while still passing soft health. Budget=2 would allow both to be missing
#   (defeats purpose). Budget=0 is equivalent to strict health.
#
# ==============================================================================

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PARQUET_COMPRESSION = "zstd"

# Microstructure-friendly defaults (start-point for B1a soft-health usability)
DEFAULT_USABILITY_WINDOW_S = 60          # trailing window length (seconds)
DEFAULT_USABILITY_MIN_RATIO = 0.95       # >= 95% healthy in window (soft health)
DEFAULT_USABILITY_MAX_BAD_STREAK = 5     # no >5 consecutive bad seconds (soft health)

# Soft-health budget (B1a): allow up to M missing L2 combos per bucket.
# With 2 combos (Binance lobdeep spot + fut), M=1 means at most one side
# can be missing. M=2 would trivially pass everything. M=0 = strict.
DEFAULT_SOFT_MISSING_BUDGET = 1

# -----------------------------------------------------------------------------
# Usability reason codes (int8)
# -----------------------------------------------------------------------------
REASON_OK = np.int8(0)
REASON_WINDOW_WARMUP = np.int8(1)          # too little history for full window
REASON_HEALTH_RATIO_BELOW_MIN = np.int8(2) # rolling health ratio below min
REASON_BAD_STREAK_EXCEEDED = np.int8(3)    # streak exceeded max_bad_streak
REASON_EXPLICIT_HEALTH_BAD = np.int8(4)    # current bucket health == 0
REASON_UNKNOWN = np.int8(9)                # fallback


# -----------------------------------------------------------------------------
# Imports (single-source-of-truth, repo-layout tolerant)
# -----------------------------------------------------------------------------
def _try_imports():
    """
    Import helpers in a repo-layout tolerant way.
    Defensive because scripts may be run from different cwd roots.
    """
    import importlib

    def _try(mod_path: str, attr: str):
        try:
            m = importlib.import_module(mod_path)
            return getattr(m, attr)
        except Exception:
            return None

    # Calendar engine
    compute_calendar_features = _try("etl.engine.s0_calendar", "compute_calendar_features")

    # Calendar spec helper (feature names come from the spec module; engine is a fallback)
    get_calendar_feature_names = (
        _try("etl.spec.s0.s0_calendar_spec", "get_calendar_feature_names")
        or _try("etl.engine.s0_calendar", "get_calendar_feature_names")
    )

    # Health engine
    compute_health_features = _try("etl.engine.s0_health", "compute_health_features")
    HealthSourcePaths = _try("etl.engine.s0_health", "SourcePaths")

    # Health spec helper (feature names come from the spec module; engine is a fallback)
    get_health_feature_names = (
        _try("etl.spec.s0.s0_health_spec", "get_health_feature_names")
        or _try("etl.engine.s0_health", "get_health_feature_names")
    )

    return (
        compute_calendar_features,
        get_calendar_feature_names,
        compute_health_features,
        HealthSourcePaths,
        get_health_feature_names,
    )


(
    compute_calendar_features,
    get_calendar_feature_names,
    compute_health_features,
    HealthSourcePaths,
    get_health_feature_names,
) = _try_imports()


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
@dataclass
class S0ContextConfig:
    """
    Configuration for the S0 context build.

    Note:
      - Context is computed on a fixed bucket grid of `resample`.
      - Usability is a rolling rule on the soft health flag.
      - We do NOT drop bad buckets. We only label them.
      - When date_str + hour are provided, the bucket grid is built
        deterministically (HH:00:00 .. HH:59:59 UTC, exactly 3600 rows at 1s).
        Without them the grid falls back to observed min/max (compat behaviour).
    """
    resample: str = "1s"
    usability_window_s: int = DEFAULT_USABILITY_WINDOW_S
    usability_min_ratio: float = DEFAULT_USABILITY_MIN_RATIO
    usability_max_bad_streak: int = DEFAULT_USABILITY_MAX_BAD_STREAK

    # Soft-health (B1a): allow up to M missing L2 combos per bucket.
    # Default 1 (out of 2 combos): at most one of spot/fut can be missing.
    soft_missing_budget: int = DEFAULT_SOFT_MISSING_BUDGET

    include_calendar: bool = True
    include_health: bool = True
    verbose: bool = True

    # If True, require lobdeep paths to exist when health is enabled.
    require_lob_deep: bool = True

    # Deterministic grid anchoring (preferred).  When both are set,
    # _ensure_bucket_grid uses these instead of observed min/max.
    date_str: Optional[str] = None   # "YYYY-MM-DD"
    hour: Optional[int] = None       # 0..23


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _log(enabled: bool, msg: str) -> None:
    if enabled:
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        print(f"[{ts}] [S0_CONTEXT] {msg}")


def _parse_resample_seconds(resample: str) -> int:
    """Parse resample interval into seconds (best effort)."""
    s = str(resample).strip().lower()
    if s.endswith("s"):
        return int(s[:-1])
    if s.endswith("m") or s.endswith("min"):
        v = int(s.replace("min", "").replace("m", ""))
        return v * 60
    if s.endswith("h"):
        return int(s[:-1]) * 3600
    return int(s) if s.isdigit() else 1


def _dedupe_by_bucket_last(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse duplicate bucket_dt_utc rows using 'last' semantics."""
    if df.empty or "bucket_dt_utc" not in df.columns:
        return df
    if df["bucket_dt_utc"].duplicated().any():
        df = df.sort_values("bucket_dt_utc")
        df = df.groupby("bucket_dt_utc", as_index=False).last()
    return df


def _ensure_bucket_grid(
    result: pd.DataFrame,
    resample: str,
    date_str: Optional[str] = None,
    hour: Optional[int] = None,
) -> pd.DataFrame:
    """
    Ensure a continuous, deterministic time grid.

    Preferred mode (date_str + hour provided):
        Grid = HH:00:00 .. HH:59:59 UTC  (exactly 3600 rows at 1s).
        Buckets outside this window are DROPPED (shouldn't exist).
        Missing buckets inside the window are inserted as NaN rows.

    Fallback mode (no date_str / hour):
        Grid = observed min .. observed max (compat, non-deterministic).
        Logs a warning.
    """
    if result.empty or "bucket_dt_utc" not in result.columns:
        return result

    result = result.sort_values("bucket_dt_utc").reset_index(drop=True)

    if date_str is not None and hour is not None:
        # -- Deterministic anchor --
        t0 = pd.Timestamp(f"{date_str} {hour:02d}:00:00", tz="UTC")
        resample_s = _parse_resample_seconds(resample)
        # Exactly (3600 / resample_s) rows: 00:00:00 → 00:59:59 for 1s
        n_buckets = 3600 // resample_s
        full = pd.date_range(start=t0, periods=n_buckets, freq=resample, tz="UTC")
    else:
        # -- Compat fallback --
        print(
            "[S0_CONTEXT] WARN: _ensure_bucket_grid called without date_str/hour — "
            "grid anchored to observed min/max (non-deterministic). "
            "Pass date_str + hour via S0ContextConfig for stable row counts."
        )
        t0 = pd.to_datetime(result["bucket_dt_utc"].min(), utc=True)
        t1 = pd.to_datetime(result["bucket_dt_utc"].max(), utc=True)
        if pd.isna(t0) or pd.isna(t1):
            return result
        full = pd.date_range(start=t0, end=t1, freq=resample, tz="UTC")

    result = result.set_index("bucket_dt_utc").reindex(full).reset_index()
    result = result.rename(columns={"index": "bucket_dt_utc"})
    return result


def _compute_soft_health_b1a(res: pd.DataFrame, missing_budget: int) -> pd.Series:
    """
    Option B1a (soft health):
      - Allow up to M missing L2 combos per bucket
      - But NO other component failures (invalid_ts/reconnect/gap/crossed)

    With 2 L2 combos (Binance lobdeep spot + fut), budget=1 means at most
    one side can be missing. budget=0 is strict. budget=2 passes everything.

    Requires diagnostics columns from s0_health aggregated output.
    Falls back to strict data_health_flag if diagnostics are missing.
    """
    need = [
        "data_health_flag",
        "l2_missing_combos",
        "l2_invalid_ts_combos",
        "l2_reconnect_combos",
        "l2_gap_combos",
        "l2_crossed_combos",
    ]
    if any(c not in res.columns for c in need):
        missing_cols = [c for c in need if c not in res.columns]
        print(
            f"[S0_CONTEXT] WARN: _compute_soft_health_b1a: diagnostic columns missing "
            f"{missing_cols} — falling back to strict data_health_flag. "
            "This should not happen in a normal pipeline run; check s0_health output."
        )
        return pd.to_numeric(res.get("data_health_flag", 0), errors="coerce").fillna(0).astype("int8")

    miss = pd.to_numeric(res["l2_missing_combos"], errors="coerce").fillna(0).astype(int)
    inv = pd.to_numeric(res["l2_invalid_ts_combos"], errors="coerce").fillna(0).astype(int)
    rec = pd.to_numeric(res["l2_reconnect_combos"], errors="coerce").fillna(0).astype(int)
    gap = pd.to_numeric(res["l2_gap_combos"], errors="coerce").fillna(0).astype(int)
    crs = pd.to_numeric(res["l2_crossed_combos"], errors="coerce").fillna(0).astype(int)

    soft_ok = (miss <= int(missing_budget)) & (inv == 0) & (rec == 0) & (gap == 0) & (crs == 0)
    return soft_ok.astype("int8")


def _compute_usability_flag_on_grid_with_debug(
    df: pd.DataFrame,
    resample: str,
    window_s: int,
    health_col: str,
    min_health_ratio: float = DEFAULT_USABILITY_MIN_RATIO,
    max_bad_streak: int = DEFAULT_USABILITY_MAX_BAD_STREAK,
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Compute a usability flag on a continuous grid using two rules:
      (A) health ratio in the trailing window >= min_health_ratio
      (B) no bad streak longer than max_bad_streak inside the trailing window

    Additionally persists diagnostics:
      - usability_bad_ratio_win, usability_bad_count_win
      - usability_max_bad_streak_win, usability_warmup_flag
      - unusable_reason_code (int8)

    Reason code priority:
      1) health==0          -> 4 (explicit health bad)
      2) warmup             -> 1 (window warmup)
      3) streak > max       -> 3 (bad streak exceeded)
      4) ratio below min    -> 2 (health ratio below min)
      5) else               -> 0 (ok)
    """
    n = len(df)
    if df.empty or "bucket_dt_utc" not in df.columns or health_col not in df.columns:
        flag = pd.Series([0] * n, dtype="int8")
        dbg = pd.DataFrame({
            "usability_bad_ratio_win": np.full(n, np.nan, dtype=np.float32),
            "usability_bad_count_win": np.zeros(n, dtype=np.int16),
            "usability_max_bad_streak_win": np.zeros(n, dtype=np.int16),
            "usability_warmup_flag": np.ones(n, dtype=np.int8),
            "unusable_reason_code": np.full(n, REASON_UNKNOWN, dtype=np.int8),
        })
        return flag, dbg

    resample_s = _parse_resample_seconds(resample)
    window_buckets = max(1, int(window_s // resample_s))

    h = pd.to_numeric(df[health_col], errors="coerce").fillna(0).astype(int).clip(0, 1)
    bad = (h == 0).astype(int)

    # (A) Ratio rule
    roll_health_ratio = h.rolling(window=window_buckets, min_periods=window_buckets).mean()
    ok_ratio = roll_health_ratio >= float(min_health_ratio)

    roll_bad_ratio = (1.0 - roll_health_ratio).astype("float32")
    roll_bad_count = bad.rolling(window=window_buckets, min_periods=window_buckets).sum()

    # (B) Streak rule
    streak_len = bad.groupby((bad == 0).cumsum()).cumsum()
    max_streak_in_window = streak_len.rolling(window=window_buckets, min_periods=window_buckets).max()
    ok_streak = max_streak_in_window <= int(max_bad_streak)

    warmup = roll_health_ratio.isna()
    usable = (ok_ratio & ok_streak).fillna(False)

    # Reason codes
    reason = np.full(n, REASON_OK, dtype=np.int8)

    m_explicit_bad = (h.values == 0)
    reason[m_explicit_bad] = REASON_EXPLICIT_HEALTH_BAD

    m_warm = (~m_explicit_bad) & warmup.values
    reason[m_warm] = REASON_WINDOW_WARMUP

    m_streak_bad = (~m_explicit_bad) & (~warmup.values) & (max_streak_in_window.fillna(0).values > int(max_bad_streak))
    reason[m_streak_bad] = REASON_BAD_STREAK_EXCEEDED

    m_ratio_bad = (
        (~m_explicit_bad) & (~warmup.values) & (~m_streak_bad)
        & (~ok_ratio.fillna(False).values)
    )
    reason[m_ratio_bad] = REASON_HEALTH_RATIO_BELOW_MIN

    m_unusable = (~usable.values)
    m_none = m_unusable & (reason == REASON_OK)
    reason[m_none] = REASON_UNKNOWN

    dbg = pd.DataFrame({
        "usability_bad_ratio_win": roll_bad_ratio.values.astype(np.float32),
        "usability_bad_count_win": roll_bad_count.fillna(0).values.astype(np.int16),
        "usability_max_bad_streak_win": max_streak_in_window.fillna(0).values.astype(np.int16),
        "usability_warmup_flag": warmup.fillna(True).astype("int8").values,
        "unusable_reason_code": reason.astype(np.int8),
    })

    return usable.astype("int8"), dbg


def _compute_usability_flag_on_grid(
    df: pd.DataFrame, resample: str, window_s: int, health_col: str,
    min_health_ratio: float = DEFAULT_USABILITY_MIN_RATIO,
    max_bad_streak: int = DEFAULT_USABILITY_MAX_BAD_STREAK,
) -> pd.Series:
    flag, _dbg = _compute_usability_flag_on_grid_with_debug(
        df=df, resample=resample, window_s=window_s, health_col=health_col,
        min_health_ratio=min_health_ratio, max_bad_streak=max_bad_streak,
    )
    return flag


def _require_exists(path: Optional[str], label: str) -> None:
    if not path:
        raise FileNotFoundError(f"[S0_CONTEXT] Missing required path for {label} (got None)")
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"[S0_CONTEXT] Missing required file for {label}: {path}")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# S0 Context Builder (Health + Calendar + Usability)
# -----------------------------------------------------------------------------
class S0ContextBuilder:
    """
    Build a per-bucket context table (wide, one row per bucket) for one asset.

    Binance-only pipeline: no lob20 sources. Only lobdeep_spot + lobdeep_fut.
    Output is joined with downstream feature tables (S0 market features, S1, etc.).
    """

    def __init__(
        self,
        trades_spot: Optional[str] = None,
        trades_fut: Optional[str] = None,
        lobdeep_spot: Optional[str] = None,
        lobdeep_fut: Optional[str] = None,
        config: Optional[S0ContextConfig] = None,
    ):
        self.trades_spot = trades_spot
        self.trades_fut = trades_fut
        self.lobdeep_spot = lobdeep_spot
        self.lobdeep_fut = lobdeep_fut
        self.config = config or S0ContextConfig()

        if self.config.include_health and self.config.require_lob_deep:
            _require_exists(self.lobdeep_spot, "lobdeep_spot")
            _require_exists(self.lobdeep_fut, "lobdeep_fut")

        if HealthSourcePaths is None:
            self._health_paths = None
        else:
            # SourcePaths has 4 fields (no lob20):
            #   trades_spot, trades_fut, lobdeep_spot, lobdeep_fut
            self._health_paths = HealthSourcePaths(
                trades_spot=trades_spot,
                trades_fut=trades_fut,
                lobdeep_spot=lobdeep_spot,
                lobdeep_fut=lobdeep_fut,
            )

    def compute_health(self) -> pd.DataFrame:
        if not self.config.include_health:
            return pd.DataFrame()

        if compute_health_features is None or self._health_paths is None:
            _log(self.config.verbose, "Health: SKIPPED (health module not importable)")
            return pd.DataFrame()

        _log(self.config.verbose, "Computing health features (GLOBAL L2 semantics, Binance-only)...")
        t0 = time.time()

        df = compute_health_features(self._health_paths, resample=self.config.resample, aggregated=True)

        if df is None or df.empty:
            _log(self.config.verbose, "  Health: empty")
            return pd.DataFrame()

        needed = {"bucket_dt_utc", "data_health_flag", "l2_coverage_flag"}
        missing = needed - set(df.columns)
        if missing:
            raise RuntimeError(f"[S0_CONTEXT] Health output missing required columns: {sorted(missing)}")

        df = _dedupe_by_bucket_last(df)
        df = df.sort_values("bucket_dt_utc").reset_index(drop=True)

        _log(self.config.verbose, f"  Health: {len(df)} buckets in {time.time() - t0:.2f}s")
        return df

    def compute_calendar(self, buckets: pd.Series) -> pd.DataFrame:
        if not self.config.include_calendar:
            return pd.DataFrame()

        if compute_calendar_features is None:
            _log(self.config.verbose, "Calendar: SKIPPED (calendar module not importable)")
            return pd.DataFrame()

        _log(self.config.verbose, "Computing calendar features (deterministic from bucket_dt_utc)...")
        t0 = time.time()

        df = compute_calendar_features(buckets)
        if df is None or df.empty:
            _log(self.config.verbose, "  Calendar: empty")
            return pd.DataFrame()

        if "bucket_dt_utc" not in df.columns:
            raise RuntimeError("[S0_CONTEXT] Calendar output missing bucket_dt_utc")

        df = _dedupe_by_bucket_last(df).sort_values("bucket_dt_utc").reset_index(drop=True)
        _log(self.config.verbose, f"  Calendar: {len(df)} buckets in {time.time()-t0:.2f}s")
        return df

    def build(self) -> pd.DataFrame:
        _log(self.config.verbose, "=" * 60)
        _log(self.config.verbose, "Building S0 Context (Health + Calendar + Usability)")
        _log(self.config.verbose, "  Pipeline: Binance-only, lobdeep only (no lob20)")
        _log(self.config.verbose, "=" * 60)
        t_start = time.time()

        parts: List[pd.DataFrame] = []

        health_df = self.compute_health()
        if not health_df.empty:
            parts.append(health_df)

        if not parts:
            _log(self.config.verbose, "No context built (health empty or disabled).")
            return pd.DataFrame()

        res = parts[0]
        for p in parts[1:]:
            res = res.merge(p, on="bucket_dt_utc", how="outer")

        res = _dedupe_by_bucket_last(res).sort_values("bucket_dt_utc").reset_index(drop=True)
        res = _ensure_bucket_grid(
            res,
            self.config.resample,
            date_str=self.config.date_str,
            hour=self.config.hour,
        )

        if self.config.include_calendar:
            cal = self.compute_calendar(res["bucket_dt_utc"])
            if not cal.empty:
                res = res.merge(cal, on="bucket_dt_utc", how="left")
                res = _dedupe_by_bucket_last(res)

        # -----------------------------------------------------------------
        # Soft health (B1a) + Usability on continuous grid
        # -----------------------------------------------------------------
        if "data_health_flag" in res.columns:
            res["data_health_flag_soft"] = _compute_soft_health_b1a(
                res, missing_budget=int(self.config.soft_missing_budget)
            )

            uflag, udbg = _compute_usability_flag_on_grid_with_debug(
                res,
                resample=self.config.resample,
                window_s=self.config.usability_window_s,
                health_col="data_health_flag_soft",
                min_health_ratio=self.config.usability_min_ratio,
                max_bad_streak=self.config.usability_max_bad_streak,
            )
            res["data_usability_flag"] = uflag
            for c in udbg.columns:
                res[c] = udbg[c].values
        else:
            res["data_health_flag_soft"] = np.int8(0)
            res["data_usability_flag"] = np.int8(0)
            res["usability_bad_ratio_win"] = np.nan
            res["usability_bad_count_win"] = np.int16(0)
            res["usability_max_bad_streak_win"] = np.int16(0)
            res["usability_warmup_flag"] = np.int8(1)
            res["unusable_reason_code"] = np.int8(REASON_UNKNOWN)

        # Column ordering: bucket + health + calendar + remaining
        col_order = ["bucket_dt_utc"]

        if get_health_feature_names is not None:
            health_cols = [c for c in get_health_feature_names() if c in res.columns]
        else:
            health_cols = [c for c in ["data_health_flag", "l2_coverage_flag", "data_usability_flag"] if c in res.columns]

        for c in [
            "data_health_flag_soft",
            "data_usability_flag",
            "unusable_reason_code",
            "usability_warmup_flag",
            "usability_bad_ratio_win",
            "usability_bad_count_win",
            "usability_max_bad_streak_win",
        ]:
            if c in res.columns and c not in health_cols:
                health_cols.append(c)

        col_order += sorted(set(health_cols))

        if get_calendar_feature_names is not None:
            cal_cols = [c for c in get_calendar_feature_names() if c in res.columns]
        else:
            cal_cols = [
                c for c in [
                    "session_sydney", "session_tokyo", "session_asia",
                    "session_london", "session_newyork",
                    "session_overlap_flag", "us_holiday", "us_rth",
                ] if c in res.columns
            ]
        col_order += sorted(set(cal_cols))

        remaining = [c for c in res.columns if c not in set(col_order)]
        col_order += sorted(remaining)
        res = res[col_order]

        _log(self.config.verbose, "=" * 60)
        _log(self.config.verbose, f"Context build complete in {time.time()-t_start:.2f}s")
        _log(self.config.verbose, f"  Buckets:   {len(res)}")
        _log(self.config.verbose, f"  Columns:   {len(res.columns)} (incl bucket)")
        if len(res) > 0:
            _log(self.config.verbose, f"  Time:      {res['bucket_dt_utc'].min()} -> {res['bucket_dt_utc'].max()}")
        _log(self.config.verbose, "=" * 60)

        return res

    def build_and_save(self, output_path: str) -> pd.DataFrame:
        df = self.build()
        if df.empty:
            _log(self.config.verbose, "No data to save!")
            return df

        out_p = Path(output_path)
        _ensure_dir(out_p.parent)

        _log(self.config.verbose, f"Saving context to {output_path} ...")
        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(table, str(out_p), compression=PARQUET_COMPRESSION)

        mb = out_p.stat().st_size / (1024 * 1024)
        _log(self.config.verbose, f"Saved: {mb:.2f} MB")
        return df


# ==============================================================================
# Convention-based build (date + hour + asset)
# ==============================================================================

def build_context_hour(
    l0_dir: str,
    date_str: str,
    hour: int,
    asset: str = "btc",
    output_dir: Optional[str] = None,
    config: Optional[S0ContextConfig] = None,
) -> pd.DataFrame:
    """
    Build S0 context for one (asset, date, hour) using convention-based file naming.

    File naming convention:
      Input:  trades_{asset}_spot_{date}_{hour:02d}.parquet
      Output: s0_context_{asset}_{date}_{hour:02d}.parquet
    """
    l0_dir_p = Path(l0_dir)
    out_dir_p = Path(output_dir) if output_dir else (l0_dir_p.parent / "s0_context")
    _ensure_dir(out_dir_p)

    asset_lower = asset.lower()
    hour_str = f"{hour:02d}"
    suffix = f"{date_str}_{hour_str}.parquet"

    builder = S0ContextBuilder(
        trades_spot=str(l0_dir_p / f"trades_{asset_lower}_spot_{suffix}"),
        trades_fut=str(l0_dir_p / f"trades_{asset_lower}_fut_{suffix}"),
        lobdeep_spot=str(l0_dir_p / f"lobdeep_{asset_lower}_spot_{suffix}"),
        lobdeep_fut=str(l0_dir_p / f"lobdeep_{asset_lower}_fut_{suffix}"),
        config=config,
    )

    # Inject deterministic grid anchors into the config if not already set
    if builder.config.date_str is None or builder.config.hour is None:
        import dataclasses
        builder.config = dataclasses.replace(
            builder.config,
            date_str=date_str,
            hour=hour,
        )

    out_path = out_dir_p / f"s0_context_{asset_lower}_{suffix}"
    return builder.build_and_save(str(out_path))


# ==============================================================================
# CLI Entry Point
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Build S0 context (Health + Calendar + Usability) for one asset-hour."
    )

    # Explicit file paths (alternative to convention-based)
    parser.add_argument("--trades-spot", type=str)
    parser.add_argument("--trades-fut", type=str)
    parser.add_argument("--lobdeep-spot", type=str)
    parser.add_argument("--lobdeep-fut", type=str)

    # Convention-based mode
    parser.add_argument("--l0-dir", type=str, default="data/l0")
    parser.add_argument("--date", type=str)
    parser.add_argument("--hour", type=int)
    parser.add_argument("--asset", type=str, default="btc",
                        help="Asset to process: btc, eth, bnb (default: btc)")

    parser.add_argument("--output", "-o", type=str)
    parser.add_argument("--output-dir", type=str)

    parser.add_argument("--resample", type=str, default="1s")

    # Usability knobs
    parser.add_argument("--usability-window", type=int, default=DEFAULT_USABILITY_WINDOW_S)
    parser.add_argument("--usability-min-ratio", type=float, default=DEFAULT_USABILITY_MIN_RATIO)
    parser.add_argument("--usability-max-bad-streak", type=int, default=DEFAULT_USABILITY_MAX_BAD_STREAK)

    # Soft-health (B1a)
    parser.add_argument("--soft-missing-budget", type=int, default=DEFAULT_SOFT_MISSING_BUDGET)

    parser.add_argument("--no-calendar", action="store_true")
    parser.add_argument("--no-health", action="store_true")
    parser.add_argument("--quiet", "-q", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    config = S0ContextConfig(
        resample=args.resample,
        usability_window_s=int(args.usability_window),
        usability_min_ratio=float(args.usability_min_ratio),
        usability_max_bad_streak=int(args.usability_max_bad_streak),
        soft_missing_budget=int(args.soft_missing_budget),
        include_calendar=not args.no_calendar,
        include_health=not args.no_health,
        verbose=not args.quiet,
    )

    # Convention-based mode (--date + --hour + --asset)
    if args.date and args.hour is not None:
        if args.dry_run:
            out_dir = args.output_dir or str(Path(args.l0_dir).parent / "s0_context")
            print(
                f"Would build context: l0_dir={args.l0_dir} asset={args.asset} "
                f"date={args.date} hour={args.hour} -> {out_dir}"
            )
            return
        build_context_hour(
            args.l0_dir, args.date, args.hour,
            asset=args.asset,
            output_dir=args.output_dir, config=config,
        )
        return

    # Explicit file paths mode
    explicit_any = any([
        args.trades_spot, args.trades_fut,
        args.lobdeep_spot, args.lobdeep_fut,
    ])
    if explicit_any:
        if args.dry_run:
            print("Would build context from explicit files.")
            return

        builder = S0ContextBuilder(
            trades_spot=args.trades_spot,
            trades_fut=args.trades_fut,
            lobdeep_spot=args.lobdeep_spot,
            lobdeep_fut=args.lobdeep_fut,
            config=config,
        )

        if args.output:
            builder.build_and_save(args.output)
        else:
            df = builder.build()
            print(df.head(10).to_string(index=False))
        return

    parser.print_help()
    print("\nError: Provide either --date/--hour/--asset (convention-based) OR explicit file paths.")
    sys.exit(1)


if __name__ == "__main__":
    main()