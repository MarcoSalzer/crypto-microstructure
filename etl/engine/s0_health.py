# ==============================================================================
# S0 Health Features — Data Quality Flags per Bucket
#
# PURPOSE:
#   Compute per-bucket data quality flags for the Binance multi-asset pipeline.
#   Health flags indicate whether each bucket has complete, valid, gap-free,
#   non-crossed L2 orderbook data from all expected source streams.
#
# ARCHITECTURE CONTEXT:
#   Pipeline: Binance-only, multi-asset (BTC + ETH).
#   Data sources per asset:
#     - trades_{asset}_spot / trades_{asset}_fut   (Binance trade streams)
#     - lobdeep_{asset}_spot / lobdeep_{asset}_fut  (Binance deep L2, up to 1000 levels)
#
#   NOTE: The old pipeline also had lob20 (shallow 20-level book) from four
#   venues (Binance, OKX, Bybit, Coinbase). The current pipeline produces
#   only lobdeep from Binance. All lob20 references have been removed.
#
# GLOBAL HEALTH SEMANTICS:
#   data_health_flag == 1 ONLY IF *every* expected L2 source is healthy:
#     - lobdeep_spot healthy (has data, valid ts, no reconnect, no gap, no crossed)
#     - lobdeep_fut  healthy
#   Trades are bursty/event-based and NOT mandatory for global health.
#
# EXPECTED L2 COMBINATIONS (Binance-only):
#   Venue="Binance" x Market=["Spot","Futures"] x Source="lob_deep" = 2 combos
#   (Old pipeline had 4 venues x 2 markets x 2 sources = 16 combos)
#
# OUTPUT (aggregated mode, one row per bucket):
#   Core:
#     data_health_flag      int8   Strict L2 health across all combos
#     l2_coverage_flag      int8   Any L2 coverage (has data + depth >= 1)
#     depth_availability    int16  Min depth across all L2 sources
#     depth_lobdeep_global  int16  Min depth across lob_deep sources
#     lob50_health_flag     int8   1 iff depth_lobdeep_global >= 50
#     trades_coverage_flag  int8   1 iff any trades stream has data
#
#   Diagnostics (aggregated):
#     l2_total_combos       int16  L2 combos evaluated (2 in Binance-only)
#     l2_bad_combos         int16  Combos where data_health_flag == 0
#     l2_missing_combos     int16  Combos where has_data == 0
#     l2_invalid_ts_combos  int16  Combos with invalid timestamps (if has_data)
#     l2_reconnect_combos   int16  Combos with reconnect events (if has_data)
#     l2_gap_combos         int16  Combos with gaps (if has_data)
#     l2_crossed_combos     int16  Combos with crossed books (if has_data)
#     l2_bad_bitmask        int16  Bitmask summary per bucket
#     health_reason_code    int8   Prioritized reason for health failure
#
# health_reason_code:
#   0 = OK, 1 = missing_data, 2 = invalid_ts, 3 = reconnect,
#   4 = gap, 5 = crossed_book, 9 = unknown
#
# NOTE: data_usability_flag is computed in s0_context_batch.py (rolling logic).
#
# MINIMAL FIX SEMANTICS (preserved from original):
#   (1) After reindex: missing buckets only set has_data=0, other components
#       default to 1 (not falsely accused of invalid/reconnect/gap/crossed).
#   (2) Diagnostics: invalid/reconnect/gap/crossed counted only where has_data==1.
#
# ==============================================================================

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


# ==============================================================================
# Configuration / Thresholds
# ==============================================================================

# Maximum allowed gap between consecutive observed buckets, measured as a
# time-difference in milliseconds.  At 1 s resample:
#   gap_ms == 1000 → consecutive (no missing bucket)
#   gap_ms == 2000 → exactly 1 missing bucket in between  ← tolerated
#   gap_ms == 3000 → 2 missing buckets → flagged as gap
# The value 2000 therefore means: tolerate at most 1 missing bucket.
MAX_GAP_WIDTH_MS = 2000          # was MAX_GAP_BUCKETS = 2 (see above)
MIN_VALID_TS_MS = 946684800000   # 2000-01-01 UTC in ms

# Binance-only pipeline: single venue, two markets
DEFAULT_VENUES = ["Binance"]
DEFAULT_MARKETS = ["Spot", "Futures"]

# Depth gate for lobdeep health flag (>= 50 levels = healthy deep book)
LOBDEEP_GATE_K = 50


# ==============================================================================
# Source Paths
# ==============================================================================
# Per-asset source paths. The S0 context builder creates one SourcePaths per
# asset (BTC, ETH) pointing to the asset-specific Parquet files.
#
# NOTE: lob20 fields removed. The old pipeline had separate lob20 (shallow
# 20-level) and lobdeep (full depth) streams. The current Binance-only
# pipeline produces only lobdeep.

@dataclass
class SourcePaths:
    trades_spot: Optional[str] = None
    trades_fut: Optional[str] = None
    lobdeep_spot: Optional[str] = None
    lobdeep_fut: Optional[str] = None


# ==============================================================================
# Column normalization (support older parquet schemas)
# ==============================================================================

COLUMN_ALIASES = {
    "venue": "venue_scope",
    "exchange": "venue_scope",
    "market_type": "market_scope",
    "market": "market_scope",
    "instrument_type": "market_scope",
}

MARKET_VALUE_MAP = {
    "spot": "Spot",
    "Spot": "Spot",
    "SPOT": "Spot",
    "fut": "Futures",       # new pipeline uses "fut" as market_type
    "futures": "Futures",
    "Futures": "Futures",
    "FUTURES": "Futures",
    "perp": "Futures",
    "PERP": "Futures",
    "perpetual": "Futures",
    "PERPETUAL": "Futures",
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename compat column names to canonical names."""
    if df.empty:
        return df
    rename_map = {}
    for old, new in COLUMN_ALIASES.items():
        if old in df.columns and new not in df.columns:
            rename_map[old] = new
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def _normalize_market_scope_values(df: pd.DataFrame) -> pd.DataFrame:
    """Canonicalize market_scope values (e.g. 'fut' -> 'Futures')."""
    if df.empty or "market_scope" not in df.columns:
        return df
    df["market_scope"] = df["market_scope"].map(lambda x: MARKET_VALUE_MAP.get(str(x), str(x)))
    return df


# ==============================================================================
# Utility
# ==============================================================================

def _read_parquet_safe(path: Optional[str], columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Read parquet defensively (missing file, missing columns, schema drift)."""
    if not path:
        return pd.DataFrame()
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()

    try:
        if columns:
            schema = pq.read_schema(str(p))
            available = set(schema.names)

            cols_to_read: List[str] = []
            for c in columns:
                if c in available:
                    cols_to_read.append(c)
                    continue
                for alias, canonical in COLUMN_ALIASES.items():
                    if canonical == c and alias in available:
                        cols_to_read.append(alias)
                        break

            if not cols_to_read:
                return pd.DataFrame()

            tbl = pq.read_table(str(p), columns=cols_to_read)
        else:
            tbl = pq.read_table(str(p))

        df = tbl.to_pandas()
        df = _normalize_columns(df)
        df = _normalize_market_scope_values(df)
        return df
    except Exception as e:
        print(f"[HEALTH] Error reading {path}: {e}")
        return pd.DataFrame()


def _to_bucket_dt_utc(exch_ts_ms: pd.Series, resample: str) -> pd.Series:
    """Convert exch_ts_ms (epoch ms) to floored bucket timestamps."""
    dt = pd.to_datetime(exch_ts_ms.astype("int64"), unit="ms", utc=True)
    return dt.dt.floor(resample)


def _parse_resample_ms(resample: str) -> int:
    """Convert pandas-style resample string to milliseconds."""
    s = str(resample).strip().lower()
    if s.endswith("s"):
        return int(s[:-1]) * 1000
    if s.endswith("m") or s.endswith("min"):
        val = int(s.replace("min", "").replace("m", ""))
        return val * 60 * 1000
    if s.endswith("h"):
        return int(s[:-1]) * 3600 * 1000
    try:
        return int(s) * 1000
    except Exception:
        return 1000


def _empty_grouped(axis: pd.DatetimeIndex, venue_scope: str, market_scope: str, source_kind: str) -> pd.DataFrame:
    """
    Build an empty per-(source, venue, market) frame aligned to the global axis.

    Fill semantics for missing buckets:
      - has_data = 0 (no data)
      - has_valid_ts / no_reconnect / no_gap / no_crossed = 1
        (missing data is NOT counted as additional failure; captured by has_data=0)
      - depth_availability = NaN (no depth to report)
    """
    out = pd.DataFrame({"bucket_dt_utc": axis})
    out["venue_scope"] = venue_scope
    out["market_scope"] = market_scope
    out["source_kind"] = source_kind

    out["has_data"] = np.int8(0)
    out["has_valid_ts"] = np.int8(1)
    out["no_reconnect"] = np.int8(1)
    out["no_gap"] = np.int8(1)
    out["no_crossed"] = np.int8(1)

    out["data_health_flag"] = np.int8(0)
    out["l2_coverage_flag"] = np.int8(0)
    out["depth_availability"] = np.nan
    return out


# ==============================================================================
# Health Engine
# ==============================================================================

class HealthEngine:
    """
    Compute per-bucket health flags from raw Parquet source files.

    With Binance-only pipeline, the L2 combinations are:
      - lob_deep x Binance x Spot
      - lob_deep x Binance x Futures
    (Old pipeline had 4 venues x 2 markets x 2 sources = 16 combos)
    """

    def __init__(self, source_paths: SourcePaths):
        self.source_paths = source_paths
        self._cache: Dict[Tuple[str, str], pd.DataFrame] = {}

    def _path_for(self, source_kind: str, market_scope: str) -> Optional[str]:
        """Map (source_kind, market_scope) to a file path."""
        m = {
            ("trades", "Spot"): self.source_paths.trades_spot,
            ("trades", "Futures"): self.source_paths.trades_fut,
            # lob20 removed: no longer produced by the pipeline
            ("lob_deep", "Spot"): self.source_paths.lobdeep_spot,
            ("lob_deep", "Futures"): self.source_paths.lobdeep_fut,
        }
        return m.get((source_kind, market_scope))

    def _has_source(self, source_kind: str, market_scope: str) -> bool:
        p = self._path_for(source_kind, market_scope)
        return bool(p and Path(p).exists())

    def _require_lob_deep(self, markets: List[str]) -> None:
        """lob_deep is mandatory because deep features (e.g. K50) depend on it."""
        missing: List[str] = []
        for m in markets:
            p = self._path_for("lob_deep", m)
            if not p or not Path(p).exists():
                missing.append(f"lob_deep/{m} -> {p}")
        if missing:
            msg = (
                "[HEALTH] lob_deep is MANDATORY but missing for expected markets:\n"
                + "\n".join(f"  - {x}" for x in missing)
                + "\nFix: ensure pipeline writes lobdeep_{asset}_spot/fut parquets."
            )
            raise FileNotFoundError(msg)

    def _load_source(self, source_kind: str, market_scope: str, columns: List[str]) -> pd.DataFrame:
        key = (source_kind, market_scope)
        if key in self._cache:
            return self._cache[key].copy()
        p = self._path_for(source_kind, market_scope)
        df = _read_parquet_safe(p, columns)
        self._cache[key] = df
        return df.copy()

    def _compute_grouped(
        self,
        df: pd.DataFrame,
        source_kind: str,
        market_scope: str,
        venue_scope: str,
        resample: str,
        axis: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """
        Compute per-(source, venue, market) health signals in bucket space
        and reindex to the full axis.
        """
        if df.empty:
            return _empty_grouped(axis, venue_scope, market_scope, source_kind)

        if "venue_scope" in df.columns:
            df = df[df["venue_scope"] == venue_scope].copy()
        if "market_scope" in df.columns:
            df = df[df["market_scope"] == market_scope].copy()

        if df.empty or "exch_ts_ms" not in df.columns:
            return _empty_grouped(axis, venue_scope, market_scope, source_kind)

        df["bucket_dt_utc"] = _to_bucket_dt_utc(df["exch_ts_ms"], resample)

        agg_dict: Dict[str, object] = {"exch_ts_ms": ["min", "count"]}
        if "reconnect_flag" in df.columns:
            agg_dict["reconnect_flag"] = "max"

        # L2 source: only lob_deep in Binance-only pipeline (lob20 removed)
        if source_kind == "lob_deep":
            if "best_bid" in df.columns and "best_ask" in df.columns:
                df["_crossed"] = (df["best_bid"] >= df["best_ask"]).astype(int)
                agg_dict["_crossed"] = "max"
            if "depth_actual" in df.columns:
                agg_dict["depth_actual"] = "min"

        grouped = df.groupby("bucket_dt_utc").agg(agg_dict)
        grouped.columns = ["_".join(c).strip("_") for c in grouped.columns]
        grouped = grouped.reset_index()
        grouped = grouped.sort_values("bucket_dt_utc").reset_index(drop=True)

        # Component flags
        grouped["has_data"] = (grouped["exch_ts_ms_count"] > 0).astype("int8")
        grouped["has_valid_ts"] = (grouped["exch_ts_ms_min"] >= MIN_VALID_TS_MS).astype("int8")

        if "reconnect_flag_max" in grouped.columns:
            grouped["no_reconnect"] = (grouped["reconnect_flag_max"] == 0).astype("int8")
        else:
            grouped["no_reconnect"] = np.int8(1)

        resample_ms = _parse_resample_ms(resample)
        # MAX_GAP_WIDTH_MS is an absolute threshold (not a multiple of resample).
        # At 1s: 2000 ms → exactly 1 missing bucket tolerated.
        # At other resample rates, re-scale proportionally to keep "1 bucket" semantics.
        max_gap_ms = max(MAX_GAP_WIDTH_MS, 2 * resample_ms)
        # .values.asi8 returns nanoseconds as int64 (works on tz-aware series)
        bucket_ms = pd.Series(grouped["bucket_dt_utc"].array.asi8 // 10**6, index=grouped.index)
        gap_ms = bucket_ms.diff()
        grouped["no_gap"] = ((gap_ms.isna()) | (gap_ms <= max_gap_ms)).astype("int8")

        if "_crossed_max" in grouped.columns:
            grouped["no_crossed"] = (grouped["_crossed_max"] == 0).astype("int8")
        else:
            grouped["no_crossed"] = np.int8(1)

        # Composite source-level health: ALL components must be 1
        grouped["data_health_flag"] = (
            (grouped["has_data"] == 1)
            & (grouped["has_valid_ts"] == 1)
            & (grouped["no_reconnect"] == 1)
            & (grouped["no_gap"] == 1)
            & (grouped["no_crossed"] == 1)
        ).astype("int8")

        if source_kind == "lob_deep":
            depth = (
                grouped["depth_actual_min"]
                if "depth_actual_min" in grouped.columns
                else pd.Series([np.nan] * len(grouped))
            )
            grouped["l2_coverage_flag"] = ((grouped["has_data"] == 1) & (depth >= 1)).astype("int8")
            grouped["depth_availability"] = depth.astype("float64")
        else:
            grouped["l2_coverage_flag"] = np.int8(0)
            grouped["depth_availability"] = np.nan

        out = grouped[[
            "bucket_dt_utc", "data_health_flag", "l2_coverage_flag",
            "depth_availability", "has_data", "has_valid_ts",
            "no_reconnect", "no_gap", "no_crossed",
        ]].copy()
        out["venue_scope"] = venue_scope
        out["market_scope"] = market_scope
        out["source_kind"] = source_kind

        # Reindex to global axis: missing buckets become explicit
        out = out.set_index("bucket_dt_utc").reindex(axis)
        out.index.name = "bucket_dt_utc"
        out = out.reset_index()

        out["venue_scope"] = venue_scope
        out["market_scope"] = market_scope
        out["source_kind"] = source_kind

        # Fill semantics after reindex (missing buckets)
        out["data_health_flag"] = out["data_health_flag"].fillna(0).astype("int8")
        out["l2_coverage_flag"] = out["l2_coverage_flag"].fillna(0).astype("int8")
        out["has_data"] = out["has_data"].fillna(0).astype("int8")
        # Missing buckets: do NOT accuse invalid/reconnect/gap/crossed
        out["has_valid_ts"] = out["has_valid_ts"].fillna(1).astype("int8")
        out["no_reconnect"] = out["no_reconnect"].fillna(1).astype("int8")
        out["no_gap"] = out["no_gap"].fillna(1).astype("int8")
        out["no_crossed"] = out["no_crossed"].fillna(1).astype("int8")
        out["depth_availability"] = out["depth_availability"].astype("float64")

        return out

    def _build_bucket_axis(
        self,
        resample: str,
        venues: List[str],
        markets: List[str],
        include_sources: List[str],
    ) -> pd.DatetimeIndex:
        """Build master bucket axis from all available sources (including trades for grid stability)."""
        buckets: List[pd.Series] = []

        for source_kind in include_sources:
            for market in markets:
                if not self._has_source(source_kind, market):
                    continue

                cols = ["exch_ts_ms", "venue_scope", "market_scope"]
                if source_kind == "trades":
                    cols += ["reconnect_flag"]
                else:
                    cols += ["reconnect_flag", "best_bid", "best_ask", "depth_actual", "depth_target"]

                df = self._load_source(source_kind, market, cols)
                if df.empty or "exch_ts_ms" not in df.columns:
                    continue

                if "venue_scope" in df.columns:
                    df = df[df["venue_scope"].isin(venues)]
                if "market_scope" in df.columns:
                    df = df[df["market_scope"].isin(markets)]
                if df.empty:
                    continue

                b = _to_bucket_dt_utc(df["exch_ts_ms"], resample)
                buckets.append(b)

        if not buckets:
            return pd.DatetimeIndex([], tz="UTC")

        axis = pd.DatetimeIndex(pd.concat(buckets).dropna().unique())
        axis = axis.sort_values()
        if axis.tz is None:
            axis = axis.tz_localize("UTC")
        else:
            axis = axis.tz_convert("UTC")
        return axis

    def compute_all_health(
        self,
        resample: str = "1s",
        venues: Optional[List[str]] = None,
        markets: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Return unaggregated health per (source_kind, venue_scope, market_scope) x bucket."""
        venues = venues or DEFAULT_VENUES
        markets = markets or DEFAULT_MARKETS

        self._require_lob_deep(markets)

        # Only trades + lob_deep in Binance-only pipeline (lob20 removed)
        include_sources = ["trades", "lob_deep"]
        axis = self._build_bucket_axis(resample, venues, markets, include_sources)
        if len(axis) == 0:
            return pd.DataFrame()

        rows: List[pd.DataFrame] = []
        for source_kind in include_sources:
            for market_scope in markets:
                if source_kind == "trades" and not self._has_source(source_kind, market_scope):
                    continue

                cols = ["exch_ts_ms", "venue_scope", "market_scope", "reconnect_flag"]
                if source_kind != "trades":
                    cols += ["best_bid", "best_ask", "depth_actual", "depth_target"]

                df_src = self._load_source(source_kind, market_scope, cols)

                for venue_scope in venues:
                    g = self._compute_grouped(
                        df=df_src, source_kind=source_kind,
                        market_scope=market_scope, venue_scope=venue_scope,
                        resample=resample, axis=axis,
                    )
                    rows.append(g)

        if not rows:
            return pd.DataFrame()

        return pd.concat(rows, ignore_index=True)

    def compute_aggregated_health(
        self,
        resample: str = "1s",
        venues: Optional[List[str]] = None,
        markets: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Return aggregated global health (one row per bucket).
        Global data_health_flag is STRICT and L2-only (lob_deep).
        """
        venues = venues or DEFAULT_VENUES
        markets = markets or DEFAULT_MARKETS

        all_health = self.compute_all_health(resample=resample, venues=venues, markets=markets)
        if all_health.empty:
            return pd.DataFrame(columns=[
                "bucket_dt_utc", "data_health_flag", "l2_coverage_flag",
                "depth_availability", "depth_lobdeep_global", "lob50_health_flag",
                "trades_coverage_flag",
                "l2_total_combos", "l2_bad_combos", "l2_missing_combos",
                "l2_invalid_ts_combos", "l2_reconnect_combos",
                "l2_gap_combos", "l2_crossed_combos",
                "l2_bad_bitmask", "health_reason_code",
            ])

        # L2-only strict health (only lob_deep in this pipeline; lob20 removed)
        l2 = all_health[all_health["source_kind"] == "lob_deep"].copy()

        if l2.empty:
            agg = all_health.groupby("bucket_dt_utc").agg(
                data_health_flag=("data_health_flag", "min"),
                l2_coverage_flag=("l2_coverage_flag", "max"),
            ).reset_index()
            agg["depth_availability"] = 0
            agg["depth_lobdeep_global"] = 0
            for c in [
                "l2_total_combos", "l2_bad_combos", "l2_missing_combos",
                "l2_invalid_ts_combos", "l2_reconnect_combos",
                "l2_gap_combos", "l2_crossed_combos", "l2_bad_bitmask",
            ]:
                agg[c] = np.int16(0)
            agg["health_reason_code"] = np.int8(9)
        else:
            # --- Main strict health + depth aggregation ---
            agg = l2.groupby("bucket_dt_utc").agg(
                data_health_flag=("data_health_flag", "min"),
                l2_coverage_flag=("l2_coverage_flag", "max"),
                depth_availability=("depth_availability", "min"),
            ).reset_index()
            agg["depth_availability"] = agg["depth_availability"].fillna(0)

            # Depth gate: lobdeep only (lob20 split removed)
            ddeep = l2.groupby("bucket_dt_utc")["depth_availability"].min().reset_index()
            ddeep = ddeep.rename(columns={"depth_availability": "depth_lobdeep_global"})
            ddeep["depth_lobdeep_global"] = ddeep["depth_lobdeep_global"].fillna(0)
            agg = agg.merge(ddeep, on="bucket_dt_utc", how="left")
            agg["depth_lobdeep_global"] = agg["depth_lobdeep_global"].fillna(0)

            # --- Diagnostics aggregation ---
            diag = l2.groupby("bucket_dt_utc").agg(
                l2_total_combos=("data_health_flag", "size"),
                l2_bad_combos=("data_health_flag", lambda s: int((s == 0).sum())),
                l2_missing_combos=("has_data", lambda s: int((pd.to_numeric(s, errors="coerce").fillna(0).astype(int) == 0).sum())),
            ).reset_index()

            # Count failures only where has_data == 1
            l2c = l2.copy()
            l2c["_has"] = (l2c["has_data"].fillna(0).astype(int) == 1)

            def _count_fail_only_if_has(col: str) -> pd.Series:
                x = pd.to_numeric(l2c[col], errors="coerce").fillna(1).astype(int)
                fail = ((l2c["_has"]) & (x == 0)).astype(int)
                return fail.groupby(l2c["bucket_dt_utc"]).sum()

            diag = diag.set_index("bucket_dt_utc")
            diag["l2_invalid_ts_combos"] = _count_fail_only_if_has("has_valid_ts")
            diag["l2_reconnect_combos"] = _count_fail_only_if_has("no_reconnect")
            diag["l2_gap_combos"] = _count_fail_only_if_has("no_gap")
            diag["l2_crossed_combos"] = _count_fail_only_if_has("no_crossed")
            diag = diag.reset_index()

            # Bitmask: 1=missing, 2=invalid_ts, 4=reconnect, 8=gap, 16=crossed
            def _bitmask_row(r) -> int:
                m = 0
                if int(r.get("l2_missing_combos", 0)) > 0: m |= 1
                if int(r.get("l2_invalid_ts_combos", 0)) > 0: m |= 2
                if int(r.get("l2_reconnect_combos", 0)) > 0: m |= 4
                if int(r.get("l2_gap_combos", 0)) > 0: m |= 8
                if int(r.get("l2_crossed_combos", 0)) > 0: m |= 16
                return m

            diag["l2_bad_bitmask"] = diag.apply(_bitmask_row, axis=1).astype("int16")

            # Prioritized reason code
            def _reason_row(r) -> int:
                if int(r.get("l2_bad_combos", 0)) == 0: return 0
                if int(r.get("l2_missing_combos", 0)) > 0: return 1
                if int(r.get("l2_invalid_ts_combos", 0)) > 0: return 2
                if int(r.get("l2_reconnect_combos", 0)) > 0: return 3
                if int(r.get("l2_gap_combos", 0)) > 0: return 4
                if int(r.get("l2_crossed_combos", 0)) > 0: return 5
                return 9

            diag["health_reason_code"] = diag.apply(_reason_row, axis=1).astype("int8")
            agg = agg.merge(diag, on="bucket_dt_utc", how="left")

        # Trades coverage (presence-only; NOT part of global health)
        tr = all_health[all_health["source_kind"] == "trades"].copy()
        if tr.empty:
            agg["trades_coverage_flag"] = np.int8(0)
        else:
            tr_cov = tr.groupby("bucket_dt_utc").agg(trades_coverage_flag=("has_data", "max")).reset_index()
            agg = agg.merge(tr_cov, on="bucket_dt_utc", how="left")
            agg["trades_coverage_flag"] = agg["trades_coverage_flag"].fillna(0).astype("int8")

        # Depth gate: lobdeep >= 50 levels (lob20 gate removed)
        agg["lob50_health_flag"] = (agg["depth_lobdeep_global"].fillna(0) >= LOBDEEP_GATE_K).astype("int8")

        # --- Final dtype enforcement ---
        agg["data_health_flag"] = agg["data_health_flag"].fillna(0).astype("int8")
        agg["l2_coverage_flag"] = agg["l2_coverage_flag"].fillna(0).astype("int8")
        agg["depth_availability"] = agg["depth_availability"].fillna(0).astype("int16")
        agg["depth_lobdeep_global"] = agg["depth_lobdeep_global"].fillna(0).astype("int16")
        agg["lob50_health_flag"] = agg["lob50_health_flag"].astype("int8")
        if isinstance(agg.get("trades_coverage_flag"), pd.Series):
            agg["trades_coverage_flag"] = agg["trades_coverage_flag"].fillna(0).astype("int8")

        for c in [
            "l2_total_combos", "l2_bad_combos", "l2_missing_combos",
            "l2_invalid_ts_combos", "l2_reconnect_combos",
            "l2_gap_combos", "l2_crossed_combos", "l2_bad_bitmask",
        ]:
            if c not in agg.columns:
                agg[c] = 0
            agg[c] = pd.to_numeric(agg[c], errors="coerce").fillna(0).astype("int16")

        if "health_reason_code" not in agg.columns:
            agg["health_reason_code"] = np.int8(9)
        else:
            agg["health_reason_code"] = pd.to_numeric(agg["health_reason_code"], errors="coerce").fillna(9).astype("int8")

        return agg


# ==============================================================================
# Convenience
# ==============================================================================

def compute_health_features(
    source_paths: SourcePaths,
    resample: str = "1s",
    aggregated: bool = True,
) -> pd.DataFrame:
    """High-level entry point: compute health features from source paths."""
    engine = HealthEngine(source_paths)
    if aggregated:
        return engine.compute_aggregated_health(resample=resample)
    return engine.compute_all_health(resample=resample)


def get_health_feature_names() -> List[str]:
    """Ordered list of health feature columns (depth_lob20_global / lob20_health_flag removed)."""
    return [
        "data_health_flag",
        "l2_coverage_flag",
        "depth_availability",
        "depth_lobdeep_global",
        "lob50_health_flag",
        "trades_coverage_flag",
        # diagnostics
        "l2_total_combos", "l2_bad_combos", "l2_missing_combos",
        "l2_invalid_ts_combos", "l2_reconnect_combos",
        "l2_gap_combos", "l2_crossed_combos",
        "l2_bad_bitmask", "health_reason_code",
        # computed later in s0_context_batch.py
        "data_usability_flag",
    ]


def get_health_feature_dtypes() -> dict:
    return {
        "data_health_flag": "int8",
        "l2_coverage_flag": "int8",
        "depth_availability": "int16",
        "depth_lobdeep_global": "int16",
        "lob50_health_flag": "int8",
        "trades_coverage_flag": "int8",
        "l2_total_combos": "int16", "l2_bad_combos": "int16",
        "l2_missing_combos": "int16", "l2_invalid_ts_combos": "int16",
        "l2_reconnect_combos": "int16", "l2_gap_combos": "int16",
        "l2_crossed_combos": "int16", "l2_bad_bitmask": "int16",
        "health_reason_code": "int8",
        "data_usability_flag": "int8",
    }