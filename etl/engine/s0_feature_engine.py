# ==============================================================================
# S0 Feature Engine — Binance-only, Multi-Asset (BTC + ETH + BNB)
#
# PURPOSE:
#   Compute S0 market features from raw L0 parquet streams (trades + lobdeep),
#   left-join them onto the s0_context bucket grid, and write a wide parquet.
#
# CONTRACT (CLEAN S0):
#   - FeatureSpec.params only contain keys that the S0 operators actually use.
#   - No venue_scope / window_s / depth_mode / compat aliases.
#   - Operators must match s0_operators.py exactly (no alias fallback).
#
# FILL CONTRACT:
#   - Trades features: missing bucket -> 0.0
#   - Notional depth features: missing bucket -> 0.0
#     BUT: NaN if source arrays empty/None or mid invalid (corrupt data).
#   - Price, BPS metadata, imbalance, LWP, max_liq_distance: missing bucket -> NaN
#   - Imbalance: NaN if BOTH sides are 0.0, or either side NaN.
#
# POST-BUILD ARCHIVE:
#   After successful feature computation the engine moves consumed source
#   files (raw data + s0_context) into a date-partitioned archive directory:
#       data_storage/data_archive/{date_str}/raw_data/      ← trades_, lobdeep_
#       data_storage/data_archive/{date_str}/context_data/  ← s0_context_
#   This keeps raw_data/ and s0_context/ clean for the next hour's pipeline.
# ==============================================================================

from __future__ import annotations
from common.paths import DATA_ROOT

import argparse
import math
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from etl.operators.s0_operators import S0_OPERATORS

from etl.spec import FeatureSpec
from etl.spec.s0.s0_price import S0_PRICE_FEATURES
from etl.spec.s0.s0_activity import S0_ACTIVITY_FEATURES
from etl.spec.s0.s0_aggression import S0_AGGRESSION_FEATURES
from etl.spec.s0.s0_bookshape import S0_BOOKSHAPE_FEATURES
from etl.spec.s0.s0_imbalance import S0_IMBALANCE_FEATURES

PARQUET_COMPRESSION = "zstd"

_ENGINE_DIR = Path(__file__).resolve().parent
_DEFAULT_RAW_DIR = DATA_ROOT / "raw_data"
_DEFAULT_CTX_DIR = DATA_ROOT / "s0_context"
_DEFAULT_OUT_DIR = DATA_ROOT / "s0_features"
_DEFAULT_ARCHIVE_DIR = DATA_ROOT / "data_archive"


# =============================================================================
# Utilities
# =============================================================================

def _log(enabled: bool, msg: str) -> None:
    if enabled:
        print(f"[{pd.Timestamp.utcnow().isoformat()}] [S0_FEATURE_ENGINE] {msg}")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _require_cols(df: pd.DataFrame, cols: Iterable[str], ctx: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{ctx}: missing required columns: {missing}. Have: {list(df.columns)}")


def _parse_resample(resample: str) -> str:
    s = str(resample).strip()
    return s or "1s"


def _to_bucket_dt_utc_ms(exch_ts_ms: pd.Series, resample: str) -> pd.Series:
    dt = pd.to_datetime(exch_ts_ms.astype("int64"), unit="ms", utc=True)
    return dt.dt.floor(_parse_resample(resample))


def _read_parquet(path: str, columns: Optional[Iterable[str]] = None) -> pd.DataFrame:
    if columns:
        schema = pq.read_schema(path)
        available = set(schema.names)
        cols_to_read = [c for c in columns if c in available]
        tbl = pq.read_table(path, columns=cols_to_read if cols_to_read else None)
    else:
        tbl = pq.read_table(path)
    return tbl.to_pandas()


def _normalize_market_scope(ms: Any) -> str:
    s = str(ms).strip().lower()
    if s == "spot":
        return "Spot"
    if s in ("fut", "futures", "future", "perp", "perps", "perpetual"):
        return "Futures"
    if str(ms).strip() in ("Spot", "Futures"):
        return str(ms).strip()
    raise ValueError(f"Unsupported market_scope: {ms!r}")


# =============================================================================
# BPS Depth Helpers
# =============================================================================

def _safe_array(val: Any) -> np.ndarray:
    if val is None:
        return np.array([], dtype=np.float64)
    try:
        a = np.asarray(val, dtype=np.float64)
        return a if a.ndim == 1 else np.array([], dtype=np.float64)
    except Exception:
        return np.array([], dtype=np.float64)


def _compute_mid_vec(best_bid: pd.Series, best_ask: pd.Series) -> pd.Series:
    bb = pd.to_numeric(best_bid, errors="coerce")
    ba = pd.to_numeric(best_ask, errors="coerce")
    valid = (bb > 0) & (ba > 0) & np.isfinite(bb) & np.isfinite(ba)
    mid = (bb + ba) * 0.5
    return mid.where(valid, np.nan).astype("float64")


def _max_bps_side_from_px(px_arr: np.ndarray, mid: float, side: str) -> float:
    if len(px_arr) == 0 or not math.isfinite(mid) or mid <= 0:
        return float("nan")
    outermost = float(px_arr[-1])
    if not math.isfinite(outermost) or outermost <= 0:
        return float("nan")
    if side == "bid":
        return (mid - outermost) / mid * 10_000
    return (outermost - mid) / mid * 10_000


def _notional_within_bps(
    px_arr: np.ndarray,
    qty_arr: np.ndarray,
    mid: float,
    side: str,
    bps_lo: float,
    bps_hi: float,
) -> float:
    """
    Sum notional (px * qty) for levels within [bps_lo, bps_hi] bps from mid.

    Returns:
        NaN  — if arrays are empty/None/mismatched or mid is invalid.
        0.0  — if arrays are valid but no levels fall within the BPS window.
        >0   — actual notional sum.
    """
    n = min(len(px_arr), len(qty_arr))
    if n == 0:
        return float("nan")                     # no book data → NaN
    if not math.isfinite(mid) or mid <= 0:
        return float("nan")                     # bad mid → NaN

    px = px_arr[:n]
    qty = qty_arr[:n]

    if side == "bid":
        bps_dist = (mid - px) / mid * 10_000
    else:
        bps_dist = (px - mid) / mid * 10_000

    mask = (bps_dist >= bps_lo) & (bps_dist <= bps_hi)
    if not mask.any():
        return 0.0                               # book exists, window empty → 0

    return float(np.nansum(px[mask] * qty[mask]))


def _imbalance(bid_val: float, ask_val: float, eps: float = 1e-12) -> float:
    """
    Compute notional imbalance: (bid - ask) / (bid + ask + eps).

    Returns:
        NaN  — if either input is NaN/None, or both sides are exactly 0.0.
        +1/-1 — if only one side is 0.0 (valid asymmetry).
        float — normal imbalance in (-1, +1).
    """
    # NaN propagation: if either side is NaN, result is NaN
    if bid_val is None or (isinstance(bid_val, float) and math.isnan(bid_val)):
        return float("nan")
    if ask_val is None or (isinstance(ask_val, float) and math.isnan(ask_val)):
        return float("nan")

    b = float(bid_val)
    a = float(ask_val)
    denom = b + a

    # Both sides zero → no information → NaN (not 0/0)
    if denom <= 0:
        return float("nan")

    return (b - a) / (denom + eps)


def _lwp_within_bps(
    px_arr: np.ndarray,
    qty_arr: np.ndarray,
    mid: float,
    side: str,
    bps_lo: float,
    bps_hi: float,
) -> float:
    """
    Liquidity-weighted price within a BPS window.
    LWP = sum(px * qty) / sum(qty) for levels where bps_dist in [bps_lo, bps_hi].

    Returns:
        NaN  — if arrays empty/None/mismatched, mid invalid,
               no levels in window, or sum(qty) == 0.
        float — the liquidity-weighted average price.
    """
    n = min(len(px_arr), len(qty_arr))
    if n == 0:
        return float("nan")
    if not math.isfinite(mid) or mid <= 0:
        return float("nan")

    px = px_arr[:n]
    qty = qty_arr[:n]

    if side == "bid":
        bps_dist = (mid - px) / mid * 10_000
    else:
        bps_dist = (px - mid) / mid * 10_000

    mask = (bps_dist >= bps_lo) & (bps_dist <= bps_hi)
    if not mask.any():
        return float("nan")

    px_in = px[mask]
    qty_in = qty[mask]
    total_qty = float(np.nansum(qty_in))

    if total_qty <= 0:
        return float("nan")

    return float(np.nansum(px_in * qty_in)) / total_qty


def _max_liq_distance_within_bps(
    px_arr: np.ndarray,
    qty_arr: np.ndarray,
    mid: float,
    side: str,
    bps_lo: float,
    bps_hi: float,
) -> float:
    """
    BPS distance from mid to the level with maximum notional (px * qty)
    within a fixed BPS window [bps_lo, bps_hi] on the given side.

    Returns:
        NaN  — if arrays are empty/None/mismatched, mid is invalid (<=0 or not
               finite), no levels fall within the window, or all notional/dist
               values are non-finite.
        float >= 0  — BPS distance of the max-notional level from mid.
    """
    n = min(len(px_arr), len(qty_arr))
    if n == 0:
        return float("nan")
    if not math.isfinite(mid) or mid <= 0:
        return float("nan")

    px = px_arr[:n]
    qty = qty_arr[:n]

    if side == "bid":
        bps_dist = (mid - px) / mid * 10_000
    else:
        bps_dist = (px - mid) / mid * 10_000

    mask = (bps_dist >= bps_lo) & (bps_dist <= bps_hi)
    if not mask.any():
        return float("nan")

    notional = px[mask] * qty[mask]
    dist_in = bps_dist[mask]

    # Guard: filter out any non-finite notional values
    finite_mask = np.isfinite(notional) & np.isfinite(dist_in)
    if not finite_mask.any():
        return float("nan")

    best_idx = int(np.argmax(notional[finite_mask]))
    return float(dist_in[finite_mask][best_idx])


# =============================================================================
# Source Paths
# =============================================================================

@dataclass(frozen=True)
class SourcePaths:
    trades_spot: str
    trades_fut: str
    lobdeep_spot: str
    lobdeep_fut: str

    def all_paths(self) -> List[str]:
        """Return all four source paths as a list (for archive operations)."""
        return [self.trades_spot, self.trades_fut, self.lobdeep_spot, self.lobdeep_fut]


def _path_for_source_kind(source_kind: str, market_scope: str, paths: SourcePaths) -> str:
    ms = _normalize_market_scope(market_scope)
    if source_kind == "source:trades":
        return paths.trades_fut if ms == "Futures" else paths.trades_spot
    if source_kind == "source:lobdeep":
        return paths.lobdeep_fut if ms == "Futures" else paths.lobdeep_spot
    raise ValueError(f"Unsupported source_kind: {source_kind}")


def _infer_source_kind_from_dep(spec: Any) -> Optional[str]:
    """
    Extract the source kind (e.g. "source:trades") from a FeatureSpec's
    depends_on list.  In S0 every Dep stores the source selector in its
    `name` attribute (e.g. Dep("source:lobdeep", ...)).
    """
    deps = getattr(spec, "depends_on", None)
    if not deps:
        return None
    for d in deps:
        name = getattr(d, "name", None)
        if isinstance(name, str) and name.startswith("source:"):
            return name
    return None


# =============================================================================
# Engine
# =============================================================================

class S0FeatureEngine:
    """
    Compute S0 features from raw L0 parquet streams and align to context grid.
    """

    def __init__(self, paths: SourcePaths, verbose: bool = True):
        self.paths = paths
        self.verbose = verbose
        self._raw_cache: Dict[Tuple[str, str], pd.DataFrame] = {}
        self._snap_cache: Dict[Tuple[str, str], pd.DataFrame] = {}

    def _load_source(self, source_kind: str, market_scope: str) -> pd.DataFrame:
        ms = _normalize_market_scope(market_scope)
        key = (source_kind, ms)
        if key in self._raw_cache:
            return self._raw_cache[key]

        path = _path_for_source_kind(source_kind, ms, self.paths)

        if source_kind == "source:trades":
            cols = ["exch_ts_ms", "side", "qty", "price"]
        elif source_kind == "source:lobdeep":
            cols = ["exch_ts_ms", "best_bid", "best_ask", "bids_px", "bids_qty", "asks_px", "asks_qty"]
        else:
            raise ValueError(f"Unsupported source_kind: {source_kind}")

        df = _read_parquet(path, columns=cols)
        _require_cols(df, ["exch_ts_ms"], f"load({source_kind},{ms})")
        self._raw_cache[key] = df
        return df

    def _get_lob_snapshots(self, market_scope: str, resample: str) -> pd.DataFrame:
        ms = _normalize_market_scope(market_scope)
        rr = str(resample)
        cache_key = (ms, rr)
        if cache_key in self._snap_cache:
            return self._snap_cache[cache_key]

        df = self._load_source("source:lobdeep", ms)
        if df.empty:
            self._snap_cache[cache_key] = df
            return df

        df = df.copy()
        df["bucket_dt_utc"] = _to_bucket_dt_utc_ms(df["exch_ts_ms"], rr)
        df = df.sort_values("exch_ts_ms", kind="mergesort")
        snap = df.drop_duplicates(subset=["bucket_dt_utc"], keep="last").copy()

        _require_cols(snap, ["best_bid", "best_ask"], f"lob_snapshots({ms})")
        snap["_mid"] = _compute_mid_vec(snap["best_bid"], snap["best_ask"])

        bids_px_list = snap["bids_px"].tolist()
        asks_px_list = snap["asks_px"].tolist()

        max_bps_bid = np.full(len(snap), np.nan, dtype=np.float64)
        max_bps_ask = np.full(len(snap), np.nan, dtype=np.float64)

        mids = snap["_mid"].to_numpy(dtype=np.float64, copy=False)
        for i in range(len(snap)):
            mid = float(mids[i])
            bpx = _safe_array(bids_px_list[i])
            apx = _safe_array(asks_px_list[i])
            max_bps_bid[i] = _max_bps_side_from_px(bpx, mid, "bid")
            max_bps_ask[i] = _max_bps_side_from_px(apx, mid, "ask")

        snap["_max_bps_bid"] = max_bps_bid
        snap["_max_bps_ask"] = max_bps_ask
        snap["_bps_sym"] = np.minimum(max_bps_bid, max_bps_ask)

        self._snap_cache[cache_key] = snap
        return snap

    def _pick_source_kind(self, spec: Any, op: Any) -> str:
        if len(op.input_kinds) == 1:
            return op.input_kinds[0]
        dep_kind = _infer_source_kind_from_dep(spec)
        if dep_kind is not None:
            return dep_kind
        return op.input_kinds[0]

    def _reindex_to_context_axis(
        self,
        out: pd.DataFrame,
        context_buckets: pd.DatetimeIndex,
        feature_col: str,
        fill_value: Optional[float],
    ) -> pd.DataFrame:
        if out.empty:
            base = pd.DataFrame({"bucket_dt_utc": context_buckets, feature_col: pd.Series([pd.NA] * len(context_buckets))})
            if fill_value is not None:
                base[feature_col] = pd.to_numeric(base[feature_col], errors="coerce").fillna(float(fill_value))
            return base

        _require_cols(out, ["bucket_dt_utc", feature_col], f"reindex({feature_col})")
        tmp = out.set_index("bucket_dt_utc").reindex(context_buckets)
        if fill_value is not None:
            tmp[feature_col] = pd.to_numeric(tmp[feature_col], errors="coerce").fillna(float(fill_value))
        return tmp.reset_index()

    # =========================================================================
    # Compute one feature
    # =========================================================================

    def compute_one_feature_on_context(self, spec: FeatureSpec, context_buckets: pd.DatetimeIndex) -> pd.DataFrame:
        op = S0_OPERATORS.get(spec.operator)
        if op is None:
            raise ValueError(f"Unknown operator: {spec.operator}")

        for rp in op.required_params:
            if rp not in spec.params:
                raise ValueError(f"{spec.name}: missing required param '{rp}' for operator '{op.name}'")

        market_scope = _normalize_market_scope(spec.params["market_scope"])
        resample = str(spec.params.get("resample", op.optional_params_defaults.get("resample", "1s")))

        # TRADES
        if op.name.startswith("trades."):
            source_kind = self._pick_source_kind(spec, op)
            df = self._load_source(source_kind, market_scope)
            if df.empty:
                return self._reindex_to_context_axis(
                    pd.DataFrame({"bucket_dt_utc": [], spec.name: []}),
                    context_buckets,
                    spec.name,
                    fill_value=0.0,
                )
            df = df.copy()
            df["bucket_dt_utc"] = _to_bucket_dt_utc_ms(df["exch_ts_ms"], resample)
            return self._compute_trades_op(op.name, spec, df, context_buckets)

        # L2 TOP
        if op.name in ("l2.best_bid", "l2.best_ask", "l2.mid", "l2.spread"):
            snap = self._get_lob_snapshots(market_scope, resample)
            if snap.empty:
                return self._reindex_to_context_axis(
                    pd.DataFrame({"bucket_dt_utc": [], spec.name: []}),
                    context_buckets,
                    spec.name,
                    fill_value=None,
                )
            return self._compute_l2_price_op(op.name, spec, snap, context_buckets)

        # DEPTH BPS
        if op.name.startswith("depth_bps."):
            snap = self._get_lob_snapshots(market_scope, resample)
            if snap.empty:
                fill = 0.0 if op.name in ("depth_bps.notional_fixed_bps", "depth_bps.notional_struct_alpha") else None
                return self._reindex_to_context_axis(
                    pd.DataFrame({"bucket_dt_utc": [], spec.name: []}),
                    context_buckets,
                    spec.name,
                    fill_value=fill,
                )
            return self._compute_depth_bps_op(op.name, spec, snap, context_buckets)

        raise ValueError(f"{spec.name}: operator '{op.name}' not implemented in S0FeatureEngine")

    # =========================================================================
    # Trades ops
    # =========================================================================

    def _compute_trades_op(self, op_name: str, spec: FeatureSpec, df: pd.DataFrame, ctx: pd.DatetimeIndex) -> pd.DataFrame:
        name = spec.name

        if op_name == "trades.trade_count":
            out = df.groupby("bucket_dt_utc", as_index=False).size().rename(columns={"size": name})
            return self._reindex_to_context_axis(out, ctx, name, fill_value=0.0)

        if op_name == "trades.volume":
            _require_cols(df, ["qty"], name)
            out = df.groupby("bucket_dt_utc", as_index=False)["qty"].sum().rename(columns={"qty": name})
            return self._reindex_to_context_axis(out, ctx, name, fill_value=0.0)

        if op_name == "trades.notional":
            _require_cols(df, ["qty", "price"], name)
            df2 = df.copy()
            df2["_notional"] = (
                pd.to_numeric(df2["qty"], errors="coerce").fillna(0.0)
                * pd.to_numeric(df2["price"], errors="coerce").fillna(0.0)
            )
            out = df2.groupby("bucket_dt_utc", as_index=False)["_notional"].sum().rename(columns={"_notional": name})
            return self._reindex_to_context_axis(out, ctx, name, fill_value=0.0)

        if op_name == "trades.taker_buy_volume":
            _require_cols(df, ["qty", "side"], name)
            b = df[df["side"] == "buy"]
            out = (
                b.groupby("bucket_dt_utc", as_index=False)["qty"].sum().rename(columns={"qty": name})
                if not b.empty
                else pd.DataFrame({"bucket_dt_utc": [], name: []})
            )
            return self._reindex_to_context_axis(out, ctx, name, fill_value=0.0)

        if op_name == "trades.taker_sell_volume":
            _require_cols(df, ["qty", "side"], name)
            s = df[df["side"] == "sell"]
            out = (
                s.groupby("bucket_dt_utc", as_index=False)["qty"].sum().rename(columns={"qty": name})
                if not s.empty
                else pd.DataFrame({"bucket_dt_utc": [], name: []})
            )
            return self._reindex_to_context_axis(out, ctx, name, fill_value=0.0)

        if op_name == "trades.signed_volume":
            _require_cols(df, ["qty", "side"], name)
            df2 = df[df["side"].isin(["buy", "sell"])].copy()
            if df2.empty:
                out = pd.DataFrame({"bucket_dt_utc": [], name: []})
            else:
                sign = df2["side"].map({"buy": 1.0, "sell": -1.0}).astype("float64")
                df2["_sv"] = pd.to_numeric(df2["qty"], errors="coerce").fillna(0.0) * sign
                out = df2.groupby("bucket_dt_utc", as_index=False)["_sv"].sum().rename(columns={"_sv": name})
            return self._reindex_to_context_axis(out, ctx, name, fill_value=0.0)

        if op_name == "trades.taker_buy_notional":
            _require_cols(df, ["qty", "price", "side"], name)
            b = df[df["side"] == "buy"].copy()
            if b.empty:
                out = pd.DataFrame({"bucket_dt_utc": [], name: []})
            else:
                b["_not"] = (
                    pd.to_numeric(b["qty"], errors="coerce").fillna(0.0)
                    * pd.to_numeric(b["price"], errors="coerce").fillna(0.0)
                )
                out = b.groupby("bucket_dt_utc", as_index=False)["_not"].sum().rename(columns={"_not": name})
            return self._reindex_to_context_axis(out, ctx, name, fill_value=0.0)

        if op_name == "trades.taker_sell_notional":
            _require_cols(df, ["qty", "price", "side"], name)
            s = df[df["side"] == "sell"].copy()
            if s.empty:
                out = pd.DataFrame({"bucket_dt_utc": [], name: []})
            else:
                s["_not"] = (
                    pd.to_numeric(s["qty"], errors="coerce").fillna(0.0)
                    * pd.to_numeric(s["price"], errors="coerce").fillna(0.0)
                )
                out = s.groupby("bucket_dt_utc", as_index=False)["_not"].sum().rename(columns={"_not": name})
            return self._reindex_to_context_axis(out, ctx, name, fill_value=0.0)

        if op_name == "trades.signed_notional":
            _require_cols(df, ["qty", "price", "side"], name)
            df2 = df[df["side"].isin(["buy", "sell"])].copy()
            if df2.empty:
                out = pd.DataFrame({"bucket_dt_utc": [], name: []})
            else:
                sign = df2["side"].map({"buy": 1.0, "sell": -1.0}).astype("float64")
                notional = (
                    pd.to_numeric(df2["qty"], errors="coerce").fillna(0.0)
                    * pd.to_numeric(df2["price"], errors="coerce").fillna(0.0)
                )
                df2["_sn"] = notional * sign
                out = df2.groupby("bucket_dt_utc", as_index=False)["_sn"].sum().rename(columns={"_sn": name})
            return self._reindex_to_context_axis(out, ctx, name, fill_value=0.0)

        raise ValueError(f"Unknown trades operator: {op_name}")

    # =========================================================================
    # L2 price ops
    # =========================================================================

    def _compute_l2_price_op(self, op_name: str, spec: FeatureSpec, snap: pd.DataFrame, ctx: pd.DatetimeIndex) -> pd.DataFrame:
        name = spec.name
        _require_cols(snap, ["best_bid", "best_ask"], name)

        if op_name == "l2.best_bid":
            out = snap[["bucket_dt_utc", "best_bid"]].rename(columns={"best_bid": name})
        elif op_name == "l2.best_ask":
            out = snap[["bucket_dt_utc", "best_ask"]].rename(columns={"best_ask": name})
        elif op_name == "l2.mid":
            _require_cols(snap, ["_mid"], name)
            out = snap[["bucket_dt_utc", "_mid"]].rename(columns={"_mid": name})
        else:  # l2.spread
            out = snap[["bucket_dt_utc"]].copy()
            out[name] = pd.to_numeric(snap["best_ask"], errors="coerce") - pd.to_numeric(snap["best_bid"], errors="coerce")

        return self._reindex_to_context_axis(out, ctx, name, fill_value=None)

    # =========================================================================
    # Depth BPS ops
    # =========================================================================

    def _compute_depth_bps_op(self, op_name: str, spec: FeatureSpec, snap: pd.DataFrame, ctx: pd.DatetimeIndex) -> pd.DataFrame:
        name = spec.name
        params = spec.params

        _require_cols(snap, ["_mid"], name)

        if op_name == "depth_bps.max_bps_side":
            side = str(params["side"]).strip().lower()
            if side not in ("bid", "ask"):
                raise ValueError(f"{name}: invalid side={side!r}")
            col = "_max_bps_bid" if side == "bid" else "_max_bps_ask"
            _require_cols(snap, [col], name)
            out = snap[["bucket_dt_utc", col]].rename(columns={col: name})
            return self._reindex_to_context_axis(out, ctx, name, fill_value=None)

        if op_name == "depth_bps.bps_sym":
            _require_cols(snap, ["_bps_sym"], name)
            out = snap[["bucket_dt_utc", "_bps_sym"]].rename(columns={"_bps_sym": name})
            return self._reindex_to_context_axis(out, ctx, name, fill_value=None)

        if op_name == "depth_bps.notional_fixed_bps":
            side = str(params["side"]).strip().lower()
            bps_lo = float(params["bps_lo"])
            bps_hi = float(params["bps_hi"])
            _require_cols(snap, ["bids_px", "bids_qty", "asks_px", "asks_qty"], name)

            if side == "bid":
                px_col, qty_col = "bids_px", "bids_qty"
            elif side == "ask":
                px_col, qty_col = "asks_px", "asks_qty"
            else:
                raise ValueError(f"{name}: invalid side={side!r}")

            out = snap[["bucket_dt_utc"]].copy()
            mids = snap["_mid"].to_numpy(dtype=np.float64, copy=False)
            px_list = snap[px_col].tolist()
            qty_list = snap[qty_col].tolist()

            vals = np.full(len(snap), np.nan, dtype=np.float64)
            for i in range(len(snap)):
                vals[i] = _notional_within_bps(
                    _safe_array(px_list[i]),
                    _safe_array(qty_list[i]),
                    float(mids[i]),
                    side,
                    bps_lo,
                    bps_hi,
                )
            out[name] = vals
            return self._reindex_to_context_axis(out, ctx, name, fill_value=0.0)

        if op_name == "depth_bps.imbalance_fixed_bps":
            bps_lo = float(params["bps_lo"])
            bps_hi = float(params["bps_hi"])
            _require_cols(snap, ["bids_px", "bids_qty", "asks_px", "asks_qty"], name)

            out = snap[["bucket_dt_utc"]].copy()
            mids = snap["_mid"].to_numpy(dtype=np.float64, copy=False)

            bpx_list = snap["bids_px"].tolist()
            bqy_list = snap["bids_qty"].tolist()
            apx_list = snap["asks_px"].tolist()
            aqy_list = snap["asks_qty"].tolist()

            vals = np.full(len(snap), np.nan, dtype=np.float64)
            for i in range(len(snap)):
                mid = float(mids[i])
                bid_not = _notional_within_bps(_safe_array(bpx_list[i]), _safe_array(bqy_list[i]), mid, "bid", bps_lo, bps_hi)
                ask_not = _notional_within_bps(_safe_array(apx_list[i]), _safe_array(aqy_list[i]), mid, "ask", bps_lo, bps_hi)
                vals[i] = _imbalance(bid_not, ask_not)
            out[name] = vals
            return self._reindex_to_context_axis(out, ctx, name, fill_value=None)

        if op_name == "depth_bps.notional_struct_alpha":
            side = str(params["side"]).strip().lower()
            alpha = float(params["alpha"])
            _require_cols(snap, ["_bps_sym", "bids_px", "bids_qty", "asks_px", "asks_qty"], name)

            if side == "bid":
                px_col, qty_col = "bids_px", "bids_qty"
            elif side == "ask":
                px_col, qty_col = "asks_px", "asks_qty"
            else:
                raise ValueError(f"{name}: invalid side={side!r}")

            out = snap[["bucket_dt_utc"]].copy()
            mids = snap["_mid"].to_numpy(dtype=np.float64, copy=False)
            bps_sym = snap["_bps_sym"].to_numpy(dtype=np.float64, copy=False)
            px_list = snap[px_col].tolist()
            qty_list = snap[qty_col].tolist()

            vals = np.full(len(snap), np.nan, dtype=np.float64)
            for i in range(len(snap)):
                mid = float(mids[i])
                bs = float(bps_sym[i])
                if not (math.isfinite(bs) and bs > 0):
                    continue                     # stays NaN (bps_sym invalid)
                bps_hi = alpha * bs
                vals[i] = _notional_within_bps(_safe_array(px_list[i]), _safe_array(qty_list[i]), mid, side, 0.0, bps_hi)
            out[name] = vals
            return self._reindex_to_context_axis(out, ctx, name, fill_value=0.0)

        if op_name == "depth_bps.imbalance_struct_alpha":
            alpha = float(params["alpha"])
            _require_cols(snap, ["_bps_sym", "bids_px", "bids_qty", "asks_px", "asks_qty"], name)

            out = snap[["bucket_dt_utc"]].copy()
            mids = snap["_mid"].to_numpy(dtype=np.float64, copy=False)
            bps_sym = snap["_bps_sym"].to_numpy(dtype=np.float64, copy=False)

            bpx_list = snap["bids_px"].tolist()
            bqy_list = snap["bids_qty"].tolist()
            apx_list = snap["asks_px"].tolist()
            aqy_list = snap["asks_qty"].tolist()

            vals = np.full(len(snap), np.nan, dtype=np.float64)
            for i in range(len(snap)):
                mid = float(mids[i])
                bs = float(bps_sym[i])
                if not (math.isfinite(bs) and bs > 0):
                    vals[i] = float("nan")
                    continue
                bps_hi = alpha * bs
                bid_not = _notional_within_bps(_safe_array(bpx_list[i]), _safe_array(bqy_list[i]), mid, "bid", 0.0, bps_hi)
                ask_not = _notional_within_bps(_safe_array(apx_list[i]), _safe_array(aqy_list[i]), mid, "ask", 0.0, bps_hi)
                vals[i] = _imbalance(bid_not, ask_not)

            out[name] = vals
            return self._reindex_to_context_axis(out, ctx, name, fill_value=None)

        # =====================================================================
        # LWP — FIXED BPS (one side)
        # =====================================================================
        if op_name == "depth_bps.lwp_fixed_bps":
            side = str(params["side"]).strip().lower()
            bps_lo = float(params["bps_lo"])
            bps_hi = float(params["bps_hi"])
            _require_cols(snap, ["bids_px", "bids_qty", "asks_px", "asks_qty"], name)

            if side == "bid":
                px_col, qty_col = "bids_px", "bids_qty"
            elif side == "ask":
                px_col, qty_col = "asks_px", "asks_qty"
            else:
                raise ValueError(f"{name}: invalid side={side!r}")

            out = snap[["bucket_dt_utc"]].copy()
            mids = snap["_mid"].to_numpy(dtype=np.float64, copy=False)
            px_list = snap[px_col].tolist()
            qty_list = snap[qty_col].tolist()

            vals = np.full(len(snap), np.nan, dtype=np.float64)
            for i in range(len(snap)):
                vals[i] = _lwp_within_bps(
                    _safe_array(px_list[i]),
                    _safe_array(qty_list[i]),
                    float(mids[i]),
                    side,
                    bps_lo,
                    bps_hi,
                )
            out[name] = vals
            return self._reindex_to_context_axis(out, ctx, name, fill_value=None)

        # =====================================================================
        # LWP MID — FIXED BPS
        # =====================================================================
        if op_name == "depth_bps.lwp_mid_fixed_bps":
            bps_lo = float(params["bps_lo"])
            bps_hi = float(params["bps_hi"])
            _require_cols(snap, ["bids_px", "bids_qty", "asks_px", "asks_qty"], name)

            out = snap[["bucket_dt_utc"]].copy()
            mids = snap["_mid"].to_numpy(dtype=np.float64, copy=False)

            bpx_list = snap["bids_px"].tolist()
            bqy_list = snap["bids_qty"].tolist()
            apx_list = snap["asks_px"].tolist()
            aqy_list = snap["asks_qty"].tolist()

            vals = np.full(len(snap), np.nan, dtype=np.float64)
            for i in range(len(snap)):
                mid = float(mids[i])
                lwp_bid = _lwp_within_bps(_safe_array(bpx_list[i]), _safe_array(bqy_list[i]), mid, "bid", bps_lo, bps_hi)
                lwp_ask = _lwp_within_bps(_safe_array(apx_list[i]), _safe_array(aqy_list[i]), mid, "ask", bps_lo, bps_hi)
                # NaN propagation: if either side NaN, mid is NaN
                if math.isfinite(lwp_bid) and math.isfinite(lwp_ask):
                    vals[i] = (lwp_bid + lwp_ask) * 0.5

            out[name] = vals
            return self._reindex_to_context_axis(out, ctx, name, fill_value=None)

        # =====================================================================
        # LWP — STRUCTURAL (one side)
        # =====================================================================
        if op_name == "depth_bps.lwp_struct_alpha":
            side = str(params["side"]).strip().lower()
            alpha = float(params["alpha"])
            _require_cols(snap, ["_bps_sym", "bids_px", "bids_qty", "asks_px", "asks_qty"], name)

            if side == "bid":
                px_col, qty_col = "bids_px", "bids_qty"
            elif side == "ask":
                px_col, qty_col = "asks_px", "asks_qty"
            else:
                raise ValueError(f"{name}: invalid side={side!r}")

            out = snap[["bucket_dt_utc"]].copy()
            mids = snap["_mid"].to_numpy(dtype=np.float64, copy=False)
            bps_sym = snap["_bps_sym"].to_numpy(dtype=np.float64, copy=False)
            px_list = snap[px_col].tolist()
            qty_list = snap[qty_col].tolist()

            vals = np.full(len(snap), np.nan, dtype=np.float64)
            for i in range(len(snap)):
                mid = float(mids[i])
                bs = float(bps_sym[i])
                if not (math.isfinite(bs) and bs > 0):
                    continue  # stays NaN
                bps_hi = alpha * bs
                vals[i] = _lwp_within_bps(_safe_array(px_list[i]), _safe_array(qty_list[i]), mid, side, 0.0, bps_hi)
            out[name] = vals
            return self._reindex_to_context_axis(out, ctx, name, fill_value=None)

        # =====================================================================
        # LWP MID — STRUCTURAL
        # =====================================================================
        if op_name == "depth_bps.lwp_mid_struct_alpha":
            alpha = float(params["alpha"])
            _require_cols(snap, ["_bps_sym", "bids_px", "bids_qty", "asks_px", "asks_qty"], name)

            out = snap[["bucket_dt_utc"]].copy()
            mids = snap["_mid"].to_numpy(dtype=np.float64, copy=False)
            bps_sym = snap["_bps_sym"].to_numpy(dtype=np.float64, copy=False)

            bpx_list = snap["bids_px"].tolist()
            bqy_list = snap["bids_qty"].tolist()
            apx_list = snap["asks_px"].tolist()
            aqy_list = snap["asks_qty"].tolist()

            vals = np.full(len(snap), np.nan, dtype=np.float64)
            for i in range(len(snap)):
                mid = float(mids[i])
                bs = float(bps_sym[i])
                if not (math.isfinite(bs) and bs > 0):
                    continue  # stays NaN
                bps_hi = alpha * bs
                lwp_bid = _lwp_within_bps(_safe_array(bpx_list[i]), _safe_array(bqy_list[i]), mid, "bid", 0.0, bps_hi)
                lwp_ask = _lwp_within_bps(_safe_array(apx_list[i]), _safe_array(aqy_list[i]), mid, "ask", 0.0, bps_hi)
                if math.isfinite(lwp_bid) and math.isfinite(lwp_ask):
                    vals[i] = (lwp_bid + lwp_ask) * 0.5

            out[name] = vals
            return self._reindex_to_context_axis(out, ctx, name, fill_value=None)

        # =====================================================================
        # MAX LIQUIDITY DISTANCE — FIXED BPS
        # =====================================================================
        if op_name == "depth_bps.max_liq_distance_fixed_bps":
            side = str(params["side"]).strip().lower()
            bps_lo = float(params["bps_lo"])
            bps_hi = float(params["bps_hi"])
            _require_cols(snap, ["bids_px", "bids_qty", "asks_px", "asks_qty", "_mid"], name)

            if side == "bid":
                px_col, qty_col = "bids_px", "bids_qty"
            elif side == "ask":
                px_col, qty_col = "asks_px", "asks_qty"
            else:
                raise ValueError(f"{name}: invalid side={side!r}")

            out = snap[["bucket_dt_utc"]].copy()
            mids = snap["_mid"].to_numpy(dtype=np.float64, copy=False)
            px_list = snap[px_col].tolist()
            qty_list = snap[qty_col].tolist()

            vals = np.full(len(snap), np.nan, dtype=np.float64)
            for i in range(len(snap)):
                vals[i] = _max_liq_distance_within_bps(
                    _safe_array(px_list[i]),
                    _safe_array(qty_list[i]),
                    float(mids[i]),
                    side,
                    bps_lo,
                    bps_hi,
                )
            out[name] = vals
            return self._reindex_to_context_axis(out, ctx, name, fill_value=None)

        # =====================================================================
        # MAX LIQUIDITY DISTANCE — STRUCTURAL (adaptive)
        # =====================================================================
        if op_name == "depth_bps.max_liq_distance_struct_alpha":
            side = str(params["side"]).strip().lower()
            alpha = float(params["alpha"])
            _require_cols(snap, ["_bps_sym", "bids_px", "bids_qty", "asks_px", "asks_qty", "_mid"], name)

            if side == "bid":
                px_col, qty_col = "bids_px", "bids_qty"
            elif side == "ask":
                px_col, qty_col = "asks_px", "asks_qty"
            else:
                raise ValueError(f"{name}: invalid side={side!r}")

            out = snap[["bucket_dt_utc"]].copy()
            mids = snap["_mid"].to_numpy(dtype=np.float64, copy=False)
            bps_sym = snap["_bps_sym"].to_numpy(dtype=np.float64, copy=False)
            px_list = snap[px_col].tolist()
            qty_list = snap[qty_col].tolist()

            vals = np.full(len(snap), np.nan, dtype=np.float64)
            for i in range(len(snap)):
                bs = float(bps_sym[i])
                if not (math.isfinite(bs) and bs > 0):
                    continue  # stays NaN
                bps_hi = alpha * bs
                vals[i] = _max_liq_distance_within_bps(
                    _safe_array(px_list[i]),
                    _safe_array(qty_list[i]),
                    float(mids[i]),
                    side,
                    0.0,
                    bps_hi,
                )
            out[name] = vals
            return self._reindex_to_context_axis(out, ctx, name, fill_value=None)

        raise ValueError(f"{name}: unknown depth_bps operator: {op_name}")
    # =========================================================================
    # Bulk compute
    # =========================================================================

    def compute_all_on_context(
        self,
        context_df: pd.DataFrame,
        specs: List[FeatureSpec],
        features_filter: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        _require_cols(context_df, ["bucket_dt_utc"], "context_df")

        ctx = context_df.copy()
        ctx = ctx.sort_values("bucket_dt_utc").reset_index(drop=True)
        ctx["bucket_dt_utc"] = pd.to_datetime(ctx["bucket_dt_utc"], utc=True)
        context_buckets = pd.DatetimeIndex(ctx["bucket_dt_utc"])

        if features_filter:
            wanted = set(features_filter)
            specs = [s for s in specs if s.name in wanted]

        _log(self.verbose, f"Computing S0 features: {len(specs)} specs")
        t0 = time.time()
        computed, errors = 0, 0

        for spec in specs:
            try:
                feat = self.compute_one_feature_on_context(spec, context_buckets=context_buckets)
                ctx = ctx.merge(feat, on="bucket_dt_utc", how="left")
                computed += 1
            except Exception as e:
                errors += 1
                if self.verbose:
                    print(f"[WARN] {spec.name}: {e}")

        _log(self.verbose, f"Done. computed={computed} errors={errors} in {time.time() - t0:.2f}s")
        return ctx


# =============================================================================
# Feature Registry
# =============================================================================

ALL_S0_FEATURES: List[FeatureSpec] = (
    list(S0_PRICE_FEATURES)
    + list(S0_ACTIVITY_FEATURES)
    + list(S0_AGGRESSION_FEATURES)
    + list(S0_BOOKSHAPE_FEATURES)
    + list(S0_IMBALANCE_FEATURES)
)

# ---------------------------------------------------------------------------
# Fail-fast uniqueness checks (catches copy-paste / renumber errors early)
# ---------------------------------------------------------------------------
def _assert_feature_registry_unique(features: List[FeatureSpec]) -> None:
    seen_names: dict = {}
    seen_ids: dict = {}
    collisions: List[str] = []

    for spec in features:
        if spec.name in seen_names:
            collisions.append(
                f"  DUPLICATE NAME : '{spec.name}' "
                f"(ids {seen_names[spec.name]} and {spec.feature_id})"
            )
        else:
            seen_names[spec.name] = spec.feature_id

        if spec.feature_id in seen_ids:
            collisions.append(
                f"  DUPLICATE ID   : feature_id={spec.feature_id} "
                f"→ '{seen_ids[spec.feature_id]}' and '{spec.name}'"
            )
        else:
            seen_ids[spec.feature_id] = spec.name

    if collisions:
        raise AssertionError(
            "S0 feature registry has collisions — fix before running the pipeline:\n"
            + "\n".join(collisions)
        )

_assert_feature_registry_unique(ALL_S0_FEATURES)


def _find_feature_by_name(features: Iterable[FeatureSpec], name: str) -> FeatureSpec:
    for f in features:
        if f.name == name:
            return f
    raise KeyError(f"Feature not found: {name}")


# =============================================================================
# I/O Helpers
# =============================================================================

def _paths_for_hour(
    raw_dir: str,
    ctx_dir: str,
    out_dir: str,
    asset: str,
    date_str: str,
    hour: int,
) -> Tuple[SourcePaths, Path, Path]:
    raw_base = Path(raw_dir)
    hh = f"{int(hour):02d}"
    suffix = f"{date_str}_{hh}.parquet"
    a = asset.lower()

    raw = SourcePaths(
        trades_spot=str(raw_base / f"trades_{a}_spot_{suffix}"),
        trades_fut=str(raw_base / f"trades_{a}_fut_{suffix}"),
        lobdeep_spot=str(raw_base / f"lobdeep_{a}_spot_{suffix}"),
        lobdeep_fut=str(raw_base / f"lobdeep_{a}_fut_{suffix}"),
    )

    ctx_path = Path(ctx_dir) / f"s0_context_{a}_{suffix}"
    out_path = Path(out_dir) / f"s0_features_{a}_{suffix}"
    return raw, ctx_path, out_path


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
    Move consumed source files into a date-partitioned archive folder.

    Target layout:
        data_archive/{date_str}/raw_data/trades_btc_spot_2026-02-16_03.parquet
        data_archive/{date_str}/raw_data/lobdeep_btc_fut_2026-02-16_03.parquet
        data_archive/{date_str}/context_data/s0_context_btc_2026-02-16_03.parquet

    Args:
        files_to_move: List of file paths to archive.
        archive_dir:   Base archive directory (e.g. data_storage/data_archive).
        date_str:      Date partition key (e.g. "2026-02-16").
        sub_dir:       Subdirectory within the date folder ("raw_data" or "context_data").
                       If empty, files go directly into {date_str}/.
        verbose:       Print progress logs.

    Files that don't exist (e.g. already archived) are silently skipped.
    Files that already exist in the archive are OVERWRITTEN — this supports
    re-runs after deleting downstream feature files. The newly-consumed
    source file (which produced the current feature output) is the one
    preserved.
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
            dest_path.unlink()
            _log(verbose, f"Archive overwrite: {dest_path.name}")

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

def build_s0_features_for_hour(
    raw_dir: str,
    ctx_dir: str,
    out_dir: str,
    asset: str,
    date_str: str,
    hour: int,
    features_filter: Optional[List[str]] = None,
    archive_dir: Optional[str] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Main entry point: compute S0 features for one asset-hour, write parquet,
    then optionally archive consumed raw data + context files.

    Args:
        raw_dir:          Directory containing raw L0 parquets.
        ctx_dir:          Directory containing s0_context parquets.
        out_dir:          Directory to write s0_features parquets.
        asset:            "btc", "eth", or "bnb".
        date_str:         Date string, e.g. "2026-02-16".
        hour:             Hour (0–23).
        features_filter:  Optional list of feature names to compute (None = all).
        archive_dir:      If set, move raw + context files here after success.
                          Files land in {archive_dir}/{date_str}/.
        verbose:          Print progress logs.

    Returns:
        The computed S0 feature DataFrame (already written to disk).
    """
    raw_paths, ctx_path, out_path = _paths_for_hour(raw_dir, ctx_dir, out_dir, asset, date_str, hour)

    if not ctx_path.exists():
        raise FileNotFoundError(f"Missing context file: {ctx_path}")

    _ensure_dir(out_path.parent)
    _log(verbose, f"Loading context: {ctx_path}")
    context_df = _read_parquet(str(ctx_path))

    engine = S0FeatureEngine(raw_paths, verbose=verbose)
    df = engine.compute_all_on_context(context_df, specs=ALL_S0_FEATURES, features_filter=features_filter)

    _log(verbose, f"Saving s0_features to: {out_path}")
    _atomic_write_parquet(df, out_path)

    mb = out_path.stat().st_size / (1024 * 1024)
    _log(verbose, f"Saved: {mb:.2f} MB | rows={len(df)} cols={len(df.columns)}")

    # -----------------------------------------------------------------
    # Archive consumed source files (raw data + context)
    # -----------------------------------------------------------------
    if archive_dir is not None:
        _archive_files(
            files_to_move=[Path(p) for p in raw_paths.all_paths()],
            archive_dir=Path(archive_dir),
            date_str=date_str,
            sub_dir="raw_data",
            verbose=verbose,
        )
        _archive_files(
            files_to_move=[ctx_path],
            archive_dir=Path(archive_dir),
            date_str=date_str,
            sub_dir="context_data",
            verbose=verbose,
        )

    return df


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="S0 feature engine: compute S0 features from L0 parquets onto s0_context grid."
    )
    ap.add_argument("--raw-dir", type=str, default=str(_DEFAULT_RAW_DIR))
    ap.add_argument("--ctx-dir", type=str, default=str(_DEFAULT_CTX_DIR))
    ap.add_argument("--out-dir", type=str, default=str(_DEFAULT_OUT_DIR))
    ap.add_argument("--archive-dir", type=str, default=str(_DEFAULT_ARCHIVE_DIR),
                     help="Archive directory for consumed raw + context files. "
                          "Files are moved into {archive-dir}/{date}/.")
    ap.add_argument("--no-archive", action="store_true",
                     help="Skip archiving (keep raw + context files in place).")
    ap.add_argument("--asset", type=str, required=True, choices=["btc", "eth", "bnb"])
    ap.add_argument("--date", type=str, required=True)
    ap.add_argument("--hour", type=int, required=True)

    ap.add_argument("--features", type=str, nargs="+")
    ap.add_argument("--feature", type=str)
    ap.add_argument("--tail", type=int, default=10)
    ap.add_argument("--quiet", "-q", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--format", choices=["table", "csv"], default="table")

    args = ap.parse_args()
    verbose = not args.quiet

    if args.dry_run:
        _, ctx_path, out_path = _paths_for_hour(args.raw_dir, args.ctx_dir, args.out_dir, args.asset, args.date, args.hour)
        archive_label = "disabled" if args.no_archive else args.archive_dir
        print(f"Would read context: {ctx_path}")
        print(f"Would write output:  {out_path}")
        print(f"Archive dir:         {archive_label}")
        return

    # Single-feature debug mode (no archive, no write)
    if args.feature:
        raw_paths, ctx_path, _ = _paths_for_hour(args.raw_dir, args.ctx_dir, args.out_dir, args.asset, args.date, args.hour)
        if not ctx_path.exists():
            raise FileNotFoundError(f"Missing context file: {ctx_path}")

        context_df = _read_parquet(str(ctx_path))
        _require_cols(context_df, ["bucket_dt_utc"], "context_df")
        context_buckets = pd.DatetimeIndex(pd.to_datetime(context_df["bucket_dt_utc"], utc=True))

        spec = _find_feature_by_name(ALL_S0_FEATURES, args.feature)
        engine = S0FeatureEngine(raw_paths, verbose=verbose)
        out = engine.compute_one_feature_on_context(spec, context_buckets=context_buckets).tail(args.tail)

        try:
            print(out.to_csv(index=False) if args.format == "csv" else out.to_string(index=False))
        except BrokenPipeError:
            pass
        return

    # Full build
    build_s0_features_for_hour(
        raw_dir=args.raw_dir,
        ctx_dir=args.ctx_dir,
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