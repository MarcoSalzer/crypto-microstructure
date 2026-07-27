#!/usr/bin/env python3
# etl/audit/audit_s0_to_s5_features.py
# ==============================================================================
# Unified Pipeline Audit — S0 → S5  (fast single-pass)
#
# Single-pass design:
#   - Read only needed columns when --stage is set (pyarrow columns=...)
#   - Convert all feature columns to numeric ONCE per file (no per-column to_numeric repeats)
#   - Aggregate NaN/Inf stats during the pass (no second full reread/concat)
#   - Progress logging every N files
#   - Optional --stats uses streaming min/max/mean/std + sampled median approximation
# ==============================================================================

# -----------------------------------------------------------------------------
# etl/audit/audit_s0_to_s5_features.py
# Reference Feature Corpus Audit (Thesis 3.3) over the MERGED corpus, stages S0-S5.
#   Every column is classified into its originating stage (S0-S5) and checked with
#   that stage's NaN/Inf, semantic and cross-column rules; --stage narrows the run
#   to one stage. This is the sole auditor for S0-S5; the former per-stage
#   audit_s0..s5_features scripts were removed as redundant.
#   S6 is covered separately by audit_s6_features.
#
# EXTERNAL DATA (standalone QA tool): reads the external, uncommitted ~94 GB
#   feature/data store, resolved via common.paths.DATA_ROOT (env THESIS_DATA_ROOT
#   or configs/paths.yaml). It does NOT run inside the repo without that store,
#   and is intentionally NOT wired into etl.run_all.
# START:  python -m etl.audit.audit_s0_to_s5_features --help
# -----------------------------------------------------------------------------

from __future__ import annotations

import argparse
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from common.paths import DATA_ROOT


# ---------------------------------------------------------------------------

_DEFAULT_FEATURES_DIR = DATA_ROOT / "s5_features"

# ANSI
_B, _G, _Y, _R, _D, _RST = "\033[1m", "\033[92m", "\033[93m", "\033[91m", "\033[2m", "\033[0m"
_CYAN = "\033[96m"

def _header(t):     print(f"\n{_B}{'=' * 76}{_RST}\n{_B}  {t}{_RST}\n{_B}{'=' * 76}{_RST}")
def _section(t):    print(f"\n{_CYAN}{_B}▸ {t}{_RST}")
def _ok(m):         print(f"  {_G}{m}{_RST}")
def _warn(m):       print(f"  {_Y}{m}{_RST}")
def _fail(m):       print(f"  {_R}{m}{_RST}")
def _kv(k, v, i=4): print(f"  {' '*i}{_D}{k}:{_RST} {v}")


# ==============================================================================
# S5 column prefix sets (must be checked BEFORE generic S0–S4 rules)
# ==============================================================================
_S5_SHOCK_PREFIXES   = ("absorption_break_shock_", "vacuum_score_shock_")
_S5_MAD_PREFIXES     = ("mad_absorption_break_",   "mad_vacuum_score_")
_S5_MEDIAN_PREFIXES  = ("median_absorption_break_", "median_vacuum_score_")
_S5_ZSCORE_PREFIXES  = (
    "z_net_add_sf_", "z_net_cancel_sf_",
    "net_add_robust_z_", "net_cancel_robust_z_",
)
_S5_PERSIST_PREFIXES = ("net_add_persist_", "net_cancel_persist_")

# S5 features that are structurally allowed to be all-NaN in some files.
# net_add_persist_sf_5bps_900s uses signal_persist with min_periods=900 over
# net_add_sf_5bps_900s which becomes fragmented during volatile market phases.
# The longest consecutive valid block can drop below 900, making all output NaN
# — this is correct behaviour, not a pipeline bug.
# Note: refill_vs_pull_ratio_* removed from whitelist — those features have been
# dropped from s3_liquidity_events.py and no longer exist in the pipeline.
_S5_ALLOWED_ALL_NAN: frozenset = frozenset({
    "net_add_persist_sf_5bps_900s",   # signal_persist 900s over net_add_sf — sparse in volatile phases
})


# ==============================================================================
# Unified feature classifier
# ==============================================================================
def _classify(name: str) -> Tuple[str, str]:
    if name == "bucket_dt_utc":
        return ("meta", "index")

    if any(name.startswith(p) for p in _S5_SHOCK_PREFIXES):
        return ("S5", "shock_s5")
    if any(name.startswith(p) for p in _S5_MAD_PREFIXES):
        return ("S5", "mad_s5")
    if any(name.startswith(p) for p in _S5_MEDIAN_PREFIXES):
        return ("S5", "median_s5")
    if any(name.startswith(p) for p in _S5_ZSCORE_PREFIXES):
        return ("S5", "zscore_s5")
    if any(name.startswith(p) for p in _S5_PERSIST_PREFIXES):
        return ("S5", "persist_s5")

    # S2 rolling z-scores have z_ prefix but are NOT S1 features
    # (exclude taker_imbalance — z_taker_imbalance is a proper S1 feature)
    if name.startswith("z_") and any(
        kw in name for kw in (
            "absorb_refill", "queue_pressure", "net_pressure",
            "queue_imb", "impact", "pull_rate", "refill_rate",
        )
    ):
        return ("S2", "zscore_s2")
    if name.startswith("z_"):
        return ("S1", "zscore_s1")

    if "pct_rank" in name:
        return ("S4", "pct_rank")
    if "robust_shock" in name:
        return ("S4", "shock_s4")
    if "rolling_mad" in name:
        return ("S4", "mad_s4")
    if "rolling_median" in name:
        return ("S4", "rolling_median")
    if "roll_sum" in name:
        return ("S4", "rolling_sum")
    if "passthrough" in name:
        return ("S4", "passthrough")
    if "zscore" in name or "z_score" in name:
        return ("S4", "zscore_s4")

    if "signal_flip" in name or "flip_rate" in name:
        return ("S3", "flip_rate")
    if "signal_persist" in name:
        return ("S3", "persistence")
    if "absorption_break_flag" in name:
        return ("S3", "break_flag")
    if "absorption_break" in name and "shock" in name:
        return ("S3", "absorption_break_shock")
    if "absorption_break" in name:
        return ("S3", "absorption_break")
    if "absorb_refill_mid" in name:
        return ("S3", "absorb_refill")
    if "vacuum" in name and "shock" in name:
        return ("S4", "shock_s4")
    if "vacuum" in name:
        return ("S3", "vacuum")
    if "qp_depth_coherence" in name:
        return ("S3", "qp_coherence")
    if "qp_depth_curvature" in name:
        return ("S3", "qp_curvature")
    if "qp_depth_slope" in name:
        return ("S3", "qp_slope")
    if "cross_div_delta" in name:
        return ("S3", "cross_div_delta")
    if "refill_vs_pull" in name and (
        "_div_fut_minus_spot_" in name or "_div_spot_minus_fut_" in name
    ):
        return ("S3", "cross_div")
    if "refill_vs_pull" in name:
        return ("S3", "refill_vs_pull")
    if "cross_div" in name and "cross_market_div" not in name:
        return ("S3", "cross_div")
    if "cross_persist" in name:
        return ("S3", "cross_persist")
    if "cross_share" in name:
        return ("S3", "cross_share")
    if "dir_consistency_asym" in name:
        return ("S3", "dir_consistency_asym")
    if "dir_consistency_persist" in name:
        return ("S3", "dir_consistency_persist")
    if "trade_absorption_ratio_bps" in name:
        return ("S3", "absorption_bps")
    if "temporal_d2" in name:
        return ("S3", "temporal_d2")
    if "temporal_d1" in name:
        return ("S3", "temporal_d1")
    if "cross_market_div" in name:
        return ("S4", "cross_div_s4")

    if "autocorr" in name:
        return ("S2", "autocorr")
    if "price_acceleration" in name or (
        ("_d2_" in name or name.startswith("d2_")) and "temporal" not in name
    ):
        return ("S2", "acceleration")
    if ("_d1_" in name or name.startswith("d1_")) and "temporal" not in name:
        return ("S2", "velocity")
    if "breakout" in name or "regime_flag" in name or ("shock" in name and "detect" in name):
        return ("S2", "flag")
    if "ofi_shock" in name:
        return ("S2", "shock_s2")
    if "shock" in name and "absorption" not in name and "robust" not in name:
        return ("S2", "shock_s2")
    if "dir_consistency" in name and "asym" not in name and "persist" not in name:
        return ("S2", "dir_consistency")
    if "unidir_ratio" in name:
        return ("S2", "unidir_ratio")
    if "depth_coherence" in name:
        return ("S2", "depth_coherence")
    if "depth_curvature" in name:
        return ("S2", "depth_curvature")
    if "depth_slope" in name:
        return ("S2", "depth_slope")
    if "mid_touch_dev" in name or "price_deviation" in name:
        return ("S2", "price_deviation")
    if "impact" in name and "absorption" not in name:
        return ("S2", "impact")
    if "absorption_asymmetry" in name:
        return ("S2", "absorption_asymmetry")
    if "churn" in name:
        return ("S2", "churn")
    if "cancel_rate" in name:
        return ("S2", "cancel_rate")
    if "aggressor_absorption" in name or "aggr_absorp" in name:
        return ("S2", "aggressor_absorption")
    # absorption_volume is a new S2 feature type
    if "absorption_volume" in name:
        return ("S2", "absorption_volume")
    if (("refill_rate" in name or "pull_rate" in name) and "persist" not in name and "shock" not in name):
        return ("S2", "flow_rate")
    if "net_add_pressure" in name or "net_cancel_pressure" in name:
        return ("S2", "net_pressure_s2")
    if ("mad" in name or "_mad_" in name) and "rolling_mad" not in name:
        return ("S2", "mad_s2")
    if "ret_vwap" in name:
        return ("S2", "return_s2")
    # ret_mid_* and ret_Ns rolling returns are S2 features
    if name.startswith("ret_mid_") or re.match(r"ret_\d+s$", name):
        return ("S2", "return_s2")
    # trade_count cross-market ratios/shares → S2
    if name.startswith("trade_count_") and "_share_" in name:
        return ("S2", "activity_share")
    if name.startswith("trade_count_") and "_div_" in name:
        return ("S2", "vol_div_s2")  # divergence/difference — unbounded, no [0,1] check
    # volume cross-market ratios → S2
    if "volume_" in name and ("_div_delta_" in name or "_sf_div_" in name):
        return ("S2", "vol_ratio_s2")
    # queue_imb (shorter than queue_imbalance, added in S2)
    if "queue_imb_persist" in name:
        return ("S2", "queue_imb_persist")
    if name.startswith("queue_imb_"):
        return ("S2", "queue_imb")
    # net_add / net_cancel raw flows (no pressure suffix) → S3
    if (name.startswith("net_add_") or name.startswith("net_cancel_")) and "pressure" not in name:
        return ("S3", "net_flow_s3")
    if ("roll_mean" in name or ("roll_median" in name and "rolling_median" not in name) or "roll_sum" in name):
        return ("S2", "rolling_agg")
    if "taker_imbalance_bucket" in name:
        return ("S2", "imbalance_bucket")

    if "log_return" in name or "ret_fwd" in name:
        return ("S1", "return_s1")
    if "range_pct" in name:
        return ("S1", "range_pct")
    if "range_pos" in name:
        return ("S1", "range_pos")
    if "basis_bps" in name:
        return ("S1", "basis_bps")
    if "basis" in name:
        return ("S1", "basis")
    if "taker_imbalance" in name:
        return ("S1", "taker_imbalance")
    # book_asymmetry_div_* are cross-market differential features → S3
    if "book_asymmetry" in name and "_div_" in name:
        return ("S3", "cross_div")
    if ("queue_imbalance" in name or "book_asymmetry" in name or "liq_cluster_asym" in name):
        return ("S1", "imbalance_s1")
    if "queue_pressure_log" in name:
        return ("S1", "pressure_log")
    if "net_pressure" in name:
        return ("S1", "net_pressure_s1")
    if "queue_pressure" in name:
        return ("S1", "pressure")
    if "vwap" in name:
        return ("S1", "vwap")
    if "avg_trade_size" in name:
        return ("S1", "avg_trade_size")
    if "participation_rate" in name:
        return ("S1", "participation_rate")
    # pull_rate / refill_rate with persist suffix → S3
    if ("pull_rate" in name or "refill_rate" in name) and "persist" in name:
        return ("S3", "persistence")
    if "absorb" in name or "refill" in name or "fill_rate" in name:
        return ("S1", "absorption_s1")
    if "add_rate" in name:
        return ("S1", "flow_rate_s1")
    if ("liq_sum" in name or "liq_concentration" in name or "max_liq_distance" in name):
        return ("S1", "liquidity")
    if "lwp" in name or "mid_touch" in name:
        return ("S1", "derived_price")
    # taker_activity_share_sf_* variant and spot_fut form
    if "taker_activity_share" in name or "spot_fut_taker_activity" in name:
        return ("S1", "activity_share")
    if "l2_update_count" in name:
        return ("S1", "count")
    if "depth_gradient" in name:
        return ("S1", "depth_gradient")
    # liq_imb cross-market div/persist → S3; plain → S1
    if "liq_imb" in name and "_div_" in name:
        return ("S3", "cross_div")
    if "liq_imb" in name and "persist" in name:
        return ("S3", "persistence")
    if "liq_imb" in name:
        return ("S1", "liq_imb")
    # trade_absorption_ratio without _bps suffix → S2
    if "trade_absorption_ratio" in name:
        return ("S2", "absorption_bps")

    def _is_raw_s0(n: str) -> bool:
        if any(tag in n for tag in ("_div_", "_delta_", "_ratio_")):
            return False
        m = re.search(r"_(\d+)s$", n)
        if m:
            return int(m.group(1)) <= 1
        return True

    if name.startswith("best_bid") or name.startswith("best_ask"):
        return ("S0", "price_level")
    if name.startswith("mid_"):
        return ("S0", "mid_price") if _is_raw_s0(name) else ("S1", "rolling_agg")
    if name.startswith("spread_"):
        return ("S0", "spread") if _is_raw_s0(name) else ("S1", "rolling_agg")
    if name.startswith("trade_count_"):
        return ("S0", "trade_count") if _is_raw_s0(name) else ("S1", "rolling_agg")
    if name.startswith("volume_"):
        return ("S0", "volume") if _is_raw_s0(name) else ("S1", "rolling_agg")
    if name.startswith("taker_buy_vol_"):
        return ("S0", "taker_buy_vol") if _is_raw_s0(name) else ("S1", "rolling_agg")
    if name.startswith("taker_sell_vol_"):
        return ("S0", "taker_sell_vol") if _is_raw_s0(name) else ("S1", "rolling_agg")
    if name.startswith("taker_buy_notional_"):
        return ("S0", "taker_buy_notional") if _is_raw_s0(name) else ("S1", "rolling_agg")
    if name.startswith("taker_sell_notional_"):
        return ("S0", "taker_sell_notional") if _is_raw_s0(name) else ("S1", "rolling_agg")
    if name.startswith("signed_vol_"):
        return ("S0", "signed_vol") if _is_raw_s0(name) else ("S1", "rolling_agg")
    if name.startswith("signed_notional_"):
        return ("S0", "signed_notional") if _is_raw_s0(name) else ("S1", "rolling_agg")
    if name.startswith("notional_"):
        return ("S0", "notional") if _is_raw_s0(name) else ("S1", "rolling_agg")
    if name.startswith("depth_imbalance_"):
        return ("S0", "depth_imbalance")
    if name.startswith("depth_notional_"):
        return ("S0", "depth_notional") if _is_raw_s0(name) else ("S1", "rolling_agg")
    if name.startswith("max_bps_") or name.startswith("bps_sym_"):
        return ("S0", "bps_metadata")

    # S1 OHLC reference values and range features
    if name.startswith("day_high_") or name.startswith("day_low_") \
            or name.startswith("day_open_") or name.startswith("day_close_"):
        return ("S1", "ohlc_ref")
    if name.startswith("day_range_bps_") or name.startswith("dist_to_day_high_") \
            or name.startswith("dist_to_day_low_"):
        return ("S1", "ohlc_ref")

    # S0 calendar / session features
    if name.startswith("session_") or name in ("us_holiday", "us_rth"):
        return ("S0", "calendar")

    # S0 health / usability flags
    if name in ("data_health_flag", "data_health_flag_soft", "data_usability_flag",
                "depth_availability", "depth_lobdeep_global", "trades_coverage_flag"):
        return ("S0", "health_meta")
    if name.startswith("l2_") or name == "lob50_health_flag":
        return ("S0", "health_meta")
    if name in ("health_reason_code", "unusable_reason_code"):
        return ("S0", "health_meta")
    if name.startswith("usability_"):
        return ("S0", "health_meta")

    # S1 flow-depth alignment (order flow × book shape composite)
    if name.startswith("flow_depth_align_"):
        return ("S1", "flow_depth_align")

    # S2 cross-market ratio features
    if name.startswith("trade_size_sf_ratio_"):
        return ("S2", "cross_ratio_s2")

    return ("meta", "other")


# ==============================================================================
# Thresholds + group sets (same as original)
# ==============================================================================
_NAN_THRESH: Dict[str, int] = {
    "price_level": 5, "mid_price": 5, "spread": 5, "bps_metadata": 5,
    "depth_imbalance": 5,
    "trade_count": 0, "volume": 0, "notional": 0,
    "taker_buy_vol": 0, "taker_sell_vol": 0,
    "taker_buy_notional": 0, "taker_sell_notional": 0,
    "signed_vol": 0, "signed_notional": 0, "depth_notional": 0,
    "zscore_s1": 25, "return_s1": 5, "range_pct": 25, "range_pos": 25,
    "basis_bps": 10, "basis": 10, "taker_imbalance": 10,
    "imbalance_s1": 10, "pressure_log": 10, "net_pressure_s1": 10,
    "pressure": 10, "vwap": 10, "avg_trade_size": 10,
    "participation_rate": 10, "absorption_s1": 10, "flow_rate_s1": 10,
    "liquidity": 10, "derived_price": 10, "activity_share": 10,
    "count": 5, "depth_gradient": 10, "rolling_agg": 25,
    "liq_imb": 25,
    "autocorr": 25, "acceleration": 25, "velocity": 25, "flag": 20,
    "shock_s2": 25, "dir_consistency": 20, "unidir_ratio": 20,
    "depth_coherence": 25, "depth_curvature": 25, "depth_slope": 25,
    "price_deviation": 10, "impact": 10,
    "absorption_asymmetry": 10, "churn": 10, "cancel_rate": 10,
    "aggressor_absorption": 90,
    "flow_rate": 10, "net_pressure_s2": 10, "mad_s2": 25,
    "return_s2": 10, "imbalance_bucket": 10,
    # new S2 groups
    "zscore_s2": 25, "absorption_volume": 15, "queue_imb": 20,
    "queue_imb_persist": 35, "net_flow_s3": 20, "vol_ratio_s2": 10, "vol_div_s2": 15,
    "flip_rate": 25, "persistence": 25, "break_flag": 20,
    "absorption_break": 20, "absorption_break_shock": 25, "absorb_refill": 10,
    "vacuum": 15, "qp_coherence": 5, "qp_curvature": 5, "qp_slope": 5,
    "cross_div": 5, "cross_div_delta": 10, "cross_persist": 20,
    "cross_share": 5, "dir_consistency_asym": 15, "dir_consistency_persist": 15,
    "refill_vs_pull": 10, "absorption_bps": 10,
    "temporal_d1": 10, "temporal_d2": 15,
    "shock_s3": 25, "mad_s3": 25,
    "pct_rank": 35, "zscore_s4": 35, "shock_s4": 35, "rolling_median": 35,
    "rolling_sum": 35, "mad_s4": 35, "passthrough": 10,
    "cross_div_s4": 15,
    "median_s5":  5, "mad_s5": 5, "shock_s5": 10, "zscore_s5": 15, "persist_s5": 15,
    "ohlc_ref": 50,
    "health_meta": 100,
    "calendar": 5,
    "flow_depth_align": 15,
    "cross_ratio_s2": 30,
}

_S0_ZERO_FILL = {
    "trade_count", "volume", "notional", "taker_buy_vol", "taker_sell_vol",
    "taker_buy_notional", "taker_sell_notional", "signed_vol", "signed_notional",
    "depth_notional",
}
_S0_NONNEG = {
    "trade_count", "volume", "notional", "taker_buy_vol", "taker_sell_vol",
    "taker_buy_notional", "taker_sell_notional", "depth_notional",
}

_ZSCORE_TAIL_SKIP = (
    "participation_rate", "pull_rate", "refill_rate", "volume_asym", "liq_imb",
    "taker_imbalance", "trade_absorption_ratio", "queue_pressure",
    "book_asymmetry", "liq_concentration", "depth_gradient",
    "net_cancel", "net_add", "mid_touch_dev", "impact_per_signed",
)

_STAGE_LABEL = {"S0":"S0","S1":"S1","S2":"S2","S3":"S3","S4":"S4","S5":"S5","meta":"  "}


# ==============================================================================
# File discovery
# ==============================================================================
_PAT = re.compile(r"^s5_features_(\w+)_(\d{4}-\d{2}-\d{2})_(\d{2})\.parquet$")

def _discover(features_dir: str, asset: Optional[str] = None, date: Optional[str] = None):
    d = Path(features_dir)
    if not d.exists():
        return []
    out = []
    for f in d.iterdir():
        m = _PAT.match(f.name)
        if not m:
            continue
        a, ds, h = m.group(1), m.group(2), int(m.group(3))
        if asset and a != asset:
            continue
        if date and ds != date:
            continue
        out.append((a, ds, h, f))
    return sorted(out)


# ==============================================================================
# NaN diagnosis reason (same function as original, kept)
# ==============================================================================
def _diagnose_nan(col: str) -> str:
    stage, g = _classify(col)

    if stage == "S5":
        m = re.search(r"_(\d+)s", col)
        w = m.group(1) if m else "?"
        if g == "median_s5":
            return f"S5 rolling median (window={w}s) — warmup NaN; mostly eliminated by context loading"
        if g == "mad_s5":
            return f"S5 rolling MAD (window={w}s) — NaN where median NaN + zero-MAD in flat periods"
        if g == "shock_s5":
            return f"S5 robust shock (window={w}s) — NaN where MAD=0 (zero-MAD guard)"
        if g == "zscore_s5":
            return f"S5 robust z-score (window={w}s) — warmup NaN + zero-MAD guard"
        if g == "persist_s5":
            return f"S5 signal persistence (window={w}s, min_periods={w}) — full window required"

    if "ret_fwd" in col:
        m = re.search(r"(\d+)s$", col)
        return f"forward shift ({m.group(1) if m else '?'}s) → tail NaN"
    if "log_return" in col:
        return "log(price/price.shift(1)) → 1 leading NaN"
    if "_persist_" in col or col.endswith("_persist"):
        m = re.search(r"_persist_?(\d+)s", col)
        w = m.group(1) if m else "?"
        return f"rolling persistence (window={w}s) → warmup NaN"
    if "_robust_z_" in col:
        m = re.search(r"_(\d+)s$", col)
        w = m.group(1) if m else "?"
        return f"robust z-score (window={w}s) → warmup NaN"
    if "_flip_rate_" in col:
        m = re.search(r"_(\d+)s$", col)
        w = m.group(1) if m else "?"
        return f"rolling flip rate (window={w}s) → warmup NaN"
    if "_pct_rank_" in col:
        m = re.search(r"_(\d+)s$", col)
        w = m.group(1) if m else "?"
        return f"rolling pct_rank (window={w}s) → warmup NaN"

    # keep same generic mapping for rest (shortened here is OK for speed script)
    if stage == "S1" and g == "zscore_s1":
        m = re.search(r"_(\d+)s", col)
        w = m.group(1) if m else "?"
        if w == "1":
            return "rolling z-score (window=1s) → warmup NaN (window too small)"
        return f"rolling z-score (window={w}s) → warmup NaN"

    if stage in ("S0","S1","S2","S3","S4"):
        m = re.search(r"_(\d+)s", col)
        w = m.group(1) if m else "?"
        if w != "?":
            return f"{stage} rolling/stat feature (window={w}s) → warmup NaN / upstream NaN"
        # No rolling window suffix — classify by group
        if g in ("ohlc_ref", "range_pos", "range_pct"):
            return "daily OHLC/range reference — NaN before first daily bar"
        if g == "health_meta":
            return "S0 health/usability flag — NaN from data gaps"
        if g in ("calendar",):
            return "S0 calendar feature — NaN from missing session data"
        # S0 raw snapshots (price_level, mid_price, spread, depth_imbalance, …)
        return f"S0 raw snapshot — NaN from data gaps ({g})"

    # meta/other — truly unclassified column
    return f"unclassified ({stage}/{g})"


# ==============================================================================
# Aggregation structs
# ==============================================================================
@dataclass
class FileAudit:
    asset: str
    date_str: str
    hour: int
    path: Path
    rows: int = 0
    cols: int = 0
    size_mb: float = 0.0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    ok: bool = True

@dataclass
class ColAgg:
    n: int = 0               # total rows seen
    nan: int = 0
    inf: int = 0
    finite: int = 0
    min: float = np.nan
    max: float = np.nan
    mean: float = 0.0
    m2: float = 0.0          # Welford
    # median approximation via reservoir-like sampling
    sample: List[float] = field(default_factory=list)

    def update(self, arr: np.ndarray, sample_k: int = 0) -> None:
        self.n += arr.size
        # nan/inf counts
        nan_mask = np.isnan(arr)
        self.nan += int(nan_mask.sum())
        # finite mask excludes nan and inf
        finite_mask = np.isfinite(arr)
        self.inf += int(np.isinf(arr).sum())
        fin = arr[finite_mask]
        self.finite += int(fin.size)
        if fin.size == 0:
            return
        fmin = float(np.min(fin))
        fmax = float(np.max(fin))
        self.min = fmin if not np.isfinite(self.min) else min(self.min, fmin)
        self.max = fmax if not np.isfinite(self.max) else max(self.max, fmax)

        # Chan's parallel Welford update (fully vectorized, no Python loop)
        n_b    = fin.size
        mean_b = float(np.mean(fin))
        m2_b   = float(np.var(fin) * n_b)   # sum of squared deviations for batch
        n_a    = self.finite                 # count BEFORE adding fin
        if n_a == 0:
            self.mean = mean_b
            self.m2   = m2_b
        else:
            n_new     = n_a + n_b
            delta     = mean_b - self.mean
            self.mean += delta * n_b / n_new
            self.m2   += m2_b + delta ** 2 * n_a * n_b / n_new

        # sampling for median approx
        if sample_k > 0:
            # take small subsample from this chunk
            take = min(sample_k, fin.size)
            if take > 0:
                idx = np.random.choice(fin.size, size=take, replace=False)
                self.sample.extend(fin[idx].tolist())
                # cap memory
                if len(self.sample) > sample_k * 200:
                    self.sample = self.sample[-sample_k * 200 :]

    def std(self) -> float:
        if self.finite <= 1:
            return np.nan
        return float(np.sqrt(self.m2 / (self.finite - 1)))

    def median_approx(self) -> float:
        if not self.sample:
            return np.nan
        return float(np.median(np.array(self.sample, dtype=np.float64)))


# ==============================================================================
# Reading helper: only needed columns if stage_filter
# ==============================================================================
def _wanted_cols_from_schema(colnames: List[str], stage_filter: Optional[str]) -> List[str]:
    if stage_filter is None:
        return colnames
    wanted = ["bucket_dt_utc"]
    for c in colnames:
        if c == "bucket_dt_utc":
            continue
        st, _ = _classify(c)
        if st == stage_filter:
            wanted.append(c)
    # Always keep upstream cols needed for cross-column checks even with filter
    if stage_filter == "S0":
        needed = []
        for mk in ("fut","spot"):
            needed += [f"best_bid_{mk}_1s", f"best_ask_{mk}_1s", f"mid_{mk}_1s", f"spread_{mk}_1s",
                       f"volume_{mk}_1s", f"taker_buy_vol_{mk}_1s", f"taker_sell_vol_{mk}_1s", f"signed_vol_{mk}_1s"]
        for c in needed:
            if c in colnames and c not in wanted:
                wanted.append(c)
    return wanted


# ==============================================================================
# Core per-file audit (same checks, but fast numeric handling)
# ==============================================================================
_MIN_EXPECTED_COLS = 100

def _audit_one_file(asset: str, date_str: str, hour: int, path: Path,
                    stage_filter: Optional[str],
                    agg: Dict[str, ColAgg],
                    sample_k_for_stats: int,
                    nan_only: bool) -> FileAudit:
    r = FileAudit(asset=asset, date_str=date_str, hour=hour, path=path)
    r.size_mb = path.stat().st_size / (1024 * 1024)

    # Read schema first
    pf = pq.ParquetFile(path)
    colnames = pf.schema_arrow.names
    wanted_cols = _wanted_cols_from_schema(colnames, stage_filter)

    try:
        df = pq.read_table(str(path), columns=wanted_cols).to_pandas()
    except Exception as e:
        r.errors.append(f"read_error:{e}")
        r.ok = False
        return r

    n = len(df)
    r.rows, r.cols = n, len(df.columns)

    if n == 0:
        r.errors.append("empty_df")
        r.ok = False
        return r
    if n < 3590:
        r.warnings.append(f"low_rows:{n}")
    if r.cols < _MIN_EXPECTED_COLS and stage_filter is None:
        r.warnings.append(f"low_col_count:{r.cols} (expected ≥{_MIN_EXPECTED_COLS})")

    if "bucket_dt_utc" not in df.columns:
        r.errors.append("missing_upstream_col:bucket_dt_utc")
    else:
        ts = pd.to_datetime(df["bucket_dt_utc"], utc=True).sort_values()
        nd = int(ts.duplicated().sum())
        if nd:
            r.errors.append(f"dup_timestamps:{nd}")
        if len(ts) > 1:
            gaps = ts.diff().dropna()
            big_gaps = gaps[gaps > pd.Timedelta("1s")]
            if len(big_gaps):
                r.warnings.append(f"gaps:{len(big_gaps)}")

    feat_cols = [c for c in df.columns if c != "bucket_dt_utc"]
    if not feat_cols:
        r.ok = len(r.errors) == 0
        return r

    # Convert all feature cols to numeric ONCE
    feat = df[feat_cols]
    num = feat.apply(pd.to_numeric, errors="coerce")  # DataFrame float/object -> float where possible

    # Inf + NaN check + aggregation (single pass)
    for col in feat_cols:
        st, g = _classify(col)
        if stage_filter and st != stage_filter:
            continue
        s = num[col].to_numpy(dtype=np.float64, copy=False)
        # count inf
        inf_ct = int(np.isinf(s).sum())
        if inf_ct:
            r.errors.append(f"inf:{col}:{inf_ct}")

        nan_pct = float(np.isnan(s).sum()) / n * 100.0
        thresh = _NAN_THRESH.get(g, 10)

        if st == "S0" and g in _S0_ZERO_FILL and nan_pct > 0:
            r.errors.append(f"nan_zerofill:{col}:{nan_pct:.1f}%")
        elif nan_pct > thresh:
            r.warnings.append(f"nan:{col}:{nan_pct:.1f}%")

        # global aggregation
        agg[col].update(s, sample_k=sample_k_for_stats)

    if nan_only:
        r.ok = len(r.errors) == 0
        return r

    # Semantic checks (same logic as original, now using num[col])
    for col in feat_cols:
        st, g = _classify(col)
        if stage_filter and st != stage_filter:
            continue
        s = num[col].to_numpy(dtype=np.float64, copy=False)
        sf = s[np.isfinite(s)]
        if sf.size == 0:
            continue
        vmin, vmax = float(np.min(sf)), float(np.max(sf))

        # S0
        if g in ("price_level", "mid_price"):
            if (sf <= 0).any():
                r.errors.append(f"non_positive_price:{col}:{int((sf<=0).sum())}")
        elif g == "spread":
            if (sf < 0).any():
                r.errors.append(f"negative_spread:{col}:{int((sf<0).sum())}")
        elif g == "depth_imbalance":
            out = int(((sf < -1.001) | (sf > 1.001)).sum())
            if out:
                r.errors.append(f"imbalance_oob:{col}:{out}")
        elif g == "bps_metadata":
            if (sf < 0).any():
                r.errors.append(f"negative_bps:{col}:{int((sf<0).sum())}")
        elif st == "S0" and g in _S0_NONNEG:
            if (sf < 0).any():
                r.errors.append(f"negative:{col}:{int((sf<0).sum())}")

        # S1
        elif g == "zscore_s1":
            extreme_pct = float(((sf < -10) | (sf > 10)).sum()) / sf.size * 100
            if not any(skip in col for skip in _ZSCORE_TAIL_SKIP) and extreme_pct > 50:
                r.warnings.append(f"zscore_tail:{col}:{extreme_pct:.1f}%")
            if abs(vmax) > 50 or abs(vmin) > 50:
                r.errors.append(f"zscore_extreme:{col}:[{vmin:.1f},{vmax:.1f}]")
        elif g == "range_pos":
            out = int(((sf < -0.001) | (sf > 1.001)).sum())
            if out:
                r.errors.append(f"range_pos_oob:{col}:{out}")
        elif g == "range_pct":
            if (sf < -0.001).any():
                r.errors.append(f"range_pct_neg:{col}:{int((sf<-0.001).sum())}")
        elif g == "taker_imbalance":
            out = int(((sf < -1.001) | (sf > 1.001)).sum())
            if out:
                r.errors.append(f"imbalance_oob:{col}:{out}")
        elif g in ("vwap", "derived_price"):
            if (sf <= 0).any():
                r.errors.append(f"neg_price:{col}:{int((sf<=0).sum())}")
        elif g in ("avg_trade_size", "participation_rate"):
            if (sf < 0).any():
                r.errors.append(f"negative:{col}:{int((sf<0).sum())}")
        elif g == "activity_share":
            out = int(((sf < -0.001) | (sf > 1.001)).sum())
            if out:
                r.errors.append(f"activity_oob:{col}:{out}")
        elif g in ("flow_rate_s1", "count"):
            if (sf < 0).any():
                r.errors.append(f"negative:{col}:{int((sf<0).sum())}")
        elif g == "return_s1":
            if abs(vmin) > 0.1 or abs(vmax) > 0.1:
                r.warnings.append(f"extreme_return:{col}:[{vmin:.6f},{vmax:.6f}]")
        elif g == "basis_bps":
            if abs(vmax) > 500 or abs(vmin) > 500:
                r.warnings.append(f"extreme_basis:{col}:[{vmin:.1f},{vmax:.1f}]")

        # S2
        elif g == "autocorr":
            out = int(((sf < -1.001) | (sf > 1.001)).sum())
            if out:
                r.errors.append(f"autocorr_oob:{col}:{out}")
        elif g == "flag":
            non_binary = set(np.unique(sf)) - {0.0, 1.0}
            if non_binary:
                r.errors.append(f"flag_non_binary:{col}:{sorted(list(non_binary))[:5]}")
        elif g == "dir_consistency":
            out = int(((sf < -1.001) | (sf > 1.001)).sum())
            if out:
                r.errors.append(f"dir_consistency_oob:{col}:{out}")
        elif g == "unidir_ratio":
            out = int(((sf < -0.001) | (sf > 1.001)).sum())
            if out:
                r.errors.append(f"unidir_ratio_oob:{col}:{out}")
        elif g == "shock_s2":
            if abs(vmax) > 100 or abs(vmin) > 100:
                r.warnings.append(f"extreme_shock:{col}:[{vmin:.2f},{vmax:.2f}]")
        elif g == "mad_s2":
            if (sf < -1e-10).any():
                r.errors.append(f"mad_negative:{col}:{int((sf<-1e-10).sum())}")
        elif g == "depth_coherence":
            out = int(((sf < -1.001) | (sf > 1.001)).sum())
            if out:
                r.warnings.append(f"depth_coherence_oob:{col}:{out}")
        elif g == "absorption_asymmetry":
            out = int(((sf < -1.001) | (sf > 1.001)).sum())
            if out:
                r.errors.append(f"absorption_asym_oob:{col}:{out}")
        elif g == "imbalance_bucket":
            out = int(((sf < -1.001) | (sf > 1.001)).sum())
            if out:
                r.errors.append(f"imbalance_oob:{col}:{out}")

        # S3
        elif g == "flip_rate":
            out = int(((sf < -0.001) | (sf > 1.001)).sum())
            if out:
                r.errors.append(f"flip_rate_oob:{col}:{out}")
        elif g == "persistence":
            out = int(((sf < -0.001) | (sf > 1.001)).sum())
            if out:
                r.errors.append(f"persistence_oob:{col}:{out}")
        elif g == "break_flag":
            non_binary = set(np.unique(sf)) - {0.0, 1.0}
            if non_binary:
                r.errors.append(f"flag_non_binary:{col}:{sorted(list(non_binary))[:5]}")
            fire_rate = float((sf == 1.0).sum() / sf.size * 100)
            if fire_rate > 50:
                r.warnings.append(f"break_flag_always_on:{col}:{fire_rate:.1f}%")
        elif g == "absorption_break":
            out = int(((sf < -0.16) | (sf > 0.86)).sum())
            if out:
                r.warnings.append(f"absorption_break_oob:{col}:{out}")
        elif g == "vacuum":
            if abs(vmax) > 40.001 or abs(vmin) > 40.001:
                r.errors.append(f"vacuum_exceeds_clip:{col}:[{vmin:.1f},{vmax:.1f}]")
        elif g == "qp_coherence":
            non_binary = set(np.unique(np.round(sf, 6))) - {0.0, 1.0}
            if non_binary:
                r.errors.append(f"qp_coherence_non_binary:{col}:{sorted(list(non_binary))[:5]}")
        elif g in ("qp_curvature", "qp_slope"):
            if abs(vmax) > 1e6 or abs(vmin) > 1e6:
                r.warnings.append(f"qp_shape_extreme:{col}:[{vmin:.2e},{vmax:.2e}]")
        elif g == "cross_share":
            out = int(((sf < -0.001) | (sf > 1.001)).sum())
            if out:
                r.errors.append(f"cross_share_oob:{col}:{out}")
        elif g == "cross_persist":
            out = int(((sf < -0.001) | (sf > 1.001)).sum())
            if out:
                r.warnings.append(f"cross_persist_oob:{col}:{out}")
        elif g == "dir_consistency_persist":
            non_binary = set(np.unique(sf)) - {0.0, 1.0}
            if non_binary:
                r.warnings.append(f"dir_consist_persist_non_binary:{col}")
        elif g == "dir_consistency_asym":
            out = int(((sf < -1.001) | (sf > 1.001)).sum())
            if out:
                r.warnings.append(f"dir_consist_asym_oob:{col}:{out}")
        elif g == "refill_vs_pull":
            neg = int((sf < -0.001).sum())
            if neg > n * 0.01:
                r.warnings.append(f"negative:{col}:{neg}")
            extreme_pct = float((sf > 1e4).sum()) / sf.size * 100
            if extreme_pct > 5:
                r.warnings.append(f"refill_vs_pull_extreme:{col}:{extreme_pct:.1f}%")
        elif g in ("temporal_d1", "temporal_d2"):
            if abs(vmax) > 100 or abs(vmin) > 100:
                r.warnings.append(f"extreme_deriv:{col}:[{vmin:.4f},{vmax:.4f}]")
        elif g in ("shock_s3", "mad_s3"):
            if (sf < -1e-10).any():
                r.errors.append(f"mad_negative:{col}:{int((sf<-1e-10).sum())}")

        # S4
        elif g == "pct_rank":
            out = int(((sf < -0.001) | (sf > 1.001)).sum())
            if out:
                r.errors.append(f"pct_rank_oob:{col}:{out}")
            if sf.size > 100 and float(np.std(sf)) < 1e-10:
                r.warnings.append(f"pct_rank_constant:{col}")
        elif g == "zscore_s4":
            extreme_pct = float(((sf < -10) | (sf > 10)).sum()) / sf.size * 100
            if extreme_pct > 50:
                r.warnings.append(f"zscore_tail:{col}:{extreme_pct:.1f}%")
            if abs(vmax) > 50 or abs(vmin) > 50:
                r.errors.append(f"zscore_extreme:{col}:[{vmin:.1f},{vmax:.1f}]")
        elif g == "shock_s4":
            if abs(vmax) > 100 or abs(vmin) > 100:
                r.warnings.append(f"shock_extreme:{col}:[{vmin:.2f},{vmax:.2f}]")
        elif g == "mad_s4":
            if (sf < -1e-10).any():
                r.errors.append(f"mad_negative:{col}:{int((sf<-1e-10).sum())}")
        elif g == "absorption_break_shock":
            if abs(vmax) > 100 or abs(vmin) > 100:
                r.warnings.append(f"shock_extreme:{col}:[{vmin:.2f},{vmax:.2f}]")

        # S5
        elif g == "zscore_s5":
            extreme_pct = float(((sf < -10) | (sf > 10)).sum()) / sf.size * 100
            if extreme_pct > 50:
                r.warnings.append(f"zscore_tail:{col}:{extreme_pct:.1f}%")
            if abs(vmax) > 20 or abs(vmin) > 20:
                r.errors.append(f"zscore_clip_violated:{col}:[{vmin:.2f},{vmax:.2f}]")
            else:
                near_clip_pct = float(((sf > 15) | (sf < -15)).sum()) / sf.size * 100
                if near_clip_pct > 95:
                    r.warnings.append(f"zscore_near_clip:{col}:{near_clip_pct:.1f}%")
            if float(np.std(sf)) < 1e-10:
                r.warnings.append(f"zscore_constant:{col}")
        elif g == "shock_s5":
            if vmin < -1e-6:
                r.errors.append(f"shock_negative:{col}:{vmin:.6f}")
            if vmax > 50.01:
                r.errors.append(f"shock_clip_violated:{col}:{vmax:.2f}")
            else:
                near_clip_pct = float((sf > 40).sum()) / sf.size * 100
                if near_clip_pct > 95:
                    r.warnings.append(f"shock_near_clip:{col}:{near_clip_pct:.1f}%")
        elif g == "persist_s5":
            out = int(((sf < -0.001) | (sf > 1.001)).sum())
            if out:
                r.errors.append(f"persistence_oob:{col}:{out}")
            if float(np.std(sf)) < 1e-10:
                r.warnings.append(f"persistence_constant:{col}")
        elif g == "mad_s5":
            neg = int((sf < -1e-10).sum())
            if neg:
                r.errors.append(f"mad_negative:{col}:{neg}")

    # Cross-column checks (same as original)
    if not stage_filter or stage_filter == "S0":
        for mk in ("fut", "spot"):
            bid_c = f"best_bid_{mk}_1s"; ask_c = f"best_ask_{mk}_1s"
            mid_c = f"mid_{mk}_1s";     sp_c  = f"spread_{mk}_1s"
            if not all(c in df.columns for c in [bid_c, ask_c, mid_c, sp_c]):
                continue
            bid = pd.to_numeric(df[bid_c], errors="coerce")
            ask = pd.to_numeric(df[ask_c], errors="coerce")
            mid = pd.to_numeric(df[mid_c], errors="coerce")
            sp  = pd.to_numeric(df[sp_c],  errors="coerce")
            ok_mask = bid.notna() & ask.notna() & mid.notna() & sp.notna()
            if ok_mask.sum() > 0:
                mid_err = (mid[ok_mask] - (bid[ok_mask] + ask[ok_mask]) / 2).abs()
                if (mid_err > 0.01).any():
                    r.errors.append(f"s0_{mk}:mid_ne_(bid+ask)/2:{int((mid_err>0.01).sum())}")
                sp_err = (sp[ok_mask] - (ask[ok_mask] - bid[ok_mask])).abs()
                if (sp_err > 0.01).any():
                    r.errors.append(f"s0_{mk}:spread_ne_ask-bid:{int((sp_err>0.01).sum())}")
                crossed = int((bid[ok_mask] > ask[ok_mask]).sum())
                if crossed:
                    r.errors.append(f"s0_{mk}:crossed_book:{crossed}")
        for mk in ("fut","spot"):
            vol_c  = f"volume_{mk}_1s"; buy_c = f"taker_buy_vol_{mk}_1s"
            sell_c = f"taker_sell_vol_{mk}_1s"; sgn_c = f"signed_vol_{mk}_1s"
            if not all(c in df.columns for c in [vol_c, buy_c, sell_c, sgn_c]):
                continue
            vol  = pd.to_numeric(df[vol_c],  errors="coerce").fillna(0)
            buy  = pd.to_numeric(df[buy_c],  errors="coerce").fillna(0)
            sell = pd.to_numeric(df[sell_c], errors="coerce").fillna(0)
            sgn  = pd.to_numeric(df[sgn_c],  errors="coerce").fillna(0)
            tol  = vol * 1e-4 + 1e-6
            if ((vol - (buy + sell)).abs() > tol).any():
                r.errors.append(f"s0_{mk}:buy+sell_ne_vol")
            if ((sgn - (buy - sell)).abs() > tol).any():
                r.errors.append(f"s0_{mk}:signed_ne_buy-sell")

    # S3 cross-column: d2 NaN ≥ d1 NaN (subset as original)
    if not stage_filter or stage_filter in ("S3","S4","S5"):
        d1_cols = [c for c in feat_cols if _classify(c)[1] == "temporal_d1"]
        d2_cols = [c for c in feat_cols if _classify(c)[1] == "temporal_d2"]
        for d1c in d1_cols[:3]:
            for d2c in d2_cols[:3]:
                b1 = re.sub(r"^(d1_|temporal_d1_)", "", d1c)
                b2 = re.sub(r"^(d2_|temporal_d2_)", "", d2c)
                if b1 != b2:
                    continue
                nan1 = int(num[d1c].isna().sum()) if d1c in num.columns else 0
                nan2 = int(num[d2c].isna().sum()) if d2c in num.columns else 0
                if nan2 < nan1:
                    r.warnings.append(f"cross:d2_fewer_nan:{d2c}={nan2}<{d1c}={nan1}")

    # S4 constant columns (same)
    if not stage_filter or stage_filter == "S4":
        s4_cols = [c for c in feat_cols if _classify(c)[0] == "S4"]
        constant_count = 0
        for c in s4_cols:
            sc = num[c].dropna()
            if len(sc) > 100 and float(sc.std()) < 1e-15:
                constant_count += 1
        if constant_count > 3:
            r.warnings.append(f"constant_s4_cols:{constant_count}")

    # S5 cross-column checks (same)
    if not stage_filter or stage_filter == "S5":
        s5_cols = [c for c in feat_cols if _classify(c)[0] == "S5"]

        median_names = [c for c in s5_cols if _classify(c)[1] == "median_s5"]
        for median_col in median_names:
            base = re.sub(r"^median_", "", median_col)
            mad_col = f"mad_{base}"
            shock_col = next(
                (c for c in s5_cols if _classify(c)[1] == "shock_s5" and base in c),
                None,
            )
            if mad_col in num.columns:
                nan_median = int(num[median_col].isna().sum())
                nan_mad    = int(num[mad_col].isna().sum())
                if nan_mad < nan_median:
                    r.warnings.append(
                        f"cross_check:mad_fewer_nan_than_median:{mad_col}={nan_mad}<{median_col}={nan_median}"
                    )
            if shock_col and shock_col in num.columns and mad_col in num.columns:
                nan_mad   = int(num[mad_col].isna().sum())
                nan_shock = int(num[shock_col].isna().sum())
                if nan_shock < nan_mad:
                    r.warnings.append(
                        f"cross_check:shock_fewer_nan_than_mad:{shock_col}={nan_shock}<{mad_col}={nan_mad}"
                    )

        add_persist    = {c for c in s5_cols if "net_add_persist" in c}
        cancel_persist = {c for c in s5_cols if "net_cancel_persist" in c}
        for add_col in add_persist:
            cancel_col = add_col.replace("net_add_persist", "net_cancel_persist")
            if cancel_col not in num.columns:
                r.warnings.append(f"cross_check:missing_cancel_persist:{cancel_col}")
        for cancel_col in cancel_persist:
            add_col = cancel_col.replace("net_cancel_persist", "net_add_persist")
            if add_col not in num.columns:
                r.warnings.append(f"cross_check:missing_add_persist:{add_col}")

        add_z    = {c for c in s5_cols if "net_add_robust_z" in c}
        cancel_z = {c for c in s5_cols if "net_cancel_robust_z" in c}
        for add_col in add_z:
            cancel_col = add_col.replace("net_add_robust_z", "net_cancel_robust_z")
            if cancel_col not in num.columns:
                r.warnings.append(f"cross_check:missing_cancel_robust_z:{cancel_col}")
        for cancel_col in cancel_z:
            add_col = cancel_col.replace("net_cancel_robust_z", "net_add_robust_z")
            if add_col not in num.columns:
                r.warnings.append(f"cross_check:missing_add_robust_z:{add_col}")

        all_nan_count = 0
        for c in s5_cols:
            if c in num.columns and num[c].isna().all():
                if c not in _S5_ALLOWED_ALL_NAN:
                    all_nan_count += 1
        if all_nan_count > 0:
            r.errors.append(f"all_nan_s5_cols:{all_nan_count}")

        z_cols = [c for c in s5_cols if _classify(c)[1] == "zscore_s5" and c in num.columns]
        constant_z = 0
        for c in z_cols:
            sc = num[c].dropna()
            if len(sc) > 100 and float(sc.std()) < 1e-10:
                constant_z += 1
        if constant_z > 3:
            r.warnings.append(f"cross_check:many_constant_s5_zscores:{constant_z}")

    r.ok = len(r.errors) == 0
    return r


# ==============================================================================
# Issue aggregation (same)
# ==============================================================================
def _aggregate_issues(results: List[FileAudit]) -> None:
    err_counter: Counter = Counter()
    warn_counter: Counter = Counter()
    for r in results:
        for e in r.errors:
            key = ":".join(e.split(":")[:2]) if ":" in e else e
            err_counter[key] += 1
        for w in r.warnings:
            if w.startswith("nan:"):
                continue
            key = ":".join(w.split(":")[:2]) if ":" in w else w
            warn_counter[key] += 1

    if err_counter:
        _section(f"Errors — grouped by type ({sum(err_counter.values())} total)")
        for issue, count in err_counter.most_common(30):
            print(f"    {_R}{issue}  ({count} file{'s' if count > 1 else ''}){_RST}")
    if warn_counter:
        _section(f"Warnings (non-NaN) — grouped by type ({sum(warn_counter.values())} total)")
        for issue, count in warn_counter.most_common(30):
            print(f"    {_Y}{issue}  ({count} file{'s' if count > 1 else ''}){_RST}")


# ==============================================================================
# Print NaN diagnosis from aggregated counts (no reread)
# ==============================================================================
def _nan_diagnosis_from_agg(agg: Dict[str, ColAgg], total_rows: int, assets: List[str]) -> None:
    # This audit reads s5_features (already merged stages), so we do global diagnosis.
    nan_info = []
    for col, a in agg.items():
        if col == "bucket_dt_utc":
            continue
        if a.n == 0:
            continue
        nan_pct = a.nan / max(a.n, 1) * 100.0
        if nan_pct < 0.1:
            continue
        nan_info.append((col, a.nan, nan_pct, _diagnose_nan(col)))

    if not nan_info:
        _section("NaN Diagnosis — no significant NaN")
        return

    reason_groups: Dict[str, list] = defaultdict(list)
    for col, ct, pct, reason in nan_info:
        reason_groups[reason].append((col, ct, pct))

    sorted_groups = sorted(
        reason_groups.items(),
        key=lambda x: sum(ct for _, ct, _ in x[1]),
        reverse=True,
    )

    _section(f"NaN Diagnosis — aggregated ({total_rows:,} rows)")
    print(f"  {len(nan_info)} columns with >0.1% NaN\n")

    for reason, cols in sorted_groups:
        total_nan = sum(ct for _, ct, _ in cols)
        print(f"  {_B}{reason}{_RST}  ({len(cols)} feature{'s' if len(cols)>1 else ''}, {total_nan:,} NaN)")
        for i, (col, ct, pct) in enumerate(sorted(cols, key=lambda x: -x[2])):
            if i < 5:
                short = col if len(col) <= 50 else col[:47] + "..."
                print(f"    {_D}│{_RST} {short:<52s} {pct:>6.1f}%  ({ct:>9,} NaN)")
            elif i == 5:
                print(f"    {_D}│ ... and {len(cols)-5} more{_RST}")
                break
        print()


# ==============================================================================
# Stats printer from agg (median approx, others exact)
# ==============================================================================
def _print_stats_from_agg(agg: Dict[str, ColAgg], stage_filter: Optional[str]) -> None:
    _section(f"Stats — aggregated (median approx via sampling){f'  [stage={stage_filter}]' if stage_filter else ''}")
    hdr = f"    {'stg':<4s} {'feature':<48s} {'nan%':>5s}  {'min':>12s}  {'median~':>12s}  {'max':>12s}  {'std':>12s}"
    print(hdr)
    print(f"    {'─'*104}")

    # group by stage/group
    grouped: Dict[Tuple[str,str], List[str]] = defaultdict(list)
    for col in agg.keys():
        if col == "bucket_dt_utc":
            continue
        st, grp = _classify(col)
        if stage_filter and st != stage_filter:
            continue
        grouped[(st, grp)].append(col)

    # deterministic ordering
    keys = sorted(grouped.keys(), key=lambda k: (k[0], k[1]))
    for st, grp in keys:
        cols = sorted(grouped[(st, grp)])
        print(f"\n    {_B}{_STAGE_LABEL.get(st,st)} {grp}{_RST} ({len(cols)})")
        for col in cols:
            a = agg[col]
            nan_pct = a.nan / max(a.n, 1) * 100.0
            vmin = a.min
            vmax = a.max
            vstd = a.std()
            vmed = a.median_approx()
            short = col if len(col) <= 46 else col[:43] + "..."
            fmt = ",.2f" if (np.isfinite(vmax) and abs(vmax) > 1000) else ".4f" if (np.isfinite(vmax) and abs(vmax) > 1) else ".6f"
            if a.finite == 0:
                print(f"    {_STAGE_LABEL.get(st,st):<4s} {short:<48s} {nan_pct:>5.1f}  {'(all NaN)':>12s}")
                continue
            print(
                f"    {_STAGE_LABEL.get(st,st):<4s} {short:<48s} {nan_pct:>5.1f}"
                f"  {vmin:>12{fmt}}  {vmed:>12{fmt}}  {vmax:>12{fmt}}  {vstd:>12{fmt}}"
            )


# ==============================================================================
# Main
# ==============================================================================
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Unified pipeline audit (fast) — single pass over s5_features.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--features-dir", default=str(_DEFAULT_FEATURES_DIR))
    ap.add_argument("--asset", default=None, choices=["btc","eth","bnb"])
    ap.add_argument("--date",  default=None, help="YYYY-MM-DD")
    ap.add_argument("--stage", default=None, choices=["S0","S1","S2","S3","S4","S5"])
    ap.add_argument("--expand", "-e", action="store_true")
    ap.add_argument("--stats",  "-s", action="store_true")
    ap.add_argument("--nan-only", action="store_true",
                    help="Only NaN/Inf + NaN diagnosis, skip semantic checks")
    ap.add_argument("--file", nargs=3, metavar=("ASSET","DATE","HOUR"))
    ap.add_argument("--quick", "-q", action="store_true")
    ap.add_argument("--progress-every", type=int, default=25,
                    help="Print progress every N files")
    ap.add_argument("--sample-k", type=int, default=64,
                    help="Per-column sample size per file chunk for median approximation (used with --stats)")

    args = ap.parse_args()

    # Single-file mode
    if args.file:
        a, d, h = args.file[0], args.file[1], int(args.file[2])
        files = _discover(args.features_dir, a, d)
        match = [f for f in files if f[2] == h]
        if not match:
            _fail(f"Not found: {a.upper()} {d} H{h:02d}")
            sys.exit(1)
        a, d, h, p = match[0]
        _header(f"{'Quick' if args.quick else 'Deep'} Audit — FAST — {a.upper()} {d} H{h:02d}")
        agg = defaultdict(ColAgg)
        r = _audit_one_file(a, d, h, p, args.stage, agg, sample_k_for_stats=(args.sample_k if args.stats else 0),
                            nan_only=args.nan_only)
        status = f"{r.rows:,} rows × {r.cols} cols  │  {r.size_mb:.2f} MB"
        if r.ok and not r.warnings:
            _ok(f"{status}  │  clean")
        elif r.ok:
            _warn(f"{status}  │  {len(r.warnings)} warnings")
        else:
            _fail(f"{status}  │  {len(r.errors)} errors, {len(r.warnings)} warnings")
        for e in r.errors:
            _fail(f"    {e}")
        for w in r.warnings:
            _warn(f"    {w}")
        if args.stats:
            _print_stats_from_agg(agg, args.stage)
        _nan_diagnosis_from_agg(agg, r.rows, [a])
        sys.exit(0 if r.ok else 1)

    # Multi-file mode
    _header("Unified Pipeline Feature Audit (FAST)  (S0 → S5)")
    files = _discover(args.features_dir, args.asset, args.date)
    if not files:
        _warn(f"No s5_features files found in {args.features_dir}")
        sys.exit(0)

    _kv("Dir", args.features_dir, i=2)
    _kv("Files", str(len(files)), i=2)
    if args.stage:
        _kv("Stage filter", args.stage, i=2)

    results: List[FileAudit] = []
    agg: Dict[str, ColAgg] = defaultdict(ColAgg)

    n_ok = n_warn = n_fail = 0
    total_rows = 0
    total_mb = 0.0

    t0 = time.time()
    for i, (asset, date_str, hour, path) in enumerate(files, 1):
        r = _audit_one_file(
            asset, date_str, hour, path,
            stage_filter=args.stage,
            agg=agg,
            sample_k_for_stats=(args.sample_k if args.stats else 0),
            nan_only=args.nan_only,
        )
        results.append(r)
        total_rows += r.rows
        total_mb += r.size_mb

        label = f"{asset.upper()} {date_str} H{hour:02d}"
        if r.ok and not r.warnings:
            n_ok += 1
            if args.expand:
                print(f"  {_G}{label}{_RST}")
        elif r.ok:
            n_warn += 1
            print(f"  {_Y}{label}  │  {len(r.warnings)}w{_RST}")
            if args.expand:
                for w in r.warnings:
                    print(f"      {_Y}{w}{_RST}")
        else:
            n_fail += 1
            print(f"  {_R}{label}  │  {len(r.errors)}e {len(r.warnings)}w{_RST}")
            if args.expand:
                for e in r.errors:
                    print(f"      {_R}{e}{_RST}")
                for w in r.warnings:
                    print(f"      {_Y}{w}{_RST}")

        if args.progress_every > 0 and (i % args.progress_every == 0 or i == len(files)):
            dt = time.time() - t0
            rate = i / dt if dt > 0 else 0.0
            eta = (len(files) - i) / rate if rate > 0 else float("inf")
            print(f"  {_D}[progress]{_RST} {i}/{len(files)} files  |  {rate:.2f} files/s  |  ETA ~ {eta/60:.1f} min")

    print(f"\n  {_G}{n_ok} passed{_RST}  {_Y}{n_warn} warnings-only{_RST}  {_R}{n_fail} errors{_RST}  (of {len(results)} files)")

    if not args.nan_only:
        _aggregate_issues(results)

    _nan_diagnosis_from_agg(agg, total_rows, sorted(set(a for a,_,_,_ in files)))

    if args.stats:
        _print_stats_from_agg(agg, args.stage)

    _header("Audit Summary")
    _kv("Files", str(len(results)), i=2)
    _kv("Total rows", f"{total_rows:,}", i=2)
    _kv("Total size", f"{total_mb:.1f} MB", i=2)
    _kv("Columns/file", f"~{results[0].cols if results else 0:,}", i=2)

    if n_fail == 0 and n_warn == 0:
        print(f"\n  {_G}{_B}All checks passed{_RST}\n")
    elif n_fail == 0:
        print(f"\n  {_Y}No errors. Warnings are structural (rolling warmup / sparse data).{_RST}\n")
    else:
        print(f"\n  {_R}{_B}{n_fail} file(s) with errors — see above.{_RST}\n")

    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()