# etl/spec/s6/s6_cross_asset.py
# ==============================================================================
# S6 Cross-Asset Features — Multi-Pair (BTC ↔ ETH ↔ BNB)
# ==============================================================================
# Overview:
#   Stage 6 is structurally distinct from S0–S5: it receives N asset DataFrames
#   (BTC_S5, ETH_S5, BNB_S5), merges them on timestamp, and produces a single
#   DataFrame containing only cross-asset comparison features.
#
#   After the merge all upstream columns carry an asset suffix:
#       <col_name>_btc   (from BTC S5 output)
#       <col_name>_eth   (from ETH S5 output)
#       <col_name>_bnb   (from BNB S5 output)
#
#   Three permitted forms (enforced throughout):
#       1. bps-based depth  — all book quantities in bps, never raw notional
#       2. vol-normalised   — z / robust-z / MAD-scaled inputs from S3–S5
#       3. regime-relative  — percentile / rank / robust-z of own history
#
# ── Multi-Pair Architecture ─────────────────────────────────────────────────
#   3 asset pairs:  BTC↔ETH, BTC↔BNB, ETH↔BNB
#   Each pair gets the same set of cross-asset feature templates, plus
#   directional lead-lag features for both directions within each pair.
#
#   Feature naming convention:
#     ca_{feature_core}_{pair_tag}
#     where pair_tag ∈ {btceth, btcbnb, ethbnb}
#
#   Intermediary columns (per-asset, not per-pair):
#     ca_{intermediary}_{asset}
#     e.g. ca_bps_mid_dev_spot_5bps_btc, ca_bps_spread_spot_bnb
#
# ── Feature Families ────────────────────────────────────────────────────────
#   Per pair (× 3 pairs):
#     F1  Taker Imbalance Spread               (4)
#     F2  Net Pressure Spread                   (4)
#     F3  Absorption Refill Differential        (4)
#     F4  Impact Efficiency Differential        (2)
#     F5  Queue Pressure Differential           (4)
#     F6  Absorption Break Shock Diff.          (2)
#     F7  Microprice Deviation Differential     (2 diffs, intermediaries shared)
#     F8  Net-Add / Net-Cancel Divergence       (5 incl. composite)
#     F10 Lead-Lag Taker Shock                  (6, both directions)
#     F11 Vacuum Score Shock Differential       (8)
#     F12 Pull Rate Shock Differential          (6)
#     F13 Net Pressure Shock Differential       (8)
#     F14 Impact Efficiency Shock Diff.         (4)
#     F15 Refill vs Pull Ratio Diff.            (4)
#     F16 Net Pressure Persist Diff.            (8)
#     F17 Taker Imbalance Persist Diff.         (4)
#     F18 Depth Slope Differential              (4)
#     F19 Depth Curvature Differential          (4)
#     F20 Spread BPS Differential               (2 diffs, intermediaries shared)
#     F21 Regime Divergence / Alignment         (4)
#     F22 Taker Imbalance Shock Differential    (2)
#     F23 Forward Return Spread                 (5)
#     F24 Range / Price Position Differential   (9)  ← NEW
#     F25 Basis / Funding Differential          (3)  ← NEW
#     F26 Activity Differential                 (2 diffs, intermediaries shared)  ← NEW
#
#   Per-asset intermediaries (× 3 assets):
#     bps_mid_dev: spot + fut  (2 per asset → 6 total)
#     bps_spread:  spot + fut  (2 per asset → 6 total)
#     z_trade_count_spot_300s  (1 per asset → 3 total)  ← NEW
#     z_avg_trade_size_spot_900s (1 per asset → 3 total) ← NEW
#
#   All feature_ids numbered sequentially from 6000.
#
# ── NaN policy ────────────────────────────────────────────────────────────────
#   cross_asset_diff : NaN if either input is NaN (propagated).
#   bps_mid_dev      : NaN where mid = 0.
#   cross_lag_corr   : NaN for first (window_s + lag - 1) rows.
#   robust_zscore    : NaN for first (window_s - 1) rows (rolling warmup).
#
# ==============================================================================

from __future__ import annotations

from typing import List, Tuple

from etl.spec import FeatureSpec, Dep


# ==============================================================================
# Configuration
# ==============================================================================

ASSET_PAIRS: List[Tuple[str, str]] = [
    ("btc", "eth"),
    ("btc", "bnb"),
    ("eth", "bnb"),
]

ALL_ASSETS: List[str] = ["btc", "eth", "bnb"]


# ==============================================================================
# Sequential ID counter
# ==============================================================================

_fid_counter: int = 5999


def _next_fid() -> int:
    global _fid_counter
    _fid_counter += 1
    return _fid_counter


def _pair_tag(a: str, b: str) -> str:
    return f"{a}{b}"


# ==============================================================================
# Factory: simple cross_asset_diff  (col_{a} − col_{b})
# ==============================================================================

def _make_diff(
    name_core: str,
    upstream_core: str,
    pair: Tuple[str, str],
    description: str = "",
) -> FeatureSpec:
    a, b = pair
    tag = _pair_tag(a, b)
    col_a = f"{upstream_core}_{a}"
    col_b = f"{upstream_core}_{b}"
    return FeatureSpec(
        name=f"ca_{name_core}_{tag}",
        stage="S6",
        operator="derived.cross_asset_diff",
        params={"market_scope": "Cross-Asset", "col_a": col_a, "col_b": col_b},
        label=f"CA {name_core} {tag.upper()}",
        group="Cross-Asset",
        description=description or f"{col_a} − {col_b}.",
        depends_on=(Dep(name=col_a, kind="col"), Dep(name=col_b, kind="col")),
        feature_id=_next_fid(),
    )


# ==============================================================================
# Factory: per-asset intermediary (bps_mid_dev / bps_spread)
# ==============================================================================

def _make_bps_mid_dev(market: str, depth: str, asset: str) -> FeatureSpec:
    lwp_col = f"lwp_mid_{depth}bps_{market}_1s_{asset}"
    mid_col = f"mid_{market}_1s_{asset}"
    return FeatureSpec(
        name=f"ca_bps_mid_dev_{market}_{depth}bps_{asset}",
        stage="S6",
        operator="derived.bps_mid_dev",
        params={"market_scope": market.capitalize(), "asset": asset,
                "lwp_col": lwp_col, "mid_col": mid_col},
        label=f"BPS Mid Dev {market} {depth}bps {asset.upper()} (intermediary)",
        group="Cross-Asset-Intermediary",
        description=f"({lwp_col} − {mid_col}) / {mid_col} * 10000.",
        depends_on=(Dep(name=lwp_col, kind="col"), Dep(name=mid_col, kind="col")),
        feature_id=_next_fid(),
    )


def _make_bps_spread(market: str, asset: str) -> FeatureSpec:
    spread_col = f"spread_{market}_1s_{asset}"
    mid_col = f"mid_{market}_1s_{asset}"
    return FeatureSpec(
        name=f"ca_bps_spread_{market}_{asset}",
        stage="S6",
        operator="derived.bps_spread",
        params={"market_scope": market.capitalize(), "asset": asset,
                "spread_col": spread_col, "mid_col": mid_col},
        label=f"BPS Spread {market} {asset.upper()} (intermediary)",
        group="Cross-Asset-Intermediary",
        description=f"{spread_col} / {mid_col} * 10000.",
        depends_on=(Dep(name=spread_col, kind="col"), Dep(name=mid_col, kind="col")),
        feature_id=_next_fid(),
    )


# ==============================================================================
# Factory: per-asset robust-z intermediary (F26 Activity)
# ==============================================================================

def _make_robust_z_intermediary(
    name_suffix: str,
    input_col_core: str,
    window_s: int,
    asset: str,
    zscore_clip: float | None = None,
) -> FeatureSpec:
    """
    Compute a rolling robust z-score on a per-asset upstream column.
    Used to normalise raw counts (trade_count, avg_trade_size) before
    cross-asset differencing, since raw values are not comparable across assets
    (BTC at ~$85k has structurally different trade sizes than ETH at ~$2k).

    zscore_clip: if set, clips the z-score output to [-zscore_clip, +zscore_clip].
    Use for depth slope/curvature whose book can be near-constant (MAD ≈ 0).
    """
    input_col = f"{input_col_core}_{asset}"
    params: dict = {
        "market_scope": "Cross-Asset",
        "asset": asset,
        "input_col": input_col,
        "window_s": str(window_s),
    }
    if zscore_clip is not None:
        params["zscore_clip"] = str(zscore_clip)
    clip_note = f"  Clipped to ±{zscore_clip}." if zscore_clip else ""
    return FeatureSpec(
        name=f"ca_z_{name_suffix}_{asset}",
        stage="S6",
        operator="derived.robust_zscore",
        params=params,
        label=f"Robust Z {name_suffix} {asset.upper()} (intermediary)",
        group="Cross-Asset-Intermediary",
        description=(
            f"Robust z-score of {input_col} over {window_s}s window. "
            f"NaN for first {window_s - 1} rows (rolling warmup).{clip_note}"
        ),
        depends_on=(Dep(name=input_col, kind="col"),),
        feature_id=_next_fid(),
    )


# ==============================================================================
# Factory: intermediary diff (cross_asset_diff on intra-S6 intermediary cols)
# ==============================================================================

def _make_intermediary_diff(
    name_core: str,
    intermediary_prefix: str,
    pair: Tuple[str, str],
) -> FeatureSpec:
    a, b = pair
    tag = _pair_tag(a, b)
    col_a = f"{intermediary_prefix}_{a}"
    col_b = f"{intermediary_prefix}_{b}"
    return FeatureSpec(
        name=f"ca_{name_core}_{tag}",
        stage="S6",
        operator="derived.cross_asset_diff",
        params={"market_scope": "Cross-Asset", "col_a": col_a, "col_b": col_b},
        label=f"CA {name_core} {tag.upper()}",
        group="Cross-Asset",
        description=f"{col_a} − {col_b}.",
        depends_on=(Dep(name=col_a, kind="col"), Dep(name=col_b, kind="col")),
        feature_id=_next_fid(),
    )


# ==============================================================================
# Factory: composite (derived.sub on two intra-S6 pair features)
# ==============================================================================

def _make_composite_sub(
    name_core: str,
    dep_a_core: str,
    dep_b_core: str,
    pair: Tuple[str, str],
) -> FeatureSpec:
    tag = _pair_tag(*pair)
    col_a = f"ca_{dep_a_core}_{tag}"
    col_b = f"ca_{dep_b_core}_{tag}"
    return FeatureSpec(
        name=f"ca_{name_core}_{tag}",
        stage="S6",
        operator="derived.sub",
        params={"market_scope": "Cross-Asset",
                "input_col_a": col_a, "input_col_b": col_b},
        label=f"CA {name_core} {tag.upper()}",
        group="Cross-Asset",
        description=f"{col_a} − {col_b}. Composite intra-S6 difference.",
        depends_on=(Dep(name=col_a, kind="col"), Dep(name=col_b, kind="col")),
        feature_id=_next_fid(),
    )


# ==============================================================================
# Factory: lead-lag cross-correlation (directional)
# ==============================================================================

def _make_lag_corr(
    lead_asset: str,
    lag_asset: str,
    lead_upstream: str,
    lag_upstream: str,
    lag_s: int,
    window_s: int = 60,
) -> FeatureSpec:
    pair_tag = _pair_tag(lead_asset, lag_asset)
    lead_col = f"{lead_upstream}_{lead_asset}"
    lag_col = f"{lag_upstream}_{lag_asset}"
    return FeatureSpec(
        name=f"ca_lag_corr_{lead_asset}_taker_lead_{lag_asset}_ret_{lag_s}s",
        stage="S6",
        operator="derived.cross_lag_corr",
        params={"market_scope": "Cross-Asset",
                "lead_col": lead_col, "lag_col": lag_col,
                "lag_s": str(lag_s), "window_s": str(window_s)},
        label=f"Lag Corr {lead_asset.upper()} lead {lag_asset.upper()} {lag_s}s",
        group="Cross-Asset",
        description=(
            f"Rolling {window_s}s correlation of {lead_col}[t−{lag_s}] "
            f"with {lag_col}[t]."
        ),
        depends_on=(Dep(name=lead_col, kind="col"), Dep(name=lag_col, kind="col")),
        feature_id=_next_fid(),
    )


# ==============================================================================
# Factory: regime flags (xor / align)
# ==============================================================================

def _make_regime(
    name_core: str,
    upstream_core: str,
    operator: str,
    pair: Tuple[str, str],
) -> FeatureSpec:
    a, b = pair
    tag = _pair_tag(a, b)
    col_a = f"{upstream_core}_{a}"
    col_b = f"{upstream_core}_{b}"
    return FeatureSpec(
        name=f"ca_{name_core}_{tag}",
        stage="S6",
        operator=operator,
        params={"market_scope": "Cross-Asset", "col_a": col_a, "col_b": col_b},
        label=f"CA {name_core} {tag.upper()}",
        group="Cross-Asset",
        description=f"{operator.split('.')[-1]}({col_a}, {col_b}).",
        depends_on=(Dep(name=col_a, kind="col"), Dep(name=col_b, kind="col")),
        feature_id=_next_fid(),
    )


# ==============================================================================
# TEMPLATE DEFINITIONS
# ==============================================================================
# Each tuple: (feature_name_core, upstream_column_core)
# The factory appends _{asset} to the upstream_column_core per pair side.

# ── Phase 1: Direct z-differences ────────────────────────────────────────────

_PHASE1_TEMPLATES = [
    # F1: Taker Imbalance
    ("taker_imb_spot_spread_15s",           "z_taker_imbalance_spot_15s"),
    ("taker_imb_spot_spread_60s",           "z_taker_imbalance_spot_60s"),
    ("taker_imb_fut_spread_15s",            "z_taker_imbalance_fut_15s"),
    ("taker_imb_fut_spread_60s",            "z_taker_imbalance_fut_60s"),
    # F2: Net Pressure
    ("net_pressure_spot_5bps_spread_60s",   "net_pressure_robust_z_spot_5bps_60s"),
    ("net_pressure_spot_10bps_spread_60s",  "net_pressure_robust_z_spot_10bps_60s"),
    ("net_pressure_spot_5bps_spread_300s",  "net_pressure_robust_z_spot_5bps_300s"),
    ("net_pressure_fut_5bps_spread_60s",    "net_pressure_robust_z_fut_5bps_60s"),
    # F3: Absorption Refill
    # NOTE: absorb_refill_*_spread_1s removed (2026-05): 1-second window
    # produces 90-93% NaN — absorption/refill events are too sparse at 1s
    # granularity for a meaningful cross-asset diff.
    # F4: Impact Efficiency
    ("impact_eff_fut_spread_15s",           "z_impact_per_signed_fut_15s"),
    ("impact_eff_fut_spread_60s",           "z_impact_per_signed_fut_60s"),
    # F5: Queue Pressure
    ("queue_pressure_spot_1bps_spread_15s", "z_queue_pressure_spot_1bps_15s"),
    ("queue_pressure_spot_1bps_spread_60s", "z_queue_pressure_spot_1bps_60s"),
    ("queue_pressure_fut_1bps_spread_15s",  "z_queue_pressure_fut_1bps_15s"),
    ("queue_pressure_fut_1bps_spread_60s",  "z_queue_pressure_fut_1bps_60s"),
    # F6: Absorption Break Shock
    ("absorb_break_shock_spot_spread_15s",  "absorption_break_shock_spot_15s"),
    ("absorb_break_shock_fut_spread_15s",   "absorption_break_shock_fut_15s"),
]

# ── Phase 2: Net-Add / Net-Cancel (simple diffs) ────────────────────────────

_PHASE2_NET_ADD_CANCEL_TEMPLATES = [
    ("net_add_spot_5bps_spread_15s",    "net_add_robust_z_spot_5bps_15s"),
    ("net_cancel_spot_5bps_spread_15s", "net_cancel_robust_z_spot_5bps_15s"),
    ("net_add_spot_5bps_spread_60s",    "net_add_robust_z_spot_5bps_60s"),
    ("net_cancel_spot_5bps_spread_60s", "net_cancel_robust_z_spot_5bps_60s"),
]

# ── Round 1: Shock / Persist / Depth ─────────────────────────────────────────

_ROUND1_TEMPLATES = [
    # F11: Vacuum Score Shock
    ("vacuum_shock_spot_5bps_spread_15s",   "vacuum_score_shock_spot_5bps_15s"),
    ("vacuum_shock_spot_5bps_spread_60s",   "vacuum_score_shock_spot_5bps_60s"),
    ("vacuum_shock_spot_10bps_spread_15s",  "vacuum_score_shock_spot_10bps_15s"),
    ("vacuum_shock_spot_10bps_spread_60s",  "vacuum_score_shock_spot_10bps_60s"),
    ("vacuum_shock_fut_5bps_spread_15s",    "vacuum_score_shock_fut_5bps_15s"),
    ("vacuum_shock_fut_5bps_spread_60s",    "vacuum_score_shock_fut_5bps_60s"),
    ("vacuum_shock_fut_10bps_spread_15s",   "vacuum_score_shock_fut_10bps_15s"),
    ("vacuum_shock_fut_10bps_spread_60s",   "vacuum_score_shock_fut_10bps_60s"),
    # F12: Pull Rate Shock
    ("pull_rate_shock_spot_5bps_spread_15s",  "pull_rate_shock_spot_5bps_15s"),
    ("pull_rate_shock_spot_5bps_spread_60s",  "pull_rate_shock_spot_5bps_60s"),
    ("pull_rate_shock_spot_10bps_spread_15s", "pull_rate_shock_spot_10bps_15s"),
    ("pull_rate_shock_fut_5bps_spread_15s",   "pull_rate_shock_fut_5bps_15s"),
    ("pull_rate_shock_fut_5bps_spread_60s",   "pull_rate_shock_fut_5bps_60s"),
    ("pull_rate_shock_fut_10bps_spread_15s",  "pull_rate_shock_fut_10bps_15s"),
    # F13: Net Pressure Shock
    ("net_pressure_shock_spot_2bps_spread_15s",  "net_pressure_shock_spot_2bps_15s"),
    ("net_pressure_shock_spot_5bps_spread_15s",  "net_pressure_shock_spot_5bps_15s"),
    ("net_pressure_shock_spot_5bps_spread_60s",  "net_pressure_shock_spot_5bps_60s"),
    ("net_pressure_shock_spot_10bps_spread_60s", "net_pressure_shock_spot_10bps_60s"),
    ("net_pressure_shock_fut_2bps_spread_15s",   "net_pressure_shock_fut_2bps_15s"),
    ("net_pressure_shock_fut_5bps_spread_15s",   "net_pressure_shock_fut_5bps_15s"),
    ("net_pressure_shock_fut_5bps_spread_60s",   "net_pressure_shock_fut_5bps_60s"),
    ("net_pressure_shock_fut_10bps_spread_60s",  "net_pressure_shock_fut_10bps_60s"),
    # F14: Impact Efficiency Shock
    ("impact_shock_spot_spread_15s",  "impact_per_signed_shock_spot_15s"),
    ("impact_shock_spot_spread_60s",  "impact_per_signed_shock_spot_60s"),
    ("impact_shock_fut_spread_15s",   "impact_per_signed_shock_fut_15s"),
    ("impact_shock_fut_spread_60s",   "impact_per_signed_shock_fut_60s"),
    # F15: Refill vs Pull — see _REFILL_DIFF_TEMPLATES below (uses S5 z-scores).
    # F16: Net Pressure Persistence
    ("net_pressure_persist_spot_5bps_spread_60s",   "net_pressure_persist_spot_5bps_60s"),
    ("net_pressure_persist_spot_5bps_spread_300s",  "net_pressure_persist_spot_5bps_300s"),
    ("net_pressure_persist_spot_10bps_spread_60s",  "net_pressure_persist_spot_10bps_60s"),
    ("net_pressure_persist_fut_5bps_spread_60s",    "net_pressure_persist_fut_5bps_60s"),
    ("net_pressure_persist_fut_5bps_spread_300s",   "net_pressure_persist_fut_5bps_300s"),
    ("net_pressure_persist_fut_10bps_spread_60s",   "net_pressure_persist_fut_10bps_60s"),
    # F17: Taker Imbalance Persistence
    ("taker_imb_persist_spot_spread_15s", "taker_imbalance_spot_persist_15s"),
    ("taker_imb_persist_spot_spread_60s", "taker_imbalance_spot_persist_60s"),
    ("taker_imb_persist_fut_spread_15s",  "taker_imbalance_fut_persist_15s"),
    ("taker_imb_persist_fut_spread_60s",  "taker_imbalance_fut_persist_60s"),
    # F18/F19: Depth Slope + Curvature — moved to z-score intermediary approach.
    # Raw depth_slope/curvature values are in asset-native units (BTC book ≠ ETH
    # book in scale) → cross-asset diffs are meaningless and extreme in every file.
    # See _DEPTH_Z_INTERMEDIARIES + _DEPTH_DIFF_TEMPLATES below.
    # F22: Taker Imbalance Shock Differential
    # Both variants point directly to the S3 source columns.
    # ofi_shock_15s (S4 passthrough alias) has been removed — no indirection needed.
    ("taker_imb_shock_spread_15s", "taker_imbalance_shock_fut_15s"),
    ("taker_imb_shock_spread_5s",  "taker_imbalance_shock_fut_5s"),
]

# ── F23: Forward Return Spread ───────────────────────────────────────────────
# Horizons from S1 ret_fwd features.  Positive = asset A outperforms asset B.
# These are label-like features useful for relative-outperformance analysis.

_RET_FWD_TEMPLATES = [
    ("ret_fwd_spread_1s",   "ret_fwd_1s"),
    ("ret_fwd_spread_15s",  "ret_fwd_15s"),
    ("ret_fwd_spread_60s",  "ret_fwd_60s"),
    ("ret_fwd_spread_300s", "ret_fwd_300s"),
    ("ret_fwd_spread_900s", "ret_fwd_900s"),
]

# ── F24 (NEW): Range / Price Position Differential ───────────────────────────
# range_pos_day and dist_to_day_high/low_bps are the #1 and #2 features
# globally in both assets. Cross-asset comparison captures relative mean-
# reversion potential: "BTC at 95% of day-range while ETH at 40%" is a
# powerful relative-value signal that single-asset features cannot encode.
#
# range_pos_day: [0,1], already scale-invariant → directly comparable.
# dist_to_day_high/low_bps: in bps, already normalised → comparable.
# range_pct: in bps, already normalised → comparable.
# day_range_bps: in bps → comparable.

_RANGE_TEMPLATES = [
    # Position within day-range (spot + fut)
    ("range_pos_day_spread_fut",           "range_pos_day_fut"),
    ("range_pos_day_spread_spot",          "range_pos_day_spot"),
    # Distance to daily extremes (fut + spot)
    ("dist_to_day_high_bps_spread_fut",    "dist_to_day_high_bps_fut"),
    ("dist_to_day_high_bps_spread_spot",   "dist_to_day_high_bps_spot"),
    ("dist_to_day_low_bps_spread_fut",     "dist_to_day_low_bps_fut"),
    ("dist_to_day_low_bps_spread_spot",    "dist_to_day_low_bps_spot"),
    # Intra-window range (900s, spot + fut)
    ("range_pct_spread_spot_900s",         "range_pct_spot_900s"),
    ("range_pct_spread_fut_900s",          "range_pct_fut_900s"),
    # Day range magnitude (fut only — spot typically redundant with fut)
    ("day_range_bps_spread_fut",           "day_range_bps_fut"),
]

# ── F25 (NEW): Basis / Funding Differential ──────────────────────────────────
# basis_vwap_sf = (fut_vwap − spot_vwap) / spot_vwap * 10_000  (in bps).
# Already normalised in bps → directly comparable across assets.
# basis_btc − basis_eth > 0 signals higher BTC funding demand, often
# preceding directional divergence between the two assets.

_BASIS_TEMPLATES = [
    ("basis_vwap_sf_spread_1s",   "basis_vwap_sf_1s"),
    # basis_vwap_sf_spread_60s removed (2026-05): upstream basis_vwap_sf_60s
    # is 97.3% NaN → all-NaN in ~60% of hourly files.  The 60s rolling VWAP
    # is too sparse on 1-second tick data.  Keep 1s (instantaneous) and 300s only.
    ("basis_vwap_sf_spread_300s", "basis_vwap_sf_300s"),
]

# ── F26 (NEW): Activity Differential ─────────────────────────────────────────
# Raw trade_count and avg_trade_size are NOT comparable across assets (BTC
# at ~$85k has structurally different tick sizes, lot sizes, and trade counts
# than ETH at ~$2k or BNB at ~$600). We first compute a rolling robust
# z-score per asset (intermediary), then diff the z-scores.
#
# Intermediaries: defined as (name_suffix, input_col_core, window_s)
# Diffs:          defined as (name_core, intermediary_prefix)

_ACTIVITY_Z_INTERMEDIARIES = [
    ("trade_count_spot_300s",    "trade_count_spot_300s",    900),
    ("avg_trade_size_spot_900s", "avg_trade_size_spot_900s", 900),
]

_ACTIVITY_DIFF_TEMPLATES = [
    ("activity_trade_count_spot_300s_spread",    "ca_z_trade_count_spot_300s"),
    ("activity_avg_trade_size_spot_900s_spread", "ca_z_avg_trade_size_spot_900s"),
]

# ── F15: Refill vs Pull — direct diff on S5 pre-computed z-scores ────────────
#   S5 already ships z_refill_vs_pull_* (robust z-score, computed at ingestion).
#   Using those directly avoids a redundant S6 intermediary and the all-NaN
#   problem that occurred when the raw refill_vs_pull_* columns were empty.

_REFILL_DIFF_TEMPLATES = [
    # (feature_name_core, s5_upstream_col_core)
    # upstream col in merged df: z_refill_vs_pull_spot_1bps_15s_{asset}
    ("refill_vs_pull_spot_1bps_spread_15s", "z_refill_vs_pull_spot_1bps_15s"),
    ("refill_vs_pull_spot_2bps_spread_60s", "z_refill_vs_pull_spot_2bps_60s"),
    ("refill_vs_pull_fut_1bps_spread_15s",  "z_refill_vs_pull_fut_1bps_15s"),
    ("refill_vs_pull_fut_2bps_spread_60s",  "z_refill_vs_pull_fut_2bps_60s"),
]

# ── F18/F19 (revised): Depth Slope + Curvature — robust-z intermediary + diff
#   net_pressure_depth_slope/curvature are in asset-native units →
#   not directly comparable across assets. Z-score per asset first.
#   Window: 300s (robust history for normalisation over a 5-min horizon).

_DEPTH_Z_INTERMEDIARIES = [
    # (name_suffix, input_col_core, window_s, zscore_clip)
    # zscore_clip=10: depth slope/curvature can have near-zero MAD when the book
    # is constant → exploding z-scores. Clip at ±10 before cross-asset diff.
    ("depth_slope_spot_15s",     "net_pressure_depth_slope_spot_15s",     300, 10.0),
    ("depth_slope_spot_60s",     "net_pressure_depth_slope_spot_60s",     300, 10.0),
    ("depth_slope_fut_15s",      "net_pressure_depth_slope_fut_15s",      300, 10.0),
    ("depth_slope_fut_60s",      "net_pressure_depth_slope_fut_60s",      300, 10.0),
    ("depth_curvature_spot_15s", "net_pressure_depth_curvature_spot_15s", 300, 10.0),
    ("depth_curvature_spot_60s", "net_pressure_depth_curvature_spot_60s", 300, 10.0),
    ("depth_curvature_fut_15s",  "net_pressure_depth_curvature_fut_15s",  300, 10.0),
    ("depth_curvature_fut_60s",  "net_pressure_depth_curvature_fut_60s",  300, 10.0),
]

_DEPTH_DIFF_TEMPLATES = [
    # (feature_name_core, intermediary_col_prefix)
    ("depth_slope_spot_spread_15s",     "ca_z_depth_slope_spot_15s"),
    ("depth_slope_spot_spread_60s",     "ca_z_depth_slope_spot_60s"),
    ("depth_slope_fut_spread_15s",      "ca_z_depth_slope_fut_15s"),
    ("depth_slope_fut_spread_60s",      "ca_z_depth_slope_fut_60s"),
    ("depth_curvature_spot_spread_15s", "ca_z_depth_curvature_spot_15s"),
    ("depth_curvature_spot_spread_60s", "ca_z_depth_curvature_spot_60s"),
    ("depth_curvature_fut_spread_15s",  "ca_z_depth_curvature_fut_15s"),
    ("depth_curvature_fut_spread_60s",  "ca_z_depth_curvature_fut_60s"),
]

# ── Regime templates ──────────────────────────────────────────────────────────

_REGIME_TEMPLATES = [
    ("regime_xor_60s",    "breakout_regime_flag_60s",  "derived.regime_xor"),
    ("regime_xor_300s",   "breakout_regime_flag_300s", "derived.regime_xor"),
    ("regime_align_60s",  "breakout_regime_flag_60s",  "derived.regime_align"),
    ("regime_align_300s", "breakout_regime_flag_300s", "derived.regime_align"),
]

# ── Lead-lag config ───────────────────────────────────────────────────────────

_LAG_VALUES = [1, 3, 5]
_LEAD_UPSTREAM = "z_taker_imbalance_spot_15s"
_LAG_UPSTREAM = "ret_mid_spot_1s"


# ==============================================================================
# Feature Generation
# ==============================================================================

def _generate_all_features() -> List[FeatureSpec]:
    """Generate all S6 cross-asset features for all pairs, numbered from 6000."""
    specs: List[FeatureSpec] = []

    # ------------------------------------------------------------------
    # 1. Per-asset intermediaries (shared across pairs, generated once)
    # ------------------------------------------------------------------

    # bps_mid_dev: spot + fut, depth=5
    for asset in ALL_ASSETS:
        specs.append(_make_bps_mid_dev("spot", "5", asset))
        specs.append(_make_bps_mid_dev("fut", "5", asset))

    # bps_spread: spot + fut
    for asset in ALL_ASSETS:
        specs.append(_make_bps_spread("spot", asset))
        specs.append(_make_bps_spread("fut", asset))

    # F26: Activity robust-z intermediaries
    for name_suffix, input_col_core, window_s in _ACTIVITY_Z_INTERMEDIARIES:
        for asset in ALL_ASSETS:
            specs.append(_make_robust_z_intermediary(
                name_suffix, input_col_core, window_s, asset,
            ))

    # F18/F19 (revised): Depth Slope + Curvature — per-asset robust-z intermediaries
    for tpl in _DEPTH_Z_INTERMEDIARIES:
        name_suffix, input_col_core, window_s = tpl[0], tpl[1], tpl[2]
        zscore_clip = tpl[3] if len(tpl) > 3 else None
        for asset in ALL_ASSETS:
            specs.append(_make_robust_z_intermediary(
                name_suffix, input_col_core, window_s, asset,
                zscore_clip=zscore_clip,
            ))

    # ------------------------------------------------------------------
    # 2. Per-pair features
    # ------------------------------------------------------------------
    for pair in ASSET_PAIRS:

        # Phase 1: direct z-diffs
        for name_core, upstream_core in _PHASE1_TEMPLATES:
            specs.append(_make_diff(name_core, upstream_core, pair))

        # Phase 2: intermediary diffs (microprice dev)
        specs.append(_make_intermediary_diff(
            "microprice_dev_spot_5bps_spread_1s",
            "ca_bps_mid_dev_spot_5bps", pair,
        ))
        specs.append(_make_intermediary_diff(
            "microprice_dev_fut_5bps_spread_1s",
            "ca_bps_mid_dev_fut_5bps", pair,
        ))

        # Phase 2: net-add / net-cancel diffs
        for name_core, upstream_core in _PHASE2_NET_ADD_CANCEL_TEMPLATES:
            specs.append(_make_diff(name_core, upstream_core, pair))

        # Phase 2: composite (add minus cancel)
        specs.append(_make_composite_sub(
            "add_minus_cancel_spot_5bps_spread_15s",
            "net_add_spot_5bps_spread_15s",
            "net_cancel_spot_5bps_spread_15s",
            pair,
        ))

        # Phase 4: lead-lag cross-correlations (both directions)
        a, b = pair
        for lag in _LAG_VALUES:
            specs.append(_make_lag_corr(a, b, _LEAD_UPSTREAM, _LAG_UPSTREAM, lag))
            specs.append(_make_lag_corr(b, a, _LEAD_UPSTREAM, _LAG_UPSTREAM, lag))

        # Round 1: shock / persist diffs
        for name_core, upstream_core in _ROUND1_TEMPLATES:
            specs.append(_make_diff(name_core, upstream_core, pair))

        # F15: Refill vs Pull — diff on S5 pre-computed z-scores (direct upstream)
        for name_core, upstream_core in _REFILL_DIFF_TEMPLATES:
            specs.append(_make_diff(name_core, upstream_core, pair))

        # F18/F19 (revised): Depth Slope + Curvature — diff on z-score intermediaries
        for name_core, intermediary_prefix in _DEPTH_DIFF_TEMPLATES:
            specs.append(_make_intermediary_diff(name_core, intermediary_prefix, pair))

        # Round 2: spread BPS diffs (on intermediaries)
        specs.append(_make_intermediary_diff(
            "spread_bps_spot_spread_1s",
            "ca_bps_spread_spot", pair,
        ))
        specs.append(_make_intermediary_diff(
            "spread_bps_fut_spread_1s",
            "ca_bps_spread_fut", pair,
        ))

        # Round 2: regime flags
        for name_core, upstream_core, operator in _REGIME_TEMPLATES:
            specs.append(_make_regime(name_core, upstream_core, operator, pair))

        # F23: Forward Return Spreads
        for name_core, upstream_core in _RET_FWD_TEMPLATES:
            specs.append(_make_diff(name_core, upstream_core, pair))

        # F24 (NEW): Range / Price Position Differential
        for name_core, upstream_core in _RANGE_TEMPLATES:
            specs.append(_make_diff(name_core, upstream_core, pair))

        # F25 (NEW): Basis / Funding Differential
        for name_core, upstream_core in _BASIS_TEMPLATES:
            specs.append(_make_diff(name_core, upstream_core, pair))

        # F26 (NEW): Activity Differential (on z-score intermediaries)
        for name_core, intermediary_prefix in _ACTIVITY_DIFF_TEMPLATES:
            specs.append(_make_intermediary_diff(name_core, intermediary_prefix, pair))

    return specs


# ==============================================================================
# Module-level feature list (generated once at import time)
# ==============================================================================

S6_CROSS_ASSET_FEATURES: List[FeatureSpec] = _generate_all_features()