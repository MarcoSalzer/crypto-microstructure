# etl/spec/s0/s0_bookshape.py
# ==============================================================================
# S0 Feature Specs: Bookshape (Depth Structure)
#
# Binance-only pipeline, BTC/ETH/BNB.
# Source: lobdeep (L2 orderbook snapshots)
#
# CONTENTS (all 1s bucket):
#   1. Metadata:       max_bps_bid, max_bps_ask                   (4 features)
#   2. Fixed Depth:    depth_notional {1|2|5|10}bps               (16 features)
#   3. Struct Depth:   depth_notional struct{50|100}               (8 features)
#   4. Max Liq Dist:   max_liq_distance fixed 10bps               (4 features)
#   5. Max Liq Dist:   max_liq_distance struct{50|100}            (8 features)
#                                                           Total: 40 features
#
# MOVED OUT:
#   - bps_sym → Price (price-regime primitive)
#   - All LWP → Price (output is a price level)
#   - depth_imbalance → Imbalance (s0_imbalance.py)
#
# REMOVED:
#   - struct75 variants (pipeline decision: only struct50/100)
#   - Fixed BPS imbalance (not in target feature set)
#
# Clean S0 Contract:
#   params: market_scope, resample, side, [bps_lo, bps_hi], [alpha]
#   No venue_scope, window_s, agg, depth_mode.
#
# Feature ID blocks:
#   Metadata (max_bps): 50–53
#   Fixed depth: 100–115
#   Struct depth: 300–307
#   Max liq distance fixed 5bps: 704–707
#   Max liq distance fixed 10bps: 700–703
#   Max liq distance struct: 710–717
#
# ==============================================================================

from typing import List
from etl.spec import FeatureSpec, Dep


def _dep_lobdeep(market: str):
    return (Dep("source:lobdeep", match_params=("market_scope",)),)


# ==============================================================================
# METADATA: max_bps_bid, max_bps_ask (4 features)
# ==============================================================================

S0_BOOKSHAPE_METADATA: List[FeatureSpec] = [
    FeatureSpec(
        name="max_bps_bid_fut_1s",
        stage="S0",
        operator="depth_bps.max_bps_side",
        params={"market_scope": "Futures", "resample": "1s", "side": "bid"},
        label="Max BPS Bid (Futures) [1s]",
        group="Bookshape",
        description="Max bid-side BPS coverage from lobdeep snapshot.",
        depends_on=_dep_lobdeep("Futures"),
        feature_id=18,
    ),
    FeatureSpec(
        name="max_bps_bid_spot_1s",
        stage="S0",
        operator="depth_bps.max_bps_side",
        params={"market_scope": "Spot", "resample": "1s", "side": "bid"},
        label="Max BPS Bid (Spot) [1s]",
        group="Bookshape",
        description="Max bid-side BPS coverage from lobdeep snapshot.",
        depends_on=_dep_lobdeep("Spot"),
        feature_id=19,
    ),
    FeatureSpec(
        name="max_bps_ask_fut_1s",
        stage="S0",
        operator="depth_bps.max_bps_side",
        params={"market_scope": "Futures", "resample": "1s", "side": "ask"},
        label="Max BPS Ask (Futures) [1s]",
        group="Bookshape",
        description="Max ask-side BPS coverage from lobdeep snapshot.",
        depends_on=_dep_lobdeep("Futures"),
        feature_id=20,
    ),
    FeatureSpec(
        name="max_bps_ask_spot_1s",
        stage="S0",
        operator="depth_bps.max_bps_side",
        params={"market_scope": "Spot", "resample": "1s", "side": "ask"},
        label="Max BPS Ask (Spot) [1s]",
        group="Bookshape",
        description="Max ask-side BPS coverage from lobdeep snapshot.",
        depends_on=_dep_lobdeep("Spot"),
        feature_id=21,
    ),
]


# ==============================================================================
# FIXED BPS DEPTH NOTIONAL (4 bps × 2 sides × 2 markets = 16 features)
# ==============================================================================

def _fixed_bps_depth_specs() -> List[FeatureSpec]:
    specs = []
    fid = 100

    for bps in (1, 2, 5, 10):
        for side in ("bid", "ask"):
            for market, market_label, market_key in (
                ("Futures", "Futures", "fut"),
                ("Spot", "Spot", "spot"),
            ):
                specs.append(FeatureSpec(
                    name=f"depth_notional_{side}_{bps}bps_{market_key}_1s",
                    stage="S0",
                    operator="depth_bps.notional_fixed_bps",
                    params={
                        "market_scope": market,
                        "resample": "1s",
                        "side": side,
                        "bps_lo": "0",
                        "bps_hi": str(bps),
                    },
                    label=f"Depth Notional {side.title()} 0-{bps}bps ({market_label}) [1s]",
                    group="Bookshape",
                    description=(
                        f"{side.title()}-side notional within 0-{bps} bps from mid "
                        f"(last snapshot per 1s bucket)."
                    ),
                    depends_on=_dep_lobdeep(market),
                    feature_id=fid,
                ))
                fid += 1
    return specs


# ==============================================================================
# STRUCTURAL DEPTH NOTIONAL (struct50/100 only, 2 alpha × 2 sides × 2 markets = 8)
# ==============================================================================

def _struct_depth_specs() -> List[FeatureSpec]:
    specs = []
    fid = 300

    for alpha, alpha_label in ((0.5, "50"), (1.0, "100")):
        for side in ("bid", "ask"):
            for market, market_label, market_key in (
                ("Futures", "Futures", "fut"),
                ("Spot", "Spot", "spot"),
            ):
                specs.append(FeatureSpec(
                    name=f"depth_notional_{side}_struct{alpha_label}_{market_key}_1s",
                    stage="S0",
                    operator="depth_bps.notional_struct_alpha",
                    params={
                        "market_scope": market,
                        "resample": "1s",
                        "side": side,
                        "alpha": str(alpha),
                    },
                    label=f"Depth Notional {side.title()} struct{alpha_label}% ({market_label}) [1s]",
                    group="Bookshape",
                    description=(
                        f"{side.title()}-side notional within 0 to {alpha}*bps_sym."
                    ),
                    depends_on=_dep_lobdeep(market),
                    feature_id=fid,
                ))
                fid += 1
    return specs


# ==============================================================================
# MAX LIQUIDITY DISTANCE — FIXED 10bps (2 sides × 2 markets = 4)
# ==============================================================================
# Distance (in bps) from mid to the price level with maximum notional (px*qty).

def _max_liq_distance_fixed_specs() -> List[FeatureSpec]:
    specs = []
    fid = 700

    for side in ("bid", "ask"):
        for market, market_label, market_key in (
            ("Futures", "Futures", "fut"),
            ("Spot", "Spot", "spot"),
        ):
            specs.append(FeatureSpec(
                name=f"max_liq_distance_{side}_10bps_{market_key}_1s",
                stage="S0",
                operator="depth_bps.max_liq_distance_fixed_bps",
                params={
                    "market_scope": market,
                    "resample": "1s",
                    "side": side,
                    "bps_lo": "0",
                    "bps_hi": "10",
                },
                label=f"Max Liquidity Distance {side.title()} 0-10bps ({market_label}) [1s]",
                group="Bookshape",
                description=(
                    f"BPS distance from mid to the level with maximum notional (px*qty) "
                    f"within 0-10 bps on {side} side."
                ),
                depends_on=_dep_lobdeep(market),
                feature_id=fid,
            ))
            fid += 1
    return specs



# ==============================================================================
# MAX LIQUIDITY DISTANCE — FIXED 5bps (2 sides × 2 markets = 4)
# ==============================================================================

def _max_liq_distance_5bps_specs() -> List[FeatureSpec]:
    specs = []
    fid = 704

    for side in ("bid", "ask"):
        for market, market_label, market_key in (
            ("Futures", "Futures", "fut"),
            ("Spot", "Spot", "spot"),
        ):
            specs.append(FeatureSpec(
                name=f"max_liq_distance_{side}_5bps_{market_key}_1s",
                stage="S0",
                operator="depth_bps.max_liq_distance_fixed_bps",
                params={
                    "market_scope": market,
                    "resample": "1s",
                    "side": side,
                    "bps_lo": "0",
                    "bps_hi": "5",
                },
                label=f"Max Liquidity Distance {side.title()} 0-5bps ({market_label}) [1s]",
                group="Bookshape",
                description=(
                    f"BPS distance from mid to the level with maximum notional (px*qty) "
                    f"within 0-5 bps on {side} side."
                ),
                depends_on=_dep_lobdeep(market),
                feature_id=fid,
            ))
            fid += 1
    return specs

# ==============================================================================
# MAX LIQUIDITY DISTANCE — STRUCTURAL (struct50/100, 2 alpha × 2 sides × 2 markets = 8)
# ==============================================================================

def _max_liq_distance_struct_specs() -> List[FeatureSpec]:
    specs = []
    fid = 710

    for alpha, alpha_label in ((0.5, "50"), (1.0, "100")):
        for side in ("bid", "ask"):
            for market, market_label, market_key in (
                ("Futures", "Futures", "fut"),
                ("Spot", "Spot", "spot"),
            ):
                specs.append(FeatureSpec(
                    name=f"max_liq_distance_{side}_struct{alpha_label}_{market_key}_1s",
                    stage="S0",
                    operator="depth_bps.max_liq_distance_struct_alpha",
                    params={
                        "market_scope": market,
                        "resample": "1s",
                        "side": side,
                        "alpha": str(alpha),
                    },
                    label=f"Max Liquidity Distance {side.title()} struct{alpha_label}% ({market_label}) [1s]",
                    group="Bookshape",
                    description=(
                        f"BPS distance from mid to the level with maximum notional (px*qty) "
                        f"within 0 to {alpha}*bps_sym on {side} side."
                    ),
                    depends_on=_dep_lobdeep(market),
                    feature_id=fid,
                ))
                fid += 1
    return specs


# ==============================================================================
# FINAL ASSEMBLY
# ==============================================================================

S0_BOOKSHAPE_FEATURES: List[FeatureSpec] = (
    S0_BOOKSHAPE_METADATA
    + _fixed_bps_depth_specs()
    + _struct_depth_specs()
    + _max_liq_distance_5bps_specs()
    + _max_liq_distance_fixed_specs()
    + _max_liq_distance_struct_specs()
)