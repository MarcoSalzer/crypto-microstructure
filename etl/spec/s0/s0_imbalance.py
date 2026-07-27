# etl/spec/s0/s0_imbalance.py
# ==============================================================================
# S0 Feature Specs: Imbalance (Depth Imbalance)
#
# Binance-only pipeline, BTC/ETH/BNB.
# Source: lobdeep (L2 orderbook snapshots)
#
# CONTENTS (all 1s bucket):
#   - depth_imbalance_struct{50|100}_{fut|spot}_1s       (4 features)
#
# DESIGN NOTES:
#   - Only STRUCTURAL imbalance (struct50/100) is in the target set.
#     Fixed BPS imbalance (1/2/5/10bps) has been removed from the pipeline.
#   - Imbalance = (bid_notional - ask_notional) / (bid_notional + ask_notional)
#     within the structural window [0, alpha * bps_sym].
#   - Range [-1, 1]. NaN if both sides are 0 or either side is NaN.
#   - struct75 has been removed (only struct50/100 remain).
#   - Previously lived in s0_bookshape.py under "Book Shape" group.
#
# Clean S0 Contract:
#   params: market_scope, resample, alpha
#   No venue_scope, window_s, agg, depth_mode.
#
# Feature ID block: 400–403
#
# ==============================================================================

from typing import List
from etl.spec import FeatureSpec, Dep


def _dep_lobdeep(market: str):
    return (Dep("source:lobdeep", match_params=("market_scope",)),)


S0_IMBALANCE_FEATURES: List[FeatureSpec] = [

    # =========================================================================
    # STRUCTURAL IMBALANCE: struct50 (alpha=0.5)
    # =========================================================================
    FeatureSpec(
        name="depth_imbalance_struct50_fut_1s",
        stage="S0",
        operator="depth_bps.imbalance_struct_alpha",
        params={"market_scope": "Futures", "resample": "1s", "alpha": "0.5"},
        label="Depth Imbalance struct50% (Futures) [1s]",
        group="Imbalance",
        description=(
            "Depth imbalance (bid - ask) / (bid + ask) within 0 to 0.5*bps_sym. "
            "Range [-1, 1]. NaN if denom == 0."
        ),
        depends_on=_dep_lobdeep("Futures"),
        feature_id=36,
    ),
    FeatureSpec(
        name="depth_imbalance_struct50_spot_1s",
        stage="S0",
        operator="depth_bps.imbalance_struct_alpha",
        params={"market_scope": "Spot", "resample": "1s", "alpha": "0.5"},
        label="Depth Imbalance struct50% (Spot) [1s]",
        group="Imbalance",
        description=(
            "Depth imbalance (bid - ask) / (bid + ask) within 0 to 0.5*bps_sym. "
            "Range [-1, 1]. NaN if denom == 0."
        ),
        depends_on=_dep_lobdeep("Spot"),
        feature_id=37,
    ),

    # =========================================================================
    # STRUCTURAL IMBALANCE: struct100 (alpha=1.0)
    # =========================================================================
    FeatureSpec(
        name="depth_imbalance_struct100_fut_1s",
        stage="S0",
        operator="depth_bps.imbalance_struct_alpha",
        params={"market_scope": "Futures", "resample": "1s", "alpha": "1.0"},
        label="Depth Imbalance struct100% (Futures) [1s]",
        group="Imbalance",
        description=(
            "Depth imbalance (bid - ask) / (bid + ask) within 0 to 1.0*bps_sym. "
            "Range [-1, 1]. NaN if denom == 0."
        ),
        depends_on=_dep_lobdeep("Futures"),
        feature_id=38,
    ),
    FeatureSpec(
        name="depth_imbalance_struct100_spot_1s",
        stage="S0",
        operator="depth_bps.imbalance_struct_alpha",
        params={"market_scope": "Spot", "resample": "1s", "alpha": "1.0"},
        label="Depth Imbalance struct100% (Spot) [1s]",
        group="Imbalance",
        description=(
            "Depth imbalance (bid - ask) / (bid + ask) within 0 to 1.0*bps_sym. "
            "Range [-1, 1]. NaN if denom == 0."
        ),
        depends_on=_dep_lobdeep("Spot"),
        feature_id=39,
    ),
]