# etl/spec/s0/s0_activity.py
# ==============================================================================
# S0 Feature Specs: Activity
#
# Binance-only pipeline, BTC/ETH/BNB.
# Source: trades
#
# CONTENTS (all 1s bucket):
#   - trade_count: number of executed trades           (2 features)
#   - volume: executed base volume (sum of qty)        (2 features)
#   - notional: total executed notional (sum qty*px)   (2 features)
#                                                Total: 6 features
#
# DESIGN NOTES:
#   - volume and notional are unified under Activity because they measure
#     the overall level of market participation, not directional aggression.
#   - Directional taker flow (buy/sell splits, signed) lives in Aggression.
#   - Previously volume lived in s0_volume.py (now merged here).
#   - Previously notional lived in s0_aggression.py (now moved here).
#
# Clean S0 Contract:
#   params: market_scope, resample
#   No venue_scope, window_s, agg, depth_mode.
#
# Feature ID block: 9–12, 25–26
#
# ==============================================================================

from typing import List
from etl.spec import FeatureSpec, Dep


def _dep_trades(market: str):
    return (Dep("source:trades", match_params=("market_scope",)),)


S0_ACTIVITY_FEATURES: List[FeatureSpec] = [

    # =========================================================================
    # TRADE COUNT
    # =========================================================================
    FeatureSpec(
        name="trade_count_fut_1s",
        stage="S0",
        operator="trades.trade_count",
        params={"market_scope": "Futures", "resample": "1s"},
        label="Trade Count (Futures) [1s]",
        group="Activity",
        description="Number of executed trades in futures per 1s bucket.",
        depends_on=_dep_trades("Futures"),
        feature_id=0,
    ),
    FeatureSpec(
        name="trade_count_spot_1s",
        stage="S0",
        operator="trades.trade_count",
        params={"market_scope": "Spot", "resample": "1s"},
        label="Trade Count (Spot) [1s]",
        group="Activity",
        description="Number of executed trades in spot per 1s bucket.",
        depends_on=_dep_trades("Spot"),
        feature_id=1,
    ),

    # =========================================================================
    # VOLUME (base units: sum of qty)
    # =========================================================================
    FeatureSpec(
        name="volume_fut_1s",
        stage="S0",
        operator="trades.volume",
        params={"market_scope": "Futures", "resample": "1s"},
        label="Volume (Futures) [1s]",
        group="Activity",
        description="Executed base volume (sum of qty) per 1s bucket in Futures.",
        depends_on=_dep_trades("Futures"),
        feature_id=2,
    ),
    FeatureSpec(
        name="volume_spot_1s",
        stage="S0",
        operator="trades.volume",
        params={"market_scope": "Spot", "resample": "1s"},
        label="Volume (Spot) [1s]",
        group="Activity",
        description="Executed base volume (sum of qty) per 1s bucket in Spot.",
        depends_on=_dep_trades("Spot"),
        feature_id=3,
    ),

    # =========================================================================
    # NOTIONAL (quote units: sum of qty * price)
    # =========================================================================
    FeatureSpec(
        name="notional_fut_1s",
        stage="S0",
        operator="trades.notional",
        params={"market_scope": "Futures", "resample": "1s"},
        label="Total Notional (Futures) [1s]",
        group="Activity",
        description="Total executed notional (sum of qty*price) per 1s bucket in Futures.",
        depends_on=_dep_trades("Futures"),
        feature_id=4,
    ),
    FeatureSpec(
        name="notional_spot_1s",
        stage="S0",
        operator="trades.notional",
        params={"market_scope": "Spot", "resample": "1s"},
        label="Total Notional (Spot) [1s]",
        group="Activity",
        description="Total executed notional (sum of qty*price) per 1s bucket in Spot.",
        depends_on=_dep_trades("Spot"),
        feature_id=5,
    ),
]