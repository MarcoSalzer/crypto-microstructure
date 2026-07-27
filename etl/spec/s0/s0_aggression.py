# etl/spec/s0/s0_aggression.py
# ==============================================================================
# S0 Feature Specs: Aggression (Directional Taker Flow)
#
# Binance-only pipeline, BTC/ETH/BNB.
# Source: trades
#
# CONTENTS (all 1s bucket):
#   Volume (base units):   taker_buy_vol, taker_sell_vol, signed_vol   (6 features)
#   Notional (quote units): taker_buy_notional, taker_sell_notional,
#                           signed_notional                            (6 features)
#                                                               Total: 12 features
#
# DESIGN NOTES:
#   - Only DIRECTIONAL taker flow lives here (buy vs sell split, signed).
#   - Total volume/notional (undirected) moved to Activity group (s0_activity.py).
#   - No venue_scope, window_s, depth_mode.
#
# Feature ID block: 13–24
#
# ==============================================================================

from typing import List
from etl.spec import FeatureSpec, Dep


def _dep_trades(market: str):
    return (Dep("source:trades", match_params=("market_scope",)),)


S0_AGGRESSION_FEATURES: List[FeatureSpec] = [

    # =========================================================================
    # TAKER BUY VOLUME (base units)
    # =========================================================================
    FeatureSpec(
        name="taker_buy_vol_fut_1s",
        stage="S0",
        operator="trades.taker_buy_volume",
        params={"market_scope": "Futures", "resample": "1s"},
        label="Taker Buy Volume (Futures) [1s]",
        group="Aggression",
        description="Buyer-initiated (aggressive) executed base volume, summed per 1s bucket.",
        depends_on=_dep_trades("Futures"),
        feature_id=6,
    ),
    FeatureSpec(
        name="taker_buy_vol_spot_1s",
        stage="S0",
        operator="trades.taker_buy_volume",
        params={"market_scope": "Spot", "resample": "1s"},
        label="Taker Buy Volume (Spot) [1s]",
        group="Aggression",
        description="Buyer-initiated (aggressive) executed base volume, summed per 1s bucket.",
        depends_on=_dep_trades("Spot"),
        feature_id=7,
    ),

    # =========================================================================
    # TAKER BUY NOTIONAL (quote units)
    # =========================================================================
    FeatureSpec(
        name="taker_buy_notional_fut_1s",
        stage="S0",
        operator="trades.taker_buy_notional",
        params={"market_scope": "Futures", "resample": "1s"},
        label="Taker Buy Notional (Futures) [1s]",
        group="Aggression",
        description="Buyer-initiated executed notional (qty*price) summed per 1s bucket.",
        depends_on=_dep_trades("Futures"),
        feature_id=8,
    ),
    FeatureSpec(
        name="taker_buy_notional_spot_1s",
        stage="S0",
        operator="trades.taker_buy_notional",
        params={"market_scope": "Spot", "resample": "1s"},
        label="Taker Buy Notional (Spot) [1s]",
        group="Aggression",
        description="Buyer-initiated executed notional (qty*price) summed per 1s bucket.",
        depends_on=_dep_trades("Spot"),
        feature_id=9,
    ),

    # =========================================================================
    # TAKER SELL VOLUME (base units)
    # =========================================================================
    FeatureSpec(
        name="taker_sell_vol_fut_1s",
        stage="S0",
        operator="trades.taker_sell_volume",
        params={"market_scope": "Futures", "resample": "1s"},
        label="Taker Sell Volume (Futures) [1s]",
        group="Aggression",
        description="Seller-initiated (aggressive) executed base volume, summed per 1s bucket.",
        depends_on=_dep_trades("Futures"),
        feature_id=10,
    ),
    FeatureSpec(
        name="taker_sell_vol_spot_1s",
        stage="S0",
        operator="trades.taker_sell_volume",
        params={"market_scope": "Spot", "resample": "1s"},
        label="Taker Sell Volume (Spot) [1s]",
        group="Aggression",
        description="Seller-initiated (aggressive) executed base volume, summed per 1s bucket.",
        depends_on=_dep_trades("Spot"),
        feature_id=11,
    ),

    # =========================================================================
    # TAKER SELL NOTIONAL (quote units)
    # =========================================================================
    FeatureSpec(
        name="taker_sell_notional_fut_1s",
        stage="S0",
        operator="trades.taker_sell_notional",
        params={"market_scope": "Futures", "resample": "1s"},
        label="Taker Sell Notional (Futures) [1s]",
        group="Aggression",
        description="Seller-initiated executed notional (qty*price) summed per 1s bucket.",
        depends_on=_dep_trades("Futures"),
        feature_id=12,
    ),
    FeatureSpec(
        name="taker_sell_notional_spot_1s",
        stage="S0",
        operator="trades.taker_sell_notional",
        params={"market_scope": "Spot", "resample": "1s"},
        label="Taker Sell Notional (Spot) [1s]",
        group="Aggression",
        description="Seller-initiated executed notional (qty*price) summed per 1s bucket.",
        depends_on=_dep_trades("Spot"),
        feature_id=13,
    ),

    # =========================================================================
    # SIGNED VOLUME (base units): +qty buys, -qty sells
    # =========================================================================
    FeatureSpec(
        name="signed_vol_fut_1s",
        stage="S0",
        operator="trades.signed_volume",
        params={"market_scope": "Futures", "resample": "1s"},
        label="Signed Volume (Futures) [1s]",
        group="Aggression",
        description="Net aggressive pressure in base units: +qty for buys, -qty for sells per 1s bucket.",
        depends_on=_dep_trades("Futures"),
        feature_id=14,
    ),
    FeatureSpec(
        name="signed_vol_spot_1s",
        stage="S0",
        operator="trades.signed_volume",
        params={"market_scope": "Spot", "resample": "1s"},
        label="Signed Volume (Spot) [1s]",
        group="Aggression",
        description="Net aggressive pressure in base units: +qty for buys, -qty for sells per 1s bucket.",
        depends_on=_dep_trades("Spot"),
        feature_id=15,
    ),

    # =========================================================================
    # SIGNED NOTIONAL (quote units): +(qty*px) buys, -(qty*px) sells
    # =========================================================================
    FeatureSpec(
        name="signed_notional_fut_1s",
        stage="S0",
        operator="trades.signed_notional",
        params={"market_scope": "Futures", "resample": "1s"},
        label="Signed Notional (Futures) [1s]",
        group="Aggression",
        description="Net aggressive notional: +(qty*px) for buys, -(qty*px) for sells per 1s bucket.",
        depends_on=_dep_trades("Futures"),
        feature_id=16,
    ),
    FeatureSpec(
        name="signed_notional_spot_1s",
        stage="S0",
        operator="trades.signed_notional",
        params={"market_scope": "Spot", "resample": "1s"},
        label="Signed Notional (Spot) [1s]",
        group="Aggression",
        description="Net aggressive notional: +(qty*px) for buys, -(qty*px) for sells per 1s bucket.",
        depends_on=_dep_trades("Spot"),
        feature_id=17,
    ),
]