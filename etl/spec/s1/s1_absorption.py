# etl/spec/s1/s1_absorption.py
# ==============================================================================
# S1 Feature Specs: Absorption
#
# Binance-only pipeline | Source: S0 features (parquet)
# 20 features | Feature IDs: 1000-1019
# ==============================================================================

from typing import List
from etl.spec import FeatureSpec, Dep


S1_ABSORPTION_FEATURES: List[FeatureSpec] = [

    # === derived.roll_mean ===
    FeatureSpec(
        name="absorption_volume_ask_fut_1bps_15s",
        stage="S1",
        operator="derived.roll_mean",
        params={'market_scope': 'Futures', 'window_s': 15, 'resample': '1s'},
        group="Absorption",
        description="Rolling mean over window.",
        depends_on=(Dep(name="absorption_volume_ask_fut_1bps_1s", kind="col"),),
        feature_id=1000,
    ),

    # === l2.absorb_refill_ask ===
    FeatureSpec(
        name="absorption_volume_ask_fut_1bps_1s",
        stage="S1",
        operator="l2.absorb_refill_ask",
        params={'market_scope': 'Futures', 'resample': '1s'},
        group="Absorption",
        description="Absorption volume ask: depth_change weighted by taker sell volume.",
        depends_on=(Dep(name="depth_notional_ask_1bps_fut_1s", kind="col"), Dep(name="taker_sell_vol_fut_1s", kind="col"),),
        feature_id=1001,
    ),

    # === derived.roll_mean ===
    FeatureSpec(
        name="absorption_volume_ask_spot_1bps_15s",
        stage="S1",
        operator="derived.roll_mean",
        params={'market_scope': 'Spot', 'window_s': 15, 'resample': '1s'},
        group="Absorption",
        description="Rolling mean over window.",
        depends_on=(Dep(name="absorption_volume_ask_spot_1bps_1s", kind="col"),),
        feature_id=1002,
    ),

    # === l2.absorb_refill_ask ===
    FeatureSpec(
        name="absorption_volume_ask_spot_1bps_1s",
        stage="S1",
        operator="l2.absorb_refill_ask",
        params={'market_scope': 'Spot', 'resample': '1s'},
        group="Absorption",
        description="Absorption volume ask: depth_change weighted by taker sell volume.",
        depends_on=(Dep(name="depth_notional_ask_1bps_spot_1s", kind="col"), Dep(name="taker_sell_vol_spot_1s", kind="col"),),
        feature_id=1003,
    ),

    # === derived.roll_mean ===
    FeatureSpec(
        name="absorption_volume_bid_fut_1bps_15s",
        stage="S1",
        operator="derived.roll_mean",
        params={'market_scope': 'Futures', 'window_s': 15, 'resample': '1s'},
        group="Absorption",
        description="Rolling mean over window.",
        depends_on=(Dep(name="absorption_volume_bid_fut_1bps_1s", kind="col"),),
        feature_id=1004,
    ),

    # === l2.absorb_refill_bid ===
    FeatureSpec(
        name="absorption_volume_bid_fut_1bps_1s",
        stage="S1",
        operator="l2.absorb_refill_bid",
        params={'market_scope': 'Futures', 'resample': '1s'},
        group="Absorption",
        description="Absorption volume bid: depth_change weighted by taker buy volume.",
        depends_on=(Dep(name="depth_notional_bid_1bps_fut_1s", kind="col"), Dep(name="taker_buy_vol_fut_1s", kind="col"),),
        feature_id=1005,
    ),

    # === derived.roll_mean ===
    FeatureSpec(
        name="absorption_volume_bid_spot_1bps_15s",
        stage="S1",
        operator="derived.roll_mean",
        params={'market_scope': 'Spot', 'window_s': 15, 'resample': '1s'},
        group="Absorption",
        description="Rolling mean over window.",
        depends_on=(Dep(name="absorption_volume_bid_spot_1bps_1s", kind="col"),),
        feature_id=1006,
    ),

    # === l2.absorb_refill_bid ===
    FeatureSpec(
        name="absorption_volume_bid_spot_1bps_1s",
        stage="S1",
        operator="l2.absorb_refill_bid",
        params={'market_scope': 'Spot', 'resample': '1s'},
        group="Absorption",
        description="Absorption volume bid: depth_change weighted by taker buy volume.",
        depends_on=(Dep(name="depth_notional_bid_1bps_spot_1s", kind="col"), Dep(name="taker_buy_vol_spot_1s", kind="col"),),
        feature_id=1007,
    ),

    # === l2.aggr_absorp_ratio_ask ===
    FeatureSpec(
        name="aggr_absorp_ratio_ask_fut_10bps_1s",
        stage="S1",
        operator="l2.aggr_absorp_ratio_ask",
        params={'market_scope': 'Futures', 'resample': '1s'},
        group="Absorption",
        description="Aggression-absorption ratio ask: taker_sell / (depth_ask + eps).",
        depends_on=(Dep(name="depth_notional_ask_10bps_fut_1s", kind="col"), Dep(name="taker_sell_vol_fut_1s", kind="col"),),
        feature_id=1008,
    ),
    FeatureSpec(
        name="aggr_absorp_ratio_ask_fut_2bps_1s",
        stage="S1",
        operator="l2.aggr_absorp_ratio_ask",
        params={'market_scope': 'Futures', 'resample': '1s'},
        group="Absorption",
        description="Aggression-absorption ratio ask: taker_sell / (depth_ask + eps).",
        depends_on=(Dep(name="depth_notional_ask_2bps_fut_1s", kind="col"), Dep(name="taker_sell_vol_fut_1s", kind="col"),),
        feature_id=1009,
    ),
    FeatureSpec(
        name="aggr_absorp_ratio_ask_fut_5bps_1s",
        stage="S1",
        operator="l2.aggr_absorp_ratio_ask",
        params={'market_scope': 'Futures', 'resample': '1s'},
        group="Absorption",
        description="Aggression-absorption ratio ask: taker_sell / (depth_ask + eps).",
        depends_on=(Dep(name="depth_notional_ask_5bps_fut_1s", kind="col"), Dep(name="taker_sell_vol_fut_1s", kind="col"),),
        feature_id=1010,
    ),
    FeatureSpec(
        name="aggr_absorp_ratio_ask_spot_10bps_1s",
        stage="S1",
        operator="l2.aggr_absorp_ratio_ask",
        params={'market_scope': 'Spot', 'resample': '1s'},
        group="Absorption",
        description="Aggression-absorption ratio ask: taker_sell / (depth_ask + eps).",
        depends_on=(Dep(name="depth_notional_ask_10bps_spot_1s", kind="col"), Dep(name="taker_sell_vol_spot_1s", kind="col"),),
        feature_id=1011,
    ),
    FeatureSpec(
        name="aggr_absorp_ratio_ask_spot_2bps_1s",
        stage="S1",
        operator="l2.aggr_absorp_ratio_ask",
        params={'market_scope': 'Spot', 'resample': '1s'},
        group="Absorption",
        description="Aggression-absorption ratio ask: taker_sell / (depth_ask + eps).",
        depends_on=(Dep(name="depth_notional_ask_2bps_spot_1s", kind="col"), Dep(name="taker_sell_vol_spot_1s", kind="col"),),
        feature_id=1012,
    ),
    FeatureSpec(
        name="aggr_absorp_ratio_ask_spot_5bps_1s",
        stage="S1",
        operator="l2.aggr_absorp_ratio_ask",
        params={'market_scope': 'Spot', 'resample': '1s'},
        group="Absorption",
        description="Aggression-absorption ratio ask: taker_sell / (depth_ask + eps).",
        depends_on=(Dep(name="depth_notional_ask_5bps_spot_1s", kind="col"), Dep(name="taker_sell_vol_spot_1s", kind="col"),),
        feature_id=1013,
    ),

    # === l2.aggr_absorp_ratio_bid ===
    FeatureSpec(
        name="aggr_absorp_ratio_bid_fut_10bps_1s",
        stage="S1",
        operator="l2.aggr_absorp_ratio_bid",
        params={'market_scope': 'Futures', 'resample': '1s'},
        group="Absorption",
        description="Aggression-absorption ratio bid: taker_buy / (depth_bid + eps).",
        depends_on=(Dep(name="depth_notional_bid_10bps_fut_1s", kind="col"), Dep(name="taker_buy_vol_fut_1s", kind="col"),),
        feature_id=1014,
    ),
    FeatureSpec(
        name="aggr_absorp_ratio_bid_fut_2bps_1s",
        stage="S1",
        operator="l2.aggr_absorp_ratio_bid",
        params={'market_scope': 'Futures', 'resample': '1s'},
        group="Absorption",
        description="Aggression-absorption ratio bid: taker_buy / (depth_bid + eps).",
        depends_on=(Dep(name="depth_notional_bid_2bps_fut_1s", kind="col"), Dep(name="taker_buy_vol_fut_1s", kind="col"),),
        feature_id=1015,
    ),
    FeatureSpec(
        name="aggr_absorp_ratio_bid_fut_5bps_1s",
        stage="S1",
        operator="l2.aggr_absorp_ratio_bid",
        params={'market_scope': 'Futures', 'resample': '1s'},
        group="Absorption",
        description="Aggression-absorption ratio bid: taker_buy / (depth_bid + eps).",
        depends_on=(Dep(name="depth_notional_bid_5bps_fut_1s", kind="col"), Dep(name="taker_buy_vol_fut_1s", kind="col"),),
        feature_id=1016,
    ),
    FeatureSpec(
        name="aggr_absorp_ratio_bid_spot_10bps_1s",
        stage="S1",
        operator="l2.aggr_absorp_ratio_bid",
        params={'market_scope': 'Spot', 'resample': '1s'},
        group="Absorption",
        description="Aggression-absorption ratio bid: taker_buy / (depth_bid + eps).",
        depends_on=(Dep(name="depth_notional_bid_10bps_spot_1s", kind="col"), Dep(name="taker_buy_vol_spot_1s", kind="col"),),
        feature_id=1017,
    ),
    FeatureSpec(
        name="aggr_absorp_ratio_bid_spot_2bps_1s",
        stage="S1",
        operator="l2.aggr_absorp_ratio_bid",
        params={'market_scope': 'Spot', 'resample': '1s'},
        group="Absorption",
        description="Aggression-absorption ratio bid: taker_buy / (depth_bid + eps).",
        depends_on=(Dep(name="depth_notional_bid_2bps_spot_1s", kind="col"), Dep(name="taker_buy_vol_spot_1s", kind="col"),),
        feature_id=1018,
    ),
    FeatureSpec(
        name="aggr_absorp_ratio_bid_spot_5bps_1s",
        stage="S1",
        operator="l2.aggr_absorp_ratio_bid",
        params={'market_scope': 'Spot', 'resample': '1s'},
        group="Absorption",
        description="Aggression-absorption ratio bid: taker_buy / (depth_bid + eps).",
        depends_on=(Dep(name="depth_notional_bid_5bps_spot_1s", kind="col"), Dep(name="taker_buy_vol_spot_1s", kind="col"),),
        feature_id=1019,
    ),
]