# etl/spec/s2/s2_activity.py
# ==============================================================================
# S2 Feature Specs: Activity
#
# Binance-only pipeline | Source: S1 features (+ some S0 columns)
# 8 features | Feature IDs: 480–487
# ==============================================================================

from typing import List
from etl.spec import FeatureSpec, Dep


S2_ACTIVITY_FEATURES: List[FeatureSpec] = [

    # === derived.roll_mean ===
    FeatureSpec(
        name="avg_trade_size_fut_300s",
        stage="S2",
        operator="derived.roll_mean",
        params={'market_scope': 'Futures', 'window_s': 300, 'resample': '1s', 'min_periods': 60},
        group="Activity",
        description="Rolling mean over window.",
        depends_on=(Dep(name="avg_trade_size_fut_1s", kind="col"),),
        feature_id=2024,
    ),
    FeatureSpec(
        name="avg_trade_size_fut_60s",
        stage="S2",
        operator="derived.roll_mean",
        # is NaN ~5% of seconds (no trades). Strict 60-of-60 requirement
        # gave 72% NaN downstream; 10-of-60 gives ~5% NaN, matching input.
        params={'market_scope': 'Futures', 'window_s': 60, 'resample': '1s', 'min_periods': 10},
        group="Activity",
        description="Rolling mean over window.",
        depends_on=(Dep(name="avg_trade_size_fut_1s", kind="col"),),
        feature_id=2025,
    ),
    FeatureSpec(
        name="avg_trade_size_fut_900s",
        stage="S2",
        operator="derived.roll_mean",
        params={'market_scope': 'Futures', 'window_s': 900, 'resample': '1s', 'min_periods': 60},
        group="Activity",
        description="Rolling mean over window.",
        depends_on=(Dep(name="avg_trade_size_fut_1s", kind="col"),),
        feature_id=2026,
    ),
    FeatureSpec(
        name="avg_trade_size_spot_300s",
        stage="S2",
        operator="derived.roll_mean",
        params={'market_scope': 'Spot', 'window_s': 300, 'resample': '1s', 'min_periods': 60},
        group="Activity",
        description="Rolling mean over window.",
        depends_on=(Dep(name="avg_trade_size_spot_1s", kind="col"),),
        feature_id=2027,
    ),
    FeatureSpec(
        name="avg_trade_size_spot_60s",
        stage="S2",
        operator="derived.roll_mean",
        params={'market_scope': 'Spot', 'window_s': 60, 'resample': '1s', 'min_periods': 10},
        group="Activity",
        description="Rolling mean over window.",
        depends_on=(Dep(name="avg_trade_size_spot_1s", kind="col"),),
        feature_id=2028,
    ),
    FeatureSpec(
        name="avg_trade_size_spot_900s",
        stage="S2",
        operator="derived.roll_mean",
        params={'market_scope': 'Spot', 'window_s': 900, 'resample': '1s', 'min_periods': 60},
        group="Activity",
        description="Rolling mean over window.",
        depends_on=(Dep(name="avg_trade_size_spot_1s", kind="col"),),
        feature_id=2029,
    ),

    # === derived.roll_sum ===
    FeatureSpec(
        name="l2_update_count_fut_5bps_5s",
        stage="S2",
        operator="derived.roll_sum",
        params={'market_scope': 'Futures', 'window_s': 5, 'resample': '1s'},
        group="Activity",
        description="Rolling sum over window.",
        depends_on=(Dep(name="l2_update_count_fut_5bps_1s", kind="col"),),
        feature_id=2030,
    ),
    FeatureSpec(
        name="l2_update_count_spot_5bps_5s",
        stage="S2",
        operator="derived.roll_sum",
        params={'market_scope': 'Spot', 'window_s': 5, 'resample': '1s'},
        group="Activity",
        description="Rolling sum over window.",
        depends_on=(Dep(name="l2_update_count_spot_5bps_1s", kind="col"),),
        feature_id=2031,
    ),
]