# etl/spec/s2/s2_aggression.py
# ==============================================================================
# S2 Feature Specs: Aggression
#
# Binance-only pipeline | Source: S1 features (+ some S0 columns)
# 12 features | Feature IDs: 488–499
# ==============================================================================

from typing import List
from etl.spec import FeatureSpec, Dep

S2_AGGRESSION_FEATURES: List[FeatureSpec] = [

    # === derived.roll_mean ===
    FeatureSpec(
        name="taker_imbalance_fut_15s",
        stage="S2",
        operator="derived.roll_mean",
        params={'market_scope': 'Futures', 'window_s': 15, 'resample': '1s'},
        group="Aggression",
        description="Rolling mean over window.",
        depends_on=(Dep(name="taker_imbalance_fut_1s", kind="col"),),
        feature_id=2032,
    ),
    FeatureSpec(
        name="taker_imbalance_fut_300s",
        stage="S2",
        operator="derived.roll_mean",
        params={'market_scope': 'Futures', 'window_s': 300, 'resample': '1s'},
        group="Aggression",
        description="Rolling mean over window.",
        depends_on=(Dep(name="taker_imbalance_fut_1s", kind="col"),),
        feature_id=2033,
    ),
    FeatureSpec(
        name="taker_imbalance_fut_5s",
        stage="S2",
        operator="derived.roll_mean",
        params={'market_scope': 'Futures', 'window_s': 5, 'resample': '1s'},
        group="Aggression",
        description="Rolling mean over window.",
        depends_on=(Dep(name="taker_imbalance_fut_1s", kind="col"),),
        feature_id=2034,
    ),
    FeatureSpec(
        name="taker_imbalance_fut_900s",
        stage="S2",
        operator="derived.roll_mean",
        params={'market_scope': 'Futures', 'window_s': 900, 'resample': '1s'},
        group="Aggression",
        description="Rolling mean over window.",
        depends_on=(Dep(name="taker_imbalance_fut_1s", kind="col"),),
        feature_id=2035,
    ),
    FeatureSpec(
        name="taker_imbalance_spot_15s",
        stage="S2",
        operator="derived.roll_mean",
        params={'market_scope': 'Spot', 'window_s': 15, 'resample': '1s'},
        group="Aggression",
        description="Rolling mean over window.",
        depends_on=(Dep(name="taker_imbalance_spot_1s", kind="col"),),
        feature_id=2036,
    ),
    FeatureSpec(
        name="taker_imbalance_spot_300s",
        stage="S2",
        operator="derived.roll_mean",
        params={'market_scope': 'Spot', 'window_s': 300, 'resample': '1s'},
        group="Aggression",
        description="Rolling mean over window.",
        depends_on=(Dep(name="taker_imbalance_spot_1s", kind="col"),),
        feature_id=2037,
    ),
    FeatureSpec(
        name="taker_imbalance_spot_5s",
        stage="S2",
        operator="derived.roll_mean",
        params={'market_scope': 'Spot', 'window_s': 5, 'resample': '1s'},
        group="Aggression",
        description="Rolling mean over window.",
        depends_on=(Dep(name="taker_imbalance_spot_1s", kind="col"),),
        feature_id=2038,
    ),
    FeatureSpec(
        name="taker_imbalance_spot_900s",
        stage="S2",
        operator="derived.roll_mean",
        params={'market_scope': 'Spot', 'window_s': 900, 'resample': '1s'},
        group="Aggression",
        description="Rolling mean over window.",
        depends_on=(Dep(name="taker_imbalance_spot_1s", kind="col"),),
        feature_id=2039,
    ),

    # === trades.taker_imbalance_bucket ===
    FeatureSpec(
        name="taker_imbalance_fut_60s",
        stage="S2",
        operator="trades.taker_imbalance_bucket",
        params={'market_scope': 'Futures', 'window_s': 60, 'resample': '1s'},
        group="Aggression",
        description="Taker imbalance bucketed over window.",
        depends_on=(Dep(name="taker_imbalance_fut_1s", kind="col"),),
        feature_id=2040,
    ),
    FeatureSpec(
        name="taker_imbalance_spot_60s",
        stage="S2",
        operator="trades.taker_imbalance_bucket",
        params={'market_scope': 'Spot', 'window_s': 60, 'resample': '1s'},
        group="Aggression",
        description="Taker imbalance bucketed over window.",
        depends_on=(Dep(name="taker_imbalance_spot_1s", kind="col"),),
        feature_id=2041,
    ),
]