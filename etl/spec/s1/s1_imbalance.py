# etl/spec/s1/s1_imbalance.py
# ==============================================================================
# S1 Feature Specs: Imbalance
#
# Binance-only pipeline | Source: S0 features (parquet)
# 16 features | Feature IDs: 1118-1129
#
# ==============================================================================

from typing import List
from etl.spec import FeatureSpec, Dep

S1_IMBALANCE_FEATURES: List[FeatureSpec] = [

    # === liq_imb_fut_struct100 ===
    FeatureSpec(
        name="liq_imb_fut_struct100_60s",
        stage="S1",
        operator="derived.roll_mean",
        params={'market_scope': 'Futures', 'window_s': 60, 'resample': '1s'},
        group="Imbalance",
        description="Rolling mean of depth imbalance over window.",
        depends_on=(Dep(name="depth_imbalance_struct100_fut_1s", kind="col"),),
        feature_id=1118,
    ),
    FeatureSpec(
        name="liq_imb_fut_struct100_300s",
        stage="S1",
        operator="derived.roll_mean",
        params={'market_scope': 'Futures', 'window_s': 300, 'resample': '1s'},
        group="Imbalance",
        description="Rolling mean of depth imbalance over window.",
        depends_on=(Dep(name="depth_imbalance_struct100_fut_1s", kind="col"),),
        feature_id=1119,
    ),
    FeatureSpec(
        name="liq_imb_fut_struct100_900s",
        stage="S1",
        operator="derived.roll_mean",
        params={'market_scope': 'Futures', 'window_s': 900, 'resample': '1s'},
        group="Imbalance",
        description="Rolling mean of depth imbalance over window.",
        depends_on=(Dep(name="depth_imbalance_struct100_fut_1s", kind="col"),),
        feature_id=1120,
    ),

    # === liq_imb_fut_struct50 ===
    FeatureSpec(
        name="liq_imb_fut_struct50_60s",
        stage="S1",
        operator="derived.roll_mean",
        params={'market_scope': 'Futures', 'window_s': 60, 'resample': '1s'},
        group="Imbalance",
        description="Rolling mean of depth imbalance over window.",
        depends_on=(Dep(name="depth_imbalance_struct50_fut_1s", kind="col"),),
        feature_id=1121,
    ),
    FeatureSpec(
        name="liq_imb_fut_struct50_300s",
        stage="S1",
        operator="derived.roll_mean",
        params={'market_scope': 'Futures', 'window_s': 300, 'resample': '1s'},
        group="Imbalance",
        description="Rolling mean of depth imbalance over window.",
        depends_on=(Dep(name="depth_imbalance_struct50_fut_1s", kind="col"),),
        feature_id=1122,
    ),
    FeatureSpec(
        name="liq_imb_fut_struct50_900s",
        stage="S1",
        operator="derived.roll_mean",
        params={'market_scope': 'Futures', 'window_s': 900, 'resample': '1s'},
        group="Imbalance",
        description="Rolling mean of depth imbalance over window.",
        depends_on=(Dep(name="depth_imbalance_struct50_fut_1s", kind="col"),),
        feature_id=1123,
    ),

    # === liq_imb_spot_struct100 ===
    FeatureSpec(
        name="liq_imb_spot_struct100_60s",
        stage="S1",
        operator="derived.roll_mean",
        params={'market_scope': 'Spot', 'window_s': 60, 'resample': '1s'},
        group="Imbalance",
        description="Rolling mean of depth imbalance over window.",
        depends_on=(Dep(name="depth_imbalance_struct100_spot_1s", kind="col"),),
        feature_id=1124,
    ),
    FeatureSpec(
        name="liq_imb_spot_struct100_300s",
        stage="S1",
        operator="derived.roll_mean",
        params={'market_scope': 'Spot', 'window_s': 300, 'resample': '1s'},
        group="Imbalance",
        description="Rolling mean of depth imbalance over window.",
        depends_on=(Dep(name="depth_imbalance_struct100_spot_1s", kind="col"),),
        feature_id=1125,
    ),
    FeatureSpec(
        name="liq_imb_spot_struct100_900s",
        stage="S1",
        operator="derived.roll_mean",
        params={'market_scope': 'Spot', 'window_s': 900, 'resample': '1s'},
        group="Imbalance",
        description="Rolling mean of depth imbalance over window.",
        depends_on=(Dep(name="depth_imbalance_struct100_spot_1s", kind="col"),),
        feature_id=1126,
    ),

    # === liq_imb_spot_struct50 ===
    FeatureSpec(
        name="liq_imb_spot_struct50_60s",
        stage="S1",
        operator="derived.roll_mean",
        params={'market_scope': 'Spot', 'window_s': 60, 'resample': '1s'},
        group="Imbalance",
        description="Rolling mean of depth imbalance over window.",
        depends_on=(Dep(name="depth_imbalance_struct50_spot_1s", kind="col"),),
        feature_id=1127,
    ),
    FeatureSpec(
        name="liq_imb_spot_struct50_300s",
        stage="S1",
        operator="derived.roll_mean",
        params={'market_scope': 'Spot', 'window_s': 300, 'resample': '1s'},
        group="Imbalance",
        description="Rolling mean of depth imbalance over window.",
        depends_on=(Dep(name="depth_imbalance_struct50_spot_1s", kind="col"),),
        feature_id=1128,
    ),
    FeatureSpec(
        name="liq_imb_spot_struct50_900s",
        stage="S1",
        operator="derived.roll_mean",
        params={'market_scope': 'Spot', 'window_s': 900, 'resample': '1s'},
        group="Imbalance",
        description="Rolling mean of depth imbalance over window.",
        depends_on=(Dep(name="depth_imbalance_struct50_spot_1s", kind="col"),),
        feature_id=1129,
    ),
]