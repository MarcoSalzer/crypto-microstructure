# etl/spec/s2/s2_returns.py
# ==============================================================================
# S2 Feature Specs: Returns
#
# Binance-only pipeline | Source: S1 features (+ some S0 columns)
# 12 features | Feature IDs: 1124–1135
# ==============================================================================

from typing import List
from etl.spec import FeatureSpec, Dep

S2_RETURNS_FEATURES: List[FeatureSpec] = [

    # === derived.ret_vwap ===
    FeatureSpec(
        name="ret_vwap_fut_300s",
        stage="S2",
        operator="derived.ret_vwap",
        params={'market_scope': 'Futures', 'window_s': 300, 'resample': '1s'},
        group="Returns",
        description="Return of VWAP relative to mid: log(vwap/mid).",
        depends_on=(Dep(name="mid_fut_1s", kind="col"), Dep(name="vwap_fut_1s", kind="col"),),
        feature_id=2631,
    ),
    FeatureSpec(
        name="ret_vwap_fut_60s",
        stage="S2",
        operator="derived.ret_vwap",
        params={'market_scope': 'Futures', 'window_s': 60, 'resample': '1s'},
        group="Returns",
        description="Return of VWAP relative to mid: log(vwap/mid).",
        depends_on=(Dep(name="mid_fut_1s", kind="col"), Dep(name="vwap_fut_1s", kind="col"),),
        feature_id=2632,
    ),
    FeatureSpec(
        name="ret_vwap_fut_900s",
        stage="S2",
        operator="derived.ret_vwap",
        params={'market_scope': 'Futures', 'window_s': 900, 'resample': '1s'},
        group="Returns",
        description="Return of VWAP relative to mid: log(vwap/mid).",
        depends_on=(Dep(name="mid_fut_1s", kind="col"), Dep(name="vwap_fut_1s", kind="col"),),
        feature_id=2633,
    ),
    FeatureSpec(
        name="ret_vwap_spot_300s",
        stage="S2",
        operator="derived.ret_vwap",
        params={'market_scope': 'Spot', 'window_s': 300, 'resample': '1s'},
        group="Returns",
        description="Return of VWAP relative to mid: log(vwap/mid).",
        depends_on=(Dep(name="mid_spot_1s", kind="col"), Dep(name="vwap_spot_1s", kind="col"),),
        feature_id=2634,
    ),
    FeatureSpec(
        name="ret_vwap_spot_60s",
        stage="S2",
        operator="derived.ret_vwap",
        params={'market_scope': 'Spot', 'window_s': 60, 'resample': '1s'},
        group="Returns",
        description="Return of VWAP relative to mid: log(vwap/mid).",
        depends_on=(Dep(name="mid_spot_1s", kind="col"), Dep(name="vwap_spot_1s", kind="col"),),
        feature_id=2635,
    ),
    FeatureSpec(
        name="ret_vwap_spot_900s",
        stage="S2",
        operator="derived.ret_vwap",
        params={'market_scope': 'Spot', 'window_s': 900, 'resample': '1s'},
        group="Returns",
        description="Return of VWAP relative to mid: log(vwap/mid).",
        depends_on=(Dep(name="mid_spot_1s", kind="col"), Dep(name="vwap_spot_1s", kind="col"),),
        feature_id=2636,
    ),

    # === derived.roll_sum ===
    FeatureSpec(
        name="ret_mid_fut_15s",
        stage="S2",
        operator="derived.roll_sum",
        params={'market_scope': 'Futures', 'window_s': 15, 'resample': '1s'},
        group="Returns",
        description="Rolling sum over window.",
        depends_on=(Dep(name="ret_mid_fut_1s", kind="col"),),
        feature_id=2637,
    ),
    FeatureSpec(
        name="ret_mid_fut_60s",
        stage="S2",
        operator="derived.roll_sum",
        params={'market_scope': 'Futures', 'window_s': 60, 'resample': '1s'},
        group="Returns",
        description="Rolling sum over window.",
        depends_on=(Dep(name="ret_mid_fut_1s", kind="col"),),
        feature_id=2638,
    ),
    FeatureSpec(
        name="ret_mid_spot_15s",
        stage="S2",
        operator="derived.roll_sum",
        params={'market_scope': 'Spot', 'window_s': 15, 'resample': '1s'},
        group="Returns",
        description="Rolling sum over window.",
        depends_on=(Dep(name="ret_mid_spot_1s", kind="col"),),
        feature_id=2639,
    ),
    FeatureSpec(
        name="ret_mid_spot_60s",
        stage="S2",
        operator="derived.roll_sum",
        params={'market_scope': 'Spot', 'window_s': 60, 'resample': '1s'},
        group="Returns",
        description="Rolling sum over window.",
        depends_on=(Dep(name="ret_mid_spot_1s", kind="col"),),
        feature_id=2640,
    ),
]