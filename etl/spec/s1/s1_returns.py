# etl/spec/s1/s1_returns.py
# ==============================================================================
# S1 Feature Specs: Returns
#
# Binance-only pipeline | Source: S0 features (parquet)
# 10 features | Feature IDs: 1313-1322
#
# ==============================================================================

from typing import List
from etl.spec import FeatureSpec, Dep

S1_RETURNS_FEATURES: List[FeatureSpec] = [

    # === derived.ret_fwd (existing — original 5 horizons) ===
    FeatureSpec(
        name="ret_fwd_15s",
        stage="S1",
        operator="derived.ret_fwd",
        params={'market_scope': 'Futures', 'window_s': 15, 'resample': '1s'},
        group="Returns",
        description="Forward return: log(mid_{t+w} / mid_t).",
        depends_on=(Dep(name="mid_fut_1s", kind="col"),),
        feature_id=1313,
    ),
    FeatureSpec(
        name="ret_fwd_1s",
        stage="S1",
        operator="derived.ret_fwd",
        params={'market_scope': 'Futures', 'window_s': 1, 'resample': '1s'},
        group="Returns",
        description="Forward return: log(mid_{t+1} / mid_t).",
        depends_on=(Dep(name="mid_fut_1s", kind="col"),),
        feature_id=1314,
    ),
    FeatureSpec(
        name="ret_fwd_300s",
        stage="S1",
        operator="derived.ret_fwd",
        params={'market_scope': 'Futures', 'window_s': 300, 'resample': '1s'},
        group="Returns",
        description="Forward return: log(mid_{t+w} / mid_t).",
        depends_on=(Dep(name="mid_fut_1s", kind="col"),),
        feature_id=1315,
    ),
    FeatureSpec(
        name="ret_fwd_60s",
        stage="S1",
        operator="derived.ret_fwd",
        params={'market_scope': 'Futures', 'window_s': 60, 'resample': '1s'},
        group="Returns",
        description="Forward return: log(mid_{t+w} / mid_t).",
        depends_on=(Dep(name="mid_fut_1s", kind="col"),),
        feature_id=1316,
    ),
    FeatureSpec(
        name="ret_fwd_900s",
        stage="S1",
        operator="derived.ret_fwd",
        params={'market_scope': 'Futures', 'window_s': 900, 'resample': '1s'},
        group="Returns",
        description="Forward return: log(mid_{t+w} / mid_t).",
        depends_on=(Dep(name="mid_fut_1s", kind="col"),),
        feature_id=1317,
    ),

    # === derived.log_return (existing) ===
    FeatureSpec(
        name="ret_mid_fut_1s",
        stage="S1",
        operator="derived.log_return",
        params={'market_scope': 'Futures', 'resample': '1s'},
        group="Returns",
        description="Log return: log(price_t / price_{t-1}).",
        depends_on=(Dep(name="mid_fut_1s", kind="col"),),
        feature_id=1318,
    ),
    FeatureSpec(
        name="ret_mid_spot_1s",
        stage="S1",
        operator="derived.log_return",
        params={'market_scope': 'Spot', 'resample': '1s'},
        group="Returns",
        description="Log return: log(price_t / price_{t-1}).",
        depends_on=(Dep(name="mid_spot_1s", kind="col"),),
        feature_id=1319,
    ),

    # =========================================================================
    # PHASE 4 ADDITIONS — additional forward-return horizons (IDs 1320-1322)
    # =========================================================================
    FeatureSpec(
        name="ret_fwd_5s",
        stage="S1",
        operator="derived.ret_fwd",
        params={'market_scope': 'Futures', 'window_s': 5, 'resample': '1s'},
        group="Returns",
        description="Forward return: log(mid_{t+5} / mid_t). Very-short-horizon "
                    "target for high-frequency signals; dense companion to "
                    "ret_fwd_1s/15s for models that benefit from many close-in "
                    "horizon samples.",
        depends_on=(Dep(name="mid_fut_1s", kind="col"),),
        feature_id=1320,
    ),
    FeatureSpec(
        name="ret_fwd_30s",
        stage="S1",
        operator="derived.ret_fwd",
        params={'market_scope': 'Futures', 'window_s': 30, 'resample': '1s'},
        group="Returns",
        description="Forward return: log(mid_{t+30} / mid_t). Fills the gap "
                    "between 15s and 60s horizons; matches the typical 15-60s "
                    "band where most intra-signal MFE/MAE realises.",
        depends_on=(Dep(name="mid_fut_1s", kind="col"),),
        feature_id=1321,
    ),
    FeatureSpec(
        name="ret_fwd_120s",
        stage="S1",
        operator="derived.ret_fwd",
        params={'market_scope': 'Futures', 'window_s': 120, 'resample': '1s'},
        group="Returns",
        description="Forward return: log(mid_{t+120} / mid_t). Fills the gap "
                    "between 60s and 300s; intermediate-horizon companion that "
                    "is still short enough for intraday signal evaluation.",
        depends_on=(Dep(name="mid_fut_1s", kind="col"),),
        feature_id=1322,
    ),
]