# etl/spec/s2/s2_price.py
# ==============================================================================
# S2 Feature Specs: Price
#
# Binance-only pipeline | Source: S1 features (+ some S0 columns)
# 5 features | Feature IDs: 2608–2612
# ==============================================================================

from typing import List
from etl.spec import FeatureSpec, Dep


S2_PRICE_FEATURES: List[FeatureSpec] = [

    # === derived.mid_touch_dev ===
    # NOTE: mid_touch_dev_fut_{1|2}bps_1s removed. mid_touch_fut_1s = (bid+ask)/2 = mid_fut_1s,
    # so (mid_touch - mid) / mid * 10000 = 0 always. No unique information.
    # Same applies to all mid_touch_{fut|spot}_{60|300|900}s rolling means.

    # === derived.price_acceleration ===
    FeatureSpec(
        name="price_acceleration_spot_300s",
        stage="S2",
        operator="derived.price_acceleration",
        params={'market_scope': 'Spot', 'window_s': 300, 'resample': '1s'},
        group="Price",
        description="Second derivative of price (change of returns).",
        depends_on=(Dep(name="ret_mid_spot_1s", kind="col"),),
        feature_id=2626,
    ),
    FeatureSpec(
        name="price_acceleration_spot_60s",
        stage="S2",
        operator="derived.price_acceleration",
        params={'market_scope': 'Spot', 'window_s': 60, 'resample': '1s'},
        group="Price",
        description="Second derivative of price (change of returns).",
        depends_on=(Dep(name="ret_mid_spot_1s", kind="col"),),
        feature_id=2627,
    ),
    FeatureSpec(
        name="price_acceleration_spot_900s",
        stage="S2",
        operator="derived.price_acceleration",
        params={'market_scope': 'Spot', 'window_s': 900, 'resample': '1s'},
        group="Price",
        description="Second derivative of price (change of returns).",
        depends_on=(Dep(name="ret_mid_spot_1s", kind="col"),),
        feature_id=2628,
    ),

    # === derived.price_deviation_bps ===
    FeatureSpec(
        name="price_deviation_fut_1s",
        stage="S2",
        operator="derived.price_deviation_bps",
        params={'market_scope': 'Futures', 'window_s': 1, 'resample': '1s'},
        group="Price",
        description="VWAP-mid deviation in basis points.",
        depends_on=(Dep(name="mid_fut_1s", kind="col"), Dep(name="vwap_fut_1s", kind="col"),),
        feature_id=2629,
    ),
    FeatureSpec(
        name="price_deviation_spot_1s",
        stage="S2",
        operator="derived.price_deviation_bps",
        params={'market_scope': 'Spot', 'window_s': 1, 'resample': '1s'},
        group="Price",
        description="VWAP-mid deviation in basis points.",
        depends_on=(Dep(name="mid_spot_1s", kind="col"), Dep(name="vwap_spot_1s", kind="col"),),
        feature_id=2630,
    ),

    # === derived.roll_mean ===
    # NOTE: mid_touch_{fut|spot}_{60|300|900}s removed — rolling means of a constant
    # (mid_touch = mid always) produce the same constant. No unique information.
]