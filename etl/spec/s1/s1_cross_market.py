# etl/spec/s1/s1_cross_market.py
# ==============================================================================
# S1 Feature Specs: Cross-Market
#
# Binance-only pipeline | Source: S0 features (parquet)
# 10 features | Feature IDs: 1108-1117
# ==============================================================================

from typing import List
from etl.spec import FeatureSpec, Dep


S1_CROSS_MARKET_FEATURES: List[FeatureSpec] = [

    # === derived.basis_mid ===
    FeatureSpec(
        name="basis_sf_mid_60s",
        stage="S1",
        operator="deriv.basis_mid",
        params={'window_s': 60, 'resample': '1s'},
        group="Cross-Market",
        description="Basis: rolling mean of (mid_fut - mid_spot) over 60s.",
        depends_on=(Dep(name="mid_fut_1s", kind="col"), Dep(name="mid_spot_1s", kind="col"),),
        feature_id=1108,
    ),

    # === derived.sub ===
    FeatureSpec(
        name="liq_imb_div_fut_minus_spot_struct100_1s",
        stage="S1",
        operator="derived.sub",
        params={'market_scope': 'Futures', 'resample': '1s'},
        group="Cross-Market",
        description="Difference: col_a - col_b.",
        depends_on=(Dep(name="depth_imbalance_struct100_fut_1s", kind="col"), Dep(name="depth_imbalance_struct100_spot_1s", kind="col"),),
        feature_id=1109,
    ),
    FeatureSpec(
        name="liq_imb_div_fut_minus_spot_struct50_1s",
        stage="S1",
        operator="derived.sub",
        params={'market_scope': 'Futures', 'resample': '1s'},
        group="Cross-Market",
        description="Difference: col_a - col_b.",
        depends_on=(Dep(name="depth_imbalance_struct50_fut_1s", kind="col"), Dep(name="depth_imbalance_struct50_spot_1s", kind="col"),),
        feature_id=1110,
    ),

    # === derived.spot_fut_taker_activity_share_1s ===
    FeatureSpec(
        name="taker_activity_share_sf_1s",
        stage="S1",
        operator="deriv.spot_fut_taker_activity_share_1s",
        params={'resample': '1s'},
        group="Cross-Market",
        description="Spot share of total taker activity.",
        depends_on=(Dep(name="taker_buy_vol_spot_1s", kind="col"), Dep(name="taker_sell_vol_spot_1s", kind="col"), Dep(name="taker_buy_vol_fut_1s", kind="col"), Dep(name="taker_sell_vol_fut_1s", kind="col"),),
        feature_id=1111,
    ),

    # === derived.count_ratio === [CROSS-DIV-FIX 2026-04-27]
    # Switched from derived.ratio to derived.count_ratio so that seconds
    # without trade activity (~3-6% per asset) yield 0 instead of NaN.
    # Fixes 100% NaN in downstream rolling windows (S2 trade_count_*_div_300s/900s)
    # and S3 trade_count_sf_div_*.
    FeatureSpec(
        name="trade_count_fut_div_1s",
        stage="S1",
        operator="derived.count_ratio",
        params={'market_scope': 'Futures', 'resample': '1s'},
        group="Cross-Market",
        description="Ratio fut/spot trade count. 0/0 = 0 (no activity = no divergence).",
        depends_on=(Dep(name="trade_count_fut_1s", kind="col"), Dep(name="trade_count_spot_1s", kind="col"),),
        feature_id=1112,
    ),

    # === derived.share ===
    FeatureSpec(
        name="trade_count_fut_share_1s",
        stage="S1",
        operator="derived.share",
        params={'market_scope': 'Futures', 'resample': '1s'},
        group="Cross-Market",
        description="Share: fut / (fut + spot + eps). Futures fraction of total trade count.",
        depends_on=(Dep(name="trade_count_fut_1s", kind="col"), Dep(name="trade_count_spot_1s", kind="col"),),
        feature_id=1113,
    ),

    # === derived.count_ratio === [CROSS-DIV-FIX 2026-04-27]
    FeatureSpec(
        name="trade_count_spot_div_1s",
        stage="S1",
        operator="derived.count_ratio",
        params={'market_scope': 'Spot', 'resample': '1s'},
        group="Cross-Market",
        description="Ratio spot/fut trade count. 0/0 = 0 (no activity = no divergence).",
        depends_on=(Dep(name="trade_count_spot_1s", kind="col"), Dep(name="trade_count_fut_1s", kind="col"),),
        feature_id=1114,
    ),

    # === derived.share ===
    FeatureSpec(
        name="trade_count_spot_share_1s",
        stage="S1",
        operator="derived.share",
        params={'market_scope': 'Spot', 'resample': '1s'},
        group="Cross-Market",
        description="Share: spot / (spot + fut + eps). Spot fraction of total trade count.",
        depends_on=(Dep(name="trade_count_spot_1s", kind="col"), Dep(name="trade_count_fut_1s", kind="col"),),
        feature_id=1115,
    ),

    # === derived.count_ratio === [CROSS-DIV-FIX 2026-04-27]
    FeatureSpec(
        name="volume_fut_div_delta_1s",
        stage="S1",
        operator="derived.count_ratio",
        params={'market_scope': 'Futures', 'resample': '1s'},
        group="Cross-Market",
        description="Ratio fut/spot volume. 0/0 = 0 (no activity = no divergence).",
        depends_on=(Dep(name="volume_fut_1s", kind="col"), Dep(name="volume_spot_1s", kind="col"),),
        feature_id=1116,
    ),
    FeatureSpec(
        name="volume_spot_div_delta_1s",
        stage="S1",
        operator="derived.count_ratio",
        params={'market_scope': 'Spot', 'resample': '1s'},
        group="Cross-Market",
        description="Ratio spot/fut volume. 0/0 = 0 (no activity = no divergence).",
        depends_on=(Dep(name="volume_spot_1s", kind="col"), Dep(name="volume_fut_1s", kind="col"),),
        feature_id=1117,
    ),
]