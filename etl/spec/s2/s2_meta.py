# etl/spec/s2/s2_meta.py
# ==============================================================================
# S2 Feature Specs: Meta
#
# Binance-only pipeline | Source: S1 features (+ some S0 columns)
# 16 features | Feature IDs: 983–998
# ==============================================================================

from typing import List
from etl.spec import FeatureSpec, Dep


S2_META_FEATURES: List[FeatureSpec] = [

    # === derived.breakout_regime_flag ===
    FeatureSpec(
        name="breakout_regime_flag_300s",
        stage="S2",
        operator="derived.breakout_regime_flag",
        params={'market_scope': 'Futures', 'window_s': 300, 'resample': '1s'},
        group="Meta",
        description="Binary flag: ret and volume exceed threshold simultaneously.",
        depends_on=(Dep(name="ret_mid_fut_1s", kind="col"), Dep(name="volume_fut_1s", kind="col"),),
        feature_id=2501,
    ),
    FeatureSpec(
        name="breakout_regime_flag_60s",
        stage="S2",
        operator="derived.breakout_regime_flag",
        params={'market_scope': 'Futures', 'window_s': 60, 'resample': '1s'},
        group="Meta",
        description="Binary flag: ret and volume exceed threshold simultaneously.",
        depends_on=(Dep(name="ret_mid_fut_1s", kind="col"), Dep(name="volume_fut_1s", kind="col"),),
        feature_id=2502,
    ),

    # === derived.depth_coherence ===
    FeatureSpec(
        name="queue_pressure_depth_coherence_fut_5s",
        stage="S2",
        operator="derived.depth_coherence",
        params={'market_scope': 'Futures', 'window_s': 5, 'resample': '1s'},
        group="Meta",
        description="Cross-depth coherence: correlation of queue_pressure across BPS bands.",
        depends_on=(Dep(name="queue_pressure_fut_10bps_5s", kind="col"), Dep(name="queue_pressure_fut_1bps_5s", kind="col"), Dep(name="queue_pressure_fut_2bps_5s", kind="col"), Dep(name="queue_pressure_fut_5bps_5s", kind="col"),),
        feature_id=2503,
    ),
    FeatureSpec(
        name="queue_pressure_depth_coherence_spot_5s",
        stage="S2",
        operator="derived.depth_coherence",
        params={'market_scope': 'Spot', 'window_s': 5, 'resample': '1s'},
        group="Meta",
        description="Cross-depth coherence: correlation of queue_pressure across BPS bands.",
        depends_on=(Dep(name="queue_pressure_spot_10bps_5s", kind="col"), Dep(name="queue_pressure_spot_1bps_5s", kind="col"), Dep(name="queue_pressure_spot_2bps_5s", kind="col"), Dep(name="queue_pressure_spot_5bps_5s", kind="col"),),
        feature_id=2504,
    ),

    # === derived.depth_curvature ===
    FeatureSpec(
        name="queue_pressure_depth_curvature_fut_5s",
        stage="S2",
        operator="derived.depth_curvature",
        params={'market_scope': 'Futures', 'window_s': 5, 'resample': '1s'},
        group="Meta",
        description="Second derivative of queue_pressure across BPS bands.",
        depends_on=(Dep(name="queue_pressure_fut_10bps_5s", kind="col"), Dep(name="queue_pressure_fut_1bps_5s", kind="col"), Dep(name="queue_pressure_fut_2bps_5s", kind="col"), Dep(name="queue_pressure_fut_5bps_5s", kind="col"),),
        feature_id=2505,
    ),
    FeatureSpec(
        name="queue_pressure_depth_curvature_spot_5s",
        stage="S2",
        operator="derived.depth_curvature",
        params={'market_scope': 'Spot', 'window_s': 5, 'resample': '1s'},
        group="Meta",
        description="Second derivative of queue_pressure across BPS bands.",
        depends_on=(Dep(name="queue_pressure_spot_10bps_5s", kind="col"), Dep(name="queue_pressure_spot_1bps_5s", kind="col"), Dep(name="queue_pressure_spot_2bps_5s", kind="col"), Dep(name="queue_pressure_spot_5bps_5s", kind="col"),),
        feature_id=2506,
    ),

    # === derived.depth_slope ===
    FeatureSpec(
        name="queue_pressure_depth_slope_fut_5s",
        stage="S2",
        operator="derived.depth_slope",
        params={'market_scope': 'Futures', 'window_s': 5, 'resample': '1s'},
        group="Meta",
        description="First derivative of queue_pressure across BPS bands.",
        depends_on=(Dep(name="queue_pressure_fut_10bps_5s", kind="col"), Dep(name="queue_pressure_fut_1bps_5s", kind="col"), Dep(name="queue_pressure_fut_2bps_5s", kind="col"), Dep(name="queue_pressure_fut_5bps_5s", kind="col"),),
        feature_id=2507,
    ),
    FeatureSpec(
        name="queue_pressure_depth_slope_spot_5s",
        stage="S2",
        operator="derived.depth_slope",
        params={'market_scope': 'Spot', 'window_s': 5, 'resample': '1s'},
        group="Meta",
        description="First derivative of queue_pressure across BPS bands.",
        depends_on=(Dep(name="queue_pressure_spot_10bps_5s", kind="col"), Dep(name="queue_pressure_spot_1bps_5s", kind="col"), Dep(name="queue_pressure_spot_2bps_5s", kind="col"), Dep(name="queue_pressure_spot_5bps_5s", kind="col"),),
        feature_id=2508,
    ),

    # === derived.dir_consistency ===
    FeatureSpec(
        name="dir_consistency_fut_300s",
        stage="S2",
        operator="derived.dir_consistency",
        params={'market_scope': 'Futures', 'window_s': 300, 'resample': '1s'},
        group="Meta",
        description="Directional consistency: fraction of same-sign returns in window.",
        depends_on=(Dep(name="ret_mid_fut_1s", kind="col"),),
        feature_id=2509,
    ),
    FeatureSpec(
        name="dir_consistency_fut_60s",
        stage="S2",
        operator="derived.dir_consistency",
        params={'market_scope': 'Futures', 'window_s': 60, 'resample': '1s'},
        group="Meta",
        description="Directional consistency: fraction of same-sign returns in window.",
        depends_on=(Dep(name="ret_mid_fut_1s", kind="col"),),
        feature_id=2510,
    ),
    FeatureSpec(
        name="dir_consistency_fut_900s",
        stage="S2",
        operator="derived.dir_consistency",
        params={'market_scope': 'Futures', 'window_s': 900, 'resample': '1s'},
        group="Meta",
        description="Directional consistency: fraction of same-sign returns in window.",
        depends_on=(Dep(name="ret_mid_fut_1s", kind="col"),),
        feature_id=2511,
    ),
    FeatureSpec(
        name="dir_consistency_spot_300s",
        stage="S2",
        operator="derived.dir_consistency",
        params={'market_scope': 'Spot', 'window_s': 300, 'resample': '1s'},
        group="Meta",
        description="Directional consistency: fraction of same-sign returns in window.",
        depends_on=(Dep(name="ret_mid_spot_1s", kind="col"),),
        feature_id=2512,
    ),
    FeatureSpec(
        name="dir_consistency_spot_60s",
        stage="S2",
        operator="derived.dir_consistency",
        params={'market_scope': 'Spot', 'window_s': 60, 'resample': '1s'},
        group="Meta",
        description="Directional consistency: fraction of same-sign returns in window.",
        depends_on=(Dep(name="ret_mid_spot_1s", kind="col"),),
        feature_id=2513,
    ),
    FeatureSpec(
        name="dir_consistency_spot_900s",
        stage="S2",
        operator="derived.dir_consistency",
        params={'market_scope': 'Spot', 'window_s': 900, 'resample': '1s'},
        group="Meta",
        description="Directional consistency: fraction of same-sign returns in window.",
        depends_on=(Dep(name="ret_mid_spot_1s", kind="col"),),
        feature_id=2514,
    ),

    # === derived.unidir_ratio ===
    FeatureSpec(
        name="unidir_ratio_spot_300s",
        stage="S2",
        operator="derived.unidir_ratio",
        params={'market_scope': 'Spot', 'window_s': 300, 'resample': '1s'},
        group="Meta",
        description="Ratio of unidirectional moves to total in window.",
        depends_on=(Dep(name="ret_mid_spot_1s", kind="col"),),
        feature_id=2515,
    ),
    FeatureSpec(
        name="unidir_ratio_spot_60s",
        stage="S2",
        operator="derived.unidir_ratio",
        params={'market_scope': 'Spot', 'window_s': 60, 'resample': '1s'},
        group="Meta",
        description="Ratio of unidirectional moves to total in window.",
        depends_on=(Dep(name="ret_mid_spot_1s", kind="col"),),
        feature_id=2516,
    ),
]