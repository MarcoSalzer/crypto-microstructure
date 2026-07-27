# S4 Meta Features
# ==============================================================================
# Overview:
#   Cross-depth structural analysis of net_pressure across all BPS bands.
#   Three sub-families:
#
#   1) depth_coherence: Sign agreement fraction across {1,2,5,10}bps bands.
#      Measures whether net pressure is directionally aligned at all depth
#      levels (high coherence = strong signal).
#   2) depth_slope: OLS slope of net_pressure across log(depth) bands.
#      Positive slope = pressure increasing with depth (building support).
#   3) depth_curvature: Second derivative across depth bands.
#      Captures convexity/concavity of the pressure profile.
#
# Features (18):
#   - net_pressure_depth_coherence_{fut,spot}_{5,15,60}s (6)
#   - net_pressure_depth_slope_{fut,spot}_{5,15,60}s (6)
#   - net_pressure_depth_curvature_{fut,spot}_{5,15,60}s (6)
#
# Removed (were duplicates of S3 outputs):
#   - vacuum_score_{fut,spot}_{5,10}bps_{5,15,60}s (12 features, ids 4171–4182)
#   All 12 had identical deps and semantics to s3_meta.py versions (ids 3189–3206).
#   The S4_dynamics median/mad/shock chain consumes the S3 columns directly
#   from the cumulative DataFrame without needing S4 to recompute them.
#
# Operators used:
#   derived.depth_coherence, derived.depth_slope, derived.depth_curvature
#
# Dependencies: S3 net_pressure at {1,2,5,10}bps
# ==============================================================================

from typing import List
from etl.spec import FeatureSpec, Dep

S4_META_FEATURES: List[FeatureSpec] = [

    # === derived.depth_coherence ===

    FeatureSpec(
        name="net_pressure_depth_coherence_fut_15s",
        stage="S4",
        operator="derived.depth_coherence",
        params={"market_scope": "Futures",
                 "depth_bands_bps": "1,2,5,10",
                 "input_col_0": "net_pressure_fut_10bps_15s",
                 "input_col_1": "net_pressure_fut_1bps_15s",
                 "input_col_2": "net_pressure_fut_2bps_15s",
                 "input_col_3": "net_pressure_fut_5bps_15s"},
        label="Net Pressure Depth Coherence Fut 15S (Binance)",
        group="Meta",
        description="Sign agreement across depth bands {1,2,5,10}bps (Futures, 15s).",
        depends_on=(
            Dep(name="net_pressure_fut_10bps_15s", kind="col"),
            Dep(name="net_pressure_fut_1bps_15s", kind="col"),
            Dep(name="net_pressure_fut_2bps_15s", kind="col"),
            Dep(name="net_pressure_fut_5bps_15s", kind="col"),
        ),
        feature_id=4148,
    ),

    FeatureSpec(
        name="net_pressure_depth_coherence_fut_5s",
        stage="S4",
        operator="derived.depth_coherence",
        params={"market_scope": "Futures",
                 "depth_bands_bps": "1,2,5,10",
                 "input_col_0": "net_pressure_fut_10bps_5s",
                 "input_col_1": "net_pressure_fut_1bps_5s",
                 "input_col_2": "net_pressure_fut_2bps_5s",
                 "input_col_3": "net_pressure_fut_5bps_5s"},
        label="Net Pressure Depth Coherence Fut 5S (Binance)",
        group="Meta",
        description="Sign agreement across depth bands {1,2,5,10}bps (Futures, 5s).",
        depends_on=(
            Dep(name="net_pressure_fut_10bps_5s", kind="col"),
            Dep(name="net_pressure_fut_1bps_5s", kind="col"),
            Dep(name="net_pressure_fut_2bps_5s", kind="col"),
            Dep(name="net_pressure_fut_5bps_5s", kind="col"),
        ),
        feature_id=4149,
    ),

    FeatureSpec(
        name="net_pressure_depth_coherence_fut_60s",
        stage="S4",
        operator="derived.depth_coherence",
        params={"market_scope": "Futures",
                 "depth_bands_bps": "1,2,5,10",
                 "input_col_0": "net_pressure_fut_10bps_60s",
                 "input_col_1": "net_pressure_fut_1bps_60s",
                 "input_col_2": "net_pressure_fut_2bps_60s",
                 "input_col_3": "net_pressure_fut_5bps_60s"},
        label="Net Pressure Depth Coherence Fut 60S (Binance)",
        group="Meta",
        description="Sign agreement across depth bands {1,2,5,10}bps (Futures, 60s).",
        depends_on=(
            Dep(name="net_pressure_fut_10bps_60s", kind="col"),
            Dep(name="net_pressure_fut_1bps_60s", kind="col"),
            Dep(name="net_pressure_fut_2bps_60s", kind="col"),
            Dep(name="net_pressure_fut_5bps_60s", kind="col"),
        ),
        feature_id=4150,
    ),

    FeatureSpec(
        name="net_pressure_depth_coherence_spot_15s",
        stage="S4",
        operator="derived.depth_coherence",
        params={"market_scope": "Spot",
                 "depth_bands_bps": "1,2,5,10",
                 "input_col_0": "net_pressure_spot_10bps_15s",
                 "input_col_1": "net_pressure_spot_1bps_15s",
                 "input_col_2": "net_pressure_spot_2bps_15s",
                 "input_col_3": "net_pressure_spot_5bps_15s"},
        label="Net Pressure Depth Coherence Spot 15S (Binance)",
        group="Meta",
        description="Sign agreement across depth bands {1,2,5,10}bps (Spot, 15s).",
        depends_on=(
            Dep(name="net_pressure_spot_10bps_15s", kind="col"),
            Dep(name="net_pressure_spot_1bps_15s", kind="col"),
            Dep(name="net_pressure_spot_2bps_15s", kind="col"),
            Dep(name="net_pressure_spot_5bps_15s", kind="col"),
        ),
        feature_id=4151,
    ),

    FeatureSpec(
        name="net_pressure_depth_coherence_spot_5s",
        stage="S4",
        operator="derived.depth_coherence",
        params={"market_scope": "Spot",
                 "depth_bands_bps": "1,2,5,10",
                 "input_col_0": "net_pressure_spot_10bps_5s",
                 "input_col_1": "net_pressure_spot_1bps_5s",
                 "input_col_2": "net_pressure_spot_2bps_5s",
                 "input_col_3": "net_pressure_spot_5bps_5s"},
        label="Net Pressure Depth Coherence Spot 5S (Binance)",
        group="Meta",
        description="Sign agreement across depth bands {1,2,5,10}bps (Spot, 5s).",
        depends_on=(
            Dep(name="net_pressure_spot_10bps_5s", kind="col"),
            Dep(name="net_pressure_spot_1bps_5s", kind="col"),
            Dep(name="net_pressure_spot_2bps_5s", kind="col"),
            Dep(name="net_pressure_spot_5bps_5s", kind="col"),
        ),
        feature_id=4152,
    ),

    FeatureSpec(
        name="net_pressure_depth_coherence_spot_60s",
        stage="S4",
        operator="derived.depth_coherence",
        params={"market_scope": "Spot",
                 "depth_bands_bps": "1,2,5,10",
                 "input_col_0": "net_pressure_spot_10bps_60s",
                 "input_col_1": "net_pressure_spot_1bps_60s",
                 "input_col_2": "net_pressure_spot_2bps_60s",
                 "input_col_3": "net_pressure_spot_5bps_60s"},
        label="Net Pressure Depth Coherence Spot 60S (Binance)",
        group="Meta",
        description="Sign agreement across depth bands {1,2,5,10}bps (Spot, 60s).",
        depends_on=(
            Dep(name="net_pressure_spot_10bps_60s", kind="col"),
            Dep(name="net_pressure_spot_1bps_60s", kind="col"),
            Dep(name="net_pressure_spot_2bps_60s", kind="col"),
            Dep(name="net_pressure_spot_5bps_60s", kind="col"),
        ),
        feature_id=4153,
    ),


    # === derived.depth_curvature ===

    FeatureSpec(
        name="net_pressure_depth_curvature_fut_15s",
        stage="S4",
        operator="derived.depth_curvature",
        params={"market_scope": "Futures",
                 "depth_bands_bps": "1,2,5,10",
                 "input_col_0": "net_pressure_fut_10bps_15s",
                 "input_col_1": "net_pressure_fut_1bps_15s",
                 "input_col_2": "net_pressure_fut_2bps_15s",
                 "input_col_3": "net_pressure_fut_5bps_15s"},
        label="Net Pressure Depth Curvature Fut 15S (Binance)",
        group="Meta",
        description="Second derivative (curvature) of net pressure across depth bands {1,2,5,10}bps (Futures, 15s).",
        depends_on=(
            Dep(name="net_pressure_fut_10bps_15s", kind="col"),
            Dep(name="net_pressure_fut_1bps_15s", kind="col"),
            Dep(name="net_pressure_fut_2bps_15s", kind="col"),
            Dep(name="net_pressure_fut_5bps_15s", kind="col"),
        ),
        feature_id=4154,
    ),

    FeatureSpec(
        name="net_pressure_depth_curvature_fut_5s",
        stage="S4",
        operator="derived.depth_curvature",
        params={"market_scope": "Futures",
                 "depth_bands_bps": "1,2,5,10",
                 "input_col_0": "net_pressure_fut_10bps_5s",
                 "input_col_1": "net_pressure_fut_1bps_5s",
                 "input_col_2": "net_pressure_fut_2bps_5s",
                 "input_col_3": "net_pressure_fut_5bps_5s"},
        label="Net Pressure Depth Curvature Fut 5S (Binance)",
        group="Meta",
        description="Second derivative (curvature) of net pressure across depth bands {1,2,5,10}bps (Futures, 5s).",
        depends_on=(
            Dep(name="net_pressure_fut_10bps_5s", kind="col"),
            Dep(name="net_pressure_fut_1bps_5s", kind="col"),
            Dep(name="net_pressure_fut_2bps_5s", kind="col"),
            Dep(name="net_pressure_fut_5bps_5s", kind="col"),
        ),
        feature_id=4155,
    ),

    FeatureSpec(
        name="net_pressure_depth_curvature_fut_60s",
        stage="S4",
        operator="derived.depth_curvature",
        params={"market_scope": "Futures",
                 "depth_bands_bps": "1,2,5,10",
                 "input_col_0": "net_pressure_fut_10bps_60s",
                 "input_col_1": "net_pressure_fut_1bps_60s",
                 "input_col_2": "net_pressure_fut_2bps_60s",
                 "input_col_3": "net_pressure_fut_5bps_60s"},
        label="Net Pressure Depth Curvature Fut 60S (Binance)",
        group="Meta",
        description="Second derivative (curvature) of net pressure across depth bands {1,2,5,10}bps (Futures, 60s).",
        depends_on=(
            Dep(name="net_pressure_fut_10bps_60s", kind="col"),
            Dep(name="net_pressure_fut_1bps_60s", kind="col"),
            Dep(name="net_pressure_fut_2bps_60s", kind="col"),
            Dep(name="net_pressure_fut_5bps_60s", kind="col"),
        ),
        feature_id=4156,
    ),

    FeatureSpec(
        name="net_pressure_depth_curvature_spot_15s",
        stage="S4",
        operator="derived.depth_curvature",
        params={"market_scope": "Spot",
                 "depth_bands_bps": "1,2,5,10",
                 "input_col_0": "net_pressure_spot_10bps_15s",
                 "input_col_1": "net_pressure_spot_1bps_15s",
                 "input_col_2": "net_pressure_spot_2bps_15s",
                 "input_col_3": "net_pressure_spot_5bps_15s"},
        label="Net Pressure Depth Curvature Spot 15S (Binance)",
        group="Meta",
        description="Second derivative (curvature) of net pressure across depth bands {1,2,5,10}bps (Spot, 15s).",
        depends_on=(
            Dep(name="net_pressure_spot_10bps_15s", kind="col"),
            Dep(name="net_pressure_spot_1bps_15s", kind="col"),
            Dep(name="net_pressure_spot_2bps_15s", kind="col"),
            Dep(name="net_pressure_spot_5bps_15s", kind="col"),
        ),
        feature_id=4157,
    ),

    FeatureSpec(
        name="net_pressure_depth_curvature_spot_5s",
        stage="S4",
        operator="derived.depth_curvature",
        params={"market_scope": "Spot",
                 "depth_bands_bps": "1,2,5,10",
                 "input_col_0": "net_pressure_spot_10bps_5s",
                 "input_col_1": "net_pressure_spot_1bps_5s",
                 "input_col_2": "net_pressure_spot_2bps_5s",
                 "input_col_3": "net_pressure_spot_5bps_5s"},
        label="Net Pressure Depth Curvature Spot 5S (Binance)",
        group="Meta",
        description="Second derivative (curvature) of net pressure across depth bands {1,2,5,10}bps (Spot, 5s).",
        depends_on=(
            Dep(name="net_pressure_spot_10bps_5s", kind="col"),
            Dep(name="net_pressure_spot_1bps_5s", kind="col"),
            Dep(name="net_pressure_spot_2bps_5s", kind="col"),
            Dep(name="net_pressure_spot_5bps_5s", kind="col"),
        ),
        feature_id=4158,
    ),

    FeatureSpec(
        name="net_pressure_depth_curvature_spot_60s",
        stage="S4",
        operator="derived.depth_curvature",
        params={"market_scope": "Spot",
                 "depth_bands_bps": "1,2,5,10",
                 "input_col_0": "net_pressure_spot_10bps_60s",
                 "input_col_1": "net_pressure_spot_1bps_60s",
                 "input_col_2": "net_pressure_spot_2bps_60s",
                 "input_col_3": "net_pressure_spot_5bps_60s"},
        label="Net Pressure Depth Curvature Spot 60S (Binance)",
        group="Meta",
        description="Second derivative (curvature) of net pressure across depth bands {1,2,5,10}bps (Spot, 60s).",
        depends_on=(
            Dep(name="net_pressure_spot_10bps_60s", kind="col"),
            Dep(name="net_pressure_spot_1bps_60s", kind="col"),
            Dep(name="net_pressure_spot_2bps_60s", kind="col"),
            Dep(name="net_pressure_spot_5bps_60s", kind="col"),
        ),
        feature_id=4159,
    ),


    # === derived.depth_slope ===

    FeatureSpec(
        name="net_pressure_depth_slope_fut_15s",
        stage="S4",
        operator="derived.depth_slope",
        params={"market_scope": "Futures",
                 "depth_bands_bps": "1,2,5,10",
                 "input_col_0": "net_pressure_fut_10bps_15s",
                 "input_col_1": "net_pressure_fut_1bps_15s",
                 "input_col_2": "net_pressure_fut_2bps_15s",
                 "input_col_3": "net_pressure_fut_5bps_15s"},
        label="Net Pressure Depth Slope Fut 15S (Binance)",
        group="Meta",
        description="OLS slope of net pressure across log(depth) bands {1,2,5,10}bps (Futures, 15s).",
        depends_on=(
            Dep(name="net_pressure_fut_10bps_15s", kind="col"),
            Dep(name="net_pressure_fut_1bps_15s", kind="col"),
            Dep(name="net_pressure_fut_2bps_15s", kind="col"),
            Dep(name="net_pressure_fut_5bps_15s", kind="col"),
        ),
        feature_id=4160,
    ),

    FeatureSpec(
        name="net_pressure_depth_slope_fut_5s",
        stage="S4",
        operator="derived.depth_slope",
        params={"market_scope": "Futures",
                 "depth_bands_bps": "1,2,5,10",
                 "input_col_0": "net_pressure_fut_10bps_5s",
                 "input_col_1": "net_pressure_fut_1bps_5s",
                 "input_col_2": "net_pressure_fut_2bps_5s",
                 "input_col_3": "net_pressure_fut_5bps_5s"},
        label="Net Pressure Depth Slope Fut 5S (Binance)",
        group="Meta",
        description="OLS slope of net pressure across log(depth) bands {1,2,5,10}bps (Futures, 5s).",
        depends_on=(
            Dep(name="net_pressure_fut_10bps_5s", kind="col"),
            Dep(name="net_pressure_fut_1bps_5s", kind="col"),
            Dep(name="net_pressure_fut_2bps_5s", kind="col"),
            Dep(name="net_pressure_fut_5bps_5s", kind="col"),
        ),
        feature_id=4161,
    ),

    FeatureSpec(
        name="net_pressure_depth_slope_fut_60s",
        stage="S4",
        operator="derived.depth_slope",
        params={"market_scope": "Futures",
                 "depth_bands_bps": "1,2,5,10",
                 "input_col_0": "net_pressure_fut_10bps_60s",
                 "input_col_1": "net_pressure_fut_1bps_60s",
                 "input_col_2": "net_pressure_fut_2bps_60s",
                 "input_col_3": "net_pressure_fut_5bps_60s"},
        label="Net Pressure Depth Slope Fut 60S (Binance)",
        group="Meta",
        description="OLS slope of net pressure across log(depth) bands {1,2,5,10}bps (Futures, 60s).",
        depends_on=(
            Dep(name="net_pressure_fut_10bps_60s", kind="col"),
            Dep(name="net_pressure_fut_1bps_60s", kind="col"),
            Dep(name="net_pressure_fut_2bps_60s", kind="col"),
            Dep(name="net_pressure_fut_5bps_60s", kind="col"),
        ),
        feature_id=4162,
    ),

    FeatureSpec(
        name="net_pressure_depth_slope_spot_15s",
        stage="S4",
        operator="derived.depth_slope",
        params={"market_scope": "Spot",
                 "depth_bands_bps": "1,2,5,10",
                 "input_col_0": "net_pressure_spot_10bps_15s",
                 "input_col_1": "net_pressure_spot_1bps_15s",
                 "input_col_2": "net_pressure_spot_2bps_15s",
                 "input_col_3": "net_pressure_spot_5bps_15s"},
        label="Net Pressure Depth Slope Spot 15S (Binance)",
        group="Meta",
        description="OLS slope of net pressure across log(depth) bands {1,2,5,10}bps (Spot, 15s).",
        depends_on=(
            Dep(name="net_pressure_spot_10bps_15s", kind="col"),
            Dep(name="net_pressure_spot_1bps_15s", kind="col"),
            Dep(name="net_pressure_spot_2bps_15s", kind="col"),
            Dep(name="net_pressure_spot_5bps_15s", kind="col"),
        ),
        feature_id=4163,
    ),

    FeatureSpec(
        name="net_pressure_depth_slope_spot_5s",
        stage="S4",
        operator="derived.depth_slope",
        params={"market_scope": "Spot",
                 "depth_bands_bps": "1,2,5,10",
                 "input_col_0": "net_pressure_spot_10bps_5s",
                 "input_col_1": "net_pressure_spot_1bps_5s",
                 "input_col_2": "net_pressure_spot_2bps_5s",
                 "input_col_3": "net_pressure_spot_5bps_5s"},
        label="Net Pressure Depth Slope Spot 5S (Binance)",
        group="Meta",
        description="OLS slope of net pressure across log(depth) bands {1,2,5,10}bps (Spot, 5s).",
        depends_on=(
            Dep(name="net_pressure_spot_10bps_5s", kind="col"),
            Dep(name="net_pressure_spot_1bps_5s", kind="col"),
            Dep(name="net_pressure_spot_2bps_5s", kind="col"),
            Dep(name="net_pressure_spot_5bps_5s", kind="col"),
        ),
        feature_id=4164,
    ),

    FeatureSpec(
        name="net_pressure_depth_slope_spot_60s",
        stage="S4",
        operator="derived.depth_slope",
        params={"market_scope": "Spot",
                 "depth_bands_bps": "1,2,5,10",
                 "input_col_0": "net_pressure_spot_10bps_60s",
                 "input_col_1": "net_pressure_spot_1bps_60s",
                 "input_col_2": "net_pressure_spot_2bps_60s",
                 "input_col_3": "net_pressure_spot_5bps_60s"},
        label="Net Pressure Depth Slope Spot 60S (Binance)",
        group="Meta",
        description="OLS slope of net pressure across log(depth) bands {1,2,5,10}bps (Spot, 60s).",
        depends_on=(
            Dep(name="net_pressure_spot_10bps_60s", kind="col"),
            Dep(name="net_pressure_spot_1bps_60s", kind="col"),
            Dep(name="net_pressure_spot_2bps_60s", kind="col"),
            Dep(name="net_pressure_spot_5bps_60s", kind="col"),
        ),
        feature_id=4165,
    ),

    # NOTE: vacuum_score_{fut,spot}_{5,10}bps_{5,15,60}s are NOT defined here.
    # They are produced by S3 (s3_meta.py) with identical deps (z_pull_rate,
    # z_refill_rate). The S4_dynamics median/mad/shock chain references them
    # as Deps and finds the S3 columns in the cumulative DataFrame.

]