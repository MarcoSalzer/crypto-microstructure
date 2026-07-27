# S4 Cross-Market Features
# ==============================================================================
# Overview:
#   Futures-vs-Spot divergence features for structural orderbook metrics.
#   Two main families:
#     1) depth_gradient_div / liq_concentration_div: Futures MINUS Spot
#        structural metrics (struct50, struct100) across timeframes (60s-3600s).
#        Operator: derived.sub (fut - spot subtraction).
#     2) net_add_sf / net_cancel_sf: Ratio of Futures to Spot net-add and
#        net-cancel pressure at 5bps across longer timeframes (900s, 3600s).
#        Operator: derived.ratio.
#
# Features (22):
#   - depth_gradient_div_fut_minus_spot_struct{50,100}_{60,300,900,3600}s (8)
#   - liq_concentration_div_fut_minus_spot_struct{50,100}_{60,300,900,3600}s (8)
#   - net_add_sf_5bps_{900,3600}s (2)
#   - net_cancel_sf_5bps_{900,3600}s (2)
#
#   incorrectly using derived.cross_market_div (fut/spot ratio). Fixed to use
#   derived.sub (fut - spot difference), consistent with all other
#   _div_fut_minus_spot_ features and matching the feature names.
#
# Operators used:
#   - derived.sub                : fut_col - spot_col
#   - derived.ratio              : num_col / (|den_col| + eps)
#
# Dependencies: S3 depth_gradient_{fut,spot}, liq_concentration_{fut,spot},
#               net_add_{fut,spot}, net_cancel_{fut,spot}
# ==============================================================================

from typing import List
from etl.spec import FeatureSpec, Dep

S4_CROSS_MARKET_FEATURES: List[FeatureSpec] = [

    # === derived.sub (fut - spot difference) ===

    FeatureSpec(
        name="depth_gradient_div_fut_minus_spot_struct100_300s",
        stage="S4",
        operator="derived.sub",
        params={"market_scope": "Spot|Futures"},
        label="Depth Gradient Div Fut Minus Spot Struct100 300S (Binance)",
        group="Cross-Market",
        description="Futures / Spot ratio of depth gradient (struct100, 300s).",
        depends_on=(
            Dep(name="depth_gradient_fut_struct100_300s", kind="col"),
            Dep(name="depth_gradient_spot_struct100_300s", kind="col"),
        ),
        feature_id=4000,
    ),
    FeatureSpec(
        name="depth_gradient_div_fut_minus_spot_struct100_60s",
        stage="S4",
        operator="derived.sub",
        params={"market_scope": "Spot|Futures"},
        label="Depth Gradient Div Fut Minus Spot Struct100 60S (Binance)",
        group="Cross-Market",
        description="Futures / Spot ratio of depth gradient (struct100, 60s).",
        depends_on=(
            Dep(name="depth_gradient_fut_struct100_60s", kind="col"),
            Dep(name="depth_gradient_spot_struct100_60s", kind="col"),
        ),
        feature_id=4001,
    ),

    FeatureSpec(
        name="depth_gradient_div_fut_minus_spot_struct100_900s",
        stage="S4",
        operator="derived.sub",
        params={"market_scope": "Spot|Futures"},
        label="Depth Gradient Div Fut Minus Spot Struct100 900S (Binance)",
        group="Cross-Market",
        description="Futures / Spot ratio of depth gradient (struct100, 900s).",
        depends_on=(
            Dep(name="depth_gradient_fut_struct100_900s", kind="col"),
            Dep(name="depth_gradient_spot_struct100_900s", kind="col"),
        ),
        feature_id=4002,
    ),

    FeatureSpec(
        name="depth_gradient_div_fut_minus_spot_struct50_300s",
        stage="S4",
        operator="derived.sub",
        params={"market_scope": "Spot|Futures"},
        label="Depth Gradient Div Fut Minus Spot Struct50 300S (Binance)",
        group="Cross-Market",
        description="Futures / Spot ratio of depth gradient (struct50, 300s).",
        depends_on=(
            Dep(name="depth_gradient_fut_struct50_300s", kind="col"),
            Dep(name="depth_gradient_spot_struct50_300s", kind="col"),
        ),
        feature_id=4003,
    ),

    FeatureSpec(
        name="depth_gradient_div_fut_minus_spot_struct50_60s",
        stage="S4",
        operator="derived.sub",
        params={"market_scope": "Spot|Futures"},
        label="Depth Gradient Div Fut Minus Spot Struct50 60S (Binance)",
        group="Cross-Market",
        description="Futures / Spot ratio of depth gradient (struct50, 60s).",
        depends_on=(
            Dep(name="depth_gradient_fut_struct50_60s", kind="col"),
            Dep(name="depth_gradient_spot_struct50_60s", kind="col"),
        ),
        feature_id=4004,
    ),

    FeatureSpec(
        name="depth_gradient_div_fut_minus_spot_struct50_900s",
        stage="S4",
        operator="derived.sub",
        params={"market_scope": "Spot|Futures"},
        label="Depth Gradient Div Fut Minus Spot Struct50 900S (Binance)",
        group="Cross-Market",
        description="Futures / Spot ratio of depth gradient (struct50, 900s).",
        depends_on=(
            Dep(name="depth_gradient_fut_struct50_900s", kind="col"),
            Dep(name="depth_gradient_spot_struct50_900s", kind="col"),
        ),
        feature_id=4005,
    ),

    FeatureSpec(
        name="liq_concentration_div_fut_minus_spot_struct100_300s",
        stage="S4",
        operator="derived.sub",
        params={"market_scope": "Spot|Futures"},
        label="Liq Concentration Div Fut Minus Spot Struct100 300S (Binance)",
        group="Cross-Market",
        description="Futures / Spot ratio of liq concentration (struct100, 300s).",
        depends_on=(
            Dep(name="liq_concentration_fut_struct100_300s", kind="col"),
            Dep(name="liq_concentration_spot_struct100_300s", kind="col"),
        ),
        feature_id=4006,
    ),

    FeatureSpec(
        name="liq_concentration_div_fut_minus_spot_struct100_60s",
        stage="S4",
        operator="derived.sub",
        params={"market_scope": "Spot|Futures"},
        label="Liq Concentration Div Fut Minus Spot Struct100 60S (Binance)",
        group="Cross-Market",
        description="Futures / Spot ratio of liq concentration (struct100, 60s).",
        depends_on=(
            Dep(name="liq_concentration_fut_struct100_60s", kind="col"),
            Dep(name="liq_concentration_spot_struct100_60s", kind="col"),
        ),
        feature_id=4007,
    ),

    FeatureSpec(
        name="liq_concentration_div_fut_minus_spot_struct100_900s",
        stage="S4",
        operator="derived.sub",
        params={"market_scope": "Spot|Futures"},
        label="Liq Concentration Div Fut Minus Spot Struct100 900S (Binance)",
        group="Cross-Market",
        description="Futures / Spot ratio of liq concentration (struct100, 900s).",
        depends_on=(
            Dep(name="liq_concentration_fut_struct100_900s", kind="col"),
            Dep(name="liq_concentration_spot_struct100_900s", kind="col"),
        ),
        feature_id=4008,
    ),

    FeatureSpec(
        name="liq_concentration_div_fut_minus_spot_struct50_300s",
        stage="S4",
        operator="derived.sub",
        params={"market_scope": "Spot|Futures"},
        label="Liq Concentration Div Fut Minus Spot Struct50 300S (Binance)",
        group="Cross-Market",
        description="Futures / Spot ratio of liq concentration (struct50, 300s).",
        depends_on=(
            Dep(name="liq_concentration_fut_struct50_300s", kind="col"),
            Dep(name="liq_concentration_spot_struct50_300s", kind="col"),
        ),
        feature_id=4009,
    ),
    FeatureSpec(
        name="liq_concentration_div_fut_minus_spot_struct50_60s",
        stage="S4",
        operator="derived.sub",
        params={"market_scope": "Spot|Futures"},
        label="Liq Concentration Div Fut Minus Spot Struct50 60S (Binance)",
        group="Cross-Market",
        description="Futures / Spot ratio of liq concentration (struct50, 60s).",
        depends_on=(
            Dep(name="liq_concentration_fut_struct50_60s", kind="col"),
            Dep(name="liq_concentration_spot_struct50_60s", kind="col"),
        ),
        feature_id=4010,
    ),

    FeatureSpec(
        name="liq_concentration_div_fut_minus_spot_struct50_900s",
        stage="S4",
        operator="derived.sub",
        params={"market_scope": "Spot|Futures"},
        label="Liq Concentration Div Fut Minus Spot Struct50 900S (Binance)",
        group="Cross-Market",
        description="Futures / Spot ratio of liq concentration (struct50, 900s).",
        depends_on=(
            Dep(name="liq_concentration_fut_struct50_900s", kind="col"),
            Dep(name="liq_concentration_spot_struct50_900s", kind="col"),
        ),
        feature_id=4011,
    ),

    # === derived.ratio ===

    FeatureSpec(
        name="net_add_sf_5bps_900s",
        stage="S4",
        operator="derived.ratio",
        params={"market_scope": "Spot|Futures",
                 "num_col": "net_add_fut_5bps_900s",
                 "den_col": "net_add_spot_5bps_900s",
                 "eps": "1e-12",
                 "abs_den": "true"},
        label="Net Add Sf 5Bps 900S (Binance)",
        group="Cross-Market",
        description="Futures / Spot ratio of net pressure (900s).",
        depends_on=(
            Dep(name="net_add_fut_5bps_900s", kind="col"),
            Dep(name="net_add_spot_5bps_900s", kind="col"),
        ),
        feature_id=4012,
    ),

    FeatureSpec(
        name="net_cancel_sf_5bps_900s",
        stage="S4",
        operator="derived.ratio",
        params={"market_scope": "Spot|Futures",
                 "num_col": "net_cancel_fut_5bps_900s",
                 "den_col": "net_cancel_spot_5bps_900s",
                 "eps": "1e-12",
                 "abs_den": "true"},
        label="Net Cancel Sf 5Bps 900S (Binance)",
        group="Cross-Market",
        description="Futures / Spot ratio of net pressure (900s).",
        depends_on=(
            Dep(name="net_cancel_fut_5bps_900s", kind="col"),
            Dep(name="net_cancel_spot_5bps_900s", kind="col"),
        ),
        feature_id=4013,
    ),
]