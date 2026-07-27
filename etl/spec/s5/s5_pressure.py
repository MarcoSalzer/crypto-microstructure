# etl/spec/s5/s5_pressure.py
# ==============================================================================
# S5 Pressure Features
# ==============================================================================
# Overview:
#   Robust z-score normalisations of per-market net-add and net-cancel
#   rolling-sum pressure signals (from S4). These features transform the
#   raw rolling sums into scale-free deviation scores that are comparable
#   across different depth bands, timeframes, and market sessions.
#
#   Operator: derived.robust_zscore (self-contained inline computation):
#     z = (x - rolling_median(x)) / (1.4826 * rolling_MAD(x) + eps)
#   Window: matched to the timeframe suffix of the input column
#     (15s → window_s=15, 60s → window_s=60).
#
#   Two sub-families:
#   1) net_add_robust_z: Normalised net-add rolling sums.
#      {fut,spot} × {2,5,10}bps × {15,60}s → 12 features
#   2) net_cancel_robust_z: Normalised net-cancel rolling sums.
#      {fut,spot} × {2,5,10}bps × {15,60}s → 12 features
#
# Features (24 total):
#   - net_add_robust_z_{fut,spot}_{2,5,10}bps_{15,60}s    (12)
#   - net_cancel_robust_z_{fut,spot}_{2,5,10}bps_{15,60}s (12)
#
# Operators used:
#   derived.robust_zscore — self-contained inline robust z-score
#
# Input (S4) columns:
#   net_add_{fut,spot}_{2,5,10}bps_{15,60}s
#   net_cancel_{fut,spot}_{2,5,10}bps_{15,60}s
#
# Intra-S5 dependencies: none — all inputs are S4 columns.
# ==============================================================================

from typing import List
from etl.spec import FeatureSpec, Dep

S5_PRESSURE_FEATURES: List[FeatureSpec] = [

    # =========================================================================
    # NET-ADD ROBUST Z-SCORE — Futures
    # =========================================================================

    FeatureSpec(
        name="net_add_robust_z_fut_2bps_15s",
        stage="S5",
        operator="derived.robust_zscore",
        params={
            "market_scope": "Futures",
            "input_col": "net_add_fut_2bps_15s",
            "window_s": "15",
        },
        label="Net Add Robust Z Fut 2Bps 15S (Binance)",
        group="Pressure",
        description="Robust z-score of net_add_fut_2bps_15s (Futures, 2bps, 15s).",
        depends_on=(Dep(name="net_add_fut_2bps_15s", kind="col"),),
        feature_id=5034,
    ),

    FeatureSpec(
        name="net_add_robust_z_fut_2bps_60s",
        stage="S5",
        operator="derived.robust_zscore",
        params={
            "market_scope": "Futures",
            "input_col": "net_add_fut_2bps_60s",
            "window_s": "60",
        },
        label="Net Add Robust Z Fut 2Bps 60S (Binance)",
        group="Pressure",
        description="Robust z-score of net_add_fut_2bps_60s (Futures, 2bps, 60s).",
        depends_on=(Dep(name="net_add_fut_2bps_60s", kind="col"),),
        feature_id=5035,
    ),

    FeatureSpec(
        name="net_add_robust_z_fut_5bps_15s",
        stage="S5",
        operator="derived.robust_zscore",
        params={
            "market_scope": "Futures",
            "input_col": "net_add_fut_5bps_15s",
            "window_s": "15",
        },
        label="Net Add Robust Z Fut 5Bps 15S (Binance)",
        group="Pressure",
        description="Robust z-score of net_add_fut_5bps_15s (Futures, 5bps, 15s).",
        depends_on=(Dep(name="net_add_fut_5bps_15s", kind="col"),),
        feature_id=5036,
    ),

    FeatureSpec(
        name="net_add_robust_z_fut_5bps_60s",
        stage="S5",
        operator="derived.robust_zscore",
        params={
            "market_scope": "Futures",
            "input_col": "net_add_fut_5bps_60s",
            "window_s": "60",
        },
        label="Net Add Robust Z Fut 5Bps 60S (Binance)",
        group="Pressure",
        description="Robust z-score of net_add_fut_5bps_60s (Futures, 5bps, 60s).",
        depends_on=(Dep(name="net_add_fut_5bps_60s", kind="col"),),
        feature_id=5037,
    ),

    FeatureSpec(
        name="net_add_robust_z_fut_10bps_15s",
        stage="S5",
        operator="derived.robust_zscore",
        params={
            "market_scope": "Futures",
            "input_col": "net_add_fut_10bps_15s",
            "window_s": "15",
        },
        label="Net Add Robust Z Fut 10Bps 15S (Binance)",
        group="Pressure",
        description="Robust z-score of net_add_fut_10bps_15s (Futures, 10bps, 15s).",
        depends_on=(Dep(name="net_add_fut_10bps_15s", kind="col"),),
        feature_id=5038,
    ),

    FeatureSpec(
        name="net_add_robust_z_fut_10bps_60s",
        stage="S5",
        operator="derived.robust_zscore",
        params={
            "market_scope": "Futures",
            "input_col": "net_add_fut_10bps_60s",
            "window_s": "60",
        },
        label="Net Add Robust Z Fut 10Bps 60S (Binance)",
        group="Pressure",
        description="Robust z-score of net_add_fut_10bps_60s (Futures, 10bps, 60s).",
        depends_on=(Dep(name="net_add_fut_10bps_60s", kind="col"),),
        feature_id=5039,
    ),

    # =========================================================================
    # NET-ADD ROBUST Z-SCORE — Spot
    # =========================================================================

    FeatureSpec(
        name="net_add_robust_z_spot_2bps_15s",
        stage="S5",
        operator="derived.robust_zscore",
        params={
            "market_scope": "Spot",
            "input_col": "net_add_spot_2bps_15s",
            "window_s": "15",
        },
        label="Net Add Robust Z Spot 2Bps 15S (Binance)",
        group="Pressure",
        description="Robust z-score of net_add_spot_2bps_15s (Spot, 2bps, 15s).",
        depends_on=(Dep(name="net_add_spot_2bps_15s", kind="col"),),
        feature_id=5040,
    ),

    FeatureSpec(
        name="net_add_robust_z_spot_2bps_60s",
        stage="S5",
        operator="derived.robust_zscore",
        params={
            "market_scope": "Spot",
            "input_col": "net_add_spot_2bps_60s",
            "window_s": "60",
        },
        label="Net Add Robust Z Spot 2Bps 60S (Binance)",
        group="Pressure",
        description="Robust z-score of net_add_spot_2bps_60s (Spot, 2bps, 60s).",
        depends_on=(Dep(name="net_add_spot_2bps_60s", kind="col"),),
        feature_id=5041,
    ),

    FeatureSpec(
        name="net_add_robust_z_spot_5bps_15s",
        stage="S5",
        operator="derived.robust_zscore",
        params={
            "market_scope": "Spot",
            "input_col": "net_add_spot_5bps_15s",
            "window_s": "15",
        },
        label="Net Add Robust Z Spot 5Bps 15S (Binance)",
        group="Pressure",
        description="Robust z-score of net_add_spot_5bps_15s (Spot, 5bps, 15s).",
        depends_on=(Dep(name="net_add_spot_5bps_15s", kind="col"),),
        feature_id=5042,
    ),

    FeatureSpec(
        name="net_add_robust_z_spot_5bps_60s",
        stage="S5",
        operator="derived.robust_zscore",
        params={
            "market_scope": "Spot",
            "input_col": "net_add_spot_5bps_60s",
            "window_s": "60",
        },
        label="Net Add Robust Z Spot 5Bps 60S (Binance)",
        group="Pressure",
        description="Robust z-score of net_add_spot_5bps_60s (Spot, 5bps, 60s).",
        depends_on=(Dep(name="net_add_spot_5bps_60s", kind="col"),),
        feature_id=5043,
    ),

    FeatureSpec(
        name="net_add_robust_z_spot_10bps_15s",
        stage="S5",
        operator="derived.robust_zscore",
        params={
            "market_scope": "Spot",
            "input_col": "net_add_spot_10bps_15s",
            "window_s": "15",
        },
        label="Net Add Robust Z Spot 10Bps 15S (Binance)",
        group="Pressure",
        description="Robust z-score of net_add_spot_10bps_15s (Spot, 10bps, 15s).",
        depends_on=(Dep(name="net_add_spot_10bps_15s", kind="col"),),
        feature_id=5044,
    ),

    FeatureSpec(
        name="net_add_robust_z_spot_10bps_60s",
        stage="S5",
        operator="derived.robust_zscore",
        params={
            "market_scope": "Spot",
            "input_col": "net_add_spot_10bps_60s",
            "window_s": "60",
        },
        label="Net Add Robust Z Spot 10Bps 60S (Binance)",
        group="Pressure",
        description="Robust z-score of net_add_spot_10bps_60s (Spot, 10bps, 60s).",
        depends_on=(Dep(name="net_add_spot_10bps_60s", kind="col"),),
        feature_id=5045,
    ),

    # =========================================================================
    # NET-CANCEL ROBUST Z-SCORE — Futures
    # =========================================================================

    FeatureSpec(
        name="net_cancel_robust_z_fut_2bps_15s",
        stage="S5",
        operator="derived.robust_zscore",
        params={
            "market_scope": "Futures",
            "input_col": "net_cancel_fut_2bps_15s",
            "window_s": "15",
        },
        label="Net Cancel Robust Z Fut 2Bps 15S (Binance)",
        group="Pressure",
        description="Robust z-score of net_cancel_fut_2bps_15s (Futures, 2bps, 15s).",
        depends_on=(Dep(name="net_cancel_fut_2bps_15s", kind="col"),),
        feature_id=5046,
    ),

    FeatureSpec(
        name="net_cancel_robust_z_fut_2bps_60s",
        stage="S5",
        operator="derived.robust_zscore",
        params={
            "market_scope": "Futures",
            "input_col": "net_cancel_fut_2bps_60s",
            "window_s": "60",
        },
        label="Net Cancel Robust Z Fut 2Bps 60S (Binance)",
        group="Pressure",
        description="Robust z-score of net_cancel_fut_2bps_60s (Futures, 2bps, 60s).",
        depends_on=(Dep(name="net_cancel_fut_2bps_60s", kind="col"),),
        feature_id=5047,
    ),

    FeatureSpec(
        name="net_cancel_robust_z_fut_5bps_15s",
        stage="S5",
        operator="derived.robust_zscore",
        params={
            "market_scope": "Futures",
            "input_col": "net_cancel_fut_5bps_15s",
            "window_s": "15",
        },
        label="Net Cancel Robust Z Fut 5Bps 15S (Binance)",
        group="Pressure",
        description="Robust z-score of net_cancel_fut_5bps_15s (Futures, 5bps, 15s).",
        depends_on=(Dep(name="net_cancel_fut_5bps_15s", kind="col"),),
        feature_id=5048,
    ),

    FeatureSpec(
        name="net_cancel_robust_z_fut_5bps_60s",
        stage="S5",
        operator="derived.robust_zscore",
        params={
            "market_scope": "Futures",
            "input_col": "net_cancel_fut_5bps_60s",
            "window_s": "60",
        },
        label="Net Cancel Robust Z Fut 5Bps 60S (Binance)",
        group="Pressure",
        description="Robust z-score of net_cancel_fut_5bps_60s (Futures, 5bps, 60s).",
        depends_on=(Dep(name="net_cancel_fut_5bps_60s", kind="col"),),
        feature_id=5049,
    ),

    FeatureSpec(
        name="net_cancel_robust_z_fut_10bps_15s",
        stage="S5",
        operator="derived.robust_zscore",
        params={
            "market_scope": "Futures",
            "input_col": "net_cancel_fut_10bps_15s",
            "window_s": "15",
        },
        label="Net Cancel Robust Z Fut 10Bps 15S (Binance)",
        group="Pressure",
        description="Robust z-score of net_cancel_fut_10bps_15s (Futures, 10bps, 15s).",
        depends_on=(Dep(name="net_cancel_fut_10bps_15s", kind="col"),),
        feature_id=5050,
    ),

    FeatureSpec(
        name="net_cancel_robust_z_fut_10bps_60s",
        stage="S5",
        operator="derived.robust_zscore",
        params={
            "market_scope": "Futures",
            "input_col": "net_cancel_fut_10bps_60s",
            "window_s": "60",
        },
        label="Net Cancel Robust Z Fut 10Bps 60S (Binance)",
        group="Pressure",
        description="Robust z-score of net_cancel_fut_10bps_60s (Futures, 10bps, 60s).",
        depends_on=(Dep(name="net_cancel_fut_10bps_60s", kind="col"),),
        feature_id=5051,
    ),

    # =========================================================================
    # NET-CANCEL ROBUST Z-SCORE — Spot
    # =========================================================================

    FeatureSpec(
        name="net_cancel_robust_z_spot_2bps_15s",
        stage="S5",
        operator="derived.robust_zscore",
        params={
            "market_scope": "Spot",
            "input_col": "net_cancel_spot_2bps_15s",
            "window_s": "15",
        },
        label="Net Cancel Robust Z Spot 2Bps 15S (Binance)",
        group="Pressure",
        description="Robust z-score of net_cancel_spot_2bps_15s (Spot, 2bps, 15s).",
        depends_on=(Dep(name="net_cancel_spot_2bps_15s", kind="col"),),
        feature_id=5052,
    ),

    FeatureSpec(
        name="net_cancel_robust_z_spot_2bps_60s",
        stage="S5",
        operator="derived.robust_zscore",
        params={
            "market_scope": "Spot",
            "input_col": "net_cancel_spot_2bps_60s",
            "window_s": "60",
        },
        label="Net Cancel Robust Z Spot 2Bps 60S (Binance)",
        group="Pressure",
        description="Robust z-score of net_cancel_spot_2bps_60s (Spot, 2bps, 60s).",
        depends_on=(Dep(name="net_cancel_spot_2bps_60s", kind="col"),),
        feature_id=5053,
    ),

    FeatureSpec(
        name="net_cancel_robust_z_spot_5bps_15s",
        stage="S5",
        operator="derived.robust_zscore",
        params={
            "market_scope": "Spot",
            "input_col": "net_cancel_spot_5bps_15s",
            "window_s": "15",
        },
        label="Net Cancel Robust Z Spot 5Bps 15S (Binance)",
        group="Pressure",
        description="Robust z-score of net_cancel_spot_5bps_15s (Spot, 5bps, 15s).",
        depends_on=(Dep(name="net_cancel_spot_5bps_15s", kind="col"),),
        feature_id=5054,
    ),

    FeatureSpec(
        name="net_cancel_robust_z_spot_5bps_60s",
        stage="S5",
        operator="derived.robust_zscore",
        params={
            "market_scope": "Spot",
            "input_col": "net_cancel_spot_5bps_60s",
            "window_s": "60",
        },
        label="Net Cancel Robust Z Spot 5Bps 60S (Binance)",
        group="Pressure",
        description="Robust z-score of net_cancel_spot_5bps_60s (Spot, 5bps, 60s).",
        depends_on=(Dep(name="net_cancel_spot_5bps_60s", kind="col"),),
        feature_id=5055,
    ),

    FeatureSpec(
        name="net_cancel_robust_z_spot_10bps_15s",
        stage="S5",
        operator="derived.robust_zscore",
        params={
            "market_scope": "Spot",
            "input_col": "net_cancel_spot_10bps_15s",
            "window_s": "15",
        },
        label="Net Cancel Robust Z Spot 10Bps 15S (Binance)",
        group="Pressure",
        description="Robust z-score of net_cancel_spot_10bps_15s (Spot, 10bps, 15s).",
        depends_on=(Dep(name="net_cancel_spot_10bps_15s", kind="col"),),
        feature_id=5056,
    ),

    FeatureSpec(
        name="net_cancel_robust_z_spot_10bps_60s",
        stage="S5",
        operator="derived.robust_zscore",
        params={
            "market_scope": "Spot",
            "input_col": "net_cancel_spot_10bps_60s",
            "window_s": "60",
        },
        label="Net Cancel Robust Z Spot 10Bps 60S (Binance)",
        group="Pressure",
        description="Robust z-score of net_cancel_spot_10bps_60s (Spot, 10bps, 60s).",
        depends_on=(Dep(name="net_cancel_spot_10bps_60s", kind="col"),),
        feature_id=5057,
    ),

]