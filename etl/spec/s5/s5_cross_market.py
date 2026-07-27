# etl/spec/s5/s5_cross_market.py
# ==============================================================================
# S5 Cross-Market Features
# ==============================================================================
# Overview:
#   Signal-quality analytics for the Futures/Spot cross-market pressure ratios
#   (the "sf" family) computed in S4. Two sub-families:
#
#   1) net_add_persist_sf / net_cancel_persist_sf:
#      Directional persistence of the futures-vs-spot net-add and net-cancel
#      pressure ratios across timeframes {900s, 3600s}. Uses the S5 signal_persist
#      formula: abs(roll_mean(x)) / (roll_mean(abs(x)) + eps). Range [0, 1].
#      A value near 1 means the cross-market imbalance is sustained and
#      directionally consistent; near 0 means it is oscillating.
#
#   2) z_net_add_sf / z_net_cancel_sf:
#      Robust z-scores of the same sf ratios. Self-contained inline computation:
#      z = (x - rolling_median(x)) / (1.4826 * rolling_MAD(x) + eps).
#      Normalises the cross-market ratio to a scale-free deviation score.
#
# Features (4 total):
#   - net_add_persist_sf_5bps_900s            (1)
#   - net_cancel_persist_sf_5bps_900s         (1)
#   - z_net_add_sf_5bps_900s                  (1)
#   - z_net_cancel_sf_5bps_900s               (1)
#
# NOTE: 3600s variants removed — structural mismatch with 1h parquet context
#   window (insufficient rolling warmup). 900s covers the same directional
#   persistence signal with adequate data density.
#
# Operators used:
#   derived.signal_persist  (S5 formula: abs(roll_mean) / roll_mean(abs) + eps)
#   derived.robust_zscore   (inline median/MAD normalisation)
#
# Input (S4) columns:
#   net_add_sf_5bps_900s
#   net_cancel_sf_5bps_900s
#
# Intra-S5 dependencies: none — all inputs are S4 columns.
# ==============================================================================

from typing import List
from etl.spec import FeatureSpec, Dep

S5_CROSS_MARKET_FEATURES: List[FeatureSpec] = [

    # =========================================================================
    # derived.signal_persist — cross-market net-add persistence
    # =========================================================================

    FeatureSpec(
        name="net_add_persist_sf_5bps_900s",
        stage="S5",
        operator="derived.signal_persist",
        params={
            "market_scope": "Spot|Futures",
            "input_col": "net_add_sf_5bps_900s",
            "window_s": "900",
        },
        label="Net Add Persist SF 5Bps 900S (Binance)",
        group="Cross-Market",
        description=(
            "Directional persistence of futures/spot net-add pressure ratio "
            "(5bps, 900s). S5 formula: abs(roll_mean) / (roll_mean(abs) + eps)."
        ),
        depends_on=(Dep(name="net_add_sf_5bps_900s", kind="col"),),
        feature_id=5000,
    ),

    FeatureSpec(
        name="net_cancel_persist_sf_5bps_900s",
        stage="S5",
        operator="derived.signal_persist",
        params={
            "market_scope": "Spot|Futures",
            "input_col": "net_cancel_sf_5bps_900s",
            "window_s": "900",
        },
        label="Net Cancel Persist SF 5Bps 900S (Binance)",
        group="Cross-Market",
        description=(
            "Directional persistence of futures/spot net-cancel pressure ratio "
            "(5bps, 900s). S5 formula: abs(roll_mean) / (roll_mean(abs) + eps)."
        ),
        depends_on=(Dep(name="net_cancel_sf_5bps_900s", kind="col"),),
        feature_id=5001,
    ),

    FeatureSpec(
        name="z_net_add_sf_5bps_900s",
        stage="S5",
        operator="derived.robust_zscore",
        params={
            "market_scope": "Spot|Futures",
            "input_col": "net_add_sf_5bps_900s",
            "window_s": "900",
        },
        label="Z Net Add SF 5Bps 900S (Binance)",
        group="Cross-Market",
        description=(
            "Robust z-score of futures/spot net-add pressure ratio (5bps, 900s). "
            "Inline: (x - rolling_median) / (1.4826 * rolling_MAD + eps)."
        ),
        depends_on=(Dep(name="net_add_sf_5bps_900s", kind="col"),),
        feature_id=5002,
    ),

    FeatureSpec(
        name="z_net_cancel_sf_5bps_900s",
        stage="S5",
        operator="derived.robust_zscore",
        params={
            "market_scope": "Spot|Futures",
            "input_col": "net_cancel_sf_5bps_900s",
            "window_s": "900",
        },
        label="Z Net Cancel SF 5Bps 900S (Binance)",
        group="Cross-Market",
        description=(
            "Robust z-score of futures/spot net-cancel pressure ratio (5bps, 900s). "
            "Inline: (x - rolling_median) / (1.4826 * rolling_MAD + eps)."
        ),
        depends_on=(Dep(name="net_cancel_sf_5bps_900s", kind="col"),),
        feature_id=5003,
    ),
]