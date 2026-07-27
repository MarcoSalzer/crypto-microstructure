# etl/spec/s3/s3_cross_market.py
# ========================================================================
# S3 Cross-Market Features
# Spot-vs-Futures divergence analytics that detect structural dislocations
# between the two markets. These features are critical for basis trading,
# lead-lag detection, and regime identification.
# 
# Key concepts:
#   - basis_vwap_sf: Rolling mean of VWAP-based basis (fut - spot) over
#     multiple windows. Smoothed basis for trend detection.
#   - book_asymmetry_div: Futures book asymmetry minus Spot book asymmetry.
#     Detects when one market is structurally more bid-heavy or ask-heavy.
#   - depth_gradient_div / liq_concentration_div: Cross-market divergence in
#     orderbook shape metrics at the 1s level.
#   - liq_imb_persist_sf: Persistence of the spot-futures liquidity imbalance
#     signal — how consistently the cross-market imbalance maintains its sign.
#   - queue_pressure_div / refill_vs_pull_div: Cross-market divergence in
#     queue pressure and maker behavior (refill vs pull).
#   - trade_count_sf_div / trade_count_sf_share: Relative activity divergence
#     and market share of trade counts.
#   - volume_sf_div_delta: Volume divergence delta between markets.
# 
# Dependencies: S2 basis_vwap_sf, book_asymmetry, depth_gradient, liq_concentration,
#               liq_imb_sf, queue_pressure, refill_vs_pull, trade_count, volume.
#
# Feature count: 29  (was 34; removed 5 × 3600s features)
# [FIX-3600] Removed all window_s=3600 features (book_asymmetry_div + liq_imb_persist + volume_sf _3600s).
#            3600s rolling windows require 1h prior data even with context;
#            produce 99-100% NaN on 1h Parquet files. The 900s variants
#            provide equivalent microstructure regime coverage.
# Feature ID range: see individual entries
# ========================================================================

from typing import List
from etl.spec import FeatureSpec, Dep

S3_CROSS_MARKET_FEATURES: List[FeatureSpec] = [
    FeatureSpec(
        name="basis_vwap_sf_300s",
        stage="S3",
        operator="derived.roll_mean",
        params={"market_scope": "Spot|Futures", "window_s": "300", "min_periods": 60},
        label="Cross-Market: basis_vwap_sf_300s (Spot/Futures) [300s] (Binance)",
        group="Cross-Market",
        description="Rolling mean of basis_vwap_sf_1s over 300s window.",
        depends_on=(Dep(name="basis_vwap_sf_1s", kind="col"),),
        feature_id=3040,
    ),
    FeatureSpec(
        name="basis_vwap_sf_60s",
        stage="S3",
        operator="derived.roll_mean",
        params={"market_scope": "Spot|Futures", "window_s": "60"},
        label="Cross-Market: basis_vwap_sf_60s (Spot/Futures) [60s] (Binance)",
        group="Cross-Market",
        description="Rolling mean of basis_vwap_sf_1s over 60s window.",
        depends_on=(Dep(name="basis_vwap_sf_1s", kind="col"),),
        feature_id=3041,
    ),
    FeatureSpec(
        name="basis_vwap_sf_900s",
        stage="S3",
        operator="derived.roll_mean",
        params={"market_scope": "Spot|Futures", "window_s": "900", "min_periods": 60},
        label="Cross-Market: basis_vwap_sf_900s (Spot/Futures) [900s] (Binance)",
        group="Cross-Market",
        description="Rolling mean of basis_vwap_sf_1s over 900s window.",
        depends_on=(Dep(name="basis_vwap_sf_1s", kind="col"),),
        feature_id=3042,
    ),
    FeatureSpec(
        name="book_asymmetry_div_fut_minus_spot_struct100_300s",
        stage="S3",
        operator="s3.cross_div",
        params={"market_scope": "Futures"},
        label="Cross-Market: book_asymmetry_div_fut_minus_spot_struct100_300s (Futures) [300s] (Binance)",
        group="Cross-Market",
        description="Cross-market divergence: book_asymmetry_fut_struct100_300s - book_asymmetry_spot_struct100_300s.",
        depends_on=(
            Dep(name="book_asymmetry_fut_struct100_300s", kind="col"),
            Dep(name="book_asymmetry_spot_struct100_300s", kind="col"),
        ),
        feature_id=3043,
    ),
    FeatureSpec(
        name="book_asymmetry_div_fut_minus_spot_struct100_60s",
        stage="S3",
        operator="s3.cross_div",
        params={"market_scope": "Futures"},
        label="Cross-Market: book_asymmetry_div_fut_minus_spot_struct100_60s (Futures) [60s] (Binance)",
        group="Cross-Market",
        description="Cross-market divergence: book_asymmetry_fut_struct100_60s - book_asymmetry_spot_struct100_60s.",
        depends_on=(
            Dep(name="book_asymmetry_fut_struct100_60s", kind="col"),
            Dep(name="book_asymmetry_spot_struct100_60s", kind="col"),
        ),
        feature_id=3044,
    ),
    FeatureSpec(
        name="book_asymmetry_div_fut_minus_spot_struct100_900s",
        stage="S3",
        operator="s3.cross_div",
        params={"market_scope": "Futures"},
        label="Cross-Market: book_asymmetry_div_fut_minus_spot_struct100_900s (Futures) [900s] (Binance)",
        group="Cross-Market",
        description="Cross-market divergence: book_asymmetry_fut_struct100_900s - book_asymmetry_spot_struct100_900s.",
        depends_on=(
            Dep(name="book_asymmetry_fut_struct100_900s", kind="col"),
            Dep(name="book_asymmetry_spot_struct100_900s", kind="col"),
        ),
        feature_id=3045,
    ),
    FeatureSpec(
        name="book_asymmetry_div_fut_minus_spot_struct50_300s",
        stage="S3",
        operator="s3.cross_div",
        params={"market_scope": "Futures"},
        label="Cross-Market: book_asymmetry_div_fut_minus_spot_struct50_300s (Futures) [300s] (Binance)",
        group="Cross-Market",
        description="Cross-market divergence: book_asymmetry_fut_struct50_300s - book_asymmetry_spot_struct50_300s.",
        depends_on=(
            Dep(name="book_asymmetry_fut_struct50_300s", kind="col"),
            Dep(name="book_asymmetry_spot_struct50_300s", kind="col"),
        ),
        feature_id=3046,
    ),
    FeatureSpec(
        name="book_asymmetry_div_fut_minus_spot_struct50_60s",
        stage="S3",
        operator="s3.cross_div",
        params={"market_scope": "Futures"},
        label="Cross-Market: book_asymmetry_div_fut_minus_spot_struct50_60s (Futures) [60s] (Binance)",
        group="Cross-Market",
        description="Cross-market divergence: book_asymmetry_fut_struct50_60s - book_asymmetry_spot_struct50_60s.",
        depends_on=(
            Dep(name="book_asymmetry_fut_struct50_60s", kind="col"),
            Dep(name="book_asymmetry_spot_struct50_60s", kind="col"),
        ),
        feature_id=3047,
    ),
    FeatureSpec(
        name="book_asymmetry_div_fut_minus_spot_struct50_900s",
        stage="S3",
        operator="s3.cross_div",
        params={"market_scope": "Futures"},
        label="Cross-Market: book_asymmetry_div_fut_minus_spot_struct50_900s (Futures) [900s] (Binance)",
        group="Cross-Market",
        description="Cross-market divergence: book_asymmetry_fut_struct50_900s - book_asymmetry_spot_struct50_900s.",
        depends_on=(
            Dep(name="book_asymmetry_fut_struct50_900s", kind="col"),
            Dep(name="book_asymmetry_spot_struct50_900s", kind="col"),
        ),
        feature_id=3048,
    ),
    FeatureSpec(
        name="depth_gradient_div_fut_minus_spot_struct100_1s",
        stage="S3",
        operator="s3.cross_div",
        params={"market_scope": "Futures"},
        label="Cross-Market: depth_gradient_div_fut_minus_spot_struct100_1s (Futures) [1s] (Binance)",
        group="Cross-Market",
        description="Cross-market divergence: depth_gradient_fut_struct100_1s - depth_gradient_spot_struct100_1s.",
        depends_on=(
            Dep(name="depth_gradient_fut_struct100_1s", kind="col"),
            Dep(name="depth_gradient_spot_struct100_1s", kind="col"),
        ),
        feature_id=3049,
    ),
    FeatureSpec(
        name="depth_gradient_div_fut_minus_spot_struct50_1s",
        stage="S3",
        operator="s3.cross_div",
        params={"market_scope": "Futures"},
        label="Cross-Market: depth_gradient_div_fut_minus_spot_struct50_1s (Futures) [1s] (Binance)",
        group="Cross-Market",
        description="Cross-market divergence: depth_gradient_fut_struct50_1s - depth_gradient_spot_struct50_1s.",
        depends_on=(
            Dep(name="depth_gradient_fut_struct50_1s", kind="col"),
            Dep(name="depth_gradient_spot_struct50_1s", kind="col"),
        ),
        feature_id=3050,
    ),
    FeatureSpec(
        name="liq_concentration_div_fut_minus_spot_struct100_1s",
        stage="S3",
        operator="s3.cross_div",
        params={"market_scope": "Futures"},
        label="Cross-Market: liq_concentration_div_fut_minus_spot_struct100_1s (Futures) [1s] (Binance)",
        group="Cross-Market",
        description="Cross-market divergence: liq_concentration_fut_struct100_1s - liq_concentration_spot_struct100_1s.",
        depends_on=(
            Dep(name="liq_concentration_fut_struct100_1s", kind="col"),
            Dep(name="liq_concentration_spot_struct100_1s", kind="col"),
        ),
        feature_id=3051,
    ),
    FeatureSpec(
        name="liq_concentration_div_fut_minus_spot_struct50_1s",
        stage="S3",
        operator="s3.cross_div",
        params={"market_scope": "Futures"},
        label="Cross-Market: liq_concentration_div_fut_minus_spot_struct50_1s (Futures) [1s] (Binance)",
        group="Cross-Market",
        description="Cross-market divergence: liq_concentration_fut_struct50_1s - liq_concentration_spot_struct50_1s.",
        depends_on=(
            Dep(name="liq_concentration_fut_struct50_1s", kind="col"),
            Dep(name="liq_concentration_spot_struct50_1s", kind="col"),
        ),
        feature_id=3052,
    ),
    FeatureSpec(
        name="liq_imb_persist_sf_struct100_300s",
        stage="S3",
        operator="s3.cross_persist",
        params={"market_scope": "Spot|Futures"},
        label="Cross-Market: liq_imb_persist_sf_struct100_300s (Spot/Futures) [300s] (Binance)",
        group="Cross-Market",
        description="Cross-market persistence: sign consistency of liq_imb_sf_struct100_300s.",
        depends_on=(Dep(name="liq_imb_sf_struct100_300s", kind="col"),),
        feature_id=3053,
    ),
    FeatureSpec(
        name="liq_imb_persist_sf_struct100_900s",
        stage="S3",
        operator="s3.cross_persist",
        params={"market_scope": "Spot|Futures"},
        label="Cross-Market: liq_imb_persist_sf_struct100_900s (Spot/Futures) [900s] (Binance)",
        group="Cross-Market",
        description="Cross-market persistence: sign consistency of liq_imb_sf_struct100_900s.",
        depends_on=(Dep(name="liq_imb_sf_struct100_900s", kind="col"),),
        feature_id=3054,
    ),
    FeatureSpec(
        name="liq_imb_persist_sf_struct50_300s",
        stage="S3",
        operator="s3.cross_persist",
        params={"market_scope": "Spot|Futures"},
        label="Cross-Market: liq_imb_persist_sf_struct50_300s (Spot/Futures) [300s] (Binance)",
        group="Cross-Market",
        description="Cross-market persistence: sign consistency of liq_imb_sf_struct50_300s.",
        depends_on=(Dep(name="liq_imb_sf_struct50_300s", kind="col"),),
        feature_id=3055,
    ),
    FeatureSpec(
        name="liq_imb_persist_sf_struct50_900s",
        stage="S3",
        operator="s3.cross_persist",
        params={"market_scope": "Spot|Futures"},
        label="Cross-Market: liq_imb_persist_sf_struct50_900s (Spot/Futures) [900s] (Binance)",
        group="Cross-Market",
        description="Cross-market persistence: sign consistency of liq_imb_sf_struct50_900s.",
        depends_on=(Dep(name="liq_imb_sf_struct50_900s", kind="col"),),
        feature_id=3056,
    ),
    FeatureSpec(
        name="queue_pressure_div_fut_minus_spot_1bps_15s",
        stage="S3",
        operator="s3.cross_div",
        params={"market_scope": "Futures"},
        label="Cross-Market: queue_pressure_div_fut_minus_spot_1bps_15s (Futures) [15s] (Binance)",
        group="Cross-Market",
        description="Cross-market divergence: queue_pressure_fut_1bps_15s - queue_pressure_spot_1bps_15s.",
        depends_on=(
            Dep(name="queue_pressure_fut_1bps_15s", kind="col"),
            Dep(name="queue_pressure_spot_1bps_15s", kind="col"),
        ),
        feature_id=3057,
    ),
    FeatureSpec(
        name="queue_pressure_div_fut_minus_spot_1bps_60s",
        stage="S3",
        operator="s3.cross_div",
        params={"market_scope": "Futures"},
        label="Cross-Market: queue_pressure_div_fut_minus_spot_1bps_60s (Futures) [60s] (Binance)",
        group="Cross-Market",
        description="Cross-market divergence: queue_pressure_fut_1bps_60s - queue_pressure_spot_1bps_60s.",
        depends_on=(
            Dep(name="queue_pressure_fut_1bps_60s", kind="col"),
            Dep(name="queue_pressure_spot_1bps_60s", kind="col"),
        ),
        feature_id=3058,
    ),
    FeatureSpec(
        name="refill_vs_pull_div_fut_minus_spot_1bps_15s",
        stage="S3",
        operator="s3.cross_div",
        params={"market_scope": "Futures"},
        label="Cross-Market: refill_vs_pull_div_fut_minus_spot_1bps_15s (Futures) [15s] (Binance)",
        group="Cross-Market",
        description="Cross-market divergence: refill_vs_pull_fut_1bps_15s - refill_vs_pull_spot_1bps_15s.",
        depends_on=(
            Dep(name="refill_vs_pull_fut_1bps_15s", kind="col"),
            Dep(name="refill_vs_pull_spot_1bps_15s", kind="col"),
        ),
        feature_id=3059,
    ),
    FeatureSpec(
        name="refill_vs_pull_div_fut_minus_spot_2bps_60s",
        stage="S3",
        operator="s3.cross_div",
        params={"market_scope": "Futures"},
        label="Cross-Market: refill_vs_pull_div_fut_minus_spot_2bps_60s (Futures) [60s] (Binance)",
        group="Cross-Market",
        description="Cross-market divergence: refill_vs_pull_fut_2bps_60s - refill_vs_pull_spot_2bps_60s.",
        depends_on=(
            Dep(name="refill_vs_pull_fut_2bps_60s", kind="col"),
            Dep(name="refill_vs_pull_spot_2bps_60s", kind="col"),
        ),
        feature_id=3060,
    ),
    FeatureSpec(
        name="trade_count_sf_div_300s",
        stage="S3",
        operator="s3.cross_div",
        params={"market_scope": "Spot|Futures"},
        label="Cross-Market: trade_count_sf_div_300s (Spot/Futures) [300s] (Binance)",
        group="Cross-Market",
        description="Cross-market divergence: trade_count_fut_div_300s - trade_count_spot_div_300s.",
        depends_on=(
            Dep(name="trade_count_fut_div_300s", kind="col"),
            Dep(name="trade_count_spot_div_300s", kind="col"),
        ),
        feature_id=3061,
    ),
    FeatureSpec(
        name="trade_count_sf_div_60s",
        stage="S3",
        operator="s3.cross_div",
        params={"market_scope": "Spot|Futures"},
        label="Cross-Market: trade_count_sf_div_60s (Spot/Futures) [60s] (Binance)",
        group="Cross-Market",
        description="Cross-market divergence: trade_count_fut_div_60s - trade_count_spot_div_60s.",
        depends_on=(
            Dep(name="trade_count_fut_div_60s", kind="col"),
            Dep(name="trade_count_spot_div_60s", kind="col"),
        ),
        feature_id=3062,
    ),
    FeatureSpec(
        name="trade_count_sf_div_900s",
        stage="S3",
        operator="s3.cross_div",
        params={"market_scope": "Spot|Futures"},
        label="Cross-Market: trade_count_sf_div_900s (Spot/Futures) [900s] (Binance)",
        group="Cross-Market",
        description="Cross-market divergence: trade_count_fut_div_900s - trade_count_spot_div_900s.",
        depends_on=(
            Dep(name="trade_count_fut_div_900s", kind="col"),
            Dep(name="trade_count_spot_div_900s", kind="col"),
        ),
        feature_id=3063,
    ),
    FeatureSpec(
        name="trade_count_sf_share_300s",
        stage="S3",
        operator="s3.cross_share",
        params={"market_scope": "Spot|Futures"},
        label="Cross-Market: trade_count_sf_share_300s (Spot/Futures) [300s] (Binance)",
        group="Cross-Market",
        description="Cross-market share: trade_count_fut_share_300s / (trade_count_fut_share_300s + trade_count_spot_share_300s).",
        depends_on=(
            Dep(name="trade_count_fut_share_300s", kind="col"),
            Dep(name="trade_count_spot_share_300s", kind="col"),
        ),
        feature_id=3064,
    ),
    FeatureSpec(
        name="trade_count_sf_share_60s",
        stage="S3",
        operator="s3.cross_share",
        params={"market_scope": "Spot|Futures"},
        label="Cross-Market: trade_count_sf_share_60s (Spot/Futures) [60s] (Binance)",
        group="Cross-Market",
        description="Cross-market share: trade_count_fut_share_60s / (trade_count_fut_share_60s + trade_count_spot_share_60s).",
        depends_on=(
            Dep(name="trade_count_fut_share_60s", kind="col"),
            Dep(name="trade_count_spot_share_60s", kind="col"),
        ),
        feature_id=3065,
    ),
    FeatureSpec(
        name="trade_count_sf_share_900s",
        stage="S3",
        operator="s3.cross_share",
        params={"market_scope": "Spot|Futures"},
        label="Cross-Market: trade_count_sf_share_900s (Spot/Futures) [900s] (Binance)",
        group="Cross-Market",
        description="Cross-market share: trade_count_fut_share_900s / (trade_count_fut_share_900s + trade_count_spot_share_900s).",
        depends_on=(
            Dep(name="trade_count_fut_share_900s", kind="col"),
            Dep(name="trade_count_spot_share_900s", kind="col"),
        ),
        feature_id=3066,
    ),
    FeatureSpec(
        name="volume_sf_div_delta_300s",
        stage="S3",
        operator="s3.cross_div_delta",
        params={"market_scope": "Spot|Futures"},
        label="Cross-Market: volume_sf_div_delta_300s (Spot/Futures) [300s] (Binance)",
        group="Cross-Market",
        description="Cross-market divergence delta: volume_fut_div_delta_300s - volume_spot_div_delta_300s.",
        depends_on=(
            Dep(name="volume_fut_div_delta_300s", kind="col"),
            Dep(name="volume_spot_div_delta_300s", kind="col"),
        ),
        feature_id=3067,
    ),
    FeatureSpec(
        name="volume_sf_div_delta_900s",
        stage="S3",
        operator="s3.cross_div_delta",
        params={"market_scope": "Spot|Futures"},
        label="Cross-Market: volume_sf_div_delta_900s (Spot/Futures) [900s] (Binance)",
        group="Cross-Market",
        description="Cross-market divergence delta: volume_fut_div_delta_900s - volume_spot_div_delta_900s.",
        depends_on=(
            Dep(name="volume_fut_div_delta_900s", kind="col"),
            Dep(name="volume_spot_div_delta_900s", kind="col"),
        ),
        feature_id=3068,
    ),

]