# etl/spec/s3/s3_bookshape.py
# ========================================================================
# S3 Bookshape Features
# Structural orderbook analytics that characterize the shape and concentration
# of liquidity across depth bands (struct50 / struct100) and markets (fut / spot).
# 
# Key concepts:
#   - depth_gradient: Ratio of ask-side to bid-side depth gradient, measuring
#     how steeply liquidity drops off on each side. Computed at struct50 and
#     struct100 levels across multiple time windows (60s, 300s, 900s, 3600s).
#   - liq_concentration: Ratio of ask-side to bid-side liquidity concentration
#     (inner-band share of total depth). Same structure/time parametrization.
# 
# All features use the derived.ratio operator: num / (|den| + eps).
# 
# Dependencies: S2 depth_gradient_ask/bid, liq_concentration_ask/bid at
#               struct50/struct100 levels.
#
# Feature count: 24  (was 32; removed 8 × 3600s features)
# [FIX-3600] Removed all window_s=3600 features (depth_gradient + liq_concentration _3600s rolling means).
#            3600s rolling windows require 1h prior data even with context;
#            produce 99-100% NaN on 1h Parquet files. The 900s variants
#            provide equivalent microstructure regime coverage.
# Feature ID range: see individual entries
# ========================================================================

from typing import List
from etl.spec import FeatureSpec, Dep

S3_BOOKSHAPE_FEATURES: List[FeatureSpec] = [
    FeatureSpec(
        name="depth_gradient_fut_struct100_300s",
        stage="S3",
        operator="derived.ratio",
        params={"market_scope": "Futures"},
        label="Bookshape: depth_gradient_fut_struct100_300s (Futures) [300s] (Binance)",
        group="Bookshape",
        description="Ratio: depth_gradient_ask_fut_struct100_300s / (depth_gradient_bid_fut_struct100_300s + eps) if len(deps)>=2.",
        depends_on=(
            Dep(name="depth_gradient_ask_fut_struct100_300s", kind="col"),
            Dep(name="depth_gradient_bid_fut_struct100_300s", kind="col"),
        ),
        feature_id=3016,
    ),
    FeatureSpec(
        name="depth_gradient_fut_struct100_60s",
        stage="S3",
        operator="derived.ratio",
        params={"market_scope": "Futures"},
        label="Bookshape: depth_gradient_fut_struct100_60s (Futures) [60s] (Binance)",
        group="Bookshape",
        description="Ratio: depth_gradient_ask_fut_struct100_60s / (depth_gradient_bid_fut_struct100_60s + eps) if len(deps)>=2.",
        depends_on=(
            Dep(name="depth_gradient_ask_fut_struct100_60s", kind="col"),
            Dep(name="depth_gradient_bid_fut_struct100_60s", kind="col"),
        ),
        feature_id=3017,
    ),
    FeatureSpec(
        name="depth_gradient_fut_struct100_900s",
        stage="S3",
        operator="derived.ratio",
        params={"market_scope": "Futures"},
        label="Bookshape: depth_gradient_fut_struct100_900s (Futures) [900s] (Binance)",
        group="Bookshape",
        description="Ratio: depth_gradient_ask_fut_struct100_900s / (depth_gradient_bid_fut_struct100_900s + eps) if len(deps)>=2.",
        depends_on=(
            Dep(name="depth_gradient_ask_fut_struct100_900s", kind="col"),
            Dep(name="depth_gradient_bid_fut_struct100_900s", kind="col"),
        ),
        feature_id=3018,
    ),
    FeatureSpec(
        name="depth_gradient_fut_struct50_300s",
        stage="S3",
        operator="derived.ratio",
        params={"market_scope": "Futures"},
        label="Bookshape: depth_gradient_fut_struct50_300s (Futures) [300s] (Binance)",
        group="Bookshape",
        description="Ratio: depth_gradient_ask_fut_struct50_300s / (depth_gradient_bid_fut_struct50_300s + eps) if len(deps)>=2.",
        depends_on=(
            Dep(name="depth_gradient_ask_fut_struct50_300s", kind="col"),
            Dep(name="depth_gradient_bid_fut_struct50_300s", kind="col"),
        ),
        feature_id=3019,
    ),
    FeatureSpec(
        name="depth_gradient_fut_struct50_60s",
        stage="S3",
        operator="derived.ratio",
        params={"market_scope": "Futures"},
        label="Bookshape: depth_gradient_fut_struct50_60s (Futures) [60s] (Binance)",
        group="Bookshape",
        description="Ratio: depth_gradient_ask_fut_struct50_60s / (depth_gradient_bid_fut_struct50_60s + eps) if len(deps)>=2.",
        depends_on=(
            Dep(name="depth_gradient_ask_fut_struct50_60s", kind="col"),
            Dep(name="depth_gradient_bid_fut_struct50_60s", kind="col"),
        ),
        feature_id=3020,
    ),
    FeatureSpec(
        name="depth_gradient_fut_struct50_900s",
        stage="S3",
        operator="derived.ratio",
        params={"market_scope": "Futures"},
        label="Bookshape: depth_gradient_fut_struct50_900s (Futures) [900s] (Binance)",
        group="Bookshape",
        description="Ratio: depth_gradient_ask_fut_struct50_900s / (depth_gradient_bid_fut_struct50_900s + eps) if len(deps)>=2.",
        depends_on=(
            Dep(name="depth_gradient_ask_fut_struct50_900s", kind="col"),
            Dep(name="depth_gradient_bid_fut_struct50_900s", kind="col"),
        ),
        feature_id=3021,
    ),
    FeatureSpec(
        name="depth_gradient_spot_struct100_300s",
        stage="S3",
        operator="derived.ratio",
        params={"market_scope": "Spot"},
        label="Bookshape: depth_gradient_spot_struct100_300s (Spot) [300s] (Binance)",
        group="Bookshape",
        description="Ratio: depth_gradient_ask_spot_struct100_300s / (depth_gradient_bid_spot_struct100_300s + eps) if len(deps)>=2.",
        depends_on=(
            Dep(name="depth_gradient_ask_spot_struct100_300s", kind="col"),
            Dep(name="depth_gradient_bid_spot_struct100_300s", kind="col"),
        ),
        feature_id=3022,
    ),
    FeatureSpec(
        name="depth_gradient_spot_struct100_60s",
        stage="S3",
        operator="derived.ratio",
        params={"market_scope": "Spot"},
        label="Bookshape: depth_gradient_spot_struct100_60s (Spot) [60s] (Binance)",
        group="Bookshape",
        description="Ratio: depth_gradient_ask_spot_struct100_60s / (depth_gradient_bid_spot_struct100_60s + eps) if len(deps)>=2.",
        depends_on=(
            Dep(name="depth_gradient_ask_spot_struct100_60s", kind="col"),
            Dep(name="depth_gradient_bid_spot_struct100_60s", kind="col"),
        ),
        feature_id=3023,
    ),
    FeatureSpec(
        name="depth_gradient_spot_struct100_900s",
        stage="S3",
        operator="derived.ratio",
        params={"market_scope": "Spot"},
        label="Bookshape: depth_gradient_spot_struct100_900s (Spot) [900s] (Binance)",
        group="Bookshape",
        description="Ratio: depth_gradient_ask_spot_struct100_900s / (depth_gradient_bid_spot_struct100_900s + eps) if len(deps)>=2.",
        depends_on=(
            Dep(name="depth_gradient_ask_spot_struct100_900s", kind="col"),
            Dep(name="depth_gradient_bid_spot_struct100_900s", kind="col"),
        ),
        feature_id=3024,
    ),
    FeatureSpec(
        name="depth_gradient_spot_struct50_300s",
        stage="S3",
        operator="derived.ratio",
        params={"market_scope": "Spot"},
        label="Bookshape: depth_gradient_spot_struct50_300s (Spot) [300s] (Binance)",
        group="Bookshape",
        description="Ratio: depth_gradient_ask_spot_struct50_300s / (depth_gradient_bid_spot_struct50_300s + eps) if len(deps)>=2.",
        depends_on=(
            Dep(name="depth_gradient_ask_spot_struct50_300s", kind="col"),
            Dep(name="depth_gradient_bid_spot_struct50_300s", kind="col"),
        ),
        feature_id=3025,
    ),
    FeatureSpec(
        name="depth_gradient_spot_struct50_60s",
        stage="S3",
        operator="derived.ratio",
        params={"market_scope": "Spot"},
        label="Bookshape: depth_gradient_spot_struct50_60s (Spot) [60s] (Binance)",
        group="Bookshape",
        description="Ratio: depth_gradient_ask_spot_struct50_60s / (depth_gradient_bid_spot_struct50_60s + eps) if len(deps)>=2.",
        depends_on=(
            Dep(name="depth_gradient_ask_spot_struct50_60s", kind="col"),
            Dep(name="depth_gradient_bid_spot_struct50_60s", kind="col"),
        ),
        feature_id=3026,
    ),
    FeatureSpec(
        name="depth_gradient_spot_struct50_900s",
        stage="S3",
        operator="derived.ratio",
        params={"market_scope": "Spot"},
        label="Bookshape: depth_gradient_spot_struct50_900s (Spot) [900s] (Binance)",
        group="Bookshape",
        description="Ratio: depth_gradient_ask_spot_struct50_900s / (depth_gradient_bid_spot_struct50_900s + eps) if len(deps)>=2.",
        depends_on=(
            Dep(name="depth_gradient_ask_spot_struct50_900s", kind="col"),
            Dep(name="depth_gradient_bid_spot_struct50_900s", kind="col"),
        ),
        feature_id=3027,
    ),
    FeatureSpec(
        name="liq_concentration_fut_struct100_300s",
        stage="S3",
        operator="derived.ratio",
        params={"market_scope": "Futures"},
        label="Bookshape: liq_concentration_fut_struct100_300s (Futures) [300s] (Binance)",
        group="Bookshape",
        description="Ratio: liq_concentration_ask_fut_struct100_300s / (liq_concentration_bid_fut_struct100_300s + eps) if len(deps)>=2.",
        depends_on=(
            Dep(name="liq_concentration_ask_fut_struct100_300s", kind="col"),
            Dep(name="liq_concentration_bid_fut_struct100_300s", kind="col"),
        ),
        feature_id=3028,
    ),
    FeatureSpec(
        name="liq_concentration_fut_struct100_60s",
        stage="S3",
        operator="derived.ratio",
        params={"market_scope": "Futures"},
        label="Bookshape: liq_concentration_fut_struct100_60s (Futures) [60s] (Binance)",
        group="Bookshape",
        description="Ratio: liq_concentration_ask_fut_struct100_60s / (liq_concentration_bid_fut_struct100_60s + eps) if len(deps)>=2.",
        depends_on=(
            Dep(name="liq_concentration_ask_fut_struct100_60s", kind="col"),
            Dep(name="liq_concentration_bid_fut_struct100_60s", kind="col"),
        ),
        feature_id=3029,
    ),
    FeatureSpec(
        name="liq_concentration_fut_struct100_900s",
        stage="S3",
        operator="derived.ratio",
        params={"market_scope": "Futures"},
        label="Bookshape: liq_concentration_fut_struct100_900s (Futures) [900s] (Binance)",
        group="Bookshape",
        description="Ratio: liq_concentration_ask_fut_struct100_900s / (liq_concentration_bid_fut_struct100_900s + eps) if len(deps)>=2.",
        depends_on=(
            Dep(name="liq_concentration_ask_fut_struct100_900s", kind="col"),
            Dep(name="liq_concentration_bid_fut_struct100_900s", kind="col"),
        ),
        feature_id=3030,
    ),
    FeatureSpec(
        name="liq_concentration_fut_struct50_300s",
        stage="S3",
        operator="derived.ratio",
        params={"market_scope": "Futures"},
        label="Bookshape: liq_concentration_fut_struct50_300s (Futures) [300s] (Binance)",
        group="Bookshape",
        description="Ratio: liq_concentration_ask_fut_struct50_300s / (liq_concentration_bid_fut_struct50_300s + eps) if len(deps)>=2.",
        depends_on=(
            Dep(name="liq_concentration_ask_fut_struct50_300s", kind="col"),
            Dep(name="liq_concentration_bid_fut_struct50_300s", kind="col"),
        ),
        feature_id=3031,
    ),
    FeatureSpec(
        name="liq_concentration_fut_struct50_60s",
        stage="S3",
        operator="derived.ratio",
        params={"market_scope": "Futures"},
        label="Bookshape: liq_concentration_fut_struct50_60s (Futures) [60s] (Binance)",
        group="Bookshape",
        description="Ratio: liq_concentration_ask_fut_struct50_60s / (liq_concentration_bid_fut_struct50_60s + eps) if len(deps)>=2.",
        depends_on=(
            Dep(name="liq_concentration_ask_fut_struct50_60s", kind="col"),
            Dep(name="liq_concentration_bid_fut_struct50_60s", kind="col"),
        ),
        feature_id=3032,
    ),
    FeatureSpec(
        name="liq_concentration_fut_struct50_900s",
        stage="S3",
        operator="derived.ratio",
        params={"market_scope": "Futures"},
        label="Bookshape: liq_concentration_fut_struct50_900s (Futures) [900s] (Binance)",
        group="Bookshape",
        description="Ratio: liq_concentration_ask_fut_struct50_900s / (liq_concentration_bid_fut_struct50_900s + eps) if len(deps)>=2.",
        depends_on=(
            Dep(name="liq_concentration_ask_fut_struct50_900s", kind="col"),
            Dep(name="liq_concentration_bid_fut_struct50_900s", kind="col"),
        ),
        feature_id=3033,
    ),
    FeatureSpec(
        name="liq_concentration_spot_struct100_300s",
        stage="S3",
        operator="derived.ratio",
        params={"market_scope": "Spot"},
        label="Bookshape: liq_concentration_spot_struct100_300s (Spot) [300s] (Binance)",
        group="Bookshape",
        description="Ratio: liq_concentration_ask_spot_struct100_300s / (liq_concentration_bid_spot_struct100_300s + eps) if len(deps)>=2.",
        depends_on=(
            Dep(name="liq_concentration_ask_spot_struct100_300s", kind="col"),
            Dep(name="liq_concentration_bid_spot_struct100_300s", kind="col"),
        ),
        feature_id=3034,
    ),
    FeatureSpec(
        name="liq_concentration_spot_struct100_60s",
        stage="S3",
        operator="derived.ratio",
        params={"market_scope": "Spot"},
        label="Bookshape: liq_concentration_spot_struct100_60s (Spot) [60s] (Binance)",
        group="Bookshape",
        description="Ratio: liq_concentration_ask_spot_struct100_60s / (liq_concentration_bid_spot_struct100_60s + eps) if len(deps)>=2.",
        depends_on=(
            Dep(name="liq_concentration_ask_spot_struct100_60s", kind="col"),
            Dep(name="liq_concentration_bid_spot_struct100_60s", kind="col"),
        ),
        feature_id=3035,
    ),
    FeatureSpec(
        name="liq_concentration_spot_struct100_900s",
        stage="S3",
        operator="derived.ratio",
        params={"market_scope": "Spot"},
        label="Bookshape: liq_concentration_spot_struct100_900s (Spot) [900s] (Binance)",
        group="Bookshape",
        description="Ratio: liq_concentration_ask_spot_struct100_900s / (liq_concentration_bid_spot_struct100_900s + eps) if len(deps)>=2.",
        depends_on=(
            Dep(name="liq_concentration_ask_spot_struct100_900s", kind="col"),
            Dep(name="liq_concentration_bid_spot_struct100_900s", kind="col"),
        ),
        feature_id=3036,
    ),
    FeatureSpec(
        name="liq_concentration_spot_struct50_300s",
        stage="S3",
        operator="derived.ratio",
        params={"market_scope": "Spot"},
        label="Bookshape: liq_concentration_spot_struct50_300s (Spot) [300s] (Binance)",
        group="Bookshape",
        description="Ratio: liq_concentration_ask_spot_struct50_300s / (liq_concentration_bid_spot_struct50_300s + eps) if len(deps)>=2.",
        depends_on=(
            Dep(name="liq_concentration_ask_spot_struct50_300s", kind="col"),
            Dep(name="liq_concentration_bid_spot_struct50_300s", kind="col"),
        ),
        feature_id=3037,
    ),
    FeatureSpec(
        name="liq_concentration_spot_struct50_60s",
        stage="S3",
        operator="derived.ratio",
        params={"market_scope": "Spot"},
        label="Bookshape: liq_concentration_spot_struct50_60s (Spot) [60s] (Binance)",
        group="Bookshape",
        description="Ratio: liq_concentration_ask_spot_struct50_60s / (liq_concentration_bid_spot_struct50_60s + eps) if len(deps)>=2.",
        depends_on=(
            Dep(name="liq_concentration_ask_spot_struct50_60s", kind="col"),
            Dep(name="liq_concentration_bid_spot_struct50_60s", kind="col"),
        ),
        feature_id=3038,
    ),
    FeatureSpec(
        name="liq_concentration_spot_struct50_900s",
        stage="S3",
        operator="derived.ratio",
        params={"market_scope": "Spot"},
        label="Bookshape: liq_concentration_spot_struct50_900s (Spot) [900s] (Binance)",
        group="Bookshape",
        description="Ratio: liq_concentration_ask_spot_struct50_900s / (liq_concentration_bid_spot_struct50_900s + eps) if len(deps)>=2.",
        depends_on=(
            Dep(name="liq_concentration_ask_spot_struct50_900s", kind="col"),
            Dep(name="liq_concentration_bid_spot_struct50_900s", kind="col"),
        ),
        feature_id=3039,
    ),

]