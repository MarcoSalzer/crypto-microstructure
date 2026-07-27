# etl/spec/s3/s3_liquidity_events.py
# ========================================================================
# S3 Liquidity Events Features
# Rolling aggregations of per-second liquidity event metrics (churn)
# that smooth noisy tick-level signals into actionable multi-second indicators.
# 
# Key concepts:
#   - ask_churn / bid_churn: Rolling mean of 1s order churn (add_rate + cancel_rate)
#     at the 5bps band. High churn = active market-making or spoofing.
# 
# NOTE: refill_vs_pull_ratio features removed (feature_ids 3176–3179).
#   refill_rate and pull_rate are complementary tick-level events — in any given
#   1s bucket, either a refill OR a pull occurs, never both simultaneously.
#   This means both inputs are never jointly valid → ratio is always NaN.
#   No ML signal can be extracted from a permanently NaN feature.
# 
# Dependencies: S2 ask_churn, bid_churn.
#
# Feature count: 8  (was 12; removed 4 × refill_vs_pull_ratio)
# Feature ID range: see individual entries
# ========================================================================

from typing import List
from etl.spec import FeatureSpec, Dep

S3_LIQUIDITY_EVENTS_FEATURES: List[FeatureSpec] = [
    FeatureSpec(
        name="ask_churn_fut_5bps_15s",
        stage="S3",
        operator="derived.roll_mean",
        params={"market_scope": "Futures", "window_s": "15"},
        label="Liquidity Events: ask_churn_fut_5bps_15s (Futures) [15s] (Binance)",
        group="Liquidity Events",
        description="Rolling mean of ask_churn_fut_5bps_1s over 15s window.",
        depends_on=(Dep(name="ask_churn_fut_5bps_1s", kind="col"),),
        feature_id=3170,
    ),
    FeatureSpec(
        name="ask_churn_fut_5bps_60s",
        stage="S3",
        operator="derived.roll_mean",
        params={"market_scope": "Futures", "window_s": "60"},
        label="Liquidity Events: ask_churn_fut_5bps_60s (Futures) [60s] (Binance)",
        group="Liquidity Events",
        description="Rolling mean of ask_churn_fut_5bps_1s over 60s window.",
        depends_on=(Dep(name="ask_churn_fut_5bps_1s", kind="col"),),
        feature_id=3171,
    ),
    FeatureSpec(
        name="ask_churn_spot_5bps_15s",
        stage="S3",
        operator="derived.roll_mean",
        params={"market_scope": "Spot", "window_s": "15"},
        label="Liquidity Events: ask_churn_spot_5bps_15s (Spot) [15s] (Binance)",
        group="Liquidity Events",
        description="Rolling mean of ask_churn_spot_5bps_1s over 15s window.",
        depends_on=(Dep(name="ask_churn_spot_5bps_1s", kind="col"),),
        feature_id=3172,
    ),
    FeatureSpec(
        name="ask_churn_spot_5bps_60s",
        stage="S3",
        operator="derived.roll_mean",
        params={"market_scope": "Spot", "window_s": "60"},
        label="Liquidity Events: ask_churn_spot_5bps_60s (Spot) [60s] (Binance)",
        group="Liquidity Events",
        description="Rolling mean of ask_churn_spot_5bps_1s over 60s window.",
        depends_on=(Dep(name="ask_churn_spot_5bps_1s", kind="col"),),
        feature_id=3173,
    ),
    FeatureSpec(
        name="bid_churn_fut_5bps_15s",
        stage="S3",
        operator="derived.roll_mean",
        params={"market_scope": "Futures", "window_s": "15"},
        label="Liquidity Events: bid_churn_fut_5bps_15s (Futures) [15s] (Binance)",
        group="Liquidity Events",
        description="Rolling mean of bid_churn_fut_5bps_1s over 15s window.",
        depends_on=(Dep(name="bid_churn_fut_5bps_1s", kind="col"),),
        feature_id=3174,
    ),
    FeatureSpec(
        name="bid_churn_fut_5bps_60s",
        stage="S3",
        operator="derived.roll_mean",
        params={"market_scope": "Futures", "window_s": "60"},
        label="Liquidity Events: bid_churn_fut_5bps_60s (Futures) [60s] (Binance)",
        group="Liquidity Events",
        description="Rolling mean of bid_churn_fut_5bps_1s over 60s window.",
        depends_on=(Dep(name="bid_churn_fut_5bps_1s", kind="col"),),
        feature_id=3175,
    ),
    FeatureSpec(
        name="bid_churn_spot_5bps_15s",
        stage="S3",
        operator="derived.roll_mean",
        params={"market_scope": "Spot", "window_s": "15"},
        label="Liquidity Events: bid_churn_spot_5bps_15s (Spot) [15s] (Binance)",
        group="Liquidity Events",
        description="Rolling mean of bid_churn_spot_5bps_1s over 15s window.",
        depends_on=(Dep(name="bid_churn_spot_5bps_1s", kind="col"),),
        feature_id=3176,
    ),
    FeatureSpec(
        name="bid_churn_spot_5bps_60s",
        stage="S3",
        operator="derived.roll_mean",
        params={"market_scope": "Spot", "window_s": "60"},
        label="Liquidity Events: bid_churn_spot_5bps_60s (Spot) [60s] (Binance)",
        group="Liquidity Events",
        description="Rolling mean of bid_churn_spot_5bps_1s over 60s window.",
        depends_on=(Dep(name="bid_churn_spot_5bps_1s", kind="col"),),
        feature_id=3177,
    ),
]