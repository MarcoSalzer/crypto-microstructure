# etl/spec/s3/s3_returns.py
# ========================================================================
# S3 Returns Features
# Simple rolling return aggregations that provide standardized return signals
# at commonly used horizons.
# 
# Key concepts:
#   - ret_15s: Rolling sum of futures mid log returns over 15s.
#   - ret_60s: Rolling sum of futures mid log returns over 60s.
# 
# These are convenience aliases for the S2 ret_mid_fut rolling sums,
# exposed as canonical return features for downstream stages.
# 
# Dependencies: S2 ret_mid_fut_15s, ret_mid_fut_60s.
#
# Feature count: 2
# Feature ID range: see individual entries
# ========================================================================

from typing import List
from etl.spec import FeatureSpec, Dep

S3_RETURNS_FEATURES: List[FeatureSpec] = [
    FeatureSpec(
        name="ret_15s",
        stage="S3",
        operator="derived.roll_sum",
        params={"market_scope": "Futures", "window_s": "15"},
        label="Returns: ret_15s (Futures) [15s] (Binance)",
        group="Returns",
        description="Rolling sum of ret_mid_fut_15s over 15s window.",
        depends_on=(Dep(name="ret_mid_fut_15s", kind="col"),),
        feature_id=3357,
    ),
    FeatureSpec(
        name="ret_60s",
        stage="S3",
        operator="derived.roll_sum",
        params={"market_scope": "Futures", "window_s": "60"},
        label="Returns: ret_60s (Futures) [60s] (Binance)",
        group="Returns",
        description="Rolling sum of ret_mid_fut_60s over 60s window.",
        depends_on=(Dep(name="ret_mid_fut_60s", kind="col"),),
        feature_id=3358,
    ),

]