# etl/spec/s1/s1_aggression.py
# ==============================================================================
# S1 Feature Specs: Aggression
#
# Binance-only pipeline | Source: S0 features (parquet)
# 2 features | Feature IDs: 1038-1039
# ==============================================================================

from typing import List
from etl.spec import FeatureSpec, Dep


S1_AGGRESSION_FEATURES: List[FeatureSpec] = [

    # === trades.taker_imbalance ===
    FeatureSpec(
        name="taker_imbalance_fut_1s",
        stage="S1",
        operator="trades.taker_imbalance",
        params={'market_scope': 'Futures', 'resample': '1s'},
        group="Aggression",
        description="(buy_vol - sell_vol) / (buy_vol + sell_vol). Range [-1,1].",
        depends_on=(Dep(name="taker_buy_vol_fut_1s", kind="col"), Dep(name="taker_sell_vol_fut_1s", kind="col"),),
        feature_id=1038,
    ),
    FeatureSpec(
        name="taker_imbalance_spot_1s",
        stage="S1",
        operator="trades.taker_imbalance",
        params={'market_scope': 'Spot', 'resample': '1s'},
        group="Aggression",
        description="(buy_vol - sell_vol) / (buy_vol + sell_vol). Range [-1,1].",
        depends_on=(Dep(name="taker_buy_vol_spot_1s", kind="col"), Dep(name="taker_sell_vol_spot_1s", kind="col"),),
        feature_id=1039,
    ),
]