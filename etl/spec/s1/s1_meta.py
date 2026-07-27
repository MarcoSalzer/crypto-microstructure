# etl/spec/s1/s1_meta.py
# ==============================================================================
# S1 Feature Specs: Meta
#
# Binance-only pipeline | Source: S0 features (parquet)
# 14 features | Feature IDs: 1198–1209
# ==============================================================================

from typing import List
from etl.spec import FeatureSpec, Dep

S1_META_FEATURES: List[FeatureSpec] = [

    # === derived.range_pct ===
    FeatureSpec(
        name="range_pct_spot_300s",
        stage="S1",
        operator="derived.range_pct",
        params={'market_scope': 'Spot', 'window_s': 300, 'resample': '1s'},
        group="Meta",
        description="(max - min) / mid * 10000 bps over window.",
        depends_on=(Dep(name="mid_spot_1s", kind="col"),),
        feature_id=1198,
    ),
    FeatureSpec(
        name="range_pct_spot_60s",
        stage="S1",
        operator="derived.range_pct",
        params={'market_scope': 'Spot', 'window_s': 60, 'resample': '1s'},
        group="Meta",
        description="(max - min) / mid * 10000 bps over window.",
        depends_on=(Dep(name="mid_spot_1s", kind="col"),),
        feature_id=1199,
    ),
    FeatureSpec(
        name="range_pct_spot_900s",
        stage="S1",
        operator="derived.range_pct",
        params={'market_scope': 'Spot', 'window_s': 900, 'resample': '1s'},
        group="Meta",
        description="(max - min) / mid * 10000 bps over window.",
        depends_on=(Dep(name="mid_spot_1s", kind="col"),),
        feature_id=1200,
    ),

    # === derived.range_pos ===
    FeatureSpec(
        name="range_pos_spot_300s",
        stage="S1",
        operator="derived.range_pos",
        params={'market_scope': 'Spot', 'window_s': 300, 'resample': '1s'},
        group="Meta",
        description="(last - min) / (max - min) position in range.",
        depends_on=(Dep(name="mid_spot_1s", kind="col"),),
        feature_id=1201,
    ),
    FeatureSpec(
        name="range_pos_spot_60s",
        stage="S1",
        operator="derived.range_pos",
        params={'market_scope': 'Spot', 'window_s': 60, 'resample': '1s'},
        group="Meta",
        description="(last - min) / (max - min) position in range.",
        depends_on=(Dep(name="mid_spot_1s", kind="col"),),
        feature_id=1202,
    ),
    FeatureSpec(
        name="range_pos_spot_900s",
        stage="S1",
        operator="derived.range_pos",
        params={'market_scope': 'Spot', 'window_s': 900, 'resample': '1s'},
        group="Meta",
        description="(last - min) / (max - min) position in range.",
        depends_on=(Dep(name="mid_spot_1s", kind="col"),),
        feature_id=1203,
    ),

    # === derived.range_pct (Futures) ===
    # Mirrors the spot variants above. Futures mid is used as the price
    # reference since this pipeline targets futures trading. These fill the
    # asymmetric gap where only range_pct_fut_3600s existed previously.
    FeatureSpec(
        name="range_pct_fut_60s",
        stage="S1",
        operator="derived.range_pct",
        params={'market_scope': 'Fut', 'window_s': 60, 'resample': '1s'},
        group="Meta",
        description="(max - min) / mid * 10000 bps over window (Futures, 60s).",
        depends_on=(Dep(name="mid_fut_1s", kind="col"),),
        feature_id=1204,
    ),
    FeatureSpec(
        name="range_pct_fut_300s",
        stage="S1",
        operator="derived.range_pct",
        params={'market_scope': 'Fut', 'window_s': 300, 'resample': '1s'},
        group="Meta",
        description="(max - min) / mid * 10000 bps over window (Futures, 300s).",
        depends_on=(Dep(name="mid_fut_1s", kind="col"),),
        feature_id=1205,
    ),
    FeatureSpec(
        name="range_pct_fut_900s",
        stage="S1",
        operator="derived.range_pct",
        params={'market_scope': 'Fut', 'window_s': 900, 'resample': '1s'},
        group="Meta",
        description="(max - min) / mid * 10000 bps over window (Futures, 900s).",
        depends_on=(Dep(name="mid_fut_1s", kind="col"),),
        feature_id=1206,
    ),

    # === derived.range_pos (Futures) ===
    FeatureSpec(
        name="range_pos_fut_60s",
        stage="S1",
        operator="derived.range_pos",
        params={'market_scope': 'Fut', 'window_s': 60, 'resample': '1s'},
        group="Meta",
        description="(last - min) / (max - min) position in range (Futures, 60s).",
        depends_on=(Dep(name="mid_fut_1s", kind="col"),),
        feature_id=1207,
    ),
    FeatureSpec(
        name="range_pos_fut_300s",
        stage="S1",
        operator="derived.range_pos",
        params={'market_scope': 'Fut', 'window_s': 300, 'resample': '1s'},
        group="Meta",
        description="(last - min) / (max - min) position in range (Futures, 300s).",
        depends_on=(Dep(name="mid_fut_1s", kind="col"),),
        feature_id=1208,
    ),
    FeatureSpec(
        name="range_pos_fut_900s",
        stage="S1",
        operator="derived.range_pos",
        params={'market_scope': 'Fut', 'window_s': 900, 'resample': '1s'},
        group="Meta",
        description="(last - min) / (max - min) position in range (Futures, 900s).",
        depends_on=(Dep(name="mid_fut_1s", kind="col"),),
        feature_id=1209,
    ),
]