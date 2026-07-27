# etl/spec/s1/s1_normalization.py
# ==============================================================================
# S1 Feature Specs: Normalization
#
# Binance-only pipeline | Source: S0 features (parquet)
# 12 features | Feature IDs: 1210-1221
# ==============================================================================

from typing import List
from etl.spec import FeatureSpec, Dep


S1_NORMALIZATION_FEATURES: List[FeatureSpec] = [

    # === derived.robust_zscore ===
# [FIX-3600]  Removed 3 × window_s=3600 features: always 99%+ NaN on 1h files.
# [FIX-1s]    Added window_s=5 + min_periods=2 to 4 × z_liq_imb_*_1s (no window_s = defaults
#             to 1 in engine → MAD=0 → 100% NaN). 1s suffix = input signal resolution.
    FeatureSpec(
        name="z_basis_900s",
        stage="S1",
        operator="derived.zscore_diff",
        params={'market_scope': 'Futures', 'window_s': 900, 'resample': '1s'},
        group="Normalization",
        description="zscore_diff: z-score of (col_a - col_b) over rolling window.",
        depends_on=(Dep(name="mid_fut_1s", kind="col"), Dep(name="mid_spot_1s", kind="col"),),
        feature_id=1210,
    ),
    FeatureSpec(
        name="z_liq_imb_fut_struct100_1s",
        stage="S1",
        operator="derived.robust_zscore",
        params={'market_scope': 'Futures', 'window_s': 5, 'min_periods': 2, 'resample': '1s'},
        group="Normalization",
        description="(x - median) / (1.4826 * MAD + eps). Robust z-score.",
        depends_on=(Dep(name="depth_imbalance_struct100_fut_1s", kind="col"),),
        feature_id=1211,
    ),
    FeatureSpec(
        name="z_liq_imb_fut_struct50_1s",
        stage="S1",
        operator="derived.robust_zscore",
        params={'market_scope': 'Futures', 'window_s': 5, 'min_periods': 2, 'resample': '1s'},
        group="Normalization",
        description="(x - median) / (1.4826 * MAD + eps). Robust z-score.",
        depends_on=(Dep(name="depth_imbalance_struct50_fut_1s", kind="col"),),
        feature_id=1212,
    ),
    FeatureSpec(
        name="z_liq_imb_spot_struct100_1s",
        stage="S1",
        operator="derived.robust_zscore",
        params={'market_scope': 'Spot', 'window_s': 5, 'min_periods': 2, 'resample': '1s'},
        group="Normalization",
        description="(x - median) / (1.4826 * MAD + eps). Robust z-score.",
        depends_on=(Dep(name="depth_imbalance_struct100_spot_1s", kind="col"),),
        feature_id=1213,
    ),
    FeatureSpec(
        name="z_liq_imb_spot_struct50_1s",
        stage="S1",
        operator="derived.robust_zscore",
        params={'market_scope': 'Spot', 'window_s': 5, 'min_periods': 2, 'resample': '1s'},
        group="Normalization",
        description="(x - median) / (1.4826 * MAD + eps). Robust z-score.",
        depends_on=(Dep(name="depth_imbalance_struct50_spot_1s", kind="col"),),
        feature_id=1214,
    ),
    FeatureSpec(
        name="z_lwp_minus_mid_2bps_300s",
        stage="S1",
        operator="derived.zscore_diff",
        params={'market_scope': 'Futures', 'window_s': 300, 'resample': '1s'},
        group="Normalization",
        description="zscore_diff: z-score of (col_a - col_b) over rolling window.",
        depends_on=(Dep(name="lwp_mid_2bps_fut_1s", kind="col"), Dep(name="mid_fut_1s", kind="col"),),
        feature_id=1215,
    ),
    FeatureSpec(
        name="z_lwp_minus_mid_5bps_300s",
        stage="S1",
        operator="derived.zscore_diff",
        params={'market_scope': 'Futures', 'window_s': 300, 'resample': '1s'},
        group="Normalization",
        description="zscore_diff: z-score of (col_a - col_b) over rolling window.",
        depends_on=(Dep(name="lwp_mid_5bps_fut_1s", kind="col"), Dep(name="mid_fut_1s", kind="col"),),
        feature_id=1216,
    ),
    FeatureSpec(
        name="z_lwp_minus_mid_5bps_900s",
        stage="S1",
        operator="derived.zscore_diff",
        params={'market_scope': 'Futures', 'window_s': 900, 'resample': '1s'},
        group="Normalization",
        description="zscore_diff: z-score of (col_a - col_b) over rolling window.",
        depends_on=(Dep(name="lwp_mid_5bps_fut_1s", kind="col"), Dep(name="mid_fut_1s", kind="col"),),
        feature_id=1217,
    ),
    FeatureSpec(
        name="z_lwp_minus_mid_struct100_300s",
        stage="S1",
        operator="derived.zscore_diff",
        params={'market_scope': 'Futures', 'window_s': 300, 'resample': '1s'},
        group="Normalization",
        description="zscore_diff: z-score of (col_a - col_b) over rolling window.",
        depends_on=(Dep(name="lwp_mid_struct100_fut_1s", kind="col"), Dep(name="mid_fut_1s", kind="col"),),
        feature_id=1218,
    ),
    FeatureSpec(
        name="z_lwp_minus_mid_struct100_900s",
        stage="S1",
        operator="derived.zscore_diff",
        params={'market_scope': 'Futures', 'window_s': 900, 'resample': '1s'},
        group="Normalization",
        description="zscore_diff: z-score of (col_a - col_b) over rolling window.",
        depends_on=(Dep(name="lwp_mid_struct100_fut_1s", kind="col"), Dep(name="mid_fut_1s", kind="col"),),
        feature_id=1219,
    ),
    FeatureSpec(
        name="z_lwp_minus_mid_struct50_300s",
        stage="S1",
        operator="derived.zscore_diff",
        params={'market_scope': 'Futures', 'window_s': 300, 'resample': '1s'},
        group="Normalization",
        description="zscore_diff: z-score of (col_a - col_b) over rolling window.",
        depends_on=(Dep(name="lwp_mid_struct50_fut_1s", kind="col"), Dep(name="mid_fut_1s", kind="col"),),
        feature_id=1220,
    ),
    FeatureSpec(
        name="z_lwp_minus_mid_struct50_900s",
        stage="S1",
        operator="derived.zscore_diff",
        params={'market_scope': 'Futures', 'window_s': 900, 'resample': '1s'},
        group="Normalization",
        description="zscore_diff: z-score of (col_a - col_b) over rolling window.",
        depends_on=(Dep(name="lwp_mid_struct50_fut_1s", kind="col"), Dep(name="mid_fut_1s", kind="col"),),
        feature_id=1221,
    ),
]