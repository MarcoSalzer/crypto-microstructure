# etl/spec/s1/s1_forward_excursion.py
# ==============================================================================
# S1 Feature Specs: Forward Excursion (MAE / MFE)
#
# Binance-only pipeline | Source: S0 features (parquet)
# 8 features | Feature IDs: 1323-1330
#
# PURPOSE:
#   Maximum Adverse Excursion (MAE) and Maximum Favorable Excursion (MFE) on
#   mid_fut_1s across four forward horizons. Long-trade perspective:
#     MAE = drawdown in bps = (mid[t] - min(mid[t..t+w])) / mid[t] * 10000
#     MFE = runup   in bps = (max(mid[t..t+w]) - mid[t]) / mid[t] * 10000
#
#   Short-trade interpretation follows by symmetry (model learns the sign
#   asymmetry implicitly). Reverse-rolling implementation is O(N log W) per
#   horizon.
#
# SCOPE:
#   Cold-path only — forward windows are not computable in real-time.
#   Hot-path DAG filter (Phase 6) removes these features before computation.
#
# USAGE:
#   Targets for stop-loss / take-profit calibration, holding-time studies,
#   regime-dependent drawdown analysis.
#
# ==============================================================================

from typing import List
from etl.spec import FeatureSpec, Dep


S1_FORWARD_EXCURSION_FEATURES: List[FeatureSpec] = [

    # =========================================================================
    # MAE — Max Adverse Excursion (drawdown for long position, in bps)
    # =========================================================================

    FeatureSpec(
        name="mae_fwd_15s_bps",
        stage="S1",
        operator="derived.mae_fwd",
        params={'market_scope': 'Futures', 'window_s': 15, 'resample': '1s'},
        group="Forward Excursion",
        description=(
            "Max Adverse Excursion over next 15s (long perspective): "
            "(mid[t] - min(mid[t..t+15])) / mid[t] * 10000, in bps. "
            "Non-negative; 0 = no drawdown in window."
        ),
        depends_on=(Dep(name="mid_fut_1s", kind="col"),),
        feature_id=1323,
    ),
    FeatureSpec(
        name="mae_fwd_60s_bps",
        stage="S1",
        operator="derived.mae_fwd",
        params={'market_scope': 'Futures', 'window_s': 60, 'resample': '1s'},
        group="Forward Excursion",
        description="Max Adverse Excursion over next 60s, in bps.",
        depends_on=(Dep(name="mid_fut_1s", kind="col"),),
        feature_id=1324,
    ),
    FeatureSpec(
        name="mae_fwd_300s_bps",
        stage="S1",
        operator="derived.mae_fwd",
        params={'market_scope': 'Futures', 'window_s': 300, 'resample': '1s'},
        group="Forward Excursion",
        description="Max Adverse Excursion over next 300s (5 min), in bps.",
        depends_on=(Dep(name="mid_fut_1s", kind="col"),),
        feature_id=1325,
    ),
    FeatureSpec(
        name="mae_fwd_900s_bps",
        stage="S1",
        operator="derived.mae_fwd",
        params={'market_scope': 'Futures', 'window_s': 900, 'resample': '1s'},
        group="Forward Excursion",
        description="Max Adverse Excursion over next 900s (15 min), in bps.",
        depends_on=(Dep(name="mid_fut_1s", kind="col"),),
        feature_id=1326,
    ),

    # =========================================================================
    # MFE — Max Favorable Excursion (runup for long position, in bps)
    # =========================================================================

    FeatureSpec(
        name="mfe_fwd_15s_bps",
        stage="S1",
        operator="derived.mfe_fwd",
        params={'market_scope': 'Futures', 'window_s': 15, 'resample': '1s'},
        group="Forward Excursion",
        description=(
            "Max Favorable Excursion over next 15s (long perspective): "
            "(max(mid[t..t+15]) - mid[t]) / mid[t] * 10000, in bps. "
            "Non-negative; 0 = no runup in window."
        ),
        depends_on=(Dep(name="mid_fut_1s", kind="col"),),
        feature_id=1327,
    ),
    FeatureSpec(
        name="mfe_fwd_60s_bps",
        stage="S1",
        operator="derived.mfe_fwd",
        params={'market_scope': 'Futures', 'window_s': 60, 'resample': '1s'},
        group="Forward Excursion",
        description="Max Favorable Excursion over next 60s, in bps.",
        depends_on=(Dep(name="mid_fut_1s", kind="col"),),
        feature_id=1328,
    ),
    FeatureSpec(
        name="mfe_fwd_300s_bps",
        stage="S1",
        operator="derived.mfe_fwd",
        params={'market_scope': 'Futures', 'window_s': 300, 'resample': '1s'},
        group="Forward Excursion",
        description="Max Favorable Excursion over next 300s (5 min), in bps.",
        depends_on=(Dep(name="mid_fut_1s", kind="col"),),
        feature_id=1329,
    ),
    FeatureSpec(
        name="mfe_fwd_900s_bps",
        stage="S1",
        operator="derived.mfe_fwd",
        params={'market_scope': 'Futures', 'window_s': 900, 'resample': '1s'},
        group="Forward Excursion",
        description="Max Favorable Excursion over next 900s (15 min), in bps.",
        depends_on=(Dep(name="mid_fut_1s", kind="col"),),
        feature_id=1330,
    ),
]