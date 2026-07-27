# etl/spec/s1/s1_forward_rv.py
# ==============================================================================
# S1 Feature Specs: Forward Realized Volatility
#
# Binance-only pipeline | Source: S0 features (parquet)
# 3 features | Feature IDs: 1331-1333
#
# PURPOSE:
#   Forward realized volatility on mid_fut_1s across three horizons.
#     rv_fwd_Hs = sqrt( sum(r_1s[t+1]^2 + ... + r_1s[t+H]^2) )
#   where r_1s[k] = log(mid[k] / mid[k-1]).
#
#   No annualization applied — absolute volatility over the H-second window.
#   Complementary to the backward-looking z_rv_* features (S1/S2).
#
# USE-CASE:
#   Stop-loss calibration, regime detection (vol-expansion / compression),
#   position sizing under forward-vol assumptions.
#
# SCOPE:
#   Cold-path only — forward windows are not computable in real-time.
#   Hot-path DAG filter (Phase 6) removes these features.
#
# ==============================================================================

from typing import List
from etl.spec import FeatureSpec, Dep


S1_FORWARD_RV_FEATURES: List[FeatureSpec] = [
    FeatureSpec(
        name="rv_fwd_60s",
        stage="S1",
        operator="derived.rv_fwd",
        params={'market_scope': 'Futures', 'window_s': 60, 'resample': '1s'},
        group="Forward RV",
        description=(
            "Forward realized volatility over next 60s: "
            "sqrt(sum(r_1s^2, t+1..t+60)) where r_1s = log-return. "
            "Fractional (not annualized)."
        ),
        depends_on=(Dep(name="mid_fut_1s", kind="col"),),
        feature_id=1331,
    ),
    FeatureSpec(
        name="rv_fwd_300s",
        stage="S1",
        operator="derived.rv_fwd",
        params={'market_scope': 'Futures', 'window_s': 300, 'resample': '1s'},
        group="Forward RV",
        description=(
            "Forward realized volatility over next 300s (5 min): "
            "sqrt(sum(r_1s^2, t+1..t+300)). Fractional."
        ),
        depends_on=(Dep(name="mid_fut_1s", kind="col"),),
        feature_id=1332,
    ),
    FeatureSpec(
        name="rv_fwd_900s",
        stage="S1",
        operator="derived.rv_fwd",
        params={'market_scope': 'Futures', 'window_s': 900, 'resample': '1s'},
        group="Forward RV",
        description=(
            "Forward realized volatility over next 900s (15 min): "
            "sqrt(sum(r_1s^2, t+1..t+900)). Fractional."
        ),
        depends_on=(Dep(name="mid_fut_1s", kind="col"),),
        feature_id=1333,
    ),
]