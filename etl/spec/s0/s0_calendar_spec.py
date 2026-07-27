# etl/spec/s0/s0_calendar_spec.py
# ==============================================================================
# S0 Calendar Feature Specifications
#
# PURPOSE:
#   Declarative spec for all calendar/session features. These features are
#   completely venue- and asset-independent: derived purely from bucket_dt_utc.
#
# ARCHITECTURE CONTEXT:
#   Pipeline: Binance-only, multi-asset (BTC/ETH/BNB).
#   Calendar features are identical for all assets (same time grid).
#
# DST NOTE:
#   All session times are FIXED UTC (no DST adjustment). Intentional:
#   ensures consistency in the 24/7 crypto context with stable joins
#   and rolling windows.
#
# ==============================================================================

from __future__ import annotations

from typing import List

from etl.spec import FeatureSpec, Dep


S0_CALENDAR_FEATURES: List[FeatureSpec] = [

    # =========================================================================
    # SESSION: REGIONAL SESSIONS (Fixed UTC, no DST)
    # =========================================================================

    FeatureSpec(
        name="session_sydney",
        stage="S0",
        operator="calendar.session",
        params={"session": "sydney", "window_s": "1", "resample": "1s"},
        label="Session: Sydney",
        group="Session",
        description="1 if Sydney session active (22:00-07:00 UTC, fixed). Early Asia/Oceania trading.",
        depends_on=(),
        feature_id=22,
    ),

    FeatureSpec(
        name="session_tokyo",
        stage="S0",
        operator="calendar.session",
        params={"session": "tokyo", "window_s": "1", "resample": "1s"},
        label="Session: Tokyo",
        group="Session",
        description="1 if Tokyo session active (00:00-09:00 UTC, fixed). Core Asian trading hours.",
        depends_on=(),
        feature_id=23,
    ),

    FeatureSpec(
        name="session_asia",
        stage="S0",
        operator="calendar.session",
        params={"session": "asia", "window_s": "1", "resample": "1s"},
        label="Session: Asia",
        group="Session",
        description="1 if Asia session active (Sydney OR Tokyo, fixed UTC). Combined Asian trading window.",
        depends_on=(),
        feature_id=24,
    ),

    FeatureSpec(
        name="session_london",
        stage="S0",
        operator="calendar.session",
        params={"session": "london", "window_s": "1", "resample": "1s"},
        label="Session: London",
        group="Session",
        description="1 if London session active (08:00-17:00 UTC, fixed). European trading hours.",
        depends_on=(),
        feature_id=25,
    ),

    FeatureSpec(
        name="session_newyork",
        stage="S0",
        operator="calendar.session",
        params={"session": "newyork", "window_s": "1", "resample": "1s"},
        label="Session: New York",
        group="Session",
        description="1 if New York session active (13:00-22:00 UTC, fixed). US trading hours.",
        depends_on=(),
        feature_id=26,
    ),

    # =========================================================================
    # SESSION: OVERLAP
    # =========================================================================

    FeatureSpec(
        name="session_overlap_flag",
        stage="S0",
        operator="calendar.session_overlap",
        params={"sessions": "london,newyork", "window_s": "1", "resample": "1s"},
        label="Session: London/NY Overlap",
        group="Session",
        description="1 if London AND New York overlap (13:00-17:00 UTC, fixed). Peak global liquidity.",
        depends_on=(),
        feature_id=27,
    ),

    # =========================================================================
    # HOLIDAYS & TRADING HOURS
    # =========================================================================

    FeatureSpec(
        name="us_holiday",
        stage="S0",
        operator="calendar.holiday",
        params={"calendar": "us", "window_s": "1", "resample": "1s"},
        label="US Holiday",
        group="Session",
        description="1 if US market holiday (NYSE/CME observed). Reduced institutional volume.",
        depends_on=(),
        feature_id=28,
    ),

    FeatureSpec(
        name="us_rth",
        stage="S0",
        operator="calendar.rth",
        params={"market": "us", "window_s": "1", "resample": "1s"},
        label="US Regular Trading Hours",
        group="Session",
        description="1 if US RTH (14:30-21:00 UTC fixed, Mon-Fri, non-holiday). Core equity session.",
        depends_on=(),
        feature_id=29,
    ),
]


# ==============================================================================
# Convenience helpers
# ==============================================================================

def get_calendar_feature_names() -> List[str]:
    """Return ordered list of calendar feature column names."""
    return [fs.name for fs in S0_CALENDAR_FEATURES]


def get_calendar_feature_dtypes() -> dict:
    """Return {column_name: dtype} for all calendar features. All are int8."""
    return {fs.name: "int8" for fs in S0_CALENDAR_FEATURES}