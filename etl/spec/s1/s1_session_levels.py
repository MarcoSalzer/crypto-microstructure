# etl/spec/s1/s1_session_levels.py
# ==============================================================================
# S1 Feature Specs: Session Levels (Week / Month / Monday / Prev-Day)
#
# Binance-only pipeline | Source: S0 + weekly/monthly/ohlc join artefacts
# 18 features | Feature IDs: 1359-1376
# (14 passthrough level-columns injected via _join_weekly / _join_monthly /
#  _join_ohlc in s1_feature_engine.py are NOT FeatureSpec entries.)
#
# PURPOSE:
#   Session-level distance, range-position and range-size features that
#   anchor the current price against longer-horizon session extremes.
#   All operate on Futures mid (mid_fut_1s) against pre-computed session
#   level columns injected by the engine's context-join step:
#
#     week_open_fut, week_high_fut, week_low_fut     (from _join_weekly)
#     prev_week_high_fut, prev_week_low_fut          (from _join_weekly)
#     monday_high_fut, monday_low_fut                (from _join_weekly)
#     month_open_fut, month_high_fut, month_low_fut  (from _join_monthly)
#     prev_month_high_fut, prev_month_low_fut        (from _join_monthly)
#     prev_day_high_fut, prev_day_low_fut            (from _join_ohlc)
#
# STRUCTURE:
#   (A) 14 distance features (range.dist_to_level_bps):
#       (mid - level) / mid * 10000. Sign follows price position.
#
#   (B)  2 range-position features (range.ext_position):
#       (mid - low) / (high - low + eps) ∈ [0, 1].
#
#   (C)  2 range-size features (range.ext_range_bps):
#       (high - low) / mid * 10000. Volatility proxy.
#
# NaN BEHAVIOUR:
#   If the weekly/monthly/ohlc parquet for the relevant ISO-week / month
#   / prev-day is missing, the injected columns are NaN and every feature
#   here degrades gracefully to NaN.
#
# ==============================================================================

from typing import List
from etl.spec import FeatureSpec, Dep


S1_SESSION_LEVELS_FEATURES: List[FeatureSpec] = [

    # =========================================================================
    # (A) Distance features (14) — range.dist_to_level_bps
    #     (mid_fut - level) / mid_fut * 10000. Sign follows (mid - level).
    # =========================================================================

    # --- Weekly levels ---
    FeatureSpec(
        name="dist_to_week_open_bps_fut",
        stage="S1",
        operator="range.dist_to_level_bps",
        params={"market_scope": "Futures"},
        group="Session Levels",
        description=(
            "BPS distance from mid_fut to week_open_fut: "
            "(mid - week_open) / mid * 10000. Positive = above weekly open, "
            "negative = below. Reference for week-to-date drift."
        ),
        depends_on=(
            Dep(name="mid_fut_1s",   kind="col"),
            Dep(name="week_open_fut", kind="col"),
        ),
        feature_id=1359,
    ),
    FeatureSpec(
        name="dist_to_week_high_bps_fut",
        stage="S1",
        operator="range.dist_to_level_bps",
        params={"market_scope": "Futures"},
        group="Session Levels",
        description=(
            "BPS distance from mid_fut to week_high_fut (expanding max since "
            "week start). Typically <= 0 since week_high >= mid by construction."
        ),
        depends_on=(
            Dep(name="mid_fut_1s",   kind="col"),
            Dep(name="week_high_fut", kind="col"),
        ),
        feature_id=1360,
    ),
    FeatureSpec(
        name="dist_to_week_low_bps_fut",
        stage="S1",
        operator="range.dist_to_level_bps",
        params={"market_scope": "Futures"},
        group="Session Levels",
        description=(
            "BPS distance from mid_fut to week_low_fut (expanding min since "
            "week start). Typically >= 0 since week_low <= mid by construction."
        ),
        depends_on=(
            Dep(name="mid_fut_1s",  kind="col"),
            Dep(name="week_low_fut", kind="col"),
        ),
        feature_id=1361,
    ),
    FeatureSpec(
        name="dist_to_prev_week_high_bps_fut",
        stage="S1",
        operator="range.dist_to_level_bps",
        params={"market_scope": "Futures"},
        group="Session Levels",
        description=(
            "BPS distance from mid_fut to the preceding ISO-week's high "
            "(constant throughout the target week). Supply/resistance anchor."
        ),
        depends_on=(
            Dep(name="mid_fut_1s",        kind="col"),
            Dep(name="prev_week_high_fut", kind="col"),
        ),
        feature_id=1362,
    ),
    FeatureSpec(
        name="dist_to_prev_week_low_bps_fut",
        stage="S1",
        operator="range.dist_to_level_bps",
        params={"market_scope": "Futures"},
        group="Session Levels",
        description="BPS distance from mid_fut to the preceding ISO-week's low.",
        depends_on=(
            Dep(name="mid_fut_1s",       kind="col"),
            Dep(name="prev_week_low_fut", kind="col"),
        ),
        feature_id=1363,
    ),

    # --- Monthly levels ---
    FeatureSpec(
        name="dist_to_month_open_bps_fut",
        stage="S1",
        operator="range.dist_to_level_bps",
        params={"market_scope": "Futures"},
        group="Session Levels",
        description=(
            "BPS distance from mid_fut to month_open_fut (1st of calendar month, "
            "00:00 UTC). Reference for month-to-date drift."
        ),
        depends_on=(
            Dep(name="mid_fut_1s",    kind="col"),
            Dep(name="month_open_fut", kind="col"),
        ),
        feature_id=1364,
    ),
    FeatureSpec(
        name="dist_to_month_high_bps_fut",
        stage="S1",
        operator="range.dist_to_level_bps",
        params={"market_scope": "Futures"},
        group="Session Levels",
        description=(
            "BPS distance from mid_fut to month_high_fut (expanding max since "
            "month start)."
        ),
        depends_on=(
            Dep(name="mid_fut_1s",    kind="col"),
            Dep(name="month_high_fut", kind="col"),
        ),
        feature_id=1365,
    ),
    FeatureSpec(
        name="dist_to_month_low_bps_fut",
        stage="S1",
        operator="range.dist_to_level_bps",
        params={"market_scope": "Futures"},
        group="Session Levels",
        description=(
            "BPS distance from mid_fut to month_low_fut (expanding min since "
            "month start)."
        ),
        depends_on=(
            Dep(name="mid_fut_1s",   kind="col"),
            Dep(name="month_low_fut", kind="col"),
        ),
        feature_id=1366,
    ),
    FeatureSpec(
        name="dist_to_prev_month_high_bps_fut",
        stage="S1",
        operator="range.dist_to_level_bps",
        params={"market_scope": "Futures"},
        group="Session Levels",
        description=(
            "BPS distance from mid_fut to the preceding calendar month's high."
        ),
        depends_on=(
            Dep(name="mid_fut_1s",         kind="col"),
            Dep(name="prev_month_high_fut", kind="col"),
        ),
        feature_id=1367,
    ),
    FeatureSpec(
        name="dist_to_prev_month_low_bps_fut",
        stage="S1",
        operator="range.dist_to_level_bps",
        params={"market_scope": "Futures"},
        group="Session Levels",
        description="BPS distance from mid_fut to the preceding calendar month's low.",
        depends_on=(
            Dep(name="mid_fut_1s",        kind="col"),
            Dep(name="prev_month_low_fut", kind="col"),
        ),
        feature_id=1368,
    ),

    # --- Monday levels (crypto-specific weekly session) ---
    FeatureSpec(
        name="dist_to_monday_high_bps_fut",
        stage="S1",
        operator="range.dist_to_level_bps",
        params={"market_scope": "Futures"},
        group="Session Levels",
        description=(
            "BPS distance from mid_fut to monday_high_fut "
            "(expanding Monday max, forward-filled Tue-Sun)."
        ),
        depends_on=(
            Dep(name="mid_fut_1s",     kind="col"),
            Dep(name="monday_high_fut", kind="col"),
        ),
        feature_id=1369,
    ),
    FeatureSpec(
        name="dist_to_monday_low_bps_fut",
        stage="S1",
        operator="range.dist_to_level_bps",
        params={"market_scope": "Futures"},
        group="Session Levels",
        description="BPS distance from mid_fut to monday_low_fut.",
        depends_on=(
            Dep(name="mid_fut_1s",    kind="col"),
            Dep(name="monday_low_fut", kind="col"),
        ),
        feature_id=1370,
    ),

    # --- Prev-day levels (counterpart to the existing day_high/low_fut) ---
    FeatureSpec(
        name="dist_to_prev_day_high_bps_fut",
        stage="S1",
        operator="range.dist_to_level_bps",
        params={"market_scope": "Futures"},
        group="Session Levels",
        description=(
            "BPS distance from mid_fut to the preceding calendar day's high "
            "(constant throughout the target day)."
        ),
        depends_on=(
            Dep(name="mid_fut_1s",       kind="col"),
            Dep(name="prev_day_high_fut", kind="col"),
        ),
        feature_id=1371,
    ),
    FeatureSpec(
        name="dist_to_prev_day_low_bps_fut",
        stage="S1",
        operator="range.dist_to_level_bps",
        params={"market_scope": "Futures"},
        group="Session Levels",
        description="BPS distance from mid_fut to the preceding calendar day's low.",
        depends_on=(
            Dep(name="mid_fut_1s",      kind="col"),
            Dep(name="prev_day_low_fut", kind="col"),
        ),
        feature_id=1372,
    ),

    # =========================================================================
    # (B) Range-position features (2) — range.ext_position (reused)
    #     (mid - low) / (high - low + eps), clipped [0, 1].
    # =========================================================================

    FeatureSpec(
        name="range_pos_week_fut",
        stage="S1",
        operator="range.ext_position",
        params={"market_scope": "Futures"},
        group="Session Levels",
        description=(
            "Position of mid_fut within the week-to-date range: "
            "(mid - week_low) / (week_high - week_low + eps), clipped [0, 1]. "
            "0 = at weekly low, 1 = at weekly high."
        ),
        depends_on=(
            Dep(name="mid_fut_1s",   kind="col"),
            Dep(name="week_low_fut", kind="col"),
            Dep(name="week_high_fut", kind="col"),
        ),
        feature_id=1373,
    ),
    FeatureSpec(
        name="range_pos_month_fut",
        stage="S1",
        operator="range.ext_position",
        params={"market_scope": "Futures"},
        group="Session Levels",
        description=(
            "Position of mid_fut within the month-to-date range, clipped [0, 1]."
        ),
        depends_on=(
            Dep(name="mid_fut_1s",   kind="col"),
            Dep(name="month_low_fut", kind="col"),
            Dep(name="month_high_fut", kind="col"),
        ),
        feature_id=1374,
    ),

    # =========================================================================
    # (C) Range-size features (2) — range.ext_range_bps (reused)
    #     (high - low) / mid * 10000. Volatility proxy.
    # =========================================================================

    FeatureSpec(
        name="week_range_bps_fut",
        stage="S1",
        operator="range.ext_range_bps",
        params={"market_scope": "Futures"},
        group="Session Levels",
        description=(
            "Week-to-date range in bps: (week_high - week_low) / mid_day * 10000. "
            "Broad weekly volatility proxy. Low = compression, high = expansion."
        ),
        depends_on=(
            Dep(name="week_high_fut", kind="col"),
            Dep(name="week_low_fut",  kind="col"),
        ),
        feature_id=1375,
    ),
    FeatureSpec(
        name="month_range_bps_fut",
        stage="S1",
        operator="range.ext_range_bps",
        params={"market_scope": "Futures"},
        group="Session Levels",
        description=(
            "Month-to-date range in bps. Broad monthly volatility proxy."
        ),
        depends_on=(
            Dep(name="month_high_fut", kind="col"),
            Dep(name="month_low_fut",  kind="col"),
        ),
        feature_id=1376,
    ),

    # =========================================================================
    # IDs 1377-1390 reserved as buffer (Phase 4 planning carried a 14-slot
    # reserve to account for the Category-6 ambiguity in the original user
    # spec). Not used — next block starts at 1391 in s1_level_events.py.
    # =========================================================================
]