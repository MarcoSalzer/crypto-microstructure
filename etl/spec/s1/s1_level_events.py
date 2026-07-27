# etl/spec/s1/s1_level_events.py
# ==============================================================================
# S1 Feature Specs: Level Events (Above/Below + Reclaim/Break, Debounced)
#
# Binance-only pipeline | Source: S0 features + session level columns from
# _join_weekly / _join_ohlc context joins.
# 17 features | Feature IDs: 1391-1407
#
# PURPOSE:
#   Binary event flags around structural levels (week_open, prev_day_high/low,
#   prev_week_high/low). Three flavours:
#
#   (A) Simple above/below (5 features):
#       Instantaneous flag: 1 if price > level (or <, for below_* variants).
#       No time-window — just a per-tick comparison.
#
#   (B) Reclaim flags (2 features):
#       Debounced event: 1 for window_s seconds after price crosses UP
#       through the level, as long as mid stays above. Captures "level was
#       just reclaimed from below". Applied only to week_open (reclaim of
#       weekly open is a standard trading setup).
#
#   (C) Break flags, upward (5 features):
#       Same debounce mechanic as reclaim but applied to resistance levels
#       (prev_day_high, prev_week_high). Marks the "breakout is still fresh"
#       window where trend-continuation probability is elevated.
#
#   (D) Break flags, downward (5 features):
#       Mirror of (C) for support levels: price crosses DOWN through
#       prev_day_low / prev_week_low and stays below for window_s seconds.
#
# OPERATORS:
#   derived.above_level      (1-tick binary)
#   derived.below_level      (1-tick binary)
#   derived.reclaim_flag     (debounced upward cross)
#   derived.break_flag_high  (debounced upward cross through resistance)
#   derived.break_flag_low   (debounced downward cross through support)
#
# DEBOUNCING:
#   Flags stay active for window_s seconds after the exact cross tick,
#   subject to the post-cross price condition still holding. Avoids the
#   sparsity problem where a naive "current cross" flag is 1 for a single
#   tick and 0 for the next 3599s — unlearnable for tree models.
#
# NaN BEHAVIOUR:
#   If the level column is NaN (missing weekly/ohlc parquet), the flag
#   evaluates to 0 (boolean cast of comparison with NaN is False). Not
#   NaN. This matches how tree models interpret "no signal".
#
# ==============================================================================

from typing import List
from etl.spec import FeatureSpec, Dep


S1_LEVEL_EVENTS_FEATURES: List[FeatureSpec] = [

    # =========================================================================
    # (A) Simple above/below (5) — derived.above_level / below_level
    # =========================================================================

    FeatureSpec(
        name="above_week_open_fut",
        stage="S1",
        operator="derived.above_level",
        params={"market_scope": "Futures", "resample": "1s"},
        group="Level Events",
        description="1 if mid_fut > week_open_fut else 0. Week-to-date direction flag.",
        depends_on=(
            Dep(name="mid_fut_1s",    kind="col"),
            Dep(name="week_open_fut", kind="col"),
        ),
        feature_id=1391,
    ),
    FeatureSpec(
        name="above_prev_day_high_fut",
        stage="S1",
        operator="derived.above_level",
        params={"market_scope": "Futures", "resample": "1s"},
        group="Level Events",
        description="1 if mid_fut > prev_day_high_fut else 0.",
        depends_on=(
            Dep(name="mid_fut_1s",        kind="col"),
            Dep(name="prev_day_high_fut", kind="col"),
        ),
        feature_id=1392,
    ),
    FeatureSpec(
        name="below_prev_day_low_fut",
        stage="S1",
        operator="derived.below_level",
        params={"market_scope": "Futures", "resample": "1s"},
        group="Level Events",
        description="1 if mid_fut < prev_day_low_fut else 0.",
        depends_on=(
            Dep(name="mid_fut_1s",       kind="col"),
            Dep(name="prev_day_low_fut", kind="col"),
        ),
        feature_id=1393,
    ),
    FeatureSpec(
        name="above_prev_week_high_fut",
        stage="S1",
        operator="derived.above_level",
        params={"market_scope": "Futures", "resample": "1s"},
        group="Level Events",
        description="1 if mid_fut > prev_week_high_fut else 0.",
        depends_on=(
            Dep(name="mid_fut_1s",         kind="col"),
            Dep(name="prev_week_high_fut", kind="col"),
        ),
        feature_id=1394,
    ),
    FeatureSpec(
        name="below_prev_week_low_fut",
        stage="S1",
        operator="derived.below_level",
        params={"market_scope": "Futures", "resample": "1s"},
        group="Level Events",
        description="1 if mid_fut < prev_week_low_fut else 0.",
        depends_on=(
            Dep(name="mid_fut_1s",        kind="col"),
            Dep(name="prev_week_low_fut", kind="col"),
        ),
        feature_id=1395,
    ),

    # =========================================================================
    # (B) Reclaim flags (2) — derived.reclaim_flag (debounced upward cross)
    # =========================================================================

    FeatureSpec(
        name="reclaimed_week_open_300s_fut",
        stage="S1",
        operator="derived.reclaim_flag",
        params={"market_scope": "Futures", "window_s": 300, "resample": "1s"},
        group="Level Events",
        description=(
            "Debounced flag: 1 for 300s after mid_fut crosses UP through "
            "week_open_fut, while mid stays above. Marks 'weekly open "
            "reclaimed' window for trend-continuation setups."
        ),
        depends_on=(
            Dep(name="mid_fut_1s",    kind="col"),
            Dep(name="week_open_fut", kind="col"),
        ),
        feature_id=1396,
    ),
    FeatureSpec(
        name="reclaimed_week_open_900s_fut",
        stage="S1",
        operator="derived.reclaim_flag",
        params={"market_scope": "Futures", "window_s": 900, "resample": "1s"},
        group="Level Events",
        description="Debounced weekly-open reclaim flag, 900s window.",
        depends_on=(
            Dep(name="mid_fut_1s",    kind="col"),
            Dep(name="week_open_fut", kind="col"),
        ),
        feature_id=1397,
    ),

    # =========================================================================
    # (C) Break flags upward (5) — derived.break_flag_high
    # =========================================================================

    FeatureSpec(
        name="broke_prev_day_high_60s_fut",
        stage="S1",
        operator="derived.break_flag_high",
        params={"market_scope": "Futures", "window_s": 60, "resample": "1s"},
        group="Level Events",
        description=(
            "Debounced flag: 1 for 60s after mid_fut crosses UP through "
            "prev_day_high_fut, while price remains above. Fresh breakout."
        ),
        depends_on=(
            Dep(name="mid_fut_1s",        kind="col"),
            Dep(name="prev_day_high_fut", kind="col"),
        ),
        feature_id=1398,
    ),
    FeatureSpec(
        name="broke_prev_day_high_300s_fut",
        stage="S1",
        operator="derived.break_flag_high",
        params={"market_scope": "Futures", "window_s": 300, "resample": "1s"},
        group="Level Events",
        description="Debounced prev_day_high break flag, 300s window.",
        depends_on=(
            Dep(name="mid_fut_1s",        kind="col"),
            Dep(name="prev_day_high_fut", kind="col"),
        ),
        feature_id=1399,
    ),
    FeatureSpec(
        name="broke_prev_day_high_900s_fut",
        stage="S1",
        operator="derived.break_flag_high",
        params={"market_scope": "Futures", "window_s": 900, "resample": "1s"},
        group="Level Events",
        description="Debounced prev_day_high break flag, 900s window.",
        depends_on=(
            Dep(name="mid_fut_1s",        kind="col"),
            Dep(name="prev_day_high_fut", kind="col"),
        ),
        feature_id=1400,
    ),
    FeatureSpec(
        name="broke_prev_week_high_300s_fut",
        stage="S1",
        operator="derived.break_flag_high",
        params={"market_scope": "Futures", "window_s": 300, "resample": "1s"},
        group="Level Events",
        description="Debounced prev_week_high break flag, 300s window.",
        depends_on=(
            Dep(name="mid_fut_1s",         kind="col"),
            Dep(name="prev_week_high_fut", kind="col"),
        ),
        feature_id=1401,
    ),
    FeatureSpec(
        name="broke_prev_week_high_900s_fut",
        stage="S1",
        operator="derived.break_flag_high",
        params={"market_scope": "Futures", "window_s": 900, "resample": "1s"},
        group="Level Events",
        description="Debounced prev_week_high break flag, 900s window.",
        depends_on=(
            Dep(name="mid_fut_1s",         kind="col"),
            Dep(name="prev_week_high_fut", kind="col"),
        ),
        feature_id=1402,
    ),

    # =========================================================================
    # (D) Break flags downward (5) — derived.break_flag_low
    # =========================================================================

    FeatureSpec(
        name="broke_prev_day_low_60s_fut",
        stage="S1",
        operator="derived.break_flag_low",
        params={"market_scope": "Futures", "window_s": 60, "resample": "1s"},
        group="Level Events",
        description=(
            "Debounced flag: 1 for 60s after mid_fut crosses DOWN through "
            "prev_day_low_fut, while price remains below. Fresh breakdown."
        ),
        depends_on=(
            Dep(name="mid_fut_1s",       kind="col"),
            Dep(name="prev_day_low_fut", kind="col"),
        ),
        feature_id=1403,
    ),
    FeatureSpec(
        name="broke_prev_day_low_300s_fut",
        stage="S1",
        operator="derived.break_flag_low",
        params={"market_scope": "Futures", "window_s": 300, "resample": "1s"},
        group="Level Events",
        description="Debounced prev_day_low break flag, 300s window.",
        depends_on=(
            Dep(name="mid_fut_1s",       kind="col"),
            Dep(name="prev_day_low_fut", kind="col"),
        ),
        feature_id=1404,
    ),
    FeatureSpec(
        name="broke_prev_day_low_900s_fut",
        stage="S1",
        operator="derived.break_flag_low",
        params={"market_scope": "Futures", "window_s": 900, "resample": "1s"},
        group="Level Events",
        description="Debounced prev_day_low break flag, 900s window.",
        depends_on=(
            Dep(name="mid_fut_1s",       kind="col"),
            Dep(name="prev_day_low_fut", kind="col"),
        ),
        feature_id=1405,
    ),
    FeatureSpec(
        name="broke_prev_week_low_300s_fut",
        stage="S1",
        operator="derived.break_flag_low",
        params={"market_scope": "Futures", "window_s": 300, "resample": "1s"},
        group="Level Events",
        description="Debounced prev_week_low break flag, 300s window.",
        depends_on=(
            Dep(name="mid_fut_1s",        kind="col"),
            Dep(name="prev_week_low_fut", kind="col"),
        ),
        feature_id=1406,
    ),
    FeatureSpec(
        name="broke_prev_week_low_900s_fut",
        stage="S1",
        operator="derived.break_flag_low",
        params={"market_scope": "Futures", "window_s": 900, "resample": "1s"},
        group="Level Events",
        description="Debounced prev_week_low break flag, 900s window.",
        depends_on=(
            Dep(name="mid_fut_1s",        kind="col"),
            Dep(name="prev_week_low_fut", kind="col"),
        ),
        feature_id=1407,
    ),
]