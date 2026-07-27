# etl/spec/s1/s1_range.py
# ==============================================================================
# S1 Feature Specs: Range Position
#
# Binance-only pipeline | Source: S0 features + OHLC / weekly / monthly context
# 18 features | Feature IDs: 1301–1312, 1408–1413
#
# PURPOSE:
#   Capture where price is relative to its operating range across three
#   complementary timescales:
#
#   (A) HOURLY ROLLING — features 1301–1304
#       Use existing derived.range_pct / derived.range_pos operators on
#       mid_spot_1s / mid_fut_1s with window_s=3600. No OHLC dependency.
#
#   (B) DAILY OHLC — features 1305–1312
#       Use pre-computed day_high_spot/low/fut columns injected by the
#       S1 engine OHLC context join.
#
#   (C) FIBONACCI RETRACEMENTS — features 1408–1413  (Phase 4)
#       BPS distance from mid_fut_1s to configurable fibonacci levels
#       within the weekly/monthly ranges. Anchors mean-reversion bias
#       to specific 38.2% / 50% / 61.8% / 78.6% levels.
#
#       fib_price = low + fib_level * (high - low)
#       dist_bps  = (mid - fib_price) / mid * 10000
#
#       Weekly: 0.382 / 0.500 / 0.618 / 0.786  (all four standard Fibs)
#       Monthly: 0.382 / 0.618 only  (Golden-Zone pair; avoids Fib
#                                     over-crowding on the longer timescale)
#
# OPERATORS:
#   Hourly:  derived.range_pct, derived.range_pos
#   Daily:   range.dist_to_high_bps, range.dist_to_low_bps,
#            range.ext_position, range.ext_range_bps
#   Fibs:    derived.fib_dist_bps  (Phase 4, Deps: [price, low, high])
#
# NAMING CONVENTION:
#   Hourly: {concept}_{market}_{timescale}
#   Daily:  {concept}_day_{market}
#   Fibs:   dist_to_fib_{level_x1000}_{week,month}_bps_fut
#
# ==============================================================================

from typing import List
from etl.spec import FeatureSpec, Dep


S1_RANGE_FEATURES: List[FeatureSpec] = [

    # =========================================================================
    # (A) HOURLY ROLLING — no OHLC dependency, available immediately
    # =========================================================================

    # --- Spot ---
    FeatureSpec(
        name="range_pct_spot_3600s",
        stage="S1",
        operator="derived.range_pct",
        params={"market_scope": "Spot", "window_s": 3600},
        group="Range",
        description=(
            "Rolling 1-hour range as % of mid: (max - min) / (mid + eps). "
            "Captures intraday volatility magnitude without direction. "
            "Uses context window to avoid warmup NaN at hour boundaries."
        ),
        depends_on=(Dep(name="mid_spot_1s", kind="col"),),
        feature_id=1301,
    ),

    FeatureSpec(
        name="range_pos_spot_3600s",
        stage="S1",
        operator="derived.range_pos",
        params={"market_scope": "Spot", "window_s": 3600},
        group="Range",
        description=(
            "Position within rolling 1-hour range: (mid - min) / (max - min + eps). "
            "0 = at hourly low, 1 = at hourly high. "
            "Mean-reversion models interpret values near 0/1 as potential reversals."
        ),
        depends_on=(Dep(name="mid_spot_1s", kind="col"),),
        feature_id=1302,
    ),

    # --- Futures ---
    FeatureSpec(
        name="range_pct_fut_3600s",
        stage="S1",
        operator="derived.range_pct",
        params={"market_scope": "Futures", "window_s": 3600},
        group="Range",
        description=(
            "Rolling 1-hour range % for futures mid. "
            "Compare with range_pct_spot_3600s: divergence indicates "
            "futures-specific vol episodes (e.g. liquidation cascades)."
        ),
        depends_on=(Dep(name="mid_fut_1s", kind="col"),),
        feature_id=1303,
    ),

    FeatureSpec(
        name="range_pos_fut_3600s",
        stage="S1",
        operator="derived.range_pos",
        params={"market_scope": "Futures", "window_s": 3600},
        group="Range",
        description="Position within rolling 1-hour range for futures mid.",
        depends_on=(Dep(name="mid_fut_1s", kind="col"),),
        feature_id=1304,
    ),

    # =========================================================================
    # (B) DAILY OHLC — requires OHLC context join in S1 engine
    #     Columns day_high_spot, day_low_spot, day_high_fut, day_low_fut
    #     are injected by _join_ohlc() in s1_feature_engine.py.
    #     If OHLC parquet is absent for a date, these columns are NaN
    #     and all features below degrade gracefully to NaN.
    # =========================================================================

    # --- Spot ---
    FeatureSpec(
        name="dist_to_day_high_bps_spot",
        stage="S1",
        operator="range.dist_to_high_bps",
        params={"market_scope": "Spot"},
        group="Range",
        description=(
            "Distance from current spot mid to full-day high, in basis points: "
            "(day_high - mid) / mid * 10000. "
            "0 = at the day high, positive = below day high. "
            "Proximity to day high signals potential resistance / exhaustion zone."
        ),
        depends_on=(
            Dep(name="mid_spot_1s",    kind="col"),
            Dep(name="day_high_spot",  kind="col"),
        ),
        feature_id=1305,
    ),

    FeatureSpec(
        name="dist_to_day_low_bps_spot",
        stage="S1",
        operator="range.dist_to_low_bps",
        params={"market_scope": "Spot"},
        group="Range",
        description=(
            "Distance from current spot mid to full-day low, in basis points: "
            "(mid - day_low) / mid * 10000. "
            "0 = at the day low, positive = above day low. "
            "Proximity to day low signals potential support / mean-reversion zone."
        ),
        depends_on=(
            Dep(name="mid_spot_1s",   kind="col"),
            Dep(name="day_low_spot",  kind="col"),
        ),
        feature_id=1306,
    ),

    FeatureSpec(
        name="range_pos_day_spot",
        stage="S1",
        operator="range.ext_position",
        params={"market_scope": "Spot"},
        group="Range",
        description=(
            "Position of spot mid within full-day range: "
            "(mid - day_low) / (day_high - day_low + eps). "
            "0 = at day low, 1 = at day high. "
            "Continuous regime signal; used to condition all other microstructure "
            "features on day-level structural context."
        ),
        depends_on=(
            Dep(name="mid_spot_1s",   kind="col"),
            Dep(name="day_low_spot",  kind="col"),
            Dep(name="day_high_spot", kind="col"),
        ),
        feature_id=1307,
    ),

    FeatureSpec(
        name="day_range_bps_spot",
        stage="S1",
        operator="range.ext_range_bps",
        params={"market_scope": "Spot"},
        group="Range",
        description=(
            "Full day's price range in basis points: "
            "(day_high - day_low) / mid_day * 10000. "
            "Broad daily volatility proxy. Low values indicate compression "
            "(consolidation), high values indicate expansion (trending day)."
        ),
        depends_on=(
            Dep(name="day_high_spot", kind="col"),
            Dep(name="day_low_spot",  kind="col"),
        ),
        feature_id=1308,
    ),

    # --- Futures ---
    FeatureSpec(
        name="dist_to_day_high_bps_fut",
        stage="S1",
        operator="range.dist_to_high_bps",
        params={"market_scope": "Futures"},
        group="Range",
        description=(
            "Distance from futures mid to full-day futures high, in bps. "
            "Compares with spot variant to detect cross-market divergence at "
            "intraday extremes."
        ),
        depends_on=(
            Dep(name="mid_fut_1s",    kind="col"),
            Dep(name="day_high_fut",  kind="col"),
        ),
        feature_id=1309,
    ),

    FeatureSpec(
        name="dist_to_day_low_bps_fut",
        stage="S1",
        operator="range.dist_to_low_bps",
        params={"market_scope": "Futures"},
        group="Range",
        description=(
            "Distance from futures mid to full-day futures low, in bps."
        ),
        depends_on=(
            Dep(name="mid_fut_1s",   kind="col"),
            Dep(name="day_low_fut",  kind="col"),
        ),
        feature_id=1310,
    ),

    FeatureSpec(
        name="range_pos_day_fut",
        stage="S1",
        operator="range.ext_position",
        params={"market_scope": "Futures"},
        group="Range",
        description=(
            "Position of futures mid within full-day futures range (0=low, 1=high)."
        ),
        depends_on=(
            Dep(name="mid_fut_1s",   kind="col"),
            Dep(name="day_low_fut",  kind="col"),
            Dep(name="day_high_fut", kind="col"),
        ),
        feature_id=1311,
    ),

    FeatureSpec(
        name="day_range_bps_fut",
        stage="S1",
        operator="range.ext_range_bps",
        params={"market_scope": "Futures"},
        group="Range",
        description=(
            "Full day's futures price range in basis points. "
            "Compare with day_range_bps_spot to detect spot-futures "
            "vol divergence on a daily timescale."
        ),
        depends_on=(
            Dep(name="day_high_fut", kind="col"),
            Dep(name="day_low_fut",  kind="col"),
        ),
        feature_id=1312,
    ),

    # =========================================================================
    #     Deps: [mid_fut_1s, <range>_low_fut, <range>_high_fut]
    #     Operator: derived.fib_dist_bps (fib_level parameterised).
    #     fib_price = low + fib_level * (high - low);
    #     (mid - fib_price) / mid * 10000
    # =========================================================================

    # --- Weekly fibs (0.382 / 0.500 / 0.618 / 0.786) ---
    FeatureSpec(
        name="dist_to_fib_382_week_bps_fut",
        stage="S1",
        operator="derived.fib_dist_bps",
        params={"market_scope": "Futures", "fib_level": 0.382, "resample": "1s"},
        group="Range",
        description=(
            "BPS distance from mid_fut to the 38.2% retracement of the "
            "week-to-date range (week_low .. week_high)."
        ),
        depends_on=(
            Dep(name="mid_fut_1s",    kind="col"),
            Dep(name="week_low_fut",  kind="col"),
            Dep(name="week_high_fut", kind="col"),
        ),
        feature_id=1408,
    ),
    FeatureSpec(
        name="dist_to_fib_500_week_bps_fut",
        stage="S1",
        operator="derived.fib_dist_bps",
        params={"market_scope": "Futures", "fib_level": 0.500, "resample": "1s"},
        group="Range",
        description=(
            "BPS distance from mid_fut to the 50.0% retracement of the "
            "week-to-date range."
        ),
        depends_on=(
            Dep(name="mid_fut_1s",    kind="col"),
            Dep(name="week_low_fut",  kind="col"),
            Dep(name="week_high_fut", kind="col"),
        ),
        feature_id=1409,
    ),
    FeatureSpec(
        name="dist_to_fib_618_week_bps_fut",
        stage="S1",
        operator="derived.fib_dist_bps",
        params={"market_scope": "Futures", "fib_level": 0.618, "resample": "1s"},
        group="Range",
        description=(
            "BPS distance from mid_fut to the 61.8% retracement of the "
            "week-to-date range (Golden Ratio)."
        ),
        depends_on=(
            Dep(name="mid_fut_1s",    kind="col"),
            Dep(name="week_low_fut",  kind="col"),
            Dep(name="week_high_fut", kind="col"),
        ),
        feature_id=1410,
    ),
    FeatureSpec(
        name="dist_to_fib_786_week_bps_fut",
        stage="S1",
        operator="derived.fib_dist_bps",
        params={"market_scope": "Futures", "fib_level": 0.786, "resample": "1s"},
        group="Range",
        description=(
            "BPS distance from mid_fut to the 78.6% retracement of the "
            "week-to-date range."
        ),
        depends_on=(
            Dep(name="mid_fut_1s",    kind="col"),
            Dep(name="week_low_fut",  kind="col"),
            Dep(name="week_high_fut", kind="col"),
        ),
        feature_id=1411,
    ),

    # --- Monthly fibs (Golden-Zone pair: 0.382 / 0.618) ---
    FeatureSpec(
        name="dist_to_fib_382_month_bps_fut",
        stage="S1",
        operator="derived.fib_dist_bps",
        params={"market_scope": "Futures", "fib_level": 0.382, "resample": "1s"},
        group="Range",
        description=(
            "BPS distance from mid_fut to the 38.2% retracement of the "
            "month-to-date range (month_low .. month_high)."
        ),
        depends_on=(
            Dep(name="mid_fut_1s",     kind="col"),
            Dep(name="month_low_fut",  kind="col"),
            Dep(name="month_high_fut", kind="col"),
        ),
        feature_id=1412,
    ),
    FeatureSpec(
        name="dist_to_fib_618_month_bps_fut",
        stage="S1",
        operator="derived.fib_dist_bps",
        params={"market_scope": "Futures", "fib_level": 0.618, "resample": "1s"},
        group="Range",
        description=(
            "BPS distance from mid_fut to the 61.8% retracement of the "
            "month-to-date range (Golden Ratio)."
        ),
        depends_on=(
            Dep(name="mid_fut_1s",     kind="col"),
            Dep(name="month_low_fut",  kind="col"),
            Dep(name="month_high_fut", kind="col"),
        ),
        feature_id=1413,
    ),
]