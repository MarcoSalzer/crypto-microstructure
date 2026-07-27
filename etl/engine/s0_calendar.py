# ==============================================================================
# S0 Calendar Features — Trading Session & Holiday Flags
#
# PURPOSE:
#   Compute deterministic calendar/session features from bucket timestamps.
#   These features are completely venue- and asset-independent: they depend
#   only on the UTC timestamp of each bucket. Used for regime detection
#   (session overlaps, holidays, RTH vs ETH) in downstream feature stages.
#
# ARCHITECTURE CONTEXT:
#   Called by the S0 context builder (s0_context_batch.py) during the
#   context build step. Receives a Series of bucket_dt_utc timestamps,
#   returns a DataFrame with one row per bucket and one column per feature.
#
#   Pipeline: Binance-only, multi-asset (BTC + ETH).
#   Calendar features are identical for all assets (same time grid).
#
# DST NOTE:
#   All session windows use FIXED UTC boundaries (no DST adjustment).
#   Intentional: ensures stable joins and rolling windows in the 24/7
#   crypto context.
#
# INPUT HARDENING:
#   - Robust input coercion to datetime64[ns, UTC]
#   - NaT values dropped, duplicates dropped (keep first)
#   - Fast holiday lookup via set membership (isin)
#   - Warns if timestamps fall outside holiday calendar coverage
#
# OUTPUT COLUMNS:
#   bucket_dt_utc        datetime64[ns, UTC]
#   session_sydney       int8    1 if 22:00-07:00 UTC
#   session_tokyo        int8    1 if 00:00-09:00 UTC
#   session_asia         int8    1 if Sydney OR Tokyo
#   session_london       int8    1 if 08:00-17:00 UTC
#   session_newyork      int8    1 if 13:00-22:00 UTC
#   session_overlap_flag int8    1 if London AND New York
#   us_holiday           int8    1 if NYSE/CME observed holiday
#   us_rth               int8    1 if US RTH (14:30-21:00 UTC, Mon-Fri, non-holiday)
#
# ==============================================================================

from __future__ import annotations

from datetime import date
from typing import List, Set, Union

import pandas as pd
import numpy as np


# ==============================================================================
# US Holiday Calendar (Static List)
# ==============================================================================
# Coverage: 2025-2027. NYSE/CME-style observed holidays.
# Update this set when extending the dataset to later years.

US_HOLIDAYS: Set[date] = {
    # 2025
    date(2025, 1, 1),    # New Year's Day
    date(2025, 1, 20),   # MLK Day
    date(2025, 2, 17),   # Presidents Day
    date(2025, 4, 18),   # Good Friday
    date(2025, 5, 26),   # Memorial Day
    date(2025, 6, 19),   # Juneteenth
    date(2025, 7, 4),    # Independence Day
    date(2025, 9, 1),    # Labor Day
    date(2025, 11, 27),  # Thanksgiving
    date(2025, 12, 25),  # Christmas

    # 2026
    date(2026, 1, 1),    # New Year's Day
    date(2026, 1, 19),   # MLK Day
    date(2026, 2, 16),   # Presidents Day
    date(2026, 4, 3),    # Good Friday
    date(2026, 5, 25),   # Memorial Day
    date(2026, 6, 19),   # Juneteenth
    date(2026, 7, 3),    # Independence Day (observed)
    date(2026, 9, 7),    # Labor Day
    date(2026, 11, 26),  # Thanksgiving
    date(2026, 12, 25),  # Christmas

    # 2027
    date(2027, 1, 1),    # New Year's Day
    date(2027, 1, 18),   # MLK Day
    date(2027, 2, 15),   # Presidents Day
    date(2027, 3, 26),   # Good Friday
    date(2027, 5, 31),   # Memorial Day
    date(2027, 6, 18),   # Juneteenth (observed; 19th is Saturday)
    date(2027, 7, 5),    # Independence Day (observed; 4th is Sunday)
    date(2027, 9, 6),    # Labor Day
    date(2027, 11, 25),  # Thanksgiving
    date(2027, 12, 24),  # Christmas (observed; 25th is Saturday)
}

HOLIDAY_MIN_YEAR = 2025
HOLIDAY_MAX_YEAR = 2027

# US Regular Trading Hours boundaries (fixed UTC, no DST)
US_RTH_START_HOUR = 14
US_RTH_START_MINUTE = 30
US_RTH_END_HOUR = 21
US_RTH_END_MINUTE = 0


def compute_calendar_features(
    bucket_dt_utc: Union[pd.Series, pd.DatetimeIndex, List, np.ndarray],
    *,
    warn_on_year_outside_holiday_set: bool = True,
) -> pd.DataFrame:
    """
    Compute calendar/session features for given UTC timestamps.

    Returns DataFrame with bucket_dt_utc + all calendar feature columns (int8).
    """

    # ---- Normalize input to Series ----
    if isinstance(bucket_dt_utc, pd.DatetimeIndex):
        s = pd.Series(bucket_dt_utc)
    elif isinstance(bucket_dt_utc, pd.Series):
        s = bucket_dt_utc.copy()
    else:
        s = pd.Series(bucket_dt_utc)

    # ---- Enforce datetime UTC (invalid -> NaT -> dropped) ----
    s = pd.to_datetime(s, utc=True, errors="coerce")
    s = s.dropna()

    if s.empty:
        out = pd.DataFrame({"bucket_dt_utc": pd.Series([], dtype="datetime64[ns, UTC]")})
        for c in get_calendar_feature_names():
            out[c] = pd.Series([], dtype="int8")
        return out

    # ---- Drop duplicate buckets (join stability), preserve first occurrence ----
    s = s[~s.duplicated(keep="first")]
    df = pd.DataFrame({"bucket_dt_utc": s})

    # ---- Warn if years outside known holiday coverage ----
    if warn_on_year_outside_holiday_set:
        years = df["bucket_dt_utc"].dt.year
        min_year, max_year = int(years.min()), int(years.max())
        if min_year < HOLIDAY_MIN_YEAR or max_year > HOLIDAY_MAX_YEAR:
            print(
                "[CALENDAR] WARNING: bucket_dt_utc contains years outside US_HOLIDAYS coverage "
                f"(found {min_year}..{max_year}, coverage {HOLIDAY_MIN_YEAR}..{HOLIDAY_MAX_YEAR}). "
                "US holidays for missing years will be treated as non-holidays. "
                "Update US_HOLIDAYS when you add new years."
            )

    hours = df["bucket_dt_utc"].dt.hour
    minutes = df["bucket_dt_utc"].dt.minute
    weekdays = df["bucket_dt_utc"].dt.weekday  # 0=Mon
    dates = df["bucket_dt_utc"].dt.date

    # ---- Sessions (fixed UTC, no DST) ----
    df["session_sydney"] = ((hours >= 22) | (hours < 7)).astype("int8")
    df["session_tokyo"] = ((hours >= 0) & (hours < 9)).astype("int8")
    df["session_asia"] = ((df["session_sydney"] == 1) | (df["session_tokyo"] == 1)).astype("int8")
    df["session_london"] = ((hours >= 8) & (hours < 17)).astype("int8")
    df["session_newyork"] = ((hours >= 13) & (hours < 22)).astype("int8")
    df["session_overlap_flag"] = ((df["session_london"] == 1) & (df["session_newyork"] == 1)).astype("int8")

    # ---- US Holiday (fast set membership) ----
    df["us_holiday"] = pd.Series(dates, index=df.index).isin(US_HOLIDAYS).astype("int8")

    # ---- US RTH: 14:30-21:00 UTC, Mon-Fri, non-holiday ----
    time_in_rth = (
        ((hours > US_RTH_START_HOUR) | ((hours == US_RTH_START_HOUR) & (minutes >= US_RTH_START_MINUTE)))
        & (hours < US_RTH_END_HOUR)
    )
    is_weekday = weekdays <= 4
    df["us_rth"] = (time_in_rth & is_weekday & (df["us_holiday"] == 0)).astype("int8")

    # ---- Ensure exact column order + dtypes ----
    cols = ["bucket_dt_utc"] + get_calendar_feature_names()
    df = df[cols]
    for c in get_calendar_feature_names():
        df[c] = df[c].astype("int8")

    return df


def get_calendar_feature_names() -> List[str]:
    """Return ordered list of calendar feature column names."""
    return [
        "session_sydney",
        "session_tokyo",
        "session_asia",
        "session_london",
        "session_newyork",
        "session_overlap_flag",
        "us_holiday",
        "us_rth",
    ]


def get_calendar_feature_dtypes() -> dict:
    """Return {column_name: dtype} for all calendar features."""
    return {name: "int8" for name in get_calendar_feature_names()}