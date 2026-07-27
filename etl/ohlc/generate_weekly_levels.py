# etl/ohlc/generate_weekly_levels.py
# ==============================================================================
# Weekly Levels Generator — ISO-week high/low/open + monday/prev-week
# extrema on a 1-second grid, from S0 feature parquets.
#
# PURPOSE:
#   Compute weekly session-level context that the S1 engine joins onto
#   1s buckets, enabling features like dist_to_week_high_bps_fut,
#   range_pos_week_fut, reclaimed_week_open_*s_fut, monday_high_fut, etc.
#
#   Unlike daily OHLC (one-row-per-day), weekly levels evolve WITHIN the
#   week (expanding max/min since week start), so this artefact is stored
#   on the full 1s timeline of the week.
#
# INPUT:
#   All hourly S0 parquet files for the target ISO-week + preceding ISO-week
#   (for prev_week_* values). Located at:
#     data_storage/s0_features/s0_features_{asset}_{YYYY-MM-DD}_{HH}.parquet
#
#   An ISO-week starts Monday 00:00:00 UTC and ends Sunday 23:59:59 UTC.
#
# OUTPUT:
#   data_storage/weekly/weekly_{asset}_{iso_year}_{iso_week:02d}.parquet
#
#   Schema (one row per 1s bucket of the target week — ~604800 rows):
#     bucket_dt_utc        datetime64[ns, UTC]  — 1s-grid key, joined by S1
#     week_open_fut        float — mid_fut at first second of the week (constant)
#     week_high_fut        float — expanding max(mid_fut) since week start
#     week_low_fut         float — expanding min(mid_fut) since week start
#     monday_high_fut      float — expanding max on Monday rows only,
#                                  ffill for Tue-Sun (frozen at Monday close)
#     monday_low_fut       float — expanding min on Monday rows only, ffill
#     prev_week_high_fut   float — max(mid_fut) over the PRECEDING complete
#                                  ISO-week (constant for the full target week)
#     prev_week_low_fut    float — min(mid_fut) over the preceding ISO-week
#
# COMPLETENESS GUARD:
#   By default, requires all 168 hourly S0 parquets for the target week
#   to be present. Missing hours can be tolerated with --no-require-complete
#   for debugging; production runs should fail if data is incomplete.
#   The preceding week is optional — if unavailable, prev_week_* = NaN.
#
# IDEMPOTENCY:
#   If the output file already exists, skipped by default (--skip-existing).
#   The writer is atomic (tmp + os.replace), so a concurrent crash never
#   leaves a half-written parquet.
#
# ISO-WEEK NAMING:
#   ISO-year and ISO-week are NOT always identical to the calendar year.
#   E.g., 2021-01-01 (Fri) belongs to ISO week 53 of ISO year 2020.
#   The --iso-week CLI arg uses 'YYYY-WNN' (e.g. '2026-W03').
#
# USAGE:
#   python generate_weekly_levels.py --asset btc --iso-week 2026-W10
#   python generate_weekly_levels.py --asset eth --iso-week 2026-W12 \
#          --s0-dir /custom/s0 --out-dir /custom/weekly
#
# ==============================================================================

from __future__ import annotations
from common.paths import DATA_ROOT

import argparse
import os
import re
import tempfile
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PARQUET_COMPRESSION = "zstd"

_SCRIPT_DIR   = Path(__file__).resolve().parent
_DEFAULT_S0_DIR  = DATA_ROOT / "s0_features"
_DEFAULT_OUT_DIR = DATA_ROOT / "weekly"

_MID_FUT_CANDIDATES = ["mid_fut_1s", "mid_fut"]

_ISO_WEEK_RE = re.compile(r"^(\d{4})-W(\d{2})$")


# =============================================================================
# Utilities
# =============================================================================

def _log(msg: str, verbose: bool = True) -> None:
    if verbose:
        print(f"[{pd.Timestamp.utcnow().strftime('%H:%M:%S')}] [WEEKLY] {msg}")


def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _atomic_write_parquet(df: pd.DataFrame, out_path: Path) -> None:
    table = pa.Table.from_pandas(df, preserve_index=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".parquet.tmp", dir=str(out_path.parent))
    try:
        os.close(fd)
        pq.write_table(table, tmp, compression=PARQUET_COMPRESSION)
        os.replace(tmp, str(out_path))
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _parse_iso_week(iso_week: str) -> Tuple[int, int]:
    """Parse 'YYYY-WNN' → (iso_year, iso_week) ints. Raises ValueError on bad format."""
    m = _ISO_WEEK_RE.match(iso_week)
    if not m:
        raise ValueError(
            f"Invalid --iso-week format: {iso_week!r}. Expected 'YYYY-WNN' (e.g. 2026-W10)."
        )
    iy, iw = int(m.group(1)), int(m.group(2))
    if not (1 <= iw <= 53):
        raise ValueError(f"ISO week out of range: {iw}")
    return iy, iw


def _iso_week_bounds(iso_year: int, iso_week: int) -> Tuple[datetime, datetime]:
    """
    Return (start_utc, end_utc) timestamps for the given ISO week:
      start = Monday 00:00:00 UTC
      end   = Sunday 23:59:59 UTC (inclusive)
    """
    # ISO weekday 1 = Monday
    start_date = date.fromisocalendar(iso_year, iso_week, 1)
    end_date   = date.fromisocalendar(iso_year, iso_week, 7)
    start = datetime(start_date.year, start_date.month, start_date.day,
                     0, 0, 0, tzinfo=timezone.utc)
    end   = datetime(end_date.year, end_date.month, end_date.day,
                     23, 59, 59, tzinfo=timezone.utc)
    return start, end


def _dates_spanned(start_utc: datetime, end_utc: datetime) -> List[str]:
    """Return list of YYYY-MM-DD strings from start_utc.date() to end_utc.date() inclusive."""
    out = []
    d = start_utc.date()
    while d <= end_utc.date():
        out.append(d.strftime("%Y-%m-%d"))
        d = d + timedelta(days=1)
    return out


# =============================================================================
# S0 Loading
# =============================================================================

def _load_s0_range(
    s0_dir: str,
    asset: str,
    start_utc: datetime,
    end_utc: datetime,
    require_complete: bool,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load all hourly S0 parquets overlapping [start_utc, end_utc] inclusive
    and return a single concatenated DataFrame sliced to that exact range.

    Columns retained: bucket_dt_utc, mid_fut (auto-detected).
    Raises FileNotFoundError if require_complete and any expected hour missing.
    """
    a = asset.lower()
    s0_path = Path(s0_dir)

    # Enumerate expected (date_str, hour) pairs
    hour_tuples: List[Tuple[str, int]] = []
    cur = datetime(start_utc.year, start_utc.month, start_utc.day,
                   start_utc.hour, tzinfo=timezone.utc)
    end_hour_cur = datetime(end_utc.year, end_utc.month, end_utc.day,
                            end_utc.hour, tzinfo=timezone.utc)
    while cur <= end_hour_cur:
        hour_tuples.append((cur.strftime("%Y-%m-%d"), cur.hour))
        cur = cur + timedelta(hours=1)

    present, missing = [], []
    for date_str, h in hour_tuples:
        p = s0_path / f"s0_features_{a}_{date_str}_{h:02d}.parquet"
        if p.exists():
            present.append(p)
        else:
            missing.append((date_str, h))

    if missing and require_complete:
        miss_sample = missing[:5]
        raise FileNotFoundError(
            f"Weekly levels require all hourly S0 parquets in "
            f"[{start_utc.isoformat()}, {end_utc.isoformat()}]. "
            f"Missing {len(missing)} hours, sample: {miss_sample}"
        )
    if missing:
        _log(f"  WARN: {len(missing)} missing S0 hours (continuing with --no-require-complete)",
             verbose)

    _log(f"  Loading {len(present)} S0 hourly parquets", verbose)

    frames = []
    mid_fut_col = None
    for fp in sorted(present):
        try:
            df = pq.read_table(str(fp), columns=None).to_pandas()
            if mid_fut_col is None:
                mid_fut_col = _find_col(df, _MID_FUT_CANDIDATES)
                if mid_fut_col is None:
                    raise ValueError(
                        f"No mid_fut column found in {fp.name}. "
                        f"Tried: {_MID_FUT_CANDIDATES}"
                    )
                _log(f"  Using mid column: {mid_fut_col!r}", verbose)
            if "bucket_dt_utc" not in df.columns:
                _log(f"  WARN: skipping {fp.name} — no bucket_dt_utc column",
                     verbose)
                continue
            frames.append(df[["bucket_dt_utc", mid_fut_col]].rename(
                columns={mid_fut_col: "mid_fut"}
            ))
        except Exception as e:
            _log(f"  WARN: failed to load {fp.name}: {e}", verbose)

    if not frames:
        return pd.DataFrame(columns=["bucket_dt_utc", "mid_fut"])

    combined = pd.concat(frames, ignore_index=True)
    combined["bucket_dt_utc"] = pd.to_datetime(combined["bucket_dt_utc"], utc=True)

    # Strict slice to [start_utc, end_utc]
    mask = (combined["bucket_dt_utc"] >= pd.Timestamp(start_utc)) & \
           (combined["bucket_dt_utc"] <= pd.Timestamp(end_utc))
    combined = combined.loc[mask].copy()
    combined = combined.sort_values("bucket_dt_utc").reset_index(drop=True)
    combined = combined.drop_duplicates(subset=["bucket_dt_utc"], keep="last")

    _log(f"  Loaded {len(combined)} rows in range [{start_utc}, {end_utc}]",
         verbose)
    return combined


# =============================================================================
# Core
# =============================================================================

def generate_weekly_levels(
    s0_dir: str,
    out_dir: str,
    asset: str,
    iso_year: int,
    iso_week: int,
    require_complete: bool = True,
    skip_existing: bool = True,
    verbose: bool = True,
) -> Optional[pd.DataFrame]:
    """
    Compute weekly levels for one asset-iso-week and write the parquet.

    Returns the DataFrame that was written, or None if skipped.
    """
    a = asset.lower()
    out_path = Path(out_dir) / f"weekly_{a}_{iso_year}_{iso_week:02d}.parquet"

    if skip_existing and out_path.exists():
        _log(f"Skip (exists): {out_path.name}", verbose)
        return None

    _log(f"Computing weekly levels: {asset} ISO {iso_year}-W{iso_week:02d}", verbose)
    t0 = time.time()

    # --- Compute week bounds ---
    week_start, week_end = _iso_week_bounds(iso_year, iso_week)
    _log(f"  Target week: {week_start} → {week_end}", verbose)

    # --- Load target-week S0 (required) ---
    target_df = _load_s0_range(
        s0_dir=s0_dir, asset=asset,
        start_utc=week_start, end_utc=week_end,
        require_complete=require_complete, verbose=verbose,
    )

    if target_df.empty:
        _log(f"  WARN: no data loaded for target week — writing empty parquet "
             f"(prev_week_* may still be filled if prior week exists)", verbose)

    # --- Prev-week bounds and load (optional) ---
    # Preceding ISO week (may cross year boundary — timedelta handles it).
    prev_start_day = date.fromisocalendar(iso_year, iso_week, 1) - timedelta(days=7)
    prev_iy, prev_iw, _ = prev_start_day.isocalendar()
    prev_start, prev_end = _iso_week_bounds(prev_iy, prev_iw)

    _log(f"  Prev week: ISO {prev_iy}-W{prev_iw:02d} "
         f"({prev_start} → {prev_end})", verbose)

    prev_df = _load_s0_range(
        s0_dir=s0_dir, asset=asset,
        start_utc=prev_start, end_utc=prev_end,
        # Prev week is always optional — if incomplete, we use whatever exists.
        require_complete=False, verbose=verbose,
    )

    # --- Compute prev_week_high/low (constants, broadcast to every row) ---
    if prev_df.empty or prev_df["mid_fut"].dropna().empty:
        prev_week_high = float("nan")
        prev_week_low  = float("nan")
        _log(f"  prev_week_high/low = NaN (no prev-week data)", verbose)
    else:
        prev_week_high = float(prev_df["mid_fut"].max())
        prev_week_low  = float(prev_df["mid_fut"].min())
        _log(f"  prev_week_high={prev_week_high:.2f} "
             f"prev_week_low={prev_week_low:.2f}", verbose)

    # --- Build 1s-grid skeleton for the full week ---
    # Generate every second from week_start..week_end (inclusive).
    grid = pd.date_range(
        start=pd.Timestamp(week_start), end=pd.Timestamp(week_end),
        freq="1s", tz="UTC",
    )
    out_df = pd.DataFrame({"bucket_dt_utc": grid})

    if not target_df.empty:
        # Merge the observed mid_fut onto the grid (left join; fill missing with ffill
        # for cumulative stats — if a second has no mid, the last observed value still
        # determines cumulative max/min).
        merged = out_df.merge(target_df, on="bucket_dt_utc", how="left")
        merged["mid_fut"] = merged["mid_fut"].astype("float64").ffill()
    else:
        merged = out_df.copy()
        merged["mid_fut"] = np.nan

    # --- week_open_fut: first non-NaN value, broadcast ---
    first_valid = merged["mid_fut"].dropna()
    week_open_val = float(first_valid.iloc[0]) if len(first_valid) > 0 else float("nan")
    merged["week_open_fut"] = week_open_val

    # --- Expanding max/min over the week (NaN-skipping) ---
    # pd.Series.cummax skips NaN after a valid start; ensure we seed with mid_fut.
    merged["week_high_fut"] = merged["mid_fut"].cummax()
    merged["week_low_fut"]  = merged["mid_fut"].cummin()

    # --- Monday levels: expanding max/min on Monday rows, ffill for rest of week ---
    # weekday(): Monday=0, ..., Sunday=6
    mon_mask = merged["bucket_dt_utc"].dt.weekday == 0
    monday_mid = merged["mid_fut"].where(mon_mask)
    monday_high_ex = monday_mid.cummax()
    monday_low_ex  = monday_mid.cummin()
    # For Tuesday-Sunday: freeze the last Monday value by forward-filling.
    merged["monday_high_fut"] = monday_high_ex.ffill()
    merged["monday_low_fut"]  = monday_low_ex.ffill()

    # --- Prev-week (broadcast constants) ---
    merged["prev_week_high_fut"] = prev_week_high
    merged["prev_week_low_fut"]  = prev_week_low

    # --- Drop working mid_fut column (not an output level) ---
    out_df = merged.drop(columns=["mid_fut"])

    # --- Write ---
    _atomic_write_parquet(out_df, out_path)
    mb = out_path.stat().st_size / (1024 * 1024)
    _log(f"  Saved: {out_path.name}  ({mb:.2f} MB, {len(out_df)} rows)  "
         f"in {time.time() - t0:.2f}s", verbose)

    return out_df


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate weekly-level parquet (ISO-week grid) from S0 features."
    )
    ap.add_argument("--s0-dir",  type=str, default=str(_DEFAULT_S0_DIR))
    ap.add_argument("--out-dir", type=str, default=str(_DEFAULT_OUT_DIR))
    ap.add_argument("--asset",   type=str, required=True, choices=["btc", "eth"])
    ap.add_argument("--iso-week", type=str, required=True,
                    help="ISO week to compute (format: YYYY-WNN, e.g. 2026-W10).")
    ap.add_argument("--skip-existing", action="store_true", default=True,
                    help="Skip if output already exists (default: True).")
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    ap.add_argument("--no-require-complete", action="store_true",
                    help="Allow missing S0 hours (not recommended for production).")
    ap.add_argument("--quiet", "-q", action="store_true")

    args = ap.parse_args()
    iso_year, iso_week = _parse_iso_week(args.iso_week)

    generate_weekly_levels(
        s0_dir=args.s0_dir,
        out_dir=args.out_dir,
        asset=args.asset,
        iso_year=iso_year,
        iso_week=iso_week,
        require_complete=not args.no_require_complete,
        skip_existing=args.skip_existing,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()