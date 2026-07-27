# etl/ohlc/generate_monthly_levels.py
# ==============================================================================
# Monthly Levels Generator — Calendar-month high/low/open on a 1-second grid,
# with prev-month extrema, from S0 feature parquets.
#
# PURPOSE:
#   Compute monthly session-level context that the S1 engine joins onto 1s
#   buckets, enabling features like dist_to_month_high_bps_fut,
#   range_pos_month_fut, month_range_bps_fut, dist_to_fib_382_month_bps_fut.
#
#   Like weekly levels (and unlike daily OHLC), monthly levels evolve
#   WITHIN the month (expanding max/min since month start), so this artefact
#   is stored on the full 1s timeline of the month.
#
# INPUT:
#   All hourly S0 parquet files for the target calendar month + preceding
#   calendar month (for prev_month_* values). Located at:
#     data_storage/s0_features/s0_features_{asset}_{YYYY-MM-DD}_{HH}.parquet
#
#   Calendar month: 1st day 00:00:00 UTC through last day 23:59:59 UTC.
#
# OUTPUT:
#   data_storage/monthly/monthly_{asset}_{year}_{month:02d}.parquet
#
#   Schema (one row per 1s bucket of the target month — ~2.5-2.7M rows):
#     bucket_dt_utc         datetime64[ns, UTC]  — 1s-grid key
#     month_open_fut        float — mid_fut at first second of the month (constant)
#     month_high_fut        float — expanding max(mid_fut) since month start
#     month_low_fut         float — expanding min(mid_fut) since month start
#     prev_month_high_fut   float — max(mid_fut) over the PRECEDING calendar
#                                   month (constant for the full target month)
#     prev_month_low_fut    float — min(mid_fut) over the preceding month
#
# COMPLETENESS GUARD:
#   By default, requires all hourly S0 parquets for the target month to be
#   present. Missing hours can be tolerated with --no-require-complete.
#   The preceding month is optional — if unavailable, prev_month_* = NaN.
#
# IDEMPOTENCY:
#   If the output file already exists, skipped by default (--skip-existing).
#   The writer is atomic (tmp + os.replace).
#
# USAGE:
#   python generate_monthly_levels.py --asset btc --year 2026 --month 3
#   python generate_monthly_levels.py --asset eth --year 2026 --month 3 \
#          --s0-dir /custom/s0 --out-dir /custom/monthly
#
# ==============================================================================

from __future__ import annotations
from common.paths import DATA_ROOT

import argparse
import calendar
import os
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PARQUET_COMPRESSION = "zstd"

_SCRIPT_DIR   = Path(__file__).resolve().parent
_DEFAULT_S0_DIR  = DATA_ROOT / "s0_features"
_DEFAULT_OUT_DIR = DATA_ROOT / "monthly"

_MID_FUT_CANDIDATES = ["mid_fut_1s", "mid_fut"]


# =============================================================================
# Utilities
# =============================================================================

def _log(msg: str, verbose: bool = True) -> None:
    if verbose:
        print(f"[{pd.Timestamp.utcnow().strftime('%H:%M:%S')}] [MONTHLY] {msg}")


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


def _month_bounds(year: int, month: int) -> Tuple[datetime, datetime]:
    """Return (start_utc, end_utc) of the calendar month (inclusive-inclusive)."""
    _, last_day = calendar.monthrange(year, month)
    start = datetime(year, month, 1, 0, 0, 0, tzinfo=timezone.utc)
    end   = datetime(year, month, last_day, 23, 59, 59, tzinfo=timezone.utc)
    return start, end


def _prev_month(year: int, month: int) -> Tuple[int, int]:
    """Return (prev_year, prev_month) tuple."""
    if month == 1:
        return year - 1, 12
    return year, month - 1


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
    and return a single concatenated DataFrame sliced to that range.

    Columns retained: bucket_dt_utc, mid_fut (auto-detected from candidates).
    Raises FileNotFoundError if require_complete=True and hours are missing.
    """
    a = asset.lower()
    s0_path = Path(s0_dir)

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
            f"Monthly levels require all hourly S0 parquets in "
            f"[{start_utc.isoformat()}, {end_utc.isoformat()}]. "
            f"Missing {len(missing)} hours, sample: {miss_sample}"
        )
    if missing:
        _log(f"  WARN: {len(missing)} missing S0 hours (continuing)", verbose)

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
                _log(f"  WARN: skipping {fp.name} — no bucket_dt_utc column", verbose)
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

def generate_monthly_levels(
    s0_dir: str,
    out_dir: str,
    asset: str,
    year: int,
    month: int,
    require_complete: bool = True,
    skip_existing: bool = True,
    verbose: bool = True,
) -> Optional[pd.DataFrame]:
    """
    Compute monthly levels for one asset-(year, month) and write the parquet.

    Returns the DataFrame that was written, or None if skipped.
    """
    if not (1 <= month <= 12):
        raise ValueError(f"month out of range: {month}")

    a = asset.lower()
    out_path = Path(out_dir) / f"monthly_{a}_{year}_{month:02d}.parquet"

    if skip_existing and out_path.exists():
        _log(f"Skip (exists): {out_path.name}", verbose)
        return None

    _log(f"Computing monthly levels: {asset} {year}-{month:02d}", verbose)
    t0 = time.time()

    # --- Month bounds ---
    month_start, month_end = _month_bounds(year, month)
    _log(f"  Target month: {month_start} → {month_end}", verbose)

    # --- Load target-month S0 ---
    target_df = _load_s0_range(
        s0_dir=s0_dir, asset=asset,
        start_utc=month_start, end_utc=month_end,
        require_complete=require_complete, verbose=verbose,
    )

    if target_df.empty:
        _log(f"  WARN: no data loaded for target month — writing empty parquet "
             f"(prev_month_* may still be filled if prior month exists)", verbose)

    # --- Prev-month bounds and load ---
    prev_year, prev_month = _prev_month(year, month)
    prev_start, prev_end = _month_bounds(prev_year, prev_month)
    _log(f"  Prev month: {prev_year}-{prev_month:02d} "
         f"({prev_start} → {prev_end})", verbose)

    prev_df = _load_s0_range(
        s0_dir=s0_dir, asset=asset,
        start_utc=prev_start, end_utc=prev_end,
        require_complete=False, verbose=verbose,
    )

    # --- Prev-month extrema (constants) ---
    if prev_df.empty or prev_df["mid_fut"].dropna().empty:
        prev_month_high = float("nan")
        prev_month_low  = float("nan")
        _log(f"  prev_month_high/low = NaN (no prev-month data)", verbose)
    else:
        prev_month_high = float(prev_df["mid_fut"].max())
        prev_month_low  = float(prev_df["mid_fut"].min())
        _log(f"  prev_month_high={prev_month_high:.2f} "
             f"prev_month_low={prev_month_low:.2f}", verbose)

    # --- Build 1s-grid skeleton for the target month ---
    grid = pd.date_range(
        start=pd.Timestamp(month_start), end=pd.Timestamp(month_end),
        freq="1s", tz="UTC",
    )
    out_df = pd.DataFrame({"bucket_dt_utc": grid})

    if not target_df.empty:
        merged = out_df.merge(target_df, on="bucket_dt_utc", how="left")
        merged["mid_fut"] = merged["mid_fut"].astype("float64").ffill()
    else:
        merged = out_df.copy()
        merged["mid_fut"] = np.nan

    # --- month_open: first non-NaN mid, broadcast ---
    first_valid = merged["mid_fut"].dropna()
    month_open_val = float(first_valid.iloc[0]) if len(first_valid) > 0 else float("nan")
    merged["month_open_fut"] = month_open_val

    # --- Expanding max/min over the month ---
    merged["month_high_fut"] = merged["mid_fut"].cummax()
    merged["month_low_fut"]  = merged["mid_fut"].cummin()

    # --- Prev-month (broadcast constants) ---
    merged["prev_month_high_fut"] = prev_month_high
    merged["prev_month_low_fut"]  = prev_month_low

    # --- Drop working mid_fut column ---
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
        description="Generate monthly-level parquet (calendar-month grid) from S0 features."
    )
    ap.add_argument("--s0-dir",  type=str, default=str(_DEFAULT_S0_DIR))
    ap.add_argument("--out-dir", type=str, default=str(_DEFAULT_OUT_DIR))
    ap.add_argument("--asset",   type=str, required=True, choices=["btc", "eth"])
    ap.add_argument("--year",    type=int, required=True,
                    help="Calendar year (e.g. 2026).")
    ap.add_argument("--month",   type=int, required=True,
                    help="Calendar month (1-12).")
    ap.add_argument("--skip-existing", action="store_true", default=True,
                    help="Skip if output already exists (default: True).")
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    ap.add_argument("--no-require-complete", action="store_true",
                    help="Allow missing S0 hours (not recommended for production).")
    ap.add_argument("--quiet", "-q", action="store_true")

    args = ap.parse_args()

    generate_monthly_levels(
        s0_dir=args.s0_dir,
        out_dir=args.out_dir,
        asset=args.asset,
        year=args.year,
        month=args.month,
        require_complete=not args.no_require_complete,
        skip_existing=args.skip_existing,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()