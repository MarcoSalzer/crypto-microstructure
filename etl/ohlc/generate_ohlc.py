# etl/ohlc/generate_ohlc.py
# ==============================================================================
# Running OHLC Generator -- Per-hour cumulative high/low/open from S0 features.
#
# PURPOSE:
#   Compute day-level price range context (running OHLC) from S0 hourly
#   parquets. Unlike the previous "complete-day" mode that produced a single
#   row per day after all 24 hours were available, this generator produces
#   one row per second, with running cummax/cummin/open semantics:
#
#     day_high_spot[t] = max(mid_spot from 00:00 UTC of the day to t)
#     day_low_spot[t]  = min(mid_spot from 00:00 UTC of the day to t)
#     day_open_spot[t] = first valid mid_spot of the UTC day (constant for
#                        the rest of the day)
#
#   This makes OHLC features available at any tick, not just at end-of-day,
#   and is consistent with the hot-path which maintains running state.
#
#   day_close has been REMOVED -- it was redundant with mid_spot[t] during
#   the day, and only became meaningful at 23:59:59 UTC.
#
# ARCHITECTURE:
#   For each (asset, date, hour) combination this generator:
#     1. Loads previous-hour state file if hour > 0 (the cumulative
#        high/low/open at the end of hour-1 of the same date).
#        If hour == 0: starts fresh with day_open = first valid mid.
#     2. Reads s0_features_{asset}_{date}_{hour:02d}.parquet to obtain the
#        per-second mid_spot_1s and mid_fut_1s columns.
#     3. Computes per-second running high/low/open using vectorized
#        cummax/cummin/ffill, seeded with previous state.
#     4. Writes two output files:
#         - ohlc_running_{asset}_{date}_{hour:02d}.parquet
#               Schema: bucket_dt_utc, day_high/low/open_{spot,fut}
#               One row per second. Joined into S1 by bucket_dt_utc.
#         - ohlc_state_{asset}_{date}_{hour:02d}.parquet
#               Schema: 1 row with the final cumulative high/low/open at
#               the end of the hour. Used as init-state for the next hour.
#
# IDEMPOTENCY:
#   If both output files already exist for a given (asset, date, hour),
#   the run is skipped.
#
# ==============================================================================

from __future__ import annotations
from common.paths import DATA_ROOT

import argparse
import os
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PARQUET_COMPRESSION = "zstd"

_SCRIPT_DIR      = Path(__file__).resolve().parent
_DEFAULT_S0_DIR  = DATA_ROOT / "s0_features"
_DEFAULT_OUT_DIR = DATA_ROOT / "ohlc"

# S0 column names for mid price - first found is used
_MID_SPOT_CANDIDATES = ["mid_spot_1s", "mid_spot"]
_MID_FUT_CANDIDATES  = ["mid_fut_1s",  "mid_fut"]

# Output column names produced by this generator
RUNNING_OHLC_COLS = [
    "day_high_spot", "day_low_spot", "day_open_spot",
    "day_high_fut",  "day_low_fut",  "day_open_fut",
]


# =============================================================================
# Utilities
# =============================================================================

def _log(msg: str, verbose: bool = True) -> None:
    if verbose:
        print(f"[{pd.Timestamp.utcnow().strftime('%H:%M:%S')}] [OHLC] {msg}")


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


# =============================================================================
# Path Helpers
# =============================================================================

def _running_path(out_dir: Path, asset: str, date_str: str, hour: int) -> Path:
    return out_dir / f"ohlc_running_{asset.lower()}_{date_str}_{hour:02d}.parquet"


def _state_path(out_dir: Path, asset: str, date_str: str, hour: int) -> Path:
    return out_dir / f"ohlc_state_{asset.lower()}_{date_str}_{hour:02d}.parquet"


def _s0_path(s0_dir: Path, asset: str, date_str: str, hour: int) -> Path:
    return s0_dir / f"s0_features_{asset.lower()}_{date_str}_{hour:02d}.parquet"


# =============================================================================
# State load / save
# =============================================================================

def _load_prev_state(
    out_dir: Path,
    asset: str,
    date_str: str,
    hour: int,
    verbose: bool,
) -> Optional[dict]:
    """
    Load running-OHLC state from the previous hour of the SAME UTC day.

    Returns None if:
      - hour == 0 (fresh start of UTC day; no prev state by design)
      - prev state file missing (caller treats as gap; running starts from
        current hour values)
    """
    if hour == 0:
        return None

    prev_path = _state_path(out_dir, asset, date_str, hour - 1)
    if not prev_path.exists():
        _log(f"  Prev-hour state missing ({prev_path.name}) -- starting "
             f"from current values", verbose)
        return None

    try:
        df = pq.read_table(str(prev_path)).to_pandas()
        if df.empty:
            return None
        return df.iloc[0].to_dict()
    except Exception as e:
        _log(f"  WARN: failed to load prev state {prev_path.name}: {e}", verbose)
        return None


def _save_final_state(
    out_dir: Path,
    asset: str,
    date_str: str,
    hour: int,
    final_state: dict,
    verbose: bool,
) -> None:
    """Save the final running-OHLC state for this hour (1 row)."""
    state_df = pd.DataFrame([final_state])
    out_path = _state_path(out_dir, asset, date_str, hour)
    _atomic_write_parquet(state_df, out_path)
    _log(f"  Saved state: {out_path.name}", verbose)


# =============================================================================
# Core: build running OHLC for one hour
# =============================================================================

def build_ohlc_running_hour(
    s0_dir: str,
    out_dir: str,
    asset: str,
    date_str: str,
    hour: int,
    skip_existing: bool = True,
    verbose: bool = True,
) -> Optional[pd.DataFrame]:
    """
    Compute running OHLC for one (asset, date, hour) combination.

    Returns the running DataFrame (or None if skipped).
    """
    a = asset.lower()
    s0_dir_p  = Path(s0_dir)
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    running_path = _running_path(out_dir_p, a, date_str, hour)
    state_path   = _state_path(out_dir_p,   a, date_str, hour)

    if skip_existing and running_path.exists() and state_path.exists():
        _log(f"Skip (exists): {running_path.name}", verbose)
        return None

    s0_path = _s0_path(s0_dir_p, a, date_str, hour)
    if not s0_path.exists():
        raise FileNotFoundError(f"Missing S0 file: {s0_path}")

    _log(f"Building running OHLC: {a} {date_str} hour={hour:02d}", verbose)
    t0 = time.time()

    # ------------------------------------------------------------------
    # Load S0 file
    # ------------------------------------------------------------------
    df = pq.read_table(str(s0_path)).to_pandas()
    if "bucket_dt_utc" not in df.columns:
        raise ValueError(f"S0 file missing bucket_dt_utc column: {s0_path}")

    df = df.sort_values("bucket_dt_utc").reset_index(drop=True)
    df["bucket_dt_utc"] = pd.to_datetime(df["bucket_dt_utc"], utc=True)

    mid_spot_col = _find_col(df, _MID_SPOT_CANDIDATES)
    mid_fut_col  = _find_col(df, _MID_FUT_CANDIDATES)
    if mid_spot_col is None or mid_fut_col is None:
        raise ValueError(
            f"S0 file missing mid_spot/mid_fut columns: {s0_path}\n"
            f"  spot candidates: {_MID_SPOT_CANDIDATES}\n"
            f"  fut  candidates: {_MID_FUT_CANDIDATES}"
        )

    mid_spot = df[mid_spot_col].astype("float64").reset_index(drop=True)
    mid_fut  = df[mid_fut_col].astype("float64").reset_index(drop=True)

    # ------------------------------------------------------------------
    # Load previous-hour state (if any)
    # ------------------------------------------------------------------
    prev = _load_prev_state(out_dir_p, a, date_str, hour, verbose)

    if prev is None:
        prev_high_spot = float("nan")
        prev_low_spot  = float("nan")
        prev_open_spot = float("nan")
        prev_high_fut  = float("nan")
        prev_low_fut   = float("nan")
        prev_open_fut  = float("nan")
    else:
        prev_high_spot = float(prev.get("day_high_spot", float("nan")))
        prev_low_spot  = float(prev.get("day_low_spot",  float("nan")))
        prev_open_spot = float(prev.get("day_open_spot", float("nan")))
        prev_high_fut  = float(prev.get("day_high_fut",  float("nan")))
        prev_low_fut   = float(prev.get("day_low_fut",   float("nan")))
        prev_open_fut  = float(prev.get("day_open_fut",  float("nan")))

    # ------------------------------------------------------------------
    # Compute running high/low.
    # cummax/cummin in pandas skip NaN by default, which is exactly what
    # we want: a NaN tick (missing data) does not lower or raise the
    # running high/low.
    # ------------------------------------------------------------------

    def _running_max(series: pd.Series, prev_val: float) -> pd.Series:
        """
        Running max from start of UTC day. Seeded with prev_val (cumulative
        max of all earlier hours of this day). NaN values pass through
        without affecting the running max.

        cummax with skipna=True (default) leaves NaN-positions as NaN even
        though it carries the running max forward internally. We need to
        ffill those NaN positions so the running max is visible at every
        tick. Finally, we elementwise-max with prev_val so the running
        cumulative high never drops below earlier hours of the day.
        """
        running = series.cummax().ffill()
        if not np.isnan(prev_val):
            # Where running is still NaN (entire prefix had no valid value)
            # use prev_val. Then floor at prev_val everywhere.
            running = running.fillna(prev_val)
            running = pd.Series(np.maximum(running.values, prev_val),
                                index=series.index)
        return running

    def _running_min(series: pd.Series, prev_val: float) -> pd.Series:
        running = series.cummin().ffill()
        if not np.isnan(prev_val):
            running = running.fillna(prev_val)
            running = pd.Series(np.minimum(running.values, prev_val),
                                index=series.index)
        return running

    def _running_open(series: pd.Series, prev_val: float) -> pd.Series:
        """
        Day-open: first non-NaN value of the UTC day, then constant.
        If prev_val is known, propagate it. Otherwise pick the first
        non-NaN value in this hour and broadcast forward.
        """
        n = len(series)
        if not np.isnan(prev_val):
            # Day-open already established earlier in the day
            return pd.Series([prev_val] * n, index=series.index)

        first_valid_pos = series.first_valid_index()
        if first_valid_pos is None:
            # No data in this hour and no prev state -- all NaN
            return pd.Series([float("nan")] * n, index=series.index)

        first_val = float(series.loc[first_valid_pos])
        # Find positional index of first valid value
        position = series.index.get_loc(first_valid_pos)

        out = np.full(n, np.nan, dtype="float64")
        out[position:] = first_val
        return pd.Series(out, index=series.index)

    high_spot = _running_max(mid_spot, prev_high_spot)
    low_spot  = _running_min(mid_spot, prev_low_spot)
    open_spot = _running_open(mid_spot, prev_open_spot)

    high_fut  = _running_max(mid_fut, prev_high_fut)
    low_fut   = _running_min(mid_fut, prev_low_fut)
    open_fut  = _running_open(mid_fut, prev_open_fut)

    # ------------------------------------------------------------------
    # Build output dataframe (per-second)
    # ------------------------------------------------------------------
    out_df = pd.DataFrame({
        "bucket_dt_utc": df["bucket_dt_utc"].values,
        "day_high_spot": high_spot.values,
        "day_low_spot":  low_spot.values,
        "day_open_spot": open_spot.values,
        "day_high_fut":  high_fut.values,
        "day_low_fut":   low_fut.values,
        "day_open_fut":  open_fut.values,
    })

    if len(out_df) == 0:
        raise ValueError(f"S0 file produced 0 rows of OHLC: {s0_path}")

    last = out_df.iloc[-1]
    final_state = {
        "date_str":      date_str,
        "asset":         a,
        "hour":          int(hour),
        "day_high_spot": float(last["day_high_spot"]),
        "day_low_spot":  float(last["day_low_spot"]),
        "day_open_spot": float(last["day_open_spot"]),
        "day_high_fut":  float(last["day_high_fut"]),
        "day_low_fut":   float(last["day_low_fut"]),
        "day_open_fut":  float(last["day_open_fut"]),
    }

    _atomic_write_parquet(out_df, running_path)
    _save_final_state(out_dir_p, a, date_str, hour, final_state, verbose)

    elapsed = time.time() - t0
    spot_lo = final_state['day_low_spot']
    spot_hi = final_state['day_high_spot']
    spot_op = final_state['day_open_spot']
    _log(
        f"  Saved: {running_path.name}  ({len(out_df)} rows, {elapsed:.1f}s)  "
        f"spot=[{spot_lo:.1f}, {spot_hi:.1f}]  open={spot_op:.1f}",
        verbose,
    )

    return out_df


# =============================================================================
# Per-day driver: loop h=0..23
# =============================================================================

def generate_ohlc_for_day(
    s0_dir: str,
    out_dir: str,
    asset: str,
    date_str: str,
    skip_existing: bool = True,
    partial: bool = False,
    verbose: bool = True,
    # Backwards-compat: compat callers (e.g. run_ohlc.py) pass these.
    # Both are now obsolete in running-OHLC mode but we accept them
    # silently so existing wrappers don't break.
    require_complete: Optional[bool] = None,
    **kwargs,
) -> int:
    """
    Build running OHLC for all 24 hours of one (asset, date) sequentially.

    [RUNNING-OHLC 2026-04-26] In running-OHLC mode the old "require_complete"
    semantic (build only after all 24 hours arrived) no longer applies. We
    instead build hour-by-hour with state passing. The require_complete
    parameter is accepted for backwards compatibility but ignored; use
    `partial=True` if you want to allow missing intermediate hours.
    """
    if require_complete is False:
        # Caller explicitly asked for partial mode (e.g. current streaming day)
        partial = True

    if kwargs and verbose:
        _log(f"  Note: ignoring unexpected kwargs: {list(kwargs.keys())}", verbose)

    n_built = 0
    n_skipped = 0
    n_missing = 0

    for h in range(24):
        s0_p = _s0_path(Path(s0_dir), asset, date_str, h)
        if not s0_p.exists():
            if partial:
                _log(f"Hour {h:02d}: S0 missing -- skip (partial mode)", verbose)
                n_missing += 1
                continue
            else:
                raise FileNotFoundError(
                    f"Missing S0 file for {asset} {date_str} hour {h:02d}: {s0_p}\n"
                    f"Pass --partial to allow missing hours."
                )
        try:
            res = build_ohlc_running_hour(
                s0_dir=s0_dir,
                out_dir=out_dir,
                asset=asset,
                date_str=date_str,
                hour=h,
                skip_existing=skip_existing,
                verbose=verbose,
            )
            if res is None:
                n_skipped += 1
            else:
                n_built += 1
        except Exception as e:
            _log(f"Hour {h:02d}: ERROR -- {e}", verbose)
            raise

    _log(
        f"Done {asset} {date_str}: built={n_built}, skipped={n_skipped}, "
        f"missing={n_missing}",
        verbose,
    )
    return n_built


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate running per-hour OHLC parquet from S0 feature files."
    )
    ap.add_argument("--s0-dir",  type=str, default=str(_DEFAULT_S0_DIR))
    ap.add_argument("--out-dir", type=str, default=str(_DEFAULT_OUT_DIR))
    ap.add_argument("--asset",   type=str, required=True, choices=["btc", "eth", "bnb"])
    ap.add_argument("--date",    type=str, required=True,
                    help="Date to compute running OHLC for (YYYY-MM-DD).")
    ap.add_argument("--hour",    type=int, default=None,
                    help="If set, build only this single hour (0-23). "
                         "Otherwise loop h=0..23.")
    ap.add_argument("--skip-existing", action="store_true", default=True,
                    help="Skip hours whose output already exists (default: True).")
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    ap.add_argument("--partial", action="store_true", default=False,
                    help="Allow missing S0 hours (current streaming day).")
    ap.add_argument("--quiet", "-q", action="store_true")

    args = ap.parse_args()

    if args.hour is not None:
        build_ohlc_running_hour(
            s0_dir=args.s0_dir,
            out_dir=args.out_dir,
            asset=args.asset,
            date_str=args.date,
            hour=args.hour,
            skip_existing=args.skip_existing,
            verbose=not args.quiet,
        )
    else:
        generate_ohlc_for_day(
            s0_dir=args.s0_dir,
            out_dir=args.out_dir,
            asset=args.asset,
            date_str=args.date,
            skip_existing=args.skip_existing,
            partial=args.partial,
            verbose=not args.quiet,
        )


if __name__ == "__main__":
    main()