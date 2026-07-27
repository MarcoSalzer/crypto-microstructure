# etl/ohlc/generate_volume_profile.py
# ==============================================================================
# Volume Profile Generator — POC / VAH / VAL levels + migration on a
# 1-second grid, for three rolling windows (60m, 240m, 1d).
#
# PURPOSE:
#   Compute a volume-profile artefact that the S1 engine joins onto 1s
#   buckets. Enables features like dist_to_poc_60m_bps_fut, price_vs_va_1d_fut,
#   and includes PASSTHROUGH-style poc_migration_*_bps_fut columns (the
#   migration signals are too expensive to compute on the 1s grid in the
#   hot path, so we pre-compute them here).
#
# INPUT:
#   All hourly S0 parquet files for the target day and the preceding
#   `--lookback-dates` day(s), located at:
#     data_storage/s0_features/s0_features_{asset}_{YYYY-MM-DD}_{HH}.parquet
#
#   Required S0 columns:
#     bucket_dt_utc      datetime64[ns, UTC]  — 1s-grid key
#     volume_fut_1s      float — per-bucket traded volume (base units)
#     vwap_fut_1s        float — per-bucket volume-weighted price
#                                (fallback: mid_fut_1s if vwap column absent)
#
#   Lookback defaults to 1 day so that the 1d (86400s) rolling window is
#   fully warm at the start of the target day.
#
# OUTPUT:
#   data_storage/vp/vp_{asset}_{date}.parquet
#
#   Schema (one row per 1s bucket of the target day — 86400 rows):
#     bucket_dt_utc                datetime64[ns, UTC]
#
#     # POC (price bin with highest volume) per window
#     poc_60m_fut                  float
#     poc_240m_fut                 float
#     poc_1d_fut                   float
#
#     # Value-Area High (upper 70%-volume boundary) per window
#     vah_60m_fut                  float
#     vah_240m_fut                 float
#     vah_1d_fut                   float
#
#     # Value-Area Low
#     val_60m_fut                  float
#     val_240m_fut                 float
#     val_1d_fut                   float
#
#     # POC Migration (as additional passthrough columns — precomputed
#     # because the hot-path buffer cannot hold 12h of poc_1d_fut history).
#     # Formula: (poc[t] - poc[t-shift]) / poc[t-shift] * 10000
#     poc_migration_60m_bps_fut    float (shift_s=1800)
#     poc_migration_240m_bps_fut   float (shift_s=7200)
#     poc_migration_1d_bps_fut     float (shift_s=43200)
#
# RE-COMPUTE FREQUENCY:
#   To avoid O(N * window_size * bins) cost at 1s resolution, POC/VA are
#   recomputed at fixed intervals and forward-filled between updates:
#     60m window:  every 60 s  (1/60 of the window length)
#     240m window: every 300 s
#     1d window:   every 900 s
#   This keeps cost at roughly O(N / step * bins) per window.
#
# VALUE-AREA ALGORITHM:
#   Standard: starting at the POC bin, greedily expand to the side
#   (up or down) with larger neighbouring volume until cumulative volume
#   >= 70% of the total. VAH = upper edge of the highest included bin,
#   VAL = lower edge of the lowest.
#
# COMPLETENESS GUARD:
#   By default, all hourly S0 parquets for the target day must be present.
#   Previous-day hours can be missing (the 1d window just has less history
#   at the start of the day and values may be NaN until warmup).
#
# IDEMPOTENCY:
#   Output is skipped if it already exists (--skip-existing, default True).
#   Atomic write via tmp file + os.replace.
#
# UNIT TESTS:
#   Run with `python generate_volume_profile.py --test` to exercise the
#   compute_vp_window() helper on synthetic distributions.
#
# USAGE:
#   python generate_volume_profile.py --asset btc --date 2026-03-10
#   python generate_volume_profile.py --asset eth --date 2026-03-10 \
#          --lookback-dates 1 --s0-dir /custom/s0 --out-dir /custom/vp
#   python generate_volume_profile.py --test        # run unit tests
#
# ==============================================================================

from __future__ import annotations
from common.paths import DATA_ROOT

import argparse
import os
import sys
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
_DEFAULT_OUT_DIR = DATA_ROOT / "vp"

_PRICE_CANDIDATES  = ["vwap_fut_1s", "mid_fut_1s", "mid_fut"]
_VOLUME_CANDIDATES = ["volume_fut_1s", "volume_fut"]

# Window configuration: (label, window_seconds, recompute_interval_seconds, poc_migration_shift_seconds)
_WINDOW_CONFIGS = [
    ("60m",  3600,   60,   1800),
    ("240m", 14400,  300,  7200),
    ("1d",   86400,  900,  43200),
]

_N_BINS_DEFAULT = 50
_VA_COVERAGE    = 0.70   # Value-Area = 70% of volume


# =============================================================================
# Utilities
# =============================================================================

def _log(msg: str, verbose: bool = True) -> None:
    if verbose:
        print(f"[{pd.Timestamp.utcnow().strftime('%H:%M:%S')}] [VP] {msg}")


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
# VP Computation (pure — used in unit tests)
# =============================================================================

def compute_vp_window(
    prices: np.ndarray,
    volumes: np.ndarray,
    n_bins: int = _N_BINS_DEFAULT,
    va_coverage: float = _VA_COVERAGE,
) -> Tuple[float, float, float]:
    """
    Compute (POC, VAH, VAL) for a price/volume window.

    Args:
        prices:       1D array of prices (e.g. per-second vwap).
        volumes:      1D array of corresponding volumes (same length as prices).
        n_bins:       Number of price bins for the histogram.
        va_coverage:  Target cumulative volume fraction for the value area
                      (standard convention: 0.70).

    Returns:
        (poc, vah, val) as floats. Returns (NaN, NaN, NaN) if the input is
        empty or carries zero total volume. If prices are constant (no spread),
        all three values equal that single price.

    Algorithm:
      1. Histogram of volume against n_bins equally-spaced price bins.
      2. POC = midpoint of the bin with the highest total volume.
      3. Value area: start at the POC bin. Expand by adding the
         neighbouring bin (above or below) with the larger volume,
         until cumulative volume >= va_coverage * total_volume.
      4. VAH = upper edge of the highest included bin.
      5. VAL = lower edge of the lowest included bin.
    """
    if prices.size == 0 or volumes.size == 0:
        return float("nan"), float("nan"), float("nan")

    # Drop any rows with NaN price/volume or non-positive volume.
    mask = np.isfinite(prices) & np.isfinite(volumes) & (volumes > 0)
    if not np.any(mask):
        return float("nan"), float("nan"), float("nan")
    prices  = prices[mask]
    volumes = volumes[mask]

    total_vol = float(volumes.sum())
    if total_vol <= 0:
        return float("nan"), float("nan"), float("nan")

    p_min = float(prices.min())
    p_max = float(prices.max())

    # Degenerate: all trades at one price
    if p_max == p_min:
        return p_min, p_max, p_min

    edges = np.linspace(p_min, p_max, n_bins + 1)
    hist, _ = np.histogram(prices, bins=edges, weights=volumes)

    # POC = center of the dominant bin
    poc_idx = int(np.argmax(hist))
    poc = float((edges[poc_idx] + edges[poc_idx + 1]) / 2.0)

    # Value-area expansion around POC
    target = va_coverage * total_vol
    covered = float(hist[poc_idx])
    lo_idx = hi_idx = poc_idx

    while covered < target and (lo_idx > 0 or hi_idx < len(hist) - 1):
        next_lo = float(hist[lo_idx - 1]) if lo_idx > 0 else -1.0
        next_hi = float(hist[hi_idx + 1]) if hi_idx < len(hist) - 1 else -1.0
        # Prefer the larger side; ties break toward the high side (arbitrary but deterministic).
        if next_hi >= next_lo:
            hi_idx += 1
            covered += float(hist[hi_idx])
        else:
            lo_idx -= 1
            covered += float(hist[lo_idx])

    vah = float(edges[hi_idx + 1])
    val = float(edges[lo_idx])
    return poc, vah, val


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
    Load S0 hourly parquets overlapping [start_utc, end_utc] inclusive.
    Returns columns: bucket_dt_utc, price, volume.
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
            f"VP generator requires S0 parquets in "
            f"[{start_utc.isoformat()}, {end_utc.isoformat()}]. "
            f"Missing {len(missing)} hours, sample: {miss_sample}"
        )
    if missing:
        _log(f"  WARN: {len(missing)} missing S0 hours (continuing)", verbose)

    _log(f"  Loading {len(present)} S0 hourly parquets", verbose)

    frames = []
    price_col = None
    vol_col   = None
    for fp in sorted(present):
        try:
            df = pq.read_table(str(fp), columns=None).to_pandas()
            if price_col is None:
                price_col = _find_col(df, _PRICE_CANDIDATES)
                if price_col is None:
                    raise ValueError(
                        f"No price column in {fp.name}. Tried: {_PRICE_CANDIDATES}"
                    )
            if vol_col is None:
                vol_col = _find_col(df, _VOLUME_CANDIDATES)
                if vol_col is None:
                    raise ValueError(
                        f"No volume column in {fp.name}. Tried: {_VOLUME_CANDIDATES}"
                    )
                _log(f"  Using price={price_col!r}  volume={vol_col!r}", verbose)
            if "bucket_dt_utc" not in df.columns:
                _log(f"  WARN: skipping {fp.name} — no bucket_dt_utc column",
                     verbose)
                continue
            frames.append(df[["bucket_dt_utc", price_col, vol_col]].rename(
                columns={price_col: "price", vol_col: "volume"}
            ))
        except Exception as e:
            _log(f"  WARN: failed to load {fp.name}: {e}", verbose)

    if not frames:
        return pd.DataFrame(columns=["bucket_dt_utc", "price", "volume"])

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

def _compute_vp_column_triple(
    data: pd.DataFrame,
    grid: pd.DatetimeIndex,
    window_s: int,
    recompute_every_s: int,
    n_bins: int,
    verbose: bool = True,
    label: str = "",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For every second in `grid`, produce (poc, vah, val) values.

    To keep cost manageable, POC/VA are recomputed every `recompute_every_s`
    seconds on the window `[grid_time - window_s, grid_time]` and forward-
    filled on the intervening seconds.

    Args:
        data:               DataFrame with columns bucket_dt_utc, price, volume.
                            Must cover at least [grid[0] - window_s, grid[-1]].
        grid:               1s-step timestamps for which to produce output.
        window_s:           Look-back window size in seconds.
        recompute_every_s:  Recompute stride in seconds.
        n_bins:             Histogram bins.
        label:              Logging label (e.g. "60m").

    Returns:
        (poc_arr, vah_arr, val_arr) — np.ndarray, same length as grid.
    """
    n = len(grid)
    poc_arr = np.full(n, np.nan, dtype=np.float64)
    vah_arr = np.full(n, np.nan, dtype=np.float64)
    val_arr = np.full(n, np.nan, dtype=np.float64)

    if data.empty:
        return poc_arr, vah_arr, val_arr

    # For efficient look-ups we build an ndarray of (ts_ns, price, vol).
    ts_ns = data["bucket_dt_utc"].astype("int64").to_numpy()
    prices  = data["price"].to_numpy(dtype=np.float64)
    volumes = data["volume"].to_numpy(dtype=np.float64)

    window_ns = np.int64(window_s) * np.int64(1_000_000_000)

    last_poc = np.nan
    last_vah = np.nan
    last_val = np.nan

    # Progress logging
    report_every = max(1, n // 20)

    grid_ns = grid.astype("int64").to_numpy()

    for i in range(n):
        t_ns = int(grid_ns[i])
        # Decide if we recompute at this tick.
        # Align recompute to the grid-start: recompute at i = 0, step, 2*step, ...
        if i % recompute_every_s == 0:
            start_ns = t_ns - int(window_ns)
            # searchsorted: left-inclusive, right-exclusive
            lo = int(np.searchsorted(ts_ns, start_ns, side="left"))
            hi = int(np.searchsorted(ts_ns, t_ns,    side="right"))
            if hi - lo > 0:
                last_poc, last_vah, last_val = compute_vp_window(
                    prices[lo:hi], volumes[lo:hi], n_bins=n_bins
                )
            else:
                last_poc = last_vah = last_val = np.nan

            if verbose and (i % report_every == 0):
                _log(f"  [{label}] tick {i}/{n} — poc={last_poc}", True)

        poc_arr[i] = last_poc
        vah_arr[i] = last_vah
        val_arr[i] = last_val

    return poc_arr, vah_arr, val_arr


def _poc_migration_bps(poc_arr: np.ndarray, shift_s: int) -> np.ndarray:
    """
    (poc[t] - poc[t - shift]) / poc[t - shift] * 10000.

    For t < shift_s, the lagged value is NaN → migration is NaN. Uses numpy
    shift so we don't depend on pd.Series indices.
    """
    n = len(poc_arr)
    out = np.full(n, np.nan, dtype=np.float64)
    if shift_s >= n:
        return out
    lagged = np.empty(n, dtype=np.float64)
    lagged[:shift_s] = np.nan
    lagged[shift_s:] = poc_arr[:-shift_s]

    with np.errstate(divide="ignore", invalid="ignore"):
        delta = poc_arr - lagged
        denom = np.where(np.abs(lagged) > 1e-12, lagged, np.nan)
        out = (delta / denom) * 10000.0
    # Clip to sane bounds (same as s1_feature_engine does for *_bps operators).
    out = np.clip(out, -1e5, 1e5)
    return out


def generate_vp_for_day(
    s0_dir: str,
    out_dir: str,
    asset: str,
    date_str: str,
    lookback_dates: int = 1,
    n_bins: int = _N_BINS_DEFAULT,
    require_complete: bool = True,
    skip_existing: bool = True,
    verbose: bool = True,
) -> Optional[pd.DataFrame]:
    """
    Compute VP (POC / VAH / VAL) parquet for one asset-date.

    Args:
        s0_dir:           Directory with S0 hourly parquets.
        out_dir:          Directory to write vp_{asset}_{date}.parquet.
        asset:            'btc' or 'eth'.
        date_str:         YYYY-MM-DD for the target day.
        lookback_dates:   How many days of prior S0 history to load (for
                          warming up the 1d window at the start of the day).
                          Default 1 — required for 1d window.
        n_bins:           Histogram bins per VP computation.
        require_complete: If True, raises if any target-day hour is missing.
                          Lookback hours are always optional (treated as
                          warmup — NaN until enough history accumulates).
        skip_existing:    Skip if output already exists.
        verbose:          Print progress.

    Returns:
        Output DataFrame (86400 rows) or None if skipped.
    """
    a = asset.lower()
    out_path = Path(out_dir) / f"vp_{a}_{date_str}.parquet"

    if skip_existing and out_path.exists():
        _log(f"Skip (exists): {out_path.name}", verbose)
        return None

    _log(f"Computing volume profile: {asset} {date_str} "
         f"(lookback={lookback_dates}d)", verbose)
    t0 = time.time()

    # --- Target-day bounds ---
    target_date = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    day_start = target_date
    day_end   = target_date + timedelta(days=1) - timedelta(seconds=1)

    # --- Load target day (required per flag) ---
    target_df = _load_s0_range(
        s0_dir=s0_dir, asset=asset,
        start_utc=day_start, end_utc=day_end,
        require_complete=require_complete, verbose=verbose,
    )

    # --- Load lookback days (always optional) ---
    lookback_df = pd.DataFrame(columns=["bucket_dt_utc", "price", "volume"])
    if lookback_dates > 0:
        lb_start = day_start - timedelta(days=lookback_dates)
        lb_end   = day_start - timedelta(seconds=1)
        _log(f"  Lookback: {lb_start} → {lb_end}", verbose)
        lookback_df = _load_s0_range(
            s0_dir=s0_dir, asset=asset,
            start_utc=lb_start, end_utc=lb_end,
            require_complete=False, verbose=verbose,
        )

    # --- Combine into a single DataFrame sorted by time ---
    if target_df.empty and lookback_df.empty:
        raise ValueError(
            f"No S0 data found for {asset} {date_str} (target+lookback both empty)"
        )
    data = pd.concat([lookback_df, target_df], ignore_index=True)
    data = data.sort_values("bucket_dt_utc").reset_index(drop=True)
    data = data.drop_duplicates(subset=["bucket_dt_utc"], keep="last")
    # Drop rows with non-positive volume (no trades → nothing to profile).
    data = data[data["volume"] > 0].copy()
    _log(f"  Combined dataset: {len(data)} rows with volume > 0", verbose)

    # --- Target 1s grid (86400 rows) ---
    grid = pd.date_range(start=pd.Timestamp(day_start),
                         end=pd.Timestamp(day_end),
                         freq="1s", tz="UTC")
    out_df = pd.DataFrame({"bucket_dt_utc": grid})

    # --- Compute (POC, VAH, VAL) for each configured window ---
    for label, window_s, recomp_s, shift_s in _WINDOW_CONFIGS:
        _log(f"  Window {label}: window_s={window_s} recompute_every={recomp_s}s",
             verbose)
        poc, vah, val = _compute_vp_column_triple(
            data=data, grid=grid,
            window_s=window_s, recompute_every_s=recomp_s,
            n_bins=n_bins, verbose=verbose, label=label,
        )
        out_df[f"poc_{label}_fut"] = poc
        out_df[f"vah_{label}_fut"] = vah
        out_df[f"val_{label}_fut"] = val

        # POC migration (passthrough column, precomputed to avoid hot-path cost).
        mig = _poc_migration_bps(poc, shift_s=shift_s)
        out_df[f"poc_migration_{label}_bps_fut"] = mig

    # --- Write ---
    _atomic_write_parquet(out_df, out_path)
    mb = out_path.stat().st_size / (1024 * 1024)
    _log(f"  Saved: {out_path.name}  ({mb:.2f} MB, {len(out_df)} rows)  "
         f"in {time.time() - t0:.2f}s", verbose)
    return out_df


# =============================================================================
# Unit tests
# =============================================================================

def _run_unit_tests() -> None:
    """
    Exercise compute_vp_window() on synthetic distributions. Prints a summary
    and exits 0 on success, 1 on any assertion failure.
    """
    print("Running compute_vp_window() unit tests...")
    failures = 0

    # Test 1 — single-price trading.
    try:
        prices  = np.array([100.0, 100.0, 100.0, 100.0, 100.0])
        volumes = np.array([1.0,   2.0,   1.5,   3.0,   1.0])
        poc, vah, val = compute_vp_window(prices, volumes, n_bins=50)
        assert poc == 100.0, f"single-price POC: got {poc}"
        assert vah == 100.0, f"single-price VAH: got {vah}"
        assert val == 100.0, f"single-price VAL: got {val}"
        print("  single-price: POC=VAH=VAL=100.0")
    except AssertionError as e:
        print(f"  single-price test: {e}")
        failures += 1

    # Test 2 — uniform distribution: VA should cover ~70% of the bin range.
    try:
        np.random.seed(42)
        prices  = np.random.uniform(100.0, 200.0, size=10000)
        volumes = np.ones_like(prices)
        poc, vah, val = compute_vp_window(prices, volumes, n_bins=50)
        va_span = vah - val
        total_span = 200.0 - 100.0
        coverage = va_span / total_span
        # Allow +/- 8% slack (histogram discretization + greedy expansion).
        assert 0.60 < coverage < 0.80, \
            f"uniform VA coverage out of [0.60, 0.80]: got {coverage:.3f}"
        assert 100.0 <= val <= 200.0 and 100.0 <= vah <= 200.0, \
            f"uniform VA bounds escape [100, 200]: val={val}, vah={vah}"
        print(f"  uniform distribution: VA span {va_span:.2f}/{total_span:.0f} "
              f"= {coverage:.3f} (target ~0.70)")
    except AssertionError as e:
        print(f"  uniform-distribution test: {e}")
        failures += 1

    # Test 3 — empty arrays → all NaN.
    try:
        poc, vah, val = compute_vp_window(np.array([]), np.array([]))
        assert np.isnan(poc) and np.isnan(vah) and np.isnan(val), \
            "empty input should give all-NaN"
        print("  empty arrays: (NaN, NaN, NaN)")
    except AssertionError as e:
        print(f"  empty-input test: {e}")
        failures += 1

    # Test 4 — zero-volume input → all NaN.
    try:
        prices  = np.array([100.0, 101.0, 102.0])
        volumes = np.array([0.0, 0.0, 0.0])
        poc, vah, val = compute_vp_window(prices, volumes)
        assert np.isnan(poc) and np.isnan(vah) and np.isnan(val), \
            "zero-volume should give all-NaN"
        print("  zero-volume: (NaN, NaN, NaN)")
    except AssertionError as e:
        print(f"  zero-volume test: {e}")
        failures += 1

    # Test 5 — bimodal distribution with one dominant peak.
    try:
        # 80% of volume at price 100, 20% at price 200.
        prices  = np.array([100.0] * 80 + [200.0] * 20)
        volumes = np.array([1.0]   * 80 + [1.0]   * 20)
        poc, vah, val = compute_vp_window(prices, volumes, n_bins=50)
        # POC should be inside the dominant-mode bin.
        assert 99.5 <= poc <= 101.5, f"POC should be near 100.0: got {poc}"
        # VA must contain the dominant mode.
        assert val <= 100.0 <= vah, f"VA [{val}, {vah}] should contain 100.0"
        print(f"  bimodal (80/20 at 100/200): POC={poc:.2f}, VA=[{val:.2f}, {vah:.2f}]")
    except AssertionError as e:
        print(f"  bimodal test: {e}")
        failures += 1

    # Test 6 — NaN mixed input: should ignore NaN rows.
    try:
        prices  = np.array([100.0, np.nan, 105.0, 105.0, 105.0])
        volumes = np.array([1.0,   5.0,    2.0,   2.0,   2.0])
        poc, vah, val = compute_vp_window(prices, volumes, n_bins=10)
        # Dominant bin is around 105 (3x), not around 100 or the NaN.
        assert 104.0 <= poc <= 106.0, f"POC should be near 105.0, got {poc}"
        print(f"  NaN mixed: POC={poc:.2f} (ignored NaN row)")
    except AssertionError as e:
        print(f"  nan-mixed test: {e}")
        failures += 1

    # Test 7 — poc_migration_bps basic.
    try:
        poc = np.array([100.0, 100.0, 100.0, 110.0, 110.0])
        mig = _poc_migration_bps(poc, shift_s=3)
        # mig[3] = (110 - 100) / 100 * 10000 = 1000.0
        assert np.isnan(mig[0]) and np.isnan(mig[2]), \
            f"lagged values not NaN before shift_s: {mig[:3]}"
        assert abs(mig[3] - 1000.0) < 1e-6, \
            f"expected migration 1000.0 bps, got {mig[3]}"
        print(f"  poc_migration_bps: shift=3, migration[3]={mig[3]:.1f} bps")
    except AssertionError as e:
        print(f"  poc-migration test: {e}")
        failures += 1

    print()
    if failures == 0:
        print(f"ALL UNIT TESTS PASSED")
        sys.exit(0)
    else:
        print(f"FAILED: {failures} test(s)")
        sys.exit(1)


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate volume-profile parquet (POC/VAH/VAL + migration) "
                    "from S0 feature files."
    )
    ap.add_argument("--s0-dir",  type=str, default=str(_DEFAULT_S0_DIR))
    ap.add_argument("--out-dir", type=str, default=str(_DEFAULT_OUT_DIR))
    ap.add_argument("--asset",   type=str, choices=["btc", "eth"],
                    help="btc or eth (not required when --test).")
    ap.add_argument("--date",    type=str,
                    help="Target date YYYY-MM-DD (not required when --test).")
    ap.add_argument("--lookback-dates", type=int, default=1,
                    help="Days of prior S0 history to load for warming up "
                         "the 1d window (default: 1).")
    ap.add_argument("--n-bins", type=int, default=_N_BINS_DEFAULT,
                    help=f"Histogram bins per VP computation "
                         f"(default: {_N_BINS_DEFAULT}).")
    ap.add_argument("--skip-existing", action="store_true", default=True,
                    help="Skip if output already exists (default: True).")
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    ap.add_argument("--no-require-complete", action="store_true",
                    help="Allow missing target-day S0 hours.")
    ap.add_argument("--quiet", "-q", action="store_true")
    ap.add_argument("--test", action="store_true",
                    help="Run unit tests on compute_vp_window() and exit.")

    args = ap.parse_args()

    if args.test:
        _run_unit_tests()
        return

    if not args.asset or not args.date:
        ap.error("--asset and --date are required unless --test is used.")

    generate_vp_for_day(
        s0_dir=args.s0_dir,
        out_dir=args.out_dir,
        asset=args.asset,
        date_str=args.date,
        lookback_dates=args.lookback_dates,
        n_bins=args.n_bins,
        require_complete=not args.no_require_complete,
        skip_existing=args.skip_existing,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()