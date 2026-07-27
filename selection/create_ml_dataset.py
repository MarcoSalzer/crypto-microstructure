# ==============================================================================
# Create ML Prediction Dataset  (whitelist-based)
# ==============================================================================
# Joins S5 (BTC + ETH) and S6 (cross-asset) Parquet files on timestamp and
# keeps exactly the columns listed in feature_keep.csv.
#
# ── Column naming ─────────────────────────────────────────────────────────────
#   BTC S5 columns → suffixed _btc  (e.g. queue_pressure_log_fut_1bps_1s_btc)
#   ETH S5 columns → suffixed _eth
#   S6 columns     → kept as-is     (e.g. ca_spread_bps_fut_spread_1s_btceth)
#
# ── feature_keep.csv schema ───────────────────────────────────────────────────
#   column   : exact column name as it appears in the merged frame (keep list)
#   type     : feature | target | meta
#   asset    : btc | eth | btceth
#   source   : S5 | S6
#   bare_name: original name before asset suffix
#
# ── Input paths ───────────────────────────────────────────────────────────────
#   BTC S5:  {base_dir}/s5_features/s5_features_btc_{date}_{hh}.parquet
#   ETH S5:  {base_dir}/s5_features/s5_features_eth_{date}_{hh}.parquet
#   S6:      {base_dir}/s6_features/s6_features_*_{date}_{hh}.parquet
#   Keep:    results/selection/feature_keep.csv
#
# ── Output path ───────────────────────────────────────────────────────────────
#   {base_dir}/ml_features/ml_features_{date}_{hh}.parquet
#   Contains exactly the columns in feature_keep.csv that exist in the data,
#   indexed by bucket_dt_utc (DatetimeIndex).
#
# ── Usage ─────────────────────────────────────────────────────────────────────
#   python create_ml_dataset.py --date 2026-02-16 --hour 6 --dry-run
#   python create_ml_dataset.py --date 2026-02-16
#   python create_ml_dataset.py --start-date 2026-02-16 --end-date 2026-03-31
#   python create_ml_dataset.py --auto --overwrite
#
# ==============================================================================

from __future__ import annotations

import argparse
import gc
import logging
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


# ==============================================================================
# Configuration
# ==============================================================================

DEFAULT_BASE_DIR  = "data_storage"
DEFAULT_KEEP_LIST = "results/selection/feature_keep.csv"

_S5_SUBDIR  = "s5_features"
_S6_SUBDIR  = "s6_features_btceth"
_OUT_SUBDIR = "ml_features"

_ALL_ASSETS = ["btc", "eth"]

# Column used as the timestamp index — MUST NOT be suffixed with _btc/_eth.
_INDEX_COL = "bucket_dt_utc"


# ==============================================================================
# Path helpers
# ==============================================================================

def _s5_path(base_dir: str, asset: str, date_str: str, hour: int) -> Path:
    return (
        Path(base_dir)
        / _S5_SUBDIR
        / f"s5_features_{asset}_{date_str}_{hour:02d}.parquet"
    )


def _s6_path(base_dir: str, date_str: str, hour: int) -> Optional[Path]:
    """
    Locate the S6 Parquet for a given date/hour regardless of which assets
    are included in the filename (btceth, btcethbnb, etc.).
    Returns the best match, or None if no file is found.
    """
    pattern = f"s6_features_*_{date_str}_{hour:02d}.parquet"
    matches = sorted((Path(base_dir) / _S6_SUBDIR).glob(pattern))
    if not matches:
        return None
    if len(matches) > 1:
        # Prefer alphabetically last = most assets (btcethbnb > btceth)
        logger.warning(
            "Multiple S6 files found for %s h%02d — using: %s",
            date_str, hour, matches[-1].name,
        )
        return matches[-1]
    return matches[0]


def _out_path(base_dir: str, date_str: str, hour: int) -> Path:
    out_dir = Path(base_dir) / _OUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"ml_features_{date_str}_{hour:02d}.parquet"


# ==============================================================================
# Keep-list loading
# ==============================================================================

def load_keep_list(keep_list_path: str) -> Tuple[List[str], Dict[str, str]]:
    """
    Load feature_keep.csv and return:
      - ordered list of columns to keep
      - dict mapping column → type (feature|target|meta)
    Preserves order: features, targets, meta.
    """
    path = Path(keep_list_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Keep list not found: {path}\n"
            f"  Generate it by running:\n"
            f"    python -m selection.build_feature_keep"
        )

    df = pd.read_csv(path)

    if "column" not in df.columns:
        raise ValueError(
            f"feature_keep.csv must have a 'column' column. "
            f"Found: {list(df.columns)}"
        )

    df["column"] = df["column"].astype(str).str.strip()
    df = df[df["column"].str.len() > 0]

    # Deduplicate while preserving order
    df = df.drop_duplicates(subset=["column"], keep="first")

    cols = df["column"].tolist()

    # Build type map
    if "type" in df.columns:
        type_map = dict(zip(df["column"], df["type"]))
    else:
        type_map = {c: "feature" for c in cols}

    n_feat = sum(1 for v in type_map.values() if v == "feature")
    n_tgt  = sum(1 for v in type_map.values() if v == "target")
    n_meta = sum(1 for v in type_map.values() if v == "meta")

    logger.info(
        "Keep list loaded: %d columns  (feature=%d, target=%d, meta=%d)",
        len(cols), n_feat, n_tgt, n_meta,
    )

    return cols, type_map


# ==============================================================================
# Load helpers
# ==============================================================================

def _ensure_dt_index(df: pd.DataFrame, source_label: str) -> pd.DataFrame:
    """
    Ensure bucket_dt_utc is the DatetimeIndex.

    Three possible states in the Parquet:
      1. bucket_dt_utc is already the index → done
      2. bucket_dt_utc is a regular column  → set_index
      3. Neither exists                     → warn, use existing index
    """
    if df.index.name == _INDEX_COL:
        # Already the index — ensure datetime
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            df.index = pd.to_datetime(df.index, utc=True)
        return df

    if _INDEX_COL in df.columns:
        df[_INDEX_COL] = pd.to_datetime(df[_INDEX_COL], utc=True)
        df = df.set_index(_INDEX_COL)
        return df

    # Fallback: check for a datetime index with a different name
    if pd.api.types.is_datetime64_any_dtype(df.index):
        logger.warning(
            "%s: no '%s' column or index found, but index is datetime — using it.",
            source_label, _INDEX_COL,
        )
        df.index.name = _INDEX_COL
        return df

    logger.warning(
        "%s: no timestamp column/index found — join may fail.", source_label,
    )
    return df


def _load_s5(path: Path, asset: str) -> Optional[pd.DataFrame]:
    """Load an S5 Parquet, set timestamp index, then suffix columns with _{asset}."""
    if not path.exists():
        logger.warning("S5 not found for %s — skipping: %s", asset.upper(), path.name)
        return None

    df = pd.read_parquet(path)

    # Dedup columns (keep=first)
    dupes = df.columns[df.columns.duplicated()].tolist()
    if dupes:
        logger.warning(
            "S5 %s: %d duplicate columns removed (keep=first): %s",
            asset.upper(), len(dupes), dupes[:5],
        )
        df = df.loc[:, ~df.columns.duplicated(keep="first")]

    # Set timestamp as index BEFORE renaming — otherwise it becomes
    # bucket_dt_utc_btc and the cross-asset join breaks.
    df = _ensure_dt_index(df, f"S5-{asset.upper()}")

    # Suffix all remaining columns with _{asset}
    df.columns = [f"{c}_{asset}" for c in df.columns]

    logger.info(
        "Loaded S5 %-3s | %d rows × %d cols | %s",
        asset.upper(), len(df), len(df.columns), path.name,
    )
    return df


def _load_s6(
    path: Path,
    reference_index: Optional[pd.Index] = None,
) -> Optional[pd.DataFrame]:
    """
    Load an S6 Parquet (ca_* columns, no renaming needed).

    S6 files often lack a timestamp column/index (just a RangeIndex 0–3599).
    If so and a reference_index from S5 is provided with matching length,
    we assign it directly — safe because both S5 and S6 are generated at
    1-second resolution for the same hour window.
    """
    if not path.exists():
        logger.warning("S6 not found — skipping: %s", path.name)
        return None

    df = pd.read_parquet(path)

    # Dedup columns
    dupes = df.columns[df.columns.duplicated()].tolist()
    if dupes:
        logger.warning(
            "S6: %d duplicate columns removed (keep=first): %s",
            len(dupes), dupes[:5],
        )
        df = df.loc[:, ~df.columns.duplicated(keep="first")]

    # Ensure timestamp index — try the standard path first
    df = _ensure_dt_index(df, "S6")

    # If still no datetime index, assign from S5 reference
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        if reference_index is not None and len(df) == len(reference_index):
            logger.info(
                "S6: assigning timestamp index from S5 reference (%d rows).",
                len(df),
            )
            df.index = reference_index
        elif reference_index is not None:
            logger.error(
                "S6: row count mismatch — S6=%d vs S5 reference=%d. "
                "Cannot assign index, skipping S6.",
                len(df), len(reference_index),
            )
            return None
        else:
            logger.warning(
                "S6: no timestamp and no reference index — join will fail.",
            )

    logger.info(
        "Loaded S6     | %d rows × %d cols | %s",
        len(df), len(df.columns), path.name,
    )
    return df


# ==============================================================================
# Core: build one hour file
# ==============================================================================

def build_hour(
    date_str:  str,
    hour:      int,
    base_dir:  str,
    keep_cols: List[str],
    type_map:  Dict[str, str],
    overwrite: bool = False,
    dry_run:   bool = False,
) -> Optional[Path]:
    """
    Build the ML feature Parquet for a single date/hour.
    Returns the output Path on success, None if skipped or aborted.
    """
    out = _out_path(base_dir, date_str, hour)

    if out.exists() and not overwrite and not dry_run:
        logger.info("Exists, skipping: %s", out.name)
        return out

    logger.info("━━ Building  date=%s  hour=%02d ━━", date_str, hour)

    # ── 1. Load S5 per asset ─────────────────────────────────────────────────
    frames: List[pd.DataFrame] = []

    for asset in _ALL_ASSETS:
        df_asset = _load_s5(_s5_path(base_dir, asset, date_str, hour), asset)
        if df_asset is not None:
            frames.append(df_asset)

    if not frames:
        logger.error(
            "No S5 data available for date=%s hour=%02d — skipping.",
            date_str, hour,
        )
        return None

    # ── 2. Load S6 (optional) ─────────────────────────────────────────────────
    # S6 files may lack a timestamp index — pass the S5 index as reference
    # so we can align by position (both are 1-second resolution, same hour).
    reference_index = frames[0].index  # first S5 asset's DatetimeIndex
    s6_file = _s6_path(base_dir, date_str, hour)
    df_s6 = (
        _load_s6(s6_file, reference_index=reference_index)
        if s6_file is not None else None
    )
    if df_s6 is not None:
        frames.append(df_s6)
    else:
        logger.warning(
            "S6 data missing for %s h%02d — output will contain S5 features only.",
            date_str, hour,
        )

    # ── 3. Join all frames on timestamp index (inner) ─────────────────────────
    if len(frames) == 1:
        df = frames[0].copy()
    else:
        # Sequential inner join — each frame is indexed by bucket_dt_utc
        df = frames[0]
        for right in frames[1:]:
            df = df.join(right, how="inner")
    del frames
    gc.collect()

    if len(df) == 0:
        logger.error(
            "Empty DataFrame after join (no overlapping timestamps) — "
            "skipping %s h%02d.", date_str, hour,
        )
        return None

    logger.info("Merged: %d rows × %d cols", len(df), len(df.columns))

    # ── 4. Health gate ────────────────────────────────────────────────────────
    flag_btc = "data_usability_flag_btc"
    flag_eth = "data_usability_flag_eth"
    n_before = len(df)

    if flag_btc in df.columns and flag_eth in df.columns:
        mask = (df[flag_btc] == 1) & (df[flag_eth] == 1)
        df   = df.loc[mask]
        n_dropped = n_before - len(df)
        if n_dropped > 0:
            logger.info(
                "Health gate: %d/%d rows removed (%.1f%% usable).",
                n_dropped, n_before, 100 * len(df) / n_before,
            )
    else:
        present = [f for f in [flag_btc, flag_eth] if f in df.columns]
        logger.warning(
            "Health gate: only %d/2 usability flags found (%s) — "
            "applying partial gate.",
            len(present), present,
        )
        for flag in present:
            df = df[df[flag] == 1]

    if len(df) == 0:
        logger.error(
            "All rows filtered by health gate — skipping %s h%02d.",
            date_str, hour,
        )
        return None

    # ── 5. Apply whitelist ────────────────────────────────────────────────────
    present = [c for c in keep_cols if c in df.columns]
    missing = [c for c in keep_cols if c not in df.columns]

    if missing:
        # Classify missing columns by type for clearer diagnostics
        miss_feat = [c for c in missing if type_map.get(c) == "feature"]
        miss_tgt  = [c for c in missing if type_map.get(c) == "target"]
        miss_meta = [c for c in missing if type_map.get(c) == "meta"]

        logger.info(
            "%d requested columns absent from merged frame "
            "(feature=%d, target=%d, meta=%d).",
            len(missing), len(miss_feat), len(miss_tgt), len(miss_meta),
        )
        if miss_feat:
            logger.debug("  Missing features (first 10): %s", miss_feat[:10])

    df = df[present].copy()

    # Log output composition by type
    out_feat = [c for c in present if type_map.get(c) == "feature"]
    out_tgt  = [c for c in present if type_map.get(c) == "target"]
    out_meta = [c for c in present if type_map.get(c) == "meta"]
    n_s6     = sum(1 for c in out_feat if c.startswith("ca_"))

    logger.info(
        "Whitelist applied: %d/%d columns  "
        "(features=%d [S6=%d], targets=%d, meta=%d, absent=%d)",
        len(present), len(keep_cols),
        len(out_feat), n_s6, len(out_tgt), len(out_meta), len(missing),
    )

    # ── 6. Sanity checks ─────────────────────────────────────────────────────
    nan_frac = df.isna().mean().mean()
    if nan_frac > 0.20:
        logger.warning(
            "High overall NaN fraction: %.1f%% — check S5/S6 source data.",
            nan_frac * 100,
        )

    # Verify no forward-return look-ahead in feature columns
    lookahead_cols = [c for c in out_feat
                 if "ret_fwd" in c or "ca_ret_fwd" in c]
    if lookahead_cols:
        logger.error(
            "LOOK-AHEAD DETECTED: %d forward-return columns classified as features: %s",
            len(lookahead_cols), lookahead_cols,
        )
        raise ValueError(
            f"Forward-return look-ahead in feature columns: {lookahead_cols}. "
            f"Fix feature_keep.csv — these must be type=target."
        )

    # ── 7. Write ──────────────────────────────────────────────────────────────
    if dry_run:
        _print_summary(df, out.name, missing, type_map)
        return None

    df.to_parquet(out, index=True, compression="snappy")
    size_kb = out.stat().st_size // 1024
    logger.info(
        "Written: %s  [%d rows × %d cols, %d KB]",
        out.name, len(df), len(df.columns), size_kb,
    )
    return out


# ==============================================================================
# Dry-run summary
# ==============================================================================

def _print_summary(
    df: pd.DataFrame,
    fname: str,
    missing: List[str],
    type_map: Dict[str, str],
) -> None:
    from collections import Counter

    present_cols = list(df.columns)

    # Group by type
    by_type = Counter(type_map.get(c, "unknown") for c in present_cols)

    # Group features by family (first token of bare name)
    def _family(col: str) -> str:
        if col.startswith("ca_"):
            return "S6_cross_asset"
        base = col.removesuffix("_btc").removesuffix("_eth")
        return base.split("_")[0]

    feat_cols = [c for c in present_cols if type_map.get(c) == "feature"]
    family_counts = Counter(_family(c) for c in feat_cols)

    print(f"\n{'━'*62}")
    print(f"  [DRY-RUN]  {fname}")
    print(f"  {len(df)} rows × {len(df.columns)} cols")
    print(f"  Index: {df.index.name}  ({df.index.dtype})")
    print(f"{'─'*62}")
    print(f"  {'TYPE':<20} {'COUNT':>6}")
    print(f"  {'─'*26}")
    for t in ["feature", "target", "meta", "unknown"]:
        if by_type.get(t, 0) > 0:
            print(f"  {t:<20} {by_type[t]:>6}")
    print(f"{'─'*62}")
    print(f"  {'FEATURE FAMILY':<40} {'COUNT':>6}")
    print(f"  {'─'*46}")
    for family, n in sorted(family_counts.items(), key=lambda x: -x[1]):
        print(f"  {family:<40} {n:>6}")
    print(f"{'─'*62}")
    print(f"  Missing (absent in data): {len(missing)}")
    if missing:
        for m in missing[:10]:
            print(f"    [{type_map.get(m, '?'):>7}] {m}")
        if len(missing) > 10:
            print(f"    ... and {len(missing) - 10} more")
    print(f"{'━'*62}\n")


# ==============================================================================
# Date-range helper
# ==============================================================================

def _date_range(start: str, end: str) -> List[str]:
    d0, d1 = date.fromisoformat(start), date.fromisoformat(end)
    if d1 < d0:
        raise ValueError(f"--end-date {end} is before --start-date {start}")
    out, cur = [], d0
    while cur <= d1:
        out.append(cur.isoformat())
        cur += timedelta(days=1)
    return out


# ==============================================================================
# Auto-discovery
# ==============================================================================

def _discover_available(base_dir: str) -> List[tuple]:
    """
    Scan s5_features/ for BTC files and return sorted (date_str, hour) tuples.
    Uses BTC as the reference asset — ETH is expected to match.
    Pattern: s5_features_btc_{date}_{hh}.parquet
    """
    pattern = "s5_features_btc_*.parquet"
    files   = sorted((Path(base_dir) / _S5_SUBDIR).glob(pattern))

    jobs = []
    for f in files:
        # e.g. s5_features_btc_2026-02-16_06.parquet
        parts = f.stem.split("_")  # ['s5', 'features', 'btc', '2026-02-16', '06']
        if len(parts) < 5:
            continue
        date_str = parts[3]
        try:
            hour = int(parts[4])
        except ValueError:
            continue
        jobs.append((date_str, hour))

    return jobs


# ==============================================================================
# CLI
# ==============================================================================

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="create_ml_dataset",
        description=(
            "Build the final ML feature Parquet files.\n"
            "Joins S5 (BTC+ETH) and S6 (cross-asset), applies health gate,\n"
            "and keeps exactly the columns in feature_keep.csv.\n\n"
            "Output: data_storage/ml_features/ml_features_{date}_{hh}.parquet"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    dg = p.add_argument_group("Date / Hour")
    dg.add_argument("--date", metavar="YYYY-MM-DD",
                    help="Single date (all 24h unless --hour/--hours given).")
    dg.add_argument("--hour",  type=int, metavar="H",
                    help="Single hour 0–23 (requires --date).")
    dg.add_argument("--hours", nargs="+", type=int, metavar="H",
                    help="Explicit hour list (requires --date).")
    dg.add_argument("--start-date", metavar="YYYY-MM-DD",
                    help="Start of date range (inclusive).")
    dg.add_argument("--end-date",   metavar="YYYY-MM-DD",
                    help="End of date range (inclusive).")
    dg.add_argument("--auto", action="store_true",
                    help="Auto-discover all available files in s5_features/ "
                         "and process them.")

    pg = p.add_argument_group("Paths")
    pg.add_argument("--base-dir",  default=DEFAULT_BASE_DIR,
                    help=f"Root data directory (default: {DEFAULT_BASE_DIR}).")
    pg.add_argument("--keep-list", default=DEFAULT_KEEP_LIST,
                    help=f"Path to feature_keep.csv (default: {DEFAULT_KEEP_LIST}).")

    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing output files.")
    p.add_argument("--dry-run",   action="store_true",
                   help="Print column summary without writing files.")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    # ── Resolve jobs ─────────────────────────────────────────────────────────
    if args.auto:
        jobs = _discover_available(args.base_dir)
        if not jobs:
            logger.error(
                "--auto: no s5_features_btc_*.parquet files found in %s/%s",
                args.base_dir, _S5_SUBDIR,
            )
            sys.exit(1)
        logger.info("Mode: auto  (%d jobs discovered)", len(jobs))
    elif args.start_date and args.end_date:
        dates = _date_range(args.start_date, args.end_date)
        jobs  = [(d, h) for d in dates for h in range(24)]
        logger.info(
            "Mode: date range %s → %s  (%d dates × 24h = %d jobs)",
            args.start_date, args.end_date, len(dates), len(jobs),
        )
    elif args.date:
        hours = (
            [args.hour]  if args.hour  is not None else
            args.hours   if args.hours             else
            list(range(24))
        )
        jobs = [(args.date, h) for h in hours]
        logger.info("Mode: %s  hours=%s  (%d jobs)", args.date, hours, len(jobs))
    else:
        parser.error("Specify --auto, --date, or --start-date/--end-date.")

    # ── Load keep list once ───────────────────────────────────────────────────
    keep_cols, type_map = load_keep_list(args.keep_list)

    # ── Process ───────────────────────────────────────────────────────────────
    total = written = skipped = failed = 0

    for date_str, hour in jobs:
        total += 1
        try:
            result = build_hour(
                date_str  = date_str,
                hour      = hour,
                base_dir  = args.base_dir,
                keep_cols = keep_cols,
                type_map  = type_map,
                overwrite = args.overwrite,
                dry_run   = args.dry_run,
            )
            if result is not None:
                written += 1
            else:
                skipped += 1
        except Exception as exc:
            logger.exception(
                "Error for date=%s hour=%02d: %s", date_str, hour, exc,
            )
            failed += 1

    logger.info(
        "Done: %d/%d written, %d skipped, %d failed",
        written, total, skipped, failed,
    )
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()