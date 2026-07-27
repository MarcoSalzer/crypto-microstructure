#!/usr/bin/env python3
# ==============================================================================
# merge_s5_s6_full.py
#
# Joins FULL (unreduced) S5 features with S6 cross-asset features on their
# shared timestamp index, producing a combined DataFrame per (date, hour).
#
# This script uses the raw s5_features/ directory (as opposed to a reduced
# s5_features_reduced/ variant) so that NO features are dropped.
# Use this dataset for cluster-specific feature importance (ws4b --full-features)
# where globally unimportant features may still be relevant at cluster moments.
#
# ── Input paths ────────────────────────────────────────────────────────────────
#   S5 full:  {base_dir}/s5_features/s5_features_{asset}_{date}_{hh}.parquet
#   S6:       {base_dir}/s6_features/s6_features_{btceth}_{date}_{hh}.parquet
#
# ── Output path ───────────────────────────────────────────────────────────────
#   Merged:   {base_dir}/s6_features_s5_full/merged_btceth_{date}_{hh}.parquet
#
# ── Asset tag logic ───────────────────────────────────────────────────────────
#   Tag inferred from S6 filename. BNB excluded (moved out of pipeline).
#   S5 columns are suffixed with _{asset} to match S6 convention.
#
# ── Join policy ───────────────────────────────────────────────────────────────
#   Inner join on timestamp index. A warning is logged when the inner join
#   loses more than 0.1% of rows.
#
# ── Deduplication ─────────────────────────────────────────────────────────────
#   Duplicate columns present in both S5 and S6 are dropped,
#   keeping the S5 version (ground truth for single-asset features).
#
# Usage:
#   python -m selection.merge_s5_s6_full --base-dir data_storage
#   python -m selection.merge_s5_s6_full --base-dir data_storage --overwrite
#   python -m selection.merge_s5_s6_full --base-dir data_storage --dry-run
#   python -m selection.merge_s5_s6_full --base-dir data_storage --date 2026-03-01
# ==============================================================================

from __future__ import annotations

import os
import signal

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# ── Logging ───────────────────────────────────────────────────────────────────

LOG_DIR = "logs"


def _setup_logging() -> logging.Logger:
    os.makedirs(LOG_DIR, exist_ok=True)
    log = logging.getLogger("merge_s5_s6_full")
    log.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(
        os.path.join(LOG_DIR, "merge_s5_s6_full.log"),
        mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    if not log.handlers:
        log.addHandler(fh)
        log.addHandler(ch)
    return log


logger = logging.getLogger("merge_s5_s6_full")

# ── Constants ─────────────────────────────────────────────────────────────────

_S5_FULL_SUBDIR = "s5_features"
_S6_SUBDIR      = "s6_features_btceth"
_OUT_SUBDIR     = "s6_features_s5_full"

# Non-feature columns to drop from S5 before suffixing
_S5_DROP_COLS = {"data_usability_flag", "__index_level_0__"}

_S6_FILENAME_RE = re.compile(
    r"^s6_features_(?P<tag>[a-z]+)_(?P<date>\d{4}-\d{2}-\d{2})_(?P<hour>\d{2})\.parquet$"
)

_TAG_TO_ASSETS: Dict[str, List[str]] = {
    "btceth": ["btc", "eth"],
}

_ROW_LOSS_WARN = 0.001

# ── Path helpers ──────────────────────────────────────────────────────────────

def _s5_path(base_dir: Path, asset: str, date: str, hour: int) -> Path:
    return (base_dir / _S5_FULL_SUBDIR
            / f"s5_features_{asset}_{date}_{hour:02d}.parquet")


def _out_path(base_dir: Path, date: str, hour: int) -> Path:
    out_dir = base_dir / _OUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"merged_btceth_{date}_{hour:02d}.parquet"

# ── Discovery ─────────────────────────────────────────────────────────────────

def discover_s6_files(base_dir: Path) -> List[Tuple[str, str, int, Path]]:
    s6_dir = base_dir / _S6_SUBDIR
    if not s6_dir.exists():
        logger.error("S6 directory not found: %s", s6_dir)
        return []
    results = []
    for fpath in sorted(s6_dir.glob("s6_features_*.parquet")):
        m = _S6_FILENAME_RE.match(fpath.name)
        if not m:
            continue
        tag  = m.group("tag")
        date = m.group("date")
        hour = int(m.group("hour"))
        if tag not in _TAG_TO_ASSETS:
            logger.warning("Unknown asset tag '%s' — skipping %s", tag, fpath.name)
            continue
        results.append((tag, date, hour, fpath))
    return results


def filter_by_date(
    files: List[Tuple[str, str, int, Path]],
    date: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> List[Tuple[str, str, int, Path]]:
    if date:
        return [f for f in files if f[1] == date]
    if date_from or date_to:
        lo = date_from or "0000-00-00"
        hi = date_to   or "9999-99-99"
        return [f for f in files if lo <= f[1] <= hi]
    return files

# ── Merge ─────────────────────────────────────────────────────────────────────

def merge_one(
    base_dir: Path,
    tag: str,
    date: str,
    hour: int,
    s6_path: Path,
    overwrite: bool = False,
    dry_run: bool = False,
) -> Optional[Path]:
    out = _out_path(base_dir, date, hour)

    if out.exists() and not overwrite:
        logger.info("Exists, skipping: %s", out.name)
        return out

    if dry_run:
        logger.info("[DRY-RUN] Would merge → %s", out.name)
        return None

    assets = _TAG_TO_ASSETS[tag]

    # Load full S5 per asset
    s5_dfs: List[pd.DataFrame] = []
    for asset in assets:
        p = _s5_path(base_dir, asset, date, hour)
        if not p.exists():
            logger.error("S5 missing: %s", p)
            return None
        df = pd.read_parquet(p)
        # Drop non-feature cols before suffixing
        df = df.drop(columns=[c for c in df.columns if c in _S5_DROP_COLS],
                     errors="ignore")
        # Suffix all columns with _{asset}
        df.columns = [f"{c}_{asset}" for c in df.columns]
        logger.info("Loaded S5-full %-3s | %d rows × %d cols | %s",
                    asset.upper(), len(df), len(df.columns), p.name)
        s5_dfs.append(df)

    # Load S6
    df_s6 = pd.read_parquet(s6_path)
    logger.info("Loaded S6 %-9s | %d rows × %d cols | %s",
                tag, len(df_s6), len(df_s6.columns), s6_path.name)

    # Join S5 assets
    df_s5_all = s5_dfs[0]
    for df_asset in s5_dfs[1:]:
        df_s5_all = df_s5_all.join(df_asset, how="inner")

    n_s5 = len(df_s5_all)

    # Join S5 + S6
    merged = df_s5_all.join(df_s6, how="inner", rsuffix="_s6_dup")
    n_merged = len(merged)

    row_loss = (n_s5 - n_merged) / max(n_s5, 1)
    if row_loss > _ROW_LOSS_WARN:
        logger.warning("Inner join dropped %.2f%% of rows (%d → %d).",
                       row_loss * 100, n_s5, n_merged)

    # Drop duplicate columns (keep S5 version)
    dup_cols = [c for c in merged.columns if c.endswith("_s6_dup")]
    if dup_cols:
        logger.info("Dropping %d duplicate col(s) (keeping S5 version): %s",
                    len(dup_cols), dup_cols)
        merged = merged.drop(columns=dup_cols)

    # Write
    merged.to_parquet(out, index=True, compression="snappy")
    size_kb = out.stat().st_size // 1024
    logger.info("Written: %s | %d rows × %d cols | %d KB",
                out.name, n_merged, len(merged.columns), size_kb)

    return out

# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Merge full S5 + S6 Parquet files → s6_features_s5_full/\n\n"
            "Input:   data_storage/s5_features/s5_features_{asset}_{date}_{hh}.parquet\n"
            "         data_storage/s6_features/s6_features_btceth_{date}_{hh}.parquet\n"
            "Output:  data_storage/s6_features_s5_full/merged_btceth_{date}_{hh}.parquet"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--base-dir", default="data_storage")
    parser.add_argument("--date",      help="Single date (YYYY-MM-DD)")
    parser.add_argument("--date-from", help="Start of date range (YYYY-MM-DD)")
    parser.add_argument("--date-to",   help="End of date range (YYYY-MM-DD)")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run",   action="store_true")
    args = parser.parse_args()

    global logger
    logger = _setup_logging()

    base_dir = Path(args.base_dir)
    if not base_dir.exists():
        logger.error("Base directory not found: %s", base_dir)
        sys.exit(1)

    all_files = discover_s6_files(base_dir)
    if not all_files:
        logger.error("No S6 files found in %s", base_dir / _S6_SUBDIR)
        sys.exit(1)

    files = filter_by_date(all_files, args.date, args.date_from, args.date_to)
    if not files:
        logger.error("No S6 files match the date filter.")
        sys.exit(1)

    logger.info("Found %d S6 file(s) to process%s.", len(files),
                f" (filtered from {len(all_files)} total)"
                if len(files) < len(all_files) else "")

    written = failed = 0
    for tag, date, hour, s6_path in files:
        result = merge_one(base_dir, tag, date, hour, s6_path,
                           overwrite=args.overwrite, dry_run=args.dry_run)
        if not args.dry_run:
            if result is None:
                failed += 1
            else:
                written += 1

    print()
    print("=" * 70)
    if args.dry_run:
        print(f"DRY-RUN: {len(files)} file(s) would be merged.")
    else:
        print(f"Done — written: {written}  failed: {failed}  "
              f"skipped (exists): {len(files) - written - failed}")
    print(f"Output: {base_dir / _OUT_SUBDIR}")


if __name__ == "__main__":
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        os.setsid()
    except OSError:
        pass
    main()