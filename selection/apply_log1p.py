#!/usr/bin/env python3
"""
apply_log1p.py

Applies log1p IN-PLACE (same column name, transformed value) to a
verified list of quantity/rate features and writes the results
into a NEW folder. The original is left untouched.

Safety mechanisms:
  1. Column list comes from a CSV (log1p_final_columns.csv), NOT hardcoded.
  2. Runtime check per column per file: if a listed column has negative values
     -> the column is NOT transformed in THIS file (otherwise log1p(neg)=NaN),
        and the incident is logged. Prevents silent NaN creation.
  3. Idempotency: writes .log1p_manifest.json into the target folder. If the script
     again on the same destination folder, it aborts (no double log1p).
  4. Unlisted columns are passed through 1:1 (incl. targets/meta).
  5. The transformed column's dtype is set to float32 (log1p produces floats).

Usage:
  python apply_log1p.py \
      --src data_storage/ml_features \
      --dst data_storage/ml_features_log1p \
      --cols log1p_final_columns.csv

  # Dry run (just check, write nothing):
  python apply_log1p.py --src ... --dst ... --cols ... --dry-run

Recommendation: --dry-run first, review the report, then without --dry-run.
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import glob
import signal
from datetime import datetime

import numpy as np
import pandas as pd

try:  # long-running job hardening (as is usual in the pipeline)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
except (OSError, ValueError):
    pass

MANIFEST_NAME = ".log1p_manifest.json"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="Source folder with *.parquet")
    p.add_argument("--dst", required=True, help="Destination folder (created if missing)")
    p.add_argument("--cols", required=True,
                   help="CSV with a 'column' column listing the columns to transform")
    p.add_argument("--pattern", default="*.parquet")
    p.add_argument("--dry-run", action="store_true",
                   help="Only check + report, write nothing")
    p.add_argument("--neg-tol", type=float, default=0.0,
                   help="Tolerance for a negative min value (default 0.0 = strict)")
    return p.parse_args()


def load_target_cols(cols_csv):
    df = pd.read_csv(cols_csv)
    if "column" not in df.columns:
        sys.exit(f"ERROR: {cols_csv} has no 'column' column")
    cols = sorted(set(df["column"].astype(str).tolist()))
    if not cols:
        sys.exit("ERROR: empty column list")
    return cols


def check_idempotency(dst, dry_run):
    mpath = os.path.join(dst, MANIFEST_NAME)
    if os.path.exists(mpath):
        sys.exit(
            f"ABORT: {mpath} already exists — target folder was already "
            f"transformed. Double log1p is prevented. Delete the folder "
            f"or choose a new --dst if you want to re-transform.")
    if not dry_run:
        os.makedirs(dst, exist_ok=True)


def write_manifest(dst, src, cols, stats):
    manifest = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "source_dir": os.path.abspath(src),
        "transform": "log1p",
        "n_target_columns": len(cols),
        "target_columns": cols,
        "files_processed": stats["files"],
        "columns_skipped_due_to_negatives": stats["neg_skips"],
    }
    with open(os.path.join(dst, MANIFEST_NAME), "w") as f:
        json.dump(manifest, f, indent=2)


def main():
    a = parse_args()
    src, dst = a.src, a.dst
    target_cols = load_target_cols(a.cols)
    print(f"Columns to transform: {len(target_cols)}")

    if os.path.abspath(src) == os.path.abspath(dst):
        sys.exit("ERROR: --src and --dst must not be identical.")

    check_idempotency(dst, a.dry_run)

    files = sorted(glob.glob(os.path.join(src, a.pattern)))
    if not files:
        sys.exit(f"ERROR: no files in {src}/{a.pattern}")
    print(f"Files: {len(files)}")
    print(f"Mode: {'DRY-RUN (nothing written)' if a.dry_run else 'WRITE -> ' + dst}")
    print("=" * 90)

    target_set = set(target_cols)
    # neg_skips[col] = number of files in which col was skipped due to negative values
    neg_skips: dict[str, int] = {}
    # transform_counts[col] = number of files in which col was transformed
    transform_counts: dict[str, int] = {}
    cols_seen_in_data: set[str] = set()
    files_done = 0

    for i, fpath in enumerate(files, 1):
        try:
            df = pd.read_parquet(fpath)
        except Exception as e:
            print(f"  WARN could not read {os.path.basename(fpath)}: {e}")
            continue

        present = [c for c in df.columns if c in target_set]
        cols_seen_in_data.update(present)

        for c in present:
            col = df[c]
            if not pd.api.types.is_numeric_dtype(col):
                neg_skips[c] = neg_skips.get(c, 0) + 1
                continue
            mn = np.nanmin(col.values) if len(col) else 0.0
            if mn < -abs(a.neg_tol):
                # SAFETY NET: negative values -> do NOT transform (would produce NaN)
                neg_skips[c] = neg_skips.get(c, 0) + 1
                continue
            # log1p in-place; NaN stays NaN (log1p(nan)=nan), no new NaN from neg.
            df[c] = np.log1p(col.astype(np.float64)).astype(np.float32)
            transform_counts[c] = transform_counts.get(c, 0) + 1

        if not a.dry_run:
            out = os.path.join(dst, os.path.basename(fpath))
            df.to_parquet(out, compression="zstd", index=False)
        files_done += 1
        del df
        if i % 50 == 0 or i == len(files):
            print(f"  {i}/{len(files)} files processed")

    print("=" * 90)
    # Columns from the list that appeared in no file
    missing = sorted(target_set - cols_seen_in_data)
    print(f"\nColumns transformed (in >=1 file): {len(transform_counts)}")
    if neg_skips:
        print(f"\nWARNING: {len(neg_skips)} column(s) skipped due to negative values:")
        for c, n in sorted(neg_skips.items()):
            print(f"   {c:46s} skipped in {n} file(s)")
        print("   -> These columns are NOT consistently log1p-transformed.")
        print("   -> Check: do they really belong in the log1p list?")
    if missing:
        print(f"\nNOTE: {len(missing)} listed column(s) appeared in NO file:")
        for c in missing[:20]:
            print(f"   {c}")
        if len(missing) > 20:
            print(f"   ... (+{len(missing)-20} more)")

    stats = {"files": files_done, "neg_skips": neg_skips}
    if not a.dry_run:
        write_manifest(dst, src, target_cols, stats)
        print(f"\nManifest written: {os.path.join(dst, MANIFEST_NAME)}")
        print(f"Transformed files in: {dst}")
    else:
        print("\nDRY-RUN finished — no files written.")
        if not neg_skips and not missing:
            print("All listed columns are non-negative and present — ready for a real run.")


if __name__ == "__main__":
    main()