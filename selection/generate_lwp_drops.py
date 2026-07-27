#!/usr/bin/env python3
"""
generate_lwp_drops.py
=====================
Produces an LWP-specific hard-drop block for consolidated_drop_list.csv.

Background
-----------
The LWP family (`lwp_<side>_<depth>_<scope>_<window>s`) shows, in the
cross-concept correlation analysis, an average absolute correlation of
0.9999 across *all* axes — between 1s and 900s, between
different depths, and between bid/ask/mid. The cause is that the
variance of the LWP features is dominated by the price level, while the
differences between variants are in the sub-bps range.

Consequence: in a correlation sense the *levels* of the LWP features are a
single feature. The informative variation of a depth-vs-depth difference
is captured not by the level features but by the explicitly constructed
z_lwp_minus_mid_* features.

Reduction rule
--------------
Keep:
  - lwp_mid_10bps_fut_60s    (per asset)
  - lwp_mid_10bps_spot_60s   (per asset)
  - all z_lwp_minus_mid_*   (not touched by this script)

All other `lwp_<side>_<depth>_<scope>_<window>s` features are written into
consolidated_drop_list.csv with
    list       = 'primary'
    drop_layer = 'correlation_structural'
    reason     = 'lwp_structural_redundancy'

Usage
-----
  # Dry-run (default):
  python -m selection.generate_lwp_drops

  # Write:
  python -m selection.generate_lwp_drops --write

  # Choose a different marker (if the plan changes):
  python -m selection.generate_lwp_drops \\
      --keep-side mid --keep-depth 10bps --keep-window 60
"""
from __future__ import annotations
from common.paths import REDUCTION_DIR

import argparse
import re
import shutil
import sys
from pathlib import Path

import pandas as pd


# ─── Configuration ──────────────────────────────────────────────────────────

BASE_DIR        = REDUCTION_DIR
CATALOG_PATH    = BASE_DIR / "feature_catalog.csv"
DROP_LIST_PATH  = BASE_DIR / "consolidated_drop_list.csv"
BACKUP_PATH     = BASE_DIR / "consolidated_drop_list.csv.bak_pre_lwp"

# Default keep-rule
DEFAULT_KEEP_SIDE   = "mid"
DEFAULT_KEEP_DEPTH  = "10bps"
DEFAULT_KEEP_WINDOW = 60  # seconds
# Both scopes (spot + fut) are always retained for the chosen (side,depth,window)

# Pattern for pure LWP level features: lwp_<side>_<depth>_<scope>_<window>s
LWP_PATTERN = re.compile(
    r"^lwp_(?P<side>mid|ask|bid)_(?P<depth>1bps|2bps|5bps|10bps|struct50|struct100)"
    r"_(?P<scope>fut|spot)_(?P<window>\d+)s$"
)

# Drop metadata
DROP_LAYER = "correlation_structural"
DROP_REASON = "lwp_structural_redundancy"


def parse_lwp(bare_name: str) -> dict | None:
    """Return dict with side/depth/scope/window, or None if not a pure LWP feature."""
    m = LWP_PATTERN.match(bare_name)
    return m.groupdict() if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="Actually append to drop list (default: dry-run).")
    ap.add_argument("--keep-side",   default=DEFAULT_KEEP_SIDE,
                    choices=["mid", "ask", "bid"])
    ap.add_argument("--keep-depth",  default=DEFAULT_KEEP_DEPTH,
                    choices=["1bps", "2bps", "5bps", "10bps", "struct50", "struct100"])
    ap.add_argument("--keep-window", type=int, default=DEFAULT_KEEP_WINDOW)
    args = ap.parse_args()

    print(f"\n{'='*70}")
    print(f"generate_lwp_drops")
    print(f"{'='*70}\n")
    print(f"Keep rule: side={args.keep_side}, depth={args.keep_depth}, "
          f"window={args.keep_window}s, scope=(fut, spot)")

    # ── 1. Load catalog ─────────────────────────────────────────────────
    if not CATALOG_PATH.exists():
        print(f"FATAL: catalog not found at {CATALOG_PATH}")
        sys.exit(1)
    cat = pd.read_csv(CATALOG_PATH)
    print(f"\nCatalog: {len(cat)} rows")

    # ── 2. Identify pure LWP level features ─────────────────────────────
    lwp = cat[cat["bare_name"].str.match(r"^lwp_(mid|ask|bid)_", na=False)].copy()
    parsed = lwp["bare_name"].apply(parse_lwp)
    valid_mask = parsed.notna()
    if not valid_mask.all():
        bad = lwp.loc[~valid_mask, "bare_name"].unique()
        print(f"{len(bad)} LWP-like bare_names did not match pattern (skipped):")
        for n in bad[:5]:
            print(f"    {n}")

    lwp = lwp[valid_mask].copy()
    for k in ["side", "depth", "scope", "window"]:
        lwp[k] = parsed[valid_mask].apply(lambda d, k=k: d[k])
    lwp["window"] = lwp["window"].astype(int)

    print(f"\nPure LWP level features in catalog: {len(lwp)}")
    print(f"  Per asset: {lwp['asset'].value_counts().to_dict()}")

    # ── 3. Apply keep rule ──────────────────────────────────────────────
    keep_mask = (
        (lwp["side"]   == args.keep_side) &
        (lwp["depth"]  == args.keep_depth) &
        (lwp["window"] == args.keep_window) &
        lwp["scope"].isin({"fut", "spot"})
    )

    keep_df = lwp[keep_mask]
    drop_df = lwp[~keep_mask]

    print(f"\nKeep: {len(keep_df)} columns ({len(keep_df) // 2} unique bare_names × 2 assets)")
    for c in sorted(keep_df["column"]):
        print(f"  KEEP: {c}")

    print(f"\nDrop: {len(drop_df)} columns")
    print(f"  Per asset: {drop_df['asset'].value_counts().to_dict()}")
    print(f"  Sample (first 8 drops):")
    for c in sorted(drop_df["column"])[:8]:
        print(f"    DROP: {c}")
    print(f"    ...")

    # Sanity: each kept bare_name must exist in both assets
    bare_counts = keep_df.groupby("bare_name")["asset"].nunique()
    if (bare_counts < 2).any():
        missing = bare_counts[bare_counts < 2].index.tolist()
        print(f"\nBare names missing one asset: {missing}")
        print(f"  Verify the chosen keep-rule produces symmetric BTC/ETH coverage.")

    if len(keep_df) == 0:
        print(f"\nFATAL: keep rule matched no features. Aborting.")
        sys.exit(1)

    # ── 4. Build drop list entries ──────────────────────────────────────
    # consolidated_drop_list uses bare_name (suffix-aware expansion happens later).
    # We drop pairs (bare_name × asset), so we deduplicate to bare_name level.
    drop_bare = sorted(drop_df["bare_name"].unique())
    print(f"\nUnique bare_names to drop: {len(drop_bare)}")

    new_entries = pd.DataFrame({
        "feature_name":          drop_bare,
        "list":                  "primary",
        "drop_layer":            DROP_LAYER,
        "reason":                DROP_REASON,
        "btc_weighted_pctile":   pd.NA,
        "eth_weighted_pctile":   pd.NA,
        "bnb_weighted_pctile":   pd.NA,
        "max_cross_pctile":      pd.NA,
    })

    # ── 5. Merge into existing drop list (idempotent) ───────────────────
    existing = None
    if DROP_LIST_PATH.exists() and DROP_LIST_PATH.stat().st_size > 0:
        try:
            existing = pd.read_csv(DROP_LIST_PATH)
        except pd.errors.EmptyDataError:
            print(f"\nDrop list at {DROP_LIST_PATH} is empty/headerless — treating as new")
            existing = None

    if existing is not None and len(existing) > 0:
        print(f"\nExisting drop list: {len(existing)} rows")
        if "drop_layer" in existing.columns:
            print(f"  By layer: {existing['drop_layer'].value_counts().to_dict()}")

        # Remove any prior entries from this layer to ensure idempotency
        prior_same_layer = existing["drop_layer"] == DROP_LAYER
        if prior_same_layer.any():
            print(f"  Removing {prior_same_layer.sum()} prior '{DROP_LAYER}' entries")
            existing = existing[~prior_same_layer]

        # Also remove any LWP bare_names already in other layers — would be redundant
        overlap = existing["feature_name"].isin(drop_bare)
        if overlap.any():
            print(f"  {overlap.sum()} LWP bare_names already in drop list under other layers:")
            for _, r in existing[overlap].head(5).iterrows():
                print(f"      {r['feature_name']:<40s}  layer={r['drop_layer']}  reason={r['reason']}")
            print(f"    These will be SUPERSEDED by the new lwp_structural_redundancy entries.")
            existing = existing[~overlap]

        merged = pd.concat([existing, new_entries], ignore_index=True)
    else:
        print(f"\nNo usable existing drop list at {DROP_LIST_PATH} — creating new")
        merged = new_entries.copy()

    # ── 6. Summary ──────────────────────────────────────────────────────
    print(f"\n─── Final drop list ───")
    print(f"  Total rows: {len(merged)}")
    print(f"  By layer: {merged['drop_layer'].value_counts().to_dict()}")
    new_lwp_n = (merged["drop_layer"] == DROP_LAYER).sum()
    print(f"  New {DROP_LAYER} entries: {new_lwp_n}")

    # ── 7. Write ────────────────────────────────────────────────────────
    if args.write:
        if DROP_LIST_PATH.exists():
            if BACKUP_PATH.exists():
                BACKUP_PATH.unlink()
            shutil.copy2(DROP_LIST_PATH, BACKUP_PATH)
            print(f"\nBackup: {BACKUP_PATH}")
        merged.to_csv(DROP_LIST_PATH, index=False)
        print(f"Written: {DROP_LIST_PATH} ({len(merged)} rows)")
    else:
        print(f"\n[DRY-RUN] Not writing. Re-run with --write to apply.")


if __name__ == "__main__":
    main()