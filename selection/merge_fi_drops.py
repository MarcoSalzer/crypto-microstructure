#!/usr/bin/env python3
"""
merge_fi_drops.py
=================
Merges the universally-weak feature list produced by aggregate_fi_results.py
into consolidated_drop_list.csv.

Idempotent by design
--------------------
Each FI round writes its drops under a distinct drop_layer label
(feature_importance_round_1, feature_importance_round_2, ...). When this
script runs, it FIRST removes any existing rows carrying the same
drop_layer label, then appends the fresh set. Re-running the same round
therefore replaces rather than duplicates, and running round 2 leaves
round 1 untouched.

Schema alignment
----------------
consolidated_drop_list.csv columns:
    feature_name, list, drop_layer, reason,
    btc_weighted_pctile, eth_weighted_pctile,
    bnb_weighted_pctile, max_cross_pctile

fi_drop_candidates.csv (from aggregate_fi_results.py) provides:
    feature_name, list, drop_layer, reason,
    btc_mean_gain_over_null, eth_mean_gain_over_null

The compat percentile columns are left empty for FI rows (they belonged to
the old percentile methodology). The gain-over-null evidence is preserved
in two dedicated columns appended to the drop list if not already present.

Usage
-----
  # Dry-run (shows what would change):
  python -m selection.merge_fi_drops

  # Apply:
  python -m selection.merge_fi_drops --write

  # Custom round label:
  python -m selection.merge_fi_drops --write --layer feature_importance_round_2
"""
from __future__ import annotations
from common.paths import REDUCTION_DIR

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd


# ─── Configuration ──────────────────────────────────────────────────────────

BASE_DIR        = REDUCTION_DIR
DROP_LIST_PATH  = BASE_DIR / "consolidated_drop_list.csv"
FI_CANDIDATES   = (BASE_DIR / "results" / "s5_s6_feature_importance_aggregated"
                   / "fi_drop_candidates.csv")

# Canonical column order of the consolidated drop list
DROP_LIST_COLS = [
    "feature_name", "list", "drop_layer", "reason",
    "btc_weighted_pctile", "eth_weighted_pctile",
    "bnb_weighted_pctile", "max_cross_pctile",
]
# FI-specific columns appended to the drop list so the gain-over-null
# evidence is not lost. Added only if missing.
FI_EXTRA_COLS = ["btc_mean_gain_over_null", "eth_mean_gain_over_null"]


def _load_drop_list() -> pd.DataFrame:
    if DROP_LIST_PATH.exists() and DROP_LIST_PATH.stat().st_size > 0:
        return pd.read_csv(DROP_LIST_PATH)
    return pd.DataFrame(columns=DROP_LIST_COLS)


def _backup(path: Path) -> None:
    if path.exists():
        bak = path.with_suffix(".csv.bak_pre_fi_merge")
        shutil.copy2(path, bak)
        print(f"  Backup: {bak}")


def main():
    ap = argparse.ArgumentParser(
        description="Merge FI drop candidates into consolidated_drop_list.csv")
    ap.add_argument("--candidates", type=Path, default=FI_CANDIDATES,
                    help="Path to fi_drop_candidates.csv")
    ap.add_argument("--drop-list", type=Path, default=DROP_LIST_PATH,
                    help="Path to consolidated_drop_list.csv")
    ap.add_argument("--layer", type=str, default=None,
                    help="Override the drop_layer label. By default the label "
                         "is taken from the candidates file itself.")
    ap.add_argument("--write", action="store_true",
                    help="Apply the merge. Without this flag the script only "
                         "reports what would change.")
    args = ap.parse_args()

    print(f"\n{'='*70}")
    print(f"merge_fi_drops")
    print(f"{'='*70}\n")

    # ── Load candidates ─────────────────────────────────────────────────
    if not args.candidates.exists():
        print(f"FATAL: candidates file not found: {args.candidates}")
        print(f"       Run aggregate_fi_results.py --write first.")
        sys.exit(1)

    cand = pd.read_csv(args.candidates)
    if cand.empty:
        print("Candidates file is empty — nothing to merge.")
        print("(This is a valid outcome: it means FI found no universally "
              "weak features this round.)")
        sys.exit(0)

    required = {"feature_name", "list", "drop_layer", "reason"}
    missing = required - set(cand.columns)
    if missing:
        print(f"FATAL: candidates file missing columns: {missing}")
        sys.exit(1)

    # Determine the layer label
    layer_labels = cand["drop_layer"].unique().tolist()
    if args.layer:
        layer = args.layer
        cand["drop_layer"] = layer
    elif len(layer_labels) == 1:
        layer = layer_labels[0]
    else:
        print(f"FATAL: candidates file has multiple drop_layer values "
              f"{layer_labels}. Pass --layer to disambiguate.")
        sys.exit(1)

    print(f"Candidates:    {args.candidates}")
    print(f"  Rows:        {len(cand)}")
    print(f"  drop_layer:  {layer}")

    # ── Load existing drop list ─────────────────────────────────────────
    existing = _load_drop_list()
    print(f"\nDrop list:     {args.drop_list}")
    print(f"  Rows:        {len(existing)}")
    if len(existing) and "drop_layer" in existing.columns:
        print(f"  By layer:    {existing['drop_layer'].value_counts().to_dict()}")

    # ── Idempotency: drop any prior rows with the same layer label ──────
    n_prior_same_layer = 0
    if len(existing) and "drop_layer" in existing.columns:
        n_prior_same_layer = int((existing["drop_layer"] == layer).sum())
        if n_prior_same_layer:
            print(f"\n  Found {n_prior_same_layer} existing rows with layer "
                  f"'{layer}' — these will be REPLACED (idempotent re-run).")
            existing = existing[existing["drop_layer"] != layer].copy()

    # ── Cross-layer dedup: skip features already dropped by another layer ─
    already_other_layer = set()
    if len(existing) and "feature_name" in existing.columns:
        already_other_layer = set(existing["feature_name"])
    n_cross_layer_dup = len(set(cand["feature_name"]) & already_other_layer)
    if n_cross_layer_dup:
        print(f"  {n_cross_layer_dup} candidate features are already in the "
              f"drop list under a different layer — these are skipped "
              f"(original layer attribution kept).")
        cand = cand[~cand["feature_name"].isin(already_other_layer)].copy()

    print(f"\n  Net new rows to add: {len(cand)}")

    # ── Align schema ────────────────────────────────────────────────────
    new_rows = pd.DataFrame()
    new_rows["feature_name"] = cand["feature_name"]
    new_rows["list"]         = cand["list"] if "list" in cand.columns else "primary"
    new_rows["drop_layer"]   = layer
    new_rows["reason"]       = cand["reason"]
    for col in ["btc_weighted_pctile", "eth_weighted_pctile",
                "bnb_weighted_pctile", "max_cross_pctile"]:
        new_rows[col] = pd.NA
    for col in FI_EXTRA_COLS:
        new_rows[col] = cand[col] if col in cand.columns else pd.NA

    # Ensure existing frame has the FI extra columns too
    for col in FI_EXTRA_COLS:
        if col not in existing.columns:
            existing[col] = pd.NA

    all_cols = DROP_LIST_COLS + FI_EXTRA_COLS
    for col in all_cols:
        if col not in existing.columns:
            existing[col] = pd.NA
        if col not in new_rows.columns:
            new_rows[col] = pd.NA
    existing = existing[all_cols]
    new_rows = new_rows[all_cols]

    combined = pd.concat([existing, new_rows], ignore_index=True)

    # ── Report ──────────────────────────────────────────────────────────
    print(f"\n─── Result ───")
    print(f"  Total rows after merge: {len(combined)}")
    print(f"  By layer:")
    for lyr, n in combined["drop_layer"].value_counts().items():
        print(f"    {lyr:<35s}: {n}")

    # ── Write ───────────────────────────────────────────────────────────
    if not args.write:
        print(f"\n[DRY-RUN] Not writing. Re-run with --write to apply.")
        return

    _backup(args.drop_list)
    args.drop_list.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.drop_list, index=False)
    print(f"\nWritten: {args.drop_list} ({len(combined)} rows)")


if __name__ == "__main__":
    main()