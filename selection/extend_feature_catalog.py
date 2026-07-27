#!/usr/bin/env python3
"""
extend_feature_catalog.py
=========================
Extends feature_catalog.csv with the derived column `base_concept`.

base_concept = bare_name with stripped variant axes:
  - Window suffixes in seconds, minutes, hours, days: _\\d+[smhd]
    (at the end and in the middle, iteratively for nested cases)
  - Market-Scope-Tokens: _fut, _spot

Within one base_concept, features differ only on
(depth_band, window_s, market_scope) and share the same semantic
`group` value. This is the basis for the within-concept correlation
analysis in Section 3.4.2.

Run:
    cd <project_root>
    python -m selection.extend_feature_catalog

Output:
    results/selection/feature_catalog.csv  (in-place updated)
    results/selection/feature_catalog.csv.bak_pre_base_concept
        (backup before extension)

The validation report is printed to stdout.
"""
from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

import pandas as pd


CATALOG_PATH = Path("results/selection/feature_catalog.csv")
BACKUP_PATH  = Path("results/selection/feature_catalog.csv.bak_pre_base_concept")

# Variant axes for validation (allowed to vary within one base_concept)
VARIANT_AXES = ["depth_band", "window_s", "market_scope"]
# Semantic invariant (must be constant within one base_concept)
INVARIANT_COL = "group"


# ─── Derivation ──────────────────────────────────────────────────────────────

_WINDOW_END = re.compile(r"_(\d+)([smhd])$")
_WINDOW_MID = re.compile(r"_(\d+)([smhd])(?=_)")
_SCOPE      = re.compile(r"_(fut|spot)(?=_|$)")


def derive_base_concept(bare_name: str) -> str:
    """
    Strips window suffixes (s/m/h/d) and market-scope tokens.
    Iterated to a fixpoint (for nested patterns like _900s_15s_).
    """
    if not isinstance(bare_name, str):
        return bare_name
    n = bare_name
    prev = None
    while n != prev:
        prev = n
        n = _WINDOW_END.sub("", n)
        n = _WINDOW_MID.sub("", n)
    n = _SCOPE.sub("", n)
    return n


# ─── Validation ────────────────────────────────────────────────────────────

def validate_invariants(df: pd.DataFrame) -> list[dict]:
    """
    Within one base_concept, `group` must be constant.
    Returns list of violations.
    """
    feats = df[df["is_feature"] == True].copy()
    violations = []
    for bc, sub in feats.groupby("base_concept"):
        if len(sub) < 2:
            continue
        if sub[INVARIANT_COL].nunique(dropna=False) > 1:
            violations.append({
                "base_concept":    bc,
                "n_members":       len(sub),
                "distinct_groups": sorted(sub[INVARIANT_COL].fillna("NaN").unique().tolist()),
                "members_sample":  sub["bare_name"].head(6).tolist(),
            })
    return violations


def print_summary(df: pd.DataFrame) -> None:
    feats = df[df["is_feature"] == True].copy()

    bc_counts = feats["base_concept"].value_counts()
    reducible = bc_counts[bc_counts >= 2]
    singletons = bc_counts[bc_counts == 1]

    print(f"\n{'='*70}")
    print(f"base_concept SUMMARY")
    print(f"{'='*70}")
    print(f"Total features:       {len(feats)}")
    print(f"Unique base_concepts: {feats['base_concept'].nunique()}")
    print(f"  Reducible (>=2):    {len(reducible)}  covering {reducible.sum()} features")
    print(f"  Singletons (=1):    {len(singletons)}")

    # Per asset breakdown
    print(f"\nPer asset:")
    for asset, sub in feats.groupby("asset"):
        bc = sub["base_concept"].value_counts()
        red = (bc >= 2).sum()
        sing = (bc == 1).sum()
        print(f"  {asset:>3}: {len(sub)} features, {len(bc)} base_concepts, "
              f"{red} reducible, {sing} singletons")

    # Bucket size distribution
    print(f"\nBucket-size distribution:")
    for size, n in bc_counts.value_counts().sort_index().items():
        print(f"  size {size:>3}: {n:>4} buckets")


# ─── Main ───────────────────────────────────────────────────────────────────

def main() -> int:
    if not CATALOG_PATH.exists():
        print(f"FATAL: catalog not found at {CATALOG_PATH}", file=sys.stderr)
        return 1

    print(f"Reading catalog: {CATALOG_PATH}")
    df = pd.read_csv(CATALOG_PATH)
    n_total = len(df)
    print(f"  {n_total} rows, columns: {list(df.columns)}")

    if "base_concept" in df.columns:
        print(f"\nWarning: 'base_concept' column already exists — it will be overwritten.")

    # Derivation
    print(f"\nDeriving base_concept from bare_name...")
    df["base_concept"] = df["bare_name"].apply(derive_base_concept)

    # Validation
    print(f"Running structural validation (group-invariance)...")
    violations = validate_invariants(df)
    n_red = sum(
        1 for _, s in df[df["is_feature"] == True].groupby("base_concept") if len(s) >= 2
    )
    clean_rate = 100 * (1 - len(violations) / n_red) if n_red else 0.0

    print(f"  Reducible buckets:       {n_red}")
    print(f"  Group-invariance violations: {len(violations)}")
    print(f"  Cleanliness rate:        {clean_rate:.1f}%")

    if violations:
        print(f"\n  Violations (showing first 5):")
        for v in violations[:5]:
            print(f"    '{v['base_concept']}' (n={v['n_members']}): "
                  f"groups={v['distinct_groups']}")
            print(f"      Members: {v['members_sample']}")

        if clean_rate < 95.0:
            print(f"\nFATAL: Cleanliness rate <95% — derivation rule needs refinement.")
            print(f"       Catalog NOT written. Investigate violations above.")
            return 2
        else:
            print(f"\n  → Cleanliness >=95%, proceeding (review violations manually).")

    print_summary(df)

    # Backup + Write
    if BACKUP_PATH.exists():
        print(f"\nBackup already exists at {BACKUP_PATH} — keeping original backup.")
    else:
        print(f"\nBacking up: {CATALOG_PATH} → {BACKUP_PATH}")
        shutil.copy2(CATALOG_PATH, BACKUP_PATH)

    print(f"Writing extended catalog: {CATALOG_PATH}")
    df.to_csv(CATALOG_PATH, index=False)

    print(f"\nDone. Catalog now has {len(df.columns)} columns including 'base_concept'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())