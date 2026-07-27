#!/usr/bin/env python3
"""
build_feature_catalog.py
========================
Schema inventory and feature catalog builder for the Section 3.4 rerun.

Reads:
  - Feature specs from etl/spec/s{0..6}/*.py (FeatureSpec lists)
  - Parquet schema from one representative merged file

Writes (to --out-dir):
  - feature_catalog.csv  : one row per parquet column with full metadata
  - stage_summary.csv    : per-stage feature/meta/target counts (becomes Table 3.1)
  - unresolved.csv       : columns that could not be classified (manual review)
  - build_catalog.log    : full diagnostic log

This script does NOT modify any source data. It only reads.

Usage (run from project root, e.g. .):
  python results/selection/build_feature_catalog.py \
      --merged data_storage/s6_features_s5_full/merged_btceth_2026-02-16_03.parquet \
      --out-dir results/selection
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import pyarrow.parquet as pq


# ─── Classification rules ────────────────────────────────────────────────────
# Patterns applied to the BARE NAME (after stripping _btc/_eth/_btceth suffix).
# These are columns not registered in the FeatureSpec system but produced by
# upstream context/pre-computation layers (s0_context_batch.py and Section 3.2.3
# Pre-Computation Layer).

# META columns: health diagnostics, usability flags, internal artefacts.
META_PATTERNS: List[re.Pattern] = [
    re.compile(r"^bucket_dt_utc$"),
    re.compile(r"^data_usability_flag"),
    re.compile(r"^usability_(bad_count|bad_ratio|max_bad_streak|warmup_flag)"),
    re.compile(r"^unusable_reason_code"),
    re.compile(r"^__index_level_"),
    # L2 health diagnostics from s0_context_batch.py (not in specs)
    re.compile(r"^depth_availability$"),
    re.compile(r"^health_reason_code$"),
    re.compile(r"^l2_(bad_bitmask|bad_combos|crossed_combos|gap_combos"
               r"|invalid_ts_combos|missing_combos|reconnect_combos"
               r"|total_combos)$"),
]

# FEATURE columns from the Pre-Computation Layer (Section 3.2.3, Appendix B.9).
# These are real features (used for distance-to-level computations in S1) but
# not declared in the S0–S5 spec modules. They are attributed to stage S0
# because they feed into S1 and contain no rolling aggregation themselves.
PRECOMPUTATION_PATTERNS: List[re.Pattern] = [
    # Daily levels (spot + fut)
    re.compile(r"^(day_(open|high|low)|prev_day_(high|low))_(spot|fut)$"),
    # Weekly levels (fut only)
    re.compile(r"^(week_(open|high|low)|monday_(high|low)"
               r"|prev_week_(high|low))_fut$"),
    # Monthly levels (fut only)
    re.compile(r"^(month_(open|high|low)|prev_month_(high|low))_fut$"),
    # Volume profile artefacts (fut only): POC, VAH, VAL across 60m/240m/1d
    re.compile(r"^(poc|vah|val)_(60m|240m|1d)_fut$"),
    re.compile(r"^poc_migration_(60m|240m|1d)_bps_fut$"),
]

# Target columns by bare-name pattern (also caught at spec-file level below)
TARGET_NAME_PATTERNS: List[re.Pattern] = [
    re.compile(r"^ret_fwd_"),
    re.compile(r"^ret_mid_fwd_"),
    re.compile(r"^mfe_fwd_"),
    re.compile(r"^mae_fwd_"),
    re.compile(r"^rv_fwd_"),
    re.compile(r"^ca_ret_fwd_spread"),
    re.compile(r"^tbl_"),
    re.compile(r"^barrier_"),
]

# Spec-file basename fragments that indicate the file defines TARGETS.
TARGET_FILE_FRAGMENTS = (
    "forward_excursion",
    "forward_rv",
    "_returns",          # s1_returns, s2_returns, s3_returns
)

# Spec-file basename fragments that indicate the file defines META columns.
META_FILE_FRAGMENTS = (
    "_meta",             # s1_meta, s2_meta, s3_meta, s4_meta
    "health_spec",
    "calendar_spec",
)


# ─── Logging ─────────────────────────────────────────────────────────────────
def setup_logging(log_path: Path) -> logging.Logger:
    log = logging.getLogger("build_catalog")
    log.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    if not log.handlers:
        log.addHandler(fh)
        log.addHandler(ch)
    return log


# ─── Spec loading ────────────────────────────────────────────────────────────
def load_all_specs(
    spec_dir: Path,
    project_root: Path,
    log: logging.Logger,
    exclude_pattern: str = "",
) -> Tuple[List, Dict[str, str], int]:
    """
    Walk etl/spec/s[0-6]/*.py and load all FeatureSpec lists.

    Args:
        exclude_pattern: Regex (re.search) applied to spec.name. Matching
                         specs are skipped. Empty string disables filtering.

    Returns:
        all_specs:    list of FeatureSpec objects (flat)
        spec_origin:  dict mapping spec.name -> source file basename
        n_excluded:   number of specs filtered out by exclude_pattern
    """
    from etl.spec import FeatureSpec  # noqa: F401

    exclude_re = re.compile(exclude_pattern) if exclude_pattern else None

    all_specs: List = []
    spec_origin: Dict[str, str] = {}
    files_loaded = 0
    files_failed = 0
    n_excluded = 0

    for stage_dir in sorted(spec_dir.glob("s[0-9]*")):
        if not stage_dir.is_dir():
            continue

        for spec_file in sorted(stage_dir.glob("*.py")):
            name = spec_file.name
            if name == "__init__.py":
                continue
            if ".bak" in name:
                continue

            rel = spec_file.relative_to(project_root)
            module_path = ".".join(rel.with_suffix("").parts)

            try:
                module = importlib.import_module(module_path)
            except Exception as e:
                log.warning("  import failed: %s (%s)", module_path, e)
                files_failed += 1
                continue

            n_in_file = 0
            n_excluded_in_file = 0
            for attr_name, attr_val in inspect.getmembers(module):
                if not isinstance(attr_val, list) or not attr_val:
                    continue
                if not isinstance(attr_val[0], FeatureSpec):
                    continue
                for spec in attr_val:
                    if exclude_re is not None and exclude_re.search(spec.name):
                        n_excluded_in_file += 1
                        n_excluded += 1
                        continue
                    all_specs.append(spec)
                    spec_origin[spec.name] = spec_file.stem
                    n_in_file += 1

            if n_excluded_in_file > 0:
                log.info(
                    "  %-40s %4d specs (excluded %d by regex)",
                    spec_file.stem, n_in_file, n_excluded_in_file,
                )
            else:
                log.info("  %-40s %4d specs", spec_file.stem, n_in_file)
            files_loaded += 1

    log.info(
        "Spec loading: %d files OK, %d failed, %d total specs (excluded %d)",
        files_loaded, files_failed, len(all_specs), n_excluded,
    )
    return all_specs, spec_origin, n_excluded


# ─── Column classification ───────────────────────────────────────────────────
def strip_asset_suffix(col: str) -> Tuple[str, str]:
    """Return (bare_name, asset). Asset is one of: btc, eth, btceth, ''."""
    if col.endswith("_btc"):
        return col[:-4], "btc"
    if col.endswith("_eth"):
        return col[:-4], "eth"
    if col.endswith("_btceth"):
        return col[:-7], "btceth"
    if col.startswith("ca_"):
        # cross-asset column without pair tag suffix
        return col, "btceth"
    return col, ""


def classify_column(
    col: str,
    spec_index: Dict[str, object],
    spec_origin: Dict[str, str],
) -> dict:
    """
    Classify one parquet column. Returns a dict with all catalog fields.
    """
    record = {
        "column": col,
        "bare_name": "",
        "asset": "",
        "stage": "",
        "group": "",
        "spec_file": "",
        "depth_band": "",
        "window_s": "",
        "market_scope": "",
        "feature_id": "",
        "matched_spec_name": "",
        "is_meta": False,
        "is_target": False,
        "is_feature": False,
        "match_source": "",
    }

    # Strip suffix first so all subsequent pattern matching is on the bare name.
    bare, asset = strip_asset_suffix(col)
    record["bare_name"] = bare
    record["asset"] = asset

    # 1) Non-spec META patterns (context_batch / health diagnostics)
    for pat in META_PATTERNS:
        if pat.search(bare):
            record["is_meta"] = True
            record["stage"] = "S0"
            record["group"] = "Data Health / Context"
            record["match_source"] = "meta_pattern"
            return record

    # 2) Pre-Computation Layer features (Section 3.2.3, Appendix B.9):
    #    daily/weekly/monthly levels and volume-profile artefacts.
    #    These are real features and are attributed to stage S0.
    for pat in PRECOMPUTATION_PATTERNS:
        if pat.search(bare):
            record["is_feature"] = True
            record["stage"] = "S0"
            # Group label distinguishes the two artefact types
            if bare.startswith(("poc_", "vah_", "val_")):
                record["group"] = "Volume Profile Artefact"
            else:
                record["group"] = "Level Artefact"
            record["match_source"] = "precomputation_pattern"
            # Market scope inferable from bare name
            if bare.endswith("_spot"):
                record["market_scope"] = "Spot"
            elif bare.endswith("_fut"):
                record["market_scope"] = "Futures"
            return record

    # 3) Spec lookup
    spec = spec_index.get(bare) or spec_index.get(col)
    if spec is not None:
        record["stage"] = spec.stage
        record["group"] = spec.group
        record["feature_id"] = spec.feature_id if spec.feature_id is not None else ""
        record["spec_file"] = spec_origin.get(spec.name, "")
        record["matched_spec_name"] = spec.name
        record["match_source"] = "spec"

        params = spec.params or {}
        record["window_s"] = params.get("window_s", "")
        record["market_scope"] = params.get("market_scope", "")
        record["depth_band"] = params.get("depth_band", "")

        # File-level role detection
        sf = record["spec_file"]
        if any(frag in sf for frag in TARGET_FILE_FRAGMENTS):
            record["is_target"] = True
        elif any(frag in sf for frag in META_FILE_FRAGMENTS):
            record["is_meta"] = True
        else:
            # Belt-and-suspenders: check name pattern in case a forward target
            # slipped into a non-target file
            if any(p.search(bare) for p in TARGET_NAME_PATTERNS):
                record["is_target"] = True
            else:
                record["is_feature"] = True
        return record

    # 4) No spec match: check target name patterns
    if any(p.search(bare) for p in TARGET_NAME_PATTERNS):
        record["is_target"] = True
        record["match_source"] = "target_pattern"
        # Infer stage from name where possible
        if col.startswith("ca_"):
            record["stage"] = "S6"
        return record

    # 5) Unresolved
    record["match_source"] = "unresolved"
    return record


# ─── Main ────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build feature catalog from specs + merged parquet schema."
    )
    parser.add_argument(
        "--merged",
        required=True,
        help="Path to one representative merged parquet file",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Output directory for catalog/summary/unresolved CSVs and log",
    )
    parser.add_argument(
        "--spec-dir",
        default="etl/spec",
        help="Spec root (default: etl/spec)",
    )
    parser.add_argument(
        "--project-root",
        default=os.getcwd(),
        help="Project root for module imports (default: current working dir)",
    )
    parser.add_argument(
        "--exclude-spec-regex",
        default=r"(?:^|_)bnb(?:_|$)|_btcbnb$|_ethbnb$",
        help=(
            "Regex pattern (re.search semantics) applied to spec.name. "
            "Any spec whose name matches is excluded from the catalog. "
            "Default: BNB asset-token and BNB-pair suffixes. "
            "Pass --exclude-spec-regex '' to disable filtering."
        ),
    )
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    spec_dir = Path(args.spec_dir).resolve()
    merged_path = Path(args.merged).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Make sure project root is importable

    log = setup_logging(out_dir / "build_catalog.log")
    log.info("=" * 70)
    log.info("Feature Catalog Builder")
    log.info("=" * 70)
    log.info("project_root: %s", project_root)
    log.info("spec_dir:     %s", spec_dir)
    log.info("merged file:  %s", merged_path)
    log.info("out_dir:      %s", out_dir)

    if not spec_dir.exists():
        log.error("Spec directory not found: %s", spec_dir)
        sys.exit(1)
    if not merged_path.exists():
        log.error("Merged file not found: %s", merged_path)
        sys.exit(1)

    # 1) Load specs
    log.info("─── Loading specs ───")
    exclude_pattern = args.exclude_spec_regex.strip()
    if exclude_pattern:
        log.info("Excluding specs matching regex: %r", exclude_pattern)
    else:
        log.info("No exclusion pattern (all specs included)")
    specs, spec_origin, n_excluded_specs = load_all_specs(
        spec_dir, project_root, log, exclude_pattern=exclude_pattern,
    )
    spec_index: Dict[str, object] = {}
    duplicates = 0
    for s in specs:
        if s.name in spec_index:
            duplicates += 1
        spec_index[s.name] = s
    if duplicates:
        log.warning("Spec index has %d duplicate names (last wins)", duplicates)
    log.info("Unique spec names: %d  (excluded %d by regex)",
             len(spec_index), n_excluded_specs)

    # 2) Read parquet schema
    log.info("─── Reading parquet schema ───")
    schema = pq.read_schema(str(merged_path))
    cols = list(schema.names)
    log.info("Parquet columns: %d", len(cols))

    # 3) Classify
    log.info("─── Classifying columns ───")
    records = [classify_column(c, spec_index, spec_origin) for c in cols]
    df = pd.DataFrame(records)

    # 4) Counts and diagnostics
    n_feat = int(df["is_feature"].sum())
    n_meta = int(df["is_meta"].sum())
    n_tgt = int(df["is_target"].sum())
    n_unres = int((df["match_source"] == "unresolved").sum())
    log.info("Totals: %d feature | %d meta | %d target | %d unresolved",
             n_feat, n_meta, n_tgt, n_unres)
    if n_feat + n_meta + n_tgt + n_unres != len(df):
        log.warning("Count mismatch: sum (%d) != total cols (%d)",
                    n_feat + n_meta + n_tgt + n_unres, len(df))

    # Spec features not present in parquet (orphaned specs).
    # Use the actual matched spec.name, not the stripped bare_name. Otherwise
    # specs with a pair-tag suffix (e.g. "ca_x_btceth") falsely appear as
    # orphans because the matched bare_name is "ca_x" without the suffix.
    matched_spec_names = {r["matched_spec_name"] for r in records
                          if r["matched_spec_name"]}
    orphan_specs = sorted(set(spec_index.keys()) - matched_spec_names)
    log.info("Specs not represented in parquet (orphan): %d", len(orphan_specs))
    if orphan_specs[:10]:
        log.info("  sample orphans: %s", orphan_specs[:10])

    # 5) Per-stage summary (becomes new Table 3.1)
    stage_df = df[df["stage"] != ""].groupby("stage").agg(
        n_features=("is_feature", "sum"),
        n_meta=("is_meta", "sum"),
        n_target=("is_target", "sum"),
        n_total=("column", "count"),
    ).reset_index().sort_values("stage")
    log.info("Per-stage summary:\n%s", stage_df.to_string(index=False))

    # 6) Write outputs
    df.to_csv(out_dir / "feature_catalog.csv", index=False)
    log.info("Written: feature_catalog.csv  (%d rows)", len(df))

    stage_df.to_csv(out_dir / "stage_summary.csv", index=False)
    log.info("Written: stage_summary.csv")

    unres_df = df[df["match_source"] == "unresolved"].copy()
    unres_df.to_csv(out_dir / "unresolved.csv", index=False)
    log.info("Written: unresolved.csv  (%d rows)", len(unres_df))

    if len(unres_df) > 0:
        log.warning(
            "Unresolved columns need manual review. Sample (first 30): %s",
            unres_df["column"].head(30).tolist(),
        )

    # Orphan specs to a separate file for transparency
    if orphan_specs:
        pd.DataFrame({"orphan_spec_name": orphan_specs}).to_csv(
            out_dir / "orphan_specs.csv", index=False
        )
        log.info("Written: orphan_specs.csv  (%d rows)", len(orphan_specs))

    log.info("Done.")


if __name__ == "__main__":
    main()