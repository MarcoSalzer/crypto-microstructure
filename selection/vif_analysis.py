#!/usr/bin/env python3
"""
vif_analysis.py  (v4 — adapted for 3.4.2 rerun)
================================================
Computes Variance Inflation Factor (VIF) for the FULL (unreduced) feature
corpus on the merged S5+S6 dataset.

Changes vs v3 (per Section-3.4 rerun plan):
    - Reads merged S5+S6 parquets from s6_features_s5_full/ (one file per
      (date, hour), contains both BTC and ETH and S6 features in one schema)
    - No drop_list dependency — VIF runs on the full unreduced corpus
    - Per-asset analysis: BTC VIF uses BTC features + S6 shared features;
      same for ETH. S6 features participate in both reports.
    - Asset-aware filtering via the catalog `column` (suffixed name) vs
      `bare_name` (unsuffixed) — merged parquets carry the suffixed form.
    - EXCLUDE_GROUPS consistent with within_concept_correlation (non-microstructure
      groups excluded: Trend, Level Artefact, Session Levels, ...)

Approach:
    1. Per asset: build feature list from catalog (is_feature, EXCLUDE_GROUPS,
       asset ∈ {btc, btceth}) and intersect with merged parquet schema
    2. Sample ~500K rows from merged parquets
    3. Compute correlation matrix via pandas .corr(min_periods=500)
    4. Project to nearest positive-definite matrix (eigenvalue clipping)
    5. VIF = diagonal of R^{-1}

Usage:
    python -m selection.vif_analysis
    python -m selection.vif_analysis --sample-rows 500000
    nohup python -u -m selection.vif_analysis > /dev/null 2>&1 &
    tail -f results/selection/logs/vif_analysis.log

Outputs (in results/selection/results/vif/):
    - {asset}_vif_results.csv
    - {asset}_vif_summary.csv         (tiered counts ≤5, 5–10, 10–50, >50)
    - {asset}_vif_high_features.csv   (VIF > 10)
"""

import signal
from common.paths import REDUCTION_DIR
signal.signal(signal.SIGHUP, signal.SIG_IGN)

import argparse
import glob
import logging
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


# ─── Configuration ───────────────────────────────────────────────────────────

BASE_DIR        = str(REDUCTION_DIR)
CATALOG         = os.path.join(BASE_DIR, "feature_catalog.csv")
# Plan-compliant: merged S5+S6 corpus, no drop_list dependency
DATA_DIR        = "data_storage/s6_features_s5_full"
OUTPUT_DIR      = os.path.join(BASE_DIR, "results", "vif")
LOG_DIR         = os.path.join(BASE_DIR, "logs")

ASSETS          = ["btc", "eth"]
USABILITY_COL   = "data_usability_flag"

# Pre-exclusion (3.4.1): non-microstructure feature groups
EXCLUDE_GROUPS  = {
    "Trend",
    "Level Artefact",
    "Session Levels",
    "Level Events",
    "Volume Profile",
    "Volume Profile Artefact",
}

DEFAULT_SAMPLE  = 500_000
SAMPLE_SEED     = 42
MIN_CORR_PERIODS = 500   # minimum overlapping non-NaN rows for valid correlation

# Additional safety net: even though is_feature filter handles meta/targets,
# these prefixes catch any edge cases (e.g. compat diagnostic columns)
EXCLUDE_PREFIXES = ("ret_", "data_health", "l2_coverage", "lob50_health",
                    "trades_coverage", "depth_lobdeep")


# ─── Logging ─────────────────────────────────────────────────────────────────

def setup_logging():
    os.makedirs(LOG_DIR, exist_ok=True)
    log = logging.getLogger("vif")
    log.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(os.path.join(LOG_DIR, "vif_analysis.log"),
                             mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG); fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO); ch.setFormatter(fmt)
    if not log.handlers:
        log.addHandler(fh); log.addHandler(ch)
    return log


# ─── Feature Identification ─────────────────────────────────────────────────

def get_features_for_asset(asset: str, log) -> tuple:
    """
    Returns (column_names, meta_df) for the VIF analysis of one asset.

    Per-asset logic:
      - Single-asset features: catalog[asset=asset] (e.g. asset='btc')
      - Cross-asset features:  catalog[asset='btceth'] (S6 shared)

    Returns:
        columns: list of suffixed names (as they appear in the merged Parquet)
        meta:    DataFrame with (column, bare_name, base_concept, group, ...)
    """
    cat = pd.read_csv(CATALOG)

    required = {"column", "bare_name", "is_feature", "stage", "group",
                "base_concept", "asset"}
    missing = required - set(cat.columns)
    if missing:
        raise RuntimeError(f"Catalog missing required columns: {missing}. "
                           f"Run extend_feature_catalog.py first.")

    before = len(cat)
    cat = cat[cat["is_feature"] == True].copy()
    log.info("Catalog: %d rows -> %d features (is_feature)", before, len(cat))

    cat = cat[~cat["group"].isin(EXCLUDE_GROUPS)].copy()
    log.info("After EXCLUDE_GROUPS: %d features", len(cat))

    # Per asset: single-asset features + S6 shared features
    cat = cat[cat["asset"].isin([asset, "btceth"])].copy()
    log.info("After asset filter (%s + btceth): %d features", asset, len(cat))

    # EXCLUDE_PREFIXES safety net (matches bare_name to keep regex simple)
    before_px = len(cat)
    cat = cat[~cat["bare_name"].apply(
        lambda x: isinstance(x, str) and any(x.startswith(px) for px in EXCLUDE_PREFIXES)
    )].copy()
    if len(cat) < before_px:
        log.info("After EXCLUDE_PREFIXES: %d features (%d removed)",
                 len(cat), before_px - len(cat))

    columns = cat["column"].tolist()
    meta = cat[["column", "bare_name", "base_concept", "group", "asset",
                "stage", "window_s", "market_scope"]].copy()

    log.info("Final feature list for %s: %d columns (incl. %d S6 shared)",
             asset, len(columns), (cat["asset"] == "btceth").sum())
    return columns, meta


# ─── Data Sampling ───────────────────────────────────────────────────────────

def stream_sample(features: list, max_rows: int, log) -> pd.DataFrame:
    """
    Sample rows from merged S5+S6 parquets.

    Files: data_storage/s6_features_s5_full/merged_btceth_*.parquet
    Each file contains all features (BTC + ETH + S6) for one (date, hour).
    """
    pattern = os.path.join(DATA_DIR, "merged_btceth_*.parquet")
    files = sorted(glob.glob(pattern))
    if not files:
        log.error("No files matching %s", pattern)
        return pd.DataFrame()

    log.info("Sampling %s rows from %d merged parquets...",
             f"{max_rows:,}", len(files))

    total_rows = 0
    file_rows = []
    for f in files:
        try:
            n = pq.read_metadata(f).num_rows
        except Exception:
            n = 0
        file_rows.append(n)
        total_rows += n

    sample_rate = min(1.0, max_rows / total_rows) if total_rows > 0 else 1.0
    log.info("  Total: %s rows across all files, sample rate: %.4f",
             f"{total_rows:,}", sample_rate)

    rng = np.random.RandomState(SAMPLE_SEED)
    sampled = []
    n_so_far = 0
    dup_files = 0
    features_set = set(features)

    for idx, (filepath, n_rows) in enumerate(zip(files, file_rows)):
        n_sample = max(1, int(n_rows * sample_rate))
        try:
            schema = pq.read_schema(filepath)
            schema_cols = schema.names

            # Detect true duplicate column names
            if len(schema_cols) != len(set(schema_cols)):
                dup_names = [c for c in set(schema_cols) if schema_cols.count(c) > 1]
                dup_files += 1
                if dup_files <= 3:
                    log.warning("  %s: duplicate column names: %s",
                                os.path.basename(filepath), dup_names[:5])

            schema_set = set(schema_cols)
            load_set = features_set & schema_set
            if USABILITY_COL in schema_set:
                load_set.add(USABILITY_COL)
            load_cols = sorted(load_set)

            if not load_cols:
                continue

            chunk = pd.read_parquet(filepath, columns=load_cols)

            # Deduplicate columns
            if chunk.columns.duplicated().any():
                chunk = chunk.loc[:, ~chunk.columns.duplicated(keep="first")]

            # Drop unusable rows
            if USABILITY_COL in chunk.columns:
                chunk = chunk[chunk[USABILITY_COL] == 1].drop(columns=[USABILITY_COL])

            if len(chunk) == 0:
                continue
            if len(chunk) > n_sample:
                chunk = chunk.sample(n=n_sample, random_state=rng)

            sampled.append(chunk)
            n_so_far += len(chunk)
            del chunk
        except Exception as e:
            log.warning("  Error %s: %s", os.path.basename(filepath), e)

        if (idx + 1) % 50 == 0 or idx == len(files) - 1:
            log.info("  [%d/%d] %s rows", idx + 1, len(files), f"{n_so_far:,}")

    if dup_files > 0:
        log.warning("  %d files had duplicate column names (deduplicated)", dup_files)

    if not sampled:
        return pd.DataFrame()

    data = pd.concat(sampled, ignore_index=True)
    log.info("  Final: %s rows × %d cols", f"{len(data):,}", data.shape[1])
    return data


# ─── Nearest Positive-Definite Projection ────────────────────────────────────

def nearest_positive_definite(R: np.ndarray, min_eig: float = 1e-6) -> np.ndarray:
    """
    Project correlation matrix to nearest positive-definite matrix.
    Clips negative eigenvalues to min_eig, then reconstructs.
    Preserves correlation structure (diagonal = 1).
    """
    # Eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(R)

    # Clip negative eigenvalues
    eigvals_clipped = np.maximum(eigvals, min_eig)

    # Reconstruct
    R_psd = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T

    # Symmetrize
    R_psd = (R_psd + R_psd.T) / 2

    # Force diagonal = 1 (re-normalize to correlation matrix)
    d = np.sqrt(np.diag(R_psd))
    d[d == 0] = 1.0
    R_psd = R_psd / np.outer(d, d)
    np.fill_diagonal(R_psd, 1.0)

    return R_psd


# ─── VIF Computation ─────────────────────────────────────────────────────────

def compute_vif(data: pd.DataFrame, features: list, log) -> pd.DataFrame:
    available = [f for f in features if f in data.columns]
    log.info("Computing correlation matrix: %d features × %s rows...",
             len(available), f"{len(data):,}")

    # Step 1: Correlation matrix (pairwise NaN-safe)
    t0 = time.time()
    R = data[available].corr(min_periods=MIN_CORR_PERIODS)
    log.info("  Correlation matrix: %.1fs", time.time() - t0)

    # Step 2: Remove features with no valid correlations
    nan_per_feat = R.isna().sum()
    bad_feats = nan_per_feat[nan_per_feat > len(available) * 0.5].index.tolist()
    if bad_feats:
        log.warning("  Removing %d features with >50%% NaN correlations: %s",
                    len(bad_feats), bad_feats[:5])
        available = [f for f in available if f not in bad_feats]
        R = R.loc[available, available]

    # Step 3: Fill remaining NaN with PAIRWISE approach
    # For remaining NaN: these are pairs where min_periods wasn't met
    # Set them to 0 (uncorrelated assumption — minimal distortion)
    n_nan = R.isna().sum().sum()
    if n_nan > 0:
        log.info("  Remaining NaN correlations: %d (%.3f%% of matrix) → filling with 0",
                 n_nan, 100 * n_nan / (len(available) ** 2))
        R = R.fillna(0.0)

    R_arr = R.values.copy()

    # Ensure symmetric and diagonal = 1
    R_arr = (R_arr + R_arr.T) / 2
    np.fill_diagonal(R_arr, 1.0)

    # Step 4: Sanity checks
    off_diag = R_arr[np.triu_indices_from(R_arr, k=1)]
    log.info("  Correlations: mean|r|=%.4f, median|r|=%.4f, max|r|=%.4f",
             np.mean(np.abs(off_diag)), np.median(np.abs(off_diag)),
             np.max(np.abs(off_diag)))
    log.info("  Pairs |r|>0.9: %d, |r|>0.95: %d, |r|>0.99: %d",
             (np.abs(off_diag) > 0.9).sum(),
             (np.abs(off_diag) > 0.95).sum(),
             (np.abs(off_diag) > 0.99).sum())

    # Step 5: Eigenvalue analysis BEFORE projection
    eigvals_raw = np.linalg.eigvalsh(R_arr)
    log.info("  Eigenvalues (raw): min=%.4e, max=%.4e, n_negative=%d",
             eigvals_raw.min(), eigvals_raw.max(), (eigvals_raw < 0).sum())

    # Step 6: Project to nearest positive-definite matrix
    if eigvals_raw.min() < 1e-6:
        log.info("  Projecting to nearest positive-definite matrix (eigenvalue clipping)...")
        R_psd = nearest_positive_definite(R_arr, min_eig=1e-4)

        eigvals_new = np.linalg.eigvalsh(R_psd)
        log.info("  Eigenvalues (after PSD): min=%.4e, max=%.4e",
                 eigvals_new.min(), eigvals_new.max())

        # Show how much projection changed the matrix
        diff = np.abs(R_psd - R_arr)
        log.info("  PSD projection distortion: mean=%.6f, max=%.6f", diff.mean(), diff.max())
    else:
        R_psd = R_arr
        log.info("  Matrix already positive-definite — no projection needed")

    # Step 7: Invert
    p = R_psd.shape[0]
    log.info("  Inverting %d×%d matrix...", p, p)
    t0 = time.time()
    try:
        R_inv = np.linalg.inv(R_psd)
        vif = np.diag(R_inv)
    except np.linalg.LinAlgError:
        log.warning("  Inversion failed — using pseudo-inverse")
        R_inv = np.linalg.pinv(R_psd)
        vif = np.diag(R_inv)
    log.info("  Inversion: %.1fs", time.time() - t0)

    # Clip negatives (should not happen after PSD, but safety)
    n_neg = (vif < 1.0).sum()
    if n_neg > 0:
        log.info("  Clipping %d VIF values < 1.0 to 1.0", n_neg)
    vif = np.maximum(vif, 1.0)

    # Step 8: VIF sanity check
    log.info("  VIF: min=%.2f, median=%.2f, mean=%.2f, max=%.2f",
             vif.min(), np.median(vif), vif.mean(), vif.max())
    log.info("  VIF>5: %d, VIF>10: %d, VIF>50: %d, VIF>100: %d",
             (vif > 5).sum(), (vif > 10).sum(), (vif > 50).sum(), (vif > 100).sum())

    df = pd.DataFrame({
        "column": available,
        "vif": np.round(vif, 4),
    }).sort_values("vif", ascending=False).reset_index(drop=True)

    return df


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="VIF Analysis v4 (merged corpus)")
    parser.add_argument("--assets", nargs="+", default=ASSETS)
    parser.add_argument("--sample-rows", type=int, default=DEFAULT_SAMPLE)
    parser.add_argument("--top", type=int, default=50)
    args = parser.parse_args()

    log = setup_logging()
    t0 = time.time()

    log.info("=" * 60)
    log.info("VIF ANALYSIS v4 (merged S5+S6 corpus, PSD-projected)")
    log.info("=" * 60)
    log.info("Data dir:    %s", DATA_DIR)
    log.info("Assets:      %s", args.assets)
    log.info("Sample rows: %s", f"{args.sample_rows:,}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Build per-asset feature lists, then a union for shared sampling
    log.info("")
    log.info("─── FEATURE IDENTIFICATION ──────────────────────────")
    asset_features = {}
    asset_meta     = {}
    for asset in args.assets:
        log.info("")
        log.info("  Asset: %s", asset.upper())
        cols, meta = get_features_for_asset(asset, log)
        asset_features[asset] = cols
        asset_meta[asset] = meta

    union_features = sorted(set().union(*asset_features.values()))
    log.info("")
    log.info("Union feature set across assets: %d columns", len(union_features))

    # Single shared sample (read each parquet once, load union of cols)
    log.info("")
    log.info("─── DATA SAMPLING (shared across assets) ─────────────")
    data = stream_sample(union_features, args.sample_rows, log)
    if data.empty:
        log.error("Empty sample — aborting")
        return

    # Per-asset VIF computation
    for asset in args.assets:
        log.info("")
        log.info("═" * 60)
        log.info("  VIF for %s (single-asset + S6 shared)", asset.upper())
        log.info("═" * 60)

        cols = asset_features[asset]
        meta = asset_meta[asset]

        # Restrict the sample to this asset's columns (subset, no reload)
        cols_in_data = [c for c in cols if c in data.columns]
        log.info("  Asset cols in sample: %d / %d", len(cols_in_data), len(cols))
        if not cols_in_data:
            log.warning("  No columns available — skipping %s", asset)
            continue

        vif_df = compute_vif(data, cols_in_data, log)

        # Enrich with metadata
        vif_df = vif_df.merge(meta, on="column", how="left")

        # Summary tiers (≤5, 5-10, 10-50, >50)
        tiers = [
            ("vif_le_5",  vif_df["vif"] <= 5),
            ("vif_5_10",  (vif_df["vif"] > 5) & (vif_df["vif"] <= 10)),
            ("vif_10_50", (vif_df["vif"] > 10) & (vif_df["vif"] <= 50)),
            ("vif_gt_50", vif_df["vif"] > 50),
        ]
        summary_rows = []
        for name, mask in tiers:
            n = int(mask.sum())
            pct = round(100 * n / len(vif_df), 1)
            summary_rows.append({"tier": name, "n_features": n, "pct": pct})
            log.info("  %-15s %4d features (%5.1f%%)", name, n, pct)
        summary_df = pd.DataFrame(summary_rows)

        high_vif = vif_df[vif_df["vif"] > 10].copy()

        # Save
        vif_df.to_csv(os.path.join(OUTPUT_DIR, f"{asset}_vif_results.csv"), index=False)
        summary_df.to_csv(os.path.join(OUTPUT_DIR, f"{asset}_vif_summary.csv"), index=False)
        log.info("Saved: %s_vif_results.csv (%d features)", asset, len(vif_df))

        if len(high_vif) > 0:
            high_vif.to_csv(os.path.join(OUTPUT_DIR, f"{asset}_vif_high_features.csv"), index=False)
            log.info("Saved: %s_vif_high_features.csv (%d)", asset, len(high_vif))

        show = ["column", "vif", "base_concept", "stage", "window_s", "market_scope"]
        show = [c for c in show if c in vif_df.columns]
        log.info("")
        log.info("TOP %d HIGHEST VIF:", args.top)
        log.info("\n%s", vif_df[show].head(args.top).to_string(index=False))
        log.info("")
        log.info("LOWEST 10:")
        log.info("\n%s", vif_df[show].tail(10).to_string(index=False))

        if len(high_vif) > 0:
            log.info("")
            log.info("HIGH VIF (>10) BY BASE_CONCEPT:")
            by_c = high_vif.groupby("base_concept").agg(
                n=("vif", "count"), mean=("vif", "mean"), max=("vif", "max")
            ).sort_values("max", ascending=False)
            log.info("\n%s", by_c.head(20).to_string())

    log.info("")
    log.info("Done in %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()