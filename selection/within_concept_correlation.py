#!/usr/bin/env python3
"""
within_concept_correlation.py  (multi-asset)
=======================================
Runs the within-group correlation analysis separately for several assets.
Each asset gets its own output files under:
  <base_output_dir>/<asset>/...

Configuration via CLI or by adjusting the ASSETS dict.

Usage:
  python within_concept_correlation.py                     # BTC + ETH
  python -m selection.within_concept_correlation --assets btc   # BTC only
  python -m selection.within_concept_correlation --assets btc eth   # both explicitly
"""

import signal
signal.signal(signal.SIGHUP, signal.SIG_IGN)

import argparse
import glob
import logging
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


# ═══════════════════════════════════════════════════════════════════════════════
# ASSET CONFIG
# Per asset: path to the S5 parquet folder, catalog, and output dir.
# ═══════════════════════════════════════════════════════════════════════════════

def asset_config(asset: str, base_catalog: str) -> dict:
    """
    Builds the asset-specific configuration.
    base_catalog: shared catalog (or an asset-specific path if needed).
    """
    a = asset.lower()
    return {
        "asset":          a,
        "catalog_path":   base_catalog,
        # Asset-specific glob — cleanly separates btc_ and eth_ parquet files
        "parquet_glob":   f"data_storage/s5_features/s5_features_{a}_*.parquet",
        # All output CSVs land together in results/
        "output_dir":     "results/selection/results/within_concept_correlation",
        "log_path":       f"results/selection/logs/{a}_within_concept_correlation.log",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SHARED SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_CATALOG          = "results/selection/feature_catalog.csv"
USABILITY_COL            = "data_usability_flag"
# Pre-exclusion (3.4.1): non-microstructure feature groups are excluded from the
# within-concept correlation analysis. They stay in the feature set
# for Phase B (feature importance) and Phase C (ML), but contribute no
# methodologically meaningful contribution to the microstructure redundancy diagnostics.
EXCLUDE_GROUPS           = {
    "Trend",                    # EMAs — trend context
    "Level Artefact",           # Day/Week/Month High/Low — structural anchor
    "Session Levels",           # dist_to_*_high/low — structural metrics
    "Level Events",             # broke_prev_* — structural events
    "Volume Profile",           # POC, VAH, VAL
    "Volume Profile Artefact",
}
HIGH_CORR_THRESHOLD      = 0.95
MODERATE_CORR_THRESHOLD  = 0.85
MIN_VALID_ROWS           = 3600


# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING  (asset-aware)
# ═══════════════════════════════════════════════════════════════════════════════

def setup_logging(asset: str, log_path: str) -> logging.Logger:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log = logging.getLogger(f"corr_explorer.{asset}")
    log.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        f"%(asctime)s  [{asset.upper()}]  %(levelname)-7s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    if not log.handlers:
        log.addHandler(fh)
        log.addHandler(ch)
    return log


# ═══════════════════════════════════════════════════════════════════════════════
# CATALOG
# ═══════════════════════════════════════════════════════════════════════════════

def load_catalog(catalog_path: str, log: logging.Logger) -> pd.DataFrame:
    log.info("Loading feature catalog from %s", catalog_path)
    df = pd.read_csv(catalog_path)

    # Schema sanity: required columns
    required = {"bare_name", "is_feature", "stage", "group", "base_concept"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Catalog missing required columns: {missing}. "
                           f"Run extend_feature_catalog.py first.")

    # 1) Only features (excl. is_meta/is_target)
    before_feat = len(df)
    df = df[df["is_feature"] == True].copy()
    log.info("Catalog: %d rows -> %d features (filtered is_feature)",
             before_feat, len(df))

    # 2) Only stages S0-S6
    df = df[df["stage"].isin(["S0", "S1", "S2", "S3", "S4", "S5", "S6"])].copy()

    # 3) Pre-exclusion: non-microstructure groups
    before_grp = len(df)
    df = df[~df["group"].isin(EXCLUDE_GROUPS)].copy()
    excluded = before_grp - len(df)
    log.info("Pre-exclusion: %d features removed (groups=%s)", excluded, EXCLUDE_GROUPS)

    # 4) Deduplicate on bare_name (should not occur within one asset,
    #    but defensive)
    dupes = df[df.duplicated(subset=["bare_name", "asset"], keep=False)]
    if len(dupes) > 0:
        log.warning("Catalog: %d duplicate (bare_name, asset) tuples — deduplicating", len(dupes) // 2)
        df = df.drop_duplicates(subset=["bare_name", "asset"], keep="first")

    for col in ("depth_band", "window_s", "market_scope"):
        if col in df.columns:
            df[col] = df[col].fillna("")

    log.info("Final catalog: %d features, %d unique base_concepts",
             len(df), df["base_concept"].nunique())
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# PRE-FLIGHT
# ═══════════════════════════════════════════════════════════════════════════════

def preflight_column_check(catalog: pd.DataFrame, parquet_files: list, log) -> set:
    log.info("Pre-flight: scanning %d parquet schemas...", len(parquet_files))
    t0 = time.time()
    all_parquet_cols = set()
    for idx, f in enumerate(parquet_files, 1):
        try:
            all_parquet_cols.update(pq.read_schema(f).names)
        except Exception as e:
            log.warning("  Schema read failed: %s: %s", os.path.basename(f), e)
        if idx % 200 == 0 or idx == len(parquet_files):
            log.info("  Schema scan: %d/%d  (%.1fs)", idx, len(parquet_files), time.time() - t0)

    catalog_features = set(catalog["bare_name"].tolist())
    found   = catalog_features & all_parquet_cols
    missing = catalog_features - all_parquet_cols

    log.info("Pre-flight: %d/%d catalog features found in parquet (%.1f%%)",
             len(found), len(catalog_features),
             100 * len(found) / len(catalog_features) if catalog_features else 0)
    if missing:
        log.warning("  %d features NOT found: %s%s", len(missing),
                    sorted(missing)[:5], " ..." if len(missing) > 5 else "")
    if len(found) == 0:
        log.error("FATAL: no catalog features found in parquet files for this asset!")
        return set()
    return found


# ═══════════════════════════════════════════════════════════════════════════════
# ONLINE PEARSON
# ═══════════════════════════════════════════════════════════════════════════════

class OnlineCorr:
    def __init__(self, feature_names: list):
        self.features = feature_names
        n             = len(feature_names)
        self._n       = n
        pairs         = n * (n - 1) // 2
        self.n_xy     = np.zeros(pairs, dtype=np.int64)
        self.sum_x    = np.zeros(pairs, dtype=np.float64)
        self.sum_y    = np.zeros(pairs, dtype=np.float64)
        self.sum_x2   = np.zeros(pairs, dtype=np.float64)
        self.sum_y2   = np.zeros(pairs, dtype=np.float64)
        self.sum_xy   = np.zeros(pairs, dtype=np.float64)

    def update(self, chunk: pd.DataFrame):
        n, features = self._n, self.features
        nrows = len(chunk)
        vals  = np.full((nrows, n), np.nan, dtype=np.float64)
        for i, f in enumerate(features):
            if f in chunk.columns:
                vals[:, i] = chunk[f].to_numpy(dtype=np.float64, na_value=np.nan)
        valid = ~np.isnan(vals)
        k = 0
        for i in range(n):
            vi = valid[:, i]
            if not vi.any():
                k += n - i - 1
                continue
            xi = vals[:, i]
            for j in range(i + 1, n):
                mask = vi & valid[:, j]
                if mask.any():
                    x = xi[mask]; y = vals[mask, j]
                    self.n_xy[k]   += len(x)
                    self.sum_x[k]  += x.sum()
                    self.sum_y[k]  += y.sum()
                    self.sum_x2[k] += np.dot(x, x)
                    self.sum_y2[k] += np.dot(y, y)
                    self.sum_xy[k] += np.dot(x, y)
                k += 1

    def compute(self, min_valid: int = MIN_VALID_ROWS):
        results, features, n = [], self.features, self._n
        k = 0
        for i in range(n):
            for j in range(i + 1, n):
                cnt = self.n_xy[k]
                if cnt >= min_valid:
                    sx, sy   = self.sum_x[k], self.sum_y[k]
                    sx2, sy2 = self.sum_x2[k], self.sum_y2[k]
                    sxy      = self.sum_xy[k]
                    denom = np.sqrt(
                        max(cnt * sx2 - sx * sx, 0.0) *
                        max(cnt * sy2 - sy * sy, 0.0)
                    )
                    if denom > 0:
                        r = float(np.clip((cnt * sxy - sx * sy) / denom, -1.0, 1.0))
                        if features[i] != features[j]:
                            results.append((features[i], features[j], r))
                k += 1
        return results


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLE-PASS
# ═══════════════════════════════════════════════════════════════════════════════

def run_online_correlation(reducible: dict, parquet_files: list, log) -> dict:
    accumulators = {name: OnlineCorr(grp["bare_name"].tolist())
                    for name, grp in reducible.items()}
    all_needed = set()
    for grp in reducible.values():
        all_needed.update(grp["bare_name"].tolist())
    all_needed.add(USABILITY_COL)
    group_cols = {name: set(grp["bare_name"].tolist()) for name, grp in reducible.items()}

    n, t0 = len(parquet_files), time.time()
    total_raw = total_usable = 0
    log.info("Single-pass: %d files, %d groups", n, len(reducible))

    for idx, f in enumerate(parquet_files, 1):
        try:
            available    = set(pq.read_schema(f).names)
            cols_to_load = list(all_needed & available)
            if not cols_to_load:
                continue
            chunk = pd.read_parquet(f, columns=cols_to_load)
            total_raw += len(chunk)
            if USABILITY_COL in chunk.columns:
                chunk = chunk[chunk[USABILITY_COL] == 1].drop(columns=[USABILITY_COL])
            total_usable += len(chunk)
            if chunk.empty:
                continue
            chunk_cols = set(chunk.columns)
            for name, acc in accumulators.items():
                present = list(group_cols[name] & chunk_cols)
                if len(present) >= 2:
                    acc.update(chunk[present])
        except Exception as e:
            log.warning("  [%d/%d] skipping %s: %s", idx, n, os.path.basename(f), e)

        if idx % 50 == 0 or idx == n:
            elapsed = time.time() - t0
            rate    = idx / elapsed if elapsed > 0 else 0
            eta     = (n - idx) / rate if rate > 0 else 0
            log.info("  [%d/%d]  %s raw / %s usable  |  %.1fs  ETA ~%.0fs",
                     idx, n, f"{total_raw:,}", f"{total_usable:,}", elapsed, eta)

    pct = (1 - total_usable / total_raw) * 100 if total_raw > 0 else 0
    log.info("Pass done: %s raw -> %s usable (%.1f%% filtered)  |  %.1fs",
             f"{total_raw:,}", f"{total_usable:,}", pct, time.time() - t0)

    return {name: acc.compute() for name, acc in accumulators.items()}


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_group(group_name, group_features, pairs):
    if not pairs:
        return None, [], []
    pairs = [(fa, fb, r) for fa, fb, r in pairs if fa != fb]
    if not pairs:
        return None, [], []

    abs_corrs = [abs(r) for _, _, r in pairs]
    summary = {
        "base_concept":    group_name,
        "total_features":  len(group_features),
        "usable_features": len({f for p in pairs for f in (p[0], p[1])}),
        "num_pairs":       len(pairs),
        "mean_abs_corr":   float(np.mean(abs_corrs)),
        "median_abs_corr": float(np.median(abs_corrs)),
        "max_abs_corr":    float(np.max(abs_corrs)),
        "min_abs_corr":    float(np.min(abs_corrs)),
        "pairs_above_095": sum(1 for c in abs_corrs if c > 0.95),
        "pairs_above_085": sum(1 for c in abs_corrs if c > 0.85),
        "pct_above_095":   sum(1 for c in abs_corrs if c > 0.95) / len(abs_corrs) * 100,
    }

    high_pairs = [{"base_concept": group_name, "feature_a": fa, "feature_b": fb,
                   "correlation": r, "abs_correlation": abs(r)}
                  for fa, fb, r in pairs if abs(r) > MODERATE_CORR_THRESHOLD]

    feature_meta = group_features.set_index("bare_name")
    axis_records = []
    for fa, fb, r in pairs:
        if fa not in feature_meta.index or fb not in feature_meta.index:
            continue
        ma, mb = feature_meta.loc[fa], feature_meta.loc[fb]
        diff_axes = [ax for ax in ["depth_band", "window_s", "market_scope"]
                     if str(ma.get(ax, "")) != str(mb.get(ax, ""))]
        same_axes = [ax for ax in ["depth_band", "window_s", "market_scope"]
                     if str(ma.get(ax, "")) == str(mb.get(ax, ""))]
        axis_records.append({
            "base_concept":    group_name,
            "feature_a": fa,   "feature_b": fb,
            "correlation": r,  "abs_correlation": abs(r),
            "differs_on": ",".join(sorted(diff_axes)) if diff_axes else "same_variant",
            "same_on":    ",".join(sorted(same_axes)),
        })
    return summary, high_pairs, axis_records


def greedy_drop_candidates(group_name, pairs_df, threshold):
    high = pairs_df[pairs_df["abs_correlation"] > threshold].copy()
    high = high[high["feature_a"] != high["feature_b"]]
    if high.empty:
        return []
    freq  = pd.concat([high["feature_a"], high["feature_b"]]).value_counts()
    drops = set()
    for _, row in high.sort_values("abs_correlation", ascending=False).iterrows():
        fa, fb = row["feature_a"], row["feature_b"]
        if fa in drops or fb in drops:
            continue
        drops.add(fa if freq.get(fa, 0) >= freq.get(fb, 0) else fb)
    return [{"base_concept": group_name, "feature_name": f, "threshold": threshold}
            for f in drops]


# ═══════════════════════════════════════════════════════════════════════════════
# RUN ONE ASSET
# ═══════════════════════════════════════════════════════════════════════════════

def run_asset(cfg: dict, shared_catalog: pd.DataFrame | None = None):
    asset      = cfg["asset"]
    output_dir = cfg["output_dir"]
    log        = setup_logging(asset, cfg["log_path"])

    t0 = time.time()
    log.info("=" * 65)
    log.info("CORRELATION EXPLORER — asset=%s — %s",
             asset.upper(), datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    log.info("=" * 65)

    os.makedirs(output_dir, exist_ok=True)

    # Catalog: use the passed-in one or reload (may be per asset)
    if shared_catalog is not None:
        catalog = shared_catalog.copy()
        log.info("Using shared catalog (%d features across all assets)", len(catalog))
    else:
        catalog = load_catalog(cfg["catalog_path"], log)

    # Filter to the current asset
    before_asset = len(catalog)
    catalog = catalog[catalog["asset"] == asset].copy()
    log.info("Catalog filtered to asset=%s: %d -> %d features",
             asset, before_asset, len(catalog))

    parquet_files = sorted(glob.glob(cfg["parquet_glob"]))
    if not parquet_files:
        log.error("No parquet files matching %s — SKIPPING asset %s",
                  cfg["parquet_glob"], asset.upper())
        return

    log.info("Found %d S5 parquet files", len(parquet_files))

    found_cols       = preflight_column_check(catalog, parquet_files, log)
    if not found_cols:
        log.error("No catalog features found in parquet — SKIPPING %s", asset.upper())
        return

    catalog_filtered = catalog[catalog["bare_name"].isin(found_cols)].copy()
    log.info("Catalog after pre-flight: %d features", len(catalog_filtered))

    groups_filtered = catalog_filtered.groupby("base_concept")

    reducible  = {n: g for n, g in groups_filtered if len(g) >= 2}
    singletons = {n: g for n, g in groups_filtered if len(g) == 1}
    log.info("Reducible groups (>=2 features): %d  |  Singletons: %d",
             len(reducible), len(singletons))

    if not reducible:
        log.error("No reducible groups — check catalog base_concept column")
        return

    group_results = run_online_correlation(reducible, parquet_files, log)

    log.info("Building summaries for %d groups...", len(reducible))
    all_summaries = []; all_high_pairs = []; all_axis_records = []
    no_pairs_count = 0

    for i, (name, grp) in enumerate(sorted(reducible.items())):
        pairs = [(fa, fb, r) for fa, fb, r in group_results.get(name, []) if fa != fb]
        if not pairs:
            no_pairs_count += 1
            continue
        summary, high_pairs, axis_records = analyze_group(name, grp, pairs)
        if summary:
            all_summaries.append(summary)
            all_high_pairs.extend(high_pairs)
            all_axis_records.extend(axis_records)
        log.info("  [%d/%d] %-42s  %d pairs  max_r=%.3f  (%.0f%% > 0.95)",
                 i+1, len(reducible), name, len(pairs),
                 max(abs(r) for _, _, r in pairs),
                 summary["pct_above_095"] if summary else 0)

    summary_df    = (pd.DataFrame(all_summaries).sort_values("pct_above_095", ascending=False)
                     if all_summaries else pd.DataFrame())
    high_pairs_df = (pd.DataFrame(all_high_pairs).sort_values("abs_correlation", ascending=False)
                     if all_high_pairs else pd.DataFrame())
    axis_df       = pd.DataFrame(all_axis_records) if all_axis_records else pd.DataFrame()

    drop_095, drop_085 = [], []
    for name in reducible:
        if high_pairs_df.empty: break
        grp_pairs = high_pairs_df[high_pairs_df["base_concept"] == name]
        if not grp_pairs.empty:
            drop_095.extend(greedy_drop_candidates(name, grp_pairs, 0.95))
            drop_085.extend(greedy_drop_candidates(name, grp_pairs, 0.85))

    drop_095_df = pd.DataFrame(drop_095) if drop_095 \
        else pd.DataFrame(columns=["base_concept", "feature_name", "threshold"])
    drop_085_df = pd.DataFrame(drop_085) if drop_085 \
        else pd.DataFrame(columns=["base_concept", "feature_name", "threshold"])

    if not high_pairs_df.empty:
        sp = (high_pairs_df["feature_a"] == high_pairs_df["feature_b"]).sum()
        if sp > 0:
            log.error("BUG: %d self-pairs still in output!", sp)
        else:
            log.info("Zero self-pairs in pairwise output")

    axis_summary = pd.DataFrame()
    if not axis_df.empty:
        axis_summary = (axis_df.groupby("differs_on")["abs_correlation"]
                        .agg(["mean", "median", "count"])
                        .sort_values("mean", ascending=False))

    total_signal = len(catalog)
    n_high_095   = (len(high_pairs_df[high_pairs_df["abs_correlation"] > 0.95])
                    if not high_pairs_df.empty else 0)
    n_high_085   = len(high_pairs_df) if not high_pairs_df.empty else 0

    log.info("=" * 70)
    log.info("REPORT — %s", asset.upper())
    log.info("=" * 70)
    log.info("Signal features:          %d", total_signal)
    log.info("Found in parquet:         %d", len(found_cols))
    log.info("Groups with valid pairs:  %d", len(all_summaries))
    log.info("Total feature pairs:      %s", f"{len(all_axis_records):,}")
    log.info("Pairs |rho| > 0.95:       %s", f"{n_high_095:,}")
    log.info("Pairs |rho| > 0.85:       %s", f"{n_high_085:,}")
    log.info("Drop candidates 0.95:     %d  (retain %d = %.0f%%)",
             len(drop_095_df), total_signal - len(drop_095_df),
             (1 - len(drop_095_df)/total_signal)*100 if total_signal else 0)
    log.info("Drop candidates 0.85:     %d  (retain %d = %.0f%%)",
             len(drop_085_df), total_signal - len(drop_085_df),
             (1 - len(drop_085_df)/total_signal)*100 if total_signal else 0)

    if not axis_summary.empty:
        log.info("AXIS MEAN CORRELATION:\n%s", axis_summary.to_string())

    if not summary_df.empty:
        top = summary_df.head(15)[["base_concept", "usable_features",
                                    "mean_abs_corr", "max_abs_corr", "pct_above_095"]]
        log.info("TOP 15 MOST REDUNDANT:\n%s", top.to_string(index=False, float_format="%.3f"))

    log.info("Saving output files to %s ...", output_dir)
    for df, name in [
        (summary_df,    f"{asset}_group_correlation_summary.csv"),
        (high_pairs_df, f"{asset}_pairwise_high_corr.csv"),
        (axis_df,       f"{asset}_axis_correlation_report.csv"),
        (drop_095_df,   f"{asset}_drop_candidates_095.csv"),
        (drop_085_df,   f"{asset}_drop_candidates_085.csv"),
    ]:
        if not df.empty or "drop_candidates" in name:
            p = os.path.join(output_dir, name)
            df.to_csv(p, index=False)
            log.info("  Saved: %s", p)

    log.info("DONE %s — %.1fs", asset.upper(), time.time() - t0)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Multi-Asset Correlation Explorer")
    parser.add_argument(
        "--assets", nargs="+", default=["btc", "eth"],
        help="Assets to process (default: btc eth)"
    )
    parser.add_argument(
        "--catalog", default=DEFAULT_CATALOG,
        help="Shared feature catalog CSV (default: results/selection/feature_catalog.csv)"
    )
    parser.add_argument(
        "--shared-catalog", action="store_true", default=True,
        help="Load catalog once and share across assets (default: True)"
    )
    args = parser.parse_args()

    print(f"\nAssets: {args.assets}")
    print(f"Catalog: {args.catalog}\n")

    # Load the catalog once if shared
    shared_catalog = None
    if args.shared_catalog:
        dummy_log = logging.getLogger("catalog_loader")
        dummy_log.addHandler(logging.StreamHandler(sys.stdout))
        dummy_log.setLevel(logging.INFO)
        shared_catalog = load_catalog(args.catalog, dummy_log)

    for asset in args.assets:
        cfg = asset_config(asset, args.catalog)
        run_asset(cfg, shared_catalog=shared_catalog)
        print()

    print("All assets processed.")


if __name__ == "__main__":
    main()