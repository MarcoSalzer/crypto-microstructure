#!/usr/bin/env python3
"""
cross_asset_correlation_explorer.py
==========================
Cross-Asset (S6) Correlation Sanity Check

Validates S6 features by checking:
  A. S6 intra-correlation — are S6 features redundant with each other?
  B. S6 ↔ S5 cross-correlation — do S6 features duplicate surviving S5 features?

NOTE: Module C (BNB pair redundancy) is excluded — BNB is out of scope for
this thesis (only BTC and ETH considered).

Input:
  - s5_features/s5_features_{btc,eth}_{date}_{hh}.parquet  (FULL, unreduced)
  - s6_features_btceth/s6_features_btceth_{date}_{hh}.parquet

Output (in results/cross_asset_correlation_explorer/):
  - s6_intra_group_summary.csv
  - s6_intra_pairwise_high_corr.csv
  - s6_intra_drop_candidates.csv
  - {asset}_s6_vs_s5_pairwise.csv
  - {asset}_s6_vs_s5_family_summary.csv
  - s6_vs_s5_consensus_flags.csv

Usage:
  python cross_asset_correlation_explorer.py
  python cross_asset_correlation_explorer.py --s5-dir data_storage/s5_features
  python cross_asset_correlation_explorer.py --assets btc         # S6↔S5 for BTC only

  # Background:
  nohup python -u cross_asset_correlation_explorer.py > /dev/null 2>&1 &
  tail -f results/selection/logs/cross_asset_correlation_explorer.log
"""

import signal
from common.paths import REDUCTION_DIR
signal.signal(signal.SIGHUP, signal.SIG_IGN)

import argparse
import glob
import logging
import os
import re
import sys
import time
from datetime import datetime
from itertools import combinations

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

BASE_DIR        = str(REDUCTION_DIR)
DEFAULT_CATALOG = f"{BASE_DIR}/feature_catalog.csv"
OUTPUT_DIR      = f"{BASE_DIR}/results/cross_asset_correlation_explorer"
LOG_DIR         = f"{BASE_DIR}/logs"

# S5 full corpus (unreduced) — as specified for Phase A
S5_DIR_DEFAULT  = "data_storage/s5_features"
# S6 features — BTC/ETH only, no BNB pair variants
S6_DIR_DEFAULT  = "data_storage/s6_features_btceth"

S5_GLOB_TEMPLATE = "s5_features_{asset}_*.parquet"       # within S5_DIR

# Only btceth — btcethbnb out of scope
S6_GLOB_PATTERNS = [
    "s6_features_btceth_*.parquet",
]

USABILITY_COL           = "data_usability_flag"
# Pre-exclusion (3.4.1): non-microstructure feature groups
EXCLUDE_GROUPS          = {
    "Trend",
    "Level Artefact",
    "Session Levels",
    "Level Events",
    "Volume Profile",
    "Volume Profile Artefact",
}
HIGH_CORR_THRESHOLD     = 0.70    # store pairs above this
DROP_THRESHOLD          = 0.95    # suggest drops above this
MIN_VALID_ROWS          = 3600    # minimum valid row overlap

# ═══════════════════════════════════════════════════════════════════════════════
# DOMAIN-GUIDED S6 ↔ S5 FAMILIES
# ═══════════════════════════════════════════════════════════════════════════════
# Maps S6 stems to their related S5 stems. Matching is PREFIX-based:
# a stem matches any base_concept where bc == stem or bc.startswith(stem + '_').
# This covers depth variants automatically (e.g. 'net_pressure' matches
# 'net_pressure_5bps', 'net_pressure_10bps', 'net_pressure_persist_5bps', ...).
# Coverage verified against actual catalog: 45/45 S6 base_concepts covered.

S6_VS_S5_FAMILIES = {
    "taker_flow": {
        "s6": ["ca_taker_imb"],
        "s5": ["taker_imbalance", "z_taker_imbalance", "median_taker_imbalance"],
    },
    "pressure_flow": {
        "s6": ["ca_net_pressure"],
        "s5": ["net_pressure"],
    },
    "queue_pressure": {
        "s6": ["ca_queue_pressure"],
        "s5": ["queue_pressure", "z_queue_pressure", "queue_pressure_log"],
    },
    "absorption": {
        "s6": ["ca_absorb_refill", "ca_absorb_break"],
        "s5": ["z_absorb_refill_bid", "z_absorb_refill_ask",
               "absorption_break_flag", "absorption_volume"],
    },
    "impact": {
        "s6": ["ca_impact"],
        "s5": ["impact_per_liquidity", "impact_per_signed",
               "mad_impact_per_signed", "median_impact_per_signed"],
    },
    "depth_shape": {
        "s6": ["ca_depth_slope", "ca_depth_curvature", "ca_z_depth"],
        "s5": ["depth_gradient", "depth_imbalance"],
    },
    "vacuum_pull_refill": {
        "s6": ["ca_vacuum", "ca_pull_rate", "ca_refill_vs_pull"],
        "s5": ["vacuum_score", "pull_rate", "refill_rate", "refill_vs_pull"],
    },
    "order_flow": {
        "s6": ["ca_add_minus_cancel", "ca_net_add", "ca_net_cancel"],
        "s5": ["add_rate", "cancel_rate", "net_add"],
    },
    "microprice_spread": {
        "s6": ["ca_microprice_dev", "ca_spread_bps", "ca_bps_mid_dev",
               "ca_bps_spread", "ca_basis_vwap"],
        "s5": ["lwp_mid", "z_lwp_minus_mid", "basis_vwap"],
    },
    "regime": {
        "s6": ["ca_regime"],
        "s5": ["breakout_regime"],
    },
    "lead_lag": {
        "s6": ["ca_lag_corr"],
        "s5": ["taker_imbalance"],
    },
    "activity_xa": {
        "s6": ["ca_activity", "ca_z_trade_count", "ca_z_avg_trade_size"],
        "s5": ["trade_count", "avg_trade_size"],
    },
    "range_dist": {
        "s6": ["ca_day_range", "ca_range_pct", "ca_range_pos_day",
               "ca_dist_to_day"],
        "s5": ["day_range_bps", "range_pct", "range_pos_day",
               "dist_to_day_high", "dist_to_day_low"],
    },
}

# Top S5 features (from feature importance R3) for broad discovery check.
# ALL S6 features are correlated against these regardless of family.
TOP_S5_FEATURES = [
    "queue_pressure_log_fut_1bps_1s",
    "depth_notional_bid_1bps_fut_1s",
    "depth_notional_ask_1bps_fut_1s",
    "z_queue_pressure_fut_1bps_1s",
    "taker_imbalance_fut_1s",
    "range_pos_day_fut",
    "dist_to_day_high_bps_fut",
    "dist_to_day_low_bps_fut",
    "trade_count_spot_300s",
    "best_ask_spot_1s",
    "range_pct_spot_900s",
    "avg_trade_size_spot_900s",
    "basis_vwap_sf_1s",
    "z_taker_imbalance_spot_15s",
    "z_taker_imbalance_fut_15s",
    "net_pressure_robust_z_spot_5bps_60s",
    "net_pressure_robust_z_fut_5bps_60s",
    "liq_imb_sf_1bps_1s",
    "book_asymmetry_1bps_1s",
    "ofi_shock_15s",
]

# Pair tags for redundancy check
PAIR_TAGS = ["btceth", "btcbnb", "ethbnb"]


# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

def setup_logging() -> logging.Logger:
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, "cross_asset_correlation_explorer.log")

    log = logging.getLogger("ca_corr_explorer")
    log.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s  [CA-CORR]  %(levelname)-7s  %(message)s",
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
    log.info("Loading catalog from %s", catalog_path)
    df = pd.read_csv(catalog_path)

    required = {"bare_name", "is_feature", "stage", "group", "base_concept", "asset"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Catalog missing required columns: {missing}. "
                           f"Run extend_feature_catalog.py first.")

    before_feat = len(df)
    df = df[df["is_feature"] == True].copy()
    log.info("Catalog: %d rows -> %d features (filtered is_feature)",
             before_feat, len(df))

    df = df[df["stage"].isin(["S0", "S1", "S2", "S3", "S4", "S5", "S6"])].copy()

    before_grp = len(df)
    df = df[~df["group"].isin(EXCLUDE_GROUPS)].copy()
    log.info("Pre-exclusion: %d features removed (groups=%s)",
             before_grp - len(df), EXCLUDE_GROUPS)

    for col in ("depth_band", "window_s", "market_scope", "base_concept"):
        if col in df.columns:
            df[col] = df[col].fillna("")

    log.info("Final catalog: %d features (S5=%d, S6=%d)",
             len(df), (df["stage"] != "S6").sum(), (df["stage"] == "S6").sum())
    return df


def _matching_base_concepts(stem: str, all_bcs: set) -> list:
    """Stem-prefix match: bc == stem OR bc startswith stem + '_'."""
    return sorted(bc for bc in all_bcs
                  if bc == stem or bc.startswith(stem + "_"))


# ═══════════════════════════════════════════════════════════════════════════════
# FILE MATCHING — pair S6 and S5 parquets by date_hour
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_date_hour(filename: str) -> str | None:
    """Extract date_hour key from parquet filename.
    Handles both YYYYMMDD_HH and YYYY-MM-DD_HH formats."""
    # Try YYYY-MM-DD_HH first (S5/S6 engine output format)
    m = re.search(r"(\d{4}-\d{2}-\d{2}_\d{2})\.parquet$", filename)
    if m:
        return m.group(1)
    # Fallback: YYYYMMDD_HH
    m = re.search(r"(\d{8}_\d{2})\.parquet$", filename)
    return m.group(1) if m else None


def build_file_index(directory: str, pattern: str) -> dict:
    """Build {date_hour: filepath} index for a glob pattern within directory."""
    full_glob = os.path.join(directory, pattern)
    index = {}
    for f in sorted(glob.glob(full_glob)):
        dh = _extract_date_hour(os.path.basename(f))
        if dh:
            index[dh] = f
    return index


def build_s6_file_index(s6_dir: str) -> dict:
    """
    Build {date_hour: filepath} index for S6 files.
    Prefers btcethbnb over btceth when both exist for the same hour
    (btcethbnb has more features = superset).
    """
    index = {}
    for pattern in S6_GLOB_PATTERNS:
        full_glob = os.path.join(s6_dir, pattern)
        for f in sorted(glob.glob(full_glob)):
            dh = _extract_date_hour(os.path.basename(f))
            if dh is None:
                continue
            # btcethbnb files are globbed first (see S6_GLOB_PATTERNS order),
            # so only add btceth if no btcethbnb exists for this hour.
            if dh not in index:
                index[dh] = f
    return index


def discover_s6_files(s6_dir: str) -> list:
    """Discover all S6 parquet files (both btceth and btcethbnb)."""
    files = []
    for pattern in S6_GLOB_PATTERNS:
        files.extend(sorted(glob.glob(os.path.join(s6_dir, pattern))))
    return sorted(set(files))


# ═══════════════════════════════════════════════════════════════════════════════
# ONLINE PEARSON — within-group (symmetric, upper-triangle)
# ═══════════════════════════════════════════════════════════════════════════════

class OnlineCorr:
    """Streaming Pearson accumulator for one feature group (upper-triangle)."""

    def __init__(self, feature_names: list):
        self.features = feature_names
        n = len(feature_names)
        self._n = n
        pairs = n * (n - 1) // 2
        self.n_xy   = np.zeros(pairs, dtype=np.int64)
        self.sum_x  = np.zeros(pairs, dtype=np.float64)
        self.sum_y  = np.zeros(pairs, dtype=np.float64)
        self.sum_x2 = np.zeros(pairs, dtype=np.float64)
        self.sum_y2 = np.zeros(pairs, dtype=np.float64)
        self.sum_xy = np.zeros(pairs, dtype=np.float64)

    def update(self, chunk: pd.DataFrame):
        n, features = self._n, self.features
        nrows = len(chunk)
        vals = np.full((nrows, n), np.nan, dtype=np.float64)
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

    def compute(self, min_valid: int = MIN_VALID_ROWS) -> list:
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
# ONLINE PEARSON — cross-group (Group A × Group B)
# ═══════════════════════════════════════════════════════════════════════════════

class OnlineCrossCorr:
    """Streaming Pearson accumulator for cross-group pairs (A × B)."""

    def __init__(self, features_a: list, features_b: list):
        self.features_a = features_a
        self.features_b = features_b
        na, nb = len(features_a), len(features_b)
        self._na, self._nb = na, nb
        total = na * nb
        self.n_xy   = np.zeros(total, dtype=np.int64)
        self.sum_x  = np.zeros(total, dtype=np.float64)
        self.sum_y  = np.zeros(total, dtype=np.float64)
        self.sum_x2 = np.zeros(total, dtype=np.float64)
        self.sum_y2 = np.zeros(total, dtype=np.float64)
        self.sum_xy = np.zeros(total, dtype=np.float64)

    def update(self, chunk: pd.DataFrame):
        na, nb = self._na, self._nb
        nrows = len(chunk)
        chunk_cols = set(chunk.columns)

        vals_a = np.full((nrows, na), np.nan, dtype=np.float64)
        vals_b = np.full((nrows, nb), np.nan, dtype=np.float64)
        for i, f in enumerate(self.features_a):
            if f in chunk_cols:
                vals_a[:, i] = chunk[f].to_numpy(dtype=np.float64, na_value=np.nan)
        for j, f in enumerate(self.features_b):
            if f in chunk_cols:
                vals_b[:, j] = chunk[f].to_numpy(dtype=np.float64, na_value=np.nan)

        valid_a = ~np.isnan(vals_a)
        valid_b = ~np.isnan(vals_b)

        k = 0
        for i in range(na):
            vi = valid_a[:, i]
            if not vi.any():
                k += nb
                continue
            xi = vals_a[:, i]
            for j in range(nb):
                mask = vi & valid_b[:, j]
                if mask.any():
                    x = xi[mask]; y = vals_b[mask, j]
                    self.n_xy[k]   += len(x)
                    self.sum_x[k]  += x.sum()
                    self.sum_y[k]  += y.sum()
                    self.sum_x2[k] += np.dot(x, x)
                    self.sum_y2[k] += np.dot(y, y)
                    self.sum_xy[k] += np.dot(x, y)
                k += 1

    def compute(self, min_valid: int = MIN_VALID_ROWS) -> list:
        results = []
        fa, fb = self.features_a, self.features_b
        k = 0
        for i in range(self._na):
            for j in range(self._nb):
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
                        results.append((fa[i], fb[j], r))
                k += 1
        return results


# ═══════════════════════════════════════════════════════════════════════════════
# GREEDY DROP CANDIDATES
# ═══════════════════════════════════════════════════════════════════════════════

def greedy_drop_candidates(pairs_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Greedy: iteratively drop the feature involved in the most high-corr pairs."""
    high = pairs_df[pairs_df["abs_correlation"] > threshold].copy()
    high = high[high["feature_a"] != high["feature_b"]]
    if high.empty:
        return pd.DataFrame(columns=["feature_name", "n_redundant_pairs",
                                     "max_corr_with", "max_corr"])

    drops = []
    remaining = high.copy()

    while len(remaining) > 0:
        ca = remaining["feature_a"].value_counts()
        cb = remaining["feature_b"].value_counts()
        counts = ca.add(cb, fill_value=0).astype(int)

        feat = counts.idxmax()
        mask = (remaining["feature_a"] == feat) | (remaining["feature_b"] == feat)
        best = remaining[mask].sort_values("abs_correlation", ascending=False).iloc[0]
        partner = best["feature_b"] if best["feature_a"] == feat else best["feature_a"]

        drops.append({
            "feature_name": feat,
            "n_redundant_pairs": int(counts[feat]),
            "max_corr_with": partner,
            "max_corr": best["abs_correlation"],
        })

        remaining = remaining[(remaining["feature_a"] != feat) &
                              (remaining["feature_b"] != feat)]

    return pd.DataFrame(drops)


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE A — S6 Intra-Correlation
# ═══════════════════════════════════════════════════════════════════════════════

def run_s6_intra_correlation(
    catalog: pd.DataFrame,
    s6_dir: str,
    log: logging.Logger,
) -> tuple:
    """
    Check correlations WITHIN S6 features, grouped by base_concept.
    Also runs one global accumulator across ALL S6 features to catch
    cross-concept redundancy (e.g., same template across different pairs).
    """
    log.info("=" * 65)
    log.info("MODULE A — S6 Intra-Correlation")
    log.info("=" * 65)

    s6_catalog = catalog[catalog["stage"] == "S6"].copy()
    if s6_catalog.empty:
        log.error("No S6 features in catalog — skipping Module A")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # S6 parquet files
    s6_files = discover_s6_files(s6_dir)
    if not s6_files:
        log.error("No S6 parquet files found in %s — skipping Module A", s6_dir)
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    log.info("S6 catalog: %d features, S6 parquets: %d files", len(s6_catalog), len(s6_files))

    # Pre-flight: check which S6 features exist in parquets
    all_s6_cols = set()
    for f in s6_files[:5]:  # sample first 5 for speed
        try:
            all_s6_cols.update(pq.read_schema(f).names)
        except Exception:
            pass

    s6_features_found = sorted(
        set(s6_catalog["bare_name"]) & all_s6_cols
    )
    s6_features_missing = sorted(
        set(s6_catalog["bare_name"]) - all_s6_cols
    )

    log.info("S6 features in parquet: %d / %d", len(s6_features_found), len(s6_catalog))
    if s6_features_missing:
        log.info("  Missing (expected if BNB not yet available): %d features",
                 len(s6_features_missing))
        # Classify missing by pair tag
        for tag in PAIR_TAGS:
            n = sum(1 for f in s6_features_missing if f.endswith(f"_{tag}"))
            if n > 0:
                log.info("    %s: %d missing", tag, n)
        n_other = sum(1 for f in s6_features_missing
                      if not any(f.endswith(f"_{t}") for t in PAIR_TAGS))
        if n_other > 0:
            log.info("    intermediary/other: %d missing", n_other)

    if len(s6_features_found) < 2:
        log.error("Too few S6 features found — skipping Module A")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # ── Build accumulators ──
    # 1. Per-base-concept groups (within-concept check)
    s6_found_catalog = s6_catalog[s6_catalog["bare_name"].isin(s6_features_found)]
    groups = {n: g for n, g in s6_found_catalog.groupby("base_concept") if len(g) >= 2}

    group_accumulators = {
        name: OnlineCorr(grp["bare_name"].tolist())
        for name, grp in groups.items()
    }

    # 2. Global cross-concept accumulator (all S6 features in one)
    #    Only if feasible (< 200 features → < 20k pairs)
    global_acc = None
    if len(s6_features_found) <= 200:
        global_acc = OnlineCorr(s6_features_found)
        log.info("Global S6 accumulator: %d features → %d pairs",
                 len(s6_features_found),
                 len(s6_features_found) * (len(s6_features_found) - 1) // 2)
    else:
        log.info("S6 feature count (%d) > 200 — skipping global accumulator",
                 len(s6_features_found))

    log.info("Within-concept groups: %d (>= 2 features each)", len(groups))

    all_needed = set(s6_features_found)
    group_cols = {name: set(grp["bare_name"].tolist()) for name, grp in groups.items()}

    # ── Single pass over S6 parquets ──
    n_files = len(s6_files)
    t0 = time.time()
    total_rows = 0

    for idx, f in enumerate(s6_files, 1):
        try:
            available = set(pq.read_schema(f).names)
            cols = list(all_needed & available)
            if not cols:
                continue
            chunk = pd.read_parquet(f, columns=cols)
            total_rows += len(chunk)
            if chunk.empty:
                continue

            chunk_cols = set(chunk.columns)
            for name, acc in group_accumulators.items():
                present = list(group_cols[name] & chunk_cols)
                if len(present) >= 2:
                    acc.update(chunk[present])

            if global_acc is not None:
                global_acc.update(chunk)

        except Exception as e:
            log.warning("  [%d/%d] skipping %s: %s", idx, n_files, os.path.basename(f), e)

        if idx % 50 == 0 or idx == n_files:
            elapsed = time.time() - t0
            rate = idx / elapsed if elapsed > 0 else 0
            eta = (n_files - idx) / rate if rate > 0 else 0
            log.info("  [%d/%d]  %s rows  |  %.1fs  ETA ~%.0fs",
                     idx, n_files, f"{total_rows:,}", elapsed, eta)

    log.info("S6 intra pass done: %s rows from %d files  (%.1fs)",
             f"{total_rows:,}", n_files, time.time() - t0)

    # ── Compute results ──
    all_high_pairs = []
    all_summaries = []

    # Within-concept results
    for name, grp in sorted(groups.items()):
        pairs = group_accumulators[name].compute()
        pairs = [(fa, fb, r) for fa, fb, r in pairs if fa != fb]
        if not pairs:
            continue

        abs_corrs = [abs(r) for _, _, r in pairs]
        summary = {
            "scope":           "within_concept",
            "group":           name,
            "n_features":      len(grp),
            "n_pairs":         len(pairs),
            "mean_abs_corr":   float(np.mean(abs_corrs)),
            "max_abs_corr":    float(np.max(abs_corrs)),
            "pairs_above_070": sum(1 for c in abs_corrs if c > 0.70),
            "pairs_above_095": sum(1 for c in abs_corrs if c > 0.95),
        }
        all_summaries.append(summary)

        for fa, fb, r in pairs:
            if abs(r) > HIGH_CORR_THRESHOLD:
                all_high_pairs.append({
                    "scope": "within_concept", "group": name,
                    "feature_a": fa, "feature_b": fb,
                    "correlation": round(r, 6),
                    "abs_correlation": round(abs(r), 6),
                })

    # Global cross-concept results
    if global_acc is not None:
        global_pairs = global_acc.compute()
        global_pairs = [(fa, fb, r) for fa, fb, r in global_pairs if fa != fb]

        # Only keep cross-concept pairs (different base_concept)
        feat_to_concept = dict(zip(s6_found_catalog["bare_name"],
                                   s6_found_catalog["base_concept"]))

        cross_concept_high = []
        for fa, fb, r in global_pairs:
            if abs(r) > HIGH_CORR_THRESHOLD:
                ca = feat_to_concept.get(fa, "?")
                cb = feat_to_concept.get(fb, "?")
                if ca != cb:
                    cross_concept_high.append({
                        "scope": "cross_concept",
                        "group": f"{ca} ↔ {cb}",
                        "feature_a": fa, "feature_b": fb,
                        "correlation": round(r, 6),
                        "abs_correlation": round(abs(r), 6),
                    })

        all_high_pairs.extend(cross_concept_high)

        if cross_concept_high:
            n95 = sum(1 for p in cross_concept_high if p["abs_correlation"] > 0.95)
            log.info("Cross-concept S6 pairs |r|>0.70: %d  (|r|>0.95: %d)",
                     len(cross_concept_high), n95)

    # Build DataFrames
    summary_df    = pd.DataFrame(all_summaries).sort_values(
        "max_abs_corr", ascending=False) if all_summaries else pd.DataFrame()
    high_pairs_df = pd.DataFrame(all_high_pairs).sort_values(
        "abs_correlation", ascending=False) if all_high_pairs else pd.DataFrame()
    drop_df       = greedy_drop_candidates(high_pairs_df, DROP_THRESHOLD) \
                    if not high_pairs_df.empty else pd.DataFrame()

    # Report
    log.info("── S6 Intra Results ──")
    log.info("  Within-concept groups:   %d", len(all_summaries))
    log.info("  High-corr pairs (>%.2f): %d", HIGH_CORR_THRESHOLD, len(high_pairs_df))
    if not high_pairs_df.empty:
        log.info("  Pairs |r|>0.95:          %d",
                 (high_pairs_df["abs_correlation"] > 0.95).sum())
    log.info("  Drop candidates @%.2f:   %d", DROP_THRESHOLD, len(drop_df))

    if not summary_df.empty:
        top = summary_df.head(10)
        log.info("Top groups by max_abs_corr:\n%s",
                 top.to_string(index=False, float_format="%.3f"))

    return summary_df, high_pairs_df, drop_df


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE B — S6 ↔ S5 Cross-Correlation
# ═══════════════════════════════════════════════════════════════════════════════

def run_s6_vs_s5_correlation(
    asset: str,
    catalog: pd.DataFrame,
    s5_dir: str,
    s6_dir: str,
    log: logging.Logger,
) -> tuple:
    """
    Check correlations between S6 features and surviving S5 features for one asset.
    Uses domain-guided families + top S5 discovery.
    """
    log.info("=" * 65)
    log.info("MODULE B — S6 ↔ S5 Cross-Correlation  [%s]", asset.upper())
    log.info("=" * 65)

    s6_catalog = catalog[catalog["stage"] == "S6"].copy()
    s5_catalog = catalog[catalog["stage"] != "S6"].copy()

    # Build file indices
    s6_index = build_s6_file_index(s6_dir)
    s5_index = build_file_index(s5_dir, S5_GLOB_TEMPLATE.format(asset=asset))

    # Matched date_hours
    common_dh = sorted(set(s6_index.keys()) & set(s5_index.keys()))
    log.info("Files: %d S6, %d S5_%s, %d matched date_hours",
             len(s6_index), len(s5_index), asset.upper(), len(common_dh))

    if not common_dh:
        log.error("No matched files — skipping S6↔S5 for %s", asset.upper())
        return pd.DataFrame(), pd.DataFrame()

    # Pre-flight: identify available columns
    s6_schema = set(pq.read_schema(s6_index[common_dh[0]]).names)
    s5_schema = set(pq.read_schema(s5_index[common_dh[0]]).names)

    s6_available = sorted(set(s6_catalog["bare_name"]) & s6_schema)
    s5_available = set(s5_catalog["bare_name"]) & s5_schema

    log.info("S6 features in parquet: %d / %d", len(s6_available), len(s6_catalog))
    log.info("S5_%s features in parquet: %d / %d",
             asset.upper(), len(s5_available), len(s5_catalog))

    # ── Build domain-guided accumulators ──
    feat_to_concept_s5 = dict(zip(s5_catalog["bare_name"], s5_catalog["base_concept"]))
    feat_to_concept_s6 = dict(zip(s6_catalog["bare_name"], s6_catalog["base_concept"]))

    all_s5_bcs = set(s5_catalog["base_concept"].unique())
    all_s6_bcs = set(s6_catalog["base_concept"].unique())

    family_accumulators = {}
    all_s6_needed = set()
    all_s5_needed = set()

    for fam_name, fam_def in S6_VS_S5_FAMILIES.items():
        # Expand stems to all matching base_concepts via prefix match
        s6_target_bcs = set()
        for stem in fam_def["s6"]:
            s6_target_bcs.update(_matching_base_concepts(stem, all_s6_bcs))
        s5_target_bcs = set()
        for stem in fam_def["s5"]:
            s5_target_bcs.update(_matching_base_concepts(stem, all_s5_bcs))

        # Resolve features whose base_concept is in the target set
        s6_feats = [f for f in s6_available
                    if feat_to_concept_s6.get(f, "") in s6_target_bcs]
        s5_feats = [f for f in s5_available
                    if feat_to_concept_s5.get(f, "") in s5_target_bcs]

        if s6_feats and s5_feats:
            acc = OnlineCrossCorr(s6_feats, s5_feats)
            family_accumulators[fam_name] = {
                "acc": acc, "s6": s6_feats, "s5": s5_feats,
            }
            all_s6_needed.update(s6_feats)
            all_s5_needed.update(s5_feats)
            log.info("  %-25s  %3d S6 × %3d S5 = %s cross-pairs",
                     fam_name, len(s6_feats), len(s5_feats),
                     f"{len(s6_feats) * len(s5_feats):,}")
        else:
            log.debug("  %-25s  no features (s6=%d, s5=%d)",
                      fam_name, len(s6_feats), len(s5_feats))

    # ── Top S5 discovery accumulator ──
    top_s5_found = [f for f in TOP_S5_FEATURES if f in s5_available]
    discovery_acc = None
    if s6_available and top_s5_found:
        discovery_acc = OnlineCrossCorr(s6_available, top_s5_found)
        all_s6_needed.update(s6_available)
        all_s5_needed.update(top_s5_found)
        log.info("  %-25s  %3d S6 × %3d S5 = %s cross-pairs",
                 "TOP_S5_DISCOVERY", len(s6_available), len(top_s5_found),
                 f"{len(s6_available) * len(top_s5_found):,}")

    if not family_accumulators and discovery_acc is None:
        log.error("No accumulators built — skipping S6↔S5 for %s", asset.upper())
        return pd.DataFrame(), pd.DataFrame()

    all_needed = all_s6_needed | all_s5_needed
    all_needed.add(USABILITY_COL)

    log.info("Total columns needed: %d S6 + %d S5 + usability",
             len(all_s6_needed), len(all_s5_needed))

    # ── Single pass: load matched S6 + S5 parquets ──
    n_files = len(common_dh)
    t0 = time.time()
    total_rows = 0

    for idx, dh in enumerate(common_dh, 1):
        try:
            s6_path = s6_index[dh]
            s5_path = s5_index[dh]

            # Load S6 (only needed ca_* cols)
            s6_avail_cols = set(pq.read_schema(s6_path).names)
            s6_cols = list(all_s6_needed & s6_avail_cols)
            df_s6 = pd.read_parquet(s6_path, columns=s6_cols) if s6_cols else pd.DataFrame()

            # Load S5 (only needed cols + usability)
            s5_avail_cols = set(pq.read_schema(s5_path).names)
            s5_cols = list((all_s5_needed | {USABILITY_COL}) & s5_avail_cols)
            df_s5 = pd.read_parquet(s5_path, columns=s5_cols) if s5_cols else pd.DataFrame()

            if df_s6.empty or df_s5.empty:
                continue

            # Join on index (timestamp)
            chunk = df_s6.join(df_s5, how="inner")
            del df_s6, df_s5

            # Apply usability filter
            if USABILITY_COL in chunk.columns:
                chunk = chunk[chunk[USABILITY_COL] == 1].drop(columns=[USABILITY_COL])

            total_rows += len(chunk)
            if chunk.empty:
                continue

            # Feed to family accumulators
            for fam_name, fam in family_accumulators.items():
                fam["acc"].update(chunk)

            # Feed to discovery accumulator
            if discovery_acc is not None:
                discovery_acc.update(chunk)

        except Exception as e:
            log.warning("  [%d/%d] skipping %s: %s", idx, n_files, dh, e)

        if idx % 50 == 0 or idx == n_files:
            elapsed = time.time() - t0
            rate = idx / elapsed if elapsed > 0 else 0
            eta = (n_files - idx) / rate if rate > 0 else 0
            log.info("  [%d/%d]  %s rows  |  %.1fs  ETA ~%.0fs",
                     idx, n_files, f"{total_rows:,}", elapsed, eta)

    log.info("S6↔S5 pass done [%s]: %s rows from %d matched files  (%.1fs)",
             asset.upper(), f"{total_rows:,}", n_files, time.time() - t0)

    # ── Compute results ──
    all_pairs = []
    all_summaries = []

    # Domain-guided families
    for fam_name, fam in family_accumulators.items():
        results = fam["acc"].compute()
        high = [(fa, fb, r) for fa, fb, r in results if abs(r) > HIGH_CORR_THRESHOLD]

        n_total = len(results)
        n70 = len(high)
        n85 = sum(1 for _, _, r in high if abs(r) > 0.85)
        n95 = sum(1 for _, _, r in high if abs(r) > 0.95)

        summary = {
            "family":           fam_name,
            "scope":            "domain_guided",
            "n_s6":             len(fam["s6"]),
            "n_s5":             len(fam["s5"]),
            "n_pairs_computed": n_total,
            "n_above_070":      n70,
            "n_above_085":      n85,
            "n_above_095":      n95,
            "max_abs_corr":     max(abs(r) for _, _, r in results) if results else 0,
            "mean_abs_corr":    float(np.mean([abs(r) for _, _, r in results])) if results else 0,
        }
        all_summaries.append(summary)

        for fa, fb, r in high:
            all_pairs.append({
                "family": fam_name, "scope": "domain_guided",
                "feature_s6": fa, "feature_s5": fb,
                "correlation": round(r, 6),
                "abs_correlation": round(abs(r), 6),
            })

        log.info("  %-25s  >0.70: %d  >0.85: %d  >0.95: %d  max=%.3f",
                 fam_name, n70, n85, n95, summary["max_abs_corr"])

    # Discovery results
    if discovery_acc is not None:
        results = discovery_acc.compute()
        high = [(fa, fb, r) for fa, fb, r in results if abs(r) > HIGH_CORR_THRESHOLD]

        summary = {
            "family":           "TOP_S5_DISCOVERY",
            "scope":            "discovery",
            "n_s6":             len(s6_available),
            "n_s5":             len(top_s5_found),
            "n_pairs_computed": len(results),
            "n_above_070":      len(high),
            "n_above_085":      sum(1 for _, _, r in high if abs(r) > 0.85),
            "n_above_095":      sum(1 for _, _, r in high if abs(r) > 0.95),
            "max_abs_corr":     max(abs(r) for _, _, r in results) if results else 0,
            "mean_abs_corr":    float(np.mean([abs(r) for _, _, r in results])) if results else 0,
        }
        all_summaries.append(summary)

        for fa, fb, r in high:
            all_pairs.append({
                "family": "TOP_S5_DISCOVERY", "scope": "discovery",
                "feature_s6": fa, "feature_s5": fb,
                "correlation": round(r, 6),
                "abs_correlation": round(abs(r), 6),
            })

        log.info("  %-25s  >0.70: %d  >0.85: %d  >0.95: %d  max=%.3f",
                 "TOP_S5_DISCOVERY",
                 summary["n_above_070"], summary["n_above_085"],
                 summary["n_above_095"], summary["max_abs_corr"])

    pairs_df   = pd.DataFrame(all_pairs).sort_values(
        "abs_correlation", ascending=False) if all_pairs else pd.DataFrame()
    summary_df = pd.DataFrame(all_summaries).sort_values(
        "n_above_095", ascending=False) if all_summaries else pd.DataFrame()

    log.info("── S6↔S5 Results [%s] ──", asset.upper())
    log.info("  Total high-corr pairs: %d", len(pairs_df))
    if not pairs_df.empty:
        log.info("  Pairs |r|>0.85: %d", (pairs_df["abs_correlation"] > 0.85).sum())
        log.info("  Pairs |r|>0.95: %d", (pairs_df["abs_correlation"] > 0.95).sum())

    return pairs_df, summary_df


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE C — Pair Redundancy (3-pair linear dependency)
# ═══════════════════════════════════════════════════════════════════════════════

def run_pair_redundancy(
    catalog: pd.DataFrame,
    s6_dir: str,
    log: logging.Logger,
) -> pd.DataFrame:
    """
    For each feature template: check if ethbnb ≈ btcbnb − btceth.
    If BNB data is unavailable, skip gracefully.
    """
    log.info("=" * 65)
    log.info("MODULE C — Pair Redundancy Check (3-pair linear dependency)")
    log.info("=" * 65)

    s6_catalog = catalog[catalog["stage"] == "S6"].copy()

    # S6 files
    s6_files = discover_s6_files(s6_dir)
    if not s6_files:
        log.info("No S6 files — skipping Module C")
        return pd.DataFrame()

    # Check which pair tags exist
    s6_schema = set(pq.read_schema(s6_files[0]).names)
    tag_present = {}
    for tag in PAIR_TAGS:
        n = sum(1 for col in s6_schema if col.endswith(f"_{tag}"))
        tag_present[tag] = n > 0
        log.info("  Pair %s: %s (%d features)", tag,
                 "available" if n > 0 else "NOT available", n)

    if not (tag_present.get("btceth") and tag_present.get("btcbnb")
            and tag_present.get("ethbnb")):
        log.info("Not all 3 pairs available — skipping pair redundancy check.")
        log.info("  (This is expected until BNB data pipeline is active.)")
        return pd.DataFrame()

    # Identify template triples: same feature core across all 3 pairs
    s6_names = set(s6_catalog["bare_name"])
    triples = []
    btceth_feats = sorted(f for f in s6_schema if f.endswith("_btceth") and f in s6_names)

    for fe in btceth_feats:
        core = fe.rsplit("_btceth", 1)[0]
        fb = f"{core}_btcbnb"
        fc = f"{core}_ethbnb"
        if fb in s6_schema and fc in s6_schema:
            triples.append((core, fe, fb, fc))

    log.info("Template triples found: %d", len(triples))
    if not triples:
        return pd.DataFrame()

    # Columns needed
    needed = set()
    for _, fe, fb, fc in triples:
        needed.update([fe, fb, fc])

    # Streaming: compute corr(ethbnb, btcbnb − btceth) per template
    # Use simple running stats for each triple
    accumulators = {}
    for core, fe, fb, fc in triples:
        accumulators[core] = {
            "btceth": fe, "btcbnb": fb, "ethbnb": fc,
            "n": 0,
            "sum_x": 0.0, "sum_y": 0.0,
            "sum_x2": 0.0, "sum_y2": 0.0, "sum_xy": 0.0,
        }

    t0 = time.time()
    total_rows = 0

    for idx, f in enumerate(s6_files, 1):
        try:
            available = set(pq.read_schema(f).names)
            cols = list(needed & available)
            if not cols:
                continue
            chunk = pd.read_parquet(f, columns=cols)
            total_rows += len(chunk)

            for core, state in accumulators.items():
                fe = state["btceth"]
                fb = state["btcbnb"]
                fc = state["ethbnb"]
                if fe not in chunk.columns or fb not in chunk.columns or fc not in chunk.columns:
                    continue

                x = chunk[fc].to_numpy(dtype=np.float64)          # ethbnb (actual)
                y = chunk[fb].to_numpy(dtype=np.float64) \
                  - chunk[fe].to_numpy(dtype=np.float64)           # btcbnb − btceth (predicted)

                mask = ~(np.isnan(x) | np.isnan(y))
                if not mask.any():
                    continue

                x, y = x[mask], y[mask]
                state["n"]      += len(x)
                state["sum_x"]  += x.sum()
                state["sum_y"]  += y.sum()
                state["sum_x2"] += np.dot(x, x)
                state["sum_y2"] += np.dot(y, y)
                state["sum_xy"] += np.dot(x, y)

        except Exception as e:
            log.warning("  [%d/%d] skipping: %s", idx, len(s6_files), e)

    log.info("Pair redundancy pass: %s rows  (%.1fs)", f"{total_rows:,}", time.time() - t0)

    # Compute correlations
    rows = []
    for core, state in accumulators.items():
        cnt = state["n"]
        if cnt < MIN_VALID_ROWS:
            continue
        sx, sy = state["sum_x"], state["sum_y"]
        sx2, sy2 = state["sum_x2"], state["sum_y2"]
        sxy = state["sum_xy"]
        denom = np.sqrt(max(cnt * sx2 - sx * sx, 0.0) *
                        max(cnt * sy2 - sy * sy, 0.0))
        if denom > 0:
            r = float(np.clip((cnt * sxy - sx * sy) / denom, -1.0, 1.0))
        else:
            r = np.nan

        rows.append({
            "template_core":  core,
            "ethbnb_feature":  state["ethbnb"],
            "reconstruction":  f"{state['btcbnb']} − {state['btceth']}",
            "corr_actual_vs_reconstructed": round(r, 6),
            "abs_corr":         round(abs(r), 6) if not np.isnan(r) else np.nan,
            "is_redundant":     abs(r) > 0.999 if not np.isnan(r) else False,
            "n_valid_rows":     cnt,
        })

    result_df = pd.DataFrame(rows).sort_values("abs_corr", ascending=False) \
                if rows else pd.DataFrame()

    if not result_df.empty:
        n_redundant = result_df["is_redundant"].sum()
        log.info("── Pair Redundancy Results ──")
        log.info("  Templates checked:  %d", len(result_df))
        log.info("  Perfectly redundant: %d (|r| > 0.999)", n_redundant)
        log.info("  Mean |r|:           %.4f", result_df["abs_corr"].mean())
        log.info("  Min  |r|:           %.4f", result_df["abs_corr"].min())

    return result_df


# ═══════════════════════════════════════════════════════════════════════════════
# CONSENSUS — flag S6 features that are redundant with S5 in BOTH assets
# ═══════════════════════════════════════════════════════════════════════════════

def compute_consensus_flags(
    asset_results: dict,
    log: logging.Logger,
) -> pd.DataFrame:
    """
    Flag S6 features where |r| > 0.95 with some S5 feature in BOTH assets.
    These are strong drop candidates (following asset-uniform policy).
    """
    if len(asset_results) < 2:
        return pd.DataFrame()

    # For each asset, build set of S6 features with |r| > 0.95
    flagged_per_asset = {}
    for asset, pairs_df in asset_results.items():
        if pairs_df.empty:
            flagged_per_asset[asset] = {}
            continue
        high = pairs_df[pairs_df["abs_correlation"] > 0.95]
        flagged = {}
        for _, row in high.iterrows():
            s6f = row["feature_s6"]
            if s6f not in flagged or row["abs_correlation"] > flagged[s6f]["abs_corr"]:
                flagged[s6f] = {
                    "s5_feature": row["feature_s5"],
                    "abs_corr": row["abs_correlation"],
                }
        flagged_per_asset[asset] = flagged

    # Consensus: flagged in ALL assets
    assets = list(flagged_per_asset.keys())
    if not assets:
        return pd.DataFrame()

    consensus = set(flagged_per_asset[assets[0]].keys())
    for a in assets[1:]:
        consensus &= set(flagged_per_asset[a].keys())

    rows = []
    for s6f in sorted(consensus):
        row = {"feature_s6": s6f}
        for a in assets:
            info = flagged_per_asset[a][s6f]
            row[f"corr_with_{a}"] = info["abs_corr"]
            row[f"s5_partner_{a}"] = info["s5_feature"]
        rows.append(row)

    df = pd.DataFrame(rows)
    log.info("Consensus S6↔S5 flags (>0.95 in both assets): %d features", len(df))
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Cross-Asset (S6) Correlation Sanity Check",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--catalog", default=DEFAULT_CATALOG)
    parser.add_argument("--s5-dir", default=S5_DIR_DEFAULT,
                        help="S5 reduced parquet directory")
    parser.add_argument("--s6-dir", default=S6_DIR_DEFAULT,
                        help="S6 parquet directory")
    parser.add_argument("--assets", nargs="+", default=["btc", "eth"],
                        help="Assets for S6↔S5 check (default: btc eth)")
    parser.add_argument("--skip-intra", action="store_true",
                        help="Skip Module A (S6 intra-correlation)")
    parser.add_argument("--skip-cross", action="store_true",
                        help="Skip Module B (S6↔S5 cross-correlation)")
    parser.add_argument("--skip-redundancy", action="store_true",
                        help="Skip Module C (BNB pair redundancy). Default: skip.")
    parser.add_argument("--enable-bnb-redundancy", action="store_true",
                        help="Opt-in to Module C (BNB pair redundancy). "
                             "Default: disabled (BNB out of scope for thesis).")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING"])
    args = parser.parse_args()

    log = setup_logging()
    log.setLevel(getattr(logging, args.log_level))

    t0_global = time.time()
    log.info("=" * 70)
    log.info("CA CORRELATION EXPLORER — %s",
             datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    log.info("=" * 70)
    log.info("Catalog: %s", args.catalog)
    log.info("S5 dir:  %s", args.s5_dir)
    log.info("S6 dir:  %s", args.s6_dir)
    log.info("Assets:  %s", args.assets)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load catalog
    catalog = load_catalog(args.catalog, log)

    # ── Module A: S6 Intra-Correlation ──
    if not args.skip_intra:
        intra_summary, intra_pairs, intra_drops = run_s6_intra_correlation(
            catalog, args.s6_dir, log,
        )

        for df, name in [
            (intra_summary, "s6_intra_group_summary.csv"),
            (intra_pairs,   "s6_intra_pairwise_high_corr.csv"),
            (intra_drops,   "s6_intra_drop_candidates.csv"),
        ]:
            p = os.path.join(OUTPUT_DIR, name)
            if not df.empty or "drop" in name:
                df.to_csv(p, index=False)
                log.info("  Saved: %s (%d rows)", name, len(df))

    # ── Module B: S6 ↔ S5 Cross-Correlation ──
    asset_cross_results = {}
    if not args.skip_cross:
        for asset in args.assets:
            pairs_df, summary_df = run_s6_vs_s5_correlation(
                asset, catalog, args.s5_dir, args.s6_dir, log,
            )
            asset_cross_results[asset] = pairs_df

            for df, name in [
                (pairs_df,   f"{asset}_s6_vs_s5_pairwise.csv"),
                (summary_df, f"{asset}_s6_vs_s5_family_summary.csv"),
            ]:
                p = os.path.join(OUTPUT_DIR, name)
                if not df.empty:
                    df.to_csv(p, index=False)
                    log.info("  Saved: %s (%d rows)", name, len(df))

        # Consensus
        consensus_df = compute_consensus_flags(asset_cross_results, log)
        if not consensus_df.empty:
            p = os.path.join(OUTPUT_DIR, "s6_vs_s5_consensus_flags.csv")
            consensus_df.to_csv(p, index=False)
            log.info("  Saved: s6_vs_s5_consensus_flags.csv (%d rows)", len(consensus_df))

    # ── Module C: Pair Redundancy (BNB) ──
    # DISABLED: BNB is out of scope for this thesis (only BTC and ETH are
    # analyzed). The pair redundancy check requires all three pairs
    # (btceth, btcbnb, ethbnb) which are not generated.
    if not args.skip_redundancy and args.enable_bnb_redundancy:
        log.warning("BNB pair-redundancy module is opt-in; running anyway "
                    "because --enable-bnb-redundancy was passed.")
        redundancy_df = run_pair_redundancy(catalog, args.s6_dir, log)
        if not redundancy_df.empty:
            p = os.path.join(OUTPUT_DIR, "pair_redundancy_report.csv")
            redundancy_df.to_csv(p, index=False)
            log.info("  Saved: pair_redundancy_report.csv (%d rows)", len(redundancy_df))

    # ── Final Report ──
    log.info("=" * 70)
    log.info("ALL DONE — %.1fs total", time.time() - t0_global)
    log.info("=" * 70)
    log.info("Output directory: %s", OUTPUT_DIR)
    log.info("Log: %s/cross_asset_correlation_explorer.log", LOG_DIR)


if __name__ == "__main__":
    main()