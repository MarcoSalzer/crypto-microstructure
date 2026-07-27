#!/usr/bin/env python3
"""
cross_concept_correlation.py  (multi-asset, streaming)
=======================================================
Computes pairwise correlations BETWEEN related base_concepts using
a single-pass streaming approach (same as within_concept_correlation.py).

Never loads all data into RAM — reads one parquet file at a time,
accumulates running Pearson statistics per feature pair.

Usage:
    python cross_concept_correlation.py
    python cross_concept_correlation.py --assets btc
    python cross_concept_correlation.py --assets btc eth

    # Background:
    nohup python -u cross_concept_correlation.py > /dev/null 2>&1 &
    tail -f results/selection/logs/btc_cross_concept.log

Expects:
    - feature_catalog.csv in results/selection/
    - s5_features_{btc,eth}_YYYYMMDD.parquet in data_storage/s5_features/

Outputs (in results/selection/results/):
    - cross_concept_families.csv
    - {asset}_cross_concept_pairwise.csv
    - {asset}_cross_concept_summary.csv
    - {asset}_cross_concept_drop_candidates.csv
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
from datetime import datetime

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

BASE_DIR           = str(REDUCTION_DIR)
DEFAULT_CATALOG    = f"{BASE_DIR}/feature_catalog.csv"
OUTPUT_DIR         = f"{BASE_DIR}/results/cross_concept_correlation"
LOG_DIR            = f"{BASE_DIR}/logs"
DATA_GLOB_TEMPLATE = "data_storage/s5_features/s5_features_{asset}_*.parquet"

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
MIN_VALID_ROWS          = 3600    # minimum overlapping rows for valid correlation


# ═══════════════════════════════════════════════════════════════════════════════
# AXIS DECOMPOSITION (for differs_on annotation)
# ═══════════════════════════════════════════════════════════════════════════════

import re as _re

_WIN_PAT   = _re.compile(r"_(\d+)([smhd])(?=_|$)")
_SCOPE_PAT = _re.compile(r"_(fut|spot)(?=_|$)")
_DEPTH_PAT = _re.compile(r"_(\d+bps|struct\d+)(?=_|$)")


def decompose_bare_name(bare: str) -> dict:
    """
    Decomposes a bare_name into its axis tokens.
      stem:   remaining name after stripping all axes (= quantity identifier)
      depth:  token like '1bps', '5bps', 'struct50', or '' if none
      window: token like '15s', '60s', '5m', or '' if none
      scope:  'fut', 'spot', or '' if none

    Multiple tokens of the same axis (e.g. _2bps_5bps in logratio features)
    are merged into one — they vary as a unit.
    """
    if not isinstance(bare, str):
        return {"stem": bare, "depth": "", "window": "", "scope": ""}

    s = bare
    depths  = _DEPTH_PAT.findall(s); s = _DEPTH_PAT.sub("", s)
    windows = _WIN_PAT.findall(s);   s = _WIN_PAT.sub("", s)
    scopes  = _SCOPE_PAT.findall(s); s = _SCOPE_PAT.sub("", s)
    stem    = s

    return {
        "stem":   stem,
        "depth":  "|".join(d for d in depths)  if depths  else "",
        "window": "|".join(w + u for w, u in windows) if windows else "",
        "scope":  "|".join(scopes) if scopes else "",
    }


def differs_on(bare_a: str, bare_b: str) -> str:
    """
    Returns a sorted, '+'-joined string of axes that differ between two features.
    Possible values: 'depth', 'stem', 'window', 'scope', or combinations like 'depth+stem'.
    Empty string if features are identical.
    """
    a, b = decompose_bare_name(bare_a), decompose_bare_name(bare_b)
    axes = [ax for ax in ("stem", "depth", "window", "scope") if a[ax] != b[ax]]
    return "+".join(axes)


# ═══════════════════════════════════════════════════════════════════════════════
# CONCEPT FAMILY DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

CONCEPT_FAMILIES = {
    "pull_rate_variants": [
        "pull_rate", "mad_pull_rate", "median_pull_rate",
        "d1_pull_rate", "d2_pull_rate", "pull_rate_shock",
    ],
    "refill_rate_variants": [
        "refill_rate", "d1_refill_rate", "d2_refill_rate",
        "mad_refill_rate", "median_refill_rate",
    ],
    "pull_vs_refill": [
        "pull_rate", "refill_rate", "refill_vs_pull",
        "refill_vs_pull_div_minus",
    ],
    "net_pressure_variants": [
        "net_pressure", "net_pressure_logratio", "net_pressure_persist",
        "net_pressure_flip_rate", "net_pressure_depth_coherence",
        "d1_net_pressure", "d2_net_pressure",
        "net_add", "net_add_pressure",
    ],
    "queue_pressure_variants": [
        "queue_pressure", "d1_queue_pressure", "d2_queue_pressure",
        "median_queue_pressure", "queue_pressure_log", "queue_pressure_persist",
        "z_queue_pressure", "queue_imb", "queue_imb_persist",
    ],
    "liq_concentration_variants": [
        "liq_concentration", "liq_concentration_ask", "liq_concentration_bid",
        "liq_concentration_div_minus", "z_liq_concentration",
    ],
    "book_shape_imbalance": [
        "book_asymmetry", "book_asymmetry_div_minus", "depth_imbalance",
        "liq_imb", "liq_imb_sf", "liq_imb_persist_sf", "liq_imb_div_minus",
        "z_book_asymmetry", "z_liq_imb",
    ],
    "depth_notional_variants": [
        "depth_notional_ask", "depth_notional_bid",
        "depth_gradient", "depth_gradient_ask", "depth_gradient_bid",
        "depth_gradient_div_minus", "z_depth_gradient",
    ],
    "taker_imbalance_variants": [
        "taker_imbalance", "z_taker_imbalance", "median_taker_imbalance",
    ],
    "lwp_variants": [
        "lwp_mid", "lwp_ask", "lwp_bid", "z_lwp_minus_mid",
    ],
    "absorption_variants": [
        "trade_absorption_ratio", "trade_absorption_ratio_persist",
        "aggressor_absorption_ratio",
        "aggr_absorp_ratio_ask", "aggr_absorp_ratio_bid",
        "absorption_break_flag", "z_trade_absorption_ratio",
        "z_absorb_refill_ask", "z_absorb_refill_bid",
    ],
    "impact_variants": [
        "impact_per_liquidity", "impact_per_signed_persist",
        "mad_impact_per_signed", "median_impact_per_signed",
    ],
    "trade_activity": [
        "trade_count", "trade_count_sf_div", "trade_count_div",
        "trade_count_share", "avg_trade_size", "spot_taker_activity_share",
    ],
    "vacuum_churn": [
        "vacuum_score", "ask_churn", "bid_churn",
    ],
    "price_range_features": [
        "range_pct", "range_pos", "range_pos_day",
        "day_range_bps", "dist_to_day_high_bps", "dist_to_day_low_bps",
    ],
    "basis_features": [
        "basis", "basis_vwap_sf",
    ],
    "max_liquidity_features": [
        "max_bps_ask", "max_bps_bid",
        "max_liq_distance_ask", "max_liq_distance_bid",
    ],
    "liq_sum_persistence": [
        "liq_sum",
    ],
    "flow_depth_alignment": [
        "flow_depth_align", "z_flow_depth_align",
    ],
    "cancel_refill_behind": [
        "cancel_rate_ahead", "cancel_rate_behind", "refill_rate_behind",
    ],
}


# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

def setup_logging(asset: str) -> logging.Logger:
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"{asset}_cross_concept.log")

    log = logging.getLogger(f"cross_concept.{asset}")
    log.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        f"%(asctime)s  [{asset.upper()}]  %(levelname)-7s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    # Console handler
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

    log.info("Final catalog: %d features, %d unique base_concepts",
             len(df), df["base_concept"].nunique())
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# ONLINE CROSS-CORRELATION (streaming, pairwise NaN-safe)
# ═══════════════════════════════════════════════════════════════════════════════

class OnlineCrossCorr:
    """
    Accumulates running Pearson statistics between two feature groups.
    Same math as OnlineCorr in within_concept_correlation.py but for CROSS pairs.

    Group A: features_a (list of names)
    Group B: features_b (list of names)
    Computes correlation for every (a_i, b_j) pair.
    """

    def __init__(self, features_a: list, features_b: list):
        self.features_a = features_a
        self.features_b = features_b
        na, nb = len(features_a), len(features_b)
        self._na = na
        self._nb = nb
        total = na * nb
        self.n_xy   = np.zeros(total, dtype=np.int64)
        self.sum_x  = np.zeros(total, dtype=np.float64)
        self.sum_y  = np.zeros(total, dtype=np.float64)
        self.sum_x2 = np.zeros(total, dtype=np.float64)
        self.sum_y2 = np.zeros(total, dtype=np.float64)
        self.sum_xy = np.zeros(total, dtype=np.float64)

    def update(self, chunk: pd.DataFrame):
        """Feed one parquet chunk. Only processes columns present in chunk."""
        na, nb = self._na, self._nb
        fa, fb = self.features_a, self.features_b
        nrows = len(chunk)

        # Build value arrays — NaN where column missing
        vals_a = np.full((nrows, na), np.nan, dtype=np.float64)
        vals_b = np.full((nrows, nb), np.nan, dtype=np.float64)

        chunk_cols = set(chunk.columns)
        for i, f in enumerate(fa):
            if f in chunk_cols:
                vals_a[:, i] = chunk[f].to_numpy(dtype=np.float64, na_value=np.nan)
        for j, f in enumerate(fb):
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
                    x = xi[mask]
                    y = vals_b[mask, j]
                    n = len(x)
                    self.n_xy[k]   += n
                    self.sum_x[k]  += x.sum()
                    self.sum_y[k]  += y.sum()
                    self.sum_x2[k] += np.dot(x, x)
                    self.sum_y2[k] += np.dot(y, y)
                    self.sum_xy[k] += np.dot(x, y)
                k += 1

    def compute(self, min_valid: int = MIN_VALID_ROWS) -> list:
        """Return list of (feature_a, feature_b, pearson_r) tuples."""
        results = []
        fa, fb = self.features_a, self.features_b
        na, nb = self._na, self._nb
        k = 0
        for i in range(na):
            for j in range(nb):
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
# PRE-FLIGHT
# ═══════════════════════════════════════════════════════════════════════════════

def preflight_check(needed_features: set, parquet_files: list, log) -> set:
    """Scan parquet schemas to find which features actually exist in data."""
    log.info("Pre-flight: scanning %d parquet schemas...", len(parquet_files))
    t0 = time.time()
    all_cols = set()
    for idx, f in enumerate(parquet_files, 1):
        try:
            all_cols.update(pq.read_schema(f).names)
        except Exception as e:
            log.warning("  Schema fail: %s: %s", os.path.basename(f), e)
        if idx % 200 == 0 or idx == len(parquet_files):
            log.info("  Schema scan: %d/%d (%.1fs)", idx, len(parquet_files), time.time() - t0)

    found = needed_features & all_cols
    missing = needed_features - all_cols
    log.info("Pre-flight: %d/%d features found (%.1f%%)",
             len(found), len(needed_features),
             100 * len(found) / len(needed_features) if needed_features else 0)
    if missing:
        log.info("  Missing: %d features (%s%s)", len(missing),
                 sorted(missing)[:5], " ..." if len(missing) > 5 else "")
    return found


# ═══════════════════════════════════════════════════════════════════════════════
# STREAMING SINGLE-PASS
# ═══════════════════════════════════════════════════════════════════════════════

def run_streaming_cross_correlation(
    family_accumulators: dict,  # {family_name: [OnlineCrossCorr, ...]}
    all_needed_cols: set,
    parquet_files: list,
    log: logging.Logger,
) -> None:
    """
    Single-pass over all parquet files. For each chunk, feed it to
    every accumulator. Same pattern as within_concept_correlation.py.
    """
    n_files = len(parquet_files)
    t0 = time.time()
    total_raw = total_usable = 0
    n_accumulators = sum(len(accs) for accs in family_accumulators.values())

    log.info("Single-pass: %d files, %d families, %d cross-accumulators",
             n_files, len(family_accumulators), n_accumulators)

    for idx, f in enumerate(parquet_files, 1):
        try:
            available = set(pq.read_schema(f).names)
            cols_to_load = list(all_needed_cols & available)
            if not cols_to_load:
                continue

            chunk = pd.read_parquet(f, columns=cols_to_load)
            total_raw += len(chunk)

            # Apply usability filter
            if USABILITY_COL in chunk.columns:
                chunk = chunk[chunk[USABILITY_COL] == 1].drop(columns=[USABILITY_COL])
            total_usable += len(chunk)

            if chunk.empty:
                continue

            # Feed to all accumulators
            for fam_name, acc_list in family_accumulators.items():
                for acc in acc_list:
                    acc.update(chunk)

        except Exception as e:
            log.warning("  [%d/%d] skipping %s: %s", idx, n_files, os.path.basename(f), e)

        if idx % 50 == 0 or idx == n_files:
            elapsed = time.time() - t0
            rate = idx / elapsed if elapsed > 0 else 0
            eta = (n_files - idx) / rate if rate > 0 else 0
            log.info("  [%d/%d]  %s raw / %s usable  |  %.1fs  ETA ~%.0fs",
                     idx, n_files,
                     f"{total_raw:,}", f"{total_usable:,}",
                     elapsed, eta)

    pct = (1 - total_usable / total_raw) * 100 if total_raw > 0 else 0
    log.info("Pass done: %s raw -> %s usable (%.1f%% filtered) | %.1fs",
             f"{total_raw:,}", f"{total_usable:,}", pct, time.time() - t0)


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def build_family_results(
    family_name: str,
    concept_pairs: list,         # [(concept_a, concept_b), ...]
    accumulators: list,          # [OnlineCrossCorr, ...] in same order
    log: logging.Logger,
) -> tuple:
    """Compute correlations from accumulators and return (pairs_list, summary_dict)."""

    all_pairs = []
    for (c1, c2), acc in zip(concept_pairs, accumulators):
        results = acc.compute()
        for fa, fb, r in results:
            if abs(r) > HIGH_CORR_THRESHOLD:
                # Axis decomposition for post-hoc disaggregation (3.4.2)
                axes_diff = differs_on(fa, fb)
                all_pairs.append({
                    "family": family_name,
                    "concept_a": c1, "concept_b": c2,
                    "feature_a": fa, "feature_b": fb,
                    "correlation": round(r, 6),
                    "abs_correlation": round(abs(r), 6),
                    "differs_on": axes_diff,
                })

    n70 = len(all_pairs)
    n85 = sum(1 for p in all_pairs if p["abs_correlation"] > 0.85)
    n95 = sum(1 for p in all_pairs if p["abs_correlation"] > 0.95)

    summary = {
        "family": family_name,
        "n_concept_pairs": len(concept_pairs),
        "n_pairs_above_070": n70,
        "n_pairs_above_085": n85,
        "n_pairs_above_095": n95,
        "mean_abs_corr": round(np.mean([p["abs_correlation"] for p in all_pairs]), 4) if n70 else 0,
        "max_abs_corr": round(max(p["abs_correlation"] for p in all_pairs), 4) if n70 else 0,
    }

    if n70 > 0:
        top = sorted(all_pairs, key=lambda p: p["abs_correlation"], reverse=True)[:3]
        for p in top:
            log.info("    %s", p["feature_a"])
            log.info("      <-> %s  |r|=%.4f  diff=%s",
                     p["feature_b"], p["abs_correlation"], p["differs_on"])

    return all_pairs, summary


def greedy_drop_candidates(pairs_df: pd.DataFrame) -> pd.DataFrame:
    """Greedy: iteratively drop feature with most high-corr cross-concept pairs."""
    high = pairs_df[pairs_df["abs_correlation"] >= DROP_THRESHOLD].copy()
    if len(high) == 0:
        return pd.DataFrame()

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
            "family": best["family"],
            "concept": best["concept_a"] if best["feature_a"] == feat else best["concept_b"],
            "n_redundant_pairs": int(counts.max()),
            "max_corr_with": partner,
            "max_corr": best["abs_correlation"],
        })

        remaining = remaining[(remaining["feature_a"] != feat) &
                              (remaining["feature_b"] != feat)]

    return pd.DataFrame(drops)


# ═══════════════════════════════════════════════════════════════════════════════
# RUN ONE ASSET
# ═══════════════════════════════════════════════════════════════════════════════

def _matching_base_concepts(stem: str, all_bcs: set) -> list:
    """
    Returns all base_concepts that belong to this stem:
    bc == stem OR bc startswith stem + '_'.
    Example: stem='pull_rate' → ['pull_rate', 'pull_rate_1bps', 'pull_rate_2bps',
    'pull_rate_shock_2bps', 'pull_rate_shock_5bps', ...]
    """
    return sorted(bc for bc in all_bcs
                  if bc == stem or bc.startswith(stem + "_"))


def run_asset(asset: str, catalog: pd.DataFrame):
    log = setup_logging(asset)

    t0 = time.time()
    log.info("=" * 65)
    log.info("CROSS-CONCEPT CORRELATION — %s — %s",
             asset.upper(), datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    log.info("=" * 65)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Filter catalog to current asset
    before_asset = len(catalog)
    catalog = catalog[catalog["asset"] == asset].copy()
    log.info("Catalog filtered to asset=%s: %d -> %d features",
             asset, before_asset, len(catalog))

    # Find parquet files for this asset
    parquet_glob = DATA_GLOB_TEMPLATE.format(asset=asset)
    parquet_files = sorted(glob.glob(parquet_glob))
    if not parquet_files:
        log.error("No parquet files matching %s — SKIPPING", parquet_glob)
        return
    log.info("Found %d parquet files", len(parquet_files))

    # Validate families against catalog (PREFIX-MATCH on stems → base_concepts)
    all_bcs = set(catalog["base_concept"].unique())
    valid_families = {}  # {fam_name: {stem: [matching_base_concepts]}}
    for fam_name, stems in CONCEPT_FAMILIES.items():
        stem_to_bcs = {}
        for stem in stems:
            matched = _matching_base_concepts(stem, all_bcs)
            if matched:
                stem_to_bcs[stem] = matched
        if len(stem_to_bcs) >= 2:
            valid_families[fam_name] = stem_to_bcs
        elif stem_to_bcs:
            log.info("  Family '%s': only 1 stem matched (%s) — skipping",
                     fam_name, list(stem_to_bcs.keys()))
        else:
            log.info("  Family '%s': no stems matched any base_concept", fam_name)

    log.info("Valid families (>=2 matching stems): %d", len(valid_families))

    # Collect all needed features per family
    # Option 3: all matched base_concepts of the family are pooled;
    # cross-pairs are formed between ALL base_concept pairs,
    # independent of the stem bucket.
    all_needed = set()
    family_concept_features = {}   # {fam: {base_concept: [features]}}
    for fam_name, stem_to_bcs in valid_families.items():
        concept_feats = {}
        # Pool over all stems of the family
        all_bcs_in_fam = sorted(set(bc for bcs in stem_to_bcs.values() for bc in bcs))
        for bc in all_bcs_in_fam:
            feats = catalog[catalog["base_concept"] == bc]["bare_name"].tolist()
            if feats:
                concept_feats[bc] = feats
                all_needed.update(feats)
        family_concept_features[fam_name] = concept_feats
        log.info("  Family '%s': %d base_concepts pooled (over %d stems)",
                 fam_name, len(concept_feats), len(stem_to_bcs))

    all_needed.add(USABILITY_COL)
    log.info("Total features needed: %d", len(all_needed) - 1)

    # Pre-flight: check which features exist in parquet
    found_cols = preflight_check(all_needed, parquet_files, log)
    if not found_cols:
        log.error("No features found in parquet — SKIPPING %s", asset.upper())
        return

    # Build accumulators — one OnlineCrossCorr per concept-pair per family
    from itertools import combinations

    family_accumulators = {}   # {fam: [OnlineCrossCorr, ...]}
    family_concept_pairs = {}  # {fam: [(c1, c2), ...]}
    total_acc = 0
    total_cross_pairs = 0

    for fam_name, concept_feats in family_concept_features.items():
        # Filter to features actually found in parquet
        filtered = {}
        for c, feats in concept_feats.items():
            present = [f for f in feats if f in found_cols]
            if present:
                filtered[c] = present

        if len(filtered) < 2:
            continue

        cpairs = list(combinations(filtered.keys(), 2))
        accs = []
        for c1, c2 in cpairs:
            acc = OnlineCrossCorr(filtered[c1], filtered[c2])
            accs.append(acc)
            n_cross = len(filtered[c1]) * len(filtered[c2])
            total_cross_pairs += n_cross

        family_accumulators[fam_name] = accs
        family_concept_pairs[fam_name] = cpairs
        total_acc += len(accs)

        log.info("  %-35s  %2d concepts  %3d accumulators  ~%s cross-pairs",
                 fam_name, len(filtered), len(accs), f"{sum(len(filtered[c1])*len(filtered[c2]) for c1,c2 in cpairs):,}")

    log.info("Total: %d accumulators, ~%s cross-pairs to compute",
             total_acc, f"{total_cross_pairs:,}")

    # Also collect all needed cols (only those found)
    all_needed_found = {f for f in all_needed if f in found_cols}
    all_needed_found.add(USABILITY_COL)

    # === SINGLE PASS ===
    run_streaming_cross_correlation(
        family_accumulators, all_needed_found, parquet_files, log
    )

    # === COMPUTE RESULTS ===
    log.info("Computing correlations from accumulators...")
    all_pairs_list = []
    all_summaries = []

    for fam_idx, (fam_name, accs) in enumerate(family_accumulators.items()):
        cpairs = family_concept_pairs[fam_name]
        log.info("  [%d/%d] %s (%d concept-pairs)",
                 fam_idx + 1, len(family_accumulators), fam_name, len(cpairs))

        pairs_list, summary = build_family_results(fam_name, cpairs, accs, log)
        all_pairs_list.extend(pairs_list)
        all_summaries.append(summary)

        log.info("    >0.70: %d  >0.85: %d  >0.95: %d",
                 summary["n_pairs_above_070"],
                 summary["n_pairs_above_085"],
                 summary["n_pairs_above_095"])

    # Build DataFrames
    pairs_df = pd.DataFrame(all_pairs_list)
    if len(pairs_df) > 0:
        pairs_df = pairs_df.sort_values("abs_correlation", ascending=False)

    summary_df = pd.DataFrame(all_summaries)
    if len(summary_df) > 0:
        summary_df = summary_df.sort_values("n_pairs_above_095", ascending=False)

    drop_df = greedy_drop_candidates(pairs_df) if len(pairs_df) > 0 else pd.DataFrame()

    # Axis-Disaggregation: aggregated statistics per (family, differs_on)
    # Answers the 3.4.2 question: "how much redundancy is depth, stem, concept, ...?"
    if len(pairs_df) > 0:
        axis_summary = (
            pairs_df.groupby(["family", "differs_on"])
                    .agg(n_pairs=("abs_correlation", "size"),
                         n_above_085=("abs_correlation", lambda s: (s > 0.85).sum()),
                         n_above_095=("abs_correlation", lambda s: (s > 0.95).sum()),
                         mean_abs_corr=("abs_correlation", "mean"),
                         median_abs_corr=("abs_correlation", "median"),
                         max_abs_corr=("abs_correlation", "max"))
                    .round(4)
                    .reset_index()
                    .sort_values(["family", "n_above_095"], ascending=[True, False])
        )
    else:
        axis_summary = pd.DataFrame()

    # === SAVE ===
    pfx = f"{asset}_cross_concept"
    files_saved = []
    for df, name in [
        (pairs_df,     f"{pfx}_pairwise.csv"),
        (summary_df,   f"{pfx}_summary.csv"),
        (axis_summary, f"{pfx}_axis_summary.csv"),
        (drop_df,      f"{pfx}_drop_candidates.csv"),
    ]:
        if len(df) > 0 or "drop" in name:
            path = os.path.join(OUTPUT_DIR, name)
            df.to_csv(path, index=False)
            files_saved.append(name)
            log.info("  Saved: %s (%d rows)", path, len(df))

    # === REPORT ===
    log.info("=" * 65)
    log.info("RESULTS — %s", asset.upper())
    log.info("=" * 65)
    log.info("Cross-concept pairs |r|>0.70:  %d", len(pairs_df))
    if len(pairs_df) > 0:
        log.info("Cross-concept pairs |r|>0.85:  %d",
                 (pairs_df["abs_correlation"] > 0.85).sum())
        log.info("Cross-concept pairs |r|>0.95:  %d",
                 (pairs_df["abs_correlation"] > 0.95).sum())
    log.info("Drop candidates @%.2f:         %d", DROP_THRESHOLD, len(drop_df))
    log.info("Files saved: %s", ", ".join(files_saved))

    if len(summary_df) > 0:
        log.info("FAMILY SUMMARY:\n%s", summary_df.to_string(index=False))

    if len(axis_summary) > 0:
        log.info("AXIS-DISAGGREGATED SUMMARY (for 3.4.2):")
        log.info("\n%s", axis_summary.to_string(index=False))

    if len(drop_df) > 0:
        log.info("DROP CANDIDATES:\n%s", drop_df.to_string(index=False))

    log.info("DONE %s — %.1fs total", asset.upper(), time.time() - t0)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Cross-Concept Correlation (streaming)")
    parser.add_argument("--assets", nargs="+", default=["btc", "eth"])
    parser.add_argument("--catalog", default=DEFAULT_CATALOG)
    args = parser.parse_args()

    print(f"\nAssets: {args.assets}")
    print(f"Catalog: {args.catalog}\n", flush=True)

    # Load catalog once
    dummy_log = logging.getLogger("catalog_loader")
    dummy_log.addHandler(logging.StreamHandler(sys.stdout))
    dummy_log.setLevel(logging.INFO)
    catalog = load_catalog(args.catalog, dummy_log)

    # Save family definitions (with all matched base_concepts via prefix)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_bcs = set(catalog["base_concept"].unique())
    fam_rows = []
    for fam, stems in CONCEPT_FAMILIES.items():
        for stem in stems:
            matched_bcs = _matching_base_concepts(stem, all_bcs)
            for bc in matched_bcs:
                n = len(catalog[catalog["base_concept"] == bc])
                fam_rows.append({
                    "family": fam, "stem": stem, "base_concept": bc,
                    "n_features_per_asset": n,
                })
    fam_path = os.path.join(OUTPUT_DIR, "cross_concept_families.csv")
    pd.DataFrame(fam_rows).to_csv(fam_path, index=False)
    print(f"Saved: {fam_path}", flush=True)

    for asset in args.assets:
        run_asset(asset, catalog)
        print(flush=True)

    print("All assets processed.", flush=True)


if __name__ == "__main__":
    main()