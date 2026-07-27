#!/usr/bin/env python3
"""
aggregate_fi_results.py
========================
External aggregator for the per-target importance CSVs produced by
s5_s6_feature_importance.py.

What this does
--------------
1. Reads {asset}_{horizon}_importance.csv for both assets and all 16 targets
2. Per feature × target × asset: takes gain_mean and null_gain_mean
3. Computes per-feature aggregate metrics:
     a. gain-over-null per (target, asset) cell
     b. is_strong per (target, asset) cell (gain_mean > null_gain_mean
        and gain_over_null >= STRONG_THRESHOLD)
     c. is_weak per (target, asset) cell (gain_over_null <= 1.0)
4. Drop decision:
     a. asset_weak[asset] = feature is weak in ALL 16 targets of that asset
     b. universal_weak = asset_weak[btc] AND asset_weak[eth]   (intersection)
5. Top-feature annotations (per-asset union over BTC/ETH):
     a. top_returns        : top X% mean gain-over-null over 8 return targets
     b. top_mfe            : top X% over 4 MFE targets
     c. top_mae            : top X% over 4 MAE targets
     d. top_short_horizon  : top X% over targets with horizon <= 30s
     e. top_long_horizon   : top X% over targets with horizon >= 60s

Outputs
-------
  - fi_aggregated.csv             per-feature consolidated table
  - fi_drop_candidates.csv        universally-weak features
  - fi_top_annotations.csv        per-feature top_* flags
  - fi_summary.txt                human-readable summary

Usage
-----
  python -m selection.aggregate_fi_results
  python -m selection.aggregate_fi_results --top-pct 20
  python -m selection.aggregate_fi_results --results-dir <path>
"""
from __future__ import annotations
from common.paths import REDUCTION_DIR

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ─── Configuration ──────────────────────────────────────────────────────────

BASE_DIR    = REDUCTION_DIR
RESULTS_DIR = BASE_DIR / "results" / "s5_s6_feature_importance_full"
OUTPUT_DIR  = BASE_DIR / "results" / "s5_s6_feature_importance_aggregated"

ASSETS = ["btc", "eth"]

RETURN_HORIZONS = ["1s", "5s", "15s", "30s", "60s", "120s", "300s", "900s"]
MFE_HORIZONS    = ["15s", "60s", "300s", "900s"]
MAE_HORIZONS    = ["15s", "60s", "300s", "900s"]

# Horizons in seconds for short/long classification
HORIZON_SECONDS = {
    "1s": 1, "5s": 5, "15s": 15, "30s": 30,
    "60s": 60, "120s": 120, "300s": 300, "900s": 900,
}
SHORT_HORIZON_CUTOFF = 30  # <= 30s is short, > 30s is long

# Defaults; can be overridden via CLI
DEFAULT_TOP_PCT          = 10.0
DEFAULT_STRONG_THRESHOLD = 1.5   # gain-over-null >= 1.5 counts as "strong"


# ─── File loading ────────────────────────────────────────────────────────────

def _target_name(family: str, horizon: str) -> str:
    """Reconstruct the target column name used in importance file naming."""
    # Naming convention from s5_s6_feature_importance.py:
    #   - returns: ret_fwd_{h}_{asset}        (h is e.g. '1s')
    #   - mfe:     mfe_fwd_{h}_bps_{asset}
    #   - mae:     mae_fwd_{h}_bps_{asset}
    if family == "ret":
        return f"ret_fwd_{horizon}"
    elif family == "mfe":
        return f"mfe_fwd_{horizon}_bps"
    elif family == "mae":
        return f"mae_fwd_{horizon}_bps"
    raise ValueError(f"Unknown family: {family}")


def _importance_filename(asset: str, family: str, horizon: str) -> str:
    """Match the actual filename pattern produced by the script.

    The script uses {asset}_{horizon}_importance.csv where horizon is the
    SHORT target name like 'ret_1s', 'mfe_15s_bps', 'mae_15s_bps'.
    """
    if family == "ret":
        return f"{asset}_ret_{horizon}_importance.csv"
    elif family == "mfe":
        return f"{asset}_mfe_{horizon}_bps_importance.csv"
    elif family == "mae":
        return f"{asset}_mae_{horizon}_bps_importance.csv"
    raise ValueError(f"Unknown family: {family}")


def _all_target_specs() -> list[tuple[str, str, str]]:
    """Return list of (family, horizon, target_label) for all 16 targets."""
    specs = []
    for h in RETURN_HORIZONS:
        specs.append(("ret", h, f"ret_{h}"))
    for h in MFE_HORIZONS:
        specs.append(("mfe", h, f"mfe_{h}"))
    for h in MAE_HORIZONS:
        specs.append(("mae", h, f"mae_{h}"))
    return specs


def load_importance_tables(results_dir: Path, log_fn=print) -> dict:
    """
    Return nested dict: results[asset][target_label] = DataFrame
    with columns at least [feature_name, gain_mean, null_gain_mean, gain_over_null].
    """
    out = {}
    target_specs = _all_target_specs()

    for asset in ASSETS:
        out[asset] = {}
        for family, horizon, target_label in target_specs:
            fname = _importance_filename(asset, family, horizon)
            fpath = results_dir / fname
            if not fpath.exists():
                log_fn(f"  WARN: missing file {fpath}")
                continue
            df = pd.read_csv(fpath)
            required = {"feature_name", "gain_mean"}
            missing = required - set(df.columns)
            if missing:
                log_fn(f"  WARN: {fname} missing columns {missing}")
                continue
            # Some files may not have null_gain_mean if null baseline was skipped
            if "null_gain_mean" not in df.columns:
                log_fn(f"  WARN: {fname} has no null_gain_mean; using 0")
                df["null_gain_mean"] = 0.0
            if "gain_over_null" not in df.columns:
                df["gain_over_null"] = np.where(
                    df["null_gain_mean"] > 0,
                    df["gain_mean"] / df["null_gain_mean"],
                    np.where(df["gain_mean"] > 0, np.inf, 0.0))
            out[asset][target_label] = df[
                ["feature_name", "gain_mean", "null_gain_mean", "gain_over_null"]
            ].copy()
            log_fn(f"  Loaded {fname}: {len(df)} features")
    return out


# ─── Per-asset weakness ──────────────────────────────────────────────────────

def per_asset_weak(asset_tables: dict, target_labels: list[str]) -> pd.DataFrame:
    """
    asset_tables: dict[target_label -> DataFrame]
    Returns DataFrame:
       feature_name, n_targets_seen, n_targets_weak, weak_in_all_targets,
       max_gain_over_null, mean_gain_over_null
    """
    # Long-format: feature_name, target, gain_over_null
    rows = []
    for target_label, df in asset_tables.items():
        if target_label not in target_labels:
            continue
        for _, r in df.iterrows():
            rows.append({
                "feature_name": r["feature_name"],
                "target":       target_label,
                "gain_over_null": r["gain_over_null"],
            })
    long_df = pd.DataFrame(rows)
    if long_df.empty:
        return pd.DataFrame(columns=["feature_name", "n_targets_seen",
                                     "n_targets_weak", "weak_in_all_targets",
                                     "max_gain_over_null", "mean_gain_over_null"])

    long_df["is_weak"] = long_df["gain_over_null"] <= 1.0

    grouped = long_df.groupby("feature_name").agg(
        n_targets_seen=("target", "count"),
        n_targets_weak=("is_weak", "sum"),
        max_gain_over_null=("gain_over_null", "max"),
        mean_gain_over_null=("gain_over_null", "mean"),
    ).reset_index()
    grouped["weak_in_all_targets"] = (
        grouped["n_targets_weak"] == grouped["n_targets_seen"])
    return grouped


# ─── Top annotations ─────────────────────────────────────────────────────────

def _aggregate_for_targets(asset_tables: dict, target_labels: list[str]) -> pd.DataFrame:
    """For one asset, average gain_over_null per feature across the given targets."""
    rows = []
    for target_label, df in asset_tables.items():
        if target_label not in target_labels:
            continue
        for _, r in df.iterrows():
            gon = r["gain_over_null"]
            if np.isinf(gon):
                gon = np.nan  # treat inf as missing for averaging; the
                              # n_targets_with_signal column tracks them
            rows.append({"feature_name": r["feature_name"],
                         "gain_over_null": gon,
                         "is_signal": (r["gain_over_null"] > 1.0)})
    if not rows:
        return pd.DataFrame(columns=["feature_name", "mean_gon", "n_signal"])

    long_df = pd.DataFrame(rows)
    return long_df.groupby("feature_name").agg(
        mean_gon=("gain_over_null", "mean"),
        n_signal=("is_signal", "sum"),
    ).reset_index()


def top_features_per_asset(asset_tables: dict, target_labels: list[str],
                           top_pct: float) -> set:
    """Return set of feature names in the top-`top_pct` percentile by mean
    gain-over-null over the supplied targets, for one asset."""
    agg = _aggregate_for_targets(asset_tables, target_labels)
    if agg.empty:
        return set()
    # Higher mean_gon = better. Top X% = features with rank >= 100-top_pct.
    cutoff = np.percentile(agg["mean_gon"].dropna(), 100 - top_pct)
    return set(agg.loc[agg["mean_gon"] >= cutoff, "feature_name"])


def top_union(tables: dict, target_labels: list[str], top_pct: float) -> set:
    """Per-asset union: feature is in the top if it qualifies in BTC OR ETH."""
    top = set()
    for asset in ASSETS:
        if asset in tables:
            top.update(top_features_per_asset(
                tables[asset], target_labels, top_pct))
    return top


# ─── Target label sets ──────────────────────────────────────────────────────

def _return_labels() -> list[str]:
    return [f"ret_{h}" for h in RETURN_HORIZONS]


def _mfe_labels() -> list[str]:
    return [f"mfe_{h}" for h in MFE_HORIZONS]


def _mae_labels() -> list[str]:
    return [f"mae_{h}" for h in MAE_HORIZONS]


def _all_labels() -> list[str]:
    return _return_labels() + _mfe_labels() + _mae_labels()


def _short_horizon_labels() -> list[str]:
    return ([f"ret_{h}" for h in RETURN_HORIZONS
             if HORIZON_SECONDS[h] <= SHORT_HORIZON_CUTOFF]
            + [f"mfe_{h}" for h in MFE_HORIZONS
               if HORIZON_SECONDS[h] <= SHORT_HORIZON_CUTOFF]
            + [f"mae_{h}" for h in MAE_HORIZONS
               if HORIZON_SECONDS[h] <= SHORT_HORIZON_CUTOFF])


def _long_horizon_labels() -> list[str]:
    return ([f"ret_{h}" for h in RETURN_HORIZONS
             if HORIZON_SECONDS[h] > SHORT_HORIZON_CUTOFF]
            + [f"mfe_{h}" for h in MFE_HORIZONS
               if HORIZON_SECONDS[h] > SHORT_HORIZON_CUTOFF]
            + [f"mae_{h}" for h in MAE_HORIZONS
               if HORIZON_SECONDS[h] > SHORT_HORIZON_CUTOFF])


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=RESULTS_DIR,
                    help="Directory containing the per-target importance CSVs.")
    ap.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                    help="Output directory for the aggregated tables.")
    ap.add_argument("--top-pct", type=float, default=DEFAULT_TOP_PCT,
                    help=f"Top percent threshold for the annotations "
                         f"(default {DEFAULT_TOP_PCT}).")
    ap.add_argument("--round", type=int, default=1,
                    help="FI round number (default 1). Sets the drop_layer "
                         "label to feature_importance_round_N so that "
                         "merge_fi_drops.py keeps each round separate.")
    ap.add_argument("--write", action="store_true",
                    help="Write output files. Without --write the script "
                         "only prints the summary.")
    args = ap.parse_args()

    drop_layer_label = f"feature_importance_round_{args.round}"

    print(f"\n{'='*70}")
    print(f"aggregate_fi_results")
    print(f"{'='*70}\n")
    print(f"Results dir:  {args.results_dir}")
    print(f"Output dir:   {args.output_dir}")
    print(f"Top pct:      {args.top_pct}%")
    print(f"FI round:     {args.round}  (drop_layer = {drop_layer_label})")
    print(f"")

    if not args.results_dir.exists():
        print(f"FATAL: results dir not found: {args.results_dir}")
        sys.exit(1)

    # ── 1. Load all per-target tables ───────────────────────────────────
    print("Loading importance tables...")
    tables = load_importance_tables(args.results_dir)

    n_loaded_btc = len(tables.get("btc", {}))
    n_loaded_eth = len(tables.get("eth", {}))
    print(f"\nLoaded: BTC={n_loaded_btc}/16 targets, ETH={n_loaded_eth}/16 targets")

    if n_loaded_btc == 0 and n_loaded_eth == 0:
        print("FATAL: no importance files found. Run feature_importance first.")
        sys.exit(1)

    # ── 2. Per-asset weakness ───────────────────────────────────────────
    print("\nComputing per-asset weakness...")
    all_labels = _all_labels()
    weak_per_asset = {}
    for asset in ASSETS:
        if asset not in tables or not tables[asset]:
            print(f"  {asset}: SKIPPED (no targets loaded)")
            continue
        weak_df = per_asset_weak(tables[asset], all_labels)
        n_weak = int(weak_df["weak_in_all_targets"].sum())
        n_total = len(weak_df)
        print(f"  {asset}: {n_weak}/{n_total} features weak in ALL targets")
        weak_per_asset[asset] = weak_df

    # ── 3. Universal weak (intersection across assets) ──────────────────
    print("\nComputing universal weakness (intersection across assets)...")
    if "btc" in weak_per_asset and "eth" in weak_per_asset:
        btc_weak_set = set(
            weak_per_asset["btc"].loc[
                weak_per_asset["btc"]["weak_in_all_targets"], "feature_name"])
        eth_weak_set = set(
            weak_per_asset["eth"].loc[
                weak_per_asset["eth"]["weak_in_all_targets"], "feature_name"])
        universal_weak = btc_weak_set & eth_weak_set
        print(f"  BTC-weak: {len(btc_weak_set)}")
        print(f"  ETH-weak: {len(eth_weak_set)}")
        print(f"  Universal (intersection): {len(universal_weak)}")
        print(f"  Symmetric difference (only-BTC + only-ETH): "
              f"{len(btc_weak_set ^ eth_weak_set)}")
    else:
        print("  SKIPPED (need both assets)")
        universal_weak = set()
        btc_weak_set = set()
        eth_weak_set = set()

    # ── 4. Top-feature annotations (per-asset union over BTC/ETH) ───────
    print("\nComputing top-feature annotations...")
    top_sets = {
        "top_returns":       top_union(tables, _return_labels(),       args.top_pct),
        "top_mfe":           top_union(tables, _mfe_labels(),          args.top_pct),
        "top_mae":           top_union(tables, _mae_labels(),          args.top_pct),
        "top_short_horizon": top_union(tables, _short_horizon_labels(),args.top_pct),
        "top_long_horizon":  top_union(tables, _long_horizon_labels(), args.top_pct),
    }
    for name, s in top_sets.items():
        print(f"  {name:<20s}: {len(s)} features")

    # Overlap diagnostics
    all_top = set()
    for s in top_sets.values():
        all_top |= s
    print(f"  Any top flag set:    {len(all_top)} features")

    # ── 5. Build consolidated per-feature output table ──────────────────
    print("\nBuilding consolidated table...")
    all_features = set()
    for asset_tabs in tables.values():
        for df in asset_tabs.values():
            all_features |= set(df["feature_name"])

    out_rows = []
    for feat in sorted(all_features):
        row = {"feature_name": feat}
        for asset in ASSETS:
            if asset in weak_per_asset:
                m = weak_per_asset[asset]
                hit = m.loc[m["feature_name"] == feat]
                if not hit.empty:
                    row[f"{asset}_weak"] = bool(hit.iloc[0]["weak_in_all_targets"])
                    row[f"{asset}_mean_gain_over_null"] = float(
                        hit.iloc[0]["mean_gain_over_null"])
                    row[f"{asset}_max_gain_over_null"] = float(
                        hit.iloc[0]["max_gain_over_null"])
                else:
                    row[f"{asset}_weak"] = None
                    row[f"{asset}_mean_gain_over_null"] = None
                    row[f"{asset}_max_gain_over_null"] = None
        row["universal_weak"] = feat in universal_weak
        for name, s in top_sets.items():
            row[name] = feat in s
        out_rows.append(row)

    out_df = pd.DataFrame(out_rows)

    # ── 6. Drop candidates table (subset for consolidated_drop_list) ────
    drop_df = out_df.loc[out_df["universal_weak"]].copy()
    drop_df["list"]       = "primary"
    drop_df["drop_layer"] = drop_layer_label
    drop_df["reason"]     = "universal_weak_gain_over_null"
    drop_cols = ["feature_name", "list", "drop_layer", "reason",
                 "btc_mean_gain_over_null", "eth_mean_gain_over_null"]
    drop_df = drop_df[drop_cols]

    # ── 7. Top annotations table (subset for downstream merge) ──────────
    top_cols = ["feature_name"] + list(top_sets.keys())
    top_df = out_df[top_cols].copy()
    # Filter to features that have at least one flag set, to keep it small
    has_any_flag = top_df[list(top_sets.keys())].any(axis=1)
    top_df = top_df.loc[has_any_flag].reset_index(drop=True)

    # ── 8. Summary ──────────────────────────────────────────────────────
    summary_lines = []
    summary_lines.append("Feature Importance Aggregation — Summary")
    summary_lines.append("=" * 60)
    summary_lines.append(f"Results source: {args.results_dir}")
    summary_lines.append(f"Targets loaded: BTC={n_loaded_btc}/16, ETH={n_loaded_eth}/16")
    summary_lines.append(f"Total features evaluated: {len(out_df)}")
    summary_lines.append("")
    summary_lines.append("Weakness:")
    summary_lines.append(f"  BTC-weak                : {len(btc_weak_set)}")
    summary_lines.append(f"  ETH-weak                : {len(eth_weak_set)}")
    summary_lines.append(f"  Universal (intersection): {len(universal_weak)}")
    summary_lines.append("")
    summary_lines.append(f"Top-feature annotations (top {args.top_pct}% per asset, union):")
    for name, s in top_sets.items():
        summary_lines.append(f"  {name:<20s}: {len(s):>5d} features")
    summary_lines.append(f"  Any top flag set    : {len(all_top):>5d} features")
    summary_lines.append("")
    summary = "\n".join(summary_lines)
    print("\n" + summary)

    # ── 9. Write ─────────────────────────────────────────────────────────
    if not args.write:
        print(f"\n[DRY-RUN] Pass --write to save outputs to {args.output_dir}")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output_dir / "fi_aggregated.csv", index=False)
    drop_df.to_csv(args.output_dir / "fi_drop_candidates.csv", index=False)
    top_df.to_csv(args.output_dir / "fi_top_annotations.csv", index=False)
    with open(args.output_dir / "fi_summary.txt", "w") as f:
        f.write(summary + "\n")
    print(f"\nWritten to {args.output_dir}:")
    print(f"  fi_aggregated.csv       ({len(out_df)} rows)")
    print(f"  fi_drop_candidates.csv  ({len(drop_df)} rows)")
    print(f"  fi_top_annotations.csv  ({len(top_df)} rows)")
    print(f"  fi_summary.txt")


if __name__ == "__main__":
    main()