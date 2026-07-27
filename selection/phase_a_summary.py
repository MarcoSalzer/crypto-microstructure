#!/usr/bin/env python3
"""
phase_a_summary.py
==================
Consolidated Phase-A diagnostics report for thesis Section 3.4.2.

Aggregates the outputs of all four Phase-A scripts into a single
Markdown-Report:

  - within_concept_correlation.py     → Section 1: Within-Concept Correlation
  - cross_concept_correlation.py → Section 2: Cross-Concept Correlation
  - cross_asset_correlation_explorer.py  → Section 3: S6 Intra + S6↔S5
  - vif_analysis.py             → Section 4: VIF Distribution

Each section is modular: missing inputs are skipped with a note.
Cross-asset comparison (BTC vs ETH) is built into each section.

Output:
  <output_dir>/phase_a_report_<timestamp>.md    — Markdown-Report
  <output_dir>/cross_asset_comparison.csv       — within-concept (compat)
  <output_dir>/axis_disaggregation.csv          — cross-concept axes
  <output_dir>/vif_tier_comparison.csv          — VIF tiers BTC vs ETH

Usage:
  python -m selection.phase_a_summary
  python -m selection.phase_a_summary --assets btc eth
"""

import argparse
import os
import sys
from datetime import datetime

import pandas as pd
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_ASSETS     = ["btc", "eth"]
DEFAULT_BASE_DIR   = "."
DEFAULT_OUTPUT_DIR = "results/selection/results/phase_a_summary"

# Result folder per script (all relative to base_dir)
CE_DIR   = "results/selection/results/within_concept_correlation"
CC_DIR   = "results/selection/results/cross_concept_correlation"
CA_DIR   = "results/selection/results/cross_asset_correlation_explorer"
VIF_DIR  = "results/selection/results/vif"

# within_concept_correlation outputs (per asset)
CE_FILES = {
    "summary":    "{asset}_group_correlation_summary.csv",
    "pairs":      "{asset}_pairwise_high_corr.csv",
    "axis":       "{asset}_axis_correlation_report.csv",
    "drop_095":   "{asset}_drop_candidates_095.csv",
    "drop_085":   "{asset}_drop_candidates_085.csv",
}

# cross_concept_correlation outputs (per asset + shared)
CC_FILES = {
    "families":   "cross_concept_families.csv",                # shared
    "pairwise":   "{asset}_cross_concept_pairwise.csv",
    "summary":    "{asset}_cross_concept_summary.csv",
    "axis":       "{asset}_cross_concept_axis_summary.csv",
    "drops":      "{asset}_cross_concept_drop_candidates.csv",
}

# cross_asset_correlation_explorer outputs
CA_FILES = {
    "s6_intra_summary":   "s6_intra_group_summary.csv",           # shared
    "s6_intra_pairs":     "s6_intra_pairwise_high_corr.csv",
    "s6_intra_drops":     "s6_intra_drop_candidates.csv",
    "s6_vs_s5_pairs":     "{asset}_s6_vs_s5_pairwise.csv",
    "s6_vs_s5_summary":   "{asset}_s6_vs_s5_family_summary.csv",
    "s6_vs_s5_consensus": "s6_vs_s5_consensus_flags.csv",         # shared
}

# vif_analysis outputs (per asset)
VIF_FILES = {
    "results":   "{asset}_vif_results.csv",
    "summary":   "{asset}_vif_summary.csv",
    "high":      "{asset}_vif_high_features.csv",
}

# Compat alias for backward compatibility
OUTPUT_FILES = CE_FILES
RESULTS_DIR  = CE_DIR


# ═══════════════════════════════════════════════════════════════════════════════
# LOADER
# ═══════════════════════════════════════════════════════════════════════════════

def asset_dir(base_dir: str, asset: str) -> str:
    # All CSVs are flat in results/ — no asset subfolder
    return os.path.join(base_dir, RESULTS_DIR)


def _try_read(path: str) -> pd.DataFrame | None:
    """Generic reader with a fallback on errors."""
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        return df if not df.empty else pd.DataFrame()
    except Exception as e:
        print(f"  [WARN] Could not read {path}: {e}", file=sys.stderr)
        return None


def load_asset_data(base_dir: str, asset: str) -> dict[str, pd.DataFrame | None]:
    """Load the within_concept_correlation outputs for one asset (Section 1)."""
    adir = os.path.join(base_dir, CE_DIR)
    data = {}
    for key, pattern in CE_FILES.items():
        data[key] = _try_read(os.path.join(adir, pattern.format(asset=asset)))
    return data


def load_cross_concept_data(base_dir: str, asset: str) -> dict[str, pd.DataFrame | None]:
    """Load the cross_concept_correlation outputs (Section 2)."""
    adir = os.path.join(base_dir, CC_DIR)
    data = {}
    for key, pattern in CC_FILES.items():
        data[key] = _try_read(os.path.join(adir, pattern.format(asset=asset)))
    return data


def _normalize_ca_intra_summary(df: pd.DataFrame | None) -> pd.DataFrame | None:
    """Schema bridge to cross_asset_correlation_explorer.py (intra_summary).

    Producer writes:    scope, group, n_features, n_pairs, mean_abs_corr,
                        max_abs_corr, pairs_above_070, pairs_above_095
    Consumer expects:   base_concept, usable_features/total_features,
                        pct_above_095, mean_abs_corr, max_abs_corr

    This function fills the missing columns deterministically from the
    existing producer columns. Existing columns are NOT overwritten,
    i.e. if the producer later writes the old schema again, the
    Fix idempotent.
    """
    if df is None or df.empty:
        return df
    out = df.copy()

    # Group-Identifier
    if "base_concept" not in out.columns and "group" in out.columns:
        out["base_concept"] = out["group"]

    # Feature-Counts
    if "usable_features" not in out.columns and "n_features" in out.columns:
        out["usable_features"] = out["n_features"]
    if "total_features" not in out.columns and "n_features" in out.columns:
        out["total_features"] = out["n_features"]

    # derive percent share >0.95 from the absolute count
    if ("pct_above_095" not in out.columns
            and "pairs_above_095" in out.columns
            and "n_pairs" in out.columns):
        denom = out["n_pairs"].replace(0, np.nan)
        out["pct_above_095"] = (out["pairs_above_095"] / denom * 100).fillna(0.0)

    # The producer writes only within_concept rows into intra_summary;
    # Defensive filter in case that changes.
    if "scope" in out.columns:
        mask = out["scope"] == "within_concept"
        if mask.any():
            out = out.loc[mask].reset_index(drop=True)

    return out


def _normalize_ca_vs_s5_summary(df: pd.DataFrame | None) -> pd.DataFrame | None:
    """Schema bridge to cross_asset_correlation_explorer.py (s6_vs_s5 family_summary).

    Producer writes:    family, scope, n_s6, n_s5, n_pairs_computed,
                        n_above_070, n_above_085, n_above_095,
                        max_abs_corr, mean_abs_corr
    Consumer expects:   family, n_pairs, n_pairs_above_070/085/095,
                        mean_abs_corr, max_abs_corr
    """
    if df is None or df.empty:
        return df
    out = df.copy()

    rename_targets = {
        "n_above_070":     "n_pairs_above_070",
        "n_above_085":     "n_pairs_above_085",
        "n_above_095":     "n_pairs_above_095",
        "n_pairs_computed": "n_pairs",
    }
    for src, dst in rename_targets.items():
        if src in out.columns and dst not in out.columns:
            out[dst] = out[src]

    return out


def load_ca_data(base_dir: str, asset: str) -> dict[str, pd.DataFrame | None]:
    """Load the cross_asset_correlation_explorer outputs (Section 3)."""
    adir = os.path.join(base_dir, CA_DIR)
    data = {}
    for key, pattern in CA_FILES.items():
        data[key] = _try_read(os.path.join(adir, pattern.format(asset=asset)))

    # Schema normalisation against the current cross_asset_correlation_explorer output.
    # Render functions (build_section_ca_correlation) stay unchanged.
    data["s6_intra_summary"] = _normalize_ca_intra_summary(data.get("s6_intra_summary"))
    data["s6_vs_s5_summary"] = _normalize_ca_vs_s5_summary(data.get("s6_vs_s5_summary"))

    return data


def load_vif_data(base_dir: str, asset: str) -> dict[str, pd.DataFrame | None]:
    """Load the vif_analysis outputs (Section 4)."""
    adir = os.path.join(base_dir, VIF_DIR)
    data = {}
    for key, pattern in VIF_FILES.items():
        data[key] = _try_read(os.path.join(adir, pattern.format(asset=asset)))
    return data


# ═══════════════════════════════════════════════════════════════════════════════
# PER-ASSET STATS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_asset_stats(asset: str, d: dict) -> dict:
    stats = {"asset": asset.upper()}

    # ── Group Summary ──────────────────────────────────────────────────────────
    s = d["summary"]
    if s is not None and not s.empty:
        stats["n_groups_total"]      = len(s)
        stats["n_features_total"]    = int(s["total_features"].sum())
        stats["n_features_usable"]   = int(s["usable_features"].sum())
        stats["n_pairs_total"]       = int(s["num_pairs"].sum())
        stats["mean_abs_corr"]       = float(s["mean_abs_corr"].mean())
        stats["median_abs_corr"]     = float(s["median_abs_corr"].median())
        stats["max_abs_corr"]        = float(s["max_abs_corr"].max())
        stats["groups_above_095_pct"]= float((s["pct_above_095"] > 0).mean() * 100)
        stats["groups_fully_redundant"] = int((s["pct_above_095"] == 100).sum())
        stats["top_redundant_group"] = s.sort_values("pct_above_095", ascending=False).iloc[0]["base_concept"]
        stats["top_redundant_pct"]   = float(s["pct_above_095"].max())
    else:
        for k in ["n_groups_total","n_features_total","n_features_usable","n_pairs_total",
                  "mean_abs_corr","median_abs_corr","max_abs_corr","groups_above_095_pct",
                  "groups_fully_redundant","top_redundant_group","top_redundant_pct"]:
            stats[k] = None

    # ── High Pairs ────────────────────────────────────────────────────────────
    p = d["pairs"]
    if p is not None and not p.empty:
        stats["n_high_pairs_085"]  = int(len(p))
        stats["n_high_pairs_095"]  = int((p["abs_correlation"] > 0.95).sum())
        stats["n_perfect_pairs"]   = int((p["abs_correlation"] > 0.999).sum())
        stats["mean_high_corr"]    = float(p["abs_correlation"].mean())
    else:
        stats["n_high_pairs_085"] = stats["n_high_pairs_095"] = 0
        stats["n_perfect_pairs"]  = 0
        stats["mean_high_corr"]   = None

    # ── Axis Report ───────────────────────────────────────────────────────────
    ax = d["axis"]
    if ax is not None and not ax.empty:
        axis_grp = ax.groupby("differs_on")["abs_correlation"].agg(["mean","count"])
        # Which axis has the highest correlation (i.e. this dimension is redundant)
        if not axis_grp.empty:
            most_redundant_axis = axis_grp["mean"].idxmax()
            stats["most_redundant_axis"]      = most_redundant_axis
            stats["most_redundant_axis_corr"] = float(axis_grp.loc[most_redundant_axis, "mean"])
            # same_variant: features that differ on no axis
            sv = axis_grp[axis_grp.index == "same_variant"]
            stats["same_variant_mean_corr"]   = float(sv["mean"].iloc[0]) if not sv.empty else None
            stats["same_variant_count"]       = int(sv["count"].iloc[0]) if not sv.empty else 0
        else:
            stats["most_redundant_axis"] = None
            stats["most_redundant_axis_corr"] = None
            stats["same_variant_mean_corr"] = None
            stats["same_variant_count"] = 0
    else:
        stats["most_redundant_axis"] = None
        stats["most_redundant_axis_corr"] = None
        stats["same_variant_mean_corr"] = None
        stats["same_variant_count"] = 0

    # ── Drop Candidates ───────────────────────────────────────────────────────
    d95 = d["drop_095"]
    d85 = d["drop_085"]
    stats["drop_candidates_095"] = len(d95) if d95 is not None else 0
    stats["drop_candidates_085"] = len(d85) if d85 is not None else 0
    if stats["n_features_total"]:
        stats["retain_rate_095"] = round(
            (stats["n_features_total"] - stats["drop_candidates_095"]) / stats["n_features_total"] * 100, 1)
        stats["retain_rate_085"] = round(
            (stats["n_features_total"] - stats["drop_candidates_085"]) / stats["n_features_total"] * 100, 1)
    else:
        stats["retain_rate_095"] = stats["retain_rate_085"] = None

    return stats


# ═══════════════════════════════════════════════════════════════════════════════
# CROSS-ASSET COMPARISON DETAILS
# ═══════════════════════════════════════════════════════════════════════════════

def cross_asset_group_comparison(data_map: dict[str, dict]) -> pd.DataFrame:
    """
    Compares all base_concepts that occur in both assets.
    Columns: base_concept, btc_mean, eth_mean, delta_mean,
             btc_pct095, eth_pct095, delta_pct095, btc_max, eth_max
    """
    frames = {}
    for asset, d in data_map.items():
        s = d["summary"]
        if s is not None and not s.empty:
            frames[asset] = s.set_index("base_concept")

    if len(frames) < 2:
        return pd.DataFrame()

    assets = list(frames.keys())
    a1, a2 = assets[0], assets[1]
    common = frames[a1].index.intersection(frames[a2].index)

    rows = []
    for bc in common:
        r1, r2 = frames[a1].loc[bc], frames[a2].loc[bc]
        rows.append({
            "base_concept":         bc,
            f"{a1}_mean_abs_corr":  round(float(r1["mean_abs_corr"]), 4),
            f"{a2}_mean_abs_corr":  round(float(r2["mean_abs_corr"]), 4),
            "delta_mean_abs_corr":  round(float(r1["mean_abs_corr"]) - float(r2["mean_abs_corr"]), 4),
            f"{a1}_pct_above_095":  round(float(r1["pct_above_095"]), 1),
            f"{a2}_pct_above_095":  round(float(r2["pct_above_095"]), 1),
            "delta_pct_above_095":  round(float(r1["pct_above_095"]) - float(r2["pct_above_095"]), 1),
            f"{a1}_max_abs_corr":   round(float(r1["max_abs_corr"]), 4),
            f"{a2}_max_abs_corr":   round(float(r2["max_abs_corr"]), 4),
            f"{a1}_num_pairs":      int(r1["num_pairs"]),
            f"{a2}_num_pairs":      int(r2["num_pairs"]),
        })

    df = pd.DataFrame(rows).sort_values("delta_mean_abs_corr", ascending=False)
    return df


def cross_asset_drop_diff(data_map: dict[str, dict]) -> dict:
    """
    Which features are proposed for dropping in one asset
    but not in the other?
    """
    assets = list(data_map.keys())
    if len(assets) < 2:
        return {}
    a1, a2 = assets[0], assets[1]

    result = {}
    for thr in ("drop_095", "drop_085"):
        d1 = data_map[a1][thr]
        d2 = data_map[a2][thr]
        s1 = set(d1["feature_name"].tolist()) if d1 is not None and not d1.empty else set()
        s2 = set(d2["feature_name"].tolist()) if d2 is not None and not d2.empty else set()
        result[thr] = {
            f"only_{a1}": sorted(s1 - s2),
            f"only_{a2}": sorted(s2 - s1),
            "both":       sorted(s1 & s2),
        }
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Section builders — Cross-Concept, S6, VIF
# ═══════════════════════════════════════════════════════════════════════════════

def build_section_cross_concept(
    cc_data_map: dict[str, dict],
    assets: list[str],
    W,                                  # writer function
) -> None:
    """Section: Cross-Concept Correlation (cross_concept_correlation.py outputs)."""
    W("## 2. Cross-Concept Correlation")
    W()
    W("Correlation **between** related base_concepts within semantic "
      "Families. Each pair is annotated with `differs_on` — which axis/axes "
      "vary between the features (depth, stem, window, scope).")
    W()

    # Check availability
    any_data = any(d.get("summary") is not None and not d["summary"].empty
                   for d in cc_data_map.values())
    if not any_data:
        W("> *Cross-concept outputs not found — section skipped.*")
        W()
        W("---")
        W()
        return

    # ─── Family Summary (per asset) ───
    W("### 2.1 Per-Family Summary")
    W()
    for asset in assets:
        d = cc_data_map.get(asset.lower(), {})
        summary = d.get("summary")
        if summary is None or summary.empty:
            continue
        W(f"**{asset.upper()}** — {len(summary)} families:")
        W()
        W("| family | concept_pairs | pairs>0.70 | pairs>0.85 | pairs>0.95 | mean \\|ρ\\| | max \\|ρ\\| |")
        W("| --- | --- | --- | --- | --- | --- | --- |")
        # Top 10 by pairs_above_095, then alphabetical for rest
        top = summary.sort_values("n_pairs_above_095", ascending=False).head(15)
        for _, row in top.iterrows():
            W(f"| {row['family']} | {int(row['n_concept_pairs'])} | "
              f"{int(row['n_pairs_above_070'])} | {int(row['n_pairs_above_085'])} | "
              f"{int(row['n_pairs_above_095'])} | {float(row['mean_abs_corr']):.4f} | "
              f"{float(row['max_abs_corr']):.4f} |")
        W()

    # ─── Axis-Disaggregated Summary (KEY TABLE FOR THESIS 3.4.2) ───
    W("### 2.2 Axis-Disaggregated Summary")
    W()
    W("Key table for 3.4.2: per `differs_on` axis, which redundancy actually arises.")
    W()
    W("- `depth` → pure depth redundancy (same quantity, different depth)")
    W("- `stem` → pure statistics-operator redundancy")
    W("- `window` → time-scale redundancy")
    W("- `scope` → spot↔futures redundancy")
    W("- Combinations (e.g. `depth+stem`) → mixed effects")
    W()

    for asset in assets:
        d = cc_data_map.get(asset.lower(), {})
        axis = d.get("axis")
        if axis is None or axis.empty:
            continue
        W(f"**{asset.upper()}**:")
        W()
        # Aggregate across families per differs_on axis
        agg = (axis.groupby("differs_on")
                   .agg(n_pairs=("n_pairs", "sum"),
                        n_above_085=("n_above_085", "sum"),
                        n_above_095=("n_above_095", "sum"),
                        mean_abs_corr=("mean_abs_corr",
                                       lambda s: round(np.average(s, weights=axis.loc[s.index, "n_pairs"]), 4)),
                        max_abs_corr=("max_abs_corr", "max"))
                   .sort_values("n_above_095", ascending=False)
                   .reset_index())
        W("| differs_on | n_pairs | >0.85 | >0.95 | weighted mean \\|ρ\\| | max \\|ρ\\| |")
        W("| --- | --- | --- | --- | --- | --- |")
        for _, row in agg.iterrows():
            W(f"| `{row['differs_on']}` | {int(row['n_pairs']):,} | "
              f"{int(row['n_above_085']):,} | {int(row['n_above_095']):,} | "
              f"{row['mean_abs_corr']:.4f} | {row['max_abs_corr']:.4f} |")
        W()

    # ─── Cross-Asset Family Comparison ───
    if len(assets) >= 2:
        a1, a2 = assets[0].lower(), assets[1].lower()
        s1 = cc_data_map.get(a1, {}).get("summary")
        s2 = cc_data_map.get(a2, {}).get("summary")
        if s1 is not None and not s1.empty and s2 is not None and not s2.empty:
            W("### 2.3 Cross-Asset Family Comparison")
            W()
            m = s1.set_index("family").join(
                s2.set_index("family"), lsuffix=f"_{a1}", rsuffix=f"_{a2}", how="inner"
            )
            W(f"| family | {a1}_mean | {a2}_mean | Δ | {a1}_>0.95 | {a2}_>0.95 |")
            W("| --- | --- | --- | --- | --- | --- |")
            for fam, row in m.sort_values(f"mean_abs_corr_{a1}", ascending=False).head(15).iterrows():
                d_mean = row[f"mean_abs_corr_{a1}"] - row[f"mean_abs_corr_{a2}"]
                W(f"| {fam} | {row[f'mean_abs_corr_{a1}']:.4f} | "
                  f"{row[f'mean_abs_corr_{a2}']:.4f} | {d_mean:+.4f} | "
                  f"{int(row[f'n_pairs_above_095_{a1}'])} | {int(row[f'n_pairs_above_095_{a2}'])} |")
            W()

    W("---")
    W()


def build_section_ca_correlation(
    ca_data_map: dict[str, dict],
    assets: list[str],
    W,
) -> None:
    """Section: S6 Intra + S6↔S5 (cross_asset_correlation_explorer.py outputs)."""
    W("## 3. Cross-Asset (S6) Correlation")
    W()
    W("Diagnostics of the S6 cross-asset spread features: internal redundancy and "
      "overlap with the S5 single-asset sources.")
    W()

    # S6 intra is shared (only one file across all assets)
    s6_intra = next((d.get("s6_intra_summary") for d in ca_data_map.values()
                     if d.get("s6_intra_summary") is not None), None)

    any_s6_data = (s6_intra is not None and not s6_intra.empty) or \
                  any(d.get("s6_vs_s5_summary") is not None and not d["s6_vs_s5_summary"].empty
                      for d in ca_data_map.values())

    if not any_s6_data:
        W("> *S6 outputs not found — section skipped.*")
        W()
        W("---")
        W()
        return

    # ─── 3.1 S6 Intra-Correlation ───
    if s6_intra is not None and not s6_intra.empty:
        W("### 3.1 S6 Intra-Correlation")
        W()
        W(f"{len(s6_intra)} S6 base_concept groups analyzed.")
        W()
        red = s6_intra.sort_values("pct_above_095", ascending=False).head(10)
        W("**Top 10 redundant S6 groups:**")
        W()
        W("| base_concept | features | mean \\|ρ\\| | max \\|ρ\\| | %>0.95 |")
        W("| --- | --- | --- | --- | --- |")
        for _, row in red.iterrows():
            usable = row.get("usable_features", row.get("total_features", "n/a"))
            W(f"| {row['base_concept']} | {usable} | "
              f"{float(row['mean_abs_corr']):.4f} | {float(row['max_abs_corr']):.4f} | "
              f"{float(row['pct_above_095']):.1f}% |")
        W()

        n_fully = int((s6_intra["pct_above_095"] == 100).sum())
        n_partial = int(((s6_intra["pct_above_095"] > 0) & (s6_intra["pct_above_095"] < 100)).sum())
        n_clean = int((s6_intra["pct_above_095"] == 0).sum())
        W(f"- Fully redundant (100% of pairs >0.95): **{n_fully}** groups")
        W(f"- Partially redundant: **{n_partial}** groups")
        W(f"- Not redundant (no pairs >0.95): **{n_clean}** groups")
        W()

    # ─── 3.2 S6 ↔ S5 Cross-Correlation ───
    any_s6_s5 = any(d.get("s6_vs_s5_summary") is not None and not d["s6_vs_s5_summary"].empty
                    for d in ca_data_map.values())
    if any_s6_s5:
        W("### 3.2 S6 ↔ S5 Cross-Correlation (per asset)")
        W()
        for asset in assets:
            d = ca_data_map.get(asset.lower(), {})
            summary = d.get("s6_vs_s5_summary")
            if summary is None or summary.empty:
                continue
            W(f"**{asset.upper()}** — {len(summary)} families:")
            W()
            W("| family | n_pairs | >0.85 | >0.95 | mean \\|ρ\\| | max \\|ρ\\| |")
            W("| --- | --- | --- | --- | --- | --- |")
            cols_avail = summary.columns
            # Be tolerant about column names
            for _, row in summary.sort_values(
                "n_pairs_above_095" if "n_pairs_above_095" in cols_avail else "mean_abs_corr",
                ascending=False
            ).head(15).iterrows():
                np_ = int(row.get("n_concept_pairs", row.get("n_pairs", 0)))
                n85 = int(row.get("n_pairs_above_085", 0))
                n95 = int(row.get("n_pairs_above_095", 0))
                W(f"| {row.get('family', 'n/a')} | {np_} | {n85} | {n95} | "
                  f"{float(row.get('mean_abs_corr', 0)):.4f} | "
                  f"{float(row.get('max_abs_corr', 0)):.4f} |")
            W()

    # ─── 3.3 S6↔S5 Consensus Flags ───
    consensus = next((d.get("s6_vs_s5_consensus") for d in ca_data_map.values()
                      if d.get("s6_vs_s5_consensus") is not None), None)
    if consensus is not None and not consensus.empty:
        W("### 3.3 S6 ↔ S5 Consensus Flags")
        W()
        W(f"S6 features highly correlated with S5 sources in **both assets**: "
          f"**{len(consensus)}** features flagged.")
        W()
        if len(consensus) > 0:
            cols_show = [c for c in ["s6_feature", "feature", "family", "max_corr"]
                         if c in consensus.columns][:4]
            if cols_show:
                W("Top 10 (alphabetical):")
                W()
                W("| " + " | ".join(cols_show) + " |")
                W("| " + " | ".join(["---"] * len(cols_show)) + " |")
                for _, row in consensus.head(10).iterrows():
                    W("| " + " | ".join(str(row.get(c, "n/a")) for c in cols_show) + " |")
            W()

    W("---")
    W()


def build_section_vif(
    vif_data_map: dict[str, dict],
    assets: list[str],
    W,
) -> None:
    """Section: VIF Distribution (vif_analysis.py outputs)."""
    W("## 4. VIF Distribution")
    W()
    W("Variance inflation factor — complementary to the correlation analysis. "
      "VIF > 10 indicates problematic multicollinearity. "
      "Reporting is documentary (no drops are based on it).")
    W()

    any_data = any(d.get("results") is not None and not d["results"].empty
                   for d in vif_data_map.values())
    if not any_data:
        W("> *VIF outputs not found — section skipped.*")
        W()
        W("---")
        W()
        return

    # ─── 4.1 Tier Summary per Asset ───
    W("### 4.1 VIF Tier Distribution")
    W()
    W("| asset | ≤5 | 5–10 | 10–50 | >50 | total |")
    W("| --- | --- | --- | --- | --- | --- |")
    tier_rows_csv = []  # For CSV export
    for asset in assets:
        d = vif_data_map.get(asset.lower(), {})
        summary = d.get("summary")
        results = d.get("results")
        if summary is None or summary.empty:
            continue
        tiers = summary.set_index("tier")["n_features"].to_dict()
        total = int(results.shape[0]) if results is not None and not results.empty else 0
        le5    = int(tiers.get("vif_le_5", 0))
        b5_10  = int(tiers.get("vif_5_10", 0))
        b10_50 = int(tiers.get("vif_10_50", 0))
        gt50   = int(tiers.get("vif_gt_50", 0))
        W(f"| {asset.upper()} | {le5} ({100*le5/max(total,1):.1f}%) | "
          f"{b5_10} ({100*b5_10/max(total,1):.1f}%) | "
          f"{b10_50} ({100*b10_50/max(total,1):.1f}%) | "
          f"{gt50} ({100*gt50/max(total,1):.1f}%) | "
          f"{total} |")
        tier_rows_csv.append({
            "asset": asset.upper(), "vif_le_5": le5, "vif_5_10": b5_10,
            "vif_10_50": b10_50, "vif_gt_50": gt50, "total": total,
        })
    W()

    # ─── 4.2 High VIF Features by base_concept ───
    W("### 4.2 High VIF (>10) by base_concept")
    W()
    for asset in assets:
        d = vif_data_map.get(asset.lower(), {})
        high = d.get("high")
        if high is None or high.empty:
            continue
        if "base_concept" not in high.columns:
            continue
        by_bc = (high.groupby("base_concept")
                     .agg(n=("vif", "count"),
                          mean_vif=("vif", "mean"),
                          max_vif=("vif", "max"))
                     .sort_values("max_vif", ascending=False)
                     .head(10))
        if by_bc.empty:
            continue
        W(f"**{asset.upper()}** — top 10 base_concepts by max VIF:")
        W()
        W("| base_concept | n_features | mean VIF | max VIF |")
        W("| --- | --- | --- | --- |")
        for bc, row in by_bc.iterrows():
            W(f"| {bc} | {int(row['n'])} | {row['mean_vif']:.2f} | {row['max_vif']:.2f} |")
        W()

    # ─── 4.3 Cross-Asset VIF Comparison ───
    if len(assets) >= 2:
        a1, a2 = assets[0].lower(), assets[1].lower()
        r1 = vif_data_map.get(a1, {}).get("results")
        r2 = vif_data_map.get(a2, {}).get("results")
        if r1 is not None and not r1.empty and r2 is not None and not r2.empty:
            W("### 4.3 BTC vs ETH VIF Distribution")
            W()
            W(f"- {a1.upper()} mean VIF: {r1['vif'].mean():.2f}, median: {r1['vif'].median():.2f}, max: {r1['vif'].max():.2f}")
            W(f"- {a2.upper()} mean VIF: {r2['vif'].mean():.2f}, median: {r2['vif'].median():.2f}, max: {r2['vif'].max():.2f}")
            W()

    W("---")
    W()
    return tier_rows_csv


# ═══════════════════════════════════════════════════════════════════════════════
# MARKDOWN REPORT BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

def _fmt(val, fmt=".3f", fallback="n/a"):
    if val is None:
        return fallback
    try:
        return format(val, fmt)
    except Exception:
        return str(val)


def _pct_bar(pct: float, width: int = 20) -> str:
    """Small ASCII progress bar."""
    filled = round(pct / 100 * width)
    return "[" + "█" * filled + "░" * (width - filled) + f"] {pct:.1f}%"


def build_report(
    stats_map:    dict[str, dict],
    data_map:     dict[str, dict],
    cross_df:     pd.DataFrame,
    drop_diff:    dict,
    assets:       list[str],
    catalog_path: str,
    cc_data_map:  dict[str, dict] | None = None,
    ca_data_map:  dict[str, dict] | None = None,
    vif_data_map: dict[str, dict] | None = None,
) -> tuple[str, list]:
    """Build markdown report.
    Returns (markdown_text, vif_tier_csv_rows)."""

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = []
    W = lambda s="": lines.append(s)

    # ── Header ────────────────────────────────────────────────────────────────
    W("# Phase A — Correlation & Multicollinearity Diagnostics")
    W(f"*Thesis Section 3.4.2 input. Generated: {now}*")
    W()
    W(f"**Catalog**: `{catalog_path}`  |  **Assets**: {', '.join(a.upper() for a in assets)}")
    W()
    W("This report aggregates Phase A outputs from four diagnostics. "
      "Correlation and VIF results document the redundancy structure of the "
      "corpus but are **not** used to drop features — the reduction is performed "
      "exclusively in Section 3.4.3 via gradient-boosted feature importance.")
    W()
    W("---")
    W()

    # ── 1. Within-Concept (within_concept_correlation) ─────────────────────────────
    W("## 1. Within-Concept Correlation")
    W()
    W("Correlation between variants of the **same** base_concept "
      "(i.e. between window/scope parametrisations of the same quantity).")
    W()

    # ── 1. Quick Overview Table ───────────────────────────────────────────────
    W("### 1.1 Quick Overview")
    W()
    headers = ["Metric"] + [s["asset"] for s in stats_map.values()]
    rows_ov = [
        ("Features in catalog",        "n_features_total",    "d",    None),
        ("Features with valid pairs",   "n_features_usable",   "d",    None),
        ("Base-concept groups",         "n_groups_total",      "d",    None),
        ("Total pairs computed",        "n_pairs_total",       ",d",   None),
        ("Mean |ρ| across all groups",  "mean_abs_corr",       ".4f",  None),
        ("Median |ρ| across all groups","median_abs_corr",     ".4f",  None),
        ("Max |ρ| observed",            "max_abs_corr",        ".4f",  None),
        ("Pairs |ρ|>0.85",              "n_high_pairs_085",    ",d",   None),
        ("Pairs |ρ|>0.95",              "n_high_pairs_095",    ",d",   None),
        ("Near-perfect pairs (>0.999)", "n_perfect_pairs",     "d",    None),
        ("Groups fully redundant @0.95","groups_fully_redundant","d",  None),
        ("Drop candidates @0.95",       "drop_candidates_095", "d",    None),
        ("Drop candidates @0.85",       "drop_candidates_085", "d",    None),
        ("Retain rate @0.95 (%)",       "retain_rate_095",     ".1f",  None),
        ("Retain rate @0.85 (%)",       "retain_rate_085",     ".1f",  None),
    ]

    col_w = max(len(h) for h in headers)
    header_line = "| " + " | ".join(h.ljust(36) for h in ["Metric"] + list(stats_map.keys())) + " |"
    sep_line    = "| " + " | ".join(["-" * 36] * (1 + len(stats_map))) + " |"
    W(header_line)
    W(sep_line)
    for label, key, fmt_str, _ in rows_ov:
        vals = [_fmt(s.get(key), fmt_str) for s in stats_map.values()]
        W("| " + " | ".join([label.ljust(36)] + [v.ljust(36) for v in vals]) + " |")
    W()

    # ── 1.2 Per-Asset Detail ──────────────────────────────────────────────────
    W("### 1.2 Per-Asset Detail")
    W()

    for asset in assets:
        s  = stats_map[asset.lower()]
        d  = data_map[asset.lower()]
        su = d["summary"]
        ax = d["axis"]
        pa = d["pairs"]

        W(f"### {asset.upper()}")
        W()

        # Redundancy distribution
        W("**Redundancy distribution across groups:**")
        W()
        if su is not None and not su.empty:
            buckets = [
                ("|ρ| > 0.95 in 100% of pairs",  (su["pct_above_095"] == 100).sum()),
                ("|ρ| > 0.95 in >50% of pairs",  ((su["pct_above_095"] > 50) & (su["pct_above_095"] < 100)).sum()),
                ("|ρ| > 0.95 in >0% of pairs",   ((su["pct_above_095"] > 0)  & (su["pct_above_095"] <= 50)).sum()),
                ("No pairs |ρ| > 0.95",           (su["pct_above_095"] == 0).sum()),
            ]
            for label, count in buckets:
                pct = count / len(su) * 100 if len(su) > 0 else 0
                W(f"- {label}: **{count}** groups  {_pct_bar(pct, 15)}")
        W()

        # Top 10 most redundant groups
        W("**Top 10 most redundant base_concepts (by % pairs > 0.95):**")
        W()
        if su is not None and not su.empty:
            top10 = su.nlargest(10, "pct_above_095")[
                ["base_concept","total_features","num_pairs","mean_abs_corr","max_abs_corr","pct_above_095"]
            ]
            W("| base_concept | features | pairs | mean|ρ| | max|ρ| | %>0.95 |")
            W("| --- | --- | --- | --- | --- | --- |")
            for _, row in top10.iterrows():
                W(f"| {row['base_concept']} | {int(row['total_features'])} | {int(row['num_pairs'])} "
                  f"| {row['mean_abs_corr']:.3f} | {row['max_abs_corr']:.3f} | {row['pct_above_095']:.1f}% |")
        W()

        # Top 10 lowest redundancy (potentially safe to keep all)
        W("**Top 10 least redundant base_concepts (lowest mean |ρ|, ≥2 features):**")
        W()
        if su is not None and not su.empty:
            bot10 = su.nsmallest(10, "mean_abs_corr")[
                ["base_concept","total_features","num_pairs","mean_abs_corr","max_abs_corr","pct_above_095"]
            ]
            W("| base_concept | features | pairs | mean|ρ| | max|ρ| | %>0.95 |")
            W("| --- | --- | --- | --- | --- | --- |")
            for _, row in bot10.iterrows():
                W(f"| {row['base_concept']} | {int(row['total_features'])} | {int(row['num_pairs'])} "
                  f"| {row['mean_abs_corr']:.3f} | {row['max_abs_corr']:.3f} | {row['pct_above_095']:.1f}% |")
        W()

        # Axis analysis
        W("**Redundancy by axis (which dimension drives correlation):**")
        W()
        if ax is not None and not ax.empty:
            ax_grp = (ax.groupby("differs_on")["abs_correlation"]
                      .agg(mean="mean", median="median", count="count")
                      .sort_values("mean", ascending=False))
            W("| differs_on | n pairs | mean|ρ| | median|ρ| | interpretation |")
            W("| --- | --- | --- | --- | --- |")
            interp_map = {
                "same_variant":   "identical parameterisation — pure duplicates",
                "window_s":       "same concept, different window → time-scale redundancy",
                "depth_band":     "same concept, different depth → depth redundancy",
                "market_scope":   "same concept, fut vs spot → cross-market redundancy",
                "depth_band,window_s": "vary on both depth & window",
                "depth_band,market_scope": "vary on depth & market",
                "window_s,market_scope": "vary on window & market",
                "depth_band,market_scope,window_s": "vary on all three axes",
            }
            for idx_val, row in ax_grp.iterrows():
                interp = interp_map.get(str(idx_val), "")
                W(f"| {idx_val} | {int(row['count']):,} | {row['mean']:.3f} | {row['median']:.3f} | {interp} |")
        W()

        # Near-perfect / suspicious pairs
        if pa is not None and not pa.empty:
            perfect = pa[pa["abs_correlation"] > 0.999]
            if not perfect.empty:
                W(f"**Near-perfect pairs (|ρ| > 0.999) — {len(perfect)} pairs, possible duplicates:**")
                W()
                W("| feature_a | feature_b | ρ |")
                W("| --- | --- | --- |")
                for _, row in perfect.head(20).iterrows():
                    W(f"| {row['feature_a']} | {row['feature_b']} | {row['abs_correlation']:.5f} |")
                if len(perfect) > 20:
                    W(f"| ... | *({len(perfect)-20} more)* | |")
                W()

        # Drop summary
        d95 = d["drop_095"]
        d85 = d["drop_085"]
        W("**Drop candidate summary:**")
        W()
        n95 = len(d95) if d95 is not None and not d95.empty else 0
        n85 = len(d85) if d85 is not None and not d85.empty else 0
        nf  = s.get("n_features_total") or 1
        W(f"- Threshold 0.95: drop **{n95}** → retain **{nf - n95}** ({(nf-n95)/nf*100:.1f}%)")
        W(f"- Threshold 0.85: drop **{n85}** → retain **{nf - n85}** ({(nf-n85)/nf*100:.1f}%)")
        W()

        if d95 is not None and not d95.empty:
            by_bc = d95.groupby("base_concept")["feature_name"].count().sort_values(ascending=False)
            W("Top 10 groups by drop count @0.95:")
            W()
            W("| base_concept | drops |")
            W("| --- | --- |")
            for bc, cnt in by_bc.head(10).items():
                W(f"| {bc} | {cnt} |")
            W()

        W("---")
        W()

    # ── 1.3 Cross-Asset Comparison ────────────────────────────────────────────
    if len(assets) >= 2:
        a1, a2 = assets[0].lower(), assets[1].lower()
        W("### 1.3 Cross-Asset Comparison (within-concept)")
        W()

        # Global divergence
        s1, s2 = stats_map[a1], stats_map[a2]
        mean_diff = None
        if s1.get("mean_abs_corr") is not None and s2.get("mean_abs_corr") is not None:
            mean_diff = s1["mean_abs_corr"] - s2["mean_abs_corr"]
        W("**Global correlation level:**")
        W()
        W(f"- {a1.upper()} mean |ρ|: {_fmt(s1.get('mean_abs_corr'), '.4f')}")
        W(f"- {a2.upper()} mean |ρ|: {_fmt(s2.get('mean_abs_corr'), '.4f')}")
        if mean_diff is not None:
            direction = a1.upper() if mean_diff > 0 else a2.upper()
            W(f"- Delta: {abs(mean_diff):.4f} — **{direction} has higher overall redundancy**")
        W()

        # Groups most divergent between assets
        if not cross_df.empty:
            W(f"**Groups with largest BTC–ETH difference in mean |ρ| (top 15, {a1.upper()} > {a2.upper()}):**")
            W()
            W(f"| base_concept | {a1}_mean | {a2}_mean | Δ mean | {a1}_%>0.95 | {a2}_%>0.95 | Δ %0.95 |")
            W("| --- | --- | --- | --- | --- | --- | --- |")
            for _, row in cross_df.head(15).iterrows():
                W(f"| {row['base_concept']} "
                  f"| {row[f'{a1}_mean_abs_corr']:.3f} "
                  f"| {row[f'{a2}_mean_abs_corr']:.3f} "
                  f"| {row['delta_mean_abs_corr']:+.3f} "
                  f"| {row[f'{a1}_pct_above_095']:.1f}% "
                  f"| {row[f'{a2}_pct_above_095']:.1f}% "
                  f"| {row['delta_pct_above_095']:+.1f}% |")
            W()

            W(f"**Groups with largest BTC–ETH difference (top 15, {a2.upper()} > {a1.upper()}):**")
            W()
            W(f"| base_concept | {a1}_mean | {a2}_mean | Δ mean | {a1}_%>0.95 | {a2}_%>0.95 | Δ %0.95 |")
            W("| --- | --- | --- | --- | --- | --- | --- |")
            for _, row in cross_df.tail(15).sort_values("delta_mean_abs_corr").iterrows():
                W(f"| {row['base_concept']} "
                  f"| {row[f'{a1}_mean_abs_corr']:.3f} "
                  f"| {row[f'{a2}_mean_abs_corr']:.3f} "
                  f"| {row['delta_mean_abs_corr']:+.3f} "
                  f"| {row[f'{a1}_pct_above_095']:.1f}% "
                  f"| {row[f'{a2}_pct_above_095']:.1f}% "
                  f"| {row['delta_pct_above_095']:+.1f}% |")
            W()

            # Agreement: groups with high redundancy in BOTH
            both_high = cross_df[
                (cross_df[f"{a1}_pct_above_095"] > 50) &
                (cross_df[f"{a2}_pct_above_095"] > 50)
            ].sort_values(f"{a1}_pct_above_095", ascending=False)
            W(f"**Groups redundant in BOTH assets (>50% pairs > 0.95): {len(both_high)} groups**")
            W()
            if not both_high.empty:
                W(f"| base_concept | {a1}_%>0.95 | {a2}_%>0.95 |")
                W("| --- | --- | --- |")
                for _, row in both_high.head(20).iterrows():
                    W(f"| {row['base_concept']} | {row[f'{a1}_pct_above_095']:.1f}% | {row[f'{a2}_pct_above_095']:.1f}% |")
            W()

            # Groups where one is high, the other is low (asset-specific redundancy)
            asset_specific = cross_df[
                ((cross_df[f"{a1}_pct_above_095"] > 50) & (cross_df[f"{a2}_pct_above_095"] <= 20)) |
                ((cross_df[f"{a2}_pct_above_095"] > 50) & (cross_df[f"{a1}_pct_above_095"] <= 20))
            ]
            W(f"**Asset-specific redundancy (>50% for one, ≤20% for the other): {len(asset_specific)} groups**")
            W()
            if not asset_specific.empty:
                W(f"| base_concept | {a1}_%>0.95 | {a2}_%>0.95 | which asset |")
                W("| --- | --- | --- | --- |")
                for _, row in asset_specific.iterrows():
                    which = a1.upper() if row[f"{a1}_pct_above_095"] > 50 else a2.upper()
                    W(f"| {row['base_concept']} | {row[f'{a1}_pct_above_095']:.1f}% "
                      f"| {row[f'{a2}_pct_above_095']:.1f}% | {which} |")
            W()

        # Drop candidate overlap
        if drop_diff:
            W("**Drop candidate overlap (threshold 0.95):**")
            W()
            dd = drop_diff.get("drop_095", {})
            n_both  = len(dd.get("both", []))
            n_only1 = len(dd.get(f"only_{a1}", []))
            n_only2 = len(dd.get(f"only_{a2}", []))
            W(f"- Dropped in **both** assets: {n_both}")
            W(f"- Dropped only in **{a1.upper()}**: {n_only1}")
            W(f"- Dropped only in **{a2.upper()}**: {n_only2}")
            W()
            if dd.get(f"only_{a1}"):
                W(f"Features dropped only in {a1.upper()} @0.95 (first 20):")
                W()
                for f in dd[f"only_{a1}"][:20]:
                    W(f"  - {f}")
                W()
            if dd.get(f"only_{a2}"):
                W(f"Features dropped only in {a2.upper()} @0.95 (first 20):")
                W()
                for f in dd[f"only_{a2}"][:20]:
                    W(f"  - {f}")
                W()

        W("---")
        W()

    # ── 2. Cross-Concept Correlation ──────────────────────────────────────────
    if cc_data_map is not None:
        build_section_cross_concept(cc_data_map, assets, W)

    # ── 3. Cross-Asset (S6) Correlation ───────────────────────────────────────
    vif_tier_rows = []
    if ca_data_map is not None:
        build_section_ca_correlation(ca_data_map, assets, W)

    # ── 4. VIF Distribution ───────────────────────────────────────────────────
    if vif_data_map is not None:
        vif_tier_rows = build_section_vif(vif_data_map, assets, W) or []

    # ── 5. Main Takeaways ─────────────────────────────────────────────────────
    W("## 5. Main Takeaways")
    W()

    all_stats = list(stats_map.values())

    for s in all_stats:
        asset = s["asset"]
        W(f"### {asset}")
        W()
        # Redundancy level
        mean_r = s.get("mean_abs_corr")
        if mean_r is not None:
            if mean_r > 0.7:
                level = "**very high** — the large majority of feature pairs are highly correlated"
            elif mean_r > 0.5:
                level = "**high** — substantial redundancy in the feature set"
            elif mean_r > 0.3:
                level = "**moderate** — selective redundancy, many groups diverse"
            else:
                level = "**low** — feature set largely non-redundant"
            W(f"- Overall redundancy: {level} (mean |ρ| = {mean_r:.4f})")

        # Drop impact
        n95 = s.get("drop_candidates_095", 0)
        rr  = s.get("retain_rate_095")
        if n95 and rr:
            W(f"- Aggressive pruning @0.95: {n95} features removable, {rr:.1f}% of the set remains")

        n85 = s.get("drop_candidates_085", 0)
        rr85 = s.get("retain_rate_085")
        if n85 and rr85:
            W(f"- Moderate pruning @0.85: {n85} features removable, {rr85:.1f}% of the set remains")

        top = s.get("top_redundant_group")
        top_pct = s.get("top_redundant_pct")
        if top and top_pct:
            W(f"- Most redundant group: `{top}` ({top_pct:.1f}% of pairs > 0.95)")

        ax_name = s.get("most_redundant_axis")
        ax_corr = s.get("most_redundant_axis_corr")
        if ax_name and ax_corr:
            W(f"- main redundancy axis: `{ax_name}` (mean |ρ| = {ax_corr:.3f})")

        perf = s.get("n_perfect_pairs", 0)
        if perf:
            W(f"- {perf} near-perfect pairs (|ρ|>0.999) — check for genuine duplicates")

        W()

    if len(assets) >= 2:
        a1, a2 = assets[0].lower(), assets[1].lower()
        s1, s2 = stats_map[a1], stats_map[a2]
        W(f"### BTC vs ETH")
        W()
        if s1.get("mean_abs_corr") and s2.get("mean_abs_corr"):
            diff = s1["mean_abs_corr"] - s2["mean_abs_corr"]
            if abs(diff) < 0.01:
                W("- Both assets show a **very similar** redundancy structure (Δ mean |ρ| < 0.01)")
            elif diff > 0:
                W(f"- BTC is **more redundant** than ETH (Δ = {diff:+.4f} in mean |ρ|)")
            else:
                W(f"- ETH is **more redundant** than BTC (Δ = {diff:+.4f} in mean |ρ|)")

        if not cross_df.empty:
            both_agree = ((cross_df[f"{a1}_pct_above_095"] > 50) &
                          (cross_df[f"{a2}_pct_above_095"] > 50)).sum()
            W(f"- {both_agree} groups are strongly redundant in **both assets** "
              f"→ safe drop candidates regardless of asset")
            asset_spec = (
                ((cross_df[f"{a1}_pct_above_095"] > 50) & (cross_df[f"{a2}_pct_above_095"] <= 20)) |
                ((cross_df[f"{a2}_pct_above_095"] > 50) & (cross_df[f"{a1}_pct_above_095"] <= 20))
            ).sum()
            if asset_spec > 0:
                W(f"- {asset_spec} groups show **asset-specific** redundancy "
                  f"→ separate per-asset drop decision recommended")

        dd = drop_diff.get("drop_095", {})
        n_both = len(dd.get("both", []))
        if n_both:
            W(f"- {n_both} features identified as drop candidate @0.95 by **both assets** "
              f"→ highest priority for removal")
        W()

    W("---")
    W(f"*Report generated by `phase_a_summary.py` at {now}*")
    W()

    return "\n".join(lines), vif_tier_rows


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Phase A Consolidated Summary (3.4.2)")
    parser.add_argument("--assets",     nargs="+", default=DEFAULT_ASSETS)
    parser.add_argument("--base-dir",   default=DEFAULT_BASE_DIR,
                        help="Root directory of the project (default: .)")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help="Where reports are written")
    parser.add_argument("--catalog",    default="results/selection/feature_catalog.csv")
    parser.add_argument("--skip-cc",  action="store_true", help="Skip cross-concept section")
    parser.add_argument("--skip-ca",  action="store_true", help="Skip ca (S6) section")
    parser.add_argument("--skip-vif", action="store_true", help="Skip VIF section")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading data for assets: {args.assets}")
    data_map     = {}    # within_concept_correlation
    stats_map    = {}
    cc_data_map  = {}    # cross_concept
    ca_data_map  = {}    # ca_correlation
    vif_data_map = {}    # vif

    for asset in args.assets:
        akey = asset.lower()

        # Section 1: within_concept_correlation
        print(f"  [{asset.upper()}] within_concept_correlation ← {os.path.join(args.base_dir, CE_DIR)}")
        d = load_asset_data(args.base_dir, asset)
        data_map[akey]  = d
        stats_map[akey] = compute_asset_stats(asset, d)
        s = d["summary"]; p = d["pairs"]
        print(f"  [{asset.upper()}]   {len(s) if s is not None else 0} groups, "
              f"{len(p) if p is not None else 0} high-corr pairs")

        # Section 2: cross_concept
        if not args.skip_cc:
            cc_data_map[akey] = load_cross_concept_data(args.base_dir, asset)
            cs = cc_data_map[akey].get("summary")
            print(f"  [{asset.upper()}]   cross_concept: "
                  f"{len(cs) if cs is not None else 0} families")

        # Section 3: ca_correlation
        if not args.skip_ca:
            ca_data_map[akey] = load_ca_data(args.base_dir, asset)
            cas = ca_data_map[akey].get("s6_vs_s5_summary")
            print(f"  [{asset.upper()}]   ca_correlation: "
                  f"{len(cas) if cas is not None else 0} S6↔S5 families")

        # Section 4: VIF
        if not args.skip_vif:
            vif_data_map[akey] = load_vif_data(args.base_dir, asset)
            vr = vif_data_map[akey].get("results")
            print(f"  [{asset.upper()}]   vif: "
                  f"{len(vr) if vr is not None else 0} features")

    # Cross-asset (within-concept)
    cross_df  = cross_asset_group_comparison(data_map)
    drop_diff = cross_asset_drop_diff(data_map)

    # Build report
    report_md, vif_tier_rows = build_report(
        stats_map    = stats_map,
        data_map     = data_map,
        cross_df     = cross_df,
        drop_diff    = drop_diff,
        assets       = [a.lower() for a in args.assets],
        catalog_path = args.catalog,
        cc_data_map  = cc_data_map  if not args.skip_cc  else None,
        ca_data_map  = ca_data_map  if not args.skip_ca  else None,
        vif_data_map = vif_data_map if not args.skip_vif else None,
    )

    # Save markdown
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    md_path = os.path.join(args.output_dir, f"phase_a_report_{ts}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"\nReport saved: {md_path}")

    # CSV exports
    if not cross_df.empty:
        csv_path = os.path.join(args.output_dir, "cross_asset_comparison.csv")
        cross_df.to_csv(csv_path, index=False)
        print(f"  CSV: {csv_path}")

    # Axis disaggregation CSV (combining cross_concept axis_summary across assets)
    if not args.skip_cc and cc_data_map:
        axis_frames = []
        for asset, d in cc_data_map.items():
            ax = d.get("axis")
            if ax is not None and not ax.empty:
                ax = ax.copy(); ax.insert(0, "asset", asset.upper())
                axis_frames.append(ax)
        if axis_frames:
            axis_all = pd.concat(axis_frames, ignore_index=True)
            csv_path = os.path.join(args.output_dir, "axis_disaggregation.csv")
            axis_all.to_csv(csv_path, index=False)
            print(f"  CSV: {csv_path}")

    # VIF tier comparison CSV
    if vif_tier_rows:
        csv_path = os.path.join(args.output_dir, "vif_tier_comparison.csv")
        pd.DataFrame(vif_tier_rows).to_csv(csv_path, index=False)
        print(f"  CSV: {csv_path}")

    # Print takeaways to stdout
    print("\n" + "=" * 70)
    print("MAIN TAKEAWAYS (within-concept)")
    print("=" * 70)
    for asset in args.assets:
        s = stats_map[asset.lower()]
        print(f"\n  {s['asset']}")
        print(f"    Features: {s.get('n_features_total','n/a')}  "
              f"| Groups: {s.get('n_groups_total','n/a')}  "
              f"| Mean|ρ|: {_fmt(s.get('mean_abs_corr'),'.4f')}")
        print(f"    Drop @0.95: {s.get('drop_candidates_095',0)}  "
              f"| Retain: {_fmt(s.get('retain_rate_095'),'.1f')}%  "
              f"| Near-perfect pairs: {s.get('n_perfect_pairs',0)}")
        print(f"    Most redundant group: {s.get('top_redundant_group','n/a')} "
              f"({_fmt(s.get('top_redundant_pct'),'.1f')}% pairs > 0.95)")

    if len(args.assets) >= 2 and not cross_df.empty:
        a1, a2 = args.assets[0].lower(), args.assets[1].lower()
        both_high = ((cross_df[f"{a1}_pct_above_095"] > 50) &
                     (cross_df[f"{a2}_pct_above_095"] > 50)).sum()
        dd95_both = len(drop_diff.get("drop_095", {}).get("both", []))
        print(f"\n  CROSS-ASSET")
        print(f"    Groups redundant in both assets: {both_high}")
        print(f"    Drop candidates shared @0.95:   {dd95_both}")
    print("=" * 70)


if __name__ == "__main__":
    main()