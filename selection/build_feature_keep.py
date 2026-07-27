#!/usr/bin/env python3
"""build_feature_keep.py — deterministic construction of feature_keep.csv.

Background
----------
`feature_keep.csv` is the load-time feature selector consumed by
``common.data_loader`` (it decides, via the ``use_{profile}`` flags, which
columns each model reads from the ml_features parquets). This script constructs
it deterministically from the committed reduction artifacts, so the selection is
fully reproducible from the material in this repository.

Inputs  (all under results/selection/)
-------
  feature_catalog.csv               one row per generated column (+ is_feature/meta/target, group, stage)
  consolidated_drop_list.csv        the 239 dropped bare-names (LWP-structural + FI-universal-weak)
  reduction_inputs/{btc,eth}_vif_results.csv                     per-asset VIF; pooled here as union-min -> vif/use_linear
  reduction_inputs/{btc,eth}_drop_candidates_095.csv            intra-concept corr drops -> use_cluster
  reduction_inputs/{btc,eth}_cross_concept_drop_candidates.csv  cross-concept corr drops -> use_cluster
  reduction_inputs/fi_top_annotations.csv                       top_returns/mfe/mae/... flags

Transform (see docs/methodology.md, thesis section 3.4)
---------
  rows      = catalog features minus the 297 dropped columns, plus metadata
              (minus the __index_level_0__ pandas artefact) and targets.
  type      = feature/meta/target from the catalog.
  source    = "S6" if stage == S6 else "S5".
  bundle    = GROUP_TO_BUNDLE[group]  (features only).
  family    = group, with the six `basis_*` Price features relabelled Cross-Market.
  use_tree  = feature AND not one of the 77 absolute price/level columns.
  use_anomaly = use_tree (the neural profile reads the tree profile).
  vif       = pooled VIF = min(btc_vif, eth_vif) per bare_name (nulled for absolute levels).
  use_linear  = feature AND pooled VIF <= 10.
  use_cluster = feature AND bare-name not in the union of the correlation drop sets.
  top_*     = from fi_top_annotations.

Test criterion
--------------
  NOT byte-equality (feature_keep.csv carries a pandas __index_level_0__
  artefact row that the construction omits). Instead:
    - identical row set over `column`
    - identical values in ALL 17 columns
    - assertions: use_tree == 3270, use_linear == 363, use_cluster == 2746,
      type == feature == 3347
All of the above pass. Run ``python -m selection.build_feature_keep --verify``.
"""
from __future__ import annotations

import argparse
import re
import sys

import numpy as np
import pandas as pd

from common.paths import REDUCTION_DIR

OUT_COLUMNS = [
    "column", "type", "asset", "source", "bare_name", "bundle", "family",
    "use_tree", "use_linear", "use_cluster", "use_anomaly", "vif",
    "top_returns", "top_mfe", "top_mae", "top_short_horizon", "top_long_horizon",
]

GROUP_TO_BUNDLE = {
    "Absorption": "B3_flow", "Activity": "B3_flow", "Aggression": "B1_imbalance",
    "Bookshape": "B2_bookshape", "Cross-Asset": "B7_cross_asset",
    "Cross-Asset-Intermediary": "B7_cross_asset", "Cross-Market": "B4_cross_market",
    "Dynamics": "B5_dynamics", "Imbalance": "B1_imbalance", "Impact": "B2_bookshape",
    "Level Artefact": "B6_context", "Level Events": "B6_context",
    "Liquidity Events": "B3_flow", "Normalization": "B5_dynamics",
    "Pressure": "B1_imbalance", "Price": "B6_context", "Range": "B6_context",
    "Session Levels": "B6_context", "Trend": "B6_context",
    "Volume Profile": "B6_context", "Volume Profile Artefact": "B6_context",
}


def _rdir(name: str) -> pd.DataFrame:
    return pd.read_csv(REDUCTION_DIR / name)


def _expand(bare: str, feat_cols: set) -> set:
    return {f"{bare}_btc", f"{bare}_eth", f"{bare}_btceth", bare} & feat_cols


# The 77 absolute price/level columns excluded from the tree/anomaly profile
# (thesis 3.4.3/3.5): 27 Level-Artefact + 18 Volume-Profile-Artefact + 16 Price
# (best_bid/best_ask/mid/vwap at 1 s) + 16 Trend (EMAs). Verified == 77.
_PRICE_ABS = re.compile(r"^(best_bid|best_ask|mid|vwap)_(fut|spot)_1s$")


def absolute_level_columns(cat: pd.DataFrame, exclude_mid_vwap: bool = False) -> set:
    """The absolute price/level columns removed from the model profiles.

    exclude_mid_vwap=False -> the 77 removed from use_tree/use_anomaly.
    exclude_mid_vwap=True  -> the 69 additionally removed from use_cluster
                              (best_bid/ask kept, but mid/vwap were already
                              correlation-dropped from the cluster profile)."""
    is_feat = cat["is_feature"]
    grp = cat["group"].fillna("")
    bare = cat["bare_name"].fillna("")
    # Level artefacts: all. Volume-Profile artefacts: all EXCEPT poc_migration,
    # which the thesis keeps as the one relative VP feature (3.2.2).
    artefact = (grp == "Level Artefact") | (
        (grp == "Volume Profile Artefact") & ~bare.str.startswith("poc_migration"))
    price_abs = (grp == "Price") & bare.str.match(_PRICE_ABS)
    if exclude_mid_vwap:
        price_abs &= bare.str.startswith(("best_bid", "best_ask"))
    # Absolute EMA levels (ema_50_*, ema_200_*) — NOT the relative ema_slope_*.
    trend_abs = (grp == "Trend") & bare.str.match(r"^ema_\d")
    mask = is_feat & (artefact | price_abs | trend_abs)
    return set(cat.loc[mask, "column"])


def correlation_cluster_drops() -> set:
    """Union of intra- and cross-concept correlation drop candidates (bare names)."""
    names: set = set()
    for asset in ("btc", "eth"):
        for fn in (f"{asset}_drop_candidates_095.csv",
                   f"{asset}_cross_concept_drop_candidates.csv"):
            df = _rdir(f"reduction_inputs/{fn}")
            names.update(df["feature_name"].astype(str))
    return names


def build() -> pd.DataFrame:
    cat = _rdir("feature_catalog.csv")
    drp = _rdir("consolidated_drop_list.csv")
    fi = _rdir("reduction_inputs/fi_top_annotations.csv").set_index("feature_name")

    feat_cols = set(cat.loc[cat["is_feature"], "column"])
    drop_cols: set = set()
    for bare in drp["feature_name"]:
        drop_cols |= _expand(bare, feat_cols)

    meta_cols = set(cat.loc[cat["is_meta"], "column"]) - {"__index_level_0__"}
    tgt_cols = set(cat.loc[cat["is_target"], "column"])
    keep_cols = (feat_cols - drop_cols) | meta_cols | tgt_cols

    # Pooled VIF = union-min over the two per-asset results, keyed by bare_name.
    # A feature is linear-stable if EITHER asset's VIF is below the threshold, so
    # the stored value is min(btc_vif, eth_vif) and both asset columns inherit it.
    vif = (pd.concat([_rdir("reduction_inputs/btc_vif_results.csv"),
                      _rdir("reduction_inputs/eth_vif_results.csv")])
           .groupby("bare_name")["vif"].min())

    abs_levels = absolute_level_columns(cat)   # 77 absolute price/level columns
    corr_drops = correlation_cluster_drops()

    rows = cat[cat["column"].isin(keep_cols)].copy()
    is_feat = rows["is_feature"]
    grp = rows["group"].fillna("")

    rows["type"] = np.where(rows["is_feature"], "feature",
                    np.where(rows["is_meta"], "meta", "target"))
    rows["source"] = np.where(rows["stage"] == "S6", "S6", "S5")
    rows["bundle"] = [GROUP_TO_BUNDLE.get(g) if f else np.nan
                      for g, f in zip(grp, is_feat)]
    fam = grp.copy()
    fam = fam.where(~((grp == "Price") & rows["bare_name"].str.startswith("basis")),
                    "Cross-Market")
    rows["family"] = fam
    rows["use_tree"] = is_feat & ~rows["column"].isin(abs_levels)
    rows["use_anomaly"] = rows["use_tree"]
    rows["vif"] = rows["bare_name"].map(vif)
    # Absolute price/level columns "carry no factor" (thesis 3.4.2): they enter
    # the pipeline as construction inputs, not model features, so their VIF is
    # nulled even though the pooled VIF run assigned them one.
    rows.loc[rows["column"].isin(abs_levels), "vif"] = np.nan
    rows["use_linear"] = is_feat & (rows["vif"] <= 10)
    rows["use_cluster"] = (is_feat & ~rows["bare_name"].isin(corr_drops)
                           & ~rows["column"].isin(abs_levels))
    for c in ["top_returns", "top_mfe", "top_mae", "top_short_horizon", "top_long_horizon"]:
        rows[c] = rows["column"].isin(fi.index[fi[c]])

    return rows[OUT_COLUMNS].sort_values("column").reset_index(drop=True)


def _norm(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("column").reset_index(drop=True)
    for c in ["use_tree", "use_linear", "use_cluster", "use_anomaly",
              "top_returns", "top_mfe", "top_mae", "top_short_horizon", "top_long_horizon"]:
        df[c] = df[c].astype(bool)
    return df


def verify() -> int:
    got = _norm(build())
    ref = _norm(pd.read_csv(REDUCTION_DIR / "feature_keep.csv")[OUT_COLUMNS])

    ok = True
    if set(got["column"]) != set(ref["column"]):
        print(f"FAIL row set: got {len(got)} vs ref {len(ref)}")
        ok = False
    g = got.set_index("column"); r = ref.set_index("column")
    common = g.index.intersection(r.index)
    for col in OUT_COLUMNS[1:]:
        a = g.loc[common, col]; b = r.loc[common, col]
        if col == "vif":
            mism = int((~np.isclose(a.fillna(-1), b.fillna(-1), rtol=1e-3, atol=1e-3)).sum())
        else:
            mism = int((a.fillna("∅").astype(str) != b.fillna("∅").astype(str)).sum())
        flag = "ok" if mism == 0 else f"MISMATCH ({mism})"
        if mism:
            ok = False
        print(f"  col {col:20} {flag}")

    asserts = {
        "use_tree==3270": int(got["use_tree"].sum()) == 3270,
        "use_linear==363": int(got["use_linear"].sum()) == 363,
        "use_cluster==2746": int(got["use_cluster"].sum()) == 2746,
        "feature==3347": int((got["type"] == "feature").sum()) == 3347,
    }
    print("  assertions:", {k: ("PASS" if v else "FAIL") for k, v in asserts.items()})
    print(f"  counts: use_tree={int(got['use_tree'].sum())} use_linear={int(got['use_linear'].sum())} "
          f"use_cluster={int(got['use_cluster'].sum())} feature={int((got['type']=='feature').sum())} "
          f"abs_levels={len(absolute_level_columns(_rdir('feature_catalog.csv')))}")
    print()
    print("NOTE: `vif` is the pooled VIF, computed as min(btc_vif, eth_vif) per")
    print("      bare_name (union logic — a feature is linear-stable if EITHER")
    print("      asset's VIF <= 10).")
    print("      Absolute price/level columns carry no factor (nulled). use_linear = vif <= 10.")
    ok = ok and all(asserts.values())
    print("RESULT (row set + all 17 columns + count assertions):",
          "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None,
                    help="write the built CSV here (default: REDUCTION_DIR/feature_keep.built.csv)")
    ap.add_argument("--verify", action="store_true",
                    help="verify against the committed feature_keep.csv")
    args = ap.parse_args()
    if args.verify:
        return verify()
    df = build()
    out = args.out or (REDUCTION_DIR / "feature_keep.built.csv")
    df.to_csv(out, index=False)
    print(f"wrote {len(df)} rows to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
