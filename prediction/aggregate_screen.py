# aggregate_screen.py
# ==============================================================================
# Aggregate the trailing GRID SCREENING (Section 4.4.3), direction-agnostic.
#
# Reads every cluster_screen_*.csv produced by cluster_engine
# (--no-da-gate run) and answers the selection questions we agreed on:
#
#   * PRIMARY ranking  : MFE/MAE ratio  (tradable asymmetry in each cluster's
#                        OWN direction — continuation OR reversal)
#   * SECONDARY ranking: MFE-lift       (favourable excursion vs all-breakout mean)
#   * cluster cutoff    : n >= --min-n  (default 100)
#   * per ASSET separately
#   * trade_dir         : +1 continuation, -1 reversal (direction-agnostic screen)
#
# SELECTION-BIAS CONTROL (professor's point 1):
#   Each config carries a PERMUTATION p-value (perm_p) on its best-cluster ratio
#   (the max over the grid is inflated even under no effect). This script applies
#   a Benjamini-Hochberg FDR correction ACROSS configs and exports a null-band
#   table for the ECDF figure, so the apparently positive cells can be shown to
#   sit within / outside the null.
#
# USAGE (on the server, from .):
#   python -m prediction.aggregate_screen \
#       --screen-dir results/cluster_screen_v2 --min-n 100
# ==============================================================================
from __future__ import annotations
import argparse
import glob
import os
import re
import numpy as np
import pandas as pd


def parse_pca_k(pca_str: str):
    """pca column is like 'pca150_k10' -> (pca_dim=150, k=10). Fallback only;
    the screening CSV now writes pca_dim and k as direct columns."""
    m = re.match(r"pca(\d+)_k(\d+)", str(pca_str))
    if m:
        return int(m.group(1)), int(m.group(2))
    return np.nan, np.nan


def benjamini_hochberg(pvals):
    """BH step-up FDR. Returns q-values aligned to the input order."""
    p = np.asarray(pvals, dtype=float)
    ok = ~np.isnan(p)
    q = np.full(p.shape, np.nan)
    if ok.sum() == 0:
        return q
    pv = p[ok]
    n = pv.size
    order = np.argsort(pv)
    ranked = pv[order]
    q_ranked = ranked * n / np.arange(1, n + 1)
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q_sorted = np.empty(n)
    q_sorted[order] = np.clip(q_ranked, 0, 1)
    q[ok] = q_sorted
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen-dir", default="results/cluster_screen_v2",
                    help="folder with cluster_screen_*.csv")
    ap.add_argument("--min-n", type=int, default=100,
                    help="minimum events per cluster for the viable selection")
    ap.add_argument("--out-dir", default=None,
                    help="where to write aggregated CSVs (default: --screen-dir)")
    ap.add_argument("--ratio-min", type=float, default=1.5,
                    help="viable double filter: min MFE/MAE ratio (default 1.5)")
    ap.add_argument("--lift-min", type=float, default=1.0,
                    help="viable double filter: min MFE-lift (default 1.0)")
    ap.add_argument("--fdr", type=float, default=0.10,
                    help="FDR level for the Benjamini-Hochberg survivor count")
    a = ap.parse_args()
    out_dir = a.out_dir or a.screen_dir

    files = sorted(glob.glob(os.path.join(a.screen_dir, "cluster_screen_*.csv")))
    if not files:
        raise SystemExit(f"No cluster_screen_*.csv in {a.screen_dir}")

    frames = []
    for f in files:
        try:
            df = pd.read_csv(f)
            if len(df):
                frames.append(df)
        except Exception as e:
            print(f"[skip] {os.path.basename(f)}: {e}")
    allc = pd.concat(frames, ignore_index=True)

    # pca_dim / k are direct columns now; parse from the pca string only if missing
    if "pca_dim" not in allc.columns or allc["pca_dim"].isna().all():
        pk = allc["pca"].apply(parse_pca_k)
        allc["pca_dim"] = [x[0] for x in pk]
    if "k" not in allc.columns:
        pk = allc["pca"].apply(parse_pca_k)
        allc["k"] = [x[1] for x in pk]

    has_perm = "perm_p" in allc.columns
    has_dir = "trade_dir" in allc.columns
    cfg_cols = ["asset", "hz", "thr_bps", "pca_dim", "k"]

    print("=" * 70)
    print(f"Loaded {len(files)} screening tables, {len(allc)} cluster-rows total.")
    print(f"Configs (grid cells): {allc[cfg_cols].drop_duplicates().shape[0]}")
    for asset, g in allc.groupby("asset"):
        print(f"  {asset}: {len(g)} clusters, "
              f"{g[cfg_cols].drop_duplicates().shape[0]} configs")
    if has_dir:
        print(f"  direction: {int((allc['trade_dir']==1).sum())} continuation, "
              f"{int((allc['trade_dir']==-1).sum())} reversal clusters (all n)")

    # ── cluster cutoff ──
    big = allc[allc["n"] >= a.min_n].copy()
    print("\n" + "=" * 70)
    print(f"After cluster cutoff n >= {a.min_n}:")
    for asset, g in big.groupby("asset"):
        line = f"  {asset}: {len(g)} clusters remain"
        if has_dir:
            line += (f"  ({int((g['trade_dir']==1).sum())} cont / "
                     f"{int((g['trade_dir']==-1).sum())} rev)")
        print(line)

    # ── MFE/MAE ratio distribution per asset ──
    print("\n" + "=" * 70)
    print(f"MFE/MAE ratio distribution over clusters (n >= {a.min_n}):")
    pcts = [50, 75, 90, 95, 99, 100]
    for asset, g in big.groupby("asset"):
        r = g["mfe_mae_ratio"].dropna().values
        if r.size:
            print(f"  {asset}: " + "  ".join(f"p{p}={q:.3f}"
                                             for p, q in zip(pcts, np.percentile(r, pcts))))

    # ── SELECTION-BIAS CONTROL: permutation null + Benjamini-Hochberg ──
    sig = None
    if has_perm:
        sig_cols = [c for c in (cfg_cols + ["perm_best_ratio", "perm_p",
                    "null_ratio_p50", "null_ratio_p95", "null_ratio_p99",
                    "n_perm"]) if c in allc.columns]
        sig = (allc[sig_cols].drop_duplicates(subset=cfg_cols)
               .reset_index(drop=True))
        sig["bh_q"] = benjamini_hochberg(sig["perm_p"].values)
        if "null_ratio_p95" in sig.columns:
            sig["above_null_p95"] = sig["perm_best_ratio"] > sig["null_ratio_p95"]
        if "null_ratio_p99" in sig.columns:
            sig["above_null_p99"] = sig["perm_best_ratio"] > sig["null_ratio_p99"]

        print("\n" + "=" * 70)
        print("SELECTION-BIAS CONTROL (permutation null on best-cluster ratio)")
        n_perm_used = int(sig["n_perm"].iloc[0]) if "n_perm" in sig.columns else 0
        print(f"  {len(sig)} configs (hypotheses), {n_perm_used} permutations each")
        for asset, g in sig.groupby("asset"):
            raw_sig = int((g["perm_p"] < 0.05).sum())
            bh_sig = int((g["bh_q"] <= a.fdr).sum())
            ab95 = int(g.get("above_null_p95", pd.Series(dtype=bool)).sum())
            print(f"  {asset}: {raw_sig}/{len(g)} raw p<0.05 | "
                  f"{bh_sig}/{len(g)} survive BH at FDR {a.fdr} | "
                  f"{ab95}/{len(g)} above own null p95")
        pp = sig["perm_p"].dropna().values
        print("  perm_p distribution: " +
              "  ".join(f"p{p}={np.percentile(pp, p):.4f}" for p in [1, 5, 10, 25, 50]))

    # ── per-config BEST cluster ──
    best_per_cfg = (big.sort_values(["mfe_mae_ratio", "mfe_lift"], ascending=[False, False])
                    .groupby(cfg_cols, as_index=False).first()
                    .sort_values(["asset", "mfe_mae_ratio", "mfe_lift"],
                                 ascending=[True, False, False]))
    if sig is not None:
        best_per_cfg = best_per_cfg.merge(sig[cfg_cols + ["perm_p", "bh_q"]],
                                          on=cfg_cols, how="left")

    show = ["asset", "hz", "thr_bps", "pca_dim", "k", "cluster", "n",
            "mfe_mean_bps", "mae_mean_bps", "mfe_mae_ratio", "mfe_lift"]
    if has_dir:
        show += ["trade_dir"]
    if has_perm:
        show += ["perm_p", "bh_q"]

    print("\n" + "=" * 70)
    print("TOP 15 CONFIGS per asset (by best cluster's MFE/MAE ratio):")
    for asset, g in best_per_cfg.groupby("asset"):
        print(f"\n  ── {asset.upper()} ──")
        print(g[[c for c in show if c in g.columns]].head(15).to_string(index=False))

    # ── VIABLE double filter ──
    viable = big[(big["mfe_mae_ratio"] >= a.ratio_min) &
                 (big["mfe_lift"] >= a.lift_min)].copy()
    if sig is not None:
        viable = viable.merge(sig[cfg_cols + ["perm_p", "bh_q"]],
                              on=cfg_cols, how="left")
    viable = viable.sort_values(["asset", "mfe_mae_ratio", "mfe_lift"],
                                ascending=[True, False, False])

    print("\n" + "=" * 70)
    print(f"VIABLE double filter: MFE/MAE ratio >= {a.ratio_min} "
          f"AND MFE-lift >= {a.lift_min} AND n >= {a.min_n}")
    for asset, g in viable.groupby("asset"):
        n_cfg = g[cfg_cols].drop_duplicates().shape[0]
        line = f"  {asset}: {len(g)} clusters across {n_cfg} configs"
        if has_dir:
            line += (f"  ({int((g['trade_dir']==1).sum())} cont / "
                     f"{int((g['trade_dir']==-1).sum())} rev)")
        if has_perm:
            line += f"  | {int((g['bh_q']<=a.fdr).sum())} at BH-q<={a.fdr}"
        print(line)

    print("\n  Full surviving list:")
    for asset, g in viable.groupby("asset"):
        print(f"\n  ── {asset.upper()} ──")
        print(g[[c for c in show if c in g.columns]].to_string(index=False))

    # ── TABLE 7 pivot ──
    print("\n" + "=" * 70)
    print(f"TABLE 7 pivot  (best cluster, n >= {a.min_n})")
    hz_order = ["1s", "5s", "15s", "30s"]
    thr_order = [10, 15, 20, 30, 40]
    for metric, lab in [("mfe_mae_ratio", "best MFE/MAE ratio"),
                        ("mfe_lift", "best MFE-lift")]:
        for asset, g in big.groupby("asset"):
            piv = (g.groupby(["hz", "thr_bps"])[metric].max().reset_index()
                     .pivot(index="hz", columns="thr_bps", values=metric))
            piv = piv.reindex(index=[h for h in hz_order if h in piv.index],
                              columns=[t for t in thr_order if t in piv.columns])
            print(f"\n  [{asset.upper()}] {lab}:")
            print("    hz \\ thr   " + "  ".join(f"{t:>6}" for t in piv.columns))
            for hz_v, row in piv.iterrows():
                print(f"    {hz_v:>6}     " +
                      "  ".join((f"{v:6.2f}" if pd.notna(v) else "     —") for v in row))

    # ── save ──
    all_out = os.path.join(out_dir, f"screen_ALL_n{a.min_n}.csv")
    cfg_out = os.path.join(out_dir, f"screen_config_best_n{a.min_n}.csv")
    via_out = os.path.join(out_dir,
                           f"screen_viable_r{a.ratio_min}_l{a.lift_min}_n{a.min_n}.csv")
    big.sort_values(["asset", "mfe_mae_ratio"], ascending=[True, False]).to_csv(all_out, index=False)
    best_per_cfg[[c for c in show if c in best_per_cfg.columns]].to_csv(cfg_out, index=False)
    viable[[c for c in show if c in viable.columns]].to_csv(via_out, index=False)
    saved = [all_out, cfg_out, via_out]
    if sig is not None:
        sig_out = os.path.join(out_dir, f"screen_significance_n{a.min_n}.csv")
        sig.to_csv(sig_out, index=False)
        saved.append(sig_out)

    print("\n" + "=" * 70)
    print("Saved:")
    for s in saved:
        print(f"  {s}")


if __name__ == "__main__":
    main()