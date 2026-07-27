#!/usr/bin/env python3
"""
Server-side pre-analysis of the --full-select run.
Reads cluster_mfe_windows_*.csv + cluster_da_multihz_*.csv in the given dir,
merges per cluster, and writes two SMALL csvs:
  reselect_merged.csv    -- every cluster: ratio@each window + DA@window + n_test
  reselect_candidates.csv-- clusters passing ratio60>=1.0 AND DA_win>0.55 AND n_test>=100
Also prints the 300s CONTROL for known clusters so we can verify correctness.

Run:  python3 reselect_analyze.py <results_dir>
"""
import sys, glob, os, re
import pandas as pd, numpy as np

TAG_RE = re.compile(r"cluster_mfe_windows_(kmeans|gmm)_pca(\d+|none)_k(\d+)_(\w+?)_(\d+s)_(\d+)bps\.csv")

def parse_tag(fname):
    m = TAG_RE.search(os.path.basename(fname))
    if not m:
        return None
    method, pca, k, asset, hz, thr = m.groups()
    tag = f"{method}_pca{pca}_k{k}_{asset}_{hz}_{thr}bps"
    return dict(tag=tag, method=method, pca=pca, k=int(k),
                asset=asset, hz=hz, thr=int(thr))


if __name__ == "__main__":
    DIR = sys.argv[1] if len(sys.argv) > 1 else "."
    rows = []
    win_files = sorted(glob.glob(os.path.join(DIR, "cluster_mfe_windows_*.csv")))
    print(f"Found {len(win_files)} window files in {DIR}\n")

    for wf in win_files:
        meta = parse_tag(wf)
        if meta is None:
            continue
        daf = os.path.join(DIR, f"cluster_da_multihz_{meta['tag']}.csv")
        try:
            w = pd.read_csv(wf)
        except Exception:
            continue
        da = pd.read_csv(daf) if os.path.exists(daf) else None

        for cl in sorted(w.cluster.unique()):
            wc = w[w.cluster == cl]
            rr = {int(r.window_s): r.ratio for _, r in wc.iterrows()}
            td = int(wc.trade_dir.iloc[0])
            n_screen = int(wc.n.iloc[0])
            # DA at the config's window horizon + a few extra horizons
            da_win = da_60 = da_max = n_test = np.nan
            if da is not None:
                dc = da[da.cluster == cl]
                row_win = dc[dc.horizon == meta["hz"]]
                if len(row_win):
                    da_win = float(row_win.da_oos.iloc[0])
                    n_test = int(row_win.n_test.iloc[0])
                row60 = dc[dc.horizon == "60s"]
                if len(row60):
                    da_60 = float(row60.da_oos.iloc[0])
                if len(dc):
                    da_max = float(dc.da_oos.max())
            rows.append(dict(
                asset=meta["asset"], hz=meta["hz"], thr=meta["thr"],
                pca=meta["pca"], k=meta["k"], cluster=int(cl),
                trade_dir=td, n=n_screen,
                ratio_15=rr.get(15, np.nan), ratio_30=rr.get(30, np.nan),
                ratio_60=rr.get(60, np.nan), ratio_120=rr.get(120, np.nan),
                ratio_300=rr.get(300, np.nan),
                DA_win=da_win, DA_60=da_60, DA_max=da_max, n_test=n_test))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(DIR, "reselect_merged.csv"), index=False)
    print(f"merged -> reselect_merged.csv ({len(df)} clusters total)\n")

    # ---- 300s CONTROL: known clusters must reproduce the old screening ratios ----
    print("=== 300s CONTROL (must match old screening) ===")
    ctrl = df[(df.asset == "btc") & (df.hz == "15s") & (df.thr == 30) &
              (df.pca == "150") & (df.k == 6)]
    print("BTC 15s/30 pca150 k6 (old: Cl0 cont 1.53, Cl4 rev 0.76, Cl1 cont 0.58):")
    print(ctrl[["cluster", "trade_dir", "n", "ratio_300", "ratio_60", "DA_win", "n_test"]]
          .to_string(index=False))
    ctrl2 = df[(df.asset == "eth") & (df.hz == "5s") & (df.thr == 20) &
               (df.pca == "300") & (df.k == 6)]
    print("\nETH 5s/20 pca300 k6 (old: Cl3 rev 0.39):")
    print(ctrl2[["cluster", "trade_dir", "n", "ratio_300", "ratio_60", "DA_win", "n_test"]]
          .to_string(index=False))

    # ---- CANDIDATES: ratio@60s>=1.0 AND DA_win>0.55 AND n_test>=100 ----
    cand = df[(df.ratio_60 >= 1.0) & (df.DA_win > 0.55) & (df.n_test >= 100)].copy()
    cand = cand.sort_values(["DA_win", "ratio_60"], ascending=False)
    cand.to_csv(os.path.join(DIR, "reselect_candidates.csv"), index=False)
    print(f"\n=== CANDIDATES (ratio60>=1.0 AND DA_win>0.55 AND n_test>=100): "
          f"{len(cand)} ===")
    if len(cand):
        print(cand[["asset", "hz", "thr", "pca", "k", "cluster", "trade_dir",
                    "n", "ratio_60", "ratio_300", "DA_win", "DA_60", "n_test"]]
              .to_string(index=False))
    print("\ncandidates -> reselect_candidates.csv")
    print("\nUpload ONLY: reselect_merged.csv + reselect_candidates.csv (both small).")
