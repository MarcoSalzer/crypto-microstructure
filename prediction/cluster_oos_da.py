#!/usr/bin/env python3
"""Cluster method, honest OOS: fit scaler+PCA+KMeans on TRAIN only, fix each
cluster's majority direction on train, pick the best cluster by TRAIN DA at 60s,
assign TEST events by predict, then measure that cluster's DA across all 8
horizons. Walk-forward, 3 folds x 3 seeds. Directly comparable to lgbm_breakout_da.py
(LGBM) on the same events and the same r_cont frame.

Run:  python3 cluster_oos_da.py > cluster_oos_da.log 2>&1
"""
import sys, os, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from common.data_loader import load_dataset

HZ = ["1s", "5s", "15s", "30s", "60s", "120s", "300s", "900s"]
SEL_HZ = "60s"            # cluster chosen by train DA at this horizon
SEEDS = [42, 123, 999]
N_FOLDS = 3
THR_DEC = 30 / 1e4
EV_HZ = 30
K = 8
MIN_TR = 100            # min train events for a cluster to be selectable
CANDS = [("btc", 600, "BTC"), ("eth", 300, "ETH")]


def bca(hits, B=3000, alpha=0.05, seed=42):
    hits = np.asarray(hits, float); n = len(hits); th = hits.mean()
    if n < 20:
        return th, np.nan, np.nan
    rng = np.random.RandomState(seed)
    boot = hits[rng.randint(0, n, (B, n))].mean(1)
    z0 = stats.norm.ppf((boot < th).mean()) if 0 < (boot < th).mean() < 1 else 0.0
    jk = (hits.sum() - hits) / (n - 1); d = jk.mean() - jk
    den = 6 * (np.sum(d**2)**1.5); a = np.sum(d**3)/den if den else 0.0
    def adj(z): zz = z0 + z; return stats.norm.cdf(z0 + zz/(1 - a*zz))
    return (th, np.percentile(boot, 100*adj(stats.norm.ppf(alpha/2))),
            np.percentile(boot, 100*adj(stats.norm.ppf(1-alpha/2))))


def run(asset, pca_dim, name):
    print("=" * 60); print(f"### {name} (pca{pca_dim}, k{K})"); print("=" * 60)
    X, y30, info, fnames = load_dataset(target="ret_30s", asset=asset,
                                        profile="cluster", max_hours=1000)
    raw = {}; L = len(y30)
    for h in HZ:
        _, yh, _, _ = load_dataset(target=f"ret_{h}", asset=asset,
                                   max_hours=1000, target_only=True)
        raw[h] = yh; L = min(L, len(yh))
    L = min(L, len(X))
    tm = np.full(L, np.nan); tm[EV_HZ:] = y30[:L - EV_HZ]
    ev = np.where(np.abs(tm) > THR_DEC)[0]; ev = ev[ev < L]
    bdir = np.sign(tm[ev]); n = len(ev)
    Xev = X[ev]
    rcont = {h: bdir * raw[h][ev] for h in HZ}       # continuation(+)/reversal(-)
    print(f"  events={n}")

    folds = np.array_split(np.arange(n), N_FOLDS + 1)
    hits = {h: [] for h in HZ}
    sel_info = []
    for k in range(1, N_FOLDS + 1):
        tr = np.concatenate(folds[:k]); te = folds[k]
        for seed in SEEDS:
            # fit preprocessing + clustering on TRAIN only
            imp = SimpleImputer(strategy="median")
            sc = StandardScaler()
            vc = np.where(np.isnan(Xev[tr]).mean(axis=0) < 0.95)[0]
            Xtr = sc.fit_transform(imp.fit_transform(Xev[tr][:, vc]))
            Xtr = np.nan_to_num(Xtr, nan=0.0)
            nco = min(pca_dim, Xtr.shape[1], len(tr) - 1)
            pca = PCA(n_components=nco, random_state=42, svd_solver="auto")
            Ptr = pca.fit_transform(Xtr)
            km = KMeans(n_clusters=K, n_init=20, random_state=seed).fit(Ptr)
            ltr = km.labels_

            # majority direction + train DA at SEL_HZ per cluster
            rc_sel = rcont[SEL_HZ][tr]
            best_c, best_da, best_maj = -1, -1, 0
            for c in range(K):
                mask = ltr == c
                if mask.sum() < MIN_TR:
                    continue
                maj = 1 if np.mean(np.sign(rc_sel[mask]) > 0) >= 0.5 else -1
                da_tr = np.mean(np.sign(rc_sel[mask]) == maj)
                if da_tr > best_da:
                    best_c, best_da, best_maj = c, da_tr, maj
            if best_c < 0:
                continue
            sel_info.append((best_da, best_maj))

            # assign TEST events, measure DA of chosen cluster across horizons
            Xte = np.nan_to_num(sc.transform(imp.transform(Xev[te][:, vc])), nan=0.0)
            lte = km.predict(pca.transform(Xte))
            in_c = lte == best_c
            for h in HZ:
                rc = rcont[h][te][in_c]
                good = np.isfinite(rc) & (rc != 0)
                hits[h].extend((np.sign(rc[good]) == best_maj).astype(float).tolist())

    print(f"  avg train DA@{SEL_HZ} of chosen cluster: {np.mean([s[0] for s in sel_info]):.3f}")
    print(f"  {'hz':>5} {'DA_oos':>7} {'lo':>7} {'hi':>7} {'n':>6}")
    for h in HZ:
        da, lo, hi = bca(hits[h])
        print(f"  {h:>5} {da:7.3f} {lo:7.3f} {hi:7.3f} {len(hits[h]):>6}")


if __name__ == "__main__":
    for asset, pca_dim, name in CANDS:
        run(asset, pca_dim, name)
