#!/usr/bin/env python3
"""Bridge to the cluster: directional accuracy of a walk-forward LGBM entry model
on the SAME 30s/30bps breakouts, in the SAME frame as the cluster (sign(r_cont),
continuation/reversal). Two feature sets: all use_tree vs. top_returns.

Run:  python3 lgbm_breakout_da.py > lgbm_breakout_da.log 2>&1
"""
import sys, os, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats
import lightgbm as lgb
from common.data_loader import load_dataset

KEEP = "results/selection/feature_keep.csv"
HZ = ["1s", "5s", "15s", "30s", "60s", "120s", "300s", "900s"]
SEEDS = [42, 123, 999]
N_FOLDS = 5
THR_DEC = 30 / 1e4
EV_HZ = 30
LGB = dict(objective="binary", n_estimators=200, num_leaves=15, max_depth=4,
           learning_rate=0.05, min_child_samples=30, subsample=0.8,
           colsample_bytree=0.7, verbose=-1)
CANDS = [("btc", "BTC"), ("eth", "ETH")]


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


def run(asset, name, keep):
    print("=" * 66); print(f"### {name}"); print("=" * 66)
    tree_feats = keep[(keep.type == "feature") & (keep.use_tree)]["column"].tolist()
    topret = set(keep[keep.top_returns == True]["column"])

    X, y30_t, info, fnames = load_dataset(target="ret_30s", asset=asset,
                                          profile="all", max_hours=1000)
    n2c = {n: i for i, n in enumerate(fnames)}
    tree_cols = [n2c[f] for f in tree_feats if f in n2c]
    ret_cols = [n2c[f] for f in fnames if f in topret]
    print(f"  features: all_tree={len(tree_cols)}  top_returns={len(ret_cols)}")

    # horizon forward returns (light loads), clip to common length
    raw = {}; L = len(y30_t)
    for h in HZ:
        _, yh, _, _ = load_dataset(target=f"ret_{h}", asset=asset,
                                   max_hours=1000, target_only=True)
        raw[h] = yh; L = min(L, len(yh))
    L = min(L, len(X))
    tm = np.full(L, np.nan); tm[EV_HZ:] = y30_t[:L - EV_HZ]
    ev = np.where(np.abs(tm) > THR_DEC)[0]; ev = ev[ev < L]
    bdir = np.sign(tm[ev]); n = len(ev)
    print(f"  events={n}")
    rcont = {h: bdir * raw[h][ev] for h in HZ}
    folds = np.array_split(np.arange(n), N_FOLDS + 1)

    for fs_name, cols in [("all_tree", tree_cols), ("top_returns", ret_cols)]:
        Xev = X[ev][:, cols].astype(np.float32)
        print(f"\n  --- {fs_name} ({len(cols)} features) ---")
        for h in HZ:
            tgt = (rcont[h] > 0).astype(int)
            hits = []
            for s in SEEDS:
                p = dict(LGB); p["random_state"] = s
                for k in range(1, N_FOLDS + 1):
                    tr = np.concatenate(folds[:k]); te = folds[k]
                    if len(np.unique(tgt[tr])) < 2:
                        continue
                    m = lgb.LGBMClassifier(**p).fit(Xev[tr], tgt[tr])
                    hits.extend((m.predict(Xev[te]) == tgt[te]).astype(float).tolist())
            da, lo, hi = bca(hits)
            print(f"    {h:>5}: DA={da:.3f} [{lo:.3f},{hi:.3f}]  n={len(hits)}")


if __name__ == "__main__":
    keep = pd.read_csv(KEEP)
    for asset, name in CANDS:
        run(asset, name, keep)
