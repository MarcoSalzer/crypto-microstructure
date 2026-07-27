#!/usr/bin/env python3
"""4.4.7 Part B — walk-forward predictive test on the 50 signature features.

For each asset, over ALL 30s/30bps breakouts (no cluster prefilter):
  target  = sign(r_cont_h) = continuation(+)/reversal(-) at horizon h
  model   = LightGBM on the 50 candidate-cluster signature features only
  scheme  = 5 expanding walk-forward folds x 5 seeds, OOS
  outputs = (a) OOS DA per horizon with BCa band  -> comparable to Figure 7
            (b) feature importance (gain), aggregated
            (c) importance-rank constancy across folds/seeds

Run:  python3 feature_forward.py
"""
import sys, os
import numpy as np, pandas as pd
from scipy import stats
import lightgbm as lgb
from common.data_loader import load_dataset

HZ = ["1s", "5s", "15s", "30s", "60s", "120s", "300s", "900s"]
SEEDS = [42, 123, 999, 7, 31]
N_FOLDS = 5
THR_DEC = 30 / 1e4          # 30 bps
EV_HZ_SEC = 30              # breakout defined on the 30s trailing move
CANDS = [("btc", "cluster_signature_kmeans_pca600_k8_btc_30s_30bps.csv", 5, "BTC"),
         ("eth", "cluster_signature_kmeans_pca300_k8_eth_30s_30bps.csv", 1, "ETH")]
SIGDIR = "results/clustering/final"


def bca_ci(hits, B=3000, alpha=0.05, seed=42):
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


LGB = dict(objective="binary", n_estimators=200, num_leaves=15, max_depth=4,
           learning_rate=0.05, min_child_samples=30, subsample=0.8,
           colsample_bytree=0.7, verbose=-1)


def run_asset(asset, sigfile, cl, name):
    print("=" * 66); print(f"### {name}"); print("=" * 66)
    sig = pd.read_csv(f"{SIGDIR}/{sigfile}")
    feats50 = sig[sig.cluster == cl].sort_values("rank").head(50)["feature"].tolist()

    X, y30, info, fnames = load_dataset(target="ret_30s", asset=asset,
                                        profile="cluster", max_hours=1000)
    name_to_col = {n: i for i, n in enumerate(fnames)}
    missing = [f for f in feats50 if f not in name_to_col]
    if missing:
        print(f"  WARN {len(missing)} signature features not in matrix, e.g. {missing[:3]}")
    cols = [name_to_col[f] for f in feats50 if f in name_to_col]
    print(f"  using {len(cols)}/50 signature features")

    # events: trailing 30s move > 30 bps
    tm = np.full_like(y30, np.nan); tm[EV_HZ_SEC:] = y30[:len(y30)-EV_HZ_SEC]
    ev = np.where(np.abs(tm) > THR_DEC)[0]
    bdir = np.sign(tm[ev])
    Xev = X[ev][:, cols].astype(np.float32)
    n = len(ev); print(f"  events={n}")

    # forward returns per horizon (target_only loads are light).
    # Guard against slightly different row counts across loads (global sort on
    # non-monotonic timestamps): clip everything to a common minimum length.
    raw = {}
    Lmin = len(tm)
    for h in HZ:
        _, yh, _, _ = load_dataset(target=f"ret_{h}", asset=asset,
                                   max_hours=1000, target_only=True)
        raw[h] = yh; Lmin = min(Lmin, len(yh))
    ev = ev[ev < Lmin]
    bdir = np.sign(tm[ev])
    Xev = X[ev][:, cols].astype(np.float32)
    n = len(ev); print(f"  events={n} (clipped to common length {Lmin})")
    rcont = {h: bdir * raw[h][ev] for h in HZ}

    folds = np.array_split(np.arange(n), N_FOLDS + 1)
    imp_acc = np.zeros(len(cols))
    rows = []
    rev_prob_60, rev_true_60 = [], []   # pooled reversal prob + truth at 60s
    for h in HZ:
        tgt = (rcont[h] > 0).astype(int)      # 1 = continuation, 0 = reversal
        hits_all = []
        for s in SEEDS:
            p = dict(LGB); p["random_state"] = s
            for k in range(1, N_FOLDS + 1):
                tr = np.concatenate(folds[:k]); te = folds[k]
                if len(np.unique(tgt[tr])) < 2:
                    continue
                m = lgb.LGBMClassifier(**p).fit(Xev[tr], tgt[tr])
                pred = m.predict(Xev[te])
                hits_all.extend((pred == tgt[te]).astype(float).tolist())
                if h == "60s":
                    imp_acc += m.booster_.feature_importance(importance_type="gain")
                    proba = m.predict_proba(Xev[te])
                    rev_prob_60.extend(proba[:, 0].tolist())      # P(reversal)
                    rev_true_60.extend((tgt[te] == 0).astype(int).tolist())
        da, lo, hi = bca_ci(hits_all)
        rows.append((h, da, lo, hi, len(hits_all)))
        print(f"  {h:>5}: DA={da:.3f} [{lo:.3f},{hi:.3f}]  n_oos={len(hits_all)}")

    # Confidence-threshold test at 60s: precision (actual reversal rate) and
    # coverage among events the model calls reversal above each threshold.
    rp = np.array(rev_prob_60); rt = np.array(rev_true_60)
    base = rt.mean()
    print(f"\n  Reversal precision vs confidence threshold @60s (base rate={base:.3f}):")
    print(f"  {'thr':>5} {'precision':>10} {'coverage':>9} {'n':>7}")
    for thr in [0.50, 0.55, 0.60, 0.65, 0.70, 0.80, 0.90]:
        sel = rp >= thr
        prec = rt[sel].mean() if sel.sum() else float("nan")
        print(f"  {thr:>5.2f} {prec:>10.3f} {sel.mean():>9.3f} {int(sel.sum()):>7}")


    # feature importance (from 60s fits) + constancy note
    imp = pd.Series(imp_acc, index=[fnames[c] for c in cols]).sort_values(ascending=False)
    print("\n  Top-10 features by gain (60s, pooled over folds/seeds):")
    for f, v in imp.head(10).items():
        print(f"    {v:10.1f}  {f}")
    pd.DataFrame(rows, columns=["hz", "da", "lo", "hi", "n_oos"]).to_csv(
        f"ab_forward_{asset}.csv", index=False)
    imp.to_csv(f"ab_forward_importance_{asset}.csv")
    return rows, imp


if __name__ == "__main__":
    for asset, sigfile, cl, name in CANDS:
        run_asset(asset, sigfile, cl, name)
