#!/usr/bin/env python3
"""Compute BCa CI at the window horizon for each candidate cluster.
Reads reselect_candidates.csv + the per-config npz in <results_dir>, writes
reselect_candidates_bca.csv (small: candidates + DA + 95% BCa CI).

Run:  python -m prediction.cluster_confidence_intervals <results_dir> <candidates_csv>
"""
import sys, os
import numpy as np, pandas as pd
from scipy import stats



def bca_ci(hits, B=3000, alpha=0.05, seed=42):
    hits = np.asarray(hits, float); n = len(hits); theta = hits.mean()
    if n < 5:
        return theta, np.nan, np.nan
    rng = np.random.RandomState(seed)
    boot = hits[rng.randint(0, n, (B, n))].mean(axis=1)
    p0 = np.mean(boot < theta)
    z0 = stats.norm.ppf(p0) if 0 < p0 < 1 else 0.0
    jk = (hits.sum() - hits) / (n - 1)
    jm = jk.mean(); d = jm - jk
    den = 6 * (np.sum(d ** 2) ** 1.5)
    a = np.sum(d ** 3) / den if den != 0 else 0.0
    def adj(z):
        zz = z0 + z; return stats.norm.cdf(z0 + zz / (1 - a * zz))
    lo = np.percentile(boot, 100 * adj(stats.norm.ppf(alpha / 2)))
    hi = np.percentile(boot, 100 * adj(stats.norm.ppf(1 - alpha / 2)))
    return theta, lo, hi


if __name__ == "__main__":
    DIR = sys.argv[1]
    CAND = sys.argv[2] if len(sys.argv) > 2 else os.path.join(DIR, "reselect_candidates.csv")
    c = pd.read_csv(CAND)
    out = []
    for _, r in c.iterrows():
        pca = "none" if str(r.pca) == "none" else int(r.pca)
        tag = f"kmeans_pca{pca}_k{int(r.k)}_{r.asset}_{r.hz}_{int(r.thr)}bps"
        npz = os.path.join(DIR, f"cluster_members_{tag}.npz")
        row = r.to_dict()
        row.update(dict(DA_bca=np.nan, ci_lo=np.nan, ci_hi=np.nan, bca_n=np.nan))
        if os.path.exists(npz):
            d = np.load(npz)
            lb = d["cluster_labels"]; split = int(d["split"])
            rc = d["r_cont_by_hz"]; hz_list = list(d["horizons"])
            maj = {int(cc): int(m) for cc, m in zip(d["cluster_ids"], d["majority"])}
            if r.hz in hz_list and int(r.cluster) in maj:
                hi = hz_list.index(r.hz)
                rc_te = rc[split:, hi]; lb_te = lb[split:]
                sel = (lb_te == int(r.cluster)) & np.isfinite(rc_te) & (rc_te != 0)
                hits = (np.sign(rc_te[sel]) == np.sign(maj[int(r.cluster)])).astype(float)
                da, lo, hh = bca_ci(hits)
                row.update(dict(DA_bca=round(float(da), 4),
                                ci_lo=round(float(lo), 4) if not np.isnan(lo) else np.nan,
                                ci_hi=round(float(hh), 4) if not np.isnan(hh) else np.nan,
                                bca_n=int(sel.sum())))
        out.append(row)

    df = pd.DataFrame(out)
    df["bca_sig"] = df.ci_lo > 0.5
    df.to_csv(os.path.join(DIR, "reselect_candidates_bca.csv"), index=False)
    n_sig = int(df.bca_sig.sum())
    print(f"Computed BCa for {len(df)} candidates.")
    print(f"BCa-significant (ci_lo > 0.5): {n_sig}")
    print(f"\n-> reselect_candidates_bca.csv  (upload this one small file)")
