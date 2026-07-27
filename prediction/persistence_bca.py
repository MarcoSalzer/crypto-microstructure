#!/usr/bin/env python3
"""DA + 95% BCa CI per horizon for the 2 final candidates.
Run:  python3 persistence_bca.py <results_dir>
Outputs a small table (copy back) for the persistence figure.
"""
import sys
import numpy as np
from scipy import stats

HZ = ["1s", "5s", "15s", "30s", "60s", "120s", "300s", "900s"]
CANDS = [("kmeans_pca600_k8_btc_30s_30bps", 5, "BTC 30s/30 k8"),
         ("kmeans_pca300_k8_eth_30s_30bps", 1, "ETH 30s/30 k8")]


def bca_ci(hits, B=3000, alpha=0.05, seed=42):
    hits = np.asarray(hits, float); n = len(hits); theta = hits.mean()
    if n < 5:
        return theta, np.nan, np.nan
    rng = np.random.RandomState(seed)
    boot = hits[rng.randint(0, n, (B, n))].mean(axis=1)
    p0 = np.mean(boot < theta); z0 = stats.norm.ppf(p0) if 0 < p0 < 1 else 0.0
    jk = (hits.sum() - hits) / (n - 1); jm = jk.mean(); d = jm - jk
    den = 6 * (np.sum(d ** 2) ** 1.5); a = np.sum(d ** 3) / den if den != 0 else 0.0
    def adj(z):
        zz = z0 + z; return stats.norm.cdf(z0 + zz / (1 - a * zz))
    return (theta,
            np.percentile(boot, 100 * adj(stats.norm.ppf(alpha / 2))),
            np.percentile(boot, 100 * adj(stats.norm.ppf(1 - alpha / 2))))


if __name__ == "__main__":
    DIR = sys.argv[1]
    for tag, cl, name in CANDS:
        d = np.load(f"{DIR}/cluster_members_{tag}.npz")
        lb = d["cluster_labels"]; split = int(d["split"])
        rc = d["r_cont_by_hz"]; hzl = list(d["horizons"])
        maj = {int(c): int(m) for c, m in zip(d["cluster_ids"], d["majority"])}[cl]
        lb_te = lb[split:]
        print(f"\n=== {name} Cl{cl} (majority={maj:+d}) ===")
        print(f"{'hz':>5} {'DA':>7} {'lo':>7} {'hi':>7} {'n':>5}")
        for h in HZ:
            hi = hzl.index(h); r = rc[split:, hi]
            sel = (lb_te == cl) & np.isfinite(r) & (r != 0)
            hits = (np.sign(r[sel]) == np.sign(maj)).astype(float)
            da, lo, hh = bca_ci(hits)
            print(f"{h:>5} {da:7.3f} {lo:7.3f} {hh:7.3f} {int(sel.sum()):>5}")
