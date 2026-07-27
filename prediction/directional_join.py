# directional_join.py  (v4 — joins on event_index, the robust common key)
# ==============================================================================
# A<->B: does the breakout-state structure (B) carry directional information
# beyond the continuous baseline LGBM (A) on the SAME breakout events, at the
# SAME horizon, OUT OF SAMPLE?
#
# Join key = event_index (absolute row position in the loaded dataset). Both
# sides carry it: the cluster run dumps it in cluster_trades; the LGBM baseline
# dumps it per OOS row (lgbm_pipeline._save_oos_predictions, patched). The loader
# timestamp is unreliable, so it is NOT used for the join.
#
# A = baseline LGBM OOS predictions   columns: event_index, y_pred, y_true, fold
# B = cluster breakout events         columns: event_index, cluster/cluster_coherent,
#                                              direction  (majority derived on train)
#
# USAGE:
#   python directional_join.py \
#       --oos    .../lightgbm_btc_ret_5s_oos_predictions.parquet \
#       --events .../cluster_trades_kmeans_pca600_k6_btc_5s_15bps_lb1.csv \
#       --horizon 5s --asset btc --out ab_btc_5s_15bps.csv
# ==============================================================================
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd


def _derive_majority(ev, cluster_col, train_frac=0.60):
    """Train-derived per-cluster majority sign + split, from the realised
    'direction'. Events ordered by event_index (= time order); majority fixed on
    the first `train_frac`, evaluated on the rest -> honest out-of-sample."""
    if "direction" not in ev.columns:
        raise SystemExit("events file has neither 'majority' nor 'direction'. "
                         f"Have: {list(ev.columns)}")
    ev = ev.sort_values("event_index").reset_index(drop=True)
    n = len(ev); cut = int(n * train_frac)
    ev["split"] = np.where(np.arange(n) < cut, "train", "test")
    tr = ev[ev["split"] == "train"]
    maj = {}
    for c, g in tr.groupby(cluster_col):
        sgn = np.sign(g["direction"].mean())
        maj[c] = int(sgn) if sgn != 0 else 1
    ev["majority"] = ev[cluster_col].map(maj)
    ev = ev[ev["majority"].notna()].copy()
    ev["majority"] = ev["majority"].astype(int)
    return ev, maj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oos", required=True, help="baseline *_oos_predictions.parquet")
    ap.add_argument("--events", required=True, help="cluster_trades_*.csv")
    ap.add_argument("--horizon", default="", help="label only")
    ap.add_argument("--asset", default="", help="label only")
    ap.add_argument("--cluster-col", default=None,
                    help="cluster id column (default: cluster_coherent if present)")
    ap.add_argument("--all-events", action="store_true",
                    help="evaluate on ALL events; default keeps the test slice")
    ap.add_argument("--test-only", action="store_true")
    ap.add_argument("--out", default="ab_compare.csv")
    a = ap.parse_args()

    oos = pd.read_parquet(a.oos)
    ev = pd.read_csv(a.events)

    if "event_index" not in oos.columns:
        raise SystemExit("OOS parquet has no 'event_index' — re-run the LGBM "
                         "baseline with the patched lgbm_pipeline.py.")
    if "event_index" not in ev.columns:
        raise SystemExit(f"events file has no 'event_index'. Have: {list(ev.columns)}")

    if a.cluster_col:
        ccol = a.cluster_col
    elif "cluster_coherent" in ev.columns:
        ccol = "cluster_coherent"
    elif "cluster" in ev.columns:
        ccol = "cluster"
    else:
        raise SystemExit(f"no cluster column. Have: {list(ev.columns)}")

    derived = False
    if "majority" not in ev.columns:
        ev, maj = _derive_majority(ev, ccol)
        derived = True
        print(f"[majority derived per '{ccol}' on train 60%] -> {maj}")

    if "split" in ev.columns and ((derived and not a.all_events) or a.test_only):
        ev = ev[ev["split"] == "test"].copy()

    ev["cluster"] = ev[ccol]

    oos = oos.drop_duplicates("event_index", keep="first")
    ev = ev.drop_duplicates("event_index", keep="first")
    n_ev = len(ev)
    merged = ev.merge(oos[["event_index", "y_pred", "y_true"]],
                      on="event_index", how="inner")
    n_join = len(merged)
    cov = n_join / max(n_ev, 1)
    print(f"[{a.asset} {a.horizon}] events={n_ev}  joined_OOS={n_join}  "
          f"coverage={cov:.1%}  (rest fall in the never-tested first block)")
    if n_join == 0:
        raise SystemExit("No overlap on event_index — confirm both runs used the "
                         "same 1000h window so absolute row positions align.")

    realized = np.sign(merged["y_true"].values)
    base_dir = np.sign(merged["y_pred"].values)
    clus_dir = np.sign(merged["majority"].values)
    m = realized != 0

    def da(pred):
        return float((pred[m] == realized[m]).mean()) if m.sum() else float("nan")

    rows = [dict(scope="ALL", n=int(m.sum()),
                 baseline_da=round(da(base_dir), 4),
                 cluster_da=round(da(clus_dir), 4),
                 delta_cluster_minus_baseline=round(da(clus_dir) - da(base_dir), 4))]
    for c in sorted(merged["cluster"].unique()):
        cm = (merged["cluster"].values == c) & m
        if cm.sum() < 30:
            continue
        b = float((base_dir[cm] == realized[cm]).mean())
        k = float((clus_dir[cm] == realized[cm]).mean())
        rows.append(dict(scope=f"cluster_{int(c)}", n=int(cm.sum()),
                         baseline_da=round(b, 4), cluster_da=round(k, 4),
                         delta_cluster_minus_baseline=round(k - b, 4)))

    out = pd.DataFrame(rows)
    print("\n=== A<->B directional comparison (OOS, directed, event_index join) ===")
    print(out.to_string(index=False))
    print("\nReading: cluster_da > baseline_da -> the state adds directional info "
          "beyond the continuous model on the same events.")
    out.to_csv(a.out, index=False)
    print(f"\nSaved -> {a.out}")


if __name__ == "__main__":
    main()