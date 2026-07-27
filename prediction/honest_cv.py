"""
honest_cv.py — train-only per-fold TP/SL selection for WS3.
===================================================================

PROBLEM (Original tp_sl_optimization.py):
  The TP/SL grid (1,377 combos) is simulated over ALL trades, then via
  df_grid.nlargest(...) the best pair is chosen — on the same dataset on
  which the PnL is measured. Pure in-sample optimisation over 1,377 degrees
  → the reported "best PnL" is massively upward-biased (the same
  same pattern as the fold-internal contamination in WS4, only worse due to the larger grid).

SOLUTION:
  Expanding-Window-CV. Per fold:
    1. TRAIN: simulate the grid, pick (TP,SL) with the best SHARPE (taker cost).
    2. TEST:  simulate EXACTLY this one (TP,SL) pair, measure the test PnL.
  Averaged over the folds = honest OOS PnL of the TP/SL SELECTION RULE.

  Reports both levels:
    - GLOBAL:      one (TP,SL) over all trades of the combo
    - PER-CLUSTER: own (TP,SL) per cluster (more freedom, more overfit risk)

  Ranking cost: taker (conservative). Maker is additionally reported as diagnostics
  on the final OOS set, but NOT used for ranking.

  Cluster IDs are fold-local (from WS4 honest CV) — for PER-CLUSTER TP/SL this means
  this: per fold each cluster is optimised separately; the aggregation is done
  over the OOS PnL series, not over the cluster IDs. This is consistent because
  the trade direction is already baked into the paths (direction-adjusted).
"""

from __future__ import annotations
import numpy as np


def _sharpe(net_pnl):
    if len(net_pnl) < 2:
        return 0.0
    sd = net_pnl.std()
    return float(net_pnl.mean() / sd) if sd > 0 else 0.0


def _select_best_tp_sl(paths_tr, tp_values, sl_values, taker_cost,
                       simulate_tp_sl, min_trades=30):
    """
    Picks on TRAIN the (TP,SL) with the highest Sharpe (after taker cost).
    Returns (best_tp, best_sl, best_sharpe_train), or None if too little data.
    """
    if len(paths_tr) < min_trades:
        return None
    best = None
    for tp in tp_values:
        for sl in sl_values:
            gpnl, _, _ = simulate_tp_sl(paths_tr, tp, sl)
            net = gpnl - taker_cost
            sh = _sharpe(net)
            if best is None or sh > best[2]:
                best = (tp, sl, sh)
    return best


def _oos_metrics(net_pnl):
    """Metrics on the OOS PnL series (concatenated across all test folds)."""
    n = len(net_pnl)
    if n == 0:
        return dict(n=0, mean_net_pnl=0.0, total_net_pnl=0.0,
                    win_rate=0.0, sharpe=0.0)
    return dict(
        n=int(n),
        mean_net_pnl=round(float(net_pnl.mean()), 3),
        total_net_pnl=round(float(net_pnl.sum()), 1),
        win_rate=round(float((net_pnl > 0).mean()), 4),
        sharpe=round(_sharpe(net_pnl), 4),
    )


def honest_strategy_cv(
    paths,              # (n_trades, max_lookahead) direction-adjusted bps paths
    clusters,           # (n_trades,) Cluster-ID per trade
    event_indices,      # (n_trades,) row index per trade (temporal sorting)
    param_grid,         # list[dict]: each dict = one parameter set for sim_fn
    sim_fn,             # callable(paths, **params) -> gross_pnl  OR (gross_pnl, ...)
    taker_cost, maker_cost,
    n_folds=5,
    min_trades_train=30,
    label="strategy",
):
    """
    Strategy-AGNOSTIC per-fold CV. Works for TP/SL, time exit,
    trailing, combined — the caller passes only sim_fn + param_grid.

    sim_fn may return gross_pnl alone OR a tuple (gross_pnl, ...);
    we always take the first element as gross_pnl.

    Selection on TRAIN by SHARPE (taker cost). Test measures exactly the
    chosen parameter set. Reports GLOBAL + PER-CLUSTER, taker+maker.
    """
    def _gross(p, params):
        out = sim_fn(p, **params)
        return out[0] if isinstance(out, tuple) else out

    def _best_on_train(p_tr):
        if len(p_tr) < min_trades_train:
            return None
        best = None
        for params in param_grid:
            net = _gross(p_tr, params) - taker_cost
            sh = _sharpe(net)
            if best is None or sh > best[1]:
                best = (params, sh)
        return best[0] if best else None

    n = len(paths)
    order = np.argsort(event_indices)
    paths_s = paths[order]
    clu_s   = clusters[order]
    block = n // (n_folds + 1)
    if block < max(min_trades_train, 20):
        return None

    def _run(mask_fn):
        oos_t, oos_m, chosen = [], [], []
        for f in range(n_folds):
            tr_end = (f + 1) * block
            te_s, te_e = tr_end, min(tr_end + block, n)
            if te_e - te_s < 10:
                continue
            tr_idx = mask_fn(slice(0, tr_end))
            te_idx = mask_fn(slice(te_s, te_e))
            if tr_idx.sum() < min_trades_train or te_idx.sum() < 5:
                continue
            params = _best_on_train(paths_s[:tr_end][tr_idx])
            if params is None:
                continue
            gpnl = _gross(paths_s[te_s:te_e][te_idx], params)
            oos_t.append(gpnl - taker_cost)
            oos_m.append(gpnl - maker_cost)
            chosen.append(params)
        if not oos_t:
            return None
        return dict(
            chosen_per_fold=chosen,
            oos_taker=_oos_metrics(np.concatenate(oos_t)),
            oos_maker=_oos_metrics(np.concatenate(oos_m)),
        )

    # GLOBAL: all trades
    global_res = _run(lambda sl: np.ones(len(paths_s[sl]), dtype=bool))
    if global_res is None:
        return None

    # PER-CLUSTER
    per_cluster = {}
    for cl in sorted(set(clu_s.tolist())):
        res = _run(lambda sl, _cl=cl: clu_s[sl] == _cl)
        if res is not None:
            per_cluster[int(cl)] = res

    return dict(strategy=label, global_=global_res, per_cluster=per_cluster)


def honest_tp_sl_cv(
    paths,              # (n_trades, max_lookahead) direction-adjusted bps paths
    clusters,           # (n_trades,) Cluster-ID per trade (fold-local from WS4)
    event_indices,      # (n_trades,) row index per trade (for time sorting)
    tp_values, sl_values,
    taker_cost, maker_cost,
    simulate_tp_sl,     # function from ws3 (reused)
    n_folds=5,
    min_trades_train=30,
):
    """
    Returns dict:
      {
        "global":  {chosen_per_fold:[(tp,sl),...], oos_taker:{...}, oos_maker:{...}},
        "per_cluster": {cluster_id: {chosen_per_fold, oos_taker, oos_maker}, ...},
      }
    """
    n = len(paths)
    # Temporal sorting (index = time, files time-sorted)
    order = np.argsort(event_indices)
    paths_s = paths[order]
    clu_s   = clusters[order]
    block = n // (n_folds + 1)
    if block < max(min_trades_train, 20):
        return None

    # ── GLOBAL: one (TP,SL) per fold over all trades ────────────────────────
    g_oos_taker, g_oos_maker, g_chosen = [], [], []
    for f in range(n_folds):
        tr_end = (f + 1) * block
        te_s, te_e = tr_end, min(tr_end + block, n)
        if te_e - te_s < 10:
            continue
        sel = _select_best_tp_sl(paths_s[:tr_end], tp_values, sl_values,
                                 taker_cost, simulate_tp_sl, min_trades_train)
        if sel is None:
            continue
        tp, sl, _ = sel
        gpnl, _, _ = simulate_tp_sl(paths_s[te_s:te_e], tp, sl)
        g_oos_taker.append(gpnl - taker_cost)
        g_oos_maker.append(gpnl - maker_cost)
        g_chosen.append((tp, sl))

    if not g_oos_taker:
        return None

    global_res = dict(
        chosen_per_fold=g_chosen,
        oos_taker=_oos_metrics(np.concatenate(g_oos_taker)),
        oos_maker=_oos_metrics(np.concatenate(g_oos_maker)),
    )

    # ── PER-CLUSTER: its own (TP,SL) per cluster and fold ────────────────────
    per_cluster = {}
    for cl in sorted(set(clu_s.tolist())):
        c_oos_taker, c_oos_maker, c_chosen = [], [], []
        for f in range(n_folds):
            tr_end = (f + 1) * block
            te_s, te_e = tr_end, min(tr_end + block, n)
            if te_e - te_s < 10:
                continue
            tr_mask = clu_s[:tr_end] == cl
            te_mask = clu_s[te_s:te_e] == cl
            if tr_mask.sum() < min_trades_train or te_mask.sum() < 5:
                continue
            sel = _select_best_tp_sl(paths_s[:tr_end][tr_mask], tp_values,
                                     sl_values, taker_cost, simulate_tp_sl,
                                     min_trades_train)
            if sel is None:
                continue
            tp, sl, _ = sel
            gpnl, _, _ = simulate_tp_sl(paths_s[te_s:te_e][te_mask], tp, sl)
            c_oos_taker.append(gpnl - taker_cost)
            c_oos_maker.append(gpnl - maker_cost)
            c_chosen.append((tp, sl))
        if c_oos_taker:
            per_cluster[int(cl)] = dict(
                chosen_per_fold=c_chosen,
                oos_taker=_oos_metrics(np.concatenate(c_oos_taker)),
                oos_maker=_oos_metrics(np.concatenate(c_oos_maker)),
            )

    return dict(global_=global_res, per_cluster=per_cluster)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Self-test: rebuild simulate_tp_sl (simplified), inject signal.
    def _sim(paths, tp, sl, timeout_s=None):
        n, mt = paths.shape
        g = np.zeros(n); o = np.zeros(n, int); e = np.zeros(n)
        for i in range(n):
            p = paths[i]; vl = int(np.sum(~np.isnan(p)))
            if vl == 0: continue
            ps = p[:vl]
            tph = np.where(ps >= tp)[0]; slh = np.where(ps <= -sl)[0]
            tpt = tph[0] if len(tph) else vl + 1
            slt = slh[0] if len(slh) else vl + 1
            if tpt <= slt and tpt < vl: g[i] = tp
            elif slt < tpt and slt < vl: g[i] = -sl
            else: g[i] = ps[-1]
        return g, o, e

    rng = np.random.RandomState(0)
    n, T = 4000, 300
    # Paths: random walk in bps, slight positive drift (a real, weak signal)
    steps = rng.randn(n, T) * 2.0 + 0.02
    paths = np.cumsum(steps, axis=1)
    clusters = rng.randint(0, 4, n)
    ev = np.sort(rng.choice(10_000_000, n, replace=False))
    out = honest_tp_sl_cv(paths, clusters, ev,
                          tp_values=[10,20,30,40,60], sl_values=[10,20,30],
                          taker_cost=10.0, maker_cost=4.0,
                          simulate_tp_sl=_sim, n_folds=5)
    if out is None:
        print("Self-test: empty (check)")
    else:
        g = out["global_"]
        print(f"GLOBAL  OOS taker: mean={g['oos_taker']['mean_net_pnl']:+.2f} "
              f"sharpe={g['oos_taker']['sharpe']:.3f} n={g['oos_taker']['n']}")
        print(f"        chosen per fold: {g['chosen_per_fold']}")
        print(f"PER-CLUSTER: {len(out['per_cluster'])} clusters reported")
        print("Self-test OK.")