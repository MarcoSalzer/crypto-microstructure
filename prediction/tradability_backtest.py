#!/usr/bin/env python3
"""4.5 Cluster Tradability — walk-forward net-PnL backtest on 30s/30bps breakouts.

Entry  : LightGBM predicts direction of the 60s move (continuation/reversal).
         Two feature sets: all use_tree features vs. top_returns subset.
         Gated by confidence threshold (0.5/0.6/0.7).
Exit A : fixed TP/SL, 2x2 grid from the train MFE/MAE distribution.
Exit B : dynamic classifier on top_mfe/top_mae features, exit at prob>=thr,
         backup SL + 300s time limit.
Costs  : taker/taker 10 bps, maker/taker 7 bps (round trip).
Scheme : 5 expanding walk-forward folds x 3 seeds, everything OOS
         (entry model AND exit classifier fitted on train only).

Run:  python3 tradability_backtest.py > tradability.log 2>&1
"""
import sys, os
import numpy as np, pandas as pd
import lightgbm as lgb
from common.data_loader import load_dataset

KEEP = "results/selection/feature_keep.csv"
THR_DEC = 30 / 1e4
EV_HZ = 30                 # breakout on 30s trailing move
HOLD = 300                 # max hold seconds
ENTRY_THRS = [0.5, 0.6, 0.7]
EXIT_THRS = [0.5, 0.6, 0.7]
SEEDS = [42, 123, 999]
N_FOLDS = 5
COSTS = {"taker_taker": 10.0, "maker_taker": 7.0}
LGB = dict(n_estimators=200, num_leaves=15, max_depth=4, learning_rate=0.05,
           min_child_samples=30, subsample=0.8, colsample_bytree=0.7, verbose=-1)
CANDS = [("btc", "BTC"), ("eth", "ETH")]


def paths_from(y1, idx, tdir, T=HOLD):
    """(n,T) direction-adjusted cumulative bps paths."""
    out = np.full((len(idx), T), np.nan)
    for i, (ix, d) in enumerate(zip(idx, tdir)):
        end = min(ix + T + 1, len(y1))
        if end <= ix + 1:
            continue
        cum = np.cumsum(y1[ix + 1:end]) * d * 1e4
        out[i, :len(cum)] = cum
    return out


def sim_tpsl(path, tp, sl):
    """First touch of +tp or -sl; else terminal. Returns exit bps (gross)."""
    v = path[~np.isnan(path)]
    if len(v) == 0:
        return 0.0
    for x in v:
        if x >= tp:
            return tp
        if x <= -sl:
            return -sl
    return float(v[-1])


def sim_dyn(path, probs, thr, sl):
    """Exit at first step prob>=thr; backup -sl; else terminal."""
    v = path[~np.isnan(path)]
    if len(v) == 0:
        return 0.0
    m = min(len(v), len(probs))
    for k in range(m):
        if v[k] <= -sl:
            return -sl
        if probs[k] >= thr:
            return float(v[k])
    return float(v[m - 1]) if m else float(v[-1])


def build_exit_steps(X, ev_idx, tdir, paths, mfe, cols, taker=10.0,
                     mfe_frac=0.60, min_hold=3, min_abs=5.0):
    """Step-level exit-classifier training data (labels use hindsight; features
    do not). Only for TRAIN trades."""
    Xs, ys = [], []
    floor = taker + min_abs
    for i, (ix, d, m) in enumerate(zip(ev_idx, tdir, mfe)):
        if m < floor:
            continue
        p = paths[i]
        for k in range(min_hold, HOLD):
            fi = ix + k
            if fi >= len(X) or k - 1 >= len(p) or np.isnan(p[k - 1]):
                break
            rn = float(p[k - 1])
            fut = p[k:min(k + 10, HOLD)]
            fut = fut[~np.isnan(fut)]
            near = (fut.max() <= rn * 1.15) if len(fut) else True
            lbl = int(rn >= mfe_frac * m and rn > floor and near)
            Xs.append(X[fi, cols]); ys.append(lbl)
    if not Xs:
        return None, None
    return np.asarray(Xs, np.float32), np.asarray(ys, np.int32)


def run(asset, name, keep):
    print("=" * 70); print(f"### {name}"); print("=" * 70)
    tree_feats = keep[(keep.type == "feature") & (keep.use_tree)]["column"].tolist()
    topret = set(keep[keep.top_returns == True]["column"])
    topmfe = set(keep[(keep.top_mfe == True) | (keep.top_mae == True)]["column"])

    X, y60, info, fnames = load_dataset(target="ret_60s", asset=asset,
                                        profile="tree", max_hours=1000)
    _, y30, _, _ = load_dataset(target="ret_30s", asset=asset, max_hours=1000,
                                target_only=True)
    _, y1, _, _ = load_dataset(target="ret_1s", asset=asset, max_hours=1000,
                               target_only=True)
    L = min(len(X), len(y60), len(y30), len(y1))
    X, y60, y30, y1 = X[:L], y60[:L], y30[:L], y1[:L]
    n2c = {n: i for i, n in enumerate(fnames)}
    tree_cols = [n2c[f] for f in tree_feats if f in n2c]
    ret_cols = [n2c[f] for f in fnames if f in topret]
    mfe_cols = [n2c[f] for f in fnames if f in topmfe]
    print(f"  features: tree={len(tree_cols)} top_returns={len(ret_cols)} "
          f"top_mfe/mae={len(mfe_cols)}")

    tm = np.full(L, np.nan); tm[EV_HZ:] = y30[:L - EV_HZ]
    ev = np.where(np.abs(tm) > THR_DEC)[0]
    ev = ev[ev < L - 1]
    bdir = np.sign(tm[ev])
    rc60 = bdir * y60[ev]
    tgt = (rc60 > 0).astype(int)         # 1 continuation, 0 reversal
    n = len(ev); print(f"  events={n}")

    folds = np.array_split(np.arange(n), N_FOLDS + 1)
    rows = []
    for fs_name, fcols in [("all_tree", tree_cols), ("top_returns", ret_cols)]:
        Xev = X[ev][:, fcols].astype(np.float32)
        for k in range(1, N_FOLDS + 1):
            tr = np.concatenate(folds[:k]); te = folds[k]
            if len(np.unique(tgt[tr])) < 2:
                continue
            # TP/SL anchors from TRAIN breakout excursions (breakout dir)
            ptr = paths_from(y1, ev[tr], bdir[tr])
            mfe_tr = np.nanmax(ptr, axis=1); mae_tr = -np.nanmin(ptr, axis=1)
            TPs = [np.nanmedian(mfe_tr), np.nanpercentile(mfe_tr, 75)]
            SLs = [np.nanmedian(mae_tr), np.nanpercentile(mae_tr, 75)]

            for seed in SEEDS:
                p = dict(LGB); p["random_state"] = seed
                em = lgb.LGBMClassifier(**p).fit(Xev[tr], tgt[tr])
                proba = em.predict_proba(Xev[te])[:, 1]   # P(continuation)

                # exit classifier (dynamic) trained on TRAIN trades
                # traded dir on train uses model prediction
                ptr_tr = em.predict(Xev[tr])
                tdir_tr = bdir[tr] * np.where(ptr_tr == 1, 1, -1)
                paths_tr = paths_from(y1, ev[tr], tdir_tr)
                mfe_train = np.nanmax(paths_tr, axis=1)
                Xex, yex = build_exit_steps(X, ev[tr], tdir_tr, paths_tr,
                                            mfe_train, mfe_cols)
                exit_clf = None
                if Xex is not None and yex.sum() > 30:
                    exit_clf = lgb.LGBMClassifier(**p).fit(Xex, yex)

                for et in ENTRY_THRS:
                    # decide trades: continuation if proba>=et, reversal if <=1-et
                    take = (proba >= et) | (proba <= 1 - et)
                    pred = np.where(proba >= 0.5, 1, -1)  # continuation/reversal
                    idx_te = te[take]
                    if len(idx_te) == 0:
                        continue
                    tdir = bdir[te][take] * pred[take]
                    pth = paths_from(y1, ev[idx_te], tdir)

                    # Exit A: TP/SL grid
                    for tp in TPs:
                        for sl in SLs:
                            gross = np.array([sim_tpsl(pth[j], tp, sl)
                                              for j in range(len(pth))])
                            for cm, c in COSTS.items():
                                net = gross - c
                                rows.append(dict(asset=name, features=fs_name,
                                    entry_thr=et, exit="tpsl",
                                    exit_param=f"tp{tp:.0f}_sl{sl:.0f}",
                                    cost=cm, fold=k, seed=seed,
                                    n=len(net), mean_net=net.mean(),
                                    win=(net > 0).mean()))

                    # Exit B: dynamic classifier (full exit-threshold grid)
                    if exit_clf is not None:
                        sl_b = SLs[1]     # backup SL = p75 MAE
                        step_probs = []
                        for j, ix in enumerate(ev[idx_te]):
                            steps = min(HOLD, len(y1) - ix - 1)
                            if steps <= 0:
                                step_probs.append(np.array([])); continue
                            Xstep = X[ix + 1: ix + 1 + steps][:, mfe_cols]
                            step_probs.append(exit_clf.predict_proba(Xstep)[:, 1])
                        for xt in EXIT_THRS:
                            gross = np.array([sim_dyn(pth[j], step_probs[j], xt, sl_b)
                                              for j in range(len(pth))])
                            for cm, c in COSTS.items():
                                net = gross - c
                                rows.append(dict(asset=name, features=fs_name,
                                    entry_thr=et, exit="dynamic",
                                    exit_param=f"thr{xt}", cost=cm, fold=k,
                                    seed=seed, n=len(net), mean_net=net.mean(),
                                    win=(net > 0).mean()))

    df = pd.DataFrame(rows)
    df.to_csv(f"tradability_{asset}_raw.csv", index=False)
    # aggregate over folds/seeds
    agg = (df.groupby(["features", "entry_thr", "exit", "exit_param", "cost"])
             .agg(mean_net=("mean_net", "mean"), win=("win", "mean"),
                  n=("n", "sum")).reset_index())
    print("\n  Net PnL per trade (bps), averaged over folds/seeds:")
    print(agg.sort_values("mean_net", ascending=False).to_string(index=False))
    agg.to_csv(f"tradability_{asset}_summary.csv", index=False)
    return agg


if __name__ == "__main__":
    keep = pd.read_csv(KEEP)
    for asset, name in CANDS:
        run(asset, name, keep)
