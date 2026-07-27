#!/usr/bin/env python3
# prediction/core_feature_signal.py
# ==============================================================================
# WS5 — Direct Core-Feature Signal (no clustering)
# ==============================================================================
#
# MOTIVATION:
#   ws4c showed: the hard cluster partition is temporally UNSTABLE
#   (0/x carry-over). BUT the feature-profile test (ws4c test 4) showed that
#   73-80% of the important features keep their DIRECTION over time. The
#   information therefore sits in the features, not in the cluster assignment.
#
#   This test skips clustering entirely and asks directly:
#   does a classifier on the stable feature core carry a
#   TIME-STABLE edge — train on the past, test on the future?
#
# APPROACH (train-only, same discipline as WS4/ws3):
#   1. Identify breakouts (price moves >= threshold over the window).
#   2. Label per breakout: "profitable" = MFE >= taker_cost + margin (from the 1s path).
#   3. Feature core: only (top_short_horizon | top_long_horizon) & use_tree,
#      strictly causal (no _fwd_), at t = entry.
#   4. Expanding-window CV: classifier per fold on train ONLY, test measures.
#   5. Honest metric: test hit rate (precision of the "trade" prediction)
#      against the test base rate, averaged over folds. Plus realised
#      OOS PnL of the filtered trades (fixed TP@frac*MFE, taker).
#
# This is NOT a cluster track. It is the direct test of whether the stable core
# alone carries a tradable, time-transferable edge.
#
# USAGE:
#   python core_feature_signal.py --asset btc --hz 15s --thresholds 15 --max-hours 2000
# ==============================================================================

from __future__ import annotations
import argparse, gc, logging, sys
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MFE_LOOKAHEAD = 300
MIN_ABS_BPS   = 5.0
N_FOLDS       = 5
PROB_THRESHOLDS = [0.45, 0.50, 0.55, 0.60]   # chosen on train (best precision lift)


def _ts(): return datetime.now().strftime("%H:%M:%S")
def tprint(m=""): print(f"{_ts()}  {m}", flush=True)


def core_feature_cols(feat_names):
    """(top_short_horizon | top_long_horizon) & use_tree, causal. Column indices."""
    from common.config import RESULTS_DIR
    fk_path = None
    for cand in [Path("results/selection/feature_keep.csv"),
                 RESULTS_DIR.parent / "feature_keep.csv",
                 RESULTS_DIR.parent.parent / "analysis/feature_reduction/feature_keep.csv",
                 Path("feature_keep.csv")]:
        if cand.exists(): fk_path = cand; break
    if fk_path is None:
        raise FileNotFoundError(
            "feature_keep.csv not found — expected in results/selection/")
    fk = pd.read_csv(fk_path)
    for c in ["top_short_horizon", "top_long_horizon", "use_tree"]:
        fk[c] = fk[c].astype(str).str.lower().isin(["true", "1"])
    sel = fk[(fk["top_short_horizon"] | fk["top_long_horizon"]) & fk["use_tree"]]
    sel = sel[~sel["column"].str.contains("_fwd_|ret_fwd|mfe_fwd|mae_fwd", regex=True, na=False)]
    name_to_idx = {n: i for i, n in enumerate(feat_names)}
    cols, names = [], []
    for col in sel["column"]:
        if col in name_to_idx:
            cols.append(name_to_idx[col]); names.append(col)
    return np.array(cols, dtype=int), names


def find_breakouts(y_1s, threshold_bps, window_s):
    """
    Breakout at index i when |cumulative return over window_s from i| >= threshold.
    Returns event_indices + direction (sign). Non-overlapping (cooldown=window).
    """
    n = len(y_1s)
    cum = np.array([np.sum(y_1s[i:i+window_s]) for i in range(n - window_s)]) * 10_000
    events, dirs, last = [], [], -10**9
    for i in range(len(cum)):
        if abs(cum[i]) >= threshold_bps and i - last >= window_s:
            events.append(i); dirs.append(1 if cum[i] > 0 else -1); last = i
    return np.array(events, int), np.array(dirs, int)


def label_and_paths(y_1s, events, dirs, taker_cost, lookahead=MFE_LOOKAHEAD):
    """MFE per breakout (direction-adjusted) + profitable label (MFE>=cost+margin)."""
    n = len(y_1s); mfe = np.zeros(len(events)); paths = []
    for j, (idx, d) in enumerate(zip(events, dirs)):
        end = min(idx + lookahead + 1, n)
        if end <= idx + 1:
            paths.append(np.array([0.0])); continue
        cum = np.cumsum(y_1s[idx+1:end]) * d * 10_000
        paths.append(cum.astype(np.float32))
        mfe[j] = float(np.max(cum)) if len(cum) else 0.0
    # The label is NOT formed globally here (that was circular + too loose a
    # base rate). It arises per-fold, relative, as the top quartile of TRAIN MFE.
    return mfe, paths


def _fixed_tp(path, frac, mfe, min_hold=3):
    n = len(path)
    if n == 0: return 0.0
    thr = frac * mfe
    for k in range(min_hold, n):
        if path[k-1] >= thr: return float(path[k-1])
    return float(path[-1])


def run_ws5(assets=("btc",), horizons=("15s",), thresholds=(15,),
            window_map=None, max_hours=None, n_jobs=6):
    import lightgbm as lgb
    from sklearn.impute import SimpleImputer
    from common.data_loader import load_dataset
    from common.config import RESULTS_DIR, SPREAD_BPS, MAKER_COST_BPS

    out_dir = RESULTS_DIR / "core_signal"; out_dir.mkdir(parents=True, exist_ok=True)
    window_map = window_map or {"5s": 5, "15s": 15}

    for asset in assets:
        taker = SPREAD_BPS.get(asset, {}).get("fut", 9.0)
        maker = MAKER_COST_BPS.get(asset, {}).get("fut", 4.0)
        for hz in horizons:
            win = window_map.get(hz, 15)
            tprint(f"━━ WS5 {asset.upper()}/{hz} (window={win}s) ━━")
            try:
                X, y, info, feat_names = load_dataset(
                    target="1s", asset=asset, profile="tree", max_hours=max_hours)
            except Exception as e:
                logger.error("Load fail: %s", e); continue
            y_1s = y.astype(np.float32)

            core_cols, core_names = core_feature_cols(feat_names)
            tprint(f"  Feature core: {len(core_cols)} features (top_short|long & use_tree, causal)")
            X_core = X[:, core_cols]

            for thr in thresholds:
                events, dirs = find_breakouts(y_1s, thr, win)
                if len(events) < 500:
                    tprint(f"  {thr}bps: only {len(events)} breakouts — skip"); continue
                mfe, paths = label_and_paths(y_1s, events, dirs, taker)
                tprint(f"  {thr}bps: {len(events)} Breakouts, "
                       f"MFE median={np.median(mfe):.1f} P75={np.percentile(mfe,75):.1f}")

                order = np.argsort(events)
                Xe = X_core[events][order]
                mfe_o = mfe[order]; paths_o = [paths[i] for i in order]
                block = len(events) // (N_FOLDS + 1)
                if block < 50:
                    tprint("    too few events per fold — skip"); continue

                prec_oos, lift_oos, pnl_oos, pnl_all_oos, chosen = [], [], [], [], []
                for f in range(N_FOLDS):
                    tr_end = (f+1)*block; te_s, te_e = tr_end, min(tr_end+block, len(events))
                    if te_e - te_s < 20: continue
                    # ── Top-quartile label: threshold from TRAIN MFE ONLY ──────────
                    q75 = np.percentile(mfe_o[:tr_end], 75)
                    ytr = (mfe_o[:tr_end]    >= q75).astype(int)
                    yte = (mfe_o[te_s:te_e]  >= q75).astype(int)   # same threshold!
                    if ytr.sum() < 20 or yte.sum() < 5: continue
                    imp = SimpleImputer(strategy="median")
                    Xtr = imp.fit_transform(Xe[:tr_end]); Xte = imp.transform(Xe[te_s:te_e])
                    nval = max(int(len(Xtr)*0.15), 50)
                    clf = lgb.LGBMClassifier(n_estimators=400, num_leaves=31, max_depth=6,
                        learning_rate=0.03, class_weight="balanced", n_jobs=n_jobs,
                        verbose=-1, random_state=42)
                    clf.fit(Xtr[:-nval], ytr[:-nval], eval_set=[(Xtr[-nval:], ytr[-nval:])],
                            callbacks=[lgb.early_stopping(40), lgb.log_evaluation(0)])
                    proba_tr = clf.predict_proba(Xtr)[:,1]
                    proba_te = clf.predict_proba(Xte)[:,1]
                    # best prob threshold on TRAIN (highest precision at >=5% trade rate)
                    best_pt, best_prec = 0.5, -1
                    for pt in PROB_THRESHOLDS:
                        sel = proba_tr >= pt
                        if sel.mean() < 0.05: continue
                        pr = ytr[sel].mean() if sel.sum() else 0
                        if pr > best_prec: best_prec, best_pt = pr, pt
                    chosen.append(best_pt)
                    sel_te = proba_te >= best_pt
                    if sel_te.sum() < 5: continue
                    te_prec = yte[sel_te].mean()
                    te_base = yte.mean()           # ~0.25 by construction
                    prec_oos.append(te_prec)
                    lift_oos.append(te_prec / te_base if te_base > 0 else 0)
                    # PnL: filtered vs. ALL test trades (fixed TP@60%MFE, taker)
                    idx_te   = np.arange(te_s, te_e)
                    pnl_sel  = np.array([_fixed_tp(paths_o[i], 0.60, mfe_o[i])
                                         for i in idx_te[sel_te]]) - taker
                    pnl_allf = np.array([_fixed_tp(paths_o[i], 0.60, mfe_o[i])
                                         for i in idx_te]) - taker
                    pnl_oos.append(pnl_sel); pnl_all_oos.append(pnl_allf)

                if not prec_oos:
                    tprint("    no valid folds"); continue
                selpnl = np.concatenate(pnl_oos)
                allpnl = np.concatenate(pnl_all_oos)
                mp = float(np.mean(prec_oos)); ml = float(np.mean(lift_oos))
                sh_sel = float(selpnl.mean()/selpnl.std()) if selpnl.std()>0 else 0
                # honest economic test: does the FILTERED selection beat the
                # unfiltered average of all breakouts?
                pnl_edge = selpnl.mean() - allpnl.mean()
                tprint(f"    OOS precision={mp:.3f} vs base=0.25 → lift={ml:.2f}x "
                       f"(top-quartile discriminatory power)")
                tprint(f"    PnL filtered={selpnl.mean():+.2f} bps vs all={allpnl.mean():+.2f} bps "
                       f"→ selection edge={pnl_edge:+.2f} bps (n_filt={len(selpnl)})")
                tprint(f"    filtered: sharpe={sh_sel:.3f} WR={(selpnl>0).mean():.1%}")
                verdict = ("EDGE" if (ml > 1.15 and pnl_edge > 1.0)
                           else "NO time-stable selection edge")
                tprint(f"    → {verdict}  (prob-thr per fold: {chosen})")
                pd.DataFrame([dict(asset=asset, hz=hz, thr=thr, n_breakouts=len(events),
                    oos_precision=round(mp,4), oos_lift=round(ml,3),
                    pnl_filtered=round(float(selpnl.mean()),3),
                    pnl_all=round(float(allpnl.mean()),3),
                    selection_edge_bps=round(float(pnl_edge),3),
                    sharpe_filtered=round(sh_sel,4),
                    n_filtered=len(selpnl), verdict=verdict)]).to_csv(
                    out_dir / f"core_signal_{asset}_{hz}_{thr}bps.csv", index=False)
            del X, X_core, y_1s; gc.collect()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--asset", nargs="+", default=["btc"])
    p.add_argument("--hz", nargs="+", default=["15s"])
    p.add_argument("--thresholds", nargs="+", type=int, default=[15])
    p.add_argument("--max-hours", type=int, default=None)
    p.add_argument("--n-jobs", type=int, default=6)
    p.add_argument("--log-level", default="INFO")
    a = p.parse_args()
    logging.basicConfig(level=getattr(logging, a.log_level),
        format="%(asctime)s  %(levelname)s  %(message)s", datefmt="%H:%M:%S")
    run_ws5(assets=tuple(a.asset), horizons=tuple(a.hz),
            thresholds=tuple(a.thresholds), max_hours=a.max_hours, n_jobs=a.n_jobs)


if __name__ == "__main__":
    main()