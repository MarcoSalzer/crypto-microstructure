#!/usr/bin/env python3
# prediction/dynamic_exit.py
# ==============================================================================
# WS3d — Dynamic Exit Classifier (train-only, per-fold)
# ==============================================================================
#
# IDEA (from the user):
#   From trade start (cluster entry), predict every second: exit NOW or
#   keep holding? Features = top_mfe/top_mae-flagged microstructure columns
#   (use_cluster-capable, causal <= t, NO forward feature → causal).
#   Backup/comparison: the fixed TP/SL from ws3 (honest per-fold).
#
# WHY A SEPARATE SCRIPT:
#   ws3 provides the FIXED TP/SL baseline (honest OOS). ws3d asks: does a
#   learned dynamic exit beat this fixed rule out of sample? Only if yes,
#   the extra effort of an ML exit in the live system is worthwhile.
#
# TRAIN/TEST DISCIPLINE (identical to WS4 honest CV / ws3 honest CV):
#   - Expanding-window time CV over trades (time-sorted).
#   - The exit classifier is fitted PER FOLD on train only, applied to test.
#   - The label (good exit) comes from the forward path — that is the
#     TRAINING TARGET, which is allowed. The FEATURES are strictly <= entry+k (microstructure
#     during the trade), never forward.
#   - The fixed TP/SL baseline is measured on the same test fold → fair
#     comparison dynamic vs. fixed on identical data.
#
# IMPORTANT HONESTY:
#   WS4 phase D showed: a dynamic exit often loses to TP@x%MFE. ws3d is
#   the clean test of whether that holds robustly. A negative result (fixed beats
#   dynamic) is a valid thesis result.
#
# USAGE:
#   python dynamic_exit.py --asset btc --hz 15s --thresholds 15
#   (reads cluster_trades_*.csv from results/cluster_mfe/, like ws3)
# ==============================================================================

from __future__ import annotations
import argparse, gc, logging, sys, time, warnings
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

# sklearn warns on every predict_proba when fit used column names (DataFrame)
# and predict uses a nameless numpy array. Harmless here (pure numpy
# pipeline), but it spams the log with thousands of lines. Suppressed on purpose —
# NOT all warnings across the board.
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning)

logger = logging.getLogger(__name__)

# Defaults (modeled on ws3/ws4)
MFE_LOOKAHEAD = 300
MIN_HOLD_S    = 3
MFE_FRAC      = 0.60
MIN_ABS_BPS   = 5.0
N_FOLDS       = 5
EXIT_PROB_THRESHOLDS = [0.30, 0.40, 0.50]   # tested, best chosen on train


def _ts():
    return datetime.now().strftime("%H:%M:%S")

def tprint(msg=""):
    print(f"{_ts()}  {msg}", flush=True)


# ──────────────────────────────────────────────────────────────────────────────
def extract_price_paths(y_1s, event_indices, directions, max_lookahead=300):
    """Direction-adjusted cumulative bps paths from entry (like ws3)."""
    n = len(event_indices)
    paths = np.full((n, max_lookahead), np.nan, dtype=np.float32)
    for i, (idx, d) in enumerate(zip(event_indices, directions)):
        end = min(idx + max_lookahead + 1, len(y_1s))
        if end <= idx + 1:
            continue
        cum = np.cumsum(y_1s[idx + 1 : end]) * d * 10_000
        paths[i, : len(cum)] = cum
    return paths


def top_mfe_mae_columns(feat_names, asset):
    """
    Reads the central KEEP_LIST (config) and returns the column indices of the
    top_mfe/top_mae features — WITHOUT the use_cluster filter.

    BUGFIX (2026-06): earlier it additionally filtered '& use_cluster', which
    dropped 166 of 683 excursion features. For the TP/SL classification
    top_mfe|top_mae is the correct selection; use_cluster is irrelevant here.
    The source is now config.KEEP_LIST (= the same CSV as data_loader), not
    no longer a separately located feature_keep.csv (avoided silent divergence).

    Causality: top_mfe/top_mae features are <= t (microstructure during the
    trades), no forward — as before.
    """
    from common.config import KEEP_LIST
    fk_path = KEEP_LIST
    if not Path(fk_path).exists():
        # Fallback only if KEEP_LIST is relative and the CWD differs
        for cand in [Path("results/selection/feature_keep.csv"),
                     Path("feature_keep.csv")]:
            if cand.exists():
                fk_path = cand; break
        else:
            raise FileNotFoundError(f"KEEP_LIST not found: {KEEP_LIST}")

    fk = pd.read_csv(fk_path)
    for c in ["top_mfe", "top_mae"]:
        fk[c] = fk[c].astype(str).str.lower().isin(["true", "1"])
    sel = fk[fk["top_mfe"] | fk["top_mae"]]
    name_to_idx = {n: i for i, n in enumerate(feat_names)}
    cols, names = [], []
    for col in sel["column"]:
        if col in name_to_idx:
            cols.append(name_to_idx[col]); names.append(col)
    return np.array(cols, dtype=int), names


# ──────────────────────────────────────────────────────────────────────────────
def build_exit_steps(X_feat, ev_idx, paths, mfe_vals, taker_cost,
                     mfe_frac=MFE_FRAC, min_hold_s=MIN_HOLD_S,
                     min_abs_bps=MIN_ABS_BPS, max_lookahead=MFE_LOOKAHEAD):
    """
    Builds (X_steps, y_steps, trade_of_step, k_of_step) for the exit classification.
    X_feat: feature matrix already reduced to top_mfe/mae columns (n_rows, n_sel).
    Label = good exit (captured enough MFE, profitable, near local peak).
    Also returns the step→trade/k mapping (for per-fold separation
    WITHOUT cutting a trade across fold boundaries).
    """
    Xs, ys, t_of, k_of = [], [], [], []
    n_total = len(X_feat)
    min_ret = taker_cost + min_abs_bps
    for i, (idx, mfe) in enumerate(zip(ev_idx, mfe_vals)):
        if mfe < min_ret:
            continue
        path = paths[i]
        n_path = int(np.sum(~np.isnan(path)))
        for k in range(min_hold_s, min(max_lookahead, n_path)):
            fidx = idx + k
            if fidx >= n_total:
                break
            ret_now = float(path[k - 1])
            pct = ret_now >= mfe_frac * mfe
            prof = ret_now > min_ret
            fut = path[k:min(k + 10, n_path)]
            fmax = float(np.max(fut)) if len(fut) else ret_now
            near_peak = fmax <= ret_now * 1.15
            Xs.append(X_feat[fidx]); ys.append(int(pct and prof and near_peak))
            t_of.append(i); k_of.append(k)
    if not Xs:
        return (np.empty((0, X_feat.shape[1]), np.float32), np.empty(0, int),
                np.empty(0, int), np.empty(0, int))
    return (np.array(Xs, np.float32), np.array(ys, np.int32),
            np.array(t_of, int), np.array(k_of, int))


def build_sl_steps(X_feat, ev_idx, paths, sl_floor, min_hold_s=MIN_HOLD_S,
                   max_lookahead=MFE_LOOKAHEAD):
    """
    SL label builder (symmetric to build_exit_steps, for the SL classifier).

    Label_sl[k] = 1 if the path from step k to trade end NEVER again rises above
    -sl_floor rises (no recovery above the SL threshold -> cap now).
    Label = 0 if the path later recovers above -sl_floor.

    forward = training target (allowed, like the TP near_peak label). Features
    are X_feat at the current step (causal <= entry+k, no forward).

    ALL steps are labelled (no mfe pre-filter as in build_exit_steps),
    because an SL is relevant precisely for the loser paths that build_exit_steps
    skips via 'if mfe < min_ret: continue'.
    """
    Xs, ys, t_of, k_of = [], [], [], []
    n_total = len(X_feat)
    for i, idx in enumerate(ev_idx):
        path = paths[i]
        n_path = int(np.sum(~np.isnan(path)))
        for k in range(min_hold_s, min(max_lookahead, n_path)):
            fidx = idx + k
            if fidx >= n_total:
                break
            fut = path[k:n_path]                 # everything FROM now on (incl. terminal)
            if len(fut) == 0:
                continue
            recovers = bool(np.any(fut > -sl_floor))
            Xs.append(X_feat[fidx]); ys.append(int(not recovers))
            t_of.append(i); k_of.append(k)
    if not Xs:
        return (np.empty((0, X_feat.shape[1]), np.float32), np.empty(0, int),
                np.empty(0, int), np.empty(0, int))
    return (np.array(Xs, np.float32), np.array(ys, np.int32),
            np.array(t_of, int), np.array(k_of, int))


def simulate_dynamic_exit(paths, trade_ids_sorted, clf, X_feat, ev_idx,
                          prob_thr, taker_cost, sl_bps=22.0, min_hold_s=MIN_HOLD_S,
                          max_lookahead=MFE_LOOKAHEAD, return_diag=False):
    """
    Dynamic exit (learned profit-taking) WITH a fixed stop-loss floor.
    Per step: check SL first (ret <= -sl -> out), otherwise at P(exit)>=thr and
    ret>taker exit; otherwise terminal. Returns gross_pnl per trade.

    return_diag=True: additionally a diag dict (why each trade closes) — for
    the diagnosis of the -10bps/WR-13.4% finding.
    """
    n_total = len(X_feat)
    out = np.zeros(len(trade_ids_sorted), dtype=np.float32)
    diag = dict(sl_exit=0, tp_exit=0, terminal=0, p_tp_fire=0, steps_eval=0)
    for j, i in enumerate(trade_ids_sorted):
        path = paths[i]
        n_path = int(np.sum(~np.isnan(path)))
        exit_ret = float(path[n_path - 1]) if n_path > 0 else 0.0
        closed = "terminal"
        for k in range(min_hold_s, min(max_lookahead, n_path)):
            fidx = ev_idx[i] + k
            if fidx >= n_total:
                break
            ret_now = float(path[k - 1])
            diag["steps_eval"] += 1
            if ret_now <= -sl_bps:            # fixed stop-loss (cut losers)
                exit_ret = -sl_bps; closed = "sl_exit"; break
            if ret_now <= taker_cost:
                continue
            p = clf.predict_proba(X_feat[fidx:fidx + 1])[0, 1]
            if p >= prob_thr:
                diag["p_tp_fire"] += 1
                exit_ret = ret_now; closed = "tp_exit"; break
        out[j] = exit_ret
        diag[closed] += 1
    if return_diag:
        return out, diag
    return out


def simulate_dynamic_exit_sl(paths, trade_ids_sorted, clf_tp, clf_sl,
                             X_feat, ev_idx, thr_tp, thr_sl, taker_cost,
                             min_hold_s=MIN_HOLD_S, max_lookahead=MFE_LOOKAHEAD):
    """
    Dynamic exit with TWO learned thresholds (TP + SL), first trigger
    wins (SL check BEFORE TP, as in the user's sketch).

    Per step:
      - P_sl = clf_sl.predict_proba(...): if P_sl >= thr_sl AND ret < 0 -> SL exit
      - otherwise P_tp = clf_tp.predict_proba(...): if P_tp >= thr_tp AND ret > taker -> TP exit
      - otherwise continue; if nothing hits -> terminal.

    Returns (out_pnl, diag). diag counts per arm how many trades there
    close (sl_exit / tp_exit / terminal) and how often the thresholds fire.
    """
    n_total = len(X_feat)
    out = np.zeros(len(trade_ids_sorted), dtype=np.float32)
    diag = dict(sl_exit=0, tp_exit=0, terminal=0,
                p_sl_fire=0, p_tp_fire=0, steps_eval=0)
    for j, i in enumerate(trade_ids_sorted):
        path = paths[i]
        n_path = int(np.sum(~np.isnan(path)))
        exit_ret = float(path[n_path - 1]) if n_path > 0 else 0.0
        closed = "terminal"
        for k in range(min_hold_s, min(max_lookahead, n_path)):
            fidx = ev_idx[i] + k
            if fidx >= n_total:
                break
            ret_now = float(path[k - 1])
            diag["steps_eval"] += 1
            # 1) SL arm first (only meaningful when currently in the red)
            if ret_now < 0:
                p_sl = clf_sl.predict_proba(X_feat[fidx:fidx + 1])[0, 1]
                if p_sl >= thr_sl:
                    diag["p_sl_fire"] += 1
                    exit_ret = ret_now; closed = "sl_exit"; break
            # 2) TP arm (only meaningful when above taker cost)
            if ret_now > taker_cost:
                p_tp = clf_tp.predict_proba(X_feat[fidx:fidx + 1])[0, 1]
                if p_tp >= thr_tp:
                    diag["p_tp_fire"] += 1
                    exit_ret = ret_now; closed = "tp_exit"; break
        out[j] = exit_ret
        diag[closed] += 1
    return out, diag


def _sharpe(x):
    return float(x.mean() / x.std()) if len(x) > 1 and x.std() > 0 else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# DYNAMIC ENTRY (Experiment C/D)  —  mirrored logic to build_exit_steps /
# simulate_dynamic_exit. The entry may be shifted in time (0..ENTRY_
# WINDOW_S) or REJECTED entirely if no signal >= thr arrives.
#
# Important convention (user decision):
#   TP/SL are still measured ABSOLUTELY from the breakout point (k=0), NOT from the
#   shifted entry. The realised PnL is (exit path value) - (entry path value),
#   where the exit is the first TP/SL/terminal hit AFTER k_entry.
#   This keeps the exit profile identical to experiment A; the only difference is
#   the entry time. A rejected entry produces NO trade.
# ──────────────────────────────────────────────────────────────────────────────
ENTRY_WINDOW_S       = 60
ENTRY_PROB_THRESHOLDS = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]


def build_entry_steps(X_feat, ev_idx, paths, tp, sl, cost,
                      entry_window_s=ENTRY_WINDOW_S, min_abs_bps=MIN_ABS_BPS,
                      max_lookahead=MFE_LOOKAHEAD):
    """
    Builds (X_steps, y_steps, trade_of_step, k_of_step) for the ENTRY classification.
    Label (option 1): is an entry at second k profitable if from there one uses
    a fixed TP/SL (measured from k=0, first-touch after k) until exit?
      y = 1 if the resulting net PnL (gross - cost) > min_abs_bps.
    Features = microstructure row at time k (X_feat[ev_idx+k]) — causal <= t.
    """
    Xs, ys, t_of, k_of = [], [], [], []
    n_total = len(X_feat)
    for i, idx in enumerate(ev_idx):
        path = paths[i]
        n_path = int(np.sum(~np.isnan(path)))
        if n_path == 0:
            continue
        kmax = min(entry_window_s, n_path - 1)
        for k in range(0, kmax):
            fidx = idx + k
            if fidx >= n_total:
                break
            entry_ret = float(path[k - 1]) if k > 0 else 0.0
            # Exit: first TP/SL hit (absolute from k=0) at a step > k
            exit_ret = float(path[n_path - 1])
            for kk in range(max(k + 1, MIN_HOLD_S), min(max_lookahead, n_path)):
                v = float(path[kk - 1])
                if v >= tp:
                    exit_ret = float(tp); break
                if v <= -sl:
                    exit_ret = float(-sl); break
            gross = exit_ret - entry_ret
            net = gross - cost
            Xs.append(X_feat[fidx]); ys.append(int(net > min_abs_bps))
            t_of.append(i); k_of.append(k)
    if not Xs:
        return (np.empty((0, X_feat.shape[1]), np.float32), np.empty(0, int),
                np.empty(0, int), np.empty(0, int))
    return (np.array(Xs, np.float32), np.array(ys, np.int32),
            np.array(t_of, int), np.array(k_of, int))


def simulate_dynamic_entry(paths, trade_ids_sorted, clf, X_feat, ev_idx,
                           prob_thr, tp, sl, dyn_exit_clf=None, exit_prob_thr=None,
                           exit_sl_bps=None, entry_window_s=ENTRY_WINDOW_S,
                           min_hold_s=MIN_HOLD_S, max_lookahead=MFE_LOOKAHEAD):
    """
    Dynamic entry. Per trade:
      - Find the first k in [0, entry_window_s) with P(good_entry) >= prob_thr.
      - No such k -> entry REJECTED (no trade; gross=NaN).
      - Otherwise enter at k_entry (entry price = path[k_entry-1], 0 at k=0).
      - Exit:
          * dyn_exit_clf is None  -> fixed TP/SL from k=0, first-touch AFTER k_entry
            (Experiment C).
          * dyn_exit_clf set  -> learned exit from k_entry + fixed SL floor
            exit_sl_bps (Experiment D).
      - gross_pnl = exit_ret - entry_ret.
    Returns (gross_array_with_NaN_for_rejected, n_taken, n_rejected).
    """
    n_total = len(X_feat)
    out = np.full(len(trade_ids_sorted), np.nan, dtype=np.float32)
    n_taken = n_rejected = 0
    for j, i in enumerate(trade_ids_sorted):
        path = paths[i]
        n_path = int(np.sum(~np.isnan(path)))
        if n_path == 0:
            n_rejected += 1; continue
        # 1) find the entry time
        k_entry = None
        for k in range(0, min(entry_window_s, n_path - 1)):
            fidx = ev_idx[i] + k
            if fidx >= n_total:
                break
            p = clf.predict_proba(X_feat[fidx:fidx + 1])[0, 1]
            if p >= prob_thr:
                k_entry = k; break
        if k_entry is None:
            n_rejected += 1; continue
        n_taken += 1
        entry_ret = float(path[k_entry - 1]) if k_entry > 0 else 0.0
        # 2) Exit
        if dyn_exit_clf is None:
            # Experiment C: fixed TP/SL from k=0, first-touch after k_entry
            exit_ret = float(path[n_path - 1])
            for kk in range(max(k_entry + 1, min_hold_s), min(max_lookahead, n_path)):
                v = float(path[kk - 1])
                if v >= tp:
                    exit_ret = float(tp); break
                if v <= -sl:
                    exit_ret = float(-sl); break
        else:
            # Experiment D: learned exit from k_entry + fixed SL floor
            exit_ret = float(path[n_path - 1])
            for kk in range(max(k_entry + 1, min_hold_s), min(max_lookahead, n_path)):
                fidx = ev_idx[i] + kk
                if fidx >= n_total:
                    break
                ret_now = float(path[kk - 1])
                if exit_sl_bps is not None and ret_now <= -exit_sl_bps:
                    exit_ret = -float(exit_sl_bps); break
                if ret_now <= 0:   # learned TP only meaningful when in profit
                    continue
                pe = dyn_exit_clf.predict_proba(X_feat[fidx:fidx + 1])[0, 1]
                if pe >= exit_prob_thr:
                    exit_ret = ret_now; break
        out[j] = exit_ret - entry_ret
    return out, n_taken, n_rejected


# ──────────────────────────────────────────────────────────────────────────────
def run_entry_exit_experiments(assets=("btc",), horizons=("5s",), thresholds=(15,),
                               cluster_tag_filter=None,
                               entry_window_s=ENTRY_WINDOW_S,
                               entry_thr_grid=tuple(ENTRY_PROB_THRESHOLDS),
                               exit_thr_grid=tuple(EXIT_PROB_THRESHOLDS),
                               n_jobs=6, max_hours=None):
    """
    Experiments A-D on the SAME cluster events, per-fold honest OOS:

      A  fixed entry (k=0) + best grid TP/SL        (baseline, == WS3 logic)
      B  fixed entry (k=0) + dynamic exit             (isolates exit effect)
      C  dynamic entry     + best grid TP/SL        (isolates entry effect)
      D  dynamic entry     + dynamic exit             (both learned)

    Common rules (comparability):
      - identical event set per cluster (cluster_trades CSV)
      - NO entry-yes/no filter, NO cluster-id as feature
      - Features = top_mfe|top_mae (like exit mode)
      - best grid TP/SL per fold chosen on TRAIN (applies to A and C, and as
        SL floor in D); the same (tp,sl) is used in A/C/D within the fold
      - TP/SL measured absolutely from the breakout point (k=0) (user decision)
      - the dynamic entry MAY reject; n_taken/n_rejected are reported
      - taker + maker costs in parallel
    """
    import lightgbm as lgb
    from sklearn.impute import SimpleImputer
    from common.data_loader import load_dataset
    from common.config import RESULTS_DIR, SPREAD_BPS, MAKER_COST_BPS

    in_dir = RESULTS_DIR / "cluster_mfe"
    out_dir = RESULTS_DIR / "entry_exit_experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    def _fit_clf(Xc, ys):
        nval = max(int(len(Xc) * 0.1), 50)
        c = lgb.LGBMClassifier(n_estimators=300, num_leaves=31, max_depth=6,
                               learning_rate=0.05, class_weight="balanced",
                               n_jobs=n_jobs, verbose=-1, random_state=42)
        if len(Xc) <= nval or len(np.unique(ys[:-nval])) < 2:
            c.fit(Xc, ys)
        else:
            c.fit(Xc[:-nval], ys[:-nval],
                  eval_set=[(Xc[-nval:], ys[-nval:])],
                  callbacks=[lgb.early_stopping(40), lgb.log_evaluation(0)])
        return c

    class _C:
        def __init__(s, m, im): s.m, s.im = m, im
        def predict_proba(s, Xrow): return s.m.predict_proba(s.im.transform(Xrow))

    for asset in assets:
        taker_cost = SPREAD_BPS.get(asset, {}).get("fut", 10.0)
        maker_cost = MAKER_COST_BPS.get(asset, {}).get("fut", 4.0)
        for hz in horizons:
            tprint(f"━━ ENTRY/EXIT-EXPERIMENTS {asset.upper()}/{hz} ━━")
            try:
                X, y, info, feat_names = load_dataset(
                    target=hz, asset=asset, profile="tree", max_hours=max_hours)
            except Exception as e:
                logger.error("Load fail: %s", e); continue
            sel_cols, sel_names = top_mfe_mae_columns(feat_names, asset)
            tprint(f"  {len(sel_cols)} top_mfe|top_mae-Features")
            X_feat = X[:, sel_cols]
            try:
                _, y_1s_full, _, _ = load_dataset(
                    target="1s", asset=asset, target_only=True, max_hours=max_hours)
            except Exception as e:
                logger.error("1s load fail: %s", e); continue
            n_min = min(len(X_feat), len(y_1s_full))
            X_feat = X_feat[:n_min]; y_1s = y_1s_full[:n_min].astype(np.float32)

            for thr_bps in thresholds:
                cand = list(in_dir.glob(f"cluster_trades_*{asset}_{hz}_{thr_bps}bps*.csv"))
                if cluster_tag_filter:
                    cand = [c for c in cand if cluster_tag_filter in c.name]
                if not cand:
                    tprint(f"  {thr_bps}bps: no cluster_trades — skip"); continue

                for trades_csv in sorted(cand):
                    tag = trades_csv.stem.replace("cluster_trades_", "")
                    df = pd.read_csv(trades_csv)
                    if not {"event_index", "direction", "cluster"}.issubset(df.columns):
                        tprint(f"    {tag}: columns missing — skip"); continue
                    ev_idx = df["event_index"].values.astype(int)
                    dirs   = df["direction"].values.astype(float)
                    mfe    = (df["mfe_bps"].values.astype(float)
                              if "mfe_bps" in df else np.full(len(df), 999.0))
                    paths  = extract_price_paths(y_1s, ev_idx, dirs, MFE_LOOKAHEAD)
                    n_tr   = len(ev_idx)
                    if n_tr < 200:
                        tprint(f"    {tag}: only {n_tr} trades — skip"); continue
                    tprint(f"    === {tag}: {n_tr} Trades ===")

                    order = np.argsort(ev_idx)
                    block = n_tr // (N_FOLDS + 1)
                    if block < 30:
                        tprint(f"      too few trades per fold — skip"); continue

                    # Accumulators per experiment/cost
                    acc = {e: {"taker": [], "maker": []} for e in ("A", "B", "C", "D")}
                    take = {"C": [0, 0], "D": [0, 0]}   # [taken, rejected]

                    for f in range(N_FOLDS):
                        tr_end = (f + 1) * block
                        te_s, te_e = tr_end, min(tr_end + block, n_tr)
                        if te_e - te_s < 10:
                            continue
                        tr_tr = order[:tr_end]; te_tr = order[te_s:te_e]

                        # ── best grid TP/SL on TRAIN (applies to A and C, SL floor for D) ──
                        grid = _exit_grid_from_paths(paths[tr_tr])
                        best_tp, best_sl, best_v = grid[0][0], grid[0][1], -1e9
                        for (gtp, gsl) in grid:
                            g = np.array([_fixed_tpsl_pnl(paths[i], gtp, gsl)
                                          for i in tr_tr], np.float32)
                            v = float((g - taker_cost).mean())
                            if v > best_v:
                                best_v, best_tp, best_sl = v, gtp, gsl

                        # ── Exit classifier (B, D) on TRAIN ──
                        Xs, ys, _, _ = build_exit_steps(
                            X_feat, ev_idx[tr_tr], paths[tr_tr], mfe[tr_tr], taker_cost)
                        exit_clf = None; best_exit_thr = exit_thr_grid[0]
                        if len(Xs) >= 100 and ys.sum() >= 20:
                            imp_e = SimpleImputer(strategy="median")
                            exit_clf = _C(_fit_clf(imp_e.fit_transform(Xs), ys), imp_e)
                            # best exit-thr on TRAIN (Sharpe)
                            bsh = -1e9
                            for pt in exit_thr_grid:
                                g = simulate_dynamic_exit(
                                    paths, tr_tr, exit_clf, X_feat, ev_idx,
                                    pt, taker_cost, sl_bps=best_sl)
                                sh = _sharpe(g - taker_cost)
                                if sh > bsh: bsh, best_exit_thr = sh, pt

                        # ── Entry classifier (C, D) on TRAIN ──
                        Xen, yen, _, _ = build_entry_steps(
                            X_feat, ev_idx[tr_tr], paths[tr_tr],
                            best_tp, best_sl, taker_cost, entry_window_s)
                        entry_clf = None; best_entry_thr = entry_thr_grid[0]
                        if len(Xen) >= 100 and yen.sum() >= 20:
                            imp_n = SimpleImputer(strategy="median")
                            entry_clf = _C(_fit_clf(imp_n.fit_transform(Xen), yen), imp_n)
                            bsh = -1e9
                            for pt in entry_thr_grid:
                                g, nt, nr = simulate_dynamic_entry(
                                    paths, tr_tr, entry_clf, X_feat, ev_idx, pt,
                                    best_tp, best_sl, entry_window_s=entry_window_s)
                                gv = g[~np.isnan(g)]
                                sh = _sharpe(gv - taker_cost) if len(gv) else -1e9
                                if sh > bsh: bsh, best_entry_thr = sh, pt

                        # ── TEST fold: all four experiments ──
                        # A: fixed entry + best grid TP/SL
                        gA = np.array([_fixed_tpsl_pnl(paths[i], best_tp, best_sl)
                                       for i in te_tr], np.float32)
                        acc["A"]["taker"].append(gA - taker_cost)
                        acc["A"]["maker"].append(gA - maker_cost)

                        # B: fixed entry + dynamic exit
                        if exit_clf is not None:
                            gB = simulate_dynamic_exit(
                                paths, te_tr, exit_clf, X_feat, ev_idx,
                                best_exit_thr, taker_cost, sl_bps=best_sl)
                            acc["B"]["taker"].append(gB - taker_cost)
                            acc["B"]["maker"].append(gB - maker_cost)

                        # C: dynamic entry + best grid TP/SL
                        if entry_clf is not None:
                            gC, ntC, nrC = simulate_dynamic_entry(
                                paths, te_tr, entry_clf, X_feat, ev_idx,
                                best_entry_thr, best_tp, best_sl,
                                entry_window_s=entry_window_s)
                            mC = ~np.isnan(gC)
                            take["C"][0] += int(mC.sum()); take["C"][1] += int((~mC).sum())
                            if mC.any():
                                acc["C"]["taker"].append(gC[mC] - taker_cost)
                                acc["C"]["maker"].append(gC[mC] - maker_cost)

                        # D: dynamic entry + dynamic exit
                        if entry_clf is not None and exit_clf is not None:
                            gD, ntD, nrD = simulate_dynamic_entry(
                                paths, te_tr, entry_clf, X_feat, ev_idx,
                                best_entry_thr, best_tp, best_sl,
                                dyn_exit_clf=exit_clf, exit_prob_thr=best_exit_thr,
                                exit_sl_bps=best_sl, entry_window_s=entry_window_s)
                            mD = ~np.isnan(gD)
                            take["D"][0] += int(mD.sum()); take["D"][1] += int((~mD).sum())
                            if mD.any():
                                acc["D"]["taker"].append(gD[mD] - taker_cost)
                                acc["D"]["maker"].append(gD[mD] - maker_cost)

                    # ── Aggregation + Report ──
                    def _agg(e, cost):
                        if not acc[e][cost]:
                            return None
                        x = np.concatenate(acc[e][cost])
                        return dict(mean=float(x.mean()), sharpe=_sharpe(x),
                                    wr=float((x > 0).mean()) * 100, n=int(len(x)))

                    tprint(f"      ── Experiment results (OOS) ──")
                    labels = {"A": "fixed entry + grid TP/SL",
                              "B": "fixed entry + dynamic exit",
                              "C": "dynamic entry + grid TP/SL",
                              "D": "dynamic entry + dynamic exit"}
                    for e in ("A", "B", "C", "D"):
                        rt = _agg(e, "taker"); rm = _agg(e, "maker")
                        if rt is None:
                            tprint(f"        {e} {labels[e]:32s}: (no result)")
                            continue
                        extra = ""
                        if e in take:
                            t_, r_ = take[e]
                            tot = t_ + r_
                            extra = f"  [taken {t_}/{tot} = {100*t_/max(tot,1):.0f}%]"
                        tprint(f"        {e} {labels[e]:32s}: taker {rt['mean']:+.2f} "
                               f"(sh {rt['sharpe']:+.2f}, WR {rt['wr']:.1f}%, n={rt['n']}) | "
                               f"maker {rm['mean']:+.2f}{extra}")
                        rows.append(dict(
                            tag=tag, experiment=e, label=labels[e],
                            taker_mean=rt["mean"], taker_sharpe=rt["sharpe"],
                            taker_wr=rt["wr"], maker_mean=rm["mean"],
                            maker_sharpe=rm["sharpe"], n=rt["n"],
                            n_taken=(take[e][0] if e in take else rt["n"]),
                            n_rejected=(take[e][1] if e in take else 0)))

    if rows:
        out_csv = out_dir / "entry_exit_experiments_summary.csv"
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        tprint(f"  → {out_csv}")
    else:
        tprint("  (no results written)")


# ──────────────────────────────────────────────────────────────────────────────
def run_ws3d(assets=("btc",), horizons=("15s",), thresholds=(15,),
             cluster_tag_filter=None, exit_tp=40.0, exit_sl=22.0,
             exit_strategy="tp_only", thr_tp_grid=(0.50, 0.60, 0.70),
             thr_sl_grid=(0.50, 0.60, 0.70), n_jobs=6, max_hours=None):
    import lightgbm as lgb
    from common.data_loader import load_dataset
    from common.config import RESULTS_DIR, SPREAD_BPS, MAKER_COST_BPS

    in_dir = RESULTS_DIR / "cluster_mfe"
    out_dir = RESULTS_DIR / "dynamic_exit"
    out_dir.mkdir(parents=True, exist_ok=True)

    for asset in assets:
        taker_cost = SPREAD_BPS.get(asset, {}).get("fut", 9.0)
        maker_cost = MAKER_COST_BPS.get(asset, {}).get("fut", 4.0)

        for hz in horizons:
            tprint(f"━━ WS3d {asset.upper()}/{hz} ━━")
            try:
                X, y, info, feat_names = load_dataset(
                    target=hz, asset=asset, profile="tree", max_hours=max_hours)
            except Exception as e:
                logger.error("Load fail: %s", e); continue

            # top_mfe/mae feature selection (causal, train-only)
            sel_cols, sel_names = top_mfe_mae_columns(feat_names, asset)
            tprint(f"  {len(sel_cols)} top_mfe|top_mae features selected (profile=tree)")
            X_feat = X[:, sel_cols]

            # Load 1s returns as the path source SEPARATELY (like WS4) — y above is the
            # hz forward return, NOT 1s, and is unsuitable for path extraction.
            try:
                _, y_1s_full, _, _ = load_dataset(
                    target="1s", asset=asset, target_only=True, max_hours=max_hours)
            except Exception as e:
                logger.error("1s load fail: %s", e); continue
            n_min = min(len(X_feat), len(y_1s_full))
            X_feat = X_feat[:n_min]
            y_1s = y_1s_full[:n_min].astype(np.float32)

            for thr_bps in thresholds:
                # load cluster_trades from WS4 (same interface as ws3)
                cand = list(in_dir.glob(f"cluster_trades_*{asset}_{hz}_{thr_bps}bps*.csv"))
                if cluster_tag_filter:
                    cand = [c for c in cand if cluster_tag_filter in c.name]
                if not cand:
                    tprint(f"  {thr_bps}bps: no cluster_trades found — skip")
                    continue

                for trades_csv in sorted(cand):
                    tag = trades_csv.stem.replace("cluster_trades_", "")
                    df = pd.read_csv(trades_csv)
                    if not {"event_index", "direction", "cluster"}.issubset(df.columns):
                        tprint(f"    {tag}: columns missing — skip"); continue

                    ev_idx = df["event_index"].values.astype(int)
                    dirs   = df["direction"].values.astype(float)
                    mfe    = (df["mfe_bps"].values.astype(float)
                              if "mfe_bps" in df else np.full(len(df), 999.0))
                    paths  = extract_price_paths(y_1s, ev_idx, dirs, MFE_LOOKAHEAD)
                    n_tr   = len(ev_idx)
                    if n_tr < 200:
                        tprint(f"    {tag}: only {n_tr} trades — skip"); continue

                    tprint(f"    === {tag}: {n_tr} Trades ===")

                    # temporal sorting of the TRADES (index = time)
                    order = np.argsort(ev_idx)
                    block = n_tr // (N_FOLDS + 1)
                    if block < 30:
                        tprint(f"      too few trades per fold — skip"); continue

                    dyn_oos_taker, base_oos_taker = [], []
                    dyn_oos_maker, base_oos_maker = [], []
                    chosen_thr = []
                    # Diagnostic accumulators (summed over folds)
                    diag_tot = dict(sl_exit=0, tp_exit=0, terminal=0,
                                    p_sl_fire=0, p_tp_fire=0, steps_eval=0)
                    chosen_grid = []   # tp_sl_grid only

                    from sklearn.impute import SimpleImputer

                    class _C:
                        """Wrapper: imputes before predict_proba."""
                        def __init__(s, m, im): s.m, s.im = m, im
                        def predict_proba(s, Xrow):
                            return s.m.predict_proba(s.im.transform(Xrow))

                    def _fit_clf(Xs_c, ys):
                        nval = max(int(len(Xs_c) * 0.1), 50)
                        c = lgb.LGBMClassifier(
                            n_estimators=300, num_leaves=31, max_depth=6,
                            learning_rate=0.05, class_weight="balanced",
                            n_jobs=n_jobs, verbose=-1, random_state=42)
                        if len(Xs_c) <= nval or len(np.unique(ys[:-nval])) < 2:
                            c.fit(Xs_c, ys)
                        else:
                            c.fit(Xs_c[:-nval], ys[:-nval],
                                  eval_set=[(Xs_c[-nval:], ys[-nval:])],
                                  callbacks=[lgb.early_stopping(40), lgb.log_evaluation(0)])
                        return c

                    for f in range(N_FOLDS):
                        tr_end = (f + 1) * block
                        te_s, te_e = tr_end, min(tr_end + block, n_tr)
                        if te_e - te_s < 10:
                            continue
                        tr_trades = order[:tr_end]
                        te_trades = order[te_s:te_e]

                        # ── TP classifier (needed in both strategies) ──
                        Xs, ys, t_of, k_of = build_exit_steps(
                            X_feat, ev_idx[tr_trades], paths[tr_trades],
                            mfe[tr_trades], taker_cost)
                        if len(Xs) < 100 or ys.sum() < 20:
                            continue
                        imp = SimpleImputer(strategy="median")
                        Xs_c = imp.fit_transform(Xs)
                        clf = _fit_clf(Xs_c, ys)
                        cwrap = _C(clf, imp)

                        # BASELINE: fixed absolute TP/SL (in both strategies)
                        g_base = np.array([
                            _fixed_tpsl_pnl(paths[i], exit_tp, exit_sl)
                            for i in te_trades], dtype=np.float32)
                        base_oos_taker.append(g_base - taker_cost)
                        base_oos_maker.append(g_base - maker_cost)

                        if exit_strategy == "tp_only":
                            # choose prob_thr per fold on TRAIN (Sharpe)
                            best_thr, best_sh = EXIT_PROB_THRESHOLDS[0], -1e9
                            for pt in EXIT_PROB_THRESHOLDS:
                                g = simulate_dynamic_exit(
                                    paths, tr_trades, cwrap, X_feat, ev_idx, pt,
                                    taker_cost, sl_bps=exit_sl)
                                sh = _sharpe(g - taker_cost)
                                if sh > best_sh:
                                    best_sh, best_thr = sh, pt
                            chosen_thr.append(best_thr)
                            g_dyn, dg = simulate_dynamic_exit(
                                paths, te_trades, cwrap, X_feat, ev_idx, best_thr,
                                taker_cost, sl_bps=exit_sl, return_diag=True)
                        else:  # tp_sl_grid
                            # ── SL classifier (second label, same features) ──
                            Xs2, ys2, _, _ = build_sl_steps(
                                X_feat, ev_idx[tr_trades], paths[tr_trades],
                                sl_floor=exit_sl)
                            if len(Xs2) < 100 or ys2.sum() < 20:
                                # SL label degenerate (too few non-recoverers) -> skip fold
                                tprint(f"      Fold {f}: SL label degenerate "
                                       f"(n={len(Xs2)}, pos={int(ys2.sum())}) — skip")
                                base_oos_taker.pop(); base_oos_maker.pop()
                                del clf; gc.collect(); continue
                            imp2 = SimpleImputer(strategy="median")
                            Xs2_c = imp2.fit_transform(Xs2)
                            clf_sl = _fit_clf(Xs2_c, ys2)
                            cwrap_sl = _C(clf_sl, imp2)

                            # 2D grid (thr_tp x thr_sl) per-fold on TRAIN (Sharpe)
                            best_pair, best_sh = (thr_tp_grid[0], thr_sl_grid[0]), -1e9
                            for ptp in thr_tp_grid:
                                for psl in thr_sl_grid:
                                    g, _ = simulate_dynamic_exit_sl(
                                        paths, tr_trades, cwrap, cwrap_sl,
                                        X_feat, ev_idx, ptp, psl, taker_cost)
                                    sh = _sharpe(g - taker_cost)
                                    if sh > best_sh:
                                        best_sh, best_pair = sh, (ptp, psl)
                            chosen_grid.append(best_pair)
                            chosen_thr.append(best_pair[0])   # for log compatibility
                            g_dyn, dg = simulate_dynamic_exit_sl(
                                paths, te_trades, cwrap, cwrap_sl, X_feat, ev_idx,
                                best_pair[0], best_pair[1], taker_cost)
                            del clf_sl; gc.collect()

                        dyn_oos_taker.append(g_dyn - taker_cost)
                        dyn_oos_maker.append(g_dyn - maker_cost)
                        for kk in diag_tot:
                            diag_tot[kk] += dg.get(kk, 0)
                        del clf; gc.collect()

                    if not dyn_oos_taker:
                        tprint(f"      no valid folds — skip"); continue

                    dt = np.concatenate(dyn_oos_taker); bt = np.concatenate(base_oos_taker)
                    dm = np.concatenate(dyn_oos_maker); bm = np.concatenate(base_oos_maker)

                    tprint(f"      [{exit_strategy}] DYNAMIC  OOS taker: mean={dt.mean():+.2f} bps "
                           f"sharpe={_sharpe(dt):.3f} WR={(dt>0).mean():.1%} n={len(dt)}")
                    tprint(f"      FIXED TP OOS taker: mean={bt.mean():+.2f} bps "
                           f"sharpe={_sharpe(bt):.3f} WR={(bt>0).mean():.1%}")
                    # ── DIAGNOSIS block ──
                    n_dyn = diag_tot["sl_exit"] + diag_tot["tp_exit"] + diag_tot["terminal"]
                    if n_dyn > 0:
                        tprint(f"      DIAG OOS-Exits: terminal={diag_tot['terminal']} "
                               f"({diag_tot['terminal']/n_dyn:.0%})  "
                               f"tp_exit={diag_tot['tp_exit']} ({diag_tot['tp_exit']/n_dyn:.0%})  "
                               f"sl_exit={diag_tot['sl_exit']} ({diag_tot['sl_exit']/n_dyn:.0%})")
                        tprint(f"      DIAG fires: p_tp={diag_tot['p_tp_fire']} "
                               f"p_sl={diag_tot['p_sl_fire']}  steps_eval={diag_tot['steps_eval']}")
                        if diag_tot["terminal"] / n_dyn > 0.8:
                            tprint(f"      >80% terminal → classifier barely fires, "
                                   f"DYNAMIC≈FIXED is expected (explains identical WR)")
                    winner = "DYNAMIC" if dt.mean() > bt.mean() else "FIXED"
                    delta = dt.mean() - bt.mean()
                    qual = "meaningless (<0.5bps)" if abs(delta) < 0.5 else "real"
                    tprint(f"      → {winner} +{abs(delta):.2f}bps ({qual}). "
                           f"thr/Fold: {chosen_thr}")
                    if exit_strategy == "tp_sl_grid":
                        tprint(f"        (thr_tp,thr_sl)/Fold: {chosen_grid}")

                    rows = [
                        dict(tag=tag, strategy=f"dynamic_{exit_strategy}", cost="taker",
                             mean_net=round(float(dt.mean()),3), sharpe=round(_sharpe(dt),4),
                             win_rate=round(float((dt>0).mean()),4), n=int(len(dt))),
                        dict(tag=tag, strategy=f"dynamic_{exit_strategy}", cost="maker",
                             mean_net=round(float(dm.mean()),3), sharpe=round(_sharpe(dm),4),
                             win_rate=round(float((dm>0).mean()),4), n=int(len(dm))),
                        dict(tag=tag, strategy="fixed_tpsl", cost="taker",
                             mean_net=round(float(bt.mean()),3), sharpe=round(_sharpe(bt),4),
                             win_rate=round(float((bt>0).mean()),4), n=int(len(bt))),
                        dict(tag=tag, strategy="fixed_tpsl", cost="maker",
                             mean_net=round(float(bm.mean()),3), sharpe=round(_sharpe(bm),4),
                             win_rate=round(float((bm>0).mean()),4), n=int(len(bm))),
                        dict(tag=tag, strategy="DIAG", cost="-",
                             terminal=diag_tot["terminal"], tp_exit=diag_tot["tp_exit"],
                             sl_exit=diag_tot["sl_exit"], p_tp_fire=diag_tot["p_tp_fire"],
                             p_sl_fire=diag_tot["p_sl_fire"], steps_eval=diag_tot["steps_eval"]),
                    ]
                    pd.DataFrame(rows).to_csv(
                        out_dir / f"dynamic_vs_fixed_{tag}_{exit_strategy}.csv", index=False)

            del X, X_feat, y_1s; gc.collect()


def _fixed_tp_exit(path, frac, mfe, taker_cost, min_hold_s, max_lookahead):
    """Fixed baseline: exit at the first step >= frac*MFE (after min_hold), else terminal."""
    n_path = int(np.sum(~np.isnan(path)))
    if n_path == 0:
        return 0.0
    thr = frac * mfe
    for k in range(min_hold_s, min(max_lookahead, n_path)):
        if path[k - 1] >= thr:
            return float(path[k - 1])
    return float(path[n_path - 1])


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY PROBE (NEW): LGBM entry classifier on ALL breakouts
#  Question: is there a subset (selectable in advance from entry features) of
#  breakouts, is it OOS-profitable after costs? If not -> the exit classifier
#  is pointless. Strictly per fold, train-only.
# ══════════════════════════════════════════════════════════════════════════════
def find_breakouts(y_1s, threshold_bps, window_s):
    """CAUSAL breakout: the move over window_s is COMPLETE, entry
    only at the window end, direction = momentum of the PAST move, path = only
    future from entry. NO look-ahead — the defining move is NOT in the path.
    (Earlier: entry at the window start + forward sign -> the defining move
     in the path -> look-ahead; baseline gross would then be ~ threshold.)"""
    n = len(y_1s)
    csum = np.concatenate(([0.0], np.cumsum(y_1s, dtype=np.float64)))
    sw = (csum[window_s:] - csum[:-window_s]) * 10_000.0   # sw[j] = sum(y[j:j+window])
    events, dirs, last = [], [], -10**9
    for j in range(len(sw)):
        i = j + window_s            # entry only after the window [j, j+window) completes
        if i >= n:
            break
        if abs(sw[j]) >= threshold_bps and i - last >= window_s:
            events.append(i); dirs.append(1 if sw[j] > 0 else -1); last = i
    return np.array(events, int), np.array(dirs, int)


def use_tree_columns(feat_names, asset):
    """Entry feature indices: use_tree=True (3272 features in feature_keep).

    BUGFIX (2026-06): earlier '& use_cluster', which excluded 526 pure tree features
    (use_cluster ⊂ use_tree). For a tree-based entry classifier
    use_tree is the correct selection.

    CAUSALITY: does NOT come from use_cluster (that is a model-profile flag,
    no causality flag). It is guaranteed by the fact that only type=='feature'
    columns are loaded — the forward-looking ret_fwd_*/mfe_fwd_*/mae_fwd_*
    are type=='target' and hence not in the feature set at all. (Precondition:
    load_dataset is called with profile='tree', so feat_names exactly
    contains the use_tree columns — otherwise the intersection below applies.)
    """
    from common.config import KEEP_LIST
    fk_path = KEEP_LIST
    if not Path(fk_path).exists():
        for cand in [Path("results/selection/feature_keep.csv"),
                     Path("feature_keep.csv")]:
            if cand.exists():
                fk_path = cand; break
        else:
            raise FileNotFoundError(f"KEEP_LIST not found: {KEEP_LIST}")
    fk = pd.read_csv(fk_path)
    fk["use_tree"] = fk["use_tree"].astype(str).str.lower().isin(["true", "1"])
    sel = fk[fk["use_tree"]]
    name_to_idx = {n: i for i, n in enumerate(feat_names)}
    cols, names = [], []
    for col in sel["column"]:
        if col in name_to_idx:
            cols.append(name_to_idx[col]); names.append(col)
    return np.array(cols, dtype=int), names


def _exit_grid_from_paths(paths, max_tp=80):
    """Small TP/SL candidate grid from MFE/MAE quantiles (~4x3). Market prior;
    the selection from it is per fold on TRAIN (train-only)."""
    mfe = np.nanmax(paths, axis=1); mae = -np.nanmin(paths, axis=1)
    mfe = mfe[np.isfinite(mfe)]; mae = np.clip(mae[np.isfinite(mae)], 0, None)
    if len(mfe) < 10 or len(mae) < 10:
        return [(t, s) for t in (20, 30, 40, 50) for s in (12, 18, 25)]
    tp = sorted({int(min(max(round(v / 5) * 5, 15), max_tp)) for v in np.percentile(mfe, [40, 60, 75, 90])})
    sl = sorted({int(min(max(round(v / 2) * 2, 8), 40)) for v in np.percentile(mae, [40, 60, 80])})
    return [(t, s) for t in tp for s in sl]


def _fixed_tpsl_pnl(path, tp, sl, min_hold_s=MIN_HOLD_S, max_lookahead=MFE_LOOKAHEAD):
    """Gross PnL with a FIXED absolute TP/SL (first-touch), NO %MFE look-ahead."""
    n_path = int(np.sum(~np.isnan(path)))
    if n_path == 0:
        return 0.0
    for k in range(min_hold_s, min(max_lookahead, n_path)):
        v = path[k - 1]
        if v >= tp:
            return float(tp)
        if v <= -sl:
            return float(-sl)
    return float(path[n_path - 1])


def run_entry_probe(assets=("eth",), horizons=("5s",), thresholds=(20,),
                    exit_tp=40.0, exit_sl=22.0, exit_mode="fixed",
                    use_cluster_trades=False,
                    label_cost="maker", n_jobs=6, max_hours=None):
    import lightgbm as lgb
    from sklearn.impute import SimpleImputer
    from common.data_loader import load_dataset
    from common.config import RESULTS_DIR, SPREAD_BPS, MAKER_COST_BPS

    in_dir  = RESULTS_DIR / "cluster_mfe"
    out_dir = RESULTS_DIR / "entry_probe"; out_dir.mkdir(parents=True, exist_ok=True)

    for asset in assets:
        taker = SPREAD_BPS.get(asset, {}).get("fut", 10.0)
        maker = MAKER_COST_BPS.get(asset, {}).get("fut", 4.0)
        lbl_cost = maker if label_cost == "maker" else taker

        for hz in horizons:
            tprint(f"━━ ENTRY-PROBE {asset.upper()}/{hz}  fixed exit TP={exit_tp}/SL={exit_sl}  "
                   f"Label@{label_cost}({lbl_cost}bps) ━━")
            try:
                X, y, info, feat_names = load_dataset(
                    target=hz, asset=asset, profile="tree", max_hours=max_hours)
            except Exception as e:
                logger.error("Load fail: %s", e); continue
            sel_cols, sel_names = use_tree_columns(feat_names, asset)
            tprint(f"  Entry-Features: {len(sel_cols)} (use_tree, profile=tree). "
                   f"Note: high dimension -> overfit risk, per-fold OOS is the test.")
            X_feat = X[:, sel_cols]
            try:
                _, y1s_full, _, _ = load_dataset(
                    target="1s", asset=asset, target_only=True, max_hours=max_hours)
            except Exception as e:
                logger.error("1s load fail: %s", e); continue
            n_min = min(len(X_feat), len(y1s_full))
            X_feat = X_feat[:n_min]; y_1s = y1s_full[:n_min].astype(np.float32)
            win = {"5s": 5, "15s": 15}.get(hz, 15)

            for thr in thresholds:
                if use_cluster_trades:
                    cand = list(in_dir.glob(f"cluster_trades_*{asset}_{hz}_{thr}bps*.csv"))
                    if not cand:
                        tprint(f"  {thr}bps: no cluster_trades — skip"); continue
                    df = pd.read_csv(sorted(cand)[0])
                    ev = df["event_index"].values.astype(int)
                    dr = df["direction"].values.astype(float)
                    univ = "cluster"
                else:
                    ev, dr = find_breakouts(y_1s, thr, win); univ = "all"
                if len(ev) < 300:
                    tprint(f"  {thr}bps: only {len(ev)} events — skip"); continue

                paths = extract_price_paths(y_1s, ev, dr, MFE_LOOKAHEAD)
                if exit_mode == "grid":
                    grid = _exit_grid_from_paths(paths)
                    gross_grid = np.column_stack([
                        np.array([_fixed_tpsl_pnl(p, tp, sl) for p in paths], np.float32)
                        for (tp, sl) in grid])
                    tprint(f"    exit grid (chosen per fold on train): {len(grid)} candidates {grid}")
                else:
                    gross_fixed = np.array([_fixed_tpsl_pnl(p, exit_tp, exit_sl) for p in paths],
                                           dtype=np.float32)
                order = np.argsort(ev)
                block = len(ev) // (N_FOLDS + 1)
                if block < 50:
                    tprint(f"  {thr}bps: too few per fold — skip"); continue

                sel_t, sel_m, all_t, all_m, n_sel, chosen_pt = [], [], [], [], [], []
                chosen_exit = []
                for f in range(N_FOLDS):
                    tr_end = (f + 1) * block
                    te_s, te_e = tr_end, min(tr_end + block, len(ev))
                    if te_e - te_s < 10:
                        continue
                    tr, te = order[:tr_end], order[te_s:te_e]
                    # Exit: in grid mode, pick the best (TP,SL) per fold on TRAIN
                    if exit_mode == "grid":
                        bc = int(np.argmax((gross_grid[tr] - lbl_cost).mean(axis=0)))
                        gross_sel = gross_grid[:, bc]
                        chosen_exit.append(grid[bc])
                    else:
                        gross_sel = gross_fixed
                    ylab = (gross_sel[tr] - lbl_cost > 0).astype(int)
                    if ylab.sum() < 20 or (1 - ylab).sum() < 20:
                        continue
                    imp = SimpleImputer(strategy="median")
                    Xtr = imp.fit_transform(X_feat[ev[tr]])
                    nval = max(int(len(Xtr) * 0.1), 50)
                    clf = lgb.LGBMClassifier(
                        n_estimators=300, num_leaves=31, max_depth=6,
                        learning_rate=0.05, class_weight="balanced",
                        n_jobs=n_jobs, verbose=-1, random_state=42)
                    if len(np.unique(ylab[:-nval])) < 2:
                        clf.fit(Xtr, ylab)
                    else:
                        clf.fit(Xtr[:-nval], ylab[:-nval],
                                eval_set=[(Xtr[-nval:], ylab[-nval:])],
                                callbacks=[lgb.early_stopping(40), lgb.log_evaluation(0)])
                    # choose the prob threshold per fold on TRAIN (max train-net@lbl_cost)
                    ptr = clf.predict_proba(imp.transform(X_feat[ev[tr]]))[:, 1]
                    best_pt, best_v = 0.5, -1e9
                    for pt in (0.50, 0.55, 0.60, 0.65, 0.70):
                        m = ptr >= pt
                        if m.sum() < 20:
                            continue
                        v = float((gross_sel[tr][m] - lbl_cost).mean())
                        if v > best_v:
                            best_v, best_pt = v, pt
                    chosen_pt.append(best_pt)
                    # TEST: selected trades
                    pte = clf.predict_proba(imp.transform(X_feat[ev[te]]))[:, 1]
                    msel = pte >= best_pt
                    if msel.sum() > 0:
                        sel_t.append(gross_sel[te][msel] - taker)
                        sel_m.append(gross_sel[te][msel] - maker)
                    n_sel.append(int(msel.sum()))
                    all_t.append(gross_sel[te] - taker)
                    all_m.append(gross_sel[te] - maker)
                    del clf; gc.collect()

                if not all_t:
                    tprint(f"  {thr}bps: no valid folds — skip"); continue
                st = np.concatenate(sel_t) if sel_t else np.array([], np.float32)
                sm = np.concatenate(sel_m) if sel_m else np.array([], np.float32)
                at = np.concatenate(all_t); am = np.concatenate(all_m)
                sel_rate = len(st) / len(at) if len(at) else 0.0

                tprint(f"  {thr}bps [{univ}, {len(ev)} Events]:")
                if len(sm):
                    tprint(f"    ENTRY-FILTER OOS: taker={st.mean():+.2f}  maker={sm.mean():+.2f} bps  "
                           f"n_sel={len(sm)} ({sel_rate:.0%})  WR_maker={(sm>0).mean():.1%}")
                else:
                    tprint(f"    ENTRY-FILTER: 0 test trades selected (classifier too restrictive)")
                tprint(f"    ALL (baseline)  : taker={at.mean():+.2f}  maker={am.mean():+.2f} bps  n={len(at)}")
                edge = (sm.mean() - am.mean()) if len(sm) else float("nan")
                if len(sm) and sm.mean() > 0 and edge > 0:
                    verdict = "ENTRY-EDGE (maker>0 AND > baseline)"
                elif len(sm) and edge > 0:
                    verdict = "better than baseline, but maker<0"
                else:
                    verdict = "NO entry edge"
                tprint(f"    → maker-Δ vs baseline: {edge:+.2f} bps  | {verdict}  | prob_thr/Fold: {chosen_pt}")
                if exit_mode == "grid":
                    tprint(f"      Exit (TP,SL)/Fold: {chosen_exit}")

                tag = f"{asset}_{hz}_{thr}bps_{univ}_tp{int(exit_tp)}_sl{int(exit_sl)}"
                pd.DataFrame([
                    dict(tag=tag, arm="entry_filter", cost="taker",
                         mean_net=round(float(st.mean()), 3) if len(st) else None,
                         n=int(len(st)), sel_rate=round(sel_rate, 4)),
                    dict(tag=tag, arm="entry_filter", cost="maker",
                         mean_net=round(float(sm.mean()), 3) if len(sm) else None,
                         win_rate=round(float((sm > 0).mean()), 4) if len(sm) else None,
                         n=int(len(sm)), sel_rate=round(sel_rate, 4)),
                    dict(tag=tag, arm="all_baseline", cost="taker",
                         mean_net=round(float(at.mean()), 3), n=int(len(at))),
                    dict(tag=tag, arm="all_baseline", cost="maker",
                         mean_net=round(float(am.mean()), 3),
                         win_rate=round(float((am > 0).mean()), 4), n=int(len(am))),
                ]).to_csv(out_dir / f"entry_probe_{tag}.csv", index=False)
                tprint(f"    saved: entry_probe_{tag}.csv")

            del X, X_feat, y_1s; gc.collect()


# ══════════════════════════════════════════════════════════════════════════════
#  BREAKOUT PROBE: cluster-INDEPENDENT, three comparison arms in one run
#  Universe = find_breakouts(thr, window) — NO cluster.
#    Arm A: entry classifier (use_tree) + fixed TP/SL
#    Arm B: dynamic TP/SL exit (top_mfe|top_mae) on ALL breakouts, no entry
#    Arm C: Entry-Classifier + dyn. TP/SL-Exit
#    Baseline: ALL breakouts, fixed TP/SL (zero point)
#  Everything strictly per fold, train-only. Entry and exit classifiers use
#  deliberately different feature sets (Entry=use_tree, Exit=top_mfe|top_mae).
# ══════════════════════════════════════════════════════════════════════════════
def run_breakout_probe(assets=("eth",), horizons=("5s",), thresholds=(15, 20),
                       windows=None,
                       fix_tp_grid=(20.0, 30.0, 40.0, 50.0),
                       fix_sl_grid=(12.0, 18.0, 25.0),
                       thr_tp_grid=(0.50, 0.60, 0.70),
                       thr_sl_grid=(0.50, 0.60, 0.70),
                       label_cost="maker",
                       entry_prob_grid=(0.50, 0.55, 0.60, 0.65, 0.70),
                       n_jobs=6, max_hours=None):
    """
    Cluster-INDEPENDENT breakout probe on find_breakouts(thr, window).

    Two test arms + one reference row, strictly per fold, train-only:
      Arm A  : LGBM-Entry-Classifier (use_tree) + FIXES TP/SL.
               The fixed (tp,sl) is chosen per fold from fix_tp_grid x fix_sl_grid on
               ALL train breakouts (option 1: TP/SL first, then ONE
               entry classifier with label = 'fixExit(tp,sl) > lbl_cost').
      Arm C  : the same entry classifier + DYNAMIC TP/SL (two classifiers,
               top_mfe|top_mae, (thr_tp,thr_sl) grid per-fold on Train).
      BASELINE (reference, NO arm): all breakouts, the same per-fold chosen
               fixed (tp,sl). Serves only as an interpretation anchor for arm A.

    A and C share EXACTLY the same entry classifier and the same
    selected test subset -> difference A vs C = only the exit type (fixed vs dyn).
    """
    import lightgbm as lgb
    from sklearn.impute import SimpleImputer
    from common.data_loader import load_dataset
    from common.config import SPREAD_BPS, MAKER_COST_BPS, RESULTS_DIR

    out_dir = RESULTS_DIR / "lgbm_dynamic_tp_sl"; out_dir.mkdir(parents=True, exist_ok=True)
    fix_grid = [(tp, sl) for tp in fix_tp_grid for sl in fix_sl_grid]

    def _fit_clf(Xc, yl):
        nval = max(int(len(Xc) * 0.1), 50)
        c = lgb.LGBMClassifier(
            n_estimators=300, num_leaves=31, max_depth=6, learning_rate=0.05,
            class_weight="balanced", n_jobs=n_jobs, verbose=-1, random_state=42)
        if len(Xc) <= nval or len(np.unique(yl[:-nval])) < 2:
            c.fit(Xc, yl)
        else:
            c.fit(Xc[:-nval], yl[:-nval], eval_set=[(Xc[-nval:], yl[-nval:])],
                  callbacks=[lgb.early_stopping(40), lgb.log_evaluation(0)])
        return c

    class _C:
        def __init__(s, m, im): s.m, s.im = m, im
        def predict_proba(s, Xrow): return s.m.predict_proba(s.im.transform(Xrow))

    for asset in assets:
        taker = SPREAD_BPS.get(asset, {}).get("fut", 10.0)
        maker = MAKER_COST_BPS.get(asset, {}).get("fut", 4.0)
        lbl_cost = maker if label_cost == "maker" else taker

        for hz in horizons:
            tprint(f"━━ BREAKOUT-PROBE {asset.upper()}/{hz}  "
                   f"fix-grid={len(fix_grid)} dyn-grid={len(thr_tp_grid)*len(thr_sl_grid)}  "
                   f"Label@{label_cost}({lbl_cost}bps) ━━")
            try:
                X, y, info, feat_names = load_dataset(
                    target=hz, asset=asset, profile="tree", max_hours=max_hours)
            except Exception as e:
                logger.error("Load fail: %s", e); continue
            entry_cols, _ = use_tree_columns(feat_names, asset)
            exit_cols, _  = top_mfe_mae_columns(feat_names, asset)
            tprint(f"  Entry-Features: {len(entry_cols)} (use_tree)  |  "
                   f"Exit-Features: {len(exit_cols)} (top_mfe|top_mae)")
            X_entry = X[:, entry_cols]
            X_exit  = X[:, exit_cols]
            try:
                _, y1s_full, _, _ = load_dataset(
                    target="1s", asset=asset, target_only=True, max_hours=max_hours)
            except Exception as e:
                logger.error("1s load fail: %s", e); continue
            n_min = min(len(X_entry), len(X_exit), len(y1s_full))
            X_entry = X_entry[:n_min]; X_exit = X_exit[:n_min]
            y_1s = y1s_full[:n_min].astype(np.float32)
            del X; gc.collect()

            win_default = {"5s": 5, "15s": 15}.get(hz, 15)
            win_list = windows if windows else [win_default]

            for thr in thresholds:
                for win in win_list:
                    ev, dr = find_breakouts(y_1s, thr, win)
                    if len(ev) < 300:
                        tprint(f"  thr={thr} win={win}: only {len(ev)} breakouts — skip"); continue
                    paths = extract_price_paths(y_1s, ev, dr, MFE_LOOKAHEAD)
                    mfe_ev = np.nanmax(paths, axis=1)
                    # PnL matrix for ALL fixed (tp,sl) candidates (no training):
                    # gross_fix_grid[i, c] = fixExit PnL of breakout i under grid pair c
                    gross_fix_grid = np.column_stack([
                        np.array([_fixed_tpsl_pnl(p, tp, sl) for p in paths], np.float32)
                        for (tp, sl) in fix_grid])
                    order = np.argsort(ev)
                    block = len(ev) // (N_FOLDS + 1)
                    if block < 50:
                        tprint(f"  thr={thr} win={win}: too few per fold — skip"); continue

                    acc = {a: {"t": [], "m": []} for a in
                           ["A_entry_fix", "C_entry_dyn", "ref_base_fix"]}
                    n_sel, chosen_pt, chosen_grid, chosen_fix = [], [], [], []
                    lbl_bal = []   # (tp_pos_rate, sl_pos_rate, n_tp_steps, n_sl_steps) per fold
                    diag = dict(sl_exit=0, tp_exit=0, terminal=0,
                                p_sl_fire=0, p_tp_fire=0, steps_eval=0)

                    for f in range(N_FOLDS):
                        tr_end = (f + 1) * block
                        te_s, te_e = tr_end, min(tr_end + block, len(ev))
                        if te_e - te_s < 10:
                            continue
                        tr, te = order[:tr_end], order[te_s:te_e]

                        # ── Option 1: fixed (tp,sl) per fold on ALL train breakouts ──
                        # best mean net@lbl_cost over the fixed grid (no filter)
                        bc = int(np.argmax(
                            (gross_fix_grid[tr] - lbl_cost).mean(axis=0)))
                        tp_f, sl_f = fix_grid[bc]
                        chosen_fix.append((tp_f, sl_f))
                        gross_fixed = gross_fix_grid[:, bc]   # fixed exit of this fold

                        # ── BASELINE reference: all test breakouts, this fixed (tp,sl) ──
                        acc["ref_base_fix"]["t"].append(gross_fixed[te] - taker)
                        acc["ref_base_fix"]["m"].append(gross_fixed[te] - maker)

                        # ── ONE entry classifier, label from the chosen fixed exit ──
                        ylab = (gross_fixed[tr] - lbl_cost > 0).astype(int)
                        have_entry = ylab.sum() >= 20 and (1 - ylab).sum() >= 20
                        msel_te = None
                        if have_entry:
                            imp_e = SimpleImputer(strategy="median")
                            Xtr_e = imp_e.fit_transform(X_entry[ev[tr]])
                            clf_e = _fit_clf(Xtr_e, ylab)
                            ptr = clf_e.predict_proba(imp_e.transform(X_entry[ev[tr]]))[:, 1]
                            best_pt, best_v = entry_prob_grid[0], -1e9
                            for pt in entry_prob_grid:
                                m = ptr >= pt
                                if m.sum() < 20: continue
                                v = float((gross_fixed[tr][m] - lbl_cost).mean())
                                if v > best_v: best_v, best_pt = v, pt
                            chosen_pt.append(best_pt)
                            pte = clf_e.predict_proba(imp_e.transform(X_entry[ev[te]]))[:, 1]
                            msel_te = pte >= best_pt
                            if msel_te.sum() > 0:
                                # Arm A: selected trades, fixed exit
                                acc["A_entry_fix"]["t"].append(gross_fixed[te][msel_te] - taker)
                                acc["A_entry_fix"]["m"].append(gross_fixed[te][msel_te] - maker)
                            n_sel.append(int(msel_te.sum()))
                            del clf_e; gc.collect()
                        else:
                            chosen_pt.append(None); n_sel.append(0)

                        # ── dynamic exit classifier (for arm C) ──
                        Xs_tp, ys_tp, _, _ = build_exit_steps(
                            X_exit, ev[tr], paths[tr], mfe_ev[tr], taker)
                        Xs_sl, ys_sl, _, _ = build_sl_steps(
                            X_exit, ev[tr], paths[tr], sl_floor=sl_f)
                        # log label balance: explains whether the exit classifier
                        # can learn at all (sl_pos_rate ~0 -> SL never fires).
                        lbl_bal.append((
                            round(float(ys_tp.mean()), 4) if len(ys_tp) else None,
                            round(float(ys_sl.mean()), 4) if len(ys_sl) else None,
                            int(len(ys_tp)), int(len(ys_sl))))
                        have_exit = (len(Xs_tp) >= 100 and ys_tp.sum() >= 20 and
                                     len(Xs_sl) >= 100 and ys_sl.sum() >= 20)
                        if have_entry and have_exit and msel_te is not None and msel_te.sum() > 0:
                            imp_tp = SimpleImputer(strategy="median")
                            clf_tp = _fit_clf(imp_tp.fit_transform(Xs_tp), ys_tp)
                            cw_tp = _C(clf_tp, imp_tp)
                            imp_sl = SimpleImputer(strategy="median")
                            clf_sl = _fit_clf(imp_sl.fit_transform(Xs_sl), ys_sl)
                            cw_sl = _C(clf_sl, imp_sl)
                            # dynamic (thr_tp,thr_sl) grid per fold on the SELECTED
                            # train trades (same selection as entry, fairer to arm C).
                            # ptr is defined here because this block presupposes have_entry.
                            tr_sel = tr[ptr >= best_pt]
                            if len(tr_sel) < 20:
                                tr_sel = tr
                            best_pair, best_sh = (thr_tp_grid[0], thr_sl_grid[0]), -1e9
                            for ptp in thr_tp_grid:
                                for psl in thr_sl_grid:
                                    g, _ = simulate_dynamic_exit_sl(
                                        paths, tr_sel, cw_tp, cw_sl, X_exit, ev,
                                        ptp, psl, taker)
                                    sh = _sharpe(g - taker)
                                    if sh > best_sh: best_sh, best_pair = sh, (ptp, psl)
                            chosen_grid.append(best_pair)
                            te_sel = te[msel_te]
                            g_dyn_sel, dg = simulate_dynamic_exit_sl(
                                paths, te_sel, cw_tp, cw_sl, X_exit, ev,
                                best_pair[0], best_pair[1], taker)
                            acc["C_entry_dyn"]["t"].append(g_dyn_sel - taker)
                            acc["C_entry_dyn"]["m"].append(g_dyn_sel - maker)
                            for kk in diag: diag[kk] += dg.get(kk, 0)
                            del clf_tp, clf_sl; gc.collect()
                        else:
                            chosen_grid.append(None)

                    if not acc["ref_base_fix"]["t"]:
                        tprint(f"  thr={thr} win={win}: no valid folds — skip"); continue

                    def _cat(arm, c):
                        L = acc[arm][c]
                        return np.concatenate(L) if L else np.array([], np.float32)

                    rows = []
                    tprint(f"  thr={thr} win={win} [{len(ev)} Breakouts]:")
                    base_m = _cat("ref_base_fix", "m")
                    for arm, lab in [("ref_base_fix", "REF base all+fix"),
                                     ("A_entry_fix", "A entry+fix     "),
                                     ("C_entry_dyn", "C entry+dyn     ")]:
                        at = _cat(arm, "t"); am = _cat(arm, "m")
                        if len(am):
                            edge = am.mean() - base_m.mean()
                            tag_edge = "" if arm == "ref_base_fix" else f"  Δvs.REF_m={edge:+.2f}"
                            tprint(f"    {lab}: taker={at.mean():+.2f}  maker={am.mean():+.2f} bps "
                                   f"WR_m={(am>0).mean():.1%}  n={len(am)}{tag_edge}")
                        else:
                            tprint(f"    {lab}: 0 Trades")
                        for c, arr in [("taker", at), ("maker", am)]:
                            rows.append(dict(
                                tag=f"{asset}_{hz}_{thr}bps_win{win}", arm=arm, cost=c,
                                is_reference=(arm == "ref_base_fix"),
                                mean_net=round(float(arr.mean()), 3) if len(arr) else None,
                                sharpe=round(_sharpe(arr), 4) if len(arr) else None,
                                win_rate=round(float((arr > 0).mean()), 4) if len(arr) else None,
                                n=int(len(arr))))
                    # A vs C directly (same trades, only exit type)
                    a_m = _cat("A_entry_fix", "m"); c_m = _cat("C_entry_dyn", "m")
                    if len(a_m) and len(c_m):
                        tprint(f"    → C−A (dyn vs fix, maker): {c_m.mean()-a_m.mean():+.2f} bps")
                    n_dyn = diag["sl_exit"] + diag["tp_exit"] + diag["terminal"]
                    if n_dyn > 0:
                        tprint(f"    DIAG dyn-exit: terminal={diag['terminal']} ({diag['terminal']/n_dyn:.0%}) "
                               f"tp={diag['tp_exit']} ({diag['tp_exit']/n_dyn:.0%}) "
                               f"sl={diag['sl_exit']} ({diag['sl_exit']/n_dyn:.0%})  "
                               f"fires tp={diag['p_tp_fire']} sl={diag['p_sl_fire']}")
                        if diag["terminal"] / n_dyn > 0.8:
                            tprint(f"    >80% terminal → dyn exit barely fires, C≈A expected")
                    tprint(f"    entry_thr/Fold: {chosen_pt}")
                    tprint(f"    fix(tp,sl)/Fold: {chosen_fix}  | dyn(thr_tp,thr_sl)/Fold: {chosen_grid}")
                    tprint(f"    sel_n/Fold: {n_sel}")
                    tprint(f"    LABEL-BAL/Fold (tp_pos, sl_pos, n_tp, n_sl): {lbl_bal}")
                    rows.append(dict(tag=f"{asset}_{hz}_{thr}bps_win{win}", arm="DIAG", cost="-",
                                     terminal=diag["terminal"], tp_exit=diag["tp_exit"],
                                     sl_exit=diag["sl_exit"], p_tp_fire=diag["p_tp_fire"],
                                     p_sl_fire=diag["p_sl_fire"], steps_eval=diag["steps_eval"]))
                    pd.DataFrame(rows).to_csv(
                        out_dir / f"lgbm_dynamic_tp_sl_{asset}_{hz}_{thr}bps_win{win}.csv", index=False)
                    tprint(f"    saved: lgbm_dynamic_tp_sl_{asset}_{hz}_{thr}bps_win{win}.csv")

            del X_entry, X_exit, y_1s; gc.collect()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--asset", nargs="+", default=["eth"])
    p.add_argument("--hz", nargs="+", default=["5s"])
    p.add_argument("--thresholds", nargs="+", type=int, default=[20])
    p.add_argument("--mode", choices=["entry", "exit", "breakout", "experiments"], default="entry",
                   help="entry = LGBM entry probe (old); exit = dynamic exit (cluster); "
                        "breakout = cluster-INDEP. three arms (entry / dyn-exit / both) on find_breakouts")
    p.add_argument("--windows", nargs="+", type=int, default=None,
                   help="breakout mode: breakout window (s); default = {5s:5,15s:15} per hz")
    p.add_argument("--universe", choices=["all", "cluster"], default="all",
                   help="entry mode: all = all breakouts (find_breakouts); cluster = cluster_trades")
    p.add_argument("--exit-tp", type=float, default=40.0, help="fixed absolute TP (bps), only exit-mode=fixed")
    p.add_argument("--exit-sl", type=float, default=22.0, help="fixed absolute SL (bps), only exit-mode=fixed")
    p.add_argument("--exit-mode", choices=["fixed", "grid"], default="fixed",
                   help="fixed = one TP/SL; grid = best TP/SL chosen per fold on train (honest)")
    p.add_argument("--exit-strategy", choices=["tp_only", "tp_sl_grid"], default="tp_only",
                   help="exit mode (--mode exit): tp_only = old (learned TP + fixed SL); "
                        "tp_sl_grid = TP AND SL classifier, (thr_tp,thr_sl) per-fold grid on Train")
    p.add_argument("--thr-tp-grid", nargs="+", type=float, default=[0.50, 0.60, 0.70],
                   help="tp_sl_grid: TP prob-threshold candidates")
    p.add_argument("--thr-sl-grid", nargs="+", type=float, default=[0.50, 0.60, 0.70],
                   help="tp_sl_grid: SL prob-threshold candidates")
    p.add_argument("--fix-tp-grid", nargs="+", type=float, default=[20.0, 30.0, 40.0, 50.0],
                   help="breakout mode arm A: fixed TP candidates (bps), chosen per fold on train")
    p.add_argument("--fix-sl-grid", nargs="+", type=float, default=[12.0, 18.0, 25.0],
                   help="breakout mode arm A: fixed SL candidates (bps), chosen per fold on train")
    p.add_argument("--entry-prob-grid", nargs="+", type=float,
                   default=[0.50, 0.55, 0.60, 0.65, 0.70],
                   help="breakout mode: entry-prob thresholds (above which LGBM trades the breakout)")
    p.add_argument("--label-cost", choices=["maker", "taker"], default="maker",
                   help="cost regime against which 'profitable' is labelled")
    p.add_argument("--cluster-tag-filter", default=None,
                   help="exit mode: e.g. 'kmeans_pca150_k6' to run only this combo")
    p.add_argument("--n-jobs", type=int, default=6)
    p.add_argument("--entry-window-s", type=int, default=ENTRY_WINDOW_S,
                   help="max seconds the dynamic entry may be shifted (experiment C/D)")
    p.add_argument("--entry-thr-grid", nargs="+", type=float, default=list(ENTRY_PROB_THRESHOLDS),
                   help="thresholds for the dynamic entry classifier (best on train)")
    p.add_argument("--max-hours", type=int, default=None,
                   help="file cap as in WS4 (e.g. 1500/2000) — MUST match the cluster_trades")
    p.add_argument("--log-level", default="INFO")
    a = p.parse_args()
    logging.basicConfig(level=getattr(logging, a.log_level),
                        format="%(asctime)s  %(levelname)s  %(message)s",
                        datefmt="%H:%M:%S")
    if a.mode == "entry":
        run_entry_probe(assets=tuple(a.asset), horizons=tuple(a.hz),
                        thresholds=tuple(a.thresholds), exit_tp=a.exit_tp, exit_sl=a.exit_sl,
                        exit_mode=a.exit_mode,
                        use_cluster_trades=(a.universe == "cluster"),
                        label_cost=a.label_cost, n_jobs=a.n_jobs, max_hours=a.max_hours)
    elif a.mode == "experiments":
        run_entry_exit_experiments(
            assets=tuple(a.asset), horizons=tuple(a.hz), thresholds=tuple(a.thresholds),
            cluster_tag_filter=a.cluster_tag_filter,
            entry_window_s=a.entry_window_s,
            entry_thr_grid=tuple(a.entry_thr_grid),
            exit_thr_grid=tuple(a.thr_tp_grid),
            n_jobs=a.n_jobs, max_hours=a.max_hours)
    elif a.mode == "breakout":
        run_breakout_probe(assets=tuple(a.asset), horizons=tuple(a.hz),
                           thresholds=tuple(a.thresholds),
                           windows=tuple(a.windows) if a.windows else None,
                           fix_tp_grid=tuple(a.fix_tp_grid), fix_sl_grid=tuple(a.fix_sl_grid),
                           thr_tp_grid=tuple(a.thr_tp_grid), thr_sl_grid=tuple(a.thr_sl_grid),
                           entry_prob_grid=tuple(a.entry_prob_grid),
                           label_cost=a.label_cost, n_jobs=a.n_jobs, max_hours=a.max_hours)
    else:
        run_ws3d(assets=tuple(a.asset), horizons=tuple(a.hz),
                 thresholds=tuple(a.thresholds),
                 cluster_tag_filter=a.cluster_tag_filter,
                 exit_tp=a.exit_tp, exit_sl=a.exit_sl,
                 exit_strategy=a.exit_strategy,
                 thr_tp_grid=tuple(a.thr_tp_grid), thr_sl_grid=tuple(a.thr_sl_grid),
                 n_jobs=a.n_jobs, max_hours=a.max_hours)


if __name__ == "__main__":
    main()