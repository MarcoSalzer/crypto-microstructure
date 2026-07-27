# prediction/breakout_directional.py
# ==============================================================================
# Breakout-Conditioned Directional Backtest — LGBM trained ON breakouts only
# ==============================================================================
#
# WHAT THIS SCRIPT DOES
# ---------------------
# A breakout is a BACKWARD (already-realised, observable-at-t) move. At bar t the
# trailing return over `w` seconds is computed from the futures mid price and a
# breakout is flagged when |bwd_w[t]| > thr bps. The decision to trade is made
# AT t, AFTER the move — never using the forward outcome to select the entry.
# For each (window x bps-threshold) breakout definition it:
#   1. Restricts the dataset to breakout bars.
#   2. Trains a LightGBM regressor (forward return) ONLY on that breakout
#      population, expanding-window split + time embargo so test breakouts are
#      strictly later than train breakouts and no train forward-target overlaps
#      the test set.
#   3. Trades the sign of the OOF prediction (long/short/flat via a |pred|
#      threshold), pays the round-trip taker fee, reports net PnL / win rate /
#      directional accuracy per fold.
#   4. Runs two NULL baselines on the SAME breakout test population:
#         momentum : trade the sign of the breakout move itself (no model)
#         random   : trade a coin-flip direction (seeded)
#
# BACKWARD RETURN SOURCE
# ----------------------
# Engineered trailing-return columns (ret_mid_fut_{w}s) exist only for 1/15/60 s.
# To support an arbitrary window grid (e.g. 1/5/15/30 s, matching the clustering
# chapter), the trailing return is computed from mid_fut_1s via a TIME-based
# lookup (robust to duplicate timestamps / gaps). Where an engineered column
# exists, the computed series is validated against it (corr + MAD logged).
# NOTE: this counts breakout events differently from the raw forward-event count
# (|ret_fwd|>thr). Backward (here) vs forward (there) counts are similar but not
# identical by construction; here the breakout is a tradable entry trigger.
#
# DESIGN CHOICES
# --------------
#   - Train on breakout bars only; regressor + sign (comparable to all-bar
#     baseline, magnitude = confidence gate, no arbitrary flat-band).
#   - Stronger regularisation than all-bar model; --top-only returns to shrink.
#   - Look-ahead control: subset -> fold -> time embargo = max(window, horizon) s.
#
# Trading cost: round-trip Binance taker fee = 10.0 bps. --maker uses 4.0 bps.
#
# COMPARISON TO ALL-BAR BASELINE
# ------------------------------
# Not re-run here; if results/directional/directional_summary.csv exists its
# per-(asset,horizon) PnL is attached as `allbar_pnl_ref`.
#
# OUTPUTS (in RESULTS_DIR/breakout_directional/{profile[-top_x]}/)
#   breakout_dir_{asset}_w{w}_thr{thr}bps.csv              per-combo summary
#   breakout_dir_folds_{asset}_w{w}_thr{thr}bps_{hz}.csv   per-fold OOF metrics
#   breakout_directional_summary.csv                       top-level summary
# Checkpointing: a finished combo CSV is its skip marker on rerun.
#
# USAGE
#   python prediction/breakout_directional.py \
#       --assets btc eth \
#       --bo-windows 1 5 15 30 --bo-thresholds 10 15 20 30 40 \
#       --horizons 5s 15s 30s 60s \
#       --seeds 42 123 999 --n-folds 5 --n-jobs 8 --profile tree
# ==============================================================================
from __future__ import annotations

import os
import signal

import argparse
import ctypes
import gc
import logging
import sys
import time

import numpy as np
import pandas as pd
import lightgbm as lgb

logger = logging.getLogger(__name__)

TAKER_FEE_BPS = 10.0
MAKER_FEE_BPS = 4.0
TREE_N_FEATURES = 3349

# Engineered trailing-return columns exist for these windows (used to VALIDATE
# the from-mid computation; any window can still be requested).
ENGINEERED_BWD_WINDOWS = (1, 15, 60)

# Clustering-chapter grid.
DEFAULT_BO_WINDOWS = [1, 5, 15, 30]
DEFAULT_BO_THRESHOLDS = [10.0, 15.0, 20.0, 30.0, 40.0]

# Forward holding horizons (separate axis from the breakout window). Short by
# design: a breakout motivates a near-term continuation bet.
DEFAULT_HORIZONS = ["5s", "15s", "30s", "60s"]

MIN_TRAIN_ROWS = 500
MIN_TEST_ROWS = 100
MIN_BREAKOUTS = 1500   # per (combo, horizon) before we bother training

BREAKOUT_LGBM_PARAMS = {
    "objective":         "regression",
    "metric":            "mse",
    "num_leaves":        15,
    "max_depth":         4,
    "learning_rate":     0.05,
    "colsample_bytree":  0.5,
    "subsample":         0.8,
    "subsample_freq":    1,
    "reg_alpha":         0.5,
    "reg_lambda":        2.0,
    "min_child_samples": 30,
    "n_estimators":      500,
    "verbose":           -1,
}

TS_COL = "bucket_dt_utc"


def hz_seconds(hz):
    return int(hz.rstrip("s"))


# ==============================================================================
# RAM guard
# ==============================================================================

def check_ram_or_skip(n_rows, n_features=TREE_N_FEATURES,
                      safety_factor=1.4, min_headroom_gb=10.0):
    raw_gb = n_rows * n_features * 4 / 1e9
    need_gb = raw_gb * safety_factor + min_headroom_gb
    avail_gb = None
    try:
        import psutil
        avail_gb = psutil.virtual_memory().available / 1e9
    except Exception:
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        avail_gb = int(line.split()[1]) / 1e6
                        break
        except Exception:
            avail_gb = None
    if avail_gb is None:
        logger.warning("  RAM check unavailable — proceeding (matrix ~%.0f GB).", raw_gb)
        return True
    logger.info("  RAM check: matrix~%.0f GB, need~%.0f GB, available~%.0f GB",
                raw_gb, need_gb, avail_gb)
    if avail_gb < need_gb:
        logger.error("  INSUFFICIENT RAM — skipping. Free memory or use --max-hours.")
        return False
    return True


# ==============================================================================
# Feature selection
# ==============================================================================

def resolve_feature_columns(profile, top_only):
    """
    Feature selection via the pipeline's own convention:
      profile  -> get_feature_columns(profile=...) -> use_{profile} column
                  (tree/linear/cluster/anomaly/all), same as every other script.
      top_only -> optionally intersect with a pre-ranked top_{x} column
                  (returns/mfe/mae/short_horizon/long_horizon) to shrink the
                  feature set on the small breakout population. 'none' = off.
    """
    from common.data_loader import get_feature_columns
    from common.config import KEEP_LIST

    base = get_feature_columns(None, profile=profile)
    if top_only == "none":
        return base

    flag = f"top_{top_only}"
    keep = pd.read_csv(KEEP_LIST)
    if flag not in keep.columns:
        logger.warning("Column '%s' not in keep list — ignoring --top-only.", flag)
        return base
    top_cols = set(keep.loc[(keep["type"] == "feature") & keep[flag].astype(bool),
                            "column"])
    selected = [c for c in base if c in top_cols]
    logger.info("Profile '%s' restricted to %s: %d of %d features",
                profile, flag, len(selected), len(base))
    return selected


# ==============================================================================
# Time-based trailing return (robust to dup timestamps / gaps)
# ==============================================================================

def backward_return_bps(mid, ts_i8, window_s, tol_s=1.0):
    """
    bwd[t] = (mid[t] / mid[t - ~window_s s] - 1) * 1e4, using the bar closest to
    `window_s` seconds before t (searchsorted on the ascending clock), accepted
    only if its actual age is within `tol_s` of `window_s`. NaN otherwise.
    """
    n = len(mid)
    out = np.full(n, np.nan)
    if n == 0:
        return out
    w_ns = int(window_s * 1_000_000_000)
    tol_ns = int(tol_s * 1_000_000_000)
    j = np.searchsorted(ts_i8, ts_i8 - w_ns, side="right") - 1
    valid = j >= 0
    jj = np.clip(j, 0, n - 1)
    age = ts_i8 - ts_i8[jj]
    prev = mid[jj]
    ok = (valid & np.isfinite(mid) & np.isfinite(prev) & (prev > 0)
          & (np.abs(age - w_ns) <= tol_ns))
    out[ok] = (mid[ok] / prev[ok] - 1.0) * 1e4
    return out


# ==============================================================================
# Single-pass loader: features + forward targets + mid + ts + engineered bwd
# ==============================================================================

def load_features_targets_bwd(asset, horizons, windows, feat_names,
                              max_hours=None, seed=42):
    """
    Returns:
      X        : float32 [n, n_features]
      y_by_hz  : {hz: float64 [n]}      forward returns (NaNs preserved)
      mid      : float64 [n]            mid_fut_1s (NaNs preserved)
      ts_i8    : int64   [n]            ns clock (real time, or 1s row clock)
      eng_by_w : {w: float64 [n] bps}   engineered ret_mid_fut_{w}s (validation)
    """
    import pyarrow.parquet as pq
    from common.data_loader import discover_files
    from common.config import target_col, ML_FEATURES

    mid_col = f"mid_fut_1s_{asset}"
    tgt_cols = {hz: target_col(hz, asset) for hz in horizons}
    eng_cols = {w: f"ret_mid_fut_{w}s_{asset}"
                for w in windows if w in ENGINEERED_BWD_WINDOWS}

    files = discover_files(None, None, None)
    if not files:
        raise FileNotFoundError(f"No ml_features files in {ML_FEATURES}")

    rng = np.random.RandomState(seed)
    if max_hours and len(files) > max_hours:
        idx = rng.choice(len(files), max_hours, replace=False)
        idx.sort()
        files = [files[i] for i in idx]

    logger.info("Loading %d files for %s ONCE (mid + horizons %s + eng-bwd %s)",
                len(files), asset, list(horizons), list(eng_cols.values()))

    file_metas, total = [], 0
    for f in files:
        try:
            n = int(pq.read_metadata(f).num_rows)
            file_metas.append((f, n)); total += n
        except Exception as e:
            logger.warning("  meta read failed %s: %s", f.name, e)
    if total == 0:
        raise ValueError("No data: all parquet metadata reads failed.")

    logger.info("  Preallocating up to %s rows x %d features (~%.1f GB)",
                f"{total:,}", len(feat_names), total * len(feat_names) * 4 / 1e9)

    X = np.full((total, len(feat_names)), np.nan, dtype=np.float32)
    y_by_hz = {hz: np.full(total, np.nan, dtype=np.float64) for hz in horizons}
    eng_by_w = {w: np.full(total, np.nan, dtype=np.float64) for w in eng_cols}
    mid = np.full(total, np.nan, dtype=np.float64)
    ts_i8 = np.full(total, -1, dtype=np.int64)

    feat_index = {c: i for i, c in enumerate(feat_names)}
    off = n_proc = n_fail = ts_ok_files = 0
    for i, (f, _) in enumerate(file_metas):
        try:
            schema = set(pq.read_schema(f).names)
            load_feats = [c for c in feat_names if c in schema]
            present_tgts = {hz: c for hz, c in tgt_cols.items() if c in schema}
            present_eng = {w: c for w, c in eng_cols.items() if c in schema}
            extra = ([mid_col] if mid_col in schema else []) + \
                    ([TS_COL] if TS_COL in schema else [])
            cols = load_feats + list(present_tgts.values()) + \
                list(present_eng.values()) + extra
            df = pd.read_parquet(f, columns=cols)
            if df.index.name == TS_COL and not df.index.is_monotonic_increasing:
                df = df.sort_index()
            n = len(df); end = off + n

            col_idx = [feat_index[c] for c in load_feats]
            X[off:end][:, col_idx] = df[load_feats].to_numpy(dtype=np.float32)
            for hz, c in present_tgts.items():
                y_by_hz[hz][off:end] = df[c].to_numpy(dtype=np.float64)
            for w, c in present_eng.items():
                eng_by_w[w][off:end] = df[c].to_numpy(dtype=np.float64)
            if mid_col in df.columns:
                mid[off:end] = df[mid_col].to_numpy(dtype=np.float64)

            t = None
            if TS_COL in df.columns:
                t = pd.to_datetime(df[TS_COL], utc=True).astype("int64").to_numpy()
            elif df.index.name == TS_COL:
                idx = df.index
                if getattr(idx, "tz", None) is not None:
                    idx = idx.tz_convert("UTC")
                t = idx.astype("int64").to_numpy()
            if t is not None and len(t) == n:
                ts_i8[off:end] = t
                ts_ok_files += 1

            off = end; n_proc += 1
            if (i + 1) % 100 == 0:
                logger.info("    [%d/%d] written=%s failed=%d",
                            i + 1, len(file_metas), f"{off:,}", n_fail)
        except Exception as e:
            n_fail += 1
            logger.warning("  read failed %s: %s", f.name, e)

    X = X[:off]
    for hz in horizons:
        y_by_hz[hz] = y_by_hz[hz][:off]
    mid = mid[:off]
    ts_i8 = ts_i8[:off]
    for w in list(eng_by_w):
        b = eng_by_w[w][:off]
        finite = np.isfinite(b)
        if finite.any() and np.nanmean(np.abs(b[finite])) < 0.01:
            b = b * 1e4
        eng_by_w[w] = b

    if ts_ok_files == 0 or (ts_i8 <= 0).all():
        logger.warning("  %s column unavailable — using 1s row clock for embargo "
                       "and trailing-return lookup (bars are ~1s; gaps make the "
                       "embargo conservative, never look-ahead).", TS_COL)
        ts_i8 = np.arange(off, dtype=np.int64) * 1_000_000_000
    else:
        order = np.argsort(ts_i8, kind="mergesort")
        if not np.array_equal(order, np.arange(off)):
            logger.info("  Sorting %s rows to global chronological order.", f"{off:,}")
            X = X[order]
            for hz in horizons:
                y_by_hz[hz] = y_by_hz[hz][order]
            mid = mid[order]
            ts_i8 = ts_i8[order]
            for w in eng_by_w:
                eng_by_w[w] = eng_by_w[w][order]

    logger.info("  Loaded ONCE: X=%s, %d files, %d failed, ts cols ok=%d/%d, "
                "mid NaN=%d", X.shape, n_proc, n_fail, ts_ok_files, len(file_metas),
                int(np.isnan(mid).sum()))
    if np.isnan(mid).all():
        raise ValueError(f"mid column '{mid_col}' absent from all files.")
    return X, y_by_hz, mid, ts_i8, eng_by_w


# ==============================================================================
# LightGBM wrapper
# ==============================================================================

class LGBMWrapper:
    def __init__(self, params, val_frac=0.15, val_min=200, early_stopping=50):
        self.params = dict(params)
        self.val_frac = val_frac
        self.val_min = val_min
        self.early_stopping = early_stopping
        self.model_ = None

    def fit(self, X, y):
        n = len(X)
        n_val = max(int(n * self.val_frac), self.val_min)
        if n_val >= n:
            n_val = max(int(n * 0.2), 1)
        n_train = n - n_val
        X_fit, y_fit = X[:n_train], y[:n_train]
        X_val, y_val = X[n_train:], y[n_train:]
        p = dict(self.params)
        n_est = p.pop("n_estimators", 500)
        self.model_ = lgb.LGBMRegressor(n_estimators=n_est, **p)
        self.model_.fit(
            X_fit, y_fit,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(self.early_stopping, verbose=False),
                       lgb.log_evaluation(0)],
        )
        return self

    def predict(self, X):
        return self.model_.predict(X)


# ==============================================================================
# OOF training on a breakout subset (expanding window + time embargo)
# ==============================================================================

def train_oof_breakout(X_sub, y_sub_bps, ts_sub_i8, n_folds, seeds,
                       params, n_jobs, embargo_s):
    from common.cv_engine import expanding_window_folds

    n = len(X_sub)
    oof = np.full(n, np.nan)
    fold_metrics, boundaries = [], []
    embargo_ns = int(embargo_s) * 1_000_000_000

    folds = expanding_window_folds(n_samples=n, n_folds=n_folds)
    for fi, fold in enumerate(folds):
        tr = fold.train_idx
        te = fold.test_idx
        if len(te) == 0:
            continue
        first_test_ns = ts_sub_i8[te[0]]
        keep = tr[ts_sub_i8[tr] <= (first_test_ns - embargo_ns)]
        if len(keep) < MIN_TRAIN_ROWS or len(te) < MIN_TEST_ROWS:
            logger.info("    fold %d skipped (train=%d, test=%d after embargo)",
                        fi + 1, len(keep), len(te))
            continue

        t0 = time.time()
        X_tr, y_tr = X_sub[keep], y_sub_bps[keep]
        X_te, y_te = X_sub[te], y_sub_bps[te]

        fold_preds = []
        for s in seeds:
            p = {**params, "random_state": s, "n_jobs": n_jobs}
            m = LGBMWrapper(p)
            m.fit(X_tr, y_tr)
            fold_preds.append(m.predict(X_te))
            del m
        pm = np.mean(fold_preds, axis=0)
        oof[te] = pm
        boundaries.append((int(te[0]), int(te[-1]) + 1))

        ss_r = float(np.sum((y_te - pm) ** 2))
        ss_t = float(np.sum((y_te - y_te.mean()) ** 2))
        r2 = 1 - ss_r / (ss_t + 1e-12)
        da = float(np.mean(np.sign(pm) == np.sign(y_te)))
        fold_metrics.append({
            "fold": fi + 1, "train": len(keep), "test": len(te),
            "r2": round(r2, 6), "dir_acc": round(da, 4),
            "seconds": round(time.time() - t0, 1),
        })
        logger.info("    fold %d/%d  train=%d  test=%d  R2=%.4f  DA=%.3f  [%.0fs]",
                    fi + 1, len(folds), len(keep), len(te), r2, da, time.time() - t0)
        del fold_preds
        gc.collect()

    return oof, fold_metrics, boundaries


# ==============================================================================
# Backtest: one strategy over the breakout test population
# ==============================================================================

def backtest_strategy(strategy, pred, fwd_bps, bwd_bps, thresholds,
                      fee_bps, boundaries, rng_seed=42):
    rows = []
    thr_list = thresholds if strategy == "lgbm" else [np.nan]
    rng = np.random.RandomState(rng_seed)

    for thr in thr_list:
        for fi, (start, end) in enumerate(boundaries):
            f = fwd_bps[start:end]
            if strategy == "lgbm":
                p = pred[start:end]
                ok = ~(np.isnan(p) | np.isnan(f))
                p, f2 = p[ok], f[ok]
                sig = np.where(p > thr, 1, np.where(p < -thr, -1, 0))
            elif strategy == "momentum":
                b = bwd_bps[start:end]
                ok = ~(np.isnan(b) | np.isnan(f))
                b, f2 = b[ok], f[ok]
                sig = np.sign(b).astype(int)
            else:
                ok = ~np.isnan(f)
                f2 = f[ok]
                sig = rng.choice([-1, 1], size=len(f2))

            in_mkt = sig != 0
            n_tr = int(in_mkt.sum())
            if n_tr == 0:
                continue
            tpnl = sig[in_mkt] * f2[in_mkt] - fee_bps
            da = float(np.mean(sig[in_mkt] == np.sign(f2[in_mkt])))
            lm, sm = sig == 1, sig == -1
            long_pnl = float((f2[lm] - fee_bps).mean()) if lm.any() else np.nan
            short_pnl = float((-f2[sm] - fee_bps).mean()) if sm.any() else np.nan
            std = float(tpnl.std()) + 1e-12
            rows.append({
                "strategy": strategy,
                "fold": fi + 1,
                "decision_thr_bps": round(float(thr), 4) if not np.isnan(thr) else None,
                "n_bars": int(ok.sum()),
                "n_trades": n_tr,
                "trade_rate_pct": round(100 * n_tr / max(int(ok.sum()), 1), 1),
                "n_longs": int(lm.sum()),
                "n_shorts": int(sm.sum()),
                "pnl_per_trade_bps": round(float(tpnl.mean()), 4),
                "win_rate_pct": round(100 * float((tpnl > 0).mean()), 1),
                "dir_acc": round(da, 4),
                "long_pnl_bps": round(long_pnl, 4) if not np.isnan(long_pnl) else None,
                "short_pnl_bps": round(short_pnl, 4) if not np.isnan(short_pnl) else None,
                "sharpe": round(float(tpnl.mean()) / std * np.sqrt(n_tr), 3),
            })
    return rows


# ==============================================================================
# Main
# ==============================================================================

def main():
    pa = argparse.ArgumentParser(
        description="Breakout-conditioned directional LGBM backtest (train on breakouts only)")
    pa.add_argument("--assets", nargs="+", default=["btc", "eth"])
    pa.add_argument("--bo-windows", type=int, nargs="+", default=DEFAULT_BO_WINDOWS,
                    help="Backward windows in s (computed from mid; any value).")
    pa.add_argument("--bo-thresholds", type=float, nargs="+", default=DEFAULT_BO_THRESHOLDS)
    pa.add_argument("--horizons", nargs="+", default=DEFAULT_HORIZONS,
                    help="Forward holding horizons (separate from breakout window).")
    pa.add_argument("--decision-thresholds", type=float, nargs="+",
                    default=[0.0, 0.5, 1.0, 2.0],
                    help="Fixed |pred| gates in bps; data-driven P50/P75/P90 added per combo.")
    pa.add_argument("--profile", choices=["tree", "linear", "cluster", "anomaly", "all"],
                    default="tree",
                    help="Feature usage profile -> use_{profile} column (LGBM uses tree).")
    pa.add_argument("--top-only", choices=["none", "returns", "mfe", "mae",
                                           "short_horizon", "long_horizon"],
                    default="none",
                    help="Optionally restrict to the pre-ranked top_{x} features "
                         "(overfitting mitigation on the small breakout subset; "
                         "'returns' is the natural choice for direction).")
    pa.add_argument("--n-folds", type=int, default=5)
    pa.add_argument("--seeds", type=int, nargs="+", default=None)
    pa.add_argument("--n-jobs", type=int, default=8)
    pa.add_argument("--max-hours", type=int, default=None)
    pa.add_argument("--maker", action="store_true")
    pa.add_argument("--log-level", default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = pa.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s  %(levelname)-8s  %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout)

    from common.config import RESULTS_DIR, DEFAULT_SEEDS

    windows = sorted(set(args.bo_windows))
    seeds = tuple(args.seeds) if args.seeds else tuple(DEFAULT_SEEDS)
    fee = MAKER_FEE_BPS if args.maker else TAKER_FEE_BPS
    feat_names = resolve_feature_columns(args.profile, args.top_only)

    feature_tag = args.profile if args.top_only == "none" else f"{args.profile}-top_{args.top_only}"
    out_dir = RESULTS_DIR / "breakout_directional" / feature_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Seeds=%s  fee=%.1f bps (%s)  breakout grid: windows=%s x thr=%s bps",
                seeds, fee, "maker" if args.maker else "taker", windows, args.bo_thresholds)
    logger.info("Horizons=%s  profile=%s top_only=%s (%d cols)  folds=%d",
                args.horizons, args.profile, args.top_only, len(feat_names), args.n_folds)

    all_summaries = []

    for asset in args.assets:
        est_rows = (args.max_hours * 3600) if args.max_hours else 7_020_000
        if not check_ram_or_skip(est_rows, n_features=len(feat_names)):
            continue

        try:
            X, y_by_hz, mid, ts_i8, eng_by_w = load_features_targets_bwd(
                asset, args.horizons, windows, feat_names,
                max_hours=args.max_hours, seed=42)
        except Exception as e:
            logger.error("  load failed for %s: %s", asset, e)
            continue

        if len(ts_i8) > 2:
            d = np.diff(ts_i8[: min(500_000, len(ts_i8))]) / 1e9
            d = d[d >= 0]
            if len(d):
                logger.info("  bar spacing (s): median=%.3f p90=%.3f (expect ~1.0)",
                            float(np.median(d)), float(np.percentile(d, 90)))

        # Trailing returns for ALL windows (from mid), validated vs engineered.
        bwd_by_w = {w: backward_return_bps(mid, ts_i8, w) for w in windows}
        for w in windows:
            b = bwd_by_w[w]
            fin = np.isfinite(b)
            if fin.any():
                ab = np.abs(b[fin])
                logger.info("  bwd w=%ds: finite=%d/%d  |bwd| bps P50=%.2f P90=%.2f "
                            "P99=%.2f max=%.2f", w, int(fin.sum()), len(b),
                            float(np.percentile(ab, 50)), float(np.percentile(ab, 90)),
                            float(np.percentile(ab, 99)), float(ab.max()))
            else:
                logger.error("  bwd w=%ds: ZERO finite — mid/clock problem.", w)
            if w in eng_by_w:
                e = eng_by_w[w]
                m = np.isfinite(b) & np.isfinite(e)
                if int(m.sum()) > 1000:
                    corr = float(np.corrcoef(b[m], e[m])[0, 1])
                    mad = float(np.mean(np.abs(b[m] - e[m])))
                    logger.info("    validation w=%ds vs ret_mid_fut: corr=%.4f "
                                "MAD=%.3f bps (n=%d)", w, corr, mad, int(m.sum()))

        del eng_by_w

        union = np.zeros(len(ts_i8), dtype=bool)
        for w in windows:
            for thr in args.bo_thresholds:
                union |= (np.abs(bwd_by_w[w]) > thr) & np.isfinite(bwd_by_w[w])
        n_union = int(union.sum())
        logger.info("  %s: %d breakout bars (any combo) of %d (%.3f%%) — "
                    "subsetting and releasing full matrix.",
                    asset, n_union, len(ts_i8), 100 * n_union / max(len(ts_i8), 1))
        if n_union == 0:
            logger.error("  No breakouts for any combo.")
            del X, y_by_hz, mid, ts_i8, bwd_by_w
            gc.collect()
            continue

        X_bo = X[union]
        ts_bo = ts_i8[union]
        y_bo = {hz: y_by_hz[hz][union] for hz in args.horizons}
        bwd_bo = {w: bwd_by_w[w][union] for w in windows}
        del X, y_by_hz, mid, ts_i8, bwd_by_w
        gc.collect()
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass

        for w in windows:
            for bo_thr in args.bo_thresholds:
                combo_csv = out_dir / f"breakout_dir_{asset}_w{w}_thr{int(bo_thr)}bps.csv"
                if combo_csv.exists():
                    logger.info("== %s w=%ds thr=%.0fbps — SKIP (%s exists)",
                                asset.upper(), w, bo_thr, combo_csv.name)
                    try:
                        all_summaries.extend(pd.read_csv(combo_csv).to_dict("records"))
                    except Exception:
                        pass
                    continue

                bo_mask = (np.abs(bwd_bo[w]) > bo_thr) & np.isfinite(bwd_bo[w])
                logger.info("== %s w=%ds thr=%.0fbps  breakouts=%d",
                            asset.upper(), w, bo_thr, int(bo_mask.sum()))

                combo_rows = []
                for hz in args.horizons:
                    hsec = hz_seconds(hz)
                    y_full = y_bo[hz]
                    valid = bo_mask & np.isfinite(y_full)
                    pos = np.where(valid)[0]
                    if len(pos) < MIN_BREAKOUTS:
                        logger.info("   %s: only %d breakouts with valid %s target — skip",
                                    hz, len(pos), hz)
                        continue

                    X_pop = X_bo[pos]
                    ts_pop = ts_bo[pos]
                    bwd_pop = bwd_bo[w][pos]
                    y_raw = y_full[pos]
                    y_pop = y_raw * 1e4 if np.nanmean(np.abs(y_raw)) < 0.01 else y_raw.copy()

                    logger.info("   %s: train OOF on %d breakouts (%d feats, embargo=%ds)",
                                hz, len(pos), X_pop.shape[1], max(w, hsec))

                    oof, fold_metrics, boundaries = train_oof_breakout(
                        X_pop, y_pop, ts_pop, n_folds=args.n_folds, seeds=list(seeds),
                        params=BREAKOUT_LGBM_PARAMS, n_jobs=args.n_jobs,
                        embargo_s=max(w, hsec))

                    if not boundaries:
                        logger.warning("   %s: no usable folds — skip", hz)
                        continue

                    pd.DataFrame(fold_metrics).to_csv(
                        out_dir / f"breakout_dir_folds_{asset}_w{w}_thr{int(bo_thr)}bps_{hz}.csv",
                        index=False)

                    oof_test = oof[~np.isnan(oof)]
                    pcts = ([round(float(np.percentile(np.abs(oof_test), q)), 4)
                             for q in (50, 75, 90)] if len(oof_test) else [])
                    dthr = sorted(set([float(t) for t in args.decision_thresholds] + pcts))
                    if pcts:
                        logger.info("   %s: |pred| bps P50=%.3f P75=%.3f P90=%.3f  thr=%s",
                                    hz, pcts[0], pcts[1], pcts[2], dthr)

                    strat_rows = []
                    strat_rows += backtest_strategy("lgbm", oof, y_pop, bwd_pop,
                                                    dthr, fee, boundaries)
                    strat_rows += backtest_strategy("momentum", oof, y_pop, bwd_pop,
                                                    dthr, fee, boundaries)
                    strat_rows += backtest_strategy("random", oof, y_pop, bwd_pop,
                                                    dthr, fee, boundaries, rng_seed=42)

                    for r in strat_rows:
                        r.update({"asset": asset, "bo_window_s": w,
                                  "bo_threshold_bps": bo_thr, "horizon": hz,
                                  "n_breakouts": len(pos)})
                        combo_rows.append(r)

                if not combo_rows:
                    logger.info("   no horizons produced results for this combo")
                    continue

                cdf = pd.DataFrame(combo_rows)
                grp = (cdf.groupby(["asset", "bo_window_s", "bo_threshold_bps",
                                    "horizon", "strategy", "decision_thr_bps"],
                                   dropna=False)
                          .agg(n_breakouts=("n_breakouts", "first"),
                               mean_n_trades=("n_trades", "mean"),
                               mean_pnl_per_trade=("pnl_per_trade_bps", "mean"),
                               mean_win_rate_pct=("win_rate_pct", "mean"),
                               mean_dir_acc=("dir_acc", "mean"),
                               mean_sharpe=("sharpe", "mean"),
                               mean_long_pnl=("long_pnl_bps", "mean"),
                               mean_short_pnl=("short_pnl_bps", "mean"),
                               n_folds=("fold", "nunique"))
                          .reset_index())
                grp["mean_n_trades"] = grp["mean_n_trades"].round(0).astype(int)
                for c in ["mean_pnl_per_trade", "mean_dir_acc", "mean_sharpe",
                          "mean_long_pnl", "mean_short_pnl"]:
                    grp[c] = grp[c].round(4)
                grp["mean_win_rate_pct"] = grp["mean_win_rate_pct"].round(1)
                grp["profitable"] = grp["mean_pnl_per_trade"] > 0

                grp.to_csv(combo_csv, index=False)
                logger.info("   saved combo -> %s (%d rows)", combo_csv.name, len(grp))
                all_summaries.extend(grp.to_dict("records"))

        del X_bo, ts_bo, y_bo, bwd_bo
        gc.collect()
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass

    if not all_summaries:
        logger.info("No results produced.")
        return

    summary = pd.DataFrame(all_summaries)

    allbar = RESULTS_DIR / "directional" / "directional_summary.csv"
    if allbar.exists():
        try:
            ab = pd.read_csv(allbar)
            ab0 = (ab.sort_values("threshold_bps")
                     .groupby(["asset", "horizon"], as_index=False)
                     .first()[["asset", "horizon", "mean_pnl_per_trade"]]
                     .rename(columns={"mean_pnl_per_trade": "allbar_pnl_ref"}))
            summary = summary.merge(ab0, on=["asset", "horizon"], how="left")
            logger.info("Attached all-bar reference PnL from %s", allbar.name)
        except Exception as e:
            logger.warning("Could not attach all-bar reference: %s", e)

    sp = out_dir / "breakout_directional_summary.csv"
    summary.to_csv(sp, index=False)
    logger.info("Saved global summary -> %s (%d rows)", sp, len(summary))

    try:
        lg = summary[summary["strategy"] == "lgbm"].copy()
        mo = (summary[summary["strategy"] == "momentum"]
              [["asset", "bo_window_s", "bo_threshold_bps", "horizon",
                "mean_pnl_per_trade"]]
              .rename(columns={"mean_pnl_per_trade": "momentum_pnl"}))
        lg = lg.merge(mo, on=["asset", "bo_window_s", "bo_threshold_bps", "horizon"],
                      how="left")
        winners = lg[(lg["mean_pnl_per_trade"] > 0) &
                     (lg["mean_pnl_per_trade"] > lg["momentum_pnl"])]
        logger.info("\n%s\n  LGBM configs that clear fees AND beat momentum null\n%s",
                    "=" * 64, "=" * 64)
        if winners.empty:
            logger.info("  None. Direct direction prediction on breakouts does not "
                        "beat the naive null — motivates clustering-by-state.")
        else:
            cols = ["asset", "bo_window_s", "bo_threshold_bps", "horizon",
                    "decision_thr_bps", "mean_pnl_per_trade", "momentum_pnl",
                    "mean_dir_acc", "mean_n_trades"]
            logger.info("\n%s", winners.sort_values("mean_pnl_per_trade",
                        ascending=False)[cols].to_string(index=False))
    except Exception as e:
        logger.warning("Headline comparison failed: %s", e)

    logger.info("Done. Outputs in %s", out_dir)


if __name__ == "__main__":
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        os.setsid()
    except OSError:
        pass
    main()