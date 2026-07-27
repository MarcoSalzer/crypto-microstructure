# prediction/directional_backtest.py
# ==============================================================================
# Directional (Long/Short) Backtest — no breakout filter
# ==============================================================================
# Uses the same server infrastructure as lgbm_pipeline.py:
#   data_loader.load_dataset  — consistent feature set + target selection
#   cv_engine.expanding_window_folds — identical fold structure
#   config.LGBM_PARAMS / DEFAULT_SEEDS / RESULTS_DIR
#   LGBMWrapper — correct early-stopping pattern
#
# What this adds on top of lgbm_pipeline.py:
#   - Saves out-of-fold predictions per bar (not just fold metrics)
#   - Runs threshold-based long/short backtest on those predictions
#   - Reports P&L, win rate, long vs short breakdown per threshold per fold
#
# Every bar is a potential trade — no event or breakout filter.
# Signal: pred > +thr → LONG, pred < −thr → SHORT, else FLAT.
# P&L per trade = direction × actual_return_bps − taker_fee_bps
#
# Trading cost: round-trip Binance taker fee = 10.0 bps, HARDCODED.
#   (Standard tier taker 0.05% one-way → 5.0 bps → round-trip 10.0 bps.)
#   This is a thesis-specific throwaway script for BTC/ETH only; no need to
#   parameterise the fee. Matches config.SPREAD_BPS.
#
# Default horizons: all 8 return horizons (1s..900s) for the complete
# microstructure-decay / fee-pivot picture.
#
# Checkpointing: each (asset, horizon) writes its outputs as soon as it
# finishes, and is SKIPPED on rerun if its parquet already exists. The run
# is therefore restartable — if it dies after 20h, just relaunch and it
# resumes with the remaining horizons. Horizons run cost-DESCENDING (1s
# first) so the most important / most expensive block lands first.
#
# Estimated runtime: ~16h per asset (all 8 horizons, 3 seeds, 5 folds),
# ~30-32h total for btc+eth. Dominated by LGBM training; the backtest
# sweep itself is seconds.
#
# Usage:
#   python prediction/directional_backtest.py \
#       --assets btc eth \
#       --thresholds 0.0 0.5 1.0 2.0 3.0 5.0 \
#       --seeds 42 123 999 --n-folds 5 --n-jobs 8
#
# Outputs (in RESULTS_DIR/directional/):
#   directional_oof_{asset}_{hz}.parquet     OOF predictions + actuals (bps)
#   directional_backtest_{asset}_{hz}.csv    P&L per threshold per fold
#   directional_summary.csv                  Top-level summary all configs
# ==============================================================================
from __future__ import annotations
import os, signal

import argparse, gc, logging, sys, time
from pathlib import Path

import ctypes
import numpy as np
import pandas as pd
import lightgbm as lgb

logger = logging.getLogger(__name__)

# Round-trip Binance taker fee, hardcoded for the thesis (BTC/ETH only).
# Matches config.SPREAD_BPS["btc"|"eth"]["fut"] = 10.0.
TAKER_FEE_BPS = 10.0

# All 8 forward-return horizons, cost-descending (1s is the most expensive
# block and carries the strongest signal → run first).
DEFAULT_HORIZONS = ["1s", "5s", "15s", "30s", "60s", "120s", "300s", "900s"]

# This script uses the 'tree' feature profile = 3349 features (LightGBM is not
# multicollinearity-sensitive), so the matrix is larger than the cluster
# profile: 7.02M × 3349 × 4 bytes ≈ 94 GB.
TREE_N_FEATURES = 3349


def check_ram_or_skip(n_rows: int, n_features: int = TREE_N_FEATURES,
                      safety_factor: float = 1.6,
                      min_headroom_gb: float = 10.0) -> bool:
    """
    Decide whether it is safe to load/train on a matrix of this size before the
    data_loader's big preallocation. The raw matrix is n_rows × n_features × 4
    bytes; LightGBM holds additional transient structures during fit, captured
    by safety_factor, plus fixed headroom so we never push the host into swap.

    Returns True if safe to proceed, False if the (asset, horizon) should be
    skipped. Falls back gracefully if memory cannot be queried.
    """
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
        logger.warning("  RAM check unavailable — proceeding (matrix ≈ %.0f GB).", raw_gb)
        return True
    logger.info("  RAM check: matrix≈%.0f GB, need≈%.0f GB (×%.1f + %.0f GB), "
                "available≈%.0f GB", raw_gb, need_gb, safety_factor,
                min_headroom_gb, avail_gb)
    if avail_gb < need_gb:
        logger.error("  INSUFFICIENT RAM — skipping %d-row job. Free memory or "
                     "use --max-hours to shrink the matrix. (need≈%.0f, have≈%.0f)",
                     n_rows, need_gb, avail_gb)
        return False
    return True


def load_all_horizons(asset: str, horizons, max_hours=None, seed: int = 42):
    """
    Load the feature matrix X ONCE together with the target column of every
    requested horizon, in a single pass over the parquet files.

    Rationale: X (the 3349 'tree' features) is identical across all return
    horizons; only the target column differs. The shared data_loader reloads
    the full matrix per horizon AND drops rows whose target is NaN — so the
    row set differs per horizon and the matrices cannot be reused naively.
    This function instead keeps every row (no target-based dropping) and
    returns all horizon targets aligned to the same rows, so the caller can
    mask per-horizon NaNs itself. One load instead of eight.

    Returns:
      X         : float32 [n_rows, n_features]
      y_by_hz   : dict {horizon: float64 array [n_rows]}  (NaNs preserved)
      feat_names: list[str]
    """
    import pyarrow.parquet as pq
    from common.data_loader import discover_files, get_feature_columns
    from common.config import target_col, ML_FEATURES

    # get_feature_columns loads the keep-list itself when keep_df is None.
    feat_names = get_feature_columns(None, profile="tree")
    tgt_cols = {hz: target_col(hz, asset) for hz in horizons}

    files = discover_files(None, None, None)
    if not files:
        raise FileNotFoundError(f"No ml_features files in {ML_FEATURES}")

    rng = np.random.RandomState(seed)
    if max_hours and len(files) > max_hours:
        idx = rng.choice(len(files), max_hours, replace=False)
        idx.sort()
        files = [files[i] for i in idx]

    logger.info("Loading %d files for %s ONCE (all horizons: %s)",
                len(files), asset, list(horizons))

    # Phase 1: sum rows
    file_metas, total = [], 0
    for f in files:
        try:
            n = int(pq.read_metadata(f).num_rows)
            file_metas.append((f, n)); total += n
        except Exception as e:
            logger.warning("  meta read failed %s: %s", f.name, e)
    if total == 0:
        raise ValueError("No data: all parquet metadata reads failed.")

    x_gb = total * len(feat_names) * 4 / 1e9
    logger.info("  Preallocating: up to %s rows × %d features → X≈%.1f GB",
                f"{total:,}", len(feat_names), x_gb)

    # Phase 2: preallocate X + one y per horizon
    X = np.full((total, len(feat_names)), np.nan, dtype=np.float32)
    y_by_hz = {hz: np.full(total, np.nan, dtype=np.float64) for hz in horizons}

    # Phase 3: fill. Keep ALL rows (no target-NaN dropping) so every horizon
    # stays aligned to the same row index.
    feat_set = set(feat_names)
    off = 0
    n_proc = n_fail = 0
    for i, (f, _) in enumerate(file_metas):
        try:
            schema = set(pq.read_schema(f).names)
            load_cols = [c for c in feat_names if c in schema]
            present_tgts = {hz: c for hz, c in tgt_cols.items() if c in schema}
            all_cols = load_cols + list(present_tgts.values())
            df = pd.read_parquet(f, columns=all_cols)
            if df.index.name == "bucket_dt_utc" and not df.index.is_monotonic_increasing:
                df = df.sort_index()
            n = len(df)
            end = off + n
            # features (missing cols stay NaN via column alignment)
            feat_present = [c for c in feat_names if c in df.columns]
            col_idx = [feat_names.index(c) for c in feat_present]
            X[off:end][:, col_idx] = df[feat_present].to_numpy(dtype=np.float32)
            # targets
            for hz, c in present_tgts.items():
                y_by_hz[hz][off:end] = df[c].to_numpy(dtype=np.float64)
            off = end
            n_proc += 1
            if (i + 1) % 100 == 0:
                logger.info("    [%d/%d]  written=%s  failed=%d",
                            i + 1, len(file_metas), f"{off:,}", n_fail)
        except Exception as e:
            n_fail += 1
            logger.warning("  read failed %s: %s", f.name, e)

    # Trim to actually-written rows
    X = X[:off]
    for hz in horizons:
        y_by_hz[hz] = y_by_hz[hz][:off]
    logger.info("  Loaded ONCE: X=%s, %d files, %d failed", X.shape, n_proc, n_fail)
    return X, y_by_hz, feat_names


# ── Reuse LGBMWrapper from lgbm_pipeline.py (identical) ──────────────────────

class LGBMWrapper:
    """
    Sklearn-compatible wrapper around LGBMRegressor.
    Splits last `val_frac` of training data for early stopping.
    """
    def __init__(self, params: dict, val_frac: float = 0.10,
                 early_stopping: int = 50):
        self.params = dict(params)
        self.val_frac = val_frac
        self.early_stopping = early_stopping
        self.model_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        n = len(X)
        n_val   = max(int(n * self.val_frac), 1000)
        n_train = n - n_val
        X_fit, y_fit = X[:n_train], y[:n_train]
        X_val,  y_val  = X[n_train:], y[n_train:]

        p = dict(self.params)
        n_est = p.pop("n_estimators", 1000)
        self.model_ = lgb.LGBMRegressor(n_estimators=n_est, **p)
        self.model_.fit(
            X_fit, y_fit,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=self.early_stopping),
                lgb.log_evaluation(period=0),
            ],
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict(X)

# ── OOF training (our own loop — run_cv doesn't save predictions) ─────────────

def train_and_collect_oof(
    X: np.ndarray,
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    seeds: list[int],
    lgbm_params: dict,
    n_jobs: int,
) -> tuple[np.ndarray, list[dict], list[tuple[int, int]]]:
    """
    Expanding-window OOF: train on each fold, collect predictions.
    Returns (oof_preds_bps, fold_metrics, test_boundaries).
    oof_preds_bps[i] is NaN for bars not in any test set (fold 0 train rows).
    """
    n = len(X)
    oof_preds = np.full(n, np.nan)
    fold_metrics = []
    test_boundaries = []

    params = {**lgbm_params, "n_jobs": n_jobs}

    for fold_idx, fold in enumerate(folds):
        # Support both plain (train, test) tuples and FoldSplit namedtuple/dataclass
        if isinstance(fold, tuple):
            train_idx, test_idx = fold
        elif hasattr(fold, "train_idx"):
            train_idx, test_idx = fold.train_idx, fold.test_idx
        elif hasattr(fold, "train"):
            train_idx, test_idx = fold.train, fold.test
        else:
            raise TypeError(f"Unrecognised fold type: {type(fold)}  attrs={dir(fold)}")
        t0 = time.time()
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_te, y_te = X[test_idx],  y[test_idx]

        test_boundaries.append((int(test_idx[0]), int(test_idx[-1]) + 1))

        # Average over seeds
        fold_preds = []
        for seed in seeds:
            p = {**params, "random_state": seed}
            model = LGBMWrapper(p)
            model.fit(X_tr, y_tr)
            fold_preds.append(model.predict(X_te))
            del model

        pred_mean = np.mean(fold_preds, axis=0)
        oof_preds[test_idx] = pred_mean

        # Metrics
        valid = ~np.isnan(y_te)
        ss_r = np.sum((y_te[valid] - pred_mean[valid]) ** 2)
        ss_t = np.sum((y_te[valid] - y_te[valid].mean()) ** 2)
        r2   = float(1 - ss_r / (ss_t + 1e-12))
        da   = float(np.mean(np.sign(pred_mean[valid]) == np.sign(y_te[valid])))

        fold_metrics.append({
            "fold":       fold_idx + 1,
            "train_size": len(train_idx),
            "test_size":  len(test_idx),
            "r2":         round(r2, 6),
            "dir_acc":    round(da, 4),
            "seconds":    round(time.time() - t0, 1),
        })
        logger.info(
            "    Fold %d/%d  train=%d  test=%d  R²=%.4f  DA=%.3f  [%.0fs]",
            fold_idx + 1, len(folds),
            len(train_idx), len(test_idx), r2, da,
            time.time() - t0,
        )

        del fold_preds
        gc.collect()

    return oof_preds, fold_metrics, test_boundaries

# ── Directional backtest ──────────────────────────────────────────────────────

def directional_backtest(
    y_pred_bps: np.ndarray,
    y_true_bps: np.ndarray,
    thresholds: list[float],
    taker_fee_bps: float,
    test_boundaries: list[tuple[int, int]],
) -> pd.DataFrame:
    """
    For each threshold and each test fold:
      pred > +thr  → LONG  (+1)
      pred < -thr  → SHORT (−1)
      else         → FLAT  ( 0)

    P&L per trade = direction × actual_bps − taker_fee_bps
    """
    rows = []

    for thr in thresholds:
        for fold_idx, (start, end) in enumerate(test_boundaries):
            p = y_pred_bps[start:end]
            a = y_true_bps[start:end]

            valid = ~(np.isnan(p) | np.isnan(a))
            p, a  = p[valid], a[valid]
            if len(p) == 0:
                continue

            signal   = np.where(p > thr, 1, np.where(p < -thr, -1, 0))
            in_mkt   = signal != 0
            n_trades = int(in_mkt.sum())
            n_longs  = int((signal == 1).sum())
            n_shorts = int((signal == -1).sum())

            if n_trades == 0:
                pnl_per  = win_rate = total_pnl = sharpe = 0.0
                long_pnl = short_pnl = np.nan
            else:
                trade_pnl = signal[in_mkt] * a[in_mkt] - taker_fee_bps
                pnl_per   = float(trade_pnl.mean())
                win_rate  = float((trade_pnl > 0).mean())
                total_pnl = float(trade_pnl.sum())
                std       = float(trade_pnl.std()) + 1e-12
                sharpe    = pnl_per / std * np.sqrt(n_trades)

                lm = signal ==  1
                sm = signal == -1
                long_pnl  = float((a[lm]  - taker_fee_bps).mean())  if lm.any()  else np.nan
                short_pnl = float((-a[sm] - taker_fee_bps).mean())  if sm.any()  else np.nan

            rows.append({
                "fold":              fold_idx + 1,
                "threshold_bps":     thr,
                "n_bars":            len(p),
                "n_trades":          n_trades,
                "trade_rate_pct":    round(100 * n_trades / max(len(p), 1), 1),
                "n_longs":           n_longs,
                "n_shorts":          n_shorts,
                "pnl_per_trade_bps": round(pnl_per,   4),
                "total_pnl_bps":     round(total_pnl, 2),
                "win_rate_pct":      round(100 * win_rate, 1),
                "long_pnl_bps":      round(long_pnl,  4) if not np.isnan(long_pnl)  else None,
                "short_pnl_bps":     round(short_pnl, 4) if not np.isnan(short_pnl) else None,
                "sharpe":            round(sharpe,     3),
            })

    return pd.DataFrame(rows)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # ── CLI ──────────────────────────────────────────────────────────────────
    p = argparse.ArgumentParser(description="Directional L/S backtest — no breakout filter")
    p.add_argument("--assets",        nargs="+", default=["btc", "eth"])
    p.add_argument("--horizons",      nargs="+", default=DEFAULT_HORIZONS,
                   help="Return horizons to run (default: all 8, 1s..900s, "
                        "cost-descending).")
    p.add_argument("--thresholds",    nargs="+", type=float,
                   default=[0.0, 0.5, 1.0, 2.0, 3.0, 5.0],
                   help="Signal thresholds in bps (absolute value of prediction)")
    p.add_argument("--n-folds",       type=int, default=5)
    p.add_argument("--seeds",         type=int, nargs="+", default=None,
                   help="Random seeds (default: config.DEFAULT_SEEDS)")
    p.add_argument("--n-jobs",        type=int, default=8)
    p.add_argument("--max-hours",     type=int, default=None)
    p.add_argument("--log-level",     default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = p.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    # ── Server infrastructure imports ─────────────────────────────────────────
    from common.data_loader import load_dataset
    from common.cv_engine import expanding_window_folds
    from common.config import LGBM_PARAMS, RESULTS_DIR, DEFAULT_SEEDS

    seeds = tuple(args.seeds) if args.seeds else tuple(DEFAULT_SEEDS)
    logger.info("Seeds: %s  |  Taker fee: %.1f bps (hardcoded)  |  Thresholds: %s",
                seeds, TAKER_FEE_BPS, args.thresholds)

    out_dir = RESULTS_DIR / "directional"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = []

    for asset in args.assets:
        # Determine which horizons still need computing (checkpoint-aware), so
        # we can skip the expensive single load entirely if the asset is done.
        todo = []
        for hz in args.horizons:
            oof_path = out_dir / f"directional_oof_{asset}_{hz}.parquet"
            if oof_path.exists():
                logger.info("━━ %s / %s — SKIP (%s exists)",
                            asset.upper(), hz, oof_path.name)
            else:
                todo.append(hz)
        if not todo:
            logger.info("━━ %s — all horizons already done.", asset.upper())
            continue

        # RAM guard before the single big load (one matrix for ALL horizons).
        est_rows = (args.max_hours * 3600) if args.max_hours else 7_020_000
        if not check_ram_or_skip(est_rows):
            continue

        # ── Load X ONCE for this asset, with every still-needed horizon target.
        try:
            X, y_by_hz, feat_names = load_all_horizons(
                asset, todo, max_hours=args.max_hours, seed=42)
        except Exception as e:
            logger.error("  load_all_horizons failed for %s: %s", asset, e)
            continue

        for hz in todo:
            logger.info("━━ Directional backtest  %s / %s ━━", asset.upper(), hz)
            oof_path = out_dir / f"directional_oof_{asset}_{hz}.parquet"

            # Per-horizon NaN target mask: a forward return is undefined where
            # the future window runs off the end of the data. Drop those rows
            # for THIS horizon only; X stays shared, we index a view.
            y_full = y_by_hz[hz]
            valid = ~np.isnan(y_full)
            n_valid = int(valid.sum())
            if n_valid < 50_000:
                logger.warning("  Only %d valid rows for %s/%s — skipping.",
                               n_valid, asset, hz)
                continue
            X_hz = X[valid]
            y = y_full[valid]
            logger.info("  Rows for %s: %d valid of %d", hz, n_valid, len(y_full))

            # y from data_loader is in return fraction units → convert to bps
            if np.nanmean(np.abs(y)) < 0.01:
                y_bps = y * 10_000
                logger.info("  Converted returns to bps (×10000)")
            else:
                y_bps = y.copy()
                logger.info("  Returns appear already in bps")

            logger.info("  Return stats (bps): mean=%.4f  std=%.4f  |p99|=%.4f",
                        np.nanmean(y_bps), np.nanstd(y_bps),
                        np.nanpercentile(np.abs(y_bps), 99))

            # ── Build folds (identical to lgbm_pipeline) ──────────────────────
            folds = expanding_window_folds(n_samples=len(X_hz), n_folds=args.n_folds)

            # ── Train OOF and collect predictions ─────────────────────────────
            logger.info("  Training %d-fold OOF  (%d seeds × %d features)...",
                        args.n_folds, len(seeds), X_hz.shape[1])

            oof_preds, fold_metrics, boundaries = train_and_collect_oof(
                X=X_hz,
                y=y_bps,
                folds=folds,
                seeds=list(seeds),
                lgbm_params=LGBM_PARAMS,
                n_jobs=args.n_jobs,
            )

            # Save OOF fold metrics (same format as lgbm_pipeline fold tables)
            fold_df = pd.DataFrame(fold_metrics)
            fold_path = out_dir / f"directional_oof_folds_{asset}_{hz}.csv"
            fold_df.to_csv(fold_path, index=False)

            # ── Directional backtest ───────────────────────────────────────────
            bt_df = directional_backtest(
                y_pred_bps=oof_preds,
                y_true_bps=y_bps,
                thresholds=args.thresholds,
                taker_fee_bps=TAKER_FEE_BPS,
                test_boundaries=boundaries,
            )
            bt_df.insert(0, "asset",   asset)
            bt_df.insert(1, "horizon", hz)

            bt_path = out_dir / f"directional_backtest_{asset}_{hz}.csv"
            bt_df.to_csv(bt_path, index=False)
            logger.info("  Saved backtest → %s", bt_path)

            # Save OOF predictions as parquet LAST — its existence is the
            # checkpoint marker for skip-if-exists, so it must only appear
            # once everything for this (asset, horizon) is fully written.
            oof_df = pd.DataFrame({
                "y_true_bps": y_bps,
                "y_pred_bps": oof_preds,
            })
            oof_df.to_parquet(oof_path, index=False)
            logger.info("  Saved OOF predictions → %s", oof_path)

            # ── Terminal summary ───────────────────────────────────────────────
            logger.info(
                "  %-8s  %-9s  %-8s  %-13s  %-10s  %-10s",
                "thr_bps", "n_trades", "win_rt%", "pnl/trade_bps", "long_bps", "short_bps",
            )
            for thr in args.thresholds:
                rows = bt_df[bt_df["threshold_bps"] == thr]
                if rows.empty:
                    continue
                logger.info(
                    "  %-8.1f  %-9.0f  %-8.1f  %-13.4f  %-10.4f  %-10.4f",
                    thr,
                    rows["n_trades"].mean(),
                    rows["win_rate_pct"].mean(),
                    rows["pnl_per_trade_bps"].mean(),
                    rows["long_pnl_bps"].mean(),
                    rows["short_pnl_bps"].mean(),
                )

            # Collect summary rows
            for thr in args.thresholds:
                rows = bt_df[bt_df["threshold_bps"] == thr]
                if rows.empty:
                    continue
                all_summaries.append({
                    "asset":              asset,
                    "horizon":            hz,
                    "threshold_bps":      thr,
                    "mean_pnl_per_trade": round(rows["pnl_per_trade_bps"].mean(), 4),
                    "mean_win_rate_pct":  round(rows["win_rate_pct"].mean(), 1),
                    "mean_n_trades":      int(rows["n_trades"].mean()),
                    "mean_sharpe":        round(rows["sharpe"].mean(), 3),
                    "mean_long_pnl":      round(rows["long_pnl_bps"].mean(), 4),
                    "mean_short_pnl":     round(rows["short_pnl_bps"].mean(), 4),
                    "profitable":         rows["pnl_per_trade_bps"].mean() > 0,
                })

            # ── Per-horizon cleanup. X and y_by_hz are SHARED across horizons
            #    for this asset, so do NOT delete them here — only the
            #    horizon-local arrays.
            del X_hz, y, y_bps, oof_preds, oof_df, bt_df
            gc.collect()
            try:
                ctypes.CDLL("libc.so.6").malloc_trim(0)
            except Exception:
                pass

        # ── Per-asset cleanup: free the shared matrix before the next asset ──
        del X, y_by_hz
        gc.collect()
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass

    # ── Global summary ────────────────────────────────────────────────────────
    if all_summaries:
        summary_df = pd.DataFrame(all_summaries)
        summary_path = out_dir / "directional_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        logger.info("Saved global summary → %s", summary_path)

        logger.info("\n%s\n  Profitable configs (pnl > 0)\n%s",
                    "=" * 60, "=" * 60)
        prof = summary_df[summary_df["profitable"]].sort_values(
            "mean_pnl_per_trade", ascending=False
        )
        if prof.empty:
            logger.info("  None found — signal may not clear %.1f bps fee.", TAKER_FEE_BPS)
            logger.info("  Consider: lower thresholds, maker-fee assumption (4.0 bps), or 1s horizon.")
        else:
            logger.info(
                "\n%s",
                prof[[
                    "asset", "horizon", "threshold_bps",
                    "mean_pnl_per_trade", "mean_win_rate_pct",
                    "mean_n_trades", "mean_sharpe",
                ]].to_string(index=False),
            )

    logger.info("Done. All outputs in %s", out_dir)


if __name__ == "__main__":
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        os.setsid()
    except OSError:
        pass
    main()