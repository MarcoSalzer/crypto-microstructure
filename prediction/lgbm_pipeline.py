# prediction/lgbm_pipeline.py
# ==============================================================================
# Phase 4B — LightGBM (Non-Linear Main Model)
# ==============================================================================
# Matches server API: config.py, data_loader.load_dataset, cv_engine.run_cv.
#
# LightGBM needs early stopping with eval_set, but cv_engine calls
# model.fit(X_train, y_train) without eval_set.  Solution: a wrapper class
# that internally splits off the last 10% of training data as validation.
#
# Usage:
#   python lgbm_pipeline.py                            # all 16 targets, both assets
#   python lgbm_pipeline.py --targets ret_15s mfe_60s  # specific targets
#   python lgbm_pipeline.py --target-family ret        # all 8 return horizons
#   python lgbm_pipeline.py --target-family mfe mae    # all excursion targets
#   python lgbm_pipeline.py --tune --n-trials 100      # Optuna tuning first
#   python lgbm_pipeline.py --hp-file path.json        # pre-tuned HP
#   python lgbm_pipeline.py --seeds 42                 # single seed (fast)
#   python lgbm_pipeline.py --n-jobs 8                 # cores for LightGBM
#
# ==============================================================================
from __future__ import annotations
import os
import signal

import argparse, gc, json, logging, sys, time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


# ==============================================================================
# LightGBM wrapper with internal early stopping
# ==============================================================================

class LGBMWrapper:
    """
    Sklearn-compatible wrapper around LGBMRegressor.
    Splits last `val_frac` of training data for early stopping.
    Conforms to: model.fit(X, y) / model.predict(X)  (cv_engine contract).
    """
    def __init__(self, params: dict, val_frac: float = 0.10,
                 early_stopping: int = 50):
        import lightgbm as lgb
        self.lgb = lgb
        self.params = dict(params)
        self.val_frac = val_frac
        self.early_stopping = early_stopping
        self.model_ = None
        self.best_iteration_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        n = len(X)
        n_val = max(int(n * self.val_frac), 1000)
        n_train = n - n_val

        X_fit, y_fit = X[:n_train], y[:n_train]
        X_val, y_val = X[n_train:], y[n_train:]

        p = dict(self.params)
        n_est = p.pop("n_estimators", 1000)

        self.model_ = self.lgb.LGBMRegressor(n_estimators=n_est, **p)
        self.model_.fit(
            X_fit, y_fit,
            eval_set=[(X_val, y_val)],
            callbacks=[
                self.lgb.early_stopping(stopping_rounds=self.early_stopping),
                self.lgb.log_evaluation(period=0),
            ],
        )
        self.best_iteration_ = self.model_.best_iteration_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict(X)

    @property
    def feature_importances_(self):
        return self.model_.feature_importances_ if self.model_ else None


def make_lgbm_model_fn(params: Optional[dict] = None, val_frac=0.10,
                        early_stopping=50, n_jobs=None):
    """
    Return model_fn(seed) → LGBMWrapper.  Matches cv_engine contract.
    """
    from common.config import LGBM_PARAMS
    base = dict(LGBM_PARAMS)
    if params:
        base.update(params)
    if n_jobs is not None:
        base["n_jobs"] = n_jobs

    def model_fn(seed: int) -> LGBMWrapper:
        p = dict(base)
        p["random_state"] = seed
        return LGBMWrapper(p, val_frac=val_frac, early_stopping=early_stopping)

    return model_fn


# ==============================================================================
# Optuna tuning
# ==============================================================================

OPTUNA_SPACE = {
    "num_leaves": (15, 63), "learning_rate": (0.01, 0.1),
    "colsample_bytree": (0.3, 0.8), "subsample": (0.6, 1.0),
    "reg_alpha": (1e-3, 10.0), "reg_lambda": (1e-3, 10.0),
    "min_child_samples": (50, 500),
}

def tune_lgbm(X, y, n_trials=100, n_inner_folds=3, seed=42, timeout_s=None):
    """Bayesian HP optimisation with inner temporal CV. Returns best HP dict."""
    import lightgbm as lgb
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    from common.config import LGBM_PARAMS

    n = len(X); block = n // (n_inner_folds + 1)

    def objective(trial):
        hp = {
            "num_leaves": trial.suggest_int("num_leaves", *OPTUNA_SPACE["num_leaves"]),
            "learning_rate": trial.suggest_float("learning_rate", *OPTUNA_SPACE["learning_rate"], log=True),
            "colsample_bytree": trial.suggest_float("colsample_bytree", *OPTUNA_SPACE["colsample_bytree"]),
            "subsample": trial.suggest_float("subsample", *OPTUNA_SPACE["subsample"]),
            "reg_alpha": trial.suggest_float("reg_alpha", *OPTUNA_SPACE["reg_alpha"], log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", *OPTUNA_SPACE["reg_lambda"], log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", *OPTUNA_SPACE["min_child_samples"]),
        }
        p = dict(LGBM_PARAMS); p.update(hp); p["random_state"] = seed
        n_est = p.pop("n_estimators", 1000)

        scores = []
        for k in range(n_inner_folds):
            tr_end = (k+1)*block; ts = tr_end; te = min(ts+block, n)
            if tr_end < 50_000: continue
            # Split train into fit+val for early stopping
            nv = max(int(tr_end * 0.1), 1000); nf = tr_end - nv
            m = lgb.LGBMRegressor(n_estimators=n_est, **p)
            m.fit(X[:nf], y[:nf], eval_set=[(X[nf:tr_end], y[nf:tr_end])],
                  callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
            pred = m.predict(X[ts:te])
            ss_r = np.sum((y[ts:te] - pred)**2)
            ss_t = np.sum((y[ts:te] - y[ts:te].mean())**2)
            scores.append(1 - ss_r / (ss_t + 1e-12))
            del m; gc.collect()
        return float(np.mean(scores)) if scores else -1.0

    study = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, timeout=timeout_s)
    logger.info("Optuna best R²=%.6f (%d trials)", study.best_value, len(study.trials))
    return dict(study.best_params)


# ==============================================================================
# Run experiments
# ==============================================================================

def run_lgbm_experiments(
    assets=("btc", "eth"), targets=("ret_15s", "ret_1s"), n_folds=5,
    seeds=(42, 123, 999), params=None, tune=False, n_trials=100,
    max_hours=None, n_jobs=None,
):
    from common.data_loader import load_dataset
    from common.cv_engine import expanding_window_folds, run_cv
    from common.config import RESULTS_DIR
    import pandas as pd

    out_dir = RESULTS_DIR / "lgbm"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_summaries = []

    # Build job list: (asset, target) pairs
    jobs = [(a, t) for a in assets for t in targets]

    for asset, tgt in jobs:
        logger.info("━━ LightGBM  %s/%s ━━", asset.upper(), tgt)
        try:
            # LightGBM consumes the 'tree' feature profile (all surviving
            # features; trees are not multicollinearity-sensitive).
            X, y, info, feat_names = load_dataset(
                target=tgt, asset=asset, profile="tree", max_hours=max_hours)
        except Exception as e:
            logger.error("Load fail %s/%s: %s", asset, tgt, e); continue

        # Optional Optuna tuning
        run_params = dict(params) if params else None
        model_name = "LightGBM"
        if tune:
            logger.info("Optuna tuning (%d trials)...", n_trials)
            t0 = time.time()
            best_hp = tune_lgbm(X, y, n_trials=n_trials)
            logger.info("Tuning done in %.0fs", time.time() - t0)
            hp_path = out_dir / f"lgbm_best_hp_{asset}_{tgt}.json"
            with open(hp_path, "w") as f:
                json.dump(best_hp, f, indent=2)
            run_params = best_hp
            model_name = "LightGBM-Tuned"

        model_fn = make_lgbm_model_fn(params=run_params, n_jobs=n_jobs)
        folds = expanding_window_folds(n_samples=len(X), n_folds=n_folds)

        result = run_cv(
            X=X, y=y, model_fn=model_fn, folds=folds,
            seeds=list(seeds), feature_names=feat_names,
            horizon=tgt, asset=asset, model_name=model_name,
        )

        all_summaries.append(result.summary_dict())

        # Save fold-level detail
        result.fold_table().to_csv(
            out_dir / f"{model_name.lower().replace('-','_')}_{asset}_{tgt}_folds.csv",
            index=False)

        # Save feature importance (averaged across folds from best-seed models)
        _save_importance(result, feat_names, out_dir, model_name, asset, tgt)

        # Save per-row OOS predictions + timestamp (for the cluster A<->B join)
        _save_oos_predictions(result, info, folds, out_dir, model_name, asset, tgt)

        del X, y, info
        gc.collect()
        try:
            import ctypes
            ctypes.CDLL('libc.so.6').malloc_trim(0)  # release memory to OS (glibc/Linux only)
        except Exception:
            pass

    # Summary table
    results_df = pd.DataFrame(all_summaries)
    tag = "lgbm_tuned" if tune else "lgbm"
    results_df.to_csv(out_dir / f"{tag}_summary.csv", index=False)
    print(f"\n{'='*60}\n  LightGBM {'(Tuned)' if tune else '(Default)'}\n{'='*60}")
    print(results_df.to_string(index=False, float_format="%.6f"))
    return results_df


def _save_importance(result, feat_names, out_dir, model_name, asset, tgt):
    """Extract and save averaged feature importance from fold results.

    cv_engine v3 stores the best-seed feature_importances_ on each FoldResult
    (field `importances`). We average across folds. Estimators without
    importances (Ridge/MLP) leave the field None → nothing is saved.
    """
    import pandas as pd
    tag = model_name.lower().replace("-", "_")
    try:
        importances = []
        for fold_res in result.fold_results:
            fi = getattr(fold_res, "importances", None)
            if fi is not None and len(fi) == len(feat_names):
                importances.append(fi)
        if not importances:
            logger.info("  Feature importance: no importances stored in fold results — skipping")
            return
        avg_fi = np.mean(importances, axis=0)
        fi_df = pd.DataFrame({
            "feature": feat_names,
            "importance_mean": avg_fi,
        }).sort_values("importance_mean", ascending=False).reset_index(drop=True)
        fi_df["rank"] = fi_df.index + 1
        out_path = out_dir / f"{tag}_{asset}_{tgt}_feature_importance.csv"
        fi_df.to_csv(out_path, index=False)
        logger.info("  Feature importance saved: %s (top: %s)",
                    out_path.name, fi_df["feature"].iloc[0] if len(fi_df) else "n/a")
    except Exception as e:
        logger.warning("  Feature importance save failed: %s", e)


def _save_oos_predictions(result, info, folds, out_dir, model_name, asset, tgt):
    """Dump per-row OOS predictions + timestamp per fold (parquet).

    Used by the cluster A<->B comparison: join sign(y_pred) at each breakout
    event timestamp against the cluster's directional call. Folds are
    contiguous test ranges [test_start:test_end]; block 0 is never a test
    block, so the earliest ~1/(n_folds+1) of rows have no OOS prediction.
    """
    import pandas as pd
    tag = model_name.lower().replace("-", "_")
    try:
        ts = info["timestamp"].values
        fold_by_id = {f.fold_id: f for f in folds}
        parts = []
        for fr in result.fold_results:
            if fr.predictions is None:
                continue
            f = fold_by_id[fr.fold_id]
            n = len(fr.predictions)
            # ABSOLUTE row position of each OOS row in the full dataset. This is
            # the robust join key for the cluster A<->B comparison: cluster_trades
            # carries the same event_index. (The loader timestamp is unreliable,
            # so we do NOT rely on it for the join.)
            event_index = np.arange(f.test_start, f.test_start + n, dtype="int64")
            parts.append(pd.DataFrame(dict(
                event_index=event_index,
                timestamp=ts[f.test_start:f.test_start + n],
                y_pred=fr.predictions,
                y_true=fr.actuals,
                fold=fr.fold_id,
            )))
        if not parts:
            logger.info("  OOS predictions: none stored — skipping")
            return
        out_path = out_dir / f"{tag}_{asset}_{tgt}_oos_predictions.parquet"
        pd.concat(parts, ignore_index=True).to_parquet(out_path, index=False)
        logger.info("  OOS predictions saved: %s (%d rows)",
                    out_path.name, sum(len(p) for p in parts))
    except Exception as e:
        logger.warning("  OOS predictions save failed: %s", e)


# ==============================================================================
# CLI
# ==============================================================================

def main():
    p = argparse.ArgumentParser(description="Phase 4B: LightGBM")
    p.add_argument("--asset", choices=["btc", "eth", "both"], default="both")
    p.add_argument("--targets", nargs="+", default=None,
                   help="Flat target tokens, e.g. 'ret_15s mfe_60s mae_300s'. "
                        "A bare horizon ('15s') is treated as a return target. "
                        "Default: all 8 return + 8 MFE/MAE targets.")
    p.add_argument("--target-family", nargs="+", default=None,
                   choices=["ret", "mfe", "mae"],
                   help="Shortcut: run every horizon of the given families "
                        "instead of listing --targets explicitly.")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                   help="Random seeds (default: config.DEFAULT_SEEDS = [42,123,999])")
    p.add_argument("--tune", action="store_true")
    p.add_argument("--n-trials", type=int, default=100)
    p.add_argument("--hp-file", type=str, default=None)
    p.add_argument("--max-hours", type=int, default=None)
    p.add_argument("--n-jobs", type=int, default=8,
                   help="CPU cores for LightGBM (default: 8). Use -1 for all cores.")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    a = p.parse_args()

    logging.basicConfig(level=getattr(logging, a.log_level),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S", stream=sys.stdout)

    assets = ("btc", "eth") if a.asset == "both" else (a.asset,)
    params = None
    if a.hp_file:
        with open(a.hp_file) as f: params = json.load(f)
        logger.info("Loaded HP from %s", a.hp_file)

    # Seeds: CLI override or config default
    if a.seeds:
        seeds = tuple(a.seeds)
    else:
        from common.config import DEFAULT_SEEDS
        seeds = tuple(DEFAULT_SEEDS)

    # Resolve target list. Precedence: --targets > --target-family > full set.
    from common.config import all_targets
    if a.targets:
        targets = tuple(a.targets)
    elif a.target_family:
        targets = tuple(all_targets(tuple(a.target_family)))
    else:
        targets = tuple(all_targets())  # all 8 ret + 4 mfe + 4 mae = 16
    logger.info("Targets (%d): %s", len(targets), list(targets))

    run_lgbm_experiments(
        assets=assets, targets=targets, n_folds=a.n_folds,
        seeds=seeds, params=params, tune=a.tune, n_trials=a.n_trials,
        max_hours=a.max_hours, n_jobs=a.n_jobs,
    )

if __name__ == "__main__":
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        os.setsid()
    except OSError:
        pass
    main()