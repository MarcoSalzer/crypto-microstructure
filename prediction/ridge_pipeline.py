#!/usr/bin/env python3
"""
ridge_pipeline.py — Phase 4A: Ridge / ElasticNet Baseline
=========================================================
Runs Ridge and ElasticNet regression on the linear feature profile using
expanding-window CV. Shows whether linear signal exists in the features.

This is the fastest model to train and the first real ML result.
If Ridge R² > 0, linear signal exists. If ElasticNet drops many features,
most signal is concentrated in a subset.

Feature profile
---------------
Ridge consumes the 'linear' feature profile from feature_keep.csv: only
features with use_linear == True (VIF <= 10, ~360 features). Linear-model
coefficients are unstable under multicollinearity, so the dense, highly
collinear microstructure parameterisations are deliberately excluded for
this model class. This is the profile distinction established in the
correlation diagnostics chapter.

Usage
-----
    # Standard run (all 16 targets, both assets):
    python -m prediction.ridge_pipeline

    # Specific targets:
    python -m prediction.ridge_pipeline --targets ret_15s mfe_60s

    # All return horizons only:
    python -m prediction.ridge_pipeline --target-family ret

    # With ElasticNet:
    python -m prediction.ridge_pipeline --elasticnet

Outputs (in results/ridge/):
    ridge_summary.csv                 — R² per asset × target
    ridge_{asset}_{target}_folds.csv  — Per-fold detail
    ridge_{asset}_{target}_coefs.csv  — Top feature coefficients
"""

import argparse
import logging
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Add parent dir to path for imports

from common.config import (
    RESULTS_DIR, DEFAULT_SEEDS, N_FOLDS_MAIN,
    HORIZONS, ASSETS, SPREAD_BPS, target_col, all_targets,
)
from common.data_loader import load_dataset
from common.cv_engine import expanding_window_folds, run_cv
from common.metrics import simple_backtest, ic_ir, directional_mcc

warnings.filterwarnings("ignore")
logger = logging.getLogger("ridge_pipeline")


# ─── Prediction clipping wrapper ──────────────────────────────────────────────

class ClippedPipeline:
    """
    Wraps an sklearn Pipeline and clips predictions to ±clip_std * y_train.std().
    Prevents R² explosion from numerically unstable coefficients.
    Implements fit/predict so cv_engine can use it directly.
    """
    def __init__(self, pipeline, clip_std: float = 10.0):
        self.pipeline = pipeline
        self.clip_std = clip_std
        self._y_std   = 1.0
        self._y_mean  = 0.0

    def fit(self, X, y, **kwargs):
        MAX_N = 400_000
        if len(X) > MAX_N:
            n = len(X)
            n_recent = MAX_N // 2
            rng = np.random.RandomState(42)
            early_idx = rng.choice(n - n_recent, size=MAX_N - n_recent, replace=False)
            idx = np.sort(np.concatenate([early_idx, np.arange(n - n_recent, n)]))
            X, y = X[idx], y[idx]
        self._y_std  = max(float(np.std(y)), 1e-12)
        self._y_mean = float(np.mean(y))
        self.pipeline.fit(X, y, **kwargs)
        return self

    def predict(self, X):
        preds = self.pipeline.predict(X)
        bound = self.clip_std * self._y_std
        return np.clip(preds, self._y_mean - bound, self._y_mean + bound)

    @property
    def named_steps(self):
        return self.pipeline.named_steps


# ─── Ridge model factory ─────────────────────────────────────────────────────

def make_ridge_pipeline(seed: int, alpha: float = 100.0):
    """
    Create a Ridge regression pipeline with:
      - Median imputation (fit on train only)
      - Variance threshold (removes near-constant features)
      - Standard scaling (fit on train only)
      - Ridge regression with prediction clipping
    """
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.linear_model import Ridge

    pipe = Pipeline([
        ("impute",   SimpleImputer(strategy="median")),
        ("var_filt", VarianceThreshold(threshold=1e-8)),
        ("scale",    StandardScaler()),
        ("model",    Ridge(alpha=alpha)),
    ])
    return ClippedPipeline(pipe)


def make_ridgecv_pipeline(seed: int):
    """
    Ridge with alpha selected via expanding-window CV (time-series safe).

    sklearn's RidgeCV uses LOO-CV which shuffles examples — invalid for
    time series (future look-ahead → alpha too small → overfitting → negative R²).

    Fix: evaluate a grid of fixed alphas on a single held-out validation
    block (last 15% of training data), pick the best, then refit on all
    training data. This is O(n_alphas) fits but avoids temporal look-ahead.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.linear_model import Ridge

    # Alpha grid: strong regularisation needed for 1600+ collinear features
    ALPHAS = [100, 500, 1_000, 5_000, 10_000, 50_000, 100_000]

    from sklearn.base import BaseEstimator
    class TimeSeriesRidgeCV(BaseEstimator):
        """Ridge with temporal hold-out alpha selection."""
        def __init__(self):
            self.coef_ = None
            self.intercept_ = 0.0
            self.alpha_ = None

        def fit(self, X, y):
            n = len(X)
            # Subsample: Ridge converges well at 400k rows, no need for 5M+
            MAX_TRAIN = 400_000
            if n > MAX_TRAIN:
                import numpy as _np
                # Keep most recent 200k + random 200k from earlier
                n_recent = MAX_TRAIN // 2
                n_rand   = MAX_TRAIN - n_recent
                early_idx = _np.random.RandomState(42).choice(n - n_recent, size=n_rand, replace=False)
                recent_idx = _np.arange(n - n_recent, n)
                idx = _np.sort(_np.concatenate([early_idx, recent_idx]))
                X, y = X[idx], y[idx]
                n = len(X)
            n_val = max(int(n * 0.15), 1000)
            X_fit, y_fit = X[:n - n_val], y[:n - n_val]
            X_val, y_val = X[n - n_val:], y[n - n_val:]

            best_alpha, best_r2 = ALPHAS[0], -np.inf
            for a in ALPHAS:
                m = Ridge(alpha=a)
                m.fit(X_fit, y_fit)
                p = m.predict(X_val)
                ss_res = np.sum((y_val - p) ** 2)
                ss_tot = np.sum((y_val - y_val.mean()) ** 2)
                r2 = 1 - ss_res / (ss_tot + 1e-12)
                if r2 > best_r2:
                    best_r2, best_alpha = r2, a

            # Refit on full training data with best alpha
            final = Ridge(alpha=best_alpha)
            final.fit(X, y)
            self.coef_       = final.coef_
            self.intercept_  = final.intercept_
            self.alpha_      = best_alpha
            return self

        def predict(self, X):
            return X @ self.coef_ + self.intercept_

    pipe = Pipeline([
        ("impute",   SimpleImputer(strategy="median")),
        ("var_filt", VarianceThreshold(threshold=1e-8)),
        ("scale",    StandardScaler()),
        ("model",    TimeSeriesRidgeCV()),
    ])
    return ClippedPipeline(pipe)


def make_elasticnet_pipeline(seed: int):
    """ElasticNet with coordinate descent."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.linear_model import ElasticNetCV

    pipe = Pipeline([
        ("impute",   SimpleImputer(strategy="median")),
        ("var_filt", VarianceThreshold(threshold=1e-8)),
        ("scale",    StandardScaler()),
        ("model",    ElasticNetCV(
            l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99],
            n_alphas=30,
            cv=3,  # inner temporal CV
            max_iter=5000,
            random_state=seed,
        )),
    ])
    return ClippedPipeline(pipe)


# ─── Coefficient extraction ──────────────────────────────────────────────────

def extract_coefficients(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    model_fn,
    seed: int = 42,
    top_n: int = 50,
) -> pd.DataFrame:
    """
    Train on full data and extract top coefficients by absolute magnitude.
    This is for interpretability, not for the CV results.
    Handles VarianceThreshold reducing the feature set.
    """
    model = model_fn(seed)
    model.fit(X, y)

    # Get coefficients from the pipeline's last step (ClippedPredictor → inner estimator)
    estimator = model.named_steps["model"]
    coefs = estimator.coef_

    # Get surviving feature names after VarianceThreshold
    if "var_filt" in model.named_steps:
        var_mask = model.named_steps["var_filt"].get_support()
        surviving_names = [f for f, keep in zip(feature_names, var_mask) if keep]
    else:
        surviving_names = feature_names

    # Safety check: coefs length must match surviving names
    if len(coefs) != len(surviving_names):
        logger.warning("Coef length (%d) != surviving features (%d) — using indices.",
                        len(coefs), len(surviving_names))
        surviving_names = [f"feature_{i}" for i in range(len(coefs))]

    df = pd.DataFrame({
        "feature":  surviving_names,
        "coef":     coefs,
        "abs_coef": np.abs(coefs),
    })
    df = df.sort_values("abs_coef", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1

    # Add selected alpha if RidgeCV
    if hasattr(estimator, "alpha_"):
        df["selected_alpha"] = estimator.alpha_

    # Add nonzero count for ElasticNet
    if hasattr(estimator, "l1_ratio_"):
        n_nonzero = (np.abs(coefs) > 1e-10).sum()
        df["n_nonzero"]      = n_nonzero
        df["n_total"]        = len(coefs)
        df["selected_l1"]    = estimator.l1_ratio_
        df["selected_alpha"] = estimator.alpha_

    # Log removed features
    n_removed = len(feature_names) - len(surviving_names)
    if n_removed > 0:
        logger.info("  VarianceThreshold removed %d/%d features.",
                     n_removed, len(feature_names))

    return df.head(top_n)


# ─── Save results ─────────────────────────────────────────────────────────────

def save_results(
    cv_result,
    coefs_df: pd.DataFrame,
    output_dir: Path,
    model_name: str,
    asset: str,
    horizon: str,
    backtest_results: dict | None = None,
):
    """Save fold table, coefficients, and backtest results."""
    os.makedirs(output_dir, exist_ok=True)
    prefix = f"{model_name}_{asset}_{horizon}"

    # Fold table
    fold_df = cv_result.fold_table()
    fold_path = output_dir / f"{prefix}_folds.csv"
    fold_df.to_csv(fold_path, index=False)
    logger.info("  Saved: %s", fold_path.name)

    # Coefficients
    coef_path = output_dir / f"{prefix}_coefs.csv"
    coefs_df.to_csv(coef_path, index=False)
    logger.info("  Saved: %s", coef_path.name)

    # Backtest
    if backtest_results:
        bt_path = output_dir / f"{prefix}_backtest.csv"
        pd.DataFrame([backtest_results]).to_csv(bt_path, index=False)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Phase 4A: Ridge / ElasticNet Baseline")
    parser.add_argument("--assets", nargs="+", default=["btc", "eth"])
    parser.add_argument("--targets", nargs="+", default=None,
                        help="Flat target tokens, e.g. 'ret_15s mfe_60s'. "
                             "A bare horizon ('15s') means a return target. "
                             "Default: all 8 return + 8 MFE/MAE targets.")
    parser.add_argument("--target-family", nargs="+", default=None,
                        choices=["ret", "mfe", "mae"],
                        help="Shortcut: run every horizon of the given "
                             "families instead of listing --targets.")
    parser.add_argument("--elasticnet", action="store_true",
                        help="Also run ElasticNet (slower)")
    parser.add_argument("--max-hours", type=int, default=None,
                        help="Limit hours for quick iteration")
    parser.add_argument("--n-folds", type=int, default=N_FOLDS_MAIN)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--top-coefs", type=int, default=50)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    output_dir = RESULTS_DIR / "ridge"
    os.makedirs(output_dir, exist_ok=True)

    t0 = time.time()
    all_summaries = []

    # Resolve target list. Precedence: --targets > --target-family > full set.
    if args.targets:
        targets = list(args.targets)
    elif args.target_family:
        targets = all_targets(tuple(args.target_family))
    else:
        targets = all_targets()  # all 8 ret + 4 mfe + 4 mae = 16

    # Build job list: (asset, target) pairs
    jobs = []
    for asset in args.assets:
        for tgt in targets:
            jobs.append((asset, tgt))

    models_to_run = [("ridge", make_ridgecv_pipeline)]
    if args.elasticnet:
        models_to_run.append(("elasticnet", make_elasticnet_pipeline))

    logger.info("=" * 70)
    logger.info("RIDGE BASELINE — Phase 4A")
    logger.info("  Feature profile: linear (use_linear == True)")
    logger.info("  Jobs: %d  (%d assets x %d targets)",
                len(jobs), len(args.assets), len(targets))
    logger.info("  Targets: %s", targets)
    logger.info("  Models: %s", [m[0] for m in models_to_run])
    logger.info("  Folds: %d, Seeds: %s", args.n_folds, args.seeds)
    logger.info("  Max hours: %s", args.max_hours or "all")
    logger.info("=" * 70)

    for asset, tgt in jobs:
        logger.info("")
        logger.info("━━ %s × %s ━━", asset.upper(), tgt)

        # Load data — Ridge uses the 'linear' feature profile
        try:
            X, y, info, feature_names = load_dataset(
                target=tgt,
                asset=asset,
                profile="linear",
                max_hours=args.max_hours,
            )
        except Exception as e:
            logger.error("Failed to load data: %s", e)
            continue

        # Generate folds
        folds = expanding_window_folds(len(y), args.n_folds)

        for model_name, model_factory in models_to_run:
            logger.info("")
            logger.info("── %s ──", model_name.upper())

            # Run CV
            cv_result = run_cv(
                X=X, y=y,
                model_fn=model_factory,
                folds=folds,
                seeds=args.seeds,
                feature_names=feature_names,
                horizon=tgt,
                asset=asset,
                model_name=model_name,
            )

            # Extract coefficients (full dataset, for interpretability)
            logger.info("  Extracting top %d coefficients...", args.top_coefs)
            coefs_df = extract_coefficients(
                X, y, feature_names,
                model_factory, seed=args.seeds[0],
                top_n=args.top_coefs,
            )

            # Backtest on concatenated fold predictions
            bt_results = None
            if asset != "relative":
                all_preds = np.concatenate(
                    [f.predictions for f in cv_result.fold_results
                     if f.predictions is not None])
                all_acts  = np.concatenate(
                    [f.actuals for f in cv_result.fold_results
                     if f.actuals is not None])
                if len(all_preds) > 0:
                    spread = SPREAD_BPS[asset]["fut"]
                    bt_results = simple_backtest(all_preds, all_acts, spread)
                    logger.info("  Backtest: PnL_net=%.1f bps, Sharpe=%.2f, "
                                "hit_rate=%.4f, trades=%d (%.1f%%)",
                                bt_results["pnl_net_bps"], bt_results["sharpe"],
                                bt_results["hit_rate"], bt_results["trade_count"],
                                bt_results["trade_pct"])

            # IC_IR
            ic_ir_val = ic_ir([f.ic_mean for f in cv_result.fold_results])

            # Summary row
            summary = cv_result.summary_dict()
            summary["ic_ir"] = round(ic_ir_val, 4)
            if bt_results:
                summary.update({f"bt_{k}": v for k, v in bt_results.items()})

            # Top 5 features
            top5 = coefs_df.head(5)["feature"].tolist()
            summary["top5_features"] = " | ".join(top5)

            all_summaries.append(summary)

            # Save
            save_results(cv_result, coefs_df, output_dir,
                         model_name, asset, tgt, bt_results)

            # Print fold table
            logger.info("\n%s", cv_result.fold_table().to_string(index=False))
            logger.info("\n  Top 10 coefficients:")
            for _, row in coefs_df.head(10).iterrows():
                logger.info("    #%d  %+.6f  %s",
                            row["rank"], row["coef"], row["feature"])

        del X, y, info
        import gc; gc.collect()

    # ── Save combined summary ─────────────────────────────────────────────────
    if all_summaries:
        summary_df = pd.DataFrame(all_summaries)
        summary_path = output_dir / "ridge_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        logger.info("\n" + "=" * 70)
        logger.info("SUMMARY TABLE:")
        logger.info("=" * 70)
        show_cols = ["model", "asset", "horizon", "r2_mean", "r2_std",
                     "n_positive", "dir_acc_mean", "ic_mean", "ic_ir"]
        show_cols = [c for c in show_cols if c in summary_df.columns]
        logger.info("\n%s", summary_df[show_cols].to_string(index=False))
        logger.info("\nSaved: %s", summary_path)

    logger.info("\nDone in %.1f min.", (time.time() - t0) / 60)


if __name__ == "__main__":
    main()