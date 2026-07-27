"""
cv_engine.py — Expanding-window temporal cross-validation engine.
================================================================
Provides fold generation and a model-agnostic CV runner that works
with any sklearn-compatible estimator (Ridge, LightGBM, MLP wrapper).

Design principles:
  - Temporal ordering: train always precedes test, no look-ahead
  - Multi-seed: each fold trains N seeds, reports mean ± std
  - Fold-level reporting: never just averages (LP4)
  - Model-agnostic: pass any object with fit/predict


Usage:
    from common.cv_engine import expanding_window_folds, run_cv

    folds = expanding_window_folds(n_samples=2_000_000, n_folds=5)
    results = run_cv(X, y, model_fn, folds, seeds=[42, 123, 999])
"""

from __future__ import annotations

import gc
import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─── Fold generation ──────────────────────────────────────────────────────────

@dataclass
class FoldSplit:
    """A single train/test split with indices."""
    fold_id:     int
    train_start: int
    train_end:   int
    test_start:  int
    test_end:    int

    @property
    def train_idx(self) -> np.ndarray:
        return np.arange(self.train_start, self.train_end)

    @property
    def test_idx(self) -> np.ndarray:
        return np.arange(self.test_start, self.test_end)

    @property
    def train_size(self) -> int:
        return self.train_end - self.train_start

    @property
    def test_size(self) -> int:
        return self.test_end - self.test_start


def expanding_window_folds(n_samples: int, n_folds: int = 5) -> list[FoldSplit]:
    """
    Generate expanding-window CV folds.

    The data is divided into (n_folds + 1) equal blocks.
    Fold k uses blocks [0..k] for training and block [k+1] for testing.

    This ensures:
      - Train always precedes test temporally
      - No look-ahead
      - Train size grows with each fold
      - Each test block is used exactly once
    """
    block_size = n_samples // (n_folds + 1)
    if block_size < 1000:
        logger.warning("Very small block size: %d rows per block.", block_size)

    folds = []
    for k in range(n_folds):
        train_end  = block_size * (k + 1)
        test_start = train_end
        test_end   = min(test_start + block_size, n_samples)

        if test_end <= test_start:
            break

        folds.append(FoldSplit(
            fold_id     = k + 1,
            train_start = 0,
            train_end   = train_end,
            test_start  = test_start,
            test_end    = test_end,
        ))

    logger.info("Generated %d folds: block_size=%s, total=%s",
                len(folds), f"{block_size:,}", f"{n_samples:,}")
    for f in folds:
        logger.info("  Fold %d: train=[0:%s] test=[%s:%s]",
                     f.fold_id, f"{f.train_end:,}",
                     f"{f.test_start:,}", f"{f.test_end:,}")

    return folds


# ─── CV Results ───────────────────────────────────────────────────────────────

@dataclass
class FoldResult:
    """Results from one fold (averaged over seeds)."""
    fold_id:       int
    train_size:    int
    test_size:     int
    r2_mean:       float
    r2_std:        float
    mse_mean:      float
    mae_mean:      float
    dir_acc_mean:  float
    ic_mean:        float
    n_seeds:       int
    seconds:       float
    # Per-seed detail
    r2_per_seed:   list = field(default_factory=list)
    predictions:   Optional[np.ndarray] = None  # best-seed predictions on test
    actuals:       Optional[np.ndarray] = None   # actual test values
    importances:   Optional[np.ndarray] = None   # best-seed feature_importances_


@dataclass
class CVResult:
    """Full CV results across all folds."""
    fold_results: list[FoldResult]
    feature_names: list[str]
    horizon:       str
    asset:         str
    model_name:    str

    @property
    def n_folds(self) -> int:
        return len(self.fold_results)

    @property
    def r2_per_fold(self) -> list[float]:
        return [f.r2_mean for f in self.fold_results]

    @property
    def r2_mean(self) -> float:
        return float(np.mean(self.r2_per_fold))

    @property
    def r2_std(self) -> float:
        return float(np.std(self.r2_per_fold))

    @property
    def n_positive_folds(self) -> int:
        return sum(1 for r in self.r2_per_fold if r > 0)

    def summary_dict(self) -> dict:
        r2s = self.r2_per_fold
        return {
            "model":           self.model_name,
            "asset":           self.asset,
            "horizon":         self.horizon,
            "n_folds":         self.n_folds,
            "r2_mean":         round(self.r2_mean, 6),
            "r2_std":          round(self.r2_std, 6),
            "r2_min":          round(float(np.min(r2s)), 6),
            "r2_max":          round(float(np.max(r2s)), 6),
            "n_positive":      self.n_positive_folds,
            "mse_mean":        round(float(np.mean([f.mse_mean for f in self.fold_results])), 10),
            "mae_mean":        round(float(np.mean([f.mae_mean for f in self.fold_results])), 8),
            "dir_acc_mean":    round(float(np.mean([f.dir_acc_mean for f in self.fold_results])), 4),
            "ic_mean":         round(float(np.mean([f.ic_mean for f in self.fold_results])), 6),
        }

    def fold_table(self) -> pd.DataFrame:
        """Per-fold detail table."""
        rows = []
        for f in self.fold_results:
            rows.append({
                "fold":       f.fold_id,
                "train_size": f.train_size,
                "test_size":  f.test_size,
                "r2":         round(f.r2_mean, 6),
                "r2_std":     round(f.r2_std, 6),
                "mse":        round(f.mse_mean, 10),
                "mae":        round(f.mae_mean, 8),
                "dir_acc":    round(f.dir_acc_mean, 4),
                "ic":         round(f.ic_mean, 6),
                "n_seeds":    f.n_seeds,
                "seconds":    round(f.seconds, 1),
            })
        return pd.DataFrame(rows)


# ─── CV Runner ────────────────────────────────────────────────────────────────

def run_cv(
    X: np.ndarray,
    y: np.ndarray,
    model_fn: Callable,
    folds: list[FoldSplit],
    seeds: list[int],
    feature_names: list[str],
    horizon: str = "",
    asset: str = "",
    model_name: str = "",
    store_predictions: bool = True,
) -> CVResult:
    """
    Run cross-validation with multiple seeds per fold.

    Args:
        X: Feature matrix (n_samples, n_features), may contain NaN
        y: Target vector (n_samples,)
        model_fn: Callable(seed) → fitted-model-like object with fit/predict.
                  Called once per seed. Example:
                    lambda seed: Ridge(alpha=1.0, random_state=seed)
        folds: List of FoldSplit from expanding_window_folds
        seeds: List of random seeds for repeated training
        feature_names: Column names for X
        store_predictions: If True, store predictions from best seed per fold

    Returns:
        CVResult with per-fold and aggregate metrics
    """
    from common.metrics import compute_fold_metrics

    fold_results = []

    for fold in folds:
        t0 = time.time()

        # Slice (view) instead of fancy indexing (copy). For a 5M-row × 3349-
        # feature dataset, fancy indexing duplicates ~60 GB of RAM per fold;
        # slicing is free. fold.train_idx / fold.test_idx properties are still
        # available for callers that need explicit index arrays.
        X_train = X[fold.train_start:fold.train_end]
        y_train = y[fold.train_start:fold.train_end]
        X_test  = X[fold.test_start:fold.test_end]
        y_test  = y[fold.test_start:fold.test_end]

        logger.info("Fold %d/%d: train=%s test=%s, seeds=%s",
                     fold.fold_id, len(folds),
                     f"{fold.train_size:,}", f"{fold.test_size:,}", seeds)

        seed_metrics = []
        best_preds   = None
        best_fi      = None
        best_r2      = -np.inf

        for seed in seeds:
            model = model_fn(seed)

            try:
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
            except Exception as e:
                logger.warning("  Seed %d failed: %s", seed, e)
                continue

            metrics = compute_fold_metrics(y_test, preds)
            seed_metrics.append(metrics)

            if metrics["r2"] > best_r2:
                best_r2 = metrics["r2"]
                if store_predictions:
                    best_preds = preds.copy()
                # Capture importance from the best seed BEFORE the model is
                # freed. Estimators without feature_importances_ (Ridge, MLP)
                # return None → field stays None, no error.
                fi = getattr(model, "feature_importances_", None)
                best_fi = np.asarray(fi).copy() if fi is not None else None

            del model
            gc.collect()

        if not seed_metrics:
            logger.error("  All seeds failed for fold %d — skipping.", fold.fold_id)
            continue

        dt = time.time() - t0

        # Aggregate across seeds
        r2s      = [m["r2"] for m in seed_metrics]
        mses     = [m["mse"] for m in seed_metrics]
        maes     = [m["mae"] for m in seed_metrics]
        dir_accs = [m["dir_acc"] for m in seed_metrics]
        ics      = [m["ic"] for m in seed_metrics]

        fold_result = FoldResult(
            fold_id      = fold.fold_id,
            train_size   = fold.train_size,
            test_size    = fold.test_size,
            r2_mean      = float(np.mean(r2s)),
            r2_std       = float(np.std(r2s)),
            mse_mean     = float(np.mean(mses)),
            mae_mean     = float(np.mean(maes)),
            dir_acc_mean = float(np.mean(dir_accs)),
            ic_mean      = float(np.mean(ics)),
            n_seeds      = len(seed_metrics),
            seconds      = dt,
            r2_per_seed  = r2s,
            predictions  = best_preds,
            # y_test is a view into y; .copy() ensures the stored array is
            # independent and won't be invalidated when y is freed.
            actuals      = y_test.copy() if store_predictions else None,
            importances  = best_fi,
        )

        fold_results.append(fold_result)

        logger.info("  Fold %d: R²=%.6f (±%.6f), DA=%.4f, IC=%.6f [%.1fs]",
                     fold.fold_id, fold_result.r2_mean, fold_result.r2_std,
                     fold_result.dir_acc_mean, fold_result.ic_mean, dt)

        # X_train / X_test were views (no allocation); deleting them releases
        # only the local reference. The underlying X is unaffected.
        del X_train, y_train, X_test, y_test
        gc.collect()

    cv_result = CVResult(
        fold_results  = fold_results,
        feature_names = feature_names,
        horizon       = horizon,
        asset         = asset,
        model_name    = model_name,
    )

    logger.info("CV done: R²=%.6f (±%.6f), %d/%d folds positive, DA=%.4f",
                cv_result.r2_mean, cv_result.r2_std,
                cv_result.n_positive_folds, cv_result.n_folds,
                np.mean([f.dir_acc_mean for f in fold_results]))

    return cv_result