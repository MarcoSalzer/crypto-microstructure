# prediction/mlp_pipeline.py
# ==============================================================================
# Phase 4C — MLP Neural Baseline (PyTorch)
# ==============================================================================
# Sklearn-compatible wrapper so cv_engine.run_cv can call fit/predict.
# Internally: drop all-NaN cols → median impute → scale → clip ±5σ → train.
#
# Feature profile: 'tree' (use_tree == True, all 3349 surviving features).
# Neural networks are not multicollinearity-sensitive, so they receive the
# full feature set rather than the VIF-restricted linear profile.
#
# Usage:
#   python mlp_pipeline.py                          # all 16 targets, both assets
#   python mlp_pipeline.py --targets ret_15s mfe_60s
#   python mlp_pipeline.py --target-family ret
#   python mlp_pipeline.py --seeds 42               # single seed (fast)
#   python mlp_pipeline.py --force-rerun            # ignore existing results
#
#                    Speed fixes:
#                    1) GPU-RESIDENT TRAINING TENSORS — load Xf, yf onto
#                       GPU once at fit() start. With 300k × 3349 float32
#                       = ~4 GB, this fits easily in 49 GB VRAM and
#                       eliminates per-batch transfers entirely.
#                    2) MANUAL INDEXING in training loop instead of
#                       DataLoader — DataLoader overhead is meaningful
#                       when batches are GPU index ops; we replace it
#                       with torch.randperm() + slice.
#                    3) MIXED PRECISION (AMP) — autocast + GradScaler.
#                       ~1.7x speedup on Ampere with BatchNorm/Linear nets.
#                    4) batch_size 4096 → 8192 default (A6000 had headroom).
#                    5) OOM-FALLBACK — if CUDA OOM during fit, halve
#                       batch_size and retry up to 2x rather than crash
#                       the whole job.
#
#                    Crash-resistance:
#                    6) Per-target try/except in run_mlp_experiments —
#                       a fold/fit failure on target X no longer kills
#                       the remaining 31 targets.
#                    7) Aggressive cleanup between targets:
#                       gc.collect() + torch.cuda.empty_cache().
#                    8) Heartbeat log per epoch at INFO level (gated on
#                       --epoch-log flag) for diagnosis on slow runs.
#                    9) batch_size sanity check: error early if n_fit
#                       < batch_size rather than silently train on
#                       zero batches (drop_last=True).
#
#                    Expected speedup vs v8: 3-5x. BTC+ETH end-to-end
#                    target: ~24-36h on A6000.
# ==============================================================================
from __future__ import annotations
import argparse, gc, logging, sys, time
from pathlib import Path
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)


# ==============================================================================
# MLP Wrapper (sklearn-compatible: fit/predict)
# ==============================================================================

class MLPWrapper:
    """
    Sklearn-compatible MLP wrapper for cv_engine.
    Handles preprocessing internally:
      drop NaN cols → impute → scale X → standardise y → clip → train.

    v8: target standardisation + prediction clipping.
    """
    def __init__(self, seed: int, hidden_dims=(512, 256, 128), dropout=0.3,
                 lr=1e-3, weight_decay=1e-4, batch_size=8192, epochs=50,
                 patience=10, val_frac=0.10, clip_sigma=5.0,
                 max_train_samples=500_000,
                 predict_batch_size=16_384,
                 clip_pred_sigma=10.0,
                 use_amp=True, epoch_log=False,
                 oom_retries=2,
                 # compat args kept for backward-compat with old callers;
                 # ignored in v9 because data lives on GPU (no workers needed)
                 num_workers=0, pin_memory=False, persistent_workers=False):
        self.seed = seed
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.val_frac = val_frac
        self.clip_sigma = clip_sigma                # feature clipping (input)
        self.max_train_samples = max_train_samples
        self.predict_batch_size = predict_batch_size
        self.clip_pred_sigma = clip_pred_sigma      # prediction clipping (output)
        self.use_amp = use_amp
        self.epoch_log = epoch_log
        self.oom_retries = oom_retries
        # compat fields (unused but preserved so config files don't break)
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self._model = None
        # v9: manual float32 preprocessing state (replaces sklearn imputer/scaler)
        self._valid_cols     = None
        self._impute_medians = None  # per-column medians for NaN imputation
        self._scale_mean     = None  # per-column means for standardisation
        self._scale_std      = None  # per-column stds for standardisation
        # Compat: keep old _imputer/_scaler refs as None so any external code
        # that introspects them does not break (they were never used outside).
        self._imputer = None
        self._scaler  = None
        self._device = "cpu"
        # v8: target standardisation parameters, set during fit
        self._y_mean = 0.0
        self._y_std  = 1.0

    def fit(self, X: np.ndarray, y: np.ndarray):
        import torch
        import torch.nn as nn
        # v9: sklearn imputer/scaler dropped to avoid 13 GB float64
        # intermediate copies. Manual float32 equivalents in Step 1 below.

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device

        # ── Step -1: Subsample training data if too large ─────────────────────
        n_orig = len(X)
        if self.max_train_samples and n_orig > self.max_train_samples:
            rng = np.random.RandomState(self.seed)
            n_recent = min(self.max_train_samples // 2, n_orig)
            n_random = self.max_train_samples - n_recent
            recent_idx = np.arange(n_orig - n_recent, n_orig)
            early_idx  = rng.choice(n_orig - n_recent, size=n_random, replace=False)
            idx = np.sort(np.concatenate([early_idx, recent_idx]))
            X, y = X[idx], y[idx]
            logger.info("  MLP: subsampled %d → %d training rows (max_train_samples=%d)",
                        n_orig, len(X), self.max_train_samples)

        # ── Step 0: Drop columns that are all-NaN or near-constant ────────────
        nan_frac_per_col = np.isnan(X).mean(axis=0)
        col_std = np.nanstd(X, axis=0)
        self._valid_cols = np.where(
            (nan_frac_per_col < 0.99) & (col_std > 1e-10)
        )[0]

        if len(self._valid_cols) == 0:
            raise ValueError("No valid columns after NaN/constant filter")

        n_dropped = X.shape[1] - len(self._valid_cols)
        if n_dropped > 0:
            logger.info("  MLP: dropped %d/%d cols (all-NaN or constant), %d remain",
                        n_dropped, X.shape[1], len(self._valid_cols))

        # Subset columns. With cv_engine passing a VIEW into the full data
        # array, this materialises a fresh contiguous float32 copy of just
        # the surviving columns. After this point X is independent and
        # ~6.5 GB for 500k × 3272 cols.
        Xp = np.ascontiguousarray(X[:, self._valid_cols], dtype=np.float32)
        del X
        gc.collect()

        # ── Step 1: Manual float32 imputation + scaling + clipping ────────────
        # v9 (memory-critical fix): sklearn's SimpleImputer.fit_transform()
        # ALWAYS returns float64 internally, regardless of input dtype. For a
        # 500k × 3272 input that's a 13 GB intermediate copy that we cannot
        # avoid via `copy=False` (the parameter has no effect on the impute
        # path). On a memory-constrained server this single copy can push
        # the run over the OOM-kill threshold.
        #
        # We replace SimpleImputer + StandardScaler with hand-rolled float32
        # equivalents that operate in-place on Xp. Mathematically identical
        # to sklearn (median imputation + zero-mean unit-variance scaling)
        # — only the dtype and the elimination of the intermediate copy
        # differ.
        nan_mask = np.isnan(Xp)
        col_medians = np.nanmedian(Xp, axis=0).astype(np.float32, copy=False)

        # Vectorised imputation: lookup the right median for every NaN
        # position in one np.where + fancy-indexing assignment, avoiding the
        # 3272-iteration Python loop the earlier v9 used.
        if nan_mask.any():
            nan_rows, nan_cols = np.where(nan_mask)
            Xp[nan_rows, nan_cols] = col_medians[nan_cols]
            del nan_rows, nan_cols
        del nan_mask
        gc.collect()

        # Save fit statistics for predict()
        self._impute_medians = col_medians

        # Standardise: subtract mean, divide by std. All in-place float32.
        col_means = Xp.mean(axis=0, dtype=np.float32)
        col_stds  = Xp.std(axis=0, dtype=np.float32)
        # Guard against zero-variance columns (already filtered in Step 0,
        # but be defensive).
        col_stds[col_stds < 1e-8] = 1.0

        Xp -= col_means        # in-place subtract
        Xp /= col_stds         # in-place divide

        self._scale_mean = col_means
        self._scale_std  = col_stds

        # Final safety: replace any inf/NaN from numerical glitches, then
        # clip to ±clip_sigma. Both in-place.
        np.nan_to_num(Xp, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        np.clip(Xp, -self.clip_sigma, self.clip_sigma, out=Xp)
        gc.collect()

        logger.info("  MLP: preprocessing done (float32, no sklearn copy). "
                    "Xp shape=%s, dtype=%s, RAM≈%.1f GB",
                    Xp.shape, Xp.dtype, Xp.nbytes / 1e9)

        # ── Step 1b (v8): Standardise target — critical for small targets ────
        # Without this, MLP must learn to squash O(1) output activations down
        # to O(1e-4) target values, which often fails within the patience
        # window and produces predictions far off scale (catastrophic R²).
        self._y_mean = float(y.mean())
        self._y_std  = max(float(y.std()), 1e-12)
        yp = ((y - self._y_mean) / self._y_std).astype(np.float32)
        logger.info("  MLP: target standardisation — mean=%.4e std=%.4e",
                    self._y_mean, self._y_std)

        # ── Step 2: Temporal val split (last val_frac) ────────────────────────
        n = len(Xp)
        n_val = max(int(n * self.val_frac), 1000)
        n_val = max(n_val, 2)  # BatchNorm needs ≥2 in val pass
        n_fit = n - n_val

        # Sanity: with drop_last=True, n_fit must accommodate at least 1 batch.
        # Otherwise we'd silently iterate over zero batches and produce garbage.
        if n_fit < self.batch_size:
            raise ValueError(
                f"n_fit={n_fit} < batch_size={self.batch_size}; "
                f"either reduce batch_size or increase max_train_samples."
            )

        # v9: GPU-RESIDENT training tensors. With max_train_samples=300k and
        # 3349 features × 4 bytes = ~4 GB, this fits comfortably in 49 GB
        # VRAM and eliminates per-batch CPU→GPU transfer overhead.
        # Validation tensors also live on GPU (was already the case in v8).
        #
        # Memory-conscious: upload one slice at a time, then drop its source
        # numpy reference so the RAM peak doesn't double during transfer.
        Xf_gpu = torch.from_numpy(np.ascontiguousarray(Xp[:n_fit])).to(
            device, non_blocking=False)  # blocking to ensure RAM is freed
        yf_gpu = torch.from_numpy(np.ascontiguousarray(yp[:n_fit])).to(
            device, non_blocking=False)
        Xv_gpu = torch.from_numpy(np.ascontiguousarray(Xp[n_fit:])).to(
            device, non_blocking=False)
        yv_gpu = torch.from_numpy(np.ascontiguousarray(yp[n_fit:])).to(
            device, non_blocking=False)

        # Free large numpy arrays — the GPU tensors hold what we need.
        # X was already freed after impute (Step 1). y still referenced via yp.
        del Xp, yp, y
        gc.collect()

        # ── Step 3: Build model ───────────────────────────────────────────────
        input_dim = Xf_gpu.shape[1]
        layers = []
        prev = input_dim
        for h in self.hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(),
                       nn.Dropout(self.dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self._model = nn.Sequential(*layers).to(device)

        opt = torch.optim.AdamW(self._model.parameters(), lr=self.lr,
                                 weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        crit = nn.MSELoss()

        # v9: AMP / Mixed Precision — only on CUDA.
        amp_enabled = self.use_amp and device == "cuda"
        # New API (PyTorch ≥2.4): torch.amp.GradScaler("cuda", ...)
        # Falls back to the old one if running on older PyTorch.
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        except (AttributeError, TypeError):
            scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

        # ── Step 4: Train with early stopping + AMP + OOM retry ──────────────
        current_bs = self.batch_size
        attempts_left = self.oom_retries

        while True:  # OOM-retry loop: halves batch_size and retries on CUDA OOM
            try:
                best_vl, best_ep, best_state = float("inf"), 0, None
                # Use a CUDA-side generator for shuffle so randperm runs on GPU
                gen = (torch.Generator(device=device).manual_seed(self.seed)
                       if device == "cuda" else
                       torch.Generator().manual_seed(self.seed))

                for ep in range(self.epochs):
                    self._model.train()
                    perm = torch.randperm(n_fit, device=device, generator=gen)

                    # drop_last semantics: only full batches
                    n_full = (n_fit // current_bs) * current_bs
                    for i in range(0, n_full, current_bs):
                        idx = perm[i:i + current_bs]
                        xb = Xf_gpu[idx]
                        yb = yf_gpu[idx]
                        opt.zero_grad(set_to_none=True)
                        with (torch.amp.autocast("cuda", enabled=amp_enabled) if hasattr(torch.amp, "autocast") else torch.cuda.amp.autocast(enabled=amp_enabled)):
                            pred = self._model(xb).squeeze(-1)
                            loss = crit(pred, yb)
                        scaler.scale(loss).backward()
                        scaler.step(opt)
                        scaler.update()
                    sched.step()

                    # validation
                    self._model.eval()
                    with torch.no_grad():
                        with (torch.amp.autocast("cuda", enabled=amp_enabled) if hasattr(torch.amp, "autocast") else torch.cuda.amp.autocast(enabled=amp_enabled)):
                            vl = crit(self._model(Xv_gpu).squeeze(-1),
                                      yv_gpu).item()

                    if self.epoch_log:
                        logger.info("    MLP seed=%d ep=%2d/%d  val_loss=%.6f  "
                                    "best_ep=%d", self.seed, ep + 1,
                                    self.epochs, vl, best_ep + 1)

                    if vl < best_vl:
                        best_vl, best_ep = vl, ep
                        best_state = {
                            k: v.detach().cpu().clone()
                            for k, v in self._model.state_dict().items()
                        }
                    if ep - best_ep >= self.patience:
                        break

                # success — break out of OOM-retry loop
                break

            except torch.cuda.OutOfMemoryError as e:
                if attempts_left <= 0 or current_bs <= 256:
                    logger.error("  MLP OOM and no retries left "
                                 "(bs=%d, attempts_left=%d). Re-raising.",
                                 current_bs, attempts_left)
                    raise
                new_bs = max(256, current_bs // 2)
                logger.warning("  MLP OOM at batch_size=%d → retrying with %d "
                               "(retries left: %d). Error: %s",
                               current_bs, new_bs, attempts_left, e)
                current_bs = new_bs
                attempts_left -= 1
                # clean up partial state
                if device == "cuda":
                    torch.cuda.empty_cache()

        if best_state:
            self._model.load_state_dict(best_state)
        self._model.eval()

        # Free training-only tensors. Keep model + scalers on GPU/host.
        del Xf_gpu, yf_gpu, Xv_gpu, yv_gpu
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

        logger.debug("MLP fit: best_epoch=%d, best_val_loss=%.6f (scaled), "
                     "input_dim=%d, y_mean=%.4e, y_std=%.4e, final_bs=%d",
                     best_ep, best_vl, input_dim, self._y_mean, self._y_std,
                     current_bs)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Batched prediction, then inverse-standardise and clip.

        v9 (final): manual float32 preprocessing matching fit(). The 1.7M-row
        test set is processed as a single pass (the manual float32 path needs
        only ~6 GB for the contiguous copy + 5.5 GB for the NaN mask, which
        fits when fit() is not holding its own intermediates).

        The earlier 200k-chunked predict was tried and reverted: it was
        consistently slower because the gc.collect() per chunk and the
        repeated np.where allocations outweighed the memory-safety benefit
        on this workload. NaN imputation is vectorised.
        """
        import torch

        X_subset = X[:, self._valid_cols]
        # Materialise as float32 contiguous (input may be a view).
        Xp = np.ascontiguousarray(X_subset, dtype=np.float32)
        del X_subset
        gc.collect()

        # Vectorised NaN imputation using train-fold medians.
        nan_mask = np.isnan(Xp)
        if nan_mask.any():
            nan_rows, nan_cols = np.where(nan_mask)
            Xp[nan_rows, nan_cols] = self._impute_medians[nan_cols]
            del nan_rows, nan_cols
        del nan_mask
        gc.collect()

        # Standardise (in-place) with train-fold mean / std.
        Xp -= self._scale_mean
        Xp /= self._scale_std

        # Safety + clip (in-place)
        np.nan_to_num(Xp, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        np.clip(Xp, -self.clip_sigma, self.clip_sigma, out=Xp)

        self._model.eval()
        n = len(Xp)
        bs = self.predict_batch_size

        # Guard: avoid degenerate batch_size=1 final batch (BatchNorm edge
        # case in eval is technically safe, but explicit is better).
        if n > bs and (n % bs) == 1:
            bs = bs - 1

        amp_enabled = self.use_amp and self._device == "cuda"
        out_scaled = np.empty(n, dtype=np.float64)
        with torch.no_grad():
            for i in range(0, n, bs):
                batch = torch.from_numpy(Xp[i:i + bs]).to(
                    self._device, non_blocking=True)
                with (torch.amp.autocast("cuda", enabled=amp_enabled)
                      if hasattr(torch.amp, "autocast")
                      else torch.cuda.amp.autocast(enabled=amp_enabled)):
                    p = self._model(batch).squeeze(-1)
                # Slice [i:i+bs] auto-clips against array length, so the
                # final partial batch broadcasts correctly even when the
                # source numpy is shorter than bs.
                out_scaled[i:i + bs] = p.float().cpu().numpy().astype(np.float64)

        # v8: inverse-standardise (scaled → original units)
        out = out_scaled * self._y_std + self._y_mean

        # v8: clip predictions at ±clip_pred_sigma × y_std around y_mean.
        bound = self.clip_pred_sigma * self._y_std
        out = np.clip(out, self._y_mean - bound, self._y_mean + bound)
        return out


def make_mlp_model_fn(**kw):
    """Return model_fn(seed) → MLPWrapper. Matches cv_engine contract."""
    def model_fn(seed: int) -> MLPWrapper:
        return MLPWrapper(seed=seed, **kw)
    return model_fn


# ==============================================================================
# Run experiments
# ==============================================================================

def run_mlp_experiments(
    assets=("btc", "eth"), targets=("ret_15s", "ret_1s"), n_folds=5,
    seeds=(42, 123, 999), max_hours=None,
    torch_threads=8, max_train_samples=500_000,
    force_rerun=False, batch_size=8192,
    clip_pred_sigma=10.0,
    use_amp=True, epoch_log=False, oom_retries=2,
    **kw,
):
    import torch
    torch.set_num_threads(torch_threads)
    logger.info("PyTorch threads limited to %d", torch_threads)
    logger.info("max_train_samples per fold: %d", max_train_samples)
    logger.info("batch_size: %d  (DataLoader removed in v9, data on GPU)",
                batch_size)
    logger.info("AMP mixed precision: %s", use_amp)
    logger.info("Prediction clip sigma: %.1f", clip_pred_sigma)
    logger.info("Force rerun: %s", force_rerun)
    logger.info("CUDA available: %s  (device: %s)",
                torch.cuda.is_available(),
                torch.cuda.get_device_name(0) if torch.cuda.is_available()
                else "cpu")
    from common.data_loader import load_dataset
    from common.cv_engine import expanding_window_folds, run_cv
    from common.config import RESULTS_DIR
    import pandas as pd

    out_dir = RESULTS_DIR / "mlp"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_summaries = []
    skipped = []
    failed = []

    jobs = [(a, t) for a in assets for t in targets]
    t_run_start = time.time()

    for asset, tgt in jobs:
        result_path = out_dir / f"mlp_{asset}_{tgt}_folds.csv"

        if result_path.exists() and not force_rerun:
            logger.info("Skipping MLP %s/%s — result exists: %s",
                        asset, tgt, result_path.name)
            skipped.append((asset, tgt))
            continue

        logger.info("━━ MLP  %s/%s ━━", asset.upper(), tgt)
        t_target_start = time.time()

        # v9: wrap the WHOLE per-target body so that a crash on one
        # target does not abort the remaining 31. We always clean up
        # GPU/CPU memory afterwards in a finally block.
        X = y = info = result = None
        try:
            X, y, info, feat_names = load_dataset(
                target=tgt, asset=asset, profile="tree", max_hours=max_hours)

            model_fn = make_mlp_model_fn(
                max_train_samples=max_train_samples,
                batch_size=batch_size,
                clip_pred_sigma=clip_pred_sigma,
                use_amp=use_amp,
                epoch_log=epoch_log,
                oom_retries=oom_retries,
                **kw,
            )
            folds = expanding_window_folds(n_samples=len(X), n_folds=n_folds)

            result = run_cv(
                X=X, y=y, model_fn=model_fn, folds=folds,
                seeds=list(seeds), feature_names=feat_names,
                horizon=tgt, asset=asset, model_name="MLP",
            )

            if result.n_folds == 0:
                logger.error("MLP %s/%s: ALL folds failed — skipping.",
                             asset, tgt)
                failed.append((asset, tgt, "all_folds_failed"))
            else:
                all_summaries.append(result.summary_dict())
                result.fold_table().to_csv(result_path, index=False)
                dt = time.time() - t_target_start
                logger.info("MLP %s/%s done in %.1f min (%d/%d folds)",
                            asset, tgt, dt / 60.0,
                            result.n_folds, n_folds)

        except Exception as e:
            logger.exception("MLP %s/%s CRASHED — continuing with next "
                             "target. Error: %s", asset, tgt, e)
            failed.append((asset, tgt, str(e)[:200]))

        finally:
            # Aggressive cleanup so the next target starts clean.
            del X, y, info, result
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        elapsed_h = (time.time() - t_run_start) / 3600.0
        logger.info("  Run elapsed: %.2f h", elapsed_h)

    # Reconstruct summary from disk (covers freshly trained + previously skipped)
    fold_files = sorted(out_dir.glob("mlp_*_folds.csv"))
    final_summaries = []
    for f in fold_files:
        try:
            df = pd.read_csv(f)
            stem = f.stem
            assert stem.startswith("mlp_") and stem.endswith("_folds")
            tag = stem[len("mlp_"):-len("_folds")]
            asset_, target_ = tag.split("_", 1)
            r2s = df["r2"].astype(float)
            final_summaries.append({
                "model":        "MLP",
                "asset":        asset_,
                "horizon":      target_,
                "n_folds":      len(df),
                "r2_mean":      round(float(r2s.mean()), 6),
                "r2_std":       round(float(r2s.std()), 6),
                "r2_min":       round(float(r2s.min()), 6),
                "r2_max":       round(float(r2s.max()), 6),
                "n_positive":   int((r2s > 0).sum()),
                "mse_mean":     round(float(df["mse"].astype(float).mean()), 10),
                "mae_mean":     round(float(df["mae"].astype(float).mean()), 8),
                "dir_acc_mean": round(float(df["dir_acc"].astype(float).mean()), 4),
                "ic_mean":      round(float(df["ic"].astype(float).mean()), 6),
            })
        except Exception as e:
            logger.warning("Could not reconstruct summary from %s: %s", f.name, e)

    if final_summaries:
        results_df = pd.DataFrame(final_summaries)
        results_df.to_csv(out_dir / "mlp_summary.csv", index=False)
        print(f"\n{'='*60}\n  MLP Summary ({len(final_summaries)} pairs)\n{'='*60}")
        print(results_df.to_string(index=False, float_format="%.6f"))
        if skipped:
            print(f"\nSkipped (already done): {len(skipped)} pairs")
        if failed:
            print(f"\nFailed targets: {len(failed)}")
            for a, t, err in failed:
                print(f"   {a}/{t}: {err}")
        total_h = (time.time() - t_run_start) / 3600.0
        print(f"\nTotal wall-clock: {total_h:.2f} h")
        return results_df

    if failed:
        logger.error("No MLP results produced. Failed: %s", failed)
    else:
        logger.error("No MLP results produced.")
    return pd.DataFrame()


# ==============================================================================
# CLI
# ==============================================================================

def main():
    p = argparse.ArgumentParser(description="Phase 4C: MLP (PyTorch) — v9")
    p.add_argument("--asset", choices=["btc", "eth", "both"], default="both")
    p.add_argument("--targets", nargs="+", default=None,
                   help="Flat target tokens, e.g. 'ret_15s mfe_60s mae_300s'.")
    p.add_argument("--target-family", nargs="+", default=None,
                   choices=["ret", "mfe", "mae"])
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--max-hours", type=int, default=None)
    p.add_argument("--max-train-samples", type=int, default=500_000)
    p.add_argument("--torch-threads", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=8192,
                   help="Train batch size. v9 default 8192 (was 4096); "
                        "A6000 had headroom.")
    p.add_argument("--clip-pred-sigma", type=float, default=10.0,
                   help="Prediction clipping in units of y_train_std (default: 10).")
    p.add_argument("--no-amp", action="store_true",
                   help="Disable mixed-precision (AMP). Slower but more "
                        "deterministic. AMP is on by default on CUDA.")
    p.add_argument("--epoch-log", action="store_true",
                   help="Log val_loss every epoch. Verbose but useful when "
                        "diagnosing slow runs.")
    p.add_argument("--oom-retries", type=int, default=2,
                   help="Number of automatic batch_size halvings on CUDA OOM "
                        "before giving up on a fold.")
    p.add_argument("--force-rerun", action="store_true")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    a = p.parse_args()
    logging.basicConfig(level=getattr(logging, a.log_level),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S", stream=sys.stdout)
    assets = ("btc", "eth") if a.asset == "both" else (a.asset,)

    if a.seeds:
        seeds = tuple(a.seeds)
    else:
        from common.config import DEFAULT_SEEDS
        seeds = tuple(DEFAULT_SEEDS)

    from common.config import all_targets
    if a.targets:
        targets = tuple(a.targets)
    elif a.target_family:
        targets = tuple(all_targets(tuple(a.target_family)))
    else:
        targets = tuple(all_targets())
    logger.info("Targets (%d): %s", len(targets), list(targets))

    run_mlp_experiments(
        assets=assets, targets=targets, n_folds=a.n_folds,
        seeds=seeds, max_hours=a.max_hours,
        torch_threads=a.torch_threads,
        max_train_samples=a.max_train_samples,
        force_rerun=a.force_rerun,
        batch_size=a.batch_size,
        clip_pred_sigma=a.clip_pred_sigma,
        use_amp=not a.no_amp,
        epoch_log=a.epoch_log,
        oom_retries=a.oom_retries,
        epochs=a.epochs, patience=a.patience)

if __name__ == "__main__": main()