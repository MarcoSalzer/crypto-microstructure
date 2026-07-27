#!/usr/bin/env python3
"""
s5_s6_feature_importance.py  (v4 — feature_keep driven)
========================================================
Feature importance on the merged S5+S6 corpus.

Feature selection chain
-----------------------
The canonical post-reduction feature list lives in feature_keep.csv.
Selection here is a single read from that file plus a safety-net prefix
filter against forward-looking columns. No catalog allowlist, no drop
list, no EXCLUDE_GROUPS — these decisions are already encoded in
feature_keep.

  1. feature_keep.csv: rows with type=feature AND use_tree=True.
     This is the canonical list after hard-drops (LWP structural,
     universal-weak FI) and includes all surviving microstructure,
     context, and cross-asset features.
  2. EXCLUDE_PREFIXES: safety net against forward-looking / look-ahead
     columns that should never appear as features.
  3. Schema intersection: only columns actually present in parquets.

Data source
-----------
  data_storage/s6_features_s5_full/merged_*_*.parquet

Targets carry asset suffix (ret_fwd_15s_btc, ret_fwd_15s_eth, ...).
--assets selects which asset's returns to predict; all surviving
features (S5 single-asset + S6 cross-asset) act as inputs.

Changes vs v3
-------------
  - feature_keep.csv replaces catalog allowlist + drop list combo
  - EXCLUDE_GROUPS removed (Trend/VolumeProfile/Level etc. now in scope)
  - load_drop_list removed; --drop-list flag removed
  - get_signal_features_from_schema simplified (single allowlist)
  - --feature-keep flag added for override / R2 runs

Usage
-----
    # Round 1 (with current feature_keep):
    nohup python -u -m selection.s5_s6_feature_importance \\
        --null-baseline --sample-rows 1500000 > /dev/null 2>&1 &

    # Round 2 (after universal-weak drops applied to feature_keep):
    nohup python -u -m selection.s5_s6_feature_importance \\
        --null-baseline --sample-rows 1500000 \\
        --output-dir results/selection/results/s5_s6_feature_importance_full_r2 \\
        > /dev/null 2>&1 &

    # Logs:
    tail -f results/selection/logs/s5_s6_feature_importance.log

Outputs (in --output-dir, default s5_s6_feature_importance_full/)
-----------------------------------------------------------------
    Per target:
        {asset}_{horizon}_importance.csv        — multi-metric importance
        {asset}_{horizon}_cv_scores.csv         — per-fold MAE / R²
    Cross-target:
        {asset}_multi_target_consensus.csv      — R²-weighted consensus
        {asset}_consensus_{group}.csv           — per horizon-group consensus
        {asset}_never_important_features.csv    — bottom 20% across ALL targets
    Optional:
        {asset}_{horizon}_null_importance.csv   — null baseline gains
        {asset}_{horizon}_shap_importance.csv   — SHAP-based importance
"""

import os
from common.paths import REDUCTION_DIR
import signal

import argparse
import gc
import glob
import logging
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


# ─── Configuration ────────────────────────────────────────────────────────────

BASE_DIR        = str(REDUCTION_DIR)
FEATURE_KEEP    = os.path.join(BASE_DIR, "feature_keep.csv")
DATA_DIR        = "data_storage/s6_features_s5_full"
DEFAULT_OUTPUT  = os.path.join(BASE_DIR, "results", "s5_s6_feature_importance_full")
LOG_DIR         = os.path.join(BASE_DIR, "logs")

ASSETS          = ["btc", "eth"]
HORIZONS        = ["1s", "5s", "15s", "30s", "60s", "120s", "300s", "900s"]

# MFE/MAE targets (Maximum Favorable/Adverse Excursion)
MFE_MAE_HORIZONS = ["15s", "60s", "300s", "900s"]
USABILITY_COL   = "data_usability_flag"  # suffixed as data_usability_flag_{asset}

# 300k rows for consensus stability. Plan 7.7 states
# R²-weighted consensus is stable above ~100k rows; 300k provides safety margin
# without excessive permutation-importance compute cost.
DEFAULT_SAMPLE  = 300_000
N_FOLDS         = 5
DEFAULT_SEEDS   = [42, 123, 999]

# Safety-net prefix filter against forward-looking columns.
# After merge, all single-asset cols carry _{asset} suffix, so these
# prefixes match the un-suffixed base name.
# Most of these should already be excluded by feature_keep (type=meta or
# type=target) — this is a belt-and-braces filter against schema drift.
EXCLUDE_PREFIXES = (
    # ── Forward-looking / look-ahead ─────────────────────────────────────────
    "ret_",                # all forward returns + raw returns
    "ret_fwd_",            # forward return targets
    "ret_mid_",            # mid-price returns
    "ca_ret_fwd_spread",   # S6 cross-asset return spreads (looks ahead into target)
    "mfe_fwd_",            # Maximum Favorable Excursion (forward-looking)
    "mae_fwd_",            # Maximum Adverse Excursion (forward-looking)
    "rv_fwd_",             # Realized volatility forward (forward-looking)
    "tbl_",                # Triple Barrier Labels
    "barrier_",            # barrier labels
    "label_",              # any labeling column
    # ── Data quality / meta — no market signal ────────────────────────────
    "data_health",
    "health_reason_code",
    "data_usability",
    "usability_",
    "l2_coverage",
    "lob50_health",
    "trades_coverage",
    "depth_lobdeep",
    "__index_level_",      # pandas/pyarrow parquet index artifact
)

# Horizon groups for group-level consensus (same as v4)
# Horizon groups for group-level consensus
HORIZON_GROUPS = {
    "ultra_short": ["ret_fwd_1s", "ret_fwd_5s", "ret_fwd_15s"],
    "short":       ["ret_fwd_30s", "ret_fwd_60s", "ret_fwd_120s"],
    "medium":      ["ret_fwd_300s", "ret_fwd_900s"],
    "mfe":         ["mfe_fwd_15s_bps", "mfe_fwd_60s_bps",
                    "mfe_fwd_300s_bps", "mfe_fwd_900s_bps"],
    "mae":         ["mae_fwd_15s_bps", "mae_fwd_60s_bps",
                    "mae_fwd_300s_bps", "mae_fwd_900s_bps"],
}

# LightGBM hyperparameters — identical to v4
LGBM_PARAMS = {
    "objective":         "regression",
    "metric":            "mae",
    "n_estimators":      500,
    "num_leaves":        20,
    "max_depth":         6,
    "learning_rate":     0.05,
    "subsample":         0.8,
    "subsample_freq":    1,
    "colsample_bytree":  0.4,
    "min_child_samples": 100,
    "reg_alpha":         0.1,
    "reg_lambda":        1.0,
    "n_jobs":            -1,
    "verbose":           -1,
}

PERM_N_REPEATS = 5


# ─── Logging ──────────────────────────────────────────────────────────────────

def setup_logging():
    os.makedirs(LOG_DIR, exist_ok=True)
    log = logging.getLogger("s5_s6_feat_imp")
    log.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(
        os.path.join(LOG_DIR, "s5_s6_feature_importance.log"),
        mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    if not log.handlers:
        log.addHandler(fh)
        log.addHandler(ch)
    return log


# ─── Feature Selection ───────────────────────────────────────────────────────

def _load_feature_keep_columns(feature_keep_path: str, log) -> set | None:
    """
    Load feature_keep.csv and return the set of column names that are
    tree-usable feature signals (type=feature AND use_tree=True).

    Returns None if file missing — caller should treat as fatal since
    feature_keep is the canonical source of truth.
    """
    if not os.path.exists(feature_keep_path):
        log.error("feature_keep not found at %s — cannot proceed.",
                  feature_keep_path)
        return None

    fk = pd.read_csv(feature_keep_path)
    required = {"column", "type", "use_tree"}
    missing = required - set(fk.columns)
    if missing:
        log.error("feature_keep missing required columns %s — cannot proceed.",
                  missing)
        return None

    n_total = len(fk)
    fk_feat = fk[fk["type"] == "feature"].copy()
    n_feat = len(fk_feat)
    fk_tree = fk_feat[fk_feat["use_tree"] == True].copy()
    n_tree = len(fk_tree)

    log.info("feature_keep: %d rows -> %d features -> %d tree-usable",
             n_total, n_feat, n_tree)

    cols = set(fk_tree["column"].tolist())
    return cols


def get_signal_features_from_schema(files: list,
                                    log,
                                    feature_keep_cols: set) -> list:
    """
    Intersect parquet schema with feature_keep allowlist, then apply
    EXCLUDE_PREFIXES safety-net filter. Returns sorted list of column
    names to use as features.

    feature_keep_cols already encodes the post-reduction allowlist
    (LWP-reduced, drop-list applied, use_tree=True). The prefix filter
    is a defence-in-depth check against schema drift only.
    """
    if not files:
        log.error("No files found to infer schema from.")
        return []

    schema_cols = set(pq.read_schema(files[0]).names)

    def _is_prefix_excluded(col: str) -> bool:
        # Strip asset suffix to check against EXCLUDE_PREFIXES on base name
        base = col
        for a in ASSETS:
            if col.endswith(f"_{a}"):
                base = col[: -(len(a) + 1)]
                break
        return any(base.startswith(px) for px in EXCLUDE_PREFIXES)

    candidates = schema_cols & feature_keep_cols
    n_keep_not_in_schema = len(feature_keep_cols - schema_cols)
    n_schema_not_in_keep = len(schema_cols - feature_keep_cols)

    n_prefix_excluded = 0
    features = []
    for c in sorted(candidates):
        if _is_prefix_excluded(c):
            n_prefix_excluded += 1
            continue
        features.append(c)

    log.info("Schema cols: %d  ∩  feature_keep: %d  →  candidates: %d",
             len(schema_cols), len(feature_keep_cols), len(candidates))
    log.info("  feature_keep cols not in schema: %d", n_keep_not_in_schema)
    log.info("  schema cols not in feature_keep: %d (mostly targets/meta)",
             n_schema_not_in_keep)
    log.info("  prefix-excluded (safety net):    %d", n_prefix_excluded)
    log.info("Final signal features: %d", len(features))
    return features


# ─── Data Sampling ────────────────────────────────────────────────────────────

def stream_sample_numpy(asset: str, features: list, targets: list,
                        max_rows: int, log):
    """
    Stream-sample rows from merged parquets directly into pre-allocated
    numpy arrays.  Returns (X, targets_dict, feature_names) or None.

    NO pd.concat, NO large DataFrame — peak RAM ≈ 1 × final arrays only.
    """
    pattern = os.path.join(DATA_DIR, "merged_*_*.parquet")
    files = sorted(glob.glob(pattern))
    if not files:
        log.error("No merged parquet files found in %s", DATA_DIR)
        return None

    usability_col = f"{USABILITY_COL}_{asset}"
    asset_targets = [f"{t}_{asset}" for t in targets]

    log.info("Sampling %s rows from %d files (asset=%s)...",
             f"{max_rows:,}", len(files), asset.upper())

    # ── Pass 1: count rows, resolve columns ───────────────────────────────
    total_rows = 0
    file_rows = []
    for f in files:
        try:
            n = pq.read_metadata(f).num_rows
        except Exception:
            n = 0
        file_rows.append(n)
        total_rows += n

    sample_rate = min(1.0, max_rows / total_rows) if total_rows > 0 else 1.0
    log.info("  Total: %s rows across %d files, sample rate: %.4f",
             f"{total_rows:,}", len(files), sample_rate)

    # Resolve which feature columns actually exist in the files
    schema_cols_first = set(pq.read_schema(files[0]).names)
    feat_cols = [f for f in features if f in schema_cols_first]
    tgt_cols  = [t for t in asset_targets if t in schema_cols_first]
    if not feat_cols:
        log.error("  No feature columns found in parquet schema")
        return None

    n_feat = len(feat_cols)
    n_tgt  = len(tgt_cols)
    feat_set = set(feat_cols)
    cols_needed = list(set(feat_cols + tgt_cols + [usability_col]))

    # Build column-name → index maps for fast row copying
    feat_idx = {c: i for i, c in enumerate(feat_cols)}
    tgt_idx  = {c: i for i, c in enumerate(tgt_cols)}

    log.info("  Features: %d, Targets: %d", n_feat, n_tgt)

    # ── Pass 2: pre-allocate and fill ─────────────────────────────────────
    # Over-allocate slightly; we'll truncate at the end
    alloc_rows = int(max_rows * 1.05) + 1000
    X    = np.zeros((alloc_rows, n_feat), dtype=np.float32)
    Y    = np.full((alloc_rows, n_tgt), np.nan, dtype=np.float64)
    cursor = 0

    rng = np.random.RandomState(DEFAULT_SEEDS[0])

    for idx, (filepath, n_rows) in enumerate(zip(files, file_rows)):
        if cursor >= max_rows:
            break
        n_sample = max(1, int(n_rows * sample_rate))
        try:
            schema_cols = set(pq.read_schema(filepath).names)
            load_cols = [c for c in cols_needed if c in schema_cols]

            chunk = pd.read_parquet(filepath, columns=load_cols)
            if chunk.columns.duplicated().any():
                chunk = chunk.loc[:, ~chunk.columns.duplicated(keep="first")]

            # Usability gate
            if usability_col in chunk.columns:
                chunk = chunk[chunk[usability_col] == 1]

            if len(chunk) == 0:
                del chunk
                continue
            if len(chunk) > n_sample:
                chunk = chunk.sample(n=n_sample, random_state=rng)

            nr = len(chunk)
            end = cursor + nr

            # Copy feature columns into X row by row of columns
            for col in chunk.columns:
                if col in feat_idx:
                    vals = chunk[col].values
                    np.nan_to_num(vals, copy=False, nan=0.0)
                    X[cursor:end, feat_idx[col]] = vals.astype(np.float32)
                elif col in tgt_idx:
                    Y[cursor:end, tgt_idx[col]] = chunk[col].values.astype(np.float64)

            cursor = end
            del chunk

        except Exception as e:
            log.warning("  Error reading %s: %s", os.path.basename(filepath), e)

        if (idx + 1) % 50 == 0 or idx == len(files) - 1:
            log.info("  [%d/%d] %s rows", idx + 1, len(files), f"{cursor:,}")

    if cursor == 0:
        log.error("  No rows sampled")
        return None

    # Truncate to actual size
    X = X[:cursor]
    Y = Y[:cursor]

    log.info("  Sampled: %s rows × %d features (%.1f MB)",
             f"{cursor:,}", n_feat, X.nbytes / 1e6)

    # Build targets dict
    targets_dict = {}
    for i, t in enumerate(tgt_cols):
        col_data = Y[:, i]
        n_valid = np.isfinite(col_data).sum()
        log.info("  Target %s: %s valid rows (%.1f%%)",
                 t, f"{n_valid:,}", 100 * n_valid / cursor)
        targets_dict[t] = col_data
    del Y

    return X, targets_dict, feat_cols


# ─── Model Training ───────────────────────────────────────────────────────────
# Model-training functions (gain-over-null importance) below.

def detect_engine(log):
    try:
        import lightgbm as lgb
        log.info("Engine: LightGBM %s", lgb.__version__)
        return "lgbm"
    except ImportError:
        log.info("Engine: sklearn GBR (LightGBM not installed)")
        return "sklearn"


def train_fold_single_seed(X_train, y_train, X_test, y_test, feature_names,
                           engine, seed, log):
    from sklearn.metrics import mean_absolute_error, r2_score
    if engine == "lgbm":
        import lightgbm as lgb
        params = {**LGBM_PARAMS, "random_state": seed}
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(0)],
        )
        n_trees = model.best_iteration_ or LGBM_PARAMS["n_estimators"]
        gain  = model.booster_.feature_importance(importance_type="gain").astype(np.float64)
        split = model.booster_.feature_importance(importance_type="split").astype(np.float64)
    else:
        from sklearn.ensemble import GradientBoostingRegressor
        model = GradientBoostingRegressor(
            n_estimators=150, max_depth=5, learning_rate=0.05,
            subsample=0.8, max_features=0.4, min_samples_leaf=100,
            random_state=seed,
        )
        model.fit(X_train, y_train)
        gain = model.feature_importances_.astype(np.float64)
        split = gain.copy()
        n_trees = 150

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)
    return model, gain, split, mae, r2, n_trees


def compute_permutation_importance(model, X_test, y_test, feature_names,
                                   n_repeats, rng, log, n_jobs=8):
    from sklearn.inspection import permutation_importance as sklearn_perm_imp
    from sklearn.metrics import mean_absolute_error

    log.info("    Permutation importance: %d features × %d repeats "
             "(n_jobs=%d)...",
             len(feature_names), n_repeats, n_jobs)
    t0 = time.time()

    result = sklearn_perm_imp(
        model, X_test, y_test,
        n_repeats=n_repeats,
        scoring="neg_mean_absolute_error",
        random_state=rng,
        n_jobs=n_jobs,
    )
    # importances_mean is the mean decrease in score (negative MAE),
    # so higher = more important. Negate to get MAE increase.
    perm_imp = -result.importances_mean

    log.info("    Permutation importance done in %.1fs", time.time() - t0)
    return perm_imp


def compute_shap_importance(model, X_test, feature_names, engine, log):
    try:
        import shap
    except ImportError:
        log.warning("    SHAP not installed — skipping (pip install shap)")
        return None

    log.info("    Computing SHAP values on test set (%s rows)...", f"{len(X_test):,}")
    t0 = time.time()
    try:
        if engine == "lgbm":
            explainer = shap.TreeExplainer(model)
        else:
            max_bg = min(1000, len(X_test))
            explainer = shap.TreeExplainer(model, X_test[:max_bg])
        shap_values = explainer.shap_values(X_test)
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        log.info("    SHAP done in %.1fs", time.time() - t0)
        del shap_values, explainer
        gc.collect()
        return mean_abs_shap
    except Exception as e:
        log.warning("    SHAP failed: %s", e)
        return None


def train_null_baseline(X_train, y_train, X_test, y_test, engine, seed, log):
    rng = np.random.RandomState(seed + 1000)
    y_shuffled = y_train.copy()
    rng.shuffle(y_shuffled)
    log.info("    Null baseline: training on shuffled target...")
    t0 = time.time()
    if engine == "lgbm":
        import lightgbm as lgb
        params = {**LGBM_PARAMS, "random_state": seed, "n_estimators": 100}
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train, y_shuffled,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.early_stopping(30, verbose=False),
                       lgb.log_evaluation(0)],
        )
        null_gain = model.booster_.feature_importance(importance_type="gain").astype(np.float64)
    else:
        from sklearn.ensemble import GradientBoostingRegressor
        model = GradientBoostingRegressor(
            n_estimators=80, max_depth=5, learning_rate=0.05,
            subsample=0.8, max_features=0.4, min_samples_leaf=100,
            random_state=seed,
        )
        model.fit(X_train, y_shuffled)
        null_gain = model.feature_importances_.astype(np.float64)
    del model
    gc.collect()
    log.info("    Null baseline done in %.1fs", time.time() - t0)
    return null_gain


# ─── CV Loop ──────────────────────────────────────────────────────────────────

def run_cv(X, y, feature_names, n_folds, seeds, engine, log,
           do_permutation=True, do_shap=False, do_null=False, perm_n_jobs=8):
    n = len(y)
    fold_size = n // (n_folds + 1)
    fold_records, score_records = [], []
    perm_records, shap_records, null_records = [], [], []

    for fold in range(n_folds):
        train_end = fold_size * (fold + 1)
        test_end  = min(train_end + fold_size, n)
        if test_end <= train_end:
            break

        X_tr, y_tr = X[:train_end], y[:train_end]
        X_te, y_te = X[train_end:test_end], y[train_end:test_end]

        log.info("  Fold %d/%d: train=%s test=%s, seeds=%s",
                 fold + 1, n_folds, f"{len(X_tr):,}", f"{len(X_te):,}", seeds)
        t_fold = time.time()

        all_gain, all_split, all_mae, all_r2 = [], [], [], []
        best_model, best_r2 = None, -np.inf

        for si, seed in enumerate(seeds):
            log.info("    Seed %d/%d (seed=%d)...", si + 1, len(seeds), seed)
            model, gain, split, mae, r2, n_trees = train_fold_single_seed(
                X_tr, y_tr, X_te, y_te, feature_names, engine, seed, log)
            log.info("      trees=%d MAE=%.8f R²=%.6f", n_trees, mae, r2)
            all_gain.append(gain); all_split.append(split)
            all_mae.append(mae);   all_r2.append(r2)
            if r2 > best_r2:
                if best_model is not None:
                    del best_model
                best_model = model
                best_r2 = r2
            else:
                del model

        gain_avg  = np.mean(all_gain, axis=0)
        split_avg = np.mean(all_split, axis=0)
        mae_avg   = np.mean(all_mae)
        r2_avg    = np.mean(all_r2)
        dt = time.time() - t_fold
        log.info("    Fold %d: MAE=%.8f R²=%.6f (avg %d seeds, %.1fs)",
                 fold + 1, mae_avg, r2_avg, len(seeds), dt)

        score_records.append({
            "fold": fold + 1, "train_rows": len(X_tr), "test_rows": len(X_te),
            "mae": round(mae_avg, 8), "r2": round(r2_avg, 6),
            "r2_std": round(np.std(all_r2), 6),
            "seconds": round(dt, 1), "n_seeds": len(seeds),
        })

        g_sum = gain_avg.sum()
        s_sum = split_avg.sum()
        gain_n  = gain_avg  / g_sum if g_sum > 0 else gain_avg
        split_n = split_avg / s_sum if s_sum > 0 else split_avg

        for i, f in enumerate(feature_names):
            fold_records.append({
                "fold": fold + 1, "feature_name": f,
                "gain_importance":  round(float(gain_n[i]),  10),
                "split_importance": round(float(split_n[i]), 10),
            })

        if do_permutation:
            perm_rng  = seeds[0] + fold  # sklearn accepts int seed
            perm_imp  = compute_permutation_importance(
                best_model, X_te, y_te, feature_names, PERM_N_REPEATS, perm_rng,
                log, n_jobs=perm_n_jobs)
            perm_sum = perm_imp.sum()
            perm_n   = perm_imp / perm_sum if perm_sum > 0 else perm_imp
            for i, f in enumerate(feature_names):
                perm_records.append({
                    "fold": fold + 1, "feature_name": f,
                    "perm_importance":   round(float(perm_n[i]),   10),
                    "perm_mae_increase": round(float(perm_imp[i]), 10),
                })

        if do_shap:
            shap_imp = compute_shap_importance(
                best_model, X_te, feature_names, engine, log)
            if shap_imp is not None:
                shap_sum = shap_imp.sum()
                shap_n   = shap_imp / shap_sum if shap_sum > 0 else shap_imp
                for i, f in enumerate(feature_names):
                    shap_records.append({
                        "fold": fold + 1, "feature_name": f,
                        "shap_importance": round(float(shap_n[i]),  10),
                        "shap_mean_abs":   round(float(shap_imp[i]), 10),
                    })

        if do_null:
            null_gain = train_null_baseline(
                X_tr, y_tr, X_te, y_te, engine, seeds[0], log)
            null_sum = null_gain.sum()
            null_n   = null_gain / null_sum if null_sum > 0 else null_gain
            for i, f in enumerate(feature_names):
                null_records.append({
                    "fold": fold + 1, "feature_name": f,
                    "null_gain": round(float(null_n[i]), 10),
                })

        del best_model, X_tr, y_tr, X_te, y_te
        gc.collect()

    return fold_records, score_records, perm_records, shap_records, null_records


# ─── Aggregation ──────────────────────────────────────────────────────────────

def aggregate_importance(fold_records, perm_records=None,
                         shap_records=None, null_records=None):
    cv_df = pd.DataFrame(fold_records)
    agg = cv_df.groupby("feature_name").agg(
        gain_mean =("gain_importance",  "mean"),
        gain_std  =("gain_importance",  "std"),
        split_mean=("split_importance", "mean"),
        split_std =("split_importance", "std"),
        n_folds   =("fold",             "count"),
    ).reset_index()
    agg["gain_stability"] = np.where(
        agg["gain_std"] > 0, agg["gain_mean"] / agg["gain_std"], np.inf)

    if perm_records:
        perm_agg = pd.DataFrame(perm_records).groupby("feature_name").agg(
            perm_mean            =("perm_importance",   "mean"),
            perm_std             =("perm_importance",   "std"),
            perm_mae_increase_mean=("perm_mae_increase","mean"),
        ).reset_index()
        agg = agg.merge(perm_agg, on="feature_name", how="left")
    else:
        agg["perm_mean"] = np.nan
        agg["perm_std"]  = np.nan
        agg["perm_mae_increase_mean"] = np.nan

    if shap_records:
        shap_agg = pd.DataFrame(shap_records).groupby("feature_name").agg(
            shap_mean=("shap_importance", "mean"),
            shap_std =("shap_importance", "std"),
        ).reset_index()
        agg = agg.merge(shap_agg, on="feature_name", how="left")
    else:
        agg["shap_mean"] = np.nan

    if null_records:
        null_agg = pd.DataFrame(null_records).groupby("feature_name").agg(
            null_gain_mean=("null_gain", "mean"),
        ).reset_index()
        agg = agg.merge(null_agg, on="feature_name", how="left")
        agg["gain_over_null"] = np.where(
            agg["null_gain_mean"] > 0,
            agg["gain_mean"] / agg["null_gain_mean"],
            np.where(agg["gain_mean"] > 0, np.inf, 0.0))
    else:
        agg["null_gain_mean"] = np.nan
        agg["gain_over_null"] = np.nan

    agg["gain_rank"]  = agg["gain_mean"].rank(ascending=False)
    agg["split_rank"] = agg["split_mean"].rank(ascending=False)
    agg["perm_rank"]  = agg["perm_mean"].rank(ascending=False, na_option="bottom")

    if perm_records:
        agg["combined_rank"] = (
            (agg["gain_rank"] + agg["split_rank"] + agg["perm_rank"]) / 3
        ).round(1)
    else:
        agg["combined_rank"] = (
            (agg["gain_rank"] + agg["split_rank"]) / 2
        ).round(1)

    return agg.sort_values("combined_rank").reset_index(drop=True)


# ─── Cross-Target Consensus (R²-weighted) ─────────────────────────────────────

def build_consensus(target_results: dict, target_r2s: dict,
                    log, targets_subset=None) -> pd.DataFrame:
    if targets_subset:
        target_results = {k: v for k, v in target_results.items() if k in targets_subset}
        target_r2s     = {k: v for k, v in target_r2s.items()     if k in targets_subset}
    if not target_results:
        return pd.DataFrame()

    raw_weights = {t: max(0.0, r2) for t, r2 in target_r2s.items()
                   if t in target_results}
    w_sum = sum(raw_weights.values())
    if w_sum == 0:
        weights = {t: 1.0 / len(target_results) for t in target_results}
    else:
        weights = {t: w / w_sum for t, w in raw_weights.items()}

    log.info("  Consensus weights: %s",
             {t: f"{w:.3f}" for t, w in sorted(weights.items())})

    all_features = set()
    for df in target_results.values():
        all_features.update(df["feature_name"])

    rows = []
    for f in all_features:
        row = {"feature_name": f}
        weighted_pctiles, raw_pctiles, gains = [], [], []

        for target, df in target_results.items():
            match = df[df["feature_name"] == f]
            if len(match) > 0:
                g   = match.iloc[0]["gain_mean"]
                r   = match.iloc[0]["combined_rank"]
                pct = 100 * (1 - r / len(df))
                row[f"gain_{target}"]   = round(g,   8)
                row[f"rank_{target}"]   = round(r,   1)
                row[f"pctile_{target}"] = round(pct, 1)
                if "perm_mean" in match.columns and pd.notna(match.iloc[0].get("perm_mean")):
                    row[f"perm_{target}"] = round(match.iloc[0]["perm_mean"], 8)
                gains.append(g)
                raw_pctiles.append(pct)
                weighted_pctiles.append(pct * weights.get(target, 0))

        row["mean_gain"]       = round(np.mean(gains),          8) if gains else 0
        row["mean_pctile"]     = round(np.mean(raw_pctiles),    1) if raw_pctiles else 0
        row["weighted_pctile"] = round(sum(weighted_pctiles),   1) if weighted_pctiles else 0
        row["min_pctile"]      = round(np.min(raw_pctiles),     1) if raw_pctiles else 0
        row["max_pctile"]      = round(np.max(raw_pctiles),     1) if raw_pctiles else 0
        row["n_targets"]       = len(gains)
        row["never_important"] = all(p < 20 for p in raw_pctiles) if raw_pctiles else True
        rows.append(row)

    consensus = pd.DataFrame(rows).sort_values(
        "weighted_pctile", ascending=False).reset_index(drop=True)
    log.info("  Consensus: %d features, never_important: %d",
             len(consensus), consensus["never_important"].sum())
    return consensus


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Feature Importance v1 — combined S5-reduced + S6 (multi-metric, multi-seed)")
    parser.add_argument("--assets",      nargs="+", default=ASSETS,
                        help="Target assets to predict returns for (default: btc eth)")
    parser.add_argument("--targets",     nargs="+", default=HORIZONS,
                        help="Horizon suffixes e.g. 15s 60s (default: all 5 horizons)")
    parser.add_argument("--sample-rows", type=int,  default=DEFAULT_SAMPLE)
    parser.add_argument("--seeds",       nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--top",         type=int,  default=50)
    parser.add_argument("--shap",           action="store_true",
                        help="Compute SHAP values (slower, more RAM)")
    parser.add_argument("--null-baseline",  action="store_true",
                        help="Compute null-importance baseline")
    parser.add_argument("--no-permutation", action="store_true",
                        help="Skip permutation importance (faster)")
    parser.add_argument("--perm-jobs", type=int, default=8,
                        help="Parallelism for sklearn permutation_importance "
                             "(default: 8). Each worker holds a copy of the "
                             "test fold (~7GB per worker for 1.5M sample). "
                             "Use -1 for all cores ONLY when nothing else runs.")
    parser.add_argument("--feature-keep",   type=str, default=FEATURE_KEEP,
                        help="Path to feature_keep.csv (default: %(default)s). "
                             "Filters to type=feature AND use_tree=True.")
    parser.add_argument("--output-dir",    type=str, default=DEFAULT_OUTPUT,
                        help="Output directory for results. Use a different dir "
                             "for Round 2 (e.g. ..._full_r2/). Default: %(default)s")
    args = parser.parse_args()

    log = setup_logging()
    t0  = time.time()

    do_perm = not args.no_permutation
    do_shap = args.shap
    do_null = args.null_baseline

    # Build full target names: ret_fwd_15s, etc. (asset suffix added per loop)
    ret_targets = [f"ret_fwd_{h}" for h in args.targets]
    mfe_targets = [f"mfe_fwd_{h}_bps" for h in MFE_MAE_HORIZONS]
    mae_targets = [f"mae_fwd_{h}_bps" for h in MFE_MAE_HORIZONS]
    all_targets = ret_targets + mfe_targets + mae_targets

    log.info("=" * 70)
    log.info("S5+S6 FEATURE IMPORTANCE v4 (feature_keep driven)")
    log.info("  Data:         %s", DATA_DIR)
    log.info("  Output:       %s", args.output_dir)
    log.info("  Assets:       %s", args.assets)
    log.info("  Targets:      %s", all_targets)
    log.info("  Sample:       %s rows per asset", f"{args.sample_rows:,}")
    log.info("  Folds:        %d (expanding window)", N_FOLDS)
    log.info("  Seeds:        %s", args.seeds)
    log.info("  Permutation:  %s (n_repeats=%d, n_jobs=%d)",
             do_perm, PERM_N_REPEATS, args.perm_jobs)
    log.info("  SHAP:         %s", do_shap)
    log.info("  Null base:    %s", do_null)
    log.info("  feature_keep: %s", args.feature_keep)
    log.info("  LightGBM:     num_leaves=%d, colsample=%.2f, subsample=%.2f",
             LGBM_PARAMS["num_leaves"], LGBM_PARAMS["colsample_bytree"],
             LGBM_PARAMS["subsample"])
    log.info("=" * 70)

    os.makedirs(args.output_dir, exist_ok=True)
    engine = detect_engine(log)

    # Load canonical feature list from feature_keep.csv
    feature_keep_cols = _load_feature_keep_columns(args.feature_keep, log)
    if feature_keep_cols is None:
        log.error("Cannot proceed without feature_keep.csv. Aborting.")
        sys.exit(1)

    # Infer feature list once from schema (same for all assets)
    all_files = sorted(glob.glob(os.path.join(DATA_DIR, "merged_*_*.parquet")))
    if not all_files:
        log.error("No merged files found in %s — run merge_s5_s6_full.py first.", DATA_DIR)
        sys.exit(1)

    signal_features = get_signal_features_from_schema(
        all_files, log, feature_keep_cols=feature_keep_cols)

    if not signal_features:
        log.error("No signal features after filtering. Aborting.")
        sys.exit(1)

    for asset in args.assets:
        log.info("")
        log.info("═" * 70)
        log.info("  ASSET: %s", asset.upper())
        log.info("═" * 70)

        result = stream_sample_numpy(asset, signal_features, all_targets,
                                     args.sample_rows, log)
        if result is None:
            continue

        X_full, targets_dict, available = result
        del result
        log.info("  X_full: %s rows × %d cols (%.1f MB)",
                 f"{X_full.shape[0]:,}", X_full.shape[1],
                 X_full.nbytes / 1e6)

        # Targets in merged data: ret_fwd_{horizon}_{asset}
        asset_targets = [f"{t}_{asset}" for t in all_targets]
        valid_targets = [t for t in asset_targets if t in targets_dict]
        if not valid_targets:
            log.error("No targets found for %s — skipping", asset.upper())
            del X_full, targets_dict
            continue

        target_results = {}
        target_r2s     = {}

        for full_target in valid_targets:
            # Extract horizon: remove known prefixes and asset suffix
            _t = full_target.replace(f"_{asset}", "")
            for _pfx in ("ret_fwd_", "mfe_fwd_", "mae_fwd_"):
                if _t.startswith(_pfx):
                    _t = _pfx.rstrip("_") + "_" + _t[len(_pfx):]
                    break
            horizon = full_target.replace(f"_{asset}", "").replace("ret_fwd_", "ret_").replace("mfe_fwd_", "mfe_").replace("mae_fwd_", "mae_")
            log.info("")
            log.info("─── %s — ret_fwd_%s ───", asset.upper(), horizon)

            y_full     = targets_dict[full_target]
            valid_mask = ~np.isnan(y_full)
            X = X_full[valid_mask]
            y = y_full[valid_mask]

            log.info("  Rows: %s (%.1f%%), y: mean=%.8f std=%.8f",
                     f"{len(y):,}", 100 * len(y) / X_full.shape[0], y.mean(), y.std())

            if len(y) < 10000:
                log.warning("  Too few valid rows — skipping %s", full_target)
                continue

            log.info("  Running %d-fold CV (%d seeds)...", N_FOLDS, len(args.seeds))
            t_cv = time.time()

            fold_recs, score_recs, perm_recs, shap_recs, null_recs = run_cv(
                X, y, available, N_FOLDS, args.seeds, engine, log,
                do_permutation=do_perm, do_shap=do_shap, do_null=do_null,
                perm_n_jobs=args.perm_jobs)

            log.info("  CV done in %.1fs", time.time() - t_cv)

            scores_df = pd.DataFrame(score_recs)
            log.info("  Scores:\n%s", scores_df.to_string(index=False))

            mean_r2 = scores_df["r2"].mean()
            # Key in target_results: short form for readability in consensus output
            short_key = f"ret_fwd_{horizon}"
            target_r2s[short_key] = mean_r2
            log.info("  Mean R²: %.6f", mean_r2)

            imp_df = aggregate_importance(
                fold_recs,
                perm_recs if do_perm  else None,
                shap_recs if do_shap  else None,
                null_recs if do_null  else None)

            target_results[short_key] = imp_df

            imp_df.to_csv(
                os.path.join(args.output_dir, f"{asset}_{horizon}_importance.csv"),
                index=False)
            scores_df.to_csv(
                os.path.join(args.output_dir, f"{asset}_{horizon}_cv_scores.csv"),
                index=False)
            log.info("  Saved: %s_%s_importance.csv", asset, horizon)

            show_cols = ["feature_name", "combined_rank", "gain_mean",
                         "gain_stability", "gain_std"]
            if do_perm:  show_cols.insert(3, "perm_mean")
            if do_null and "gain_over_null" in imp_df.columns:
                show_cols.insert(4, "gain_over_null")
            show_cols = [c for c in show_cols if c in imp_df.columns]
            log.info("  TOP 20:")
            log.info(imp_df[show_cols].head(20).to_string(index=False))

            n_gain_nonzero = (imp_df["gain_mean"] > 0).sum()
            n_perm_pos     = (imp_df["perm_mean"] > 0).sum() if do_perm else 0
            log.info("  Feature usage: %d/%d gain>0, %d/%d perm>0",
                     n_gain_nonzero, len(imp_df), n_perm_pos, len(imp_df))

            del X, y
            gc.collect()

        del X_full, targets_dict
        gc.collect()

        # ── Cross-target consensus ──────────────────────────────────────────
        if len(target_results) > 1:
            log.info("")
            log.info("═── CROSS-TARGET CONSENSUS (%s) — R²-WEIGHTED ──═", asset.upper())
            log.info("  R² per target: %s",
                     {t: f"{r:.6f}" for t, r in target_r2s.items()})

            consensus = build_consensus(target_results, target_r2s, log)
            consensus.to_csv(
                os.path.join(args.output_dir, f"{asset}_multi_target_consensus.csv"),
                index=False)
            log.info("Saved: %s_multi_target_consensus.csv", asset)

            for group_name, group_targets in HORIZON_GROUPS.items():
                group_avail = [t for t in group_targets if t in target_results]
                if len(group_avail) >= 2:
                    log.info("")
                    log.info("  Group consensus: %s (%s)", group_name, group_avail)
                    group_cons = build_consensus(
                        target_results, target_r2s, log,
                        targets_subset=group_avail)
                    group_cons.to_csv(
                        os.path.join(args.output_dir,
                                     f"{asset}_consensus_{group_name}.csv"),
                        index=False)
                    log.info("  Saved: %s_consensus_%s.csv", asset, group_name)

            never = consensus[consensus["never_important"]].copy()
            if len(never) > 0:
                never.to_csv(
                    os.path.join(args.output_dir,
                                 f"{asset}_never_important_features.csv"),
                    index=False)
                log.info("Saved: %s_never_important_features.csv (%d features)",
                         asset, len(never))

            show_c = ["feature_name", "weighted_pctile", "mean_pctile",
                      "min_pctile", "max_pctile", "mean_gain", "n_targets"]
            show_c = [c for c in show_c if c in consensus.columns]
            log.info("")
            log.info("TOP %d CONSENSUS (R²-weighted):", args.top)
            log.info(consensus[show_c].head(args.top).to_string(index=False))
            log.info("")
            log.info("BOTTOM 20 CONSENSUS:")
            log.info(consensus[show_c].tail(20).to_string(index=False))
            if len(never) > 0:
                log.info("")
                log.info("NEVER IMPORTANT BY BASE CONCEPT:")
                # S6 features have no family — group by name prefix instead
                never["prefix"] = never["feature_name"].str.split("_").str[:3].str.join("_")
                log.info(never.groupby("prefix").size()
                         .sort_values(ascending=False).head(20).to_string())

        elif len(target_results) == 1:
            target = list(target_results.keys())[0]
            target_results[target].to_csv(
                os.path.join(args.output_dir, f"{asset}_multi_target_consensus.csv"),
                index=False)

    log.info("")
    log.info("=" * 70)
    log.info("All done in %.1fs (%.1f min)",
             time.time() - t0, (time.time() - t0) / 60)
    log.info("=" * 70)


if __name__ == "__main__":
    signal.signal(signal.SIGHUP, signal.SIG_IGN)   # survive terminal close (SIGINT left for manual abort)
    os.setpgrp()                                    # new process group -> detached from shell
    main()