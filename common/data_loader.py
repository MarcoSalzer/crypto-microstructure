"""
data_loader.py — Load ML feature dataset for prediction.
========================================================
Reads ml_features/ parquets (output of create_ml_dataset.py) and
resolves feature/target/meta columns from feature_keep.csv.


                   Per-file processing:
                     - Read parquet (small, ~24 MB at float32).
                     - Drop rows where target is NaN.
                     - Write valid rows into preallocated X/y/timestamps.
                     - df is released before next file.

                   Trust assumption: parquet files are named with timestamps,
                   so sorted file order = sorted time order. Each parquet
                   is internally sorted by bucket_dt_utc index. After fill,
                   we verify global monotonicity and only sort if violated.
                   The defensive sort (if needed) requires 2× X temporarily.
"""

from __future__ import annotations

import ctypes
import gc
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from common.config import ML_FEATURES, KEEP_LIST, ASSETS, target_col

logger = logging.getLogger(__name__)


# ─── Keep-list parsing ────────────────────────────────────────────────────────

def _load_keep_list() -> pd.DataFrame:
    """Load feature_keep.csv."""
    if not KEEP_LIST.exists():
        raise FileNotFoundError(f"feature_keep.csv not found at {KEEP_LIST}")
    return pd.read_csv(KEEP_LIST)


def get_feature_columns(keep_df: Optional[pd.DataFrame] = None,
                        profile: str = "tree") -> list[str]:
    """
    Return the feature columns (type=feature) for a given model profile.

    profile selects which usage flag gates the feature set:
      'tree'    -> use_tree
      'linear'  -> use_linear
      'cluster' -> use_cluster
      'anomaly' -> use_anomaly
      'all'     -> no profile filter
    """
    if keep_df is None:
        keep_df = _load_keep_list()

    feat = keep_df[keep_df["type"] == "feature"]
    if profile == "all":
        return feat["column"].tolist()

    flag_col = f"use_{profile}"
    if flag_col not in feat.columns:
        logger.warning(
            "feature_keep.csv has no '%s' column — profile '%s' falls back "
            "to all feature columns.", flag_col, profile)
        return feat["column"].tolist()

    selected = feat.loc[feat[flag_col] == True, "column"].tolist()
    logger.info("Feature profile '%s': %d of %d feature columns selected",
                profile, len(selected), len(feat))
    return selected


def get_target_columns(keep_df: Optional[pd.DataFrame] = None) -> list[str]:
    if keep_df is None:
        keep_df = _load_keep_list()
    return keep_df.loc[keep_df["type"] == "target", "column"].tolist()


def get_meta_columns(keep_df: Optional[pd.DataFrame] = None) -> list[str]:
    if keep_df is None:
        keep_df = _load_keep_list()
    return keep_df.loc[keep_df["type"] == "meta", "column"].tolist()


# ─── File discovery ───────────────────────────────────────────────────────────

def discover_files(
    data_dir: Optional[Path] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> list[Path]:
    if data_dir is None:
        data_dir = ML_FEATURES
    files = sorted(data_dir.glob("ml_features_*.parquet"))

    if start_date or end_date:
        filtered = []
        for f in files:
            parts = f.stem.split("_")
            if len(parts) >= 3:
                file_date = parts[2]
                if start_date and file_date < start_date:
                    continue
                if end_date and file_date > end_date:
                    continue
            filtered.append(f)
        files = filtered

    return files


# ─── Memory helper ────────────────────────────────────────────────────────────

def _release_memory_to_os() -> None:
    """gc + glibc malloc_trim so freed buffers return to OS."""
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


# ─── Main loader (preallocation strategy) ────────────────────────────────────

def load_dataset(
    target: str = "ret_15s",
    asset: str = "btc",
    profile: str = "tree",
    max_hours: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    data_dir: Optional[Path] = None,
    seed: int = 42,
    target_only: bool = False,
    aux_targets: Optional[list[str]] = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, list[str]]:
    """
    Load the ML dataset for a specific target and asset using preallocated
    numpy arrays. Peak RAM ≈ final RAM, no concat/cast doubling.

    target_only=True: load ONLY the target column (y) + timestamps, skip the
        feature matrix entirely. X is returned as a zero-column array
        (shape (n_rows, 0)), so it costs ~0 RAM and the defensive global sort
        no longer copies a multi-GB feature matrix. Use this when the caller
        only needs y (e.g. the 1s return vector for path analysis) — it avoids
        allocating the full ~88 GB feature matrix just to drop it.

    Returns:
        X:             np.ndarray (n_rows, n_features), float32, NaN preserved
                       (n_features == 0 when target_only=True)
        y:             np.ndarray (n_rows,), float64
        info:          pd.DataFrame [timestamp, fold_order]
        feature_names: list[str] of column names matching X column order
                       (empty when target_only=True)
    """
    import pyarrow.parquet as pq

    keep_df      = _load_keep_list()
    feature_cols = get_feature_columns(keep_df, profile=profile)
    if target_only:
        # Skip all feature loading: zero feature columns → X stays (n_rows, 0),
        # which costs no RAM and makes the defensive sort trivial.
        feature_cols = []
    feature_set  = set(feature_cols)
    feat_idx     = {c: i for i, c in enumerate(feature_cols)}
    target_name  = target_col(target, asset)
    aux_targets  = aux_targets or []
    aux_cols     = [target_col(t, asset) for t in aux_targets]

    files = discover_files(data_dir, start_date, end_date)
    if not files:
        raise FileNotFoundError(
            f"No ml_features files found in {data_dir or ML_FEATURES}")

    rng = np.random.RandomState(seed)
    if max_hours and len(files) > max_hours:
        idx = rng.choice(len(files), max_hours, replace=False)
        idx.sort()
        files = [files[i] for i in idx]

    logger.info("Loading %d files for %s %s (preallocation strategy)...",
                len(files), asset, target)

    cols_to_load = list(feature_set | {target_name} | set(aux_cols))

    # ── Phase 1: scan metadata to sum total rows ─────────────────────────
    file_metas = []
    total_rows_upper = 0
    for f in files:
        try:
            meta = pq.read_metadata(f)
            n_rows = int(meta.num_rows)
            file_metas.append((f, n_rows))
            total_rows_upper += n_rows
        except Exception as e:
            logger.warning("Metadata read failed for %s: %s", f.name, e)

    if total_rows_upper == 0:
        raise ValueError("No data: all parquet metadata reads failed.")

    x_bytes = total_rows_upper * len(feature_cols) * 4
    logger.info("Preallocating: up to %s rows × %d features → X≈%.1f GB float32",
                f"{total_rows_upper:,}", len(feature_cols), x_bytes / 1e9)

    # ── Phase 2: preallocate ──────────────────────────────────────────────
    X = np.full((total_rows_upper, len(feature_cols)), np.nan, dtype=np.float32)
    y = np.full(total_rows_upper, np.nan, dtype=np.float64)
    Y_aux = (np.full((total_rows_upper, len(aux_cols)), np.nan, dtype=np.float64)
             if aux_cols else None)
    timestamps = np.empty(total_rows_upper, dtype="datetime64[ns]")

    # ── Phase 3: read each parquet, filter NaN target, write into slot ──
    write_offset       = 0
    n_files_processed  = 0
    n_files_no_target  = 0
    n_files_failed     = 0
    features_seen      = set()
    missing_logged     = False

    for i, (f, n_rows_meta) in enumerate(file_metas):
        try:
            schema_cols = set(pq.read_schema(f).names)
            if target_name not in schema_cols:
                n_files_no_target += 1
                continue

            load_cols = [c for c in cols_to_load if c in schema_cols]
            df = pd.read_parquet(f, columns=load_cols)

            # Defensive: enforce within-file chronological order
            if df.index.name == "bucket_dt_utc" and not df.index.is_monotonic_increasing:
                df = df.sort_index()

            # Filter NaN target within this file
            y_chunk = df[target_name].to_numpy(dtype=np.float64)
            valid   = ~np.isnan(y_chunk)
            n_valid = int(valid.sum())

            if n_valid == 0:
                n_files_no_target += 1
                continue

            end = write_offset + n_valid

            # Write target
            if n_valid == len(y_chunk):
                y[write_offset:end] = y_chunk
            else:
                y[write_offset:end] = y_chunk[valid]

            # Write aux targets, aligned to the SAME valid mask as the primary
            # target, so every horizon indexes the identical rows.
            if Y_aux is not None:
                for j, ac in enumerate(aux_cols):
                    if ac in df.columns:
                        av = df[ac].to_numpy(dtype=np.float64)
                        Y_aux[write_offset:end, j] = (av[valid]
                                                      if n_valid != len(av) else av)

            # Write timestamps. Index is tz-aware UTC; convert to tz-naive
            # numpy datetime64 explicitly to avoid pandas UserWarning spam.
            if df.index.name == "bucket_dt_utc":
                idx = df.index
                if idx.tz is not None:
                    idx = idx.tz_convert("UTC").tz_localize(None)
                ts_values = idx.to_numpy()
                if n_valid == len(ts_values):
                    timestamps[write_offset:end] = ts_values
                else:
                    timestamps[write_offset:end] = ts_values[valid]

            # Write features
            feat_in_file = [c for c in feature_cols if c in df.columns]
            features_seen.update(feat_in_file)
            if not missing_logged and len(feat_in_file) < len(feature_cols):
                missing_in_file = set(feature_cols) - set(feat_in_file)
                logger.info("  %d/%d features missing in %s (S6 partial?): %s",
                            len(missing_in_file), len(feature_cols), f.name,
                            list(missing_in_file)[:5])
                missing_logged = True

            if feat_in_file:
                arr = df[feat_in_file].to_numpy(dtype=np.float32)
                if n_valid != len(arr):
                    arr = arr[valid]

                # Fast path: all features present in defined order
                if len(feat_in_file) == len(feature_cols) and \
                        feat_in_file == feature_cols:
                    X[write_offset:end, :] = arr
                else:
                    # Partial / reordered file: scatter into correct columns
                    idxs = np.array([feat_idx[c] for c in feat_in_file],
                                    dtype=np.int64)
                    X[write_offset:end, idxs] = arr
                del arr

            write_offset = end
            n_files_processed += 1
            del df, y_chunk, valid

        except Exception as e:
            logger.warning("Error reading %s: %s", f.name, e)
            n_files_failed += 1

        if (i + 1) % 100 == 0:
            logger.info("  [%d/%d]  written=%s  no_target=%d  failed=%d",
                        i + 1, len(file_metas),
                        f"{write_offset:,}", n_files_no_target, n_files_failed)

    if write_offset == 0:
        raise ValueError(
            "No valid rows after filtering. Target may be missing from "
            "all files, or all target values are NaN.")

    # Trim to actual valid row count (numpy slice = view, no copy)
    X = X[:write_offset]
    y = y[:write_offset]
    timestamps = timestamps[:write_offset]
    if Y_aux is not None:
        Y_aux = Y_aux[:write_offset]

    n_dropped = total_rows_upper - write_offset
    logger.info("Loaded: %s rows (preallocated %s, dropped %s = %.1f%%)",
                f"{write_offset:,}", f"{total_rows_upper:,}",
                f"{n_dropped:,}", 100 * n_dropped / total_rows_upper)
    logger.info("  Files: processed=%d, no_target=%d, failed=%d",
                n_files_processed, n_files_no_target, n_files_failed)

    unseen = set(feature_cols) - features_seen
    if unseen:
        logger.warning("  %d features absent from ALL files (X columns all NaN): %s",
                       len(unseen), list(unseen)[:5])

    _release_memory_to_os()

    # ── Phase 4: verify global timestamp monotonicity ────────────────────
    # Files are processed in sorted-name order, which for ISO dates equals
    # chronological order. We still verify defensively.
    needs_sort = False
    if len(timestamps) > 1:
        diffs = timestamps[1:].view("i8") - timestamps[:-1].view("i8")
        n_violations = int((diffs < 0).sum())
        if n_violations > 0:
            logger.warning(
                "Timestamps not monotonic (%d violations) — sorting globally. "
                "This requires ~2× X RAM temporarily.", n_violations)
            needs_sort = True

    if needs_sort:
        sort_idx = np.argsort(timestamps, kind="mergesort")
        X = X[sort_idx]
        y = y[sort_idx]
        timestamps = timestamps[sort_idx]
        if Y_aux is not None:
            Y_aux = Y_aux[sort_idx]
        del sort_idx
        _release_memory_to_os()

    # ── Phase 5: build info frame ────────────────────────────────────────
    info = pd.DataFrame({
        "timestamp":  timestamps,
        "fold_order": np.arange(len(y)),
    })
    # Aux targets ride along as extra info columns (named by their token, e.g.
    # 'ret_300s'), aligned row-for-row with y. Non-breaking: callers that do
    # not pass aux_targets get the same info frame as before.
    if Y_aux is not None:
        for j, t in enumerate(aux_targets):
            info[t] = Y_aux[:, j]

    # ── Stats ────────────────────────────────────────────────────────────
    nan_frac    = float(np.isnan(X).mean())
    nan_per_col = np.isnan(X).mean(axis=0)
    n_high_nan  = int((nan_per_col > 0.10).sum())
    logger.info("  X: %s × %d, NaN: %.2f%% overall, %d cols >10%% NaN",
                f"{X.shape[0]:,}", X.shape[1], nan_frac * 100, n_high_nan)
    logger.info("  y: mean=%.2e, std=%.2e, min=%.2e, max=%.2e",
                float(y.mean()), float(y.std()), float(y.min()), float(y.max()))

    return X, y, info, feature_cols