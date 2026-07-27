# prediction/cluster_engine.py
# ==============================================================================
# WS4: MFE Analysis on Cluster-Filtered Trades
# ==============================================================================
#
# PURPOSE:
#   Our cluster prediction system identifies profitable breakout types
#   (clusters with DA > 55% and positive est. PnL). But we only know the
#   15s return — not how far these moves ACTUALLY run, how fast they peak,
#   or where the price typically reverses. This script answers those questions
#   by computing Maximum Favorable Excursion (MFE) and Maximum Adverse
#   Excursion (MAE) tick-by-tick for every filtered trade, grouped by cluster.
#
#   The results feed directly into WS3 (TP/SL optimization) and WS3d
#   (dynamic exit classifier) — they show where to set take-profit, how much
#   drawdown to tolerate, and when the move is over.
#
# PIPELINE (executed in order):
#
#   PHASE A — Reproduce the cluster pipeline (identical logic)
#   ──────────────────────────────────────────────────────────────
#   1. Load data via data_loader.load_dataset from ml_features_log1p:
#      X (features), y = ret_fwd_{hz} (FORWARD return at the config horizon),
#      y_1s = ret_fwd_1s (1s forward returns, for the post-event price paths).
#   2. Identify breakout events (TRAILING): a breakout is a move that has
#      ALREADY happened by time T, i.e. |return over the PAST hz seconds|
#      > threshold. The move over [T-hz, T] equals y[T-hz] (y shifted by hz
#      rows; 1 row = 1 second). The breakout direction is OBSERVABLE at T.
#   3. Cluster the COMPLETED breakout events on the feature matrix at T
#      (the predictive pipeline reads features at T-lookback, 1-5 s earlier).
#   4. Screen / select clusters: with --no-da-gate keep every cluster with
#      >= min-cluster-events members and screen the configuration by MFE-lift
#      (grid run); otherwise apply the directed-OOS directional-accuracy + PnL
#      gate (viable-cluster evaluation step).
#   5. The forward move y[T] = ret_fwd_{hz} is the CONTINUATION after the
#      breakout — the quantity the strategy trades and the baseline predicts.
#   6. Train LightGBM classifier (expanding window, 5 folds):
#      predict "good cluster" vs "bad cluster" from pre-event features
#   7. Collect OOS predictions → filtered trades = predicted "good"
#
#   Unlike the cluster pipeline which only saves summary CSVs, this script
#   KEEPS the per-trade arrays (indices, cluster labels, directions) so we
#   can do path analysis on the actual filtered trades.
#
#   PHASE B — MFE analysis on filtered trades (NEW in WS4)
#   ──────────────────────────────────────────────────────────────
#   8. Extract tick-by-tick price paths (y_1s, up to 300s) for:
#      - filtered trades (per cluster)
#      - ALL breakouts (baseline comparison)
#      - random entry points (null hypothesis)
#   9. Compute per-trade MFE, MAE, terminal return, time-to-MFE
#  10. Aggregate per cluster: full percentile distribution (P10–P99)
#  11. Time-to-level: how fast do filtered trades reach 5, 10, 20 bps?
#  12. Average price paths: mean/median cumulative return over time
#  13. Comparison table: filtered vs all breakouts vs random
#  14. Save per-trade results (event_index, cluster, direction, MFE, MAE)
#      → directly usable by WS3 (TP/SL grid) and WS3d (dynamic exit classifier)
#
#   PHASE C — Visualization
#   ──────────────────────────────────────────────────────────────
#  15. Generate matplotlib plots:
#      - Cluster overview bar chart (DA, est. PnL, cluster size)
#      - MFE comparison bar chart (mean MFE, P90, % > taker cost)
#      - Average price paths with confidence bands per cluster
#
# OUTPUTS (in RESULTS_DIR/cluster_mfe/):
#   cluster_mfe_{tag}.csv              MFE percentile distribution per cluster
#   cluster_time_to_level_{tag}.csv    Time-to-level per cluster
#   cluster_paths_{tag}.csv            Average price paths per cluster
#   cluster_mfe_comparison_{tag}.csv   Comparison table (filtered/all/random)
#   cluster_trades_{tag}.csv           Per-trade data for WS3/WS3d
#   ws4_overview_{tag}.png             Cluster overview bar chart
#   ws4_mfe_dist_{tag}.png             MFE distribution bar chart
#   ws4_paths_{tag}.png                Average price paths plot
#
# USAGE:
#   python cluster_engine.py --asset btc --hz 15s
#   python cluster_engine.py --asset both --hz 15s
#   python cluster_engine.py --asset btc --hz 15s --lookbacks 5  # fast test
#
# REQUIREMENTS:
#   Part of the prediction/ package (this is the cluster pipeline).
#   Imports: common.data_loader, common.config (RESULTS_DIR, SPREAD_BPS, MAKER_COST_BPS)
#   Packages: numpy, pandas, scikit-learn, lightgbm, matplotlib
# ==============================================================================
from __future__ import annotations
# --- Optional deterministic mode (opt-in via WS4_DETERMINISTIC=1) -------------
# Multi-threaded BLAS makes PCA/KMeans floating-point non-bit-reproducible across
# processes, which tips degenerate (large-n, high-k) partitions between runs.
# Setting threads=1 BEFORE numpy is imported, together with svd_solver="full"
# in the PCA, gives bit-reproducible clustering. Default OFF so the existing
# cluster_final partition stays reproducible; enable only for fresh re-clusters.
import os as _os
if _os.environ.get("WS4_DETERMINISTIC") == "1":
    for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
               "NUMEXPR_NUM_THREADS"):
        _os.environ[_v] = "1"
import argparse, gc, logging, sys, time
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _ts() -> str:
    """Wall-clock HH:MM:SS — prepended to milestone logs so the logfile
    shows when each step was written."""
    return datetime.now().strftime("%H:%M:%S")


def tprint(msg: str = "") -> None:
    """print() with a timestamp prefix (for phase milestones)."""
    print(f"{_ts()}  {msg}", flush=True)

# ─── S6-full loader ──────────────────────────────────────────────────────────
_EXCLUDE_PREFIXES = (
    "ret_", "mfe_fwd_", "mae_fwd_", "rv_fwd_",
    "tbl_", "barrier_", "label_", "data_health",
    "health_reason_code", "data_usability", "usability_",
    "l2_coverage", "lob50_health", "trades_coverage",
    "session_", "us_holiday", "us_rth",
)


def load_s6_full(asset: str, hz: str, data_dir, max_files: int = 0) -> tuple:
    """Load merged S5+S6 full dataset. Returns (X, y, y_1s, feat_names).
    max_files>0: load only N evenly-spaced files as RAM guard.
    """
    import glob
    files = sorted(glob.glob(str(data_dir) + "/merged_btceth_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No merged files in {data_dir}")
    if max_files > 0 and len(files) > max_files:
        step  = max(1, len(files) // max_files)
        files = files[::step][:max_files]
        logger.info("RAM guard: using %d files (~%d rows)", len(files), len(files)*3600)
    target_col = f"ret_fwd_{hz}_{asset}"
    target_1s  = f"ret_fwd_1s_{asset}"
    # Determine feature columns from first valid file (avoid full load for schema)
    feat_cols = None
    for f in files:
        try:
            probe = pd.read_parquet(f, columns=None)
            if target_col not in probe.columns:
                continue
            feat_cols = [
                c for c in probe.columns
                if not any(c.startswith(px) for px in _EXCLUDE_PREFIXES)
                and c not in (target_col, target_1s)
                and pd.api.types.is_numeric_dtype(probe[c])
            ]
            del probe
            break
        except Exception:
            continue
    if feat_cols is None:
        raise ValueError(f"No files with column '{target_col}'")

    # Stream-load: convert each file to numpy immediately — avoids pd.concat peak RAM
    load_cols = feat_cols + [target_col] + ([target_1s] if target_1s != target_col else [])
    X_chunks, y_chunks, y1s_chunks = [], [], []
    n_loaded = 0
    for f in files:
        try:
            df = pd.read_parquet(f, columns=load_cols)
            if target_col not in df.columns:
                continue
            X_chunks.append(df[feat_cols].values.astype(np.float32))
            y_chunks.append(df[target_col].values.astype(np.float64))
            y1s_chunks.append(
                df[target_1s].values.astype(np.float64)
                if target_1s in df.columns else y_chunks[-1]
            )
            n_loaded += 1
            del df
        except Exception as e:
            logger.warning("Skip %s: %s", f, e)

    if not X_chunks:
        raise ValueError(f"No files with column '{target_col}'")

    X    = np.concatenate(X_chunks,   axis=0);  del X_chunks
    y    = np.concatenate(y_chunks,   axis=0);  del y_chunks
    y_1s = np.concatenate(y1s_chunks, axis=0);  del y1s_chunks

    logger.info("S6-full loaded: %d rows × %d features, %d files",
                len(X), X.shape[1], n_loaded)
    return X, y, y_1s, feat_cols


def estimate_matrix_gb(n_rows: int, n_features: int = 2815) -> float:
    """Estimated size of the float32 feature matrix in GB."""
    return n_rows * n_features * 4 / 1e9


def check_ram_or_skip(n_rows: int, n_features: int = 2815,
                      safety_factor: float = 1.6,
                      min_headroom_gb: float = 10.0) -> bool:
    """
    Decide whether it is safe to load/process a matrix of this size.

    The raw matrix is n_rows × n_features × 4 bytes. While loading, the shared
    data_loader holds the preallocated matrix; during clustering a scaled copy
    of the (much smaller) event subset is made, and PCA reduces dimensionality
    before the heavy clustering work. The dominant cost is therefore the raw
    matrix plus a moderate transient overhead, approximated by `safety_factor`,
    plus a fixed headroom so we never push the whole host into swap.

    Returns True if safe to proceed, False if the config should be skipped.
    Uses psutil if available; otherwise falls back to /proc/meminfo, and if
    neither works, proceeds (logging that the check was skipped).
    """
    raw_gb = estimate_matrix_gb(n_rows, n_features)
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
                        avail_gb = int(line.split()[1]) / 1e6  # kB → GB
                        break
        except Exception:
            avail_gb = None

    if avail_gb is None:
        logger.warning("  RAM check unavailable — proceeding without guard "
                       "(raw matrix ≈ %.0f GB).", raw_gb)
        return True

    logger.info("  RAM check: matrix≈%.0f GB, need≈%.0f GB (×%.1f + %.0f GB), "
                "available≈%.0f GB",
                raw_gb, need_gb, safety_factor, min_headroom_gb, avail_gb)
    if avail_gb < need_gb:
        logger.error("  INSUFFICIENT RAM — skipping this config. "
                     "Free up memory or rerun with --max-hours to shrink the "
                     "matrix. (need≈%.0f GB, have≈%.0f GB)", need_gb, avail_gb)
        return False
    return True


# ─── Constants ────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# THE DEFAULTS BELOW ARE A REDUCED "ANCHOR" CONFIGURATION FOR A QUICK SINGLE RUN,
# NOT THE FULL SCREENING GRID. HORIZON_THRESHOLDS, K_CLUSTERS and PCA_VARIANTS,
# together with the run_ws4 horizons default ("5s", "15s"), give a small run.
# The committed artifacts (results/clustering/, docs/results.md) come from the
# FULL grid:
#   horizons    1s, 5s, 15s, 30s
#   thresholds  10, 15, 20, 30, 40 bps (horizon-specific; sparse cells are
#               dropped at run time by the 500-event screening floor)
#   PCA         50, 150, 300, 600
#   k           6, 8, 10  (swept via --k; resolve_k shrinks k where a cell
#               cannot fill it at MIN_PER_CLUSTER events, so k=8/10 collapse onto
#               k=6 in sparse cells: 336 requested -> 312 distinct)
# Reproduce the committed grid (needs the external data store) with:
#   python -m prediction.cluster_engine --asset both \
#     --hz 1s 5s 15s 30s --pca-components 50 150 300 600 --k 6 8 10
# ─────────────────────────────────────────────────────────────────────────────
# Thresholds are HORIZON-SPECIFIC: short-window moves are smaller, so a high
# threshold yields too few events. Built on the original 15s / 20 bps anchor.
HORIZON_THRESHOLDS = {
    "5s":  [10, 15],
    "15s": [10, 15, 20],
}
LOOKBACKS     = [1, 2, 5, 10]

# Cluster count per (horizon, threshold): the single-config FALLBACK used only
# when --k is not given. Used by KMeans / GMM / Agglomerative; HDBSCAN infers its
# own count from density. The committed grid does NOT use this table: it sweeps
# k in {6, 8, 10} via --k (see the grid note above). Clustering runs on ALL
# events for a cell, so total event count (not per-fold) drives this.
K_CLUSTERS = {
    ("5s", 10): 4, ("5s", 15): 4,
    ("15s", 10): 3, ("15s", 15): 4, ("15s", 20): 7,
}
K_DEFAULT       = 4
MIN_PER_CLUSTER = 100         # safety floor: shrink k if events/k below this
MFE_LOOKAHEAD   = 300         # seconds to track after entry

# Minimum events a cluster must contain to be evaluated as a good-cluster
# candidate. Clusters below this are skipped (not tested). Overridable via
# --min-cluster-events; default 20 preserves the original behaviour.
MIN_CLUSTER_EVENTS = 20

# Synthetic horizons: not native targets, built on the fly from the 1s forward
# returns as ret_fwd_Ns[t] = sum_{i=0}^{N-1} ret_1s[t+i]. Value = N in seconds.
# Self-checked against the native 5s target at load time (see run_ws4).
# Return horizons for the multi-horizon Cluster-DA (regime persistence). The
# breakout window fixes the cluster direction; DA is then measured on every
# horizon. 1s..60s are comparable to the LGBM baseline; 120s..900s are a
# persistence probe (does the directional signal outlast the window?).
AUX_RET_HZ = ["1s", "5s", "15s", "30s", "60s", "120s", "300s", "900s"]
# Windows (s) for the per-cluster MFE/MAE-over-horizon check (--mfe-windows).
MFE_WINDOWS = [15, 30, 60, 120, 300]
AUX_RET_TARGETS = [f"ret_{h}" for h in AUX_RET_HZ]

# Cluster methods exposed via --cluster-method.
CLUSTER_METHODS = ("kmeans", "gmm", "hdbscan")

# PCA variants for the dimensionality comparison (--pca-components): the anchor
# default. Integers reduce to that many components; "none" clusters on the scaled
# feature matrix directly (no reduction). The committed grid uses
# --pca-components 50 150 300 600 (see the grid note above).
PCA_VARIANTS = (25, 50, "none")

# Above this feature count, GMM cannot afford full covariance matrices
# (d×d per component), so it falls back to diagonal covariance. Relevant only
# for the "none" PCA variant (d ≈ 2815).
GMM_FULL_COV_MAX_DIM = 100


def resolve_k(hz: str, thr_bps: int, n_events: int, k_override: int = None) -> int:
    """
    Look up the configured k for (horizon, threshold) and apply a safety floor:
    never request more clusters than the event count can populate at
    MIN_PER_CLUSTER events each. Deterministic and logged.
    k_override (if set) overrides the table — for the k scan.
    """
    k = k_override if k_override is not None else K_CLUSTERS.get((hz, thr_bps), K_DEFAULT)
    k_safe = max(2, min(k, n_events // MIN_PER_CLUSTER))
    if k_safe != k:
        logger.info("    k reduced %d → %d (%d events, floor %d/cluster)",
                    k, k_safe, n_events, MIN_PER_CLUSTER)
    return k_safe


def cluster_events(X_in: np.ndarray, method: str, k: int,
                   random_state: int = 42):
    """
    Cluster breakout events. X_in is either the PCA-reduced matrix or the
    full scaled feature matrix (for the "none" PCA variant).

    Returns (labels, k_effective):
      labels       : int array, one cluster id per event (-1 = noise, HDBSCAN)
      k_effective  : number of non-noise clusters actually found

    KMeans / GMM / Agglomerative use the supplied k. HDBSCAN ignores k and
    derives the cluster count from density; min_cluster_size is scaled to the
    event count so small thresholds don't shatter into noise.

    For high-dimensional input (no PCA), GMM uses diagonal covariance — full
    covariance is infeasible at d ≈ 2815. This is a documented caveat: the
    "none" GMM run is not a like-for-like covariance comparison with the
    PCA runs.
    """
    method = method.lower()
    n, d = X_in.shape

    if method == "kmeans":
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=k, n_init=20, random_state=random_state)
        labels = km.fit_predict(X_in)

    elif method == "gmm":
        from sklearn.mixture import GaussianMixture
        cov = "full" if d <= GMM_FULL_COV_MAX_DIM else "diag"
        n_init = 5 if d <= GMM_FULL_COV_MAX_DIM else 1   # diag path: keep cheap
        if cov == "diag":
            logger.info("    GMM: d=%d > %d → covariance_type='diag'",
                        d, GMM_FULL_COV_MAX_DIM)
        gmm = GaussianMixture(
            n_components=k, covariance_type=cov,
            n_init=n_init, random_state=random_state,
        )
        labels = gmm.fit_predict(X_in)

    elif method == "hdbscan":
        # HDBSCAN ships with sklearn >=1.3 as sklearn.cluster.HDBSCAN.
        try:
            from sklearn.cluster import HDBSCAN
        except ImportError as e:
            raise ImportError(
                "HDBSCAN requires scikit-learn >= 1.3. "
                "Upgrade sklearn or drop --cluster-method hdbscan."
            ) from e
        # Scale min_cluster_size to event count: at least 50, at most ~2% of
        # events, so we get a handful of dense clusters rather than dozens.
        min_cluster_size = int(np.clip(n * 0.02, 50, 2000))
        hdb = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=None,            # defaults to min_cluster_size
            cluster_selection_method="eom",
            copy=True,                   # silence 1.10 default-change warning
        )
        labels = hdb.fit_predict(X_in)

    elif method == "agglomerative":
        from sklearn.cluster import AgglomerativeClustering
        agg = AgglomerativeClustering(n_clusters=k, linkage="ward")
        labels = agg.fit_predict(X_in)

    else:
        raise ValueError(f"Unknown cluster method: {method!r}")

    k_eff = len(set(labels.tolist()) - {-1})
    return labels.astype(int), k_eff


# ═══════════════════════════════════════════════════════════════════════════════
# PATH ANALYSIS HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def extract_price_paths(y_1s, signal_indices, signal_directions, max_lookahead=300):
    """
    Extract cumulative return paths starting at each signal.
    Returns (n_signals, max_lookahead) array in bps, direction-adjusted.
    """
    n = len(signal_indices)
    paths = np.full((n, max_lookahead), np.nan)
    for i in range(n):
        idx, d = signal_indices[i], signal_directions[i]
        end = min(idx + max_lookahead + 1, len(y_1s))
        if end <= idx + 1:
            continue
        cum = np.cumsum(y_1s[idx + 1 : end]) * d * 10_000
        paths[i, :len(cum)] = cum
    return paths


def compute_mfe_mae(paths):
    """From (n, T) path array: MFE, MAE, terminal return, time-to-MFE."""
    n = paths.shape[0]
    mfe, mae, term, mfet = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    for i in range(n):
        v = paths[i][~np.isnan(paths[i])]
        if len(v) == 0:
            continue
        mfe[i], mae[i], term[i] = v.max(), v.min(), v[-1]
        mfet[i] = np.argmax(v) + 1
    return mfe, mae, term, mfet


def time_to_level(paths, levels):
    """Hit rate and median time until path reaches each level (bps)."""
    n = paths.shape[0]
    rows = []
    for lv in levels:
        hit_times = []
        for i in range(n):
            v = paths[i][~np.isnan(paths[i])]
            above = np.where(v >= lv)[0]
            if len(above) > 0:
                hit_times.append(above[0] + 1)
        nh = len(hit_times)
        if nh > 0:
            ht = np.array(hit_times)
            rows.append(dict(level_bps=lv, hit_rate=round(nh/n,4), n_hit=nh,
                             mean_time_s=round(ht.mean(),1),
                             median_time_s=round(float(np.median(ht)),1),
                             p25_time_s=round(float(np.percentile(ht,25)),1),
                             p75_time_s=round(float(np.percentile(ht,75)),1)))
        else:
            rows.append(dict(level_bps=lv, hit_rate=0, n_hit=0,
                             mean_time_s=np.nan, median_time_s=np.nan,
                             p25_time_s=np.nan, p75_time_s=np.nan))
    return pd.DataFrame(rows)


def mfe_distribution_table(mfe, mae, term, mfet, taker_cost, maker_cost):
    """Percentile table for a group of trades."""
    rows = []
    for p in [10, 25, 50, 75, 90, 95, 99]:
        rows.append(dict(
            percentile=f"P{p}",
            mfe_bps=round(np.percentile(mfe, p), 2),
            mae_bps=round(np.percentile(mae, p), 2),
            terminal_bps=round(np.percentile(term, p), 2),
            mfe_time_s=round(np.percentile(mfet, p), 1),
        ))
    rows.append(dict(percentile="MEAN",
                     mfe_bps=round(mfe.mean(),2), mae_bps=round(mae.mean(),2),
                     terminal_bps=round(term.mean(),2), mfe_time_s=round(mfet.mean(),1)))
    rows.append(dict(percentile=f"MFE>=taker({taker_cost}bps)",
                     mfe_bps=round((mfe>=taker_cost).mean()*100,2),
                     mae_bps=0, terminal_bps=0, mfe_time_s=0))
    rows.append(dict(percentile=f"MFE>=maker({maker_cost}bps)",
                     mfe_bps=round((mfe>=maker_cost).mean()*100,2),
                     mae_bps=0, terminal_bps=0, mfe_time_s=0))
    return pd.DataFrame(rows)


def average_path_table(paths, max_s=300):
    """Average price path over time: mean, median, confidence bands."""
    rows = []
    for t in range(min(max_s, paths.shape[1])):
        v = paths[:, t][~np.isnan(paths[:, t])]
        if len(v) == 0:
            continue
        rows.append(dict(
            time_s=t+1, mean_ret_bps=round(v.mean(),3),
            median_ret_bps=round(float(np.median(v)),3),
            p25_ret_bps=round(float(np.percentile(v,25)),3),
            p75_ret_bps=round(float(np.percentile(v,75)),3),
            p10_ret_bps=round(float(np.percentile(v,10)),3),
            p90_ret_bps=round(float(np.percentile(v,90)),3),
            n_valid=int(len(v)),
        ))
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION (Phase C)
# ═══════════════════════════════════════════════════════════════════════════════

def _setup_mpl():
    """Configure matplotlib for headless plotting."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.grid": True, "grid.alpha": 0.3,
        "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11,
    })
    return plt


def plot_cluster_overview(cluster_stats, good_clusters, taker_cost, maker_cost,
                          tag, out_dir):
    """
    Bar chart showing DA, est. taker PnL, and cluster size.
    Highlights good clusters in green, others in gray.
    """
    plt = _setup_mpl()

    clusters = sorted(cluster_stats.keys())
    labels   = [f"Cl {c}" for c in clusters]
    da_vals  = [cluster_stats[c]["da"] * 100 for c in clusters]
    pnl_tk   = [cluster_stats[c]["est_pnl_taker"] for c in clusters]
    n_ev     = [cluster_stats[c]["n"] for c in clusters]
    colors   = ["#1D9E75" if c in good_clusters else "#888" for c in clusters]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # DA per cluster
    ax = axes[0]
    ax.bar(labels, da_vals, color=colors, alpha=0.85, edgecolor="white")
    ax.axhline(y=55, color="gray", linestyle="--", alpha=0.5, label="55% threshold")
    ax.set_ylabel("Directional accuracy (%)"); ax.set_title("DA per cluster")
    ax.legend(fontsize=8)
    for i, v in enumerate(da_vals):
        ax.text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=8)

    # Est. PnL taker
    ax = axes[1]
    bc = ["#1D9E75" if v > 0 else "#D85A30" for v in pnl_tk]
    ax.bar(labels, pnl_tk, color=bc, alpha=0.85, edgecolor="white")
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_ylabel("Est. PnL (bps, taker)"); ax.set_title("Est. taker PnL per cluster")
    for i, v in enumerate(pnl_tk):
        ax.text(i, v + 0.3, f"{v:+.1f}", ha="center", fontsize=8)

    # Cluster size
    ax = axes[2]
    ax.bar(labels, n_ev, color=colors, alpha=0.85, edgecolor="white")
    ax.set_ylabel("N events"); ax.set_title("Cluster size")
    for i, v in enumerate(n_ev):
        ax.text(i, v + 10, f"{v:,}", ha="center", fontsize=8)

    fig.suptitle(f"Cluster overview — {tag}", fontsize=13, y=1.02)
    plt.tight_layout()
    path = out_dir / f"ws4_overview_{tag}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"      Plot: {path.name}")


def plot_mfe_distribution(df_comp, taker_cost, maker_cost, tag, out_dir):
    """
    Horizontal bar chart comparing mean MFE, P90 MFE, and MFE > taker %
    across clusters, all filtered, all breakouts, and random.
    """
    plt = _setup_mpl()

    # Sort: clusters desc by MFE, then reference groups
    cl_rows = df_comp[df_comp["group"].str.startswith("Cluster_")].sort_values("mean_mfe_bps", ascending=False)
    sp_rows = df_comp[~df_comp["group"].str.startswith("Cluster_")]
    df_s = pd.concat([cl_rows, sp_rows])

    labels = df_s["group"].apply(lambda x: x.replace("Cluster_","Cl ")).values
    colors = ["#378ADD" if g.startswith("Cluster_") else
              "#1D9E75" if g == "ALL_FILTERED" else
              "#D85A30" if g == "ALL_BREAKOUTS" else "#888"
              for g in df_s["group"]]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    y_pos = range(len(labels))

    # Mean MFE
    ax = axes[0]
    ax.barh(y_pos, df_s["mean_mfe_bps"], color=colors, alpha=0.85)
    ax.axvline(x=taker_cost, color="red", linestyle="--", alpha=0.4, label=f"Taker ({taker_cost} bps)")
    ax.axvline(x=maker_cost, color="green", linestyle="--", alpha=0.4, label=f"Maker ({maker_cost} bps)")
    ax.set_yticks(y_pos); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("bps"); ax.set_title("Mean MFE")
    ax.legend(fontsize=8); ax.invert_yaxis()

    # P90 MFE
    ax = axes[1]
    ax.barh(y_pos, df_s["p90_mfe_bps"], color=colors, alpha=0.85)
    ax.set_yticks(y_pos); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("bps"); ax.set_title("P90 MFE"); ax.invert_yaxis()

    # MFE > taker %
    ax = axes[2]
    ax.barh(y_pos, df_s["mfe_gt_taker_pct"], color=colors, alpha=0.85)
    ax.axvline(x=50, color="gray", linestyle="--", alpha=0.3)
    ax.set_yticks(y_pos); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("%"); ax.set_title("MFE > taker cost (%)"); ax.invert_yaxis()

    fig.suptitle(f"MFE comparison — {tag}", fontsize=13, y=1.02)
    plt.tight_layout()
    path = out_dir / f"ws4_mfe_dist_{tag}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"      Plot: {path.name}")


def plot_price_paths(df_paths, taker_cost, maker_cost, tag, out_dir):
    """
    Left: average price path per cluster overlaid.
    Right: confidence band (P10-P90, P25-P75) for all filtered trades.
    """
    plt = _setup_mpl()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    cmap = plt.cm.Set2

    # Left: all clusters overlaid
    ax = axes[0]
    clusters = [c for c in df_paths["cluster"].unique()
                if str(c) not in ("ALL_BREAKOUTS", "RANDOM")]
    for i, cl in enumerate(clusters):
        d = df_paths[df_paths["cluster"] == cl]
        if len(d) == 0: continue
        color = cmap(i / max(len(clusters)-1, 1))
        lbl = f"Cl {cl}" if cl != "ALL_FILTERED" else "All filtered"
        lw = 2.0 if cl == "ALL_FILTERED" else 1.2
        ax.plot(d["time_s"], d["mean_ret_bps"], label=lbl, color=color, linewidth=lw)

    rand = df_paths[df_paths["cluster"] == "RANDOM"]
    if len(rand) > 0:
        ax.plot(rand["time_s"], rand["mean_ret_bps"],
                label="Random", color="gray", linewidth=1, linestyle="--", alpha=0.5)

    ax.axhline(y=0, color="black", linewidth=0.5, alpha=0.3)
    ax.axhline(y=taker_cost, color="red", linewidth=0.5, linestyle="--", alpha=0.3)
    ax.set_xlabel("Time since entry (s)"); ax.set_ylabel("Mean return (bps)")
    ax.set_title("Average price path per cluster")
    ax.legend(fontsize=7, loc="upper left"); ax.set_xlim(0, 300)

    # Right: confidence band for ALL_FILTERED
    ax = axes[1]
    af = df_paths[df_paths["cluster"] == "ALL_FILTERED"]
    if len(af) > 0:
        t = af["time_s"]
        ax.fill_between(t, af["p10_ret_bps"], af["p90_ret_bps"],
                        alpha=0.15, color="#378ADD", label="P10-P90")
        ax.fill_between(t, af["p25_ret_bps"], af["p75_ret_bps"],
                        alpha=0.25, color="#378ADD", label="P25-P75")
        ax.plot(t, af["median_ret_bps"], color="#378ADD", linewidth=2, label="Median")
        ax.plot(t, af["mean_ret_bps"], color="#D85A30", linewidth=1.5, linestyle="--", label="Mean")

    ax.axhline(y=0, color="black", linewidth=0.5, alpha=0.3)
    ax.axhline(y=maker_cost, color="green", linewidth=0.5, linestyle="--",
               alpha=0.5, label=f"Maker ({maker_cost} bps)")
    ax.set_xlabel("Time since entry (s)"); ax.set_ylabel("Return (bps)")
    ax.set_title("All filtered trades — confidence band")
    ax.legend(fontsize=8); ax.set_xlim(0, 300)

    plt.tight_layout()
    path = out_dir / f"ws4_paths_{tag}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"      Plot: {path.name}")



# ═══════════════════════════════════════════════════════════════════════════════
# PHASE D — DYNAMIC EXIT CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════════

def build_exit_dataset(
    X: np.ndarray,
    filt_idx: np.ndarray,
    filt_dirs: np.ndarray,
    mfe_vals: np.ndarray,
    paths: np.ndarray,
    taker_cost: float,
    mfe_frac: float = 0.60,     # lowered from 0.80 → more positive labels, less imbalanced
    min_hold_s: int  = 3,
    min_abs_bps: float = 5.0,   # NEW: must be at least 5 bps above zero to label as "exit"
    max_lookahead: int = 300,
    step_s: int = 1,
) -> tuple:
    """
    Build the exit classifier dataset from filtered trades.

    For each filtered trade and each in-trade second k:
      - Feature vector X[event_index + k]  (microstructure DURING the trade)
      - Label 1 ("exit now") if ALL of:
          k >= min_hold_s                                         (past minimum hold)
          ret_now >= mfe_frac * MFE                               (captured enough of the move)
          ret_now > taker_cost + min_abs_bps                      (meaningfully profitable)
          ret_now is near the local peak in next 10s              (don't label a false peak)

    The local-peak guard prevents labelling a step as "exit" when the price
    is still trending — only label when we are near the top of the move.

    Returns:
      X_exit   — (N_steps, n_features) feature matrix
      y_exit   — (N_steps,) binary exit labels
      meta     — list of dicts {trade_idx, step_k, ret_bps} for diagnostics
    """
    X_steps, y_steps, meta = [], [], []
    n_total = len(X)
    min_ret_thresh = taker_cost + min_abs_bps  # e.g. ~8bps combined minimum

    for i, (ev_idx, direction, mfe_bps) in enumerate(zip(filt_idx, filt_dirs, mfe_vals)):
        if mfe_bps < min_ret_thresh:
            continue  # skip trades too small to have a sensible exit

        trade_path = paths[i]  # direction-adjusted returns in bps
        n_path = len(trade_path)

        for k in range(min_hold_s, min(max_lookahead, n_path), step_s):
            feat_idx = ev_idx + k
            if feat_idx >= n_total:
                break

            ret_now = float(trade_path[k - 1])

            # ── Label conditions ──────────────────────────────────────────────
            # 1. Captured enough of the total move
            pct_captured = ret_now >= mfe_frac * mfe_bps

            # 2. Meaningfully profitable (above combined cost floor)
            profitable   = ret_now > min_ret_thresh

            # 3. Near local peak: ret in next 10s doesn't exceed ret_now by >15%
            # (prevents labelling a false peak mid-run)
            future_window = trade_path[k:min(k + 10, n_path)]
            future_max    = float(np.max(future_window)) if len(future_window) > 0 else ret_now
            near_peak     = future_max <= ret_now * 1.15  # within 15% of current level

            good_exit = int(pct_captured and profitable and near_peak)

            X_steps.append(X[feat_idx])
            y_steps.append(good_exit)
            meta.append({"trade_i": i, "step_k": k, "ret_bps": round(ret_now, 3)})

    if not X_steps:
        return np.empty((0, X.shape[1])), np.empty(0), []

    return np.array(X_steps, dtype=np.float32), np.array(y_steps, dtype=np.int32), meta


def train_exit_classifier(
    X_exit: np.ndarray,
    y_exit: np.ndarray,
    n_folds: int = 5,
    n_jobs: int = 8,
) -> tuple:
    """
    Train exit classifier with expanding-window CV.

    Returns:
      fold_rows  — per-fold accuracy/AUC/precision/recall metrics
      prob_thr   — probability threshold that maximises F1 on OOS data
      oos_probs  — OOS predicted probabilities (same order as input)
    """
    from sklearn.impute import SimpleImputer
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

    n = len(X_exit)
    block = n // (n_folds + 1)
    oos_probs = np.full(n, np.nan)
    fold_rows = []

    for fold_idx in range(n_folds):
        tr_end   = (fold_idx + 1) * block
        te_start = tr_end
        te_end   = min(te_start + block, n)

        if tr_end < 200 or te_end - te_start < 50:
            continue
        if y_exit[:tr_end].sum() < 20 or y_exit[te_start:te_end].sum() < 10:
            continue

        imp = SimpleImputer(strategy="median")
        X_tr = imp.fit_transform(X_exit[:tr_end])
        X_te = imp.transform(X_exit[te_start:te_end])
        y_tr = y_exit[:tr_end]
        y_te = y_exit[te_start:te_end]

        n_val = max(int(len(X_tr) * 0.1), 50)
        model = lgb.LGBMClassifier(
            n_estimators=300, num_leaves=20, max_depth=5,
            learning_rate=0.05, colsample_bytree=0.5,
            subsample=0.8, class_weight="balanced",
            n_jobs=n_jobs, verbose=-1, random_state=42,
        )
        model.fit(
            X_tr[:-n_val], y_tr[:-n_val],
            eval_set=[(X_tr[-n_val:], y_tr[-n_val:])],
            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
        )
        probs = model.predict_proba(X_te)[:, 1]
        oos_probs[te_start:te_end] = probs

        auc = roc_auc_score(y_te, probs) if len(np.unique(y_te)) > 1 else 0.5
        # Find threshold maximising F1
        best_f1, best_thr = 0.0, 0.5
        for thr in np.arange(0.3, 0.9, 0.05):
            preds = (probs >= thr).astype(int)
            f1 = f1_score(y_te, preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_thr = f1, thr

        preds_best = (probs >= best_thr).astype(int)
        fold_rows.append(dict(
            fold=fold_idx + 1,
            train_steps=tr_end,
            test_steps=te_end - te_start,
            pos_rate=round(y_te.mean(), 3),
            auc=round(auc, 4),
            best_f1=round(best_f1, 4),
            best_thr=round(best_thr, 2),
            precision=round(precision_score(y_te, preds_best, zero_division=0), 4),
            recall=round(recall_score(y_te, preds_best, zero_division=0), 4),
        ))

    # Global OOS threshold
    valid = ~np.isnan(oos_probs)
    if valid.sum() > 0 and len(np.unique(y_exit[valid])) > 1:
        best_f1, prob_thr = 0.0, 0.5
        for thr in np.arange(0.3, 0.9, 0.05):
            preds = (oos_probs[valid] >= thr).astype(int)
            f1 = f1_score(y_exit[valid], preds, zero_division=0)
            if f1 > best_f1:
                best_f1, prob_thr = f1, thr
    else:
        prob_thr = 0.5

    return fold_rows, prob_thr, oos_probs


def simulate_dynamic_exit(
    filt_idx: np.ndarray,
    filt_dirs: np.ndarray,
    paths: np.ndarray,
    oos_probs: np.ndarray,
    prob_thr: float,
    taker_cost: float,
    mfe_vals: np.ndarray,
    min_hold_s: int = 3,
    mfe_frac: float = 0.80,
    step_s: int = 1,
) -> pd.DataFrame:
    """
    Simulate dynamic exit vs fixed strategies for each filtered trade.

    Compares:
      dynamic_exit_bps  — exit when classifier prob >= prob_thr
      tp_exit_bps       — exit at first tick >= MFE * mfe_frac (theoretical best)
      hold_to_mfe_bps   — hold to MFE (oracle)
      hold_300s_bps     — hold full 300s

    Returns DataFrame with one row per trade.
    """
    rows = []
    step_cursor = 0  # tracks position in flattened X_exit / oos_probs

    for i, (ev_idx, direction, mfe_bps) in enumerate(zip(filt_idx, filt_dirs, mfe_vals)):
        trade_path = paths[i]
        n_steps    = max(0, min(len(trade_path), 300) - min_hold_s)

        if mfe_bps < taker_cost or n_steps <= 0:
            step_cursor += n_steps
            continue

        # Dynamic exit: first step where prob >= threshold AND return is positive
        dyn_exit_bps = float(trade_path[-1])  # fallback: hold to end
        for k_rel in range(n_steps):
            flat_idx = step_cursor + k_rel
            if flat_idx < len(oos_probs) and not np.isnan(oos_probs[flat_idx]):
                if oos_probs[flat_idx] >= prob_thr:
                    candidate_ret = float(trade_path[min_hold_s + k_rel - 1])
                    # Guardrail: never exit at a loss — wait for next profitable signal
                    if candidate_ret > taker_cost:
                        dyn_exit_bps = candidate_ret
                        break

        # TP exit: first step >= mfe_frac * MFE
        tp_thr = mfe_frac * mfe_bps
        tp_exit_bps = float(trade_path[-1])
        for k in range(min_hold_s, len(trade_path)):
            if trade_path[k - 1] >= tp_thr:
                tp_exit_bps = float(trade_path[k - 1])
                break

        rows.append(dict(
            trade_i=i,
            event_index=int(ev_idx),
            direction=int(direction),
            mfe_bps=round(float(mfe_bps), 2),
            dynamic_exit_bps=round(dyn_exit_bps, 2),
            tp_exit_bps=round(tp_exit_bps, 2),
            hold_to_mfe_bps=round(float(mfe_bps), 2),
            hold_300s_bps=round(float(trade_path[-1]), 2),
            dynamic_vs_tp=round(dyn_exit_bps - tp_exit_bps, 2),
        ))
        step_cursor += n_steps

    return pd.DataFrame(rows)

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════
# TRAIN-ONLY FOLD-INTERNAL CLUSTERING
# ═══════════════════════════════════════════════════════════════════


# ─────────────────────────────────────────────────────────────────────────────
def _fit_pipeline_train(X_ev_raw_tr, pca_variant, method, k, random_state=42):
    """
    Fit impute → scale → (PCA) → cluster on TRAIN events only.
    Returns the fitted objects + the train cluster labels.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture

    GMM_FULL_COV_MAX_DIM = 100  # same as in the original

    imp = SimpleImputer(strategy="median")
    # SCALER = StandardScaler (mean/std). THIS IS THE PROJECT STANDARD.
    # Do NOT switch to RobustScaler (median/IQR): RobustScaler scales by
    # the IQR but leaves extreme outlier events undamped in the space,
    # (outliers / small IQR = enormous value). Those outliers then hijack the
    # which pulls the KMeans centroids and collapses the clustering into one giant cluster
    # (observed in the 24h run with RobustScaler). StandardScaler is the
    # choice used consistently across all runs. If outliers cause problems:
    # winsorise, do NOT swap the scaler.
    scl = StandardScaler()
    Xtr = imp.fit_transform(X_ev_raw_tr)
    Xtr = scl.fit_transform(Xtr)
    Xtr = np.nan_to_num(Xtr, nan=0.0)

    pca = None
    if pca_variant != "none":
        n_comp = min(int(pca_variant), Xtr.shape[1], Xtr.shape[0] - 1)
        pca = PCA(n_components=n_comp, random_state=random_state)
        Xtr = pca.fit_transform(Xtr)

    method = method.lower()
    if method == "kmeans":
        model = KMeans(n_clusters=k, n_init=20, random_state=random_state)
        labels_tr = model.fit_predict(Xtr)
    elif method == "gmm":
        d = Xtr.shape[1]
        cov = "full" if d <= GMM_FULL_COV_MAX_DIM else "diag"
        n_init = 5 if d <= GMM_FULL_COV_MAX_DIM else 1
        model = GaussianMixture(n_components=k, covariance_type=cov,
                                n_init=n_init, random_state=random_state)
        labels_tr = model.fit_predict(Xtr)
    else:
        # HDBSCAN cannot be assigned out of sample (no predict) → deliberately
        # unsupported here. Across the whole run HDBSCAN never found
        # clusters anyway; dropped from the method grid.
        raise ValueError(
            f"method={method!r} is not supported in honest CV "
            f"(no out-of-sample assign). Use kmeans or gmm.")

    return imp, scl, pca, model, labels_tr


def _assign(model, imp, scl, pca, X_ev_raw):
    """Assign test points to a cluster with the TRAIN-fitted pipeline."""
    Xt = imp.transform(X_ev_raw)
    Xt = scl.transform(Xt)
    Xt = np.nan_to_num(Xt, nan=0.0)
    if pca is not None:
        Xt = pca.transform(Xt)
    return model.predict(Xt).astype(int)


def _good_clusters_from_train(labels_tr, dirs_tr, abs_ret_tr_bps,
                              taker_cost, min_n=None, da_min=0.55,
                              require_pnl=True):
    """
    Good-cluster selection from train ONLY (identical rule to the original:
    est_pnl_taker > 0 AND DA > 0.55). Returns good-set + majority direction per
    clusters.
    """
    good, majority = [], {}
    if min_n is None:
        min_n = MIN_CLUSTER_EVENTS
    for c in sorted(set(labels_tr.tolist())):
        m = labels_tr == c
        if m.sum() < min_n:
            continue
        up = (dirs_tr[m] > 0).mean()
        da = max(up, 1 - up)
        majority[c] = 1 if up > 0.5 else -1
        est_pnl = abs_ret_tr_bps[m].mean() * da - taker_cost
        # DA-only gate when require_pnl=False (Section 4.4: signal only;
        # profitability is decided in 4.5). PnL still computed for reporting.
        pnl_ok = (est_pnl > 0) if require_pnl else True
        if pnl_ok and da > da_min:
            good.append(c)
    return set(good), majority


# ─────────────────────────────────────────────────────────────────────────────
def honest_cluster_classifier_cv(
    X_full,              # (n_rows, n_feat) full feature matrix (for pre-event features)
    X_ev_raw,            # (n_events, n_feat_valid) RAW event features (before impute/scale/pca)
    event_indices,      # (n_events,) row index of each event in X_full
    event_dirs,         # (n_events,) sign(y) at the event
    y,                  # (n_rows,) target (for cluster PnL)
    pca_variant,        # "none" | 25 | 50 ...
    method, k,
    lookbacks,
    taker_cost,
    n_folds=5,
    n_jobs=8,
    random_state=42,
):
    """
    Returns: dict lookback -> {
        "filt_idx":   OOS predicted-good event indices (in X_full),
        "filt_dirs":  trade direction (train majority of the assigned cluster),
        "filt_labels": fold-local cluster IDs (NOT comparable across folds),
        "fold_rows":  per-fold precision/recall/lift (honest),
        "oos_is_good": OOS label per test event (for diagnostics),
    }
    """
    import lightgbm as lgb
    from sklearn.impute import SimpleImputer

    n_rows = len(X_full)

    # 1) Sort events ONCE by time (index = time, since files are time-sorted).
    order = np.argsort(event_indices)
    ev_idx = event_indices[order]
    ev_dir = event_dirs[order]
    ev_raw = X_ev_raw[order]
    abs_ret_bps = np.abs(y[ev_idx]) * 10_000.0
    n_ev = len(ev_idx)
    block = n_ev // (n_folds + 1)
    if block < 1:
        return {}

    # 2) Per fold: fit pipeline on train, determine good-set, assign test.
    #    cache so it is shared across all lookbacks (clustering is
    #    lookback-independent).
    folds = []
    for f in range(n_folds):
        tr_end = (f + 1) * block
        te_s, te_e = tr_end, min(tr_end + block, n_ev)
        if tr_end < 100 or te_e - te_s < 50:
            continue
        try:
            imp, scl, pca, model, lab_tr = _fit_pipeline_train(
                ev_raw[:tr_end], pca_variant, method, k, random_state)
        except Exception as e:
            print(f"        [honest-cv] fold {f+1} cluster-fit failed: {e}")
            continue

        good, majority = _good_clusters_from_train(
            lab_tr, ev_dir[:tr_end], abs_ret_bps[:tr_end], taker_cost,
            require_pnl=False)
        if not good:
            # No profitable clusters from train in this fold → no signal.
            continue

        lab_te = _assign(model, imp, scl, pca, ev_raw[te_s:te_e])
        folds.append(dict(
            f=f, tr_end=tr_end, te_s=te_s, te_e=te_e,
            lab_tr=lab_tr, lab_te=lab_te, good=good, majority=majority,
        ))

    if not folds:
        return {}

    # 3) Per lookback: pre-event classifier per fold (train→test), collect OOS.
    results = {}
    for lb in lookbacks:
        pre_all = ev_idx - lb
        valid_all = (pre_all >= 0) & (pre_all < n_rows)  # event validity for this lb

        oos_pred = np.full(n_ev, -1, dtype=int)
        oos_good = np.full(n_ev, -1, dtype=int)
        oos_lab  = np.full(n_ev, -1, dtype=int)
        fold_rows = []

        for fd in folds:
            tr_end, te_s, te_e = fd["tr_end"], fd["te_s"], fd["te_e"]
            good, majority = fd["good"], fd["majority"]

            # is_good from the TRAIN good-set (no future knowledge)
            is_good_tr = np.isin(fd["lab_tr"], list(good)).astype(int)
            is_good_te = np.isin(fd["lab_te"], list(good)).astype(int)

            # Pre-event features; use only events with a valid pre_index
            v_tr = valid_all[:tr_end]
            v_te = valid_all[te_s:te_e]
            if v_tr.sum() < 100 or v_te.sum() < 50:
                continue
            Xtr = X_full[(ev_idx[:tr_end])[v_tr] - lb]
            Xte = X_full[(ev_idx[te_s:te_e])[v_te] - lb]
            ytr = is_good_tr[v_tr]
            yte = is_good_te[v_te]
            if ytr.sum() < 20 or yte.sum() < 10:
                continue

            imp2 = SimpleImputer(strategy="median")
            Xtr_c = imp2.fit_transform(Xtr)
            Xte_c = imp2.transform(Xte)

            clf = lgb.LGBMClassifier(
                n_estimators=500, num_leaves=31, max_depth=6,
                learning_rate=0.05, colsample_bytree=0.5, subsample=0.8,
                class_weight="balanced", n_jobs=n_jobs, verbose=-1,
                random_state=random_state,
            )
            n_val = max(int(len(Xtr_c) * 0.1), 50)

            y_fit = ytr[:-n_val]
            y_val = ytr[-n_val:]

            if len(np.unique(y_fit)) < 2:
                print(f"        [honest-cv] fold {fd['f']+1} lb={lb}: train split has only one class — skipping")
                continue

            if len(np.unique(y_val)) < 2:
                print(f"        [honest-cv] fold {fd['f']+1} lb={lb}: validation split has only one class — fitting without early stopping")
                clf.fit(Xtr_c, ytr)
            else:
                clf.fit(
                    Xtr_c[:-n_val], y_fit,
                    eval_set=[(Xtr_c[-n_val:], y_val)],
                    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
                )

            preds = clf.predict(Xte_c)

            # Fill OOS arrays (at the positions of the valid test events)
            te_pos = np.arange(te_s, te_e)[v_te]
            oos_pred[te_pos] = preds
            oos_good[te_pos] = yte
            oos_lab[te_pos]  = fd["lab_te"][v_te]

            # IMPORTANT: lift against the TRAIN good-rate (stable reference), NOT
            # against yte.mean(). The test good-rate depends on the per-fold
            # on the cluster assignment and collapses when clusters are temporally
            # unstable → artificially inflated lift. The train rate is
            # the prevalence the classifier actually selects against.
            base_tr = float(ytr.mean())                 # stable reference
            base_te = float(yte.mean())                 # diagnostics only
            prec = float(yte[preds == 1].mean()) if preds.sum() > 0 else 0.0
            rec  = float(preds[yte == 1].mean()) if yte.sum() > 0 else 0.0
            fold_rows.append(dict(
                fold=fd["f"] + 1, n_test=int(v_te.sum()),
                precision=round(prec, 4), recall=round(rec, 4),
                base_rate_train=round(base_tr, 4),
                base_rate_test=round(base_te, 4),
                cluster_stability=round(base_te / base_tr, 3) if base_tr > 0 else 0.0,
                lift=round(prec / base_tr, 2) if base_tr > 0 else 0.0,
            ))
            del clf

        if not fold_rows:
            results[lb] = None
            continue

        # OOS predicted-good → filtered trades
        sel = (oos_pred == 1)
        filt_pos = np.where(sel)[0]
        filt_idx = ev_idx[filt_pos]
        filt_lab = oos_lab[filt_pos]
        # Direction = train majority of the cluster from the fold in which the event was test.
        # majority is fold-specific; we map via the dict stored per fold.
        # For this, position→fold lookup:
        dir_map = np.zeros(n_ev, dtype=int)
        for fd in folds:
            for p in range(fd["te_s"], fd["te_e"]):
                lab = oos_lab[p]
                if lab >= 0:
                    dir_map[p] = fd["majority"].get(int(lab), 0)
        filt_dir = dir_map[filt_pos]

        valid_dir = filt_dir != 0
        results[lb] = dict(
            filt_idx=filt_idx[valid_dir],
            filt_dirs=filt_dir[valid_dir],
            filt_labels=filt_lab[valid_dir],
            filt_pos=filt_pos[valid_dir],          # NEW: position in the event array
            fold_rows=fold_rows,
            oos_is_good=oos_good,
        )

    return results


# ─────────────────────────────────────────────────────────────────────────────

# ═══════════════════════════════════════════════════════════════════


def _bundle_columns_to_drop(feat_names, exclude_bundles):
    """
    Returns the column indices whose feature, per feature_keep.csv, belongs to one
    of the bundles to exclude. For the microstructure-only test
    (e.g. exclude_bundles=['B6_context'] removes EMAs, day/week levels, trend).
    """
    import pandas as pd
    from pathlib import Path as _P
    fk_path = None
    for cand in [_P("results/selection/feature_keep.csv"),
                 _P("feature_keep.csv")]:
        if cand.exists():
            fk_path = cand; break
    if fk_path is None:
        logger.warning("feature_keep.csv not found for bundle exclusion — "
                       "no feature removed")
        return []
    fk = pd.read_csv(fk_path)
    excl = set(exclude_bundles)
    drop_names = set(fk[fk["bundle"].isin(excl)]["column"])
    return [i for i, name in enumerate(feat_names) if name in drop_names]



def _load_family_map(csv_path):
    """Map full feature column name -> (family, bundle, bare_name) from the
    feature_keep CSV, for labelling cluster feature signatures."""
    if not csv_path:
        return {}
    import pandas as _pd
    df = _pd.read_csv(csv_path)
    fam = {}
    for _, r in df.iterrows():
        fam[str(r["column"])] = (str(r.get("family", "")),
                                 str(r.get("bundle", "")),
                                 str(r.get("bare_name", "")))
    return fam


def _screen_best_ratio(labels, mfe_raw, mae_raw, rcont, min_n):
    """Max over clusters of the direction-agnostic MFE/MAE ratio, computed from
    the PRECOMPUTED per-event breakout-frame excursions (mfe_raw = max of the
    cumulative path, mae_raw = min of it, both as returned by compute_mfe_mae,
    NOT floored). This reproduces the screening block EXACTLY:
      trade_dir = +1 (continuation): fav = mean(mfe_raw),  adv = mean(|mae_raw|)
      trade_dir = -1 (reversal):     fav = mean(-mae_raw), adv = mean(|mfe_raw|)
    (a reversal flips the path, so MFE_-= -mae_raw and |MAE_-| = |mfe_raw|).
    One definition for BOTH the observed statistic and the permutation null."""
    best = -np.inf
    for c in np.unique(labels):
        m = labels == c
        if m.sum() < min_n:
            continue
        if (rcont[m] > 0).mean() >= 0.5:
            fav = mfe_raw[m].mean(); adv = np.abs(mae_raw[m]).mean()
        else:
            fav = (-mae_raw[m]).mean(); adv = np.abs(mfe_raw[m]).mean()
        if adv > 0:
            r = fav / adv
            if r > best:
                best = r
    return float(best) if np.isfinite(best) else float("nan")


def _permutation_null_p(labels, mfe_raw, mae_raw, rcont, min_n,
                        n_perm, seed):
    """Permutation p-value for a configuration's BEST-cluster MFE/MAE ratio
    (professor's point 1: the max over the grid is inflated even under no
    effect). Cluster labels are shuffled among events, preserving cluster
    sizes, so the null is 'same partition sizes, random membership'. The
    in-sample trade_dir choice is re-made in every permutation, so the null
    also absorbs the direction-selection freedom. Returns
    (obs_best, p_value, null_p50, null_p95, null_p99)."""
    obs = _screen_best_ratio(labels, mfe_raw, mae_raw, rcont, min_n)
    nan = float("nan")
    if not np.isfinite(obs):
        return obs, nan, nan, nan, nan
    rng = np.random.RandomState(seed)
    lab = labels.copy()
    null = np.empty(n_perm, dtype=float)
    for b in range(n_perm):
        rng.shuffle(lab)
        null[b] = _screen_best_ratio(lab, mfe_raw, mae_raw, rcont, min_n)
    null = null[np.isfinite(null)]
    if null.size == 0:
        return obs, nan, nan, nan, nan
    p = (1.0 + float(np.sum(null >= obs))) / (null.size + 1.0)
    return (obs, p, float(np.percentile(null, 50)),
            float(np.percentile(null, 95)), float(np.percentile(null, 99)))


def _cluster_feature_signature(X_events, labels, cluster_ids, feat_names,
                               fam_map, min_n, top_k=50):
    """Per cluster: the top_k features by centroid z-score, i.e. the
    standardized deviation of the cluster mean from the global mean. Describes
    which features define each cluster, for comparison against the LGBM feature
    profile (family level). Returns a list of dict rows."""
    gmean = X_events.mean(axis=0)
    gstd = X_events.std(axis=0).copy()
    gstd[gstd == 0] = 1.0
    rows = []
    for c in cluster_ids:
        m = labels == c
        nc = int(m.sum())
        if nc < min_n:
            continue
        cmean = X_events[m].mean(axis=0)
        z = (cmean - gmean) / gstd
        order = np.argsort(-np.abs(z))[:top_k]
        for rank, fi in enumerate(order, 1):
            name = feat_names[fi] if fi < len(feat_names) else f"col{fi}"
            fam, bun, bare = fam_map.get(name, ("", "", name))
            rows.append(dict(
                cluster=int(c), n=nc, rank=rank, feature=name, bare_name=bare,
                family=fam, bundle=bun, z_score=round(float(z[fi]), 4),
                cluster_mean=round(float(cmean[fi]), 6),
                global_mean=round(float(gmean[fi]), 6)))
    return rows


def run_ws4(
    assets=("btc",),
    horizons=("5s", "15s"),
    thresholds=None,
    lookbacks=None,
    cluster_methods=CLUSTER_METHODS,
    pca_variants=PCA_VARIANTS,
    n_folds=5,
    n_jobs=8,
    data_source: str = 's5_reduced',
    data_dir=None,
    max_files: int = 0,
    max_hours=None,
    skip_exit: bool = False,
    make_plots: bool = False,   # exploratory screening: PNGs OFF by default
    exclude_bundles=None,       # e.g. ['B6_context'] for microstructure-only
    k_override=None,            # overrides the k table (k scan)
    silhouette_only=False,      # lean k scan: only KMeans+silhouette
    silhouette_k_list=None,     # k values for the silhouette scan
    no_da_gate: bool = False,   # grid screening: no DA/PnL gate, only MFE lift
    config_filter=None,         # set of (asset,hz,thr,pca_dim,k) -> only these
    out_subdir: str = "cluster_mfe",   # output subfolder
    family_map=None,            # feature -> (family,bundle,bare) for the signature
    n_perm: int = 1000,         # permutations for the screening null (point 1)
    perm_seed: int = 42,
    dump_only: bool = False,    # only load+cluster+per-event dump (for BCa)
    mfe_windows: bool = False,  # MFE/MAE per cluster over several windows (15..300s)
    full_select: bool = False,  # combined: DA + BCa dump + windowed MFE in ONE run
):
    if data_source == 's5_reduced':
        from common.data_loader import load_dataset
    from common.config import RESULTS_DIR, SPREAD_BPS, MAKER_COST_BPS
    from sklearn.preprocessing import StandardScaler  # PROJECT STANDARD — not RobustScaler
    from sklearn.impute import SimpleImputer
    from sklearn.decomposition import PCA
    import lightgbm as lgb

    # thresholds=None → use the per-horizon defaults (HORIZON_THRESHOLDS).
    # An explicit list overrides for every horizon.
    if lookbacks is None:
        lookbacks = LOOKBACKS
    if full_select:
        # One-pass re-selection: run the DA branch (no_da_gate=False), dump
        # per-event BCa data, AND compute windowed MFE — for EVERY config.
        mfe_windows = True
    if not cluster_methods:
        cluster_methods = CLUSTER_METHODS
    if not pca_variants:
        pca_variants = PCA_VARIANTS

    out_dir = RESULTS_DIR / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    for asset in assets:
        taker_cost = SPREAD_BPS.get(asset, {}).get("fut", 10.0)
        maker_cost = MAKER_COST_BPS.get(asset, {}).get("fut", 4.0)

        for hz in horizons:
            t0 = time.time()
            print(f"\n{'━'*70}")
            print(f"  WS4 — {asset.upper()}/{hz}")
            print(f"{'━'*70}")

            # ── PHASE A, Step 1: Load data ────────────────────────────────
            if data_source == "s6_full":
                _dir = data_dir or "data_storage/s6_features_s5_full"
                try:
                    X, y, y_1s, feat_names = load_s6_full(asset, hz, _dir, max_files=max_files)
                    info = None
                    logger.info("S6-full: %d rows x %d features", len(X), X.shape[1])
                except Exception as e:
                    logger.error("S6 load fail %s/%s: %s", asset, hz, e)
                    continue
            else:
                # RAM guard: estimate matrix size before the loader's big
                # preallocation. With --max-hours we cap rows; otherwise assume
                # the full ~7.0M-row dataset. Skip the config (don't crash the
                # host) if memory is insufficient.
                est_rows = (max_hours * 3600) if max_hours else 7_020_000
                if not check_ram_or_skip(est_rows, n_features=2815):
                    continue
                # Native load only (synthetic horizons removed). In the DA run
                # (config-list set) we also pull all return horizons as aligned
                # aux columns on `info`, for the multi-horizon Cluster-DA.
                try:
                    X, y, info, feat_names = load_dataset(
                        target=hz, asset=asset, profile="cluster",
                        max_hours=max_hours,
                        aux_targets=(AUX_RET_TARGETS
                                     if (full_select or config_filter is not None)
                                     else None))
                except Exception as e:
                    logger.error("Load fail %s/%s: %s", asset, hz, e)
                    continue
                if hz != "1s":
                    try:
                        _, y_1s, _, _ = load_dataset(
                            target="1s", asset=asset,
                            max_hours=max_hours, target_only=True)
                        n_min = min(len(X), len(y_1s))
                        X, y, y_1s = X[:n_min], y[:n_min], y_1s[:n_min]
                        if info is not None:
                            info = info.iloc[:n_min]
                    except:
                        y_1s = y
                else:
                    y_1s = y

            n = len(X)
            print(f"  Data: {n:,} samples, {X.shape[1]} features")

            # ── Optional bundle exclusion (microstructure-only test) ───────
            # Hypothesis: Chart-Structure-Features (B6_context: EMAs, day/week-
            # levels) dominate the clusters and make them temporally unstable +
            # asset-unspecific. With --exclude-bundles B6_context exactly
            # these columns are removed BEFORE clustering, to test whether pure
            # microstructure yields more stable, more asset-specific clusters.
            if exclude_bundles:
                drop_cols = _bundle_columns_to_drop(feat_names, exclude_bundles)
                if drop_cols:
                    keep_mask = np.ones(X.shape[1], dtype=bool)
                    keep_mask[drop_cols] = False
                    X = X[:, keep_mask]
                    feat_names = [f for i, f in enumerate(feat_names) if keep_mask[i]]
                    print(f"  Bundle exclusion {exclude_bundles}: "
                          f"{len(drop_cols)} features removed → {X.shape[1]} remain")

            # Per-horizon thresholds: explicit --thresholds overrides for every
            # horizon; otherwise use the curated HORIZON_THRESHOLDS defaults.
            cell_thresholds = thresholds if thresholds else \
                HORIZON_THRESHOLDS.get(hz, [10, 15, 20])
            print(f"  Thresholds for {hz}: {cell_thresholds} bps")

            for thr_bps in cell_thresholds:
                thr_dec = thr_bps / 10_000

                # ── PHASE A, Step 2: Identify breakout events (TRAILING) ──
                # A breakout is a move that has ALREADY happened by time T:
                # the absolute return over the PAST `hz` seconds exceeds the
                # threshold. y = ret_fwd_{hz} is the FORWARD move over [t, t+hz];
                # the move over the PAST window [T-hz, T] is therefore y[T-hz],
                # i.e. y shifted forward by `hz` rows (1 row = 1 second, the
                # dataset's standing convention). The first `hz` rows have no
                # complete trailing window and are excluded. (Verified equal to
                # the cumulative 1s-return over the same window.)
                hz_sec = int(str(hz).rstrip("s"))
                trailing_move = np.full_like(y, np.nan)
                if hz_sec < len(y):
                    trailing_move[hz_sec:] = y[:len(y) - hz_sec]
                event_mask = np.abs(trailing_move) > thr_dec
                event_indices = np.where(event_mask)[0]
                n_events = len(event_indices)

                if n_events < 500:
                    print(f"  {thr_bps} bps: only {n_events} events — skipping")
                    continue

                tprint(f"── {thr_bps} bps: {n_events} breakout events (trailing) ──")

                # ── PHASE A, Step 3: Cluster the COMPLETED breakouts ──────
                # Features are read AT T, the moment the breakout has just
                # completed (the predictive pipeline below reads them at
                # T-lookback, i.e. 1-5 s earlier; see `pre_all`).
                X_events = X[event_indices].copy()
                # Direction of the COMPLETED breakout move, OBSERVABLE at T
                # (no look-ahead). It is the side the continuation is measured
                # against; the forward move y[T] stays untouched and serves
                # downstream as the CONTINUATION the strategy trades.
                event_directions = np.sign(trailing_move[event_indices])

                nan_frac = np.isnan(X_events).mean(axis=0)
                valid_cols = np.where(nan_frac < 0.95)[0]
                X_ev_clean = X_events[:, valid_cols]
                # RAW event features (with NaN, BEFORE impute/scale) for the
                # Train-only fold-internal pipeline. The global impute/
                # scale/PCA below only serves the DESCRIPTIVE clustering
                # (overview plot), not the predictive lift/precision figures.
                X_ev_raw = X_events[:, valid_cols].copy()

                imputer_cl = SimpleImputer(strategy="median")
                scaler_cl  = StandardScaler()  # PROJECT STANDARD, see _fit_pipeline_train (NOT RobustScaler)
                X_ev_clean  = imputer_cl.fit_transform(X_ev_clean)
                X_ev_scaled = scaler_cl.fit_transform(X_ev_clean)
                X_ev_scaled = np.nan_to_num(X_ev_scaled, nan=0.0)

                # ── PHASE A, Step 3: cluster under EACH PCA variant ───────
                # The scaled event matrix is shared; each PCA variant either
                # reduces it to N components or (— "none" —) clusters on the
                # full feature space. PCA is fit once per variant per cell.
                for pca_variant in pca_variants:
                    if pca_variant == "none":
                        X_cluster = X_ev_scaled
                        pca_tag = "none"
                        print(f"\n  ┄┄ PCA variant: none "
                              f"(full {X_cluster.shape[1]}-dim space) ┄┄")
                    else:
                        n_comp = min(int(pca_variant),
                                     X_ev_scaled.shape[1], n_events - 1)
                        _svd = ("full" if _os.environ.get("WS4_DETERMINISTIC") == "1"
                                else "auto")
                        pca = PCA(n_components=n_comp, random_state=42,
                                  svd_solver=_svd)
                        X_cluster = pca.fit_transform(X_ev_scaled)
                        pca_tag = f"pca{int(pca_variant)}"
                        evr = pca.explained_variance_ratio_.sum()
                        print(f"\n  ┄┄ PCA variant: {n_comp} comps "
                              f"(EVR={evr:.1%}) ┄┄")

                    # ── SILHOUETTE-ONLY fast path ─────────────────────────
                    # Data + PCA are built ONCE above. Here only
                    # KMeans + silhouette over all k, then CSV append and
                    # leave this pca variant — the entire MFE/DA/trade/
                    # the plotting apparatus is skipped.
                    if silhouette_only:
                        from sklearn.metrics import silhouette_score
                        sil_path = out_dir / f"silhouette_{asset}_{hz}.csv"
                        if not sil_path.exists():
                            sil_path.write_text(
                                "asset,hz,thr_bps,pca_tag,k,n_events,silhouette\n")
                        for k_sil in (silhouette_k_list or []):
                            try:
                                lbl, _k_eff = cluster_events(
                                    X_cluster, method="kmeans",
                                    k=int(k_sil), random_state=42)
                            except Exception as e:
                                print(f"      [silhouette] k={k_sil}: failed "
                                      f"({type(e).__name__}: {e}) — skip")
                                continue
                            uniq = set(lbl.tolist()) - {-1}
                            if len(uniq) < 2:
                                sil = float("nan")
                            else:
                                ssz = min(5000, X_cluster.shape[0])
                                sil = float(silhouette_score(
                                    X_cluster, lbl,
                                    sample_size=ssz, random_state=42))
                            with sil_path.open("a") as fh:
                                fh.write(f"{asset},{hz},{thr_bps},{pca_tag},"
                                         f"{int(k_sil)},{n_events},{sil:.5f}\n")
                            print(f"      [silhouette] k={k_sil:>2}  "
                                  f"sil={sil:.4f}  (n_events={n_events}, "
                                  f"pca={pca_tag})")
                        continue   # next pca variant; skip the method loop
                    # ──────────────────────────────────────────────────────

                    # k override (k scan): append k to pca_tag so tags +
                    # .done markers differ per k and do not collide.
                    if k_override is not None:
                        pca_tag = f"{pca_tag}_k{int(k_override)}"

                    # ── cluster with EACH requested method ────────────────
                    for method in cluster_methods:
                        # ── Checkpoint: skip if this
                        #    (asset, method, pca, threshold) is fully done.
                        #    The .done file is written only after every
                        #    lookback finished → reliable completion flag.
                        done_marker = out_dir / (
                            f".done_{method}_{pca_tag}_{asset}_{hz}_{thr_bps}bps")
                        if done_marker.exists():
                            print(f"\n  ── SKIP {method}/{pca_tag}/{thr_bps}bps "
                                  f"— already done ──")
                            continue

                        k = resolve_k(hz, thr_bps, n_events, k_override=k_override)
                        if config_filter is not None:
                            _pca_dim = int(pca_variant) if str(pca_variant) != "none" else 0
                            if (asset, hz, int(thr_bps), _pca_dim, int(k)) not in config_filter:
                                continue
                        tprint(f"━━ method={method}  pca={pca_tag}  thr={thr_bps}bps  (k={k}) ━━")
                        try:
                            cluster_labels, k_eff = cluster_events(
                                X_cluster, method=method, k=k, random_state=42)
                        except ImportError as e:
                            print(f"    {method} unavailable: {e} — skipping")
                            continue
                        except Exception as e:
                            print(f"    {method}/{pca_tag} failed: "
                                  f"{type(e).__name__}: {e} — skipping")
                            continue

                        cluster_ids = sorted(set(cluster_labels.tolist()) - {-1})
                        n_noise = int((cluster_labels == -1).sum())
                        if not cluster_ids:
                            print(f"    {method}: no clusters found "
                                  f"({n_noise} noise) — marking done (empty)")
                            (out_dir / f".done_{method}_{pca_tag}_{asset}_{hz}_"
                                       f"{thr_bps}bps").write_text(
                                "no_clusters\n")
                            continue
                        print(f"    {method}: {len(cluster_ids)} clusters, "
                              f"{n_noise} noise points")

                        # Silhouette on the actual partition (Appendix D / 4.4.3):
                        # a task-independent internal-validity measure, appended
                        # per config. Cheap; sampled for large event sets.
                        try:
                            from sklearn.metrics import silhouette_score as _sil_fn
                            _sil_path = out_dir / f"cluster_silhouette_{asset}_{hz}.csv"
                            if not _sil_path.exists():
                                _sil_path.write_text(
                                    "method,pca_tag,asset,hz,thr_bps,k,n_events,"
                                    "n_clusters,silhouette\n")
                            if len(cluster_ids) >= 2:
                                _ssz = min(5000, X_cluster.shape[0])
                                _sil = float(_sil_fn(
                                    X_cluster, cluster_labels,
                                    sample_size=_ssz, random_state=42))
                            else:
                                _sil = float("nan")
                            with open(_sil_path, "a") as _sf:
                                _sf.write(
                                    f"{method},{pca_tag},{asset},{hz},{thr_bps},"
                                    f"{k},{len(cluster_labels)},{len(cluster_ids)},"
                                    f"{_sil:.4f}\n")
                        except Exception as _e:
                            print(f"    [silhouette] skipped "
                                  f"({type(_e).__name__}: {_e})")

                        # ── PHASE A, Step 4: Cluster screening / selection ──
                        # cluster_stats keeps a fixed set of keys (da, majority,
                        # mean_abs_ret, est_pnl_taker, n, n_test, da_oos,
                        # est_pnl_oos) so the overview plot, MFE phase and trade
                        # dump downstream work unchanged in both branches.
                        if no_da_gate:
                            # GRID SCREENING run: NO directional/PnL gate. Every
                            # cluster with >= MIN_CLUSTER_EVENTS members is kept so
                            # the configuration is screened purely by MFE-lift
                            # (excursion separation). Directional accuracy and PnL
                            # selection are a SEPARATE, later step (viable-cluster
                            # evaluation); keeping them out here avoids selecting
                            # configurations on the very metric reported afterwards.
                            good_clusters = sorted(
                                c for c in cluster_ids
                                if (cluster_labels == c).sum() >= MIN_CLUSTER_EVENTS)
                            cluster_stats = {}
                            for c in cluster_ids:
                                full_mask = cluster_labels == c
                                maj = 1 if (event_directions[full_mask] > 0).mean() > 0.5 else -1
                                mean_abs_ret = np.abs(y[event_indices[full_mask]]).mean() * 10_000
                                cluster_stats[c] = dict(
                                    da=float("nan"), majority=maj,
                                    mean_abs_ret=mean_abs_ret,
                                    est_pnl_taker=float("nan"),
                                    n=int(full_mask.sum()), n_test=0,
                                    da_oos=float("nan"), est_pnl_oos=float("nan"),
                                )
                            print(f"    Clusters: {len(cluster_ids)}  "
                                  f"(MFE-lift screening, no DA gate)")
                            for c, st_ in sorted(cluster_stats.items()):
                                mark = " *" if c in good_clusters else ""
                                print(f"      Cl {c}: N={st_['n']:>5}, "
                                      f"majority={st_['majority']:+d}{mark}")
                            print(f"    Screened clusters (n>={MIN_CLUSTER_EVENTS}): "
                                  f"{good_clusters}")
                        else:
                            # HONEST directed-OOS-DA selection in the CONTINUATION
                            # FRAME. The breakout direction is observed at T, so the
                            # quantity to predict is whether the forward move continues
                            # the breakout or reverses it: r_cont = breakout_dir *
                            # forward_move. sign(r_cont) is the label; the cluster's
                            # train-majority sign is fixed on a TRAIN slice and its DA
                            # is measured on a held-out TEST slice (60/40 forward
                            # split). This DA equals the LGBM sign(ret_fwd) accuracy
                            # (metrics.compute_fold_metrics), so the two are directly
                            # comparable. Gate is DA-only (require_pnl=False); PnL is
                            # reported but decided in Section 4.5.
                            from prediction.cluster_eval import select_good_clusters_oos
                            _n_ev = len(cluster_labels)
                            _split = int(_n_ev * 0.60)
                            _lab_tr, _lab_te = cluster_labels[:_split], cluster_labels[_split:]
                            _r_cont = event_directions * y[event_indices]
                            _dir_all = np.sign(_r_cont)
                            _dir_tr, _dir_te = _dir_all[:_split], _dir_all[_split:]
                            _ret_bps = _r_cont * 10_000.0
                            good_set, majority_map, oos_stats = select_good_clusters_oos(
                                _lab_tr, _dir_tr, _lab_te, _dir_te, _ret_bps[_split:],
                                taker_cost=taker_cost, da_min=0.55,
                                min_n_test=MIN_CLUSTER_EVENTS,
                                min_n_train=MIN_CLUSTER_EVENTS,
                                require_pnl=False,
                            )

                            # ── Multi-horizon Cluster-DA (regime persistence) ──
                            # The train-fixed majority direction (from the breakout
                            # window) is tested on EVERY return horizon 1s..900s,
                            # on the same held-out test split. DA up to 60s is
                            # comparable to the LGBM baseline; 120s..900s probe
                            # whether the directional signal persists beyond the
                            # window. Direction is NOT re-fixed per horizon.
                            from prediction.cluster_eval import directed_oos_da
                            _datag = f"{method}_{pca_tag}_{asset}_{hz}_{thr_bps}bps"
                            _pca_dim_da = (int(pca_variant)
                                           if str(pca_variant) != "none" else 0)
                            # Per-event r_cont and raw y_h for every horizon,
                            # aligned to event_indices (NaN where a long horizon
                            # runs past the file end). Reused for the DA and dumped
                            # raw for the offline BCa bootstrap.
                            _hz_list = [h for h in AUX_RET_HZ
                                        if (info is not None
                                            and f"ret_{h}" in info.columns)]
                            _yh_mat = np.column_stack(
                                [info[f"ret_{h}"].to_numpy()[event_indices]
                                 for h in _hz_list]) if _hz_list else None
                            _rcont_mat = (event_directions[:, None] * _yh_mat
                                          if _yh_mat is not None else None)
                            _mh_rows = []
                            for _hi, _h in enumerate(_hz_list):
                                _rc_te = _rcont_mat[_split:, _hi]
                                _dh_te = np.sign(_rc_te)
                                for _c, _maj in majority_map.items():
                                    # valid = cluster test events with a finite,
                                    # non-zero return at THIS horizon (NaN at long
                                    # horizons is excluded, not counted as a miss).
                                    _sel = ((_lab_te == _c) & np.isfinite(_rc_te)
                                            & (_rc_te != 0))
                                    _nv = int(_sel.sum())
                                    if _nv < MIN_CLUSTER_EVENTS:
                                        continue
                                    _da = float((_dh_te[_sel] == np.sign(_maj)).mean())
                                    _mh_rows.append(dict(
                                        method=method, pca=pca_tag,
                                        pca_dim=_pca_dim_da, asset=asset, hz=hz,
                                        thr_bps=thr_bps, k=k, cluster=int(_c),
                                        trade_dir=int(_maj), horizon=_h,
                                        da_oos=round(_da, 4), n_test=_nv))
                            if _mh_rows:
                                pd.DataFrame(_mh_rows).to_csv(
                                    out_dir / f"cluster_da_multihz_{_datag}.csv",
                                    index=False)
                                print(f"    Multi-hz DA-A -> "
                                      f"cluster_da_multihz_{_datag}.csv "
                                      f"({len(majority_map)} clusters x "
                                      f"{len(_hz_list)} hz)")

                            # ── Member dump (event-overlap + offline BCa) ──
                            # Raw per-event arrays: r_cont and y_h for every
                            # horizon (NaN preserved), plus event_directions, so
                            # the BCa bootstrap can be run entirely offline.
                            np.savez_compressed(
                                out_dir / f"cluster_members_{_datag}.npz",
                                event_indices=np.asarray(event_indices),
                                cluster_labels=np.asarray(cluster_labels),
                                cluster_ids=np.asarray(cluster_ids),
                                majority=np.array([majority_map.get(int(c), 0)
                                                   for c in cluster_ids]),
                                event_directions=np.asarray(event_directions),
                                split=int(_split),
                                horizons=np.array(_hz_list),
                                r_cont_by_hz=(_rcont_mat if _rcont_mat is not None
                                              else np.empty((len(event_indices), 0))),
                                y_h_by_hz=(_yh_mat if _yh_mat is not None
                                           else np.empty((len(event_indices), 0))))

                            # In --dump-only we stop here: the per-event arrays and
                            # the DA are all we need. Skip signature, screening,
                            # classifier and the MFE phase.
                            if dump_only:
                                (out_dir / f".done_{method}_{pca_tag}_{asset}_"
                                           f"{hz}_{thr_bps}bps").write_text("dumped\n")
                                continue

                            good_clusters = sorted(good_set)
                            cluster_stats = {}
                            for c in cluster_ids:
                                full_mask = cluster_labels == c
                                st = oos_stats.get(int(c), {})
                                maj = majority_map.get(
                                    int(c),
                                    1 if (event_directions[full_mask] > 0).mean() > 0.5 else -1)
                                mean_abs_ret = np.abs(y[event_indices[full_mask]]).mean() * 10_000
                                cluster_stats[c] = dict(
                                    da=st.get("da_oos", float("nan")),
                                    majority=maj, mean_abs_ret=mean_abs_ret,
                                    est_pnl_taker=st.get("est_pnl_oos", float("nan")),
                                    n=int(full_mask.sum()),
                                    n_test=int(st.get("n_test", 0)),
                                    da_oos=st.get("da_oos", float("nan")),
                                    est_pnl_oos=st.get("est_pnl_oos", float("nan")),
                                )
                            print(f"    Clusters: {len(cluster_ids)}  "
                                  f"(directed OOS DA, 60/40 forward split)")
                            for c, st_ in sorted(cluster_stats.items()):
                                mark = " *" if c in good_clusters else ""
                                print(f"      Cl {c}: N={st_['n']:>5} (n_test={st_['n_test']:>4}), "
                                      f"DA_oos={st_['da']:.3f}, "
                                      f"PnL_oos={st_['est_pnl_taker']:+.1f} bps{mark}")
                            print(f"    Good clusters (OOS): {good_clusters}")

                        # Feature signature per cluster (centroid z-score, top 50)
                        # for comparison against the LGBM feature profile (family).
                        if family_map is not None:
                            _sig_rows = _cluster_feature_signature(
                                X_events, cluster_labels, cluster_ids,
                                feat_names, family_map,
                                min_n=MIN_CLUSTER_EVENTS, top_k=50)
                            if _sig_rows:
                                _sig_tag = f"{method}_{pca_tag}_{asset}_{hz}_{thr_bps}bps"
                                pd.DataFrame(_sig_rows).to_csv(
                                    out_dir / f"cluster_signature_{_sig_tag}.csv",
                                    index=False)
                                _ncl = len(set(r["cluster"] for r in _sig_rows))
                                print(f"    Signature -> cluster_signature_{_sig_tag}.csv "
                                      f"({_ncl} clusters x top50)")

                        if not good_clusters and not full_select:
                            print("    No profitable clusters — marking done (empty)")
                            (out_dir / f".done_{method}_{pca_tag}_{asset}_{hz}_"
                                       f"{thr_bps}bps").write_text(
                                "no_profitable_clusters\n")
                            continue
                        if not good_clusters:
                            # full_select: no DA>0.55 clusters, but screening
                            # (windowed MFE) + DA table + dump should nevertheless
                            # run — the selection happens offline.
                            print("    No DA>0.55 clusters — screening/dump only "
                                  "(full-select)")

                        # Phase C: Cluster overview plot
                        if make_plots and good_clusters:
                          plot_cluster_overview(cluster_stats, good_clusters,
                                             taker_cost, maker_cost,
                                             f"{method}_{pca_tag}_{asset}_{hz}_{thr_bps}bps", out_dir)

                        # ── PHASE B baseline: MFE for ALL breakouts ───────────────
                        print(f"\n    Baseline: MFE for all breakouts...")
                        all_paths = extract_price_paths(
                            y_1s, event_indices, event_directions, MFE_LOOKAHEAD)
                        all_mfe, all_mae, all_term, all_mfet = compute_mfe_mae(all_paths)
                        # 4.4.3 screening is defined on the SIGNAL-RELEVANT 60 s
                        # window (the 300 s lookahead overstates adverse excursion
                        # for a 15-60 s signal). The per-event 60 s excursions feed
                        # the baseline lift, the per-cluster ratio, and the
                        # permutation null so all three are consistent at 60 s.
                        SCREEN_WIN = 60
                        all_mfe60, all_mae60, _, _ = compute_mfe_mae(
                            all_paths[:, :SCREEN_WIN])
                        print(f"      Mean MFE={all_mfe.mean():.2f} bps (300s), "
                              f"{all_mfe60.mean():.2f} bps (60s), "
                              f"MFE>=taker: {(all_mfe60>=taker_cost).mean()*100:.1f}%")

                        # ── 4.4.3 SCREENING: direction-agnostic per-cluster MFE/MAE ──
                        # Each cluster is scored in ITS OWN dominant direction, fixed
                        # in-sample: trade_dir = majority sign of
                        # (breakout_dir * forward_move). +1 = continuation, -1 =
                        # reversal. The excursion is then measured in
                        # trade_dir * breakout_dir, so a reversal regime (price breaks
                        # up then falls) is scored on its true favourable side and can
                        # show a high MFE/MAE ratio just like a continuation cluster.
                        # This orients the DESCRIPTIVE screening only; the directional
                        # accuracy of Section 4.4.6 fixes the direction out-of-sample.
                        _base_mfe = float(all_mfe60.mean())   # 60 s baseline
                        _pca_dim = int(pca_variant) if str(pca_variant) != "none" else 0
                        screen_rows = []
                        window_rows = []
                        for _c in cluster_ids:
                            _cm = cluster_labels == _c
                            _nc = int(_cm.sum())
                            if _nc < MIN_CLUSTER_EVENTS:
                                continue
                            _rc = event_directions[_cm] * y[event_indices[_cm]]
                            _trade_dir = 1 if (_rc > 0).mean() >= 0.5 else -1
                            _pdir = _trade_dir * event_directions[_cm]
                            _cp = extract_price_paths(
                                y_1s, event_indices[_cm], _pdir, MFE_LOOKAHEAD)
                            # Screening ratio on the 60 s window; full 300 s path
                            # retained for the windowed breakdown below.
                            _cmfe, _cmae, _ct, _cmt = compute_mfe_mae(
                                _cp[:, :SCREEN_WIN])
                            if mfe_windows:
                                # Same trade-direction paths, truncated to each
                                # window: MFE/MAE over the first W seconds only.
                                for _W in MFE_WINDOWS:
                                    _wm, _wa, _, _ = compute_mfe_mae(_cp[:, :_W])
                                    _wmfe = float(_wm.mean())
                                    _wmae = float(np.abs(_wa).mean())
                                    window_rows.append(dict(
                                        method=method, pca=pca_tag, asset=asset,
                                        hz=hz, thr_bps=thr_bps, k=k, cluster=int(_c),
                                        trade_dir=int(_trade_dir), n=_nc,
                                        window_s=int(_W), mfe=round(_wmfe, 3),
                                        mae=round(_wmae, 3),
                                        ratio=(round(_wmfe / _wmae, 3)
                                               if _wmae > 0 else float("nan"))))
                            _mfe_mean = float(_cmfe.mean())
                            _mae_mean = float(np.abs(_cmae).mean())
                            screen_rows.append(dict(
                                method=method, pca=pca_tag, pca_dim=_pca_dim,
                                asset=asset, hz=hz, thr_bps=thr_bps, k=k,
                                cluster=int(_c), n=_nc,
                                mfe_mean_bps=round(_mfe_mean, 3),
                                mae_mean_bps=round(_mae_mean, 3),
                                mfe_mae_ratio=(round(_mfe_mean / _mae_mean, 3)
                                               if _mae_mean > 0 else float("nan")),
                                mfe_lift=(round(_mfe_mean / _base_mfe, 3)
                                          if _base_mfe > 0 else float("nan")),
                                all_breakout_mfe_bps=round(_base_mfe, 3),
                                trade_dir=int(_trade_dir)))
                        # Permutation null on the config's best-cluster ratio.
                        # Cheap: reuses the per-event all-breakout excursions
                        # (all_mfe/all_mae), no re-clustering, no path re-extraction.
                        if screen_rows and n_perm > 0:
                            _mfe_raw = np.asarray(all_mfe60, dtype=float)  # 60 s
                            _mae_raw = np.asarray(all_mae60, dtype=float)  # 60 s
                            _rc_all = event_directions * y[event_indices]
                            _obs, _pp, _n50, _n95, _n99 = _permutation_null_p(
                                cluster_labels, _mfe_raw, _mae_raw, _rc_all,
                                MIN_CLUSTER_EVENTS, n_perm, perm_seed)
                            for _r in screen_rows:
                                _r["perm_best_ratio"] = round(_obs, 3)
                                _r["perm_p"] = round(_pp, 4)
                                _r["null_ratio_p50"] = round(_n50, 3)
                                _r["null_ratio_p95"] = round(_n95, 3)
                                _r["null_ratio_p99"] = round(_n99, 3)
                                _r["n_perm"] = int(n_perm)
                        _scr_tag = f"{method}_{pca_tag}_{asset}_{hz}_{thr_bps}bps"
                        if screen_rows:
                            pd.DataFrame(screen_rows).sort_values(
                                "mfe_mae_ratio", ascending=False).to_csv(
                                out_dir / f"cluster_screen_{_scr_tag}.csv", index=False)
                            _rr = [r["mfe_mae_ratio"] for r in screen_rows
                                   if not np.isnan(r["mfe_mae_ratio"])]
                            _nrev = sum(1 for r in screen_rows if r["trade_dir"] == -1)
                            _pp_show = screen_rows[0].get("perm_p", float("nan"))
                            print(f"    Screening: {len(screen_rows)} clusters "
                                  f"({_nrev} reversal), best ratio="
                                  f"{(max(_rr) if _rr else float('nan')):.2f}, "
                                  f"perm_p={_pp_show} -> "
                                  f"cluster_screen_{_scr_tag}.csv")

                        if mfe_windows and window_rows:
                            pd.DataFrame(window_rows).to_csv(
                                out_dir / f"cluster_mfe_windows_{_scr_tag}.csv",
                                index=False)
                            print(f"    MFE/MAE windows -> "
                                  f"cluster_mfe_windows_{_scr_tag}.csv "
                                  f"({len(MFE_WINDOWS)} windows x {len(screen_rows)} clusters)")

                        # In the grid-screening run the DA/PnL gate, the classifier and
                        # the filtered-trade MFE phase are skipped: the config is fully
                        # characterised by the screening table above. Mark done + skip.
                        if no_da_gate or mfe_windows:
                            (out_dir / f".done_{method}_{pca_tag}_{asset}_{hz}_"
                                       f"{thr_bps}bps").write_text("screened\n")
                            continue

                        # ── PHASE A, Steps 5-7: TRAIN-ONLY per-fold clustering ──
                        # impute/scale/PCA, KMeans/GMM AND good-cluster selection
                        # now run PER FOLD on train only (see
                        # honest_cluster_classifier_cv). The global clustering
                        # the above stays purely DESCRIPTIVE (overview plot, in-sample).
                        # Returns honest filt_idx/dirs/labels per lookback +
                        # fold_rows (precision/lift), which never mix the label with
                        # contaminate with future knowledge.
                        tprint(f"  honest per-fold CV ({method}/{pca_tag}/{thr_bps}bps)...")
                        try:
                            cv = honest_cluster_classifier_cv(
                                X_full=X, X_ev_raw=X_ev_raw,
                                event_indices=event_indices,
                                event_dirs=np.sign(event_directions * y[event_indices]), y=y,
                                pca_variant=pca_variant, method=method, k=k,
                                lookbacks=lookbacks, taker_cost=taker_cost,
                                n_folds=n_folds, n_jobs=n_jobs,
                            )
                        except ValueError as e:
                            print(f"    honest-cv skip ({method}): {e}")
                            cv = {}

                        n_outputs_written = 0

                        for lookback in lookbacks:
                            tprint(f"  ── Lookback {lookback}s ──")

                            r = cv.get(lookback)
                            n_honest = len(r["filt_idx"]) if r else 0
                            if not r or n_honest < 20:
                                print(f"      No/too few honest filtered trades "
                                      f"({n_honest}) — skipping")
                                continue

                            filt_idx    = r["filt_idx"]
                            filt_dirs   = r["filt_dirs"]
                            filt_labels = r["filt_labels"]
                            filt_pos    = r["filt_pos"]
                            fold_rows   = r["fold_rows"]
                            # Coherent baseline label (one clustering, line ~1351)
                            # per filtered event; filt_pos indexes cluster_labels
                            # (both in ev_idx order).
                            cluster_coherent = cluster_labels[filt_pos]

                            mean_prec  = float(np.mean([fr["precision"] for fr in fold_rows]))
                            mean_lift  = float(np.mean([fr["lift"]      for fr in fold_rows]))
                            mean_btr   = float(np.mean([fr["base_rate_train"]  for fr in fold_rows]))
                            mean_stab  = float(np.mean([fr["cluster_stability"] for fr in fold_rows]))
                            print(f"      OOS (honest): precision={mean_prec:.3f}, "
                                  f"lift={mean_lift:.2f}x vs train-base={mean_btr:.3f} "
                                  f"(folds={len(fold_rows)})")
                            # DA-B (real-time identifiability, 4.4.4/4.4.5): clean
                            # per-config/lookback precision + lift, so the numbers
                            # come from a CSV, not from parsing the log.
                            _clf_path = out_dir / (
                                f"cluster_clf_{asset}_{hz}.csv")
                            if not _clf_path.exists():
                                _clf_path.write_text(
                                    "asset,hz,thr_bps,pca_tag,k,lookback,"
                                    "mean_precision,mean_lift,mean_base_rate,"
                                    "mean_stability,n_folds\n")
                            with open(_clf_path, "a") as _cf:
                                _cf.write(
                                    f"{asset},{hz},{thr_bps},{pca_tag},{k},"
                                    f"{lookback},{mean_prec:.4f},{mean_lift:.4f},"
                                    f"{mean_btr:.4f},{mean_stab:.4f},"
                                    f"{len(fold_rows)}\n")
                            # cluster_stability = Test-good-Rate / Train-good-Rate.
                            # ~1.0 = cluster temporally stable; << 1.0 = the profitable
                            # cluster disappears OOS → treat lift/precision with caution.
                            if mean_stab < 0.5:
                                print(f"      cluster_stability={mean_stab:.2f} "
                                      f"— good cluster temporally UNSTABLE (OOS prevalence "
                                      f"collapses); the economic MFE lift below is the "
                                      f"defensible metric, not the precision lift.")
                            else:
                                print(f"      cluster_stability={mean_stab:.2f}")

                            n_filt = len(filt_idx)
                            print(f"      Filtered (honest): {n_filt} trades")

                            # ══════════════════════════════════════════════════════
                            #  PHASE B: MFE ANALYSIS ON FILTERED TRADES
                            # ══════════════════════════════════════════════════════

                            tag = f"{method}_{pca_tag}_{asset}_{hz}_{thr_bps}bps_lb{lookback}"
                            tprint(f"  === Phase B: MFE analysis ({tag}) ===")

                            # ── Step 8: Extract tick-by-tick price paths ───────────
                            print(f"      Extracting price paths...")
                            paths_filt = extract_price_paths(
                                y_1s, filt_idx, filt_dirs, MFE_LOOKAHEAD)

                            # ── Step 9: Compute per-trade MFE/MAE ─────────────────
                            mfe_f, mae_f, term_f, mfet_f = compute_mfe_mae(paths_filt)
                            print(f"      Filtered: Mean MFE={mfe_f.mean():.2f}, "
                                  f"Median MFE={np.median(mfe_f):.2f}, "
                                  f"MFE>=taker={( mfe_f>=taker_cost).mean()*100:.1f}%")

                            # Random baseline (null hypothesis)
                            rng = np.random.RandomState(42)
                            n_rand = min(n_filt * 3, n - MFE_LOOKAHEAD - 1)
                            rand_idx  = rng.choice(n - MFE_LOOKAHEAD, n_rand, replace=False)
                            rand_dirs = rng.choice([-1, 1], n_rand)
                            paths_rand = extract_price_paths(y_1s, rand_idx, rand_dirs, MFE_LOOKAHEAD)
                            mfe_r, mae_r, term_r, mfet_r = compute_mfe_mae(paths_rand)

                            # ── Step 10: MFE distribution per cluster ─────────────
                            print(f"      MFE distribution per cluster...")
                            unique_cl = np.unique(filt_labels)
                            mfe_tables = []

                            for cl in unique_cl:
                                cmask = filt_labels == cl
                                nc = cmask.sum()
                                if nc < 5: continue
                                cm, ca, ct, cmt = compute_mfe_mae(paths_filt[cmask])
                                tbl = mfe_distribution_table(cm, ca, ct, cmt, taker_cost, maker_cost)
                                tbl.insert(0, "cluster", int(cl)); tbl.insert(0, "group", "filtered")
                                mfe_tables.append(tbl)
                                print(f"        Cl {cl}: N={nc}, MFE mean={cm.mean():.2f}, "
                                      f"P50={np.median(cm):.2f}, P90={np.percentile(cm,90):.2f}, "
                                      f">=taker {(cm>=taker_cost).mean()*100:.0f}%")

                            # Reference groups
                            tbl = mfe_distribution_table(mfe_f, mae_f, term_f, mfet_f, taker_cost, maker_cost)
                            tbl.insert(0, "cluster", "ALL_FILTERED"); tbl.insert(0, "group", "filtered")
                            mfe_tables.append(tbl)

                            tbl = mfe_distribution_table(all_mfe, all_mae, all_term, all_mfet, taker_cost, maker_cost)
                            tbl.insert(0, "cluster", "ALL_BREAKOUTS"); tbl.insert(0, "group", "all_breakouts")
                            mfe_tables.append(tbl)

                            tbl = mfe_distribution_table(mfe_r, mae_r, term_r, mfet_r, taker_cost, maker_cost)
                            tbl.insert(0, "cluster", "RANDOM"); tbl.insert(0, "group", "random")
                            mfe_tables.append(tbl)

                            df_mfe = pd.concat(mfe_tables, ignore_index=True)
                            df_mfe.to_csv(out_dir / f"cluster_mfe_{tag}.csv", index=False)

                            # ── Step 11: Time-to-level per cluster ────────────────
                            print(f"      Time-to-level...")
                            levels = [1, 2, 3, 5, 10, 15, 20, 30, 50]
                            ttl_tables = []

                            for cl in unique_cl:
                                cmask = filt_labels == cl
                                if cmask.sum() < 5: continue
                                t = time_to_level(paths_filt[cmask], levels)
                                t.insert(0, "cluster", int(cl)); t.insert(0, "group", "filtered")
                                ttl_tables.append(t)

                            t = time_to_level(paths_filt, levels)
                            t.insert(0, "cluster", "ALL_FILTERED"); t.insert(0, "group", "filtered")
                            ttl_tables.append(t)
                            t = time_to_level(all_paths, levels)
                            t.insert(0, "cluster", "ALL_BREAKOUTS"); t.insert(0, "group", "all_breakouts")
                            ttl_tables.append(t)
                            t = time_to_level(paths_rand, levels)
                            t.insert(0, "cluster", "RANDOM"); t.insert(0, "group", "random")
                            ttl_tables.append(t)

                            df_ttl = pd.concat(ttl_tables, ignore_index=True)
                            df_ttl.to_csv(out_dir / f"cluster_time_to_level_{tag}.csv", index=False)

                            # ── Step 12: Average price paths ──────────────────────
                            print(f"      Average price paths...")
                            path_tables = []
                            for cl in unique_cl:
                                cmask = filt_labels == cl
                                if cmask.sum() < 5: continue
                                pt = average_path_table(paths_filt[cmask])
                                pt.insert(0, "cluster", int(cl)); pt.insert(0, "group", "filtered")
                                path_tables.append(pt)

                            pt = average_path_table(paths_filt)
                            pt.insert(0, "cluster", "ALL_FILTERED"); pt.insert(0, "group", "filtered")
                            path_tables.append(pt)
                            pt = average_path_table(all_paths)
                            pt.insert(0, "cluster", "ALL_BREAKOUTS"); pt.insert(0, "group", "all_breakouts")
                            path_tables.append(pt)
                            pt = average_path_table(paths_rand)
                            pt.insert(0, "cluster", "RANDOM"); pt.insert(0, "group", "random")
                            path_tables.append(pt)

                            df_paths = pd.concat(path_tables, ignore_index=True)
                            df_paths.to_csv(out_dir / f"cluster_paths_{tag}.csv", index=False)

                            # ── Step 13: Comparison table ─────────────────────────
                            print(f"      Comparison table...")

                            def comp_row(label, mfe, mae, term, mfet, n_trades):
                                return dict(
                                    group=label, n_trades=n_trades,
                                    mean_mfe_bps=round(mfe.mean(),2),
                                    median_mfe_bps=round(float(np.median(mfe)),2),
                                    p90_mfe_bps=round(float(np.percentile(mfe,90)),2),
                                    p95_mfe_bps=round(float(np.percentile(mfe,95)),2),
                                    mean_mae_bps=round(mae.mean(),2),
                                    mean_terminal_bps=round(term.mean(),2),
                                    mean_mfe_time_s=round(mfet.mean(),1),
                                    median_mfe_time_s=round(float(np.median(mfet)),1),
                                    mfe_gt_taker_pct=round((mfe>=taker_cost).mean()*100,1),
                                    mfe_gt_maker_pct=round((mfe>=maker_cost).mean()*100,1),
                                    pct_positive_terminal=round((term>0).mean()*100,1),
                                )

                            comp_rows = []
                            for cl in unique_cl:
                                cmask = filt_labels == cl
                                if cmask.sum() < 5: continue
                                cm, ca, ct, cmt = compute_mfe_mae(paths_filt[cmask])
                                comp_rows.append(comp_row(f"Cluster_{cl}", cm, ca, ct, cmt, cmask.sum()))

                            comp_rows.append(comp_row("ALL_FILTERED", mfe_f, mae_f, term_f, mfet_f, n_filt))
                            comp_rows.append(comp_row("ALL_BREAKOUTS", all_mfe, all_mae, all_term, all_mfet, n_events))
                            comp_rows.append(comp_row("RANDOM", mfe_r, mae_r, term_r, mfet_r, n_rand))

                            df_comp = pd.DataFrame(comp_rows)
                            df_comp.to_csv(out_dir / f"cluster_mfe_comparison_{tag}.csv", index=False)

                            # Honest economic metric: MFE lift ──────────
                            # filtered vs. all breakouts (assignment-INDEPENDENT,
                            # hence the defensible number for the thesis).
                            _all_mfe_mean = float(np.mean(all_mfe))
                            _filt_mfe_mean = float(np.mean(mfe_f))
                            mfe_lift = (_filt_mfe_mean / _all_mfe_mean) if _all_mfe_mean > 0 else 0.0
                            edge_net = _filt_mfe_mean - _all_mfe_mean
                            tprint(f"  MFE-Lift (filtered/all): {mfe_lift:.2f}x  "
                                   f"({_filt_mfe_mean:.1f} vs {_all_mfe_mean:.1f} bps, "
                                   f"+{edge_net:.1f} bps edge, n={n_filt})")

                            # ── Step 14: Per-trade results for WS3/WS3d ───────────
                            print(f"      Saving per-trade results...")
                            df_trades = pd.DataFrame({
                                "event_index": filt_idx,
                                "timestamp": (info["timestamp"].values[filt_idx]
                                              if info is not None else pd.NaT),
                                "cluster": filt_labels,                 # fold-local (lift/filter)
                                "cluster_coherent": cluster_coherent,   # coherent (ws3 per-cluster)
                                "direction": filt_dirs,
                                "mfe_bps": mfe_f,
                                "mae_bps": mae_f,
                                "terminal_bps": term_f,
                                "mfe_time_s": mfet_f,
                            })
                            df_trades.to_csv(out_dir / f"cluster_trades_{tag}.csv", index=False)
                            n_outputs_written += 1

                            # ── Step 15: Plots ────────────────────────────────────
                            if make_plots:
                                print(f"      Generating plots...")
                                plot_mfe_distribution(df_comp, taker_cost, maker_cost, tag, out_dir)
                                plot_price_paths(df_paths, taker_cost, maker_cost, tag, out_dir)

                            # ══════════════════════════════════════════════════════
                            #  PHASE D: DYNAMIC EXIT CLASSIFIER
                            # ══════════════════════════════════════════════════════
                            if skip_exit:
                                print(f"    === Phase D skipped (--skip-exit-classifier) ===")
                            else:
                              tprint(f"  === Phase D: Dynamic exit classifier ({tag}) ===")

                              # Build step-level exit dataset
                              X_exit, y_exit, exit_meta = build_exit_dataset(
                                X=X,
                                filt_idx=filt_idx,
                                filt_dirs=filt_dirs,
                                mfe_vals=mfe_f,
                                paths=paths_filt,
                                taker_cost=taker_cost,
                                mfe_frac=0.60,       # 60% MFE threshold → ~25% pos_rate (balanced)
                                min_hold_s=3,
                                min_abs_bps=5.0,     # at least taker_cost+5bps to qualify as exit
                                max_lookahead=MFE_LOOKAHEAD,
                              )
                              print(f"      Exit dataset: {len(X_exit):,} steps, "
                                  f"pos_rate={y_exit.mean()*100:.1f}%")

                              if len(X_exit) < 500 or y_exit.sum() < 50:
                                print("      Too few exit samples — skipping Phase D")
                              else:
                                # Train exit classifier
                                fold_rows, prob_thr, oos_probs = train_exit_classifier(
                                    X_exit, y_exit,
                                    n_folds=n_folds, n_jobs=n_jobs,
                                )

                                if fold_rows:
                                    df_exit_cv = pd.DataFrame(fold_rows)
                                    df_exit_cv.to_csv(
                                        out_dir / f"exit_classifier_cv_{tag}.csv", index=False)
                                    mean_auc = df_exit_cv["auc"].mean()
                                    mean_f1  = df_exit_cv["best_f1"].mean()
                                    print(f"      Exit CV: mean AUC={mean_auc:.3f}, "
                                          f"mean F1={mean_f1:.3f}, "
                                          f"exit prob threshold={prob_thr:.2f}")

                                # Simulate dynamic exit vs fixed strategies
                                df_sim = simulate_dynamic_exit(
                                    filt_idx=filt_idx,
                                    filt_dirs=filt_dirs,
                                    paths=paths_filt,
                                    oos_probs=oos_probs,
                                    prob_thr=prob_thr,
                                    taker_cost=taker_cost,
                                    mfe_vals=mfe_f,
                                    min_hold_s=3,
                                    mfe_frac=0.60,
                                )
                                df_sim.to_csv(
                                    out_dir / f"exit_simulation_{tag}.csv", index=False)

                                if len(df_sim) > 0:
                                    dyn_mean  = df_sim["dynamic_exit_bps"].mean()
                                    tp_mean   = df_sim["tp_exit_bps"].mean()
                                    mfe_mean  = df_sim["mfe_bps"].mean()
                                    hold_mean = df_sim["hold_300s_bps"].mean()
                                    print(f"\n    ┌─ Exit Strategy Comparison ─────────────────┐")
                                    print(f"    │  Dynamic classifier: {dyn_mean:>7.2f} bps/trade   │")
                                    print(f"    │  TP at 80% MFE:      {tp_mean:>7.2f} bps/trade   │")
                                    print(f"    │  Hold to MFE:        {mfe_mean:>7.2f} bps/trade   │")
                                    print(f"    │  Hold 300s:          {hold_mean:>7.2f} bps/trade   │")
                                    print(f"    └────────────────────────────────────────────┘")
                                print(f"      Saved: exit_classifier_cv_{tag}.csv")
                                print(f"      Saved: exit_simulation_{tag}.csv")

                            # ── Console summary ───────────────────────────────────
                            print(f"\n    ╔{'═'*60}╗")
                            print(f"    ║  SUMMARY  {tag:>48} ║")
                            print(f"    ╠{'═'*60}╣")
                            print(f"    ║  {'Group':<20} {'MFE':>6} {'P90':>6} {'>=Tk':>6} {'>=Mk':>6} {'N':>7}  ║")
                            print(f"    ╠{'─'*60}╣")
                            for _, r in df_comp.iterrows():
                                name = str(r['group'])[:20]
                                print(f"    ║  {name:<20} {r['mean_mfe_bps']:>5.1f} "
                                      f"{r['p90_mfe_bps']:>5.1f} "
                                      f"{r['mfe_gt_taker_pct']:>5.1f}% "
                                      f"{r['mfe_gt_maker_pct']:>5.1f}% "
                                      f"{int(r['n_trades']):>6}  ║")
                            print(f"    ╚{'═'*60}╝")

                            tprint(f"  Files saved to: {out_dir}/")
                            print(f"      cluster_mfe_{tag}.csv")
                            print(f"      cluster_time_to_level_{tag}.csv")
                            print(f"      cluster_paths_{tag}.csv")
                            print(f"      cluster_mfe_comparison_{tag}.csv")
                            print(f"      cluster_trades_{tag}.csv         <- for WS3/WS3d")
                            print(f"      ws4_overview_{method}_{pca_tag}_{asset}_{hz}_{thr_bps}bps.png")
                            print(f"      ws4_mfe_dist_{tag}.png")
                            print(f"      ws4_paths_{tag}.png")

                        # ── Completion marker for this (asset, method, threshold)
                        # Written only after ALL lookbacks for this config finished,
                        # so a crash mid-config never leaves a false "done" flag.
                        done_marker = out_dir / (
                            f".done_{method}_{pca_tag}_{asset}_{hz}_{thr_bps}bps")

                        if n_outputs_written > 0:
                            done_marker.write_text("ok\n")
                            tprint(f"  config done: {done_marker.name}")
                        else:
                            tprint(f"  config produced no outputs — no .done marker written")

            elapsed = time.time() - t0
            tprint(f"━━ {asset.upper()}/{hz} done in {elapsed:.0f}s ━━")
            del X, y, y_1s; gc.collect()


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="WS4: MFE analysis on cluster-filtered trades")
    p.add_argument("--asset", choices=["btc","eth","both"], default="btc")
    p.add_argument("--hz", nargs="+", default=["5s", "15s"],
        help="Horizons to run. Default: 5s 15s.")
    p.add_argument("--cluster-method", nargs="+", default=list(CLUSTER_METHODS),
        choices=["kmeans", "gmm", "hdbscan", "agglomerative"],
        help="Clustering algorithm(s) to run. Default: kmeans gmm hdbscan.")
    p.add_argument("--pca-components", nargs="+", default=[str(v) for v in PCA_VARIANTS],
        help="PCA variants: integers reduce to N comps; 'none' = full feature "
             "space. Default: 25 50 none.")
    p.add_argument("--thresholds", nargs="+", type=int, default=None,
        help="Override breakout thresholds (bps) for ALL horizons. Default: "
             "per-horizon HORIZON_THRESHOLDS (5s: 10/15, 15s: 10/15/20).")
    p.add_argument("--lookbacks", nargs="+", type=int, default=LOOKBACKS)
    p.add_argument("--plots", action="store_true",
                   help="Produce PNG visualisations (default: off, CSV results only)")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--n-jobs", type=int, default=8)
    p.add_argument("--log-level", default="INFO")
    p.add_argument("--skip-exit-classifier", action="store_true",
        help="Skip Phase D (exit classifier) — run separately with rerun_phase_d.")
    p.add_argument("--max-files", type=int, default=0,
        help="s6_full: max parquet files to load (0=all). ~600 files keeps RAM under 30GB.")
    p.add_argument("--max-hours", type=int, default=None,
        help="Cap dataset to N hourly files (~3.6k rows each) as a RAM guard. "
             "e.g. --max-hours 800 → ~2.9M rows → matrix ≈34GB instead of 79GB. "
             "Default: all data.")
    p.add_argument("--data-source", choices=["s5_reduced","s6_full"],
        default="s5_reduced",
        help="s5_reduced: data_loader.py; s6_full: merged S5+S6 parquets")
    p.add_argument("--data-dir", default="data_storage/s6_features_s5_full",
        help="Directory for s6_full parquets")
    p.add_argument("--k", nargs="+", type=int, default=None,
        help="k scan: one or more cluster counts, e.g. --k 6 8 10. "
             "Overrides the default table (k=4). Tags/markers get _kN.")
    p.add_argument("--exclude-bundles", nargs="+", default=None,
        help="Exclude feature bundles before clustering, e.g. "
             "'B6_context' for microstructure-only (removes EMAs, day/week levels).")
    p.add_argument("--min-cluster-events", type=int, default=20,
        help="Minimum events per cluster to qualify as a good-cluster candidate "
             "is tested. Clusters below that are skipped. Default: 20.")
    p.add_argument("--no-da-gate", action="store_true",
        help="Grid screening: skips the DA/PnL good-cluster gate and "
             "keeps every cluster with >= --min-cluster-events events, so that "
             "configs are screened purely by MFE lift. The DA/PnL selection "
             "is a separate later step (viable-cluster evaluation).")
    p.add_argument("--config-list", default=None,
        help="CSV with columns asset,hz,thr_bps,pca_dim,k — ONLY exactly these "
             "configs are computed (e.g. screen_viable_*.csv). If omitted: "
             "full grid.")
    p.add_argument("--out-subdir", default="cluster_mfe",
        help="Output subfolder under results/ (e.g. cluster_mfe_viable), so that "
             "do not disturb the screening .done markers.")
    p.add_argument("--feature-map", default=None,
        help="feature_keep CSV for the feature signature (family/bundle per feature).")
    p.add_argument("--n-perm", type=int, default=1000,
        help="Permutations for the screening null (best-cluster ratio). 0 = off.")
    p.add_argument("--perm-seed", type=int, default=42)
    p.add_argument("--dump-only", action="store_true",
        help="Only load + cluster + per-event dump (r_cont/y_h per horizon) for "
             "the offline BCa. Skips classifier, MFE, signature, screening.")
    p.add_argument("--mfe-windows", action="store_true",
        help="Per cluster the MFE/MAE ratio over several windows (15/30/60/120/300s) "
             "from the trade-direction paths. Writes cluster_mfe_windows_*.csv and "
             "skips the classifier.")
    p.add_argument("--full-select", action="store_true",
        help="Combined re-selection run: DA (OOS, n_test) + per-event dump "
             "(BCa) + windowed MFE/MAE for EVERY cluster of EVERY config, in one "
             "pass. Skips only the classifier. Writes "
             "cluster_da_multihz_*.csv, cluster_members_*.npz, cluster_mfe_windows_*.csv.")
    p.add_argument("--silhouette-only", action="store_true",
        help="Lean k scan: loads data ONCE per (asset,hz), fits PCA "
             "once per (thr,pca) and then computes only KMeans+silhouette "
             "over all --k values (CSV: silhouette_{asset}_{hz}.csv). The full "
             "MFE/DA/trade/plot path is skipped. Does NOT reload the data per "
             "k — which makes the whole grid orders of magnitude faster.")
    a = p.parse_args()
    global MIN_CLUSTER_EVENTS
    MIN_CLUSTER_EVENTS = a.min_cluster_events
    logging.basicConfig(
        level=getattr(logging, a.log_level),
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S", stream=sys.stdout,
    )
    assets = ("btc","eth") if a.asset == "both" else (a.asset,)

    # Parse PCA variants: "none" stays as the string sentinel, everything
    # else becomes an int component count.
    pca_variants = []
    for v in a.pca_components:
        if str(v).lower() == "none":
            pca_variants.append("none")
        else:
            pca_variants.append(int(v))

    # k scan: if --k is set, run run_ws4 once per k value; otherwise once
    # with the table default (k_override=None).
    # EXCEPTION --silhouette-only: then run_ws4 loads the data ONCE per
    # (asset,hz) and iterates k INTERNALLY — so run_ws4 is called only once with the
    # full k list (no reload per k).
    k_list = a.k if a.k else [None]

    _config_filter = None
    if a.config_list:
        import pandas as _pd
        _cf = _pd.read_csv(a.config_list)
        _config_filter = set(
            (str(r["asset"]), str(r["hz"]), int(r["thr_bps"]),
             int(r["pca_dim"]), int(r["k"]))
            for _, r in _cf.iterrows())
        print(f"config-list: {len(_config_filter)} unique configs "
              f"from {a.config_list}")
    _family_map = _load_family_map(a.feature_map) if a.feature_map else None
    if _family_map:
        print(f"feature-map: {len(_family_map)} feature names loaded")

    common = dict(
        assets=assets, horizons=tuple(a.hz),
        thresholds=a.thresholds, lookbacks=a.lookbacks,
        cluster_methods=tuple(a.cluster_method),
        pca_variants=tuple(pca_variants),
        n_folds=a.n_folds, n_jobs=a.n_jobs,
        data_source=a.data_source, data_dir=a.data_dir,
        max_files=a.max_files,
        max_hours=a.max_hours,
        skip_exit=a.skip_exit_classifier,
        make_plots=a.plots,
        exclude_bundles=a.exclude_bundles,
        no_da_gate=a.no_da_gate,
        config_filter=_config_filter,
        out_subdir=a.out_subdir,
        family_map=_family_map,
        n_perm=a.n_perm,
        perm_seed=a.perm_seed,
        dump_only=a.dump_only,
        mfe_windows=a.mfe_windows,
        full_select=a.full_select,
    )

    if a.silhouette_only:
        sil_ks = a.k if a.k else [2,3,4,5,6,7,8,9,10,11,12,13,14,15]
        print(f"\n{'#'*70}\n#  SILHOUETTE-ONLY k-SCAN: k={sil_ks}\n"
              f"#  (data is loaded only ONCE per (asset,hz))\n{'#'*70}")
        run_ws4(**common, k_override=None,
                silhouette_only=True, silhouette_k_list=sil_ks)
    else:
        for k_ov in k_list:
            if k_ov is not None:
                print(f"\n{'#'*70}\n#  k-SCAN: k={k_ov}\n{'#'*70}")
            run_ws4(**common, k_override=k_ov,
                    silhouette_only=False, silhouette_k_list=None)


if __name__ == "__main__":
    main()