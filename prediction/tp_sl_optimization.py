# prediction/tp_sl_optimization.py
# ==============================================================================
# WS3: TP/SL Grid + Dynamic Exits for Cluster-Filtered Trades
# ==============================================================================
#
# V9 MODIFICATIONS (2026-04-12):
# ==============================================================================
# 1. TWO-PHASE GRID replaces the original 10×10 (50 bps ceiling) grid:
#      Phase A1 — Coarse grid: 5 bps steps, TP 5–120, SL 5–60 (288 combos)
#      Phase A2 — Fine grid:   2 bps steps, ±20 bps around coarse optimum
#    Rationale: C1 optimal TP ≈ 90 bps (MFE P90=80 bps), well outside the
#    original 50 bps ceiling.  Full 2 bps sweep would take 4–8 hours;
#    two-phase gives the same 2 bps precision in ~2 hours.
#    Output: tp_sl_grid_{tag}.csv (coarse), tp_sl_fine_grid_{tag}.csv (fine),
#            tp_sl_grid_full_{tag}.csv (merged, fine wins on overlap)
#
# 2. EXTENDED PHASE B RANGES for C1 momentum profile:
#      TP+timeout: TP extended to 100 bps  (was 30)
#      TRAIL_OFFSETS: added 20 bps         (was max 15)
#      Combined (TP+trail+timeout): TP 10–100, trail 5–20  (was TP 10–30, trail 5–10)
#
# 3. PROFITABLE CLUSTERS NOTE:
#    the fixed TP/SL baseline is the zero point for the comparison.
#    reviewing Phase A2 fine-grid results from this script.
#    Current known optimum from MFE/MAE approximation: C1 TP=90, SL=40.
#    This script validates that with actual 1s price paths.
# ==============================================================================
#
# Cluster trades are read from results/cluster_mfe/ (per the --tag / --trades-csv args).

# ==============================================================================
#
# PURPOSE:
#   ETH cluster-filtered trades achieve gross +10 bps per trade (maker) with
#   cluster-to-cluster exits (WS3b). This script finds the optimal TP/SL and
#   trailing stop parameters to either (a) improve on this further with a
#   tighter safety SL, or (b) establish the best static exit as a simpler
#   live-trading alternative. ETH profitable clusters have mean MFE of
#   ~33 bps (C1) and ~15 bps (C3, C5) -- there is meaningful room to
#   optimise exits above the current terminal-return baseline.
#
# DEPENDS ON:
#   WS4 outputs — specifically cluster_trades_{tag}.csv which contains
#   per-trade event_index, cluster label, and direction. WS3 re-extracts
#   the full 300s price paths from y_1s and simulates exit strategies on them.
#
# PIPELINE (executed in order):
#
#   PHASE A — Static TP/SL Grid (hybrid grid, 1,377 combos)
#   ──────────────────────────────────────────────────────────────
#   1. Load WS4 cluster_trades CSV (event_index, cluster, direction)
#   2. Load y_1s (1s returns), extract float32 paths, FREE y_1s immediately
#   3. Simulate hybrid TP/SL grid:
#      TP  ∈ [10,15,20,25] + range(28,122,2)  — 51 values
#      SL  ∈ [5,10,15]     + range(16,62,2)   — 27 values
#      = 1,377 combinations.  Fine 2 bps resolution where it matters.
#   4. Per-cluster sweep: TP_COARSE x SL_COARSE (13x8=104 combos) for speed
#
#   PHASE B — Time-Based Exits + Trailing Stops (extended ranges)
#   ──────────────────────────────────────────────────────────────
#   5. Pure timeout exits: 5–300 s
#   6. TP + timeout: TP up to 100 bps (extended for C1 momentum)
#   7. Trailing stop: offset up to 20 bps
#   8. Combined: TP (10–100) + trailing (5–20) + timeout
#
#   PHASE C — Analysis & Visualization
#   ──────────────────────────────────────────────────────────────
#   9. Best strategy per cluster, comparison vs baselines
#
#   PHASE D — Sub-cluster Heterogeneity Analysis (NEW)
#   ──────────────────────────────────────────────────────────────
#  10. Split each cluster by MAE tercile (tight/medium/loose) and
#      MFE speed (fast/slow peak) → up to 6 sub-groups per cluster
#  11. Sweep TP_COARSE x SL_COARSE (104 combos) per sub-group
#  12. Flag clusters where sub-groups diverge in optimal TP >= 10 bps
#      → these warrant feature-based investigation for live sub-routing
#
# OUTPUTS (in RESULTS_DIR/tp_sl_optimization/):
#   tp_sl_grid_{tag}.csv               Full 1,377-combo grid results
#   tp_sl_grid_by_cluster_{tag}.csv    Coarse grid results per cluster
#   time_exits_{tag}.csv               Time-based exit results
#   trailing_stop_{tag}.csv            Trailing stop results
#   combined_exits_{tag}.csv           Combined strategy results
#   optimal_strategy_{tag}.csv         Best strategy per cluster
#   sub_cluster_analysis_{tag}.csv     Phase D sub-cluster optima + flags
#
# USAGE:
#   python tp_sl_optimization.py --asset eth --hz 15s --lookbacks 2
#   python tp_sl_optimization.py --asset both --hz 15s
#
# RUNTIME: ~4-8 hours for ETH/15s/lb2 (1,377 main grid combos).
#          Designed for overnight unattended run.
#          RAM: y_1s freed immediately after path extraction.
#               Paths use float32 (~19 MB for 16k trades x 300s).
#               RAM checks abort with clear error if free RAM < 8 GB.
# ==============================================================================
from __future__ import annotations
import argparse, gc, logging, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

from prediction.honest_cv import honest_tp_sl_cv, honest_strategy_cv   # train-only per-fold choice

N_FOLDS = 5   # expanding-window folds for the honest TP/SL CV

logger = logging.getLogger(__name__)


# ─── RAM safety ──────────────────────────────────────────────────────────────

def _get_free_ram_gb():
    """Return available RAM in GB via /proc/meminfo or psutil, else None."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1_048_576   # KB -> GB
    except Exception:
        pass
    try:
        import psutil
        return psutil.virtual_memory().available / 1_073_741_824
    except ImportError:
        pass
    return None


def check_ram(min_free_gb=8.0, label=""):
    """Print RAM status and raise MemoryError if below threshold."""
    free = _get_free_ram_gb()
    if free is None:
        print(f"  RAM check [{label}]: undetectable — proceeding")
        return 999.0
    status = "OK" if free >= min_free_gb else "LOW — aborting"
    print(f"  RAM [{label}]: {free:.1f} GB free  ({status})")
    if free < min_free_gb:
        raise MemoryError(
            f"Only {free:.1f} GB free, need {min_free_gb:.1f} GB. "
            "Aborting to prevent OOM. Free memory and retry."
        )
    return free

# ─── Grid parameters ─────────────────────────────────────────────────────────
#
# HYBRID GRID — coarse steps at extremes, 2 bps resolution in the action zone
# ────────────────────────────────────────────────────────────────────────────
# TP values:
#   [10, 15, 20, 25] (5 bps steps, below C2 MFE P50=12 bps territory)
#   + range(28, 122, 2) (2 bps steps, covers C1 MFE P50=19 to P95=100 bps)
#   = 4 + 47 = 51 values
#
# SL values:
#   [5, 10, 15] (5 bps steps, below typical MAE P50)
#   + range(16, 62, 2) (2 bps steps, covers MAE P25–P95 for both clusters)
#   = 3 + 24 = 27 values
#
# Total grid: 51 x 27 = 1,377 combos.
# Estimated runtime: 4-8 hours overnight (safe for unattended run).
# RAM budget: paths array ~19 MB (float32, 16k trades x 300s).
#             y_1s is freed immediately after path extraction.
#
# PHASE D — sub-cluster analysis uses coarse subset (13x8=104 combos)
# per sub-group for speed; full grid would be redundant at this stage.
#
TP_VALUES     = [10, 15, 20, 25] + list(range(28, 122, 2))  # 51 values
SL_VALUES     = [5, 10, 15]      + list(range(16, 64, 2))   # 3 + 24 = 27 values

# Coarse-only subset used for per-cluster and sub-cluster sweeps (faster)
TP_COARSE     = [10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 120]  # 13 values
SL_COARSE     = [5, 10, 15, 20, 25, 30, 40, 50]                          # 8 values

TIMEOUTS      = [5, 10, 15, 30, 60, 120, 180, 300]
TRAIL_OFFSETS = [3, 5, 7, 10, 15, 20]   # added 20 bps for C1 momentum profile
MFE_LOOKAHEAD = 300

# DEFAULTS (2026-06): the primary and recommended entry path is --trades-file,
# which points directly at a WS4 cluster_trades_*.csv and derives the tag from
# the filename (bypassing the tag-search below). These THRESHOLDS/LOOKBACKS are
# only used for the compat tag-search path, and are aligned to the viable
# configs (Sec 4.4): thresholds 10/15/40, lead time fixed at 1s. NOTE: the
# tag-search path builds "{asset}_{hz}_{thr}bps_lb{lb}" WITHOUT the
# "{method}_{pca}_k{k}_" prefix that WS4 actually writes, so it will NOT find
# the real files — always use --trades-file for the viable configs.
THRESHOLDS    = [15]
LOOKBACKS     = [1]

# Sub-cluster analysis thresholds (Phase D)
# If two sub-groups within a cluster differ in optimal TP by >= this value,
# the cluster is flagged as heterogeneous and worth splitting for live trading.
SUBCLUSTER_TP_DIVERGENCE_BPS  = 10
SUBCLUSTER_SL_DIVERGENCE_BPS  = 8
SUBCLUSTER_MIN_N              = 40   # minimum trades per sub-group to analyse


# ═══════════════════════════════════════════════════════════════════════════════
# EXIT SIMULATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def extract_price_paths(y_1s, event_indices, directions, max_lookahead=300):
    """Extract direction-adjusted cumulative return paths in bps.
    Uses float32 (vs float64) to halve memory: ~19 MB for 16k x 300 paths."""
    n = len(event_indices)
    paths = np.full((n, max_lookahead), np.nan, dtype=np.float32)
    for i in range(n):
        idx, d = int(event_indices[i]), directions[i]
        end = min(idx + max_lookahead + 1, len(y_1s))
        if end <= idx + 1:
            continue
        cum = np.cumsum(y_1s[idx + 1 : end]) * d * 10_000
        paths[i, :len(cum)] = cum
    return paths


def simulate_tp_sl(paths, tp_bps, sl_bps, timeout_s=None):
    """
    Simulate TP/SL exit on precomputed paths.
    Returns per-trade: gross_pnl (before costs), outcome, exit_time.
    Outcome: 1=TP hit, -1=SL hit, 0=timeout.
    """
    n = paths.shape[0]
    max_t = paths.shape[1]
    if timeout_s is None:
        timeout_s = max_t

    gross_pnl  = np.zeros(n)
    outcomes   = np.zeros(n, dtype=int)
    exit_times = np.zeros(n)

    for i in range(n):
        p = paths[i]
        valid_len = int(np.sum(~np.isnan(p)))
        if valid_len == 0:
            continue

        t_limit = min(valid_len, timeout_s)
        path_slice = p[:t_limit]

        # Find first TP and SL crossings
        tp_hits = np.where(path_slice >= tp_bps)[0]
        sl_hits = np.where(path_slice <= -sl_bps)[0]

        tp_time = tp_hits[0] if len(tp_hits) > 0 else t_limit + 1
        sl_time = sl_hits[0] if len(sl_hits) > 0 else t_limit + 1

        if tp_time <= sl_time and tp_time < t_limit:
            gross_pnl[i] = tp_bps
            outcomes[i] = 1
            exit_times[i] = tp_time + 1
        elif sl_time < tp_time and sl_time < t_limit:
            gross_pnl[i] = -sl_bps
            outcomes[i] = -1
            exit_times[i] = sl_time + 1
        else:
            # Timeout: take terminal value
            gross_pnl[i] = path_slice[-1] if t_limit > 0 else 0
            outcomes[i] = 0
            exit_times[i] = t_limit

    return gross_pnl, outcomes, exit_times


def simulate_time_exit(paths, timeout_s):
    """Simple time-based exit: hold for exactly N seconds."""
    n = paths.shape[0]
    gross_pnl = np.zeros(n)
    for i in range(n):
        p = paths[i]
        valid_len = int(np.sum(~np.isnan(p)))
        if valid_len == 0:
            continue
        t = min(timeout_s, valid_len) - 1
        gross_pnl[i] = p[t] if t >= 0 else 0
    return gross_pnl


def simulate_trailing_stop(paths, trail_offset_bps, tp_bps=None, timeout_s=None):
    """
    Trailing stop: SL follows the running high.
    Once price reaches +X, stop moves to X - trail_offset.
    Optional: also exit at TP or timeout.
    """
    n = paths.shape[0]
    max_t = paths.shape[1]
    if timeout_s is None:
        timeout_s = max_t

    gross_pnl  = np.zeros(n)
    outcomes   = np.zeros(n, dtype=int)  # 1=TP, -1=trail_stop, 0=timeout
    exit_times = np.zeros(n)

    for i in range(n):
        p = paths[i]
        valid_len = int(np.sum(~np.isnan(p)))
        if valid_len == 0:
            continue

        t_limit = min(valid_len, timeout_s)
        running_high = 0.0

        for t in range(t_limit):
            val = p[t]
            if val > running_high:
                running_high = val

            # Check TP
            if tp_bps is not None and val >= tp_bps:
                gross_pnl[i] = tp_bps
                outcomes[i] = 1
                exit_times[i] = t + 1
                break

            # Check trailing stop (only active once we've been positive)
            trail_level = running_high - trail_offset_bps
            if running_high > trail_offset_bps and val <= trail_level:
                gross_pnl[i] = val  # exit at current price, not trail level
                outcomes[i] = -1
                exit_times[i] = t + 1
                break
        else:
            # Timeout
            gross_pnl[i] = p[t_limit - 1] if t_limit > 0 else 0
            outcomes[i] = 0
            exit_times[i] = t_limit

    return gross_pnl, outcomes, exit_times


def compute_strategy_metrics(gross_pnl, outcomes, exit_times, cost_bps, n_total=None):
    """Compute standard metrics for a strategy."""
    if n_total is None:
        n_total = len(gross_pnl)
    net_pnl = gross_pnl - cost_bps
    n = len(net_pnl)
    if n == 0:
        return {}

    return dict(
        n_trades    = n,
        mean_gross_pnl = round(float(gross_pnl.mean()), 3),
        mean_net_pnl   = round(float(net_pnl.mean()), 3),
        total_net_pnl  = round(float(net_pnl.sum()), 1),
        win_rate    = round(float((net_pnl > 0).mean()), 4),
        tp_rate     = round(float((outcomes == 1).mean()), 4),
        sl_rate     = round(float((outcomes == -1).mean()), 4),
        timeout_rate= round(float((outcomes == 0).mean()), 4),
        mean_exit_s = round(float(exit_times.mean()), 1),
        median_exit_s = round(float(np.median(exit_times)), 1),
        sharpe      = round(float(net_pnl.mean() / net_pnl.std()), 4) if net_pnl.std() > 0 else 0,
        profit_factor = round(float(
            net_pnl[net_pnl > 0].sum() / abs(net_pnl[net_pnl < 0].sum())
        ), 3) if (net_pnl < 0).any() and (net_pnl > 0).any() else 0,
    )


def build_tp_sl_grid_from_paths(paths, max_tp=120):
    """TP from MFE quantiles, SL from MAE quantiles of the (direction-adjusted) paths.
    Coarse market prior, ~6-8 x 5 combos. The per-fold selection WITHIN the
    Grid stays train-only (honest_cv). Adapts to thr15/thr20 automatically."""
    mfe = np.nanmax(paths, axis=1)
    mae = -np.nanmin(paths, axis=1)            # adverse magnitude (paths go negative)
    mfe = mfe[np.isfinite(mfe)]
    mae = np.clip(mae[np.isfinite(mae)], 0, None)
    if len(mfe) < 10 or len(mae) < 10:         # fallback when too little data
        return [15, 20, 25, 30, 40, 50, 60, 80], [8, 12, 16, 20, 25, 30]
    tp = sorted({int(min(max(round(v / 5) * 5, 10), max_tp))
                 for v in np.percentile(mfe, [25, 40, 55, 70, 80, 90, 95])})
    sl = sorted({int(min(max(round(v / 2) * 2, 5), 60))
                 for v in np.percentile(mae, [25, 40, 55, 70, 85])})
    return tp, sl


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION (Phase C)
# ═══════════════════════════════════════════════════════════════════════════════

def _setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.grid": True, "grid.alpha": 0.3,
        "font.size": 11, "axes.titlesize": 13,
    })
    return plt


def plot_tp_sl_heatmap(df_grid, cost_type, tag, out_dir):
    """Heatmap of mean net PnL across TP/SL grid."""
    plt = _setup_mpl()

    pnl_col = f"mean_net_pnl_{cost_type}"
    if pnl_col not in df_grid.columns:
        pnl_col = "mean_net_pnl"

    pivot = df_grid.pivot_table(index="sl_bps", columns="tp_bps", values=pnl_col)

    fig, ax = plt.subplots(figsize=(10, 8))
    vmax = max(abs(pivot.values.min()), abs(pivot.values.max()))
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto",
                   vmin=-vmax, vmax=vmax, origin="lower")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Take-profit (bps)")
    ax.set_ylabel("Stop-loss (bps)")
    ax.set_title(f"Mean net PnL ({cost_type}) — {tag}")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Mean PnL (bps)")

    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            color = "white" if abs(val) > vmax * 0.6 else "black"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                    fontsize=7, color=color)

    plt.tight_layout()
    path = out_dir / f"ws3_heatmap_{cost_type}_{tag}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"      Plot: {path.name}")


def plot_strategy_comparison(df_optimal, tag, out_dir):
    """Bar chart comparing best strategies."""
    plt = _setup_mpl()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax_idx, cost_type in enumerate(["taker", "maker"]):
        ax = axes[ax_idx]
        df = df_optimal[df_optimal["cost_type"] == cost_type].sort_values("mean_net_pnl", ascending=True)
        if len(df) == 0:
            continue

        labels = df["strategy_label"].values
        pnl = df["mean_net_pnl"].values
        colors = ["#1D9E75" if v > 0 else "#D85A30" for v in pnl]

        ax.barh(range(len(labels)), pnl, color=colors, alpha=0.85)
        ax.axvline(x=0, color="black", linewidth=0.5)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Mean net PnL (bps)")
        ax.set_title(f"Strategy comparison — {cost_type}")

        for i, v in enumerate(pnl):
            ax.text(v + 0.1, i, f"{v:+.2f}", va="center", fontsize=8)

    fig.suptitle(f"WS3 strategy comparison — {tag}", fontsize=13, y=1.02)
    plt.tight_layout()
    path = out_dir / f"ws3_comparison_{tag}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"      Plot: {path.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def run_ws3(
    assets=("btc",),
    horizons=("15s",),
    thresholds=None,
    lookbacks=None,
    trades_file=None,
):
    from common.data_loader import load_dataset
    from common.config import RESULTS_DIR, SPREAD_BPS, MAKER_COST_BPS

    if thresholds is None:
        thresholds = THRESHOLDS
    if lookbacks is None:
        lookbacks = LOOKBACKS

    ws4_dir = RESULTS_DIR / "cluster_mfe"
    out_dir = RESULTS_DIR / "tp_sl_optimization"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = out_dir / "ws3_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for asset in assets:
        taker_cost = SPREAD_BPS.get(asset, {}).get("fut", 10.0)  # config v6 round-trip taker
        maker_cost = MAKER_COST_BPS.get(asset, {}).get("fut", 4.0)

        for hz in horizons:
            t0 = time.time()
            print(f"\n{'━'*70}")
            print(f"  WS3 — {asset.upper()}/{hz}")
            print(f"{'━'*70}")

            # ── Load y_1s for path extraction ─────────────────────────────
            print(f"  Loading 1s returns...")
            check_ram(min_free_gb=12.0, label="before y_1s load")
            try:
                _, y_1s, _, _ = load_dataset(target="ret_1s", asset=asset, target_only=True)
            except Exception as e:
                logger.error("Failed to load 1s data for %s: %s", asset, e)
                continue
            print(f"  y_1s: {len(y_1s):,} samples")
            check_ram(min_free_gb=8.0, label="after y_1s load")

            # --trades-file: exactly this one file; tag from the filename. Otherwise
            # the usual thr/lb iteration (both loops then run once
            # with None; structure/indentation stays unchanged).
            _thr_iter = [None] if trades_file is not None else thresholds
            for thr_bps in _thr_iter:
                _lb_iter = [None] if trades_file is not None else lookbacks
                for lookback in _lb_iter:
                    if trades_file is not None:
                        trades_path = Path(trades_file)
                        tag = trades_path.stem.replace("cluster_trades_", "")
                    else:
                        tag = f"{asset}_{hz}_{thr_bps}bps_lb{lookback}"
                        trades_path = ws4_dir / f"cluster_trades_{tag}.csv"

                    if not trades_path.exists():
                        print(f"\n  {tag}: cluster_trades not found: {trades_path}")
                        continue

                    df_trades = pd.read_csv(trades_path)
                    n_trades = len(df_trades)
                    print(f"\n  ── {tag}: {n_trades} filtered trades ──")

                    # Prefer the coherent label (ws4 cluster_coherent);
                    # fold-local 'cluster' only as a fallback (exploratory).
                    cl_col = "cluster_coherent" if "cluster_coherent" in df_trades.columns else "cluster"
                    if cl_col == "cluster":
                        print("    WARNING: only fold-local labels -> per-cluster only exploratory")

                    event_indices = df_trades["event_index"].values.astype(int)
                    clusters      = df_trades[cl_col].values.astype(int)
                    directions    = df_trades["direction"].values.astype(float)
                    unique_cl     = np.unique(clusters)

                    # ── Step 2: Extract full price paths ──────────────────
                    print(f"    Extracting 300s price paths (float32)...")
                    paths = extract_price_paths(y_1s, event_indices, directions, MFE_LOOKAHEAD)
                    print(f"    Paths shape: {paths.shape}  "
                          f"({paths.nbytes / 1e6:.0f} MB)")

                    # Tight, data-driven grid from the actual
                    # MFE/MAE of the paths (replaces the 1377-cell grid). Market prior,
                    # NO per-cluster tuning; the per-fold selection stays OOS.
                    tp_grid, sl_grid = build_tp_sl_grid_from_paths(paths)
                    print(f"    Grid from MFE/MAE: TP={tp_grid}  SL={sl_grid}  "
                          f"({len(tp_grid)}x{len(sl_grid)}={len(tp_grid)*len(sl_grid)} combos)")
                    # Note: y_1s is kept alive through all lookback iterations
                    # (each lookback has different event_indices). It is freed
                    # after the outer lookback loop completes — see below.

                    # ══════════════════════════════════════════════════════
                    #  PHASE A: STATIC TP/SL GRID
                    # ══════════════════════════════════════════════════════
                    print(f"\n    === Phase A: TP/SL grid ({len(tp_grid)}x{len(sl_grid)} = {len(tp_grid)*len(sl_grid)} combos) ===")

                    grid_rows = []
                    for tp in tp_grid:
                        for sl in sl_grid:
                            gpnl, outcomes, etimes = simulate_tp_sl(paths, tp, sl)

                            row = dict(tp_bps=tp, sl_bps=sl, ratio=round(tp/sl, 2))
                            # Taker metrics
                            m = compute_strategy_metrics(gpnl, outcomes, etimes, taker_cost)
                            for k, v in m.items():
                                row[f"{k}_taker"] = v
                            row["mean_net_pnl_taker"] = m["mean_net_pnl"]
                            row["sharpe_taker"] = m["sharpe"]

                            # Maker metrics
                            m = compute_strategy_metrics(gpnl, outcomes, etimes, maker_cost)
                            for k, v in m.items():
                                row[f"{k}_maker"] = v
                            row["mean_net_pnl_maker"] = m["mean_net_pnl"]
                            row["sharpe_maker"] = m["sharpe"]

                            grid_rows.append(row)

                    df_grid = pd.DataFrame(grid_rows)
                    df_grid.to_csv(out_dir / f"tp_sl_grid_{tag}.csv", index=False)

                    # Best combos — IN-SAMPLE, descriptive only (NOT for the result!)
                    print(f"      [IN-SAMPLE grid — descriptive, optimistically biased:]")
                    for cost_type in ["taker", "maker"]:
                        col = f"mean_net_pnl_{cost_type}"
                        best = df_grid.nlargest(5, col)
                        print(f"      Top 5 ({cost_type}):")
                        for _, r in best.iterrows():
                            print(f"        TP={int(r['tp_bps']):>3} SL={int(r['sl_bps']):>3} "
                                  f"PnL={r[col]:>+7.2f} bps "
                                  f"WR={r[f'win_rate_{cost_type}']:.1%} "
                                  f"Sharpe={r[f'sharpe_{cost_type}']:.3f}")

                    # ══════════════════════════════════════════════════════
                    #  PHASE A HONEST: train-only per-fold TP/SL choice
                    #  (train selects by Sharpe, test measures — that is the
                    #   defensible result for the thesis.)
                    # ══════════════════════════════════════════════════════
                    print(f"\n    === Phase A (HONEST per-fold, Sharpe selection, taker ranking) ===")
                    hcv = honest_tp_sl_cv(
                        paths=paths, clusters=clusters, event_indices=event_indices,
                        tp_values=tp_grid, sl_values=sl_grid,
                        taker_cost=taker_cost, maker_cost=maker_cost,
                        simulate_tp_sl=simulate_tp_sl, n_folds=N_FOLDS,
                    )
                    if hcv is None:
                        print("      too few trades for per-fold CV — skipped")
                    else:
                        g = hcv["global_"]
                        gt, gm = g["oos_taker"], g["oos_maker"]
                        print(f"      GLOBAL  OOS: taker mean={gt['mean_net_pnl']:+.2f} bps "
                              f"sharpe={gt['sharpe']:.3f} WR={gt['win_rate']:.1%} n={gt['n']}")
                        print(f"              (maker mean={gm['mean_net_pnl']:+.2f} bps "
                              f"sharpe={gm['sharpe']:.3f})")
                        chosen = g["chosen_per_fold"]
                        print(f"              chosen (TP,SL) per fold: {chosen}")
                        # Stability of the choice as diagnostics
                        uniq = len(set(chosen))
                        print(f"              {'STABLE' if uniq <= 2 else 'UNSTABLE'} "
                              f"TP/SL choice ({uniq} distinct over {len(chosen)} folds)")

                        # honest result as CSV
                        hrows = [dict(scope="GLOBAL", cluster=-1, **gt,
                                      sharpe_maker=gm["sharpe"],
                                      mean_net_pnl_maker=gm["mean_net_pnl"])]
                        if hcv["per_cluster"]:
                            print(f"      PER-CLUSTER OOS (taker):")
                            for cl, r in sorted(hcv["per_cluster"].items()):
                                ct = r["oos_taker"]; cm = r["oos_maker"]
                                print(f"        Cl {cl}: mean={ct['mean_net_pnl']:+.2f} bps "
                                      f"sharpe={ct['sharpe']:.3f} WR={ct['win_rate']:.1%} "
                                      f"n={ct['n']}  (TP/SL per fold: {r['chosen_per_fold']})")
                                hrows.append(dict(scope="CLUSTER", cluster=cl, **ct,
                                                  sharpe_maker=cm["sharpe"],
                                                  mean_net_pnl_maker=cm["mean_net_pnl"]))
                        pd.DataFrame(hrows).to_csv(
                            out_dir / f"tp_sl_honest_oos_{tag}.csv", index=False)

                    # Heatmaps
                    # plots disabled -- results only
                    # plot_tp_sl_heatmap(df_grid, "taker", tag, plot_dir)
                    # plot_tp_sl_heatmap(df_grid, "maker", tag, plot_dir)

                    # ── Per-cluster grid (coarse only for speed) ──────────
                    print(f"\n    Per-cluster TP/SL grid (coarse: {len(TP_COARSE)}x{len(SL_COARSE)} combos)...")
                    cluster_grid_rows = []

                    for cl in unique_cl:
                        cl_mask = clusters == cl
                        cl_paths = paths[cl_mask]
                        n_cl = cl_mask.sum()
                        if n_cl < 20:
                            continue

                        for tp in TP_COARSE:
                            for sl in SL_COARSE:
                                gpnl, outcomes, etimes = simulate_tp_sl(cl_paths, tp, sl)
                                m_tk = compute_strategy_metrics(gpnl, outcomes, etimes, taker_cost)
                                m_mk = compute_strategy_metrics(gpnl, outcomes, etimes, maker_cost)
                                cluster_grid_rows.append(dict(
                                    cluster=int(cl), tp_bps=tp, sl_bps=sl,
                                    n_trades=n_cl,
                                    mean_net_pnl_taker=m_tk["mean_net_pnl"],
                                    sharpe_taker=m_tk["sharpe"],
                                    win_rate_taker=m_tk["win_rate"],
                                    mean_net_pnl_maker=m_mk["mean_net_pnl"],
                                    sharpe_maker=m_mk["sharpe"],
                                    win_rate_maker=m_mk["win_rate"],
                                ))

                        # Print best for this cluster
                        cl_df = pd.DataFrame([r for r in cluster_grid_rows if r["cluster"] == int(cl)])
                        if len(cl_df) > 0:
                            best_mk = cl_df.nlargest(1, "mean_net_pnl_maker").iloc[0]
                            print(f"      Cl {cl} (N={n_cl}): Best maker TP={int(best_mk['tp_bps'])} "
                                  f"SL={int(best_mk['sl_bps'])} "
                                  f"PnL={best_mk['mean_net_pnl_maker']:+.2f} bps")

                    df_cluster_grid = pd.DataFrame(cluster_grid_rows)
                    df_cluster_grid.to_csv(out_dir / f"tp_sl_grid_by_cluster_{tag}.csv", index=False)

                    # ══════════════════════════════════════════════════════
                    #  PHASE B: TIME-BASED EXITS + TRAILING STOPS
                    # ══════════════════════════════════════════════════════
                    print(f"\n    === Phase B: Time exits + trailing stops ===")

                    # ── Step 7: Pure time exits ───────────────────────────
                    print(f"      Time exits...")
                    time_rows = []
                    for timeout in TIMEOUTS:
                        gpnl = simulate_time_exit(paths, timeout)
                        for cost_type, cost in [("taker", taker_cost), ("maker", maker_cost)]:
                            net = gpnl - cost
                            time_rows.append(dict(
                                strategy="time_exit", timeout_s=timeout,
                                cost_type=cost_type,
                                n_trades=n_trades,
                                mean_gross_pnl=round(float(gpnl.mean()), 3),
                                mean_net_pnl=round(float(net.mean()), 3),
                                total_net_pnl=round(float(net.sum()), 1),
                                win_rate=round(float((net > 0).mean()), 4),
                                sharpe=round(float(net.mean()/net.std()), 4) if net.std() > 0 else 0,
                            ))

                    df_time = pd.DataFrame(time_rows)
                    df_time.to_csv(out_dir / f"time_exits_{tag}.csv", index=False)

                    for cost_type in ["taker", "maker"]:
                        best = df_time[df_time["cost_type"]==cost_type].nlargest(3, "mean_net_pnl")
                        for _, r in best.iterrows():
                            print(f"        {cost_type} timeout={int(r['timeout_s'])}s: "
                                  f"PnL={r['mean_net_pnl']:+.2f} bps, "
                                  f"WR={r['win_rate']:.1%}")

                    # ── Step 8: TP + timeout ──────────────────────────────
                    # Extended TP range to 100 bps to cover C1 momentum profile
                    print(f"      TP + timeout combos...")
                    tp_timeout_rows = []
                    for tp in [5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
                        for timeout in [15, 30, 60, 120, 300]:
                            gpnl, outcomes, etimes = simulate_tp_sl(
                                paths, tp_bps=tp, sl_bps=9999, timeout_s=timeout)
                            for cost_type, cost in [("taker", taker_cost), ("maker", maker_cost)]:
                                m = compute_strategy_metrics(gpnl, outcomes, etimes, cost)
                                tp_timeout_rows.append(dict(
                                    strategy="tp_timeout", tp_bps=tp, timeout_s=timeout,
                                    cost_type=cost_type, **m,
                                ))

                    df_tp_timeout = pd.DataFrame(tp_timeout_rows)

                    # ── Step 9: Trailing stops ────────────────────────────
                    print(f"      Trailing stops...")
                    trail_rows = []
                    for trail in TRAIL_OFFSETS:
                        # Pure trailing stop
                        gpnl, outcomes, etimes = simulate_trailing_stop(paths, trail)
                        for cost_type, cost in [("taker", taker_cost), ("maker", maker_cost)]:
                            m = compute_strategy_metrics(gpnl, outcomes, etimes, cost)
                            trail_rows.append(dict(
                                strategy="trailing_only", trail_offset=trail,
                                tp_bps=0, timeout_s=300,
                                cost_type=cost_type, **m,
                            ))

                    # ── Step 10: TP + trailing + timeout (triple) ─────────
                    # Extended TP range for C1 (TP 30–100), trail offsets 5–20
                    print(f"      Combined: TP + trailing + timeout...")
                    for tp in [10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
                        for trail in [5, 7, 10, 15, 20]:
                            for timeout in [60, 120, 300]:
                                gpnl, outcomes, etimes = simulate_trailing_stop(
                                    paths, trail, tp_bps=tp, timeout_s=timeout)
                                for cost_type, cost in [("taker", taker_cost), ("maker", maker_cost)]:
                                    m = compute_strategy_metrics(gpnl, outcomes, etimes, cost)
                                    trail_rows.append(dict(
                                        strategy="tp_trail_timeout",
                                        trail_offset=trail, tp_bps=tp, timeout_s=timeout,
                                        cost_type=cost_type, **m,
                                    ))

                    df_trail = pd.DataFrame(trail_rows)
                    df_trail.to_csv(out_dir / f"trailing_stop_{tag}.csv", index=False)

                    # Best trailing strategies
                    for cost_type in ["taker", "maker"]:
                        best = df_trail[df_trail["cost_type"]==cost_type].nlargest(3, "mean_net_pnl")
                        print(f"      Best trailing ({cost_type}):")
                        for _, r in best.iterrows():
                            s = r["strategy"]
                            tp_str = f"TP={int(r['tp_bps'])}" if r["tp_bps"] > 0 else "no TP"
                            print(f"        {s}: trail={int(r['trail_offset'])} {tp_str} "
                                  f"t/o={int(r['timeout_s'])}s → PnL={r['mean_net_pnl']:+.2f} bps "
                                  f"Sharpe={r['sharpe']:.3f}")

                    # Combine all exit strategies
                    df_combined = pd.concat([df_tp_timeout, df_trail], ignore_index=True)
                    df_combined.to_csv(out_dir / f"combined_exits_{tag}.csv", index=False)

                    # ══════════════════════════════════════════════════════
                    #  PHASE C: FIND OPTIMAL + COMPARISON
                    # ══════════════════════════════════════════════════════
                    print(f"\n    === Phase C: Optimal strategy ===")

                    optimal_rows = []

                    # Baseline: simple 15s hold (= what we have now)
                    baseline_pnl = simulate_time_exit(paths, 15)
                    for cost_type, cost in [("taker", taker_cost), ("maker", maker_cost)]:
                        net = baseline_pnl - cost
                        optimal_rows.append(dict(
                            strategy_label="Baseline (15s hold)",
                            cost_type=cost_type,
                            mean_net_pnl=round(float(net.mean()), 3),
                            total_net_pnl=round(float(net.sum()), 1),
                            win_rate=round(float((net > 0).mean()), 4),
                            sharpe=round(float(net.mean()/net.std()), 4) if net.std() > 0 else 0,
                            details="hold=15s",
                        ))

                    # Baseline: full 300s hold
                    hold300_pnl = simulate_time_exit(paths, 300)
                    for cost_type, cost in [("taker", taker_cost), ("maker", maker_cost)]:
                        net = hold300_pnl - cost
                        optimal_rows.append(dict(
                            strategy_label="Hold 300s (no exit)",
                            cost_type=cost_type,
                            mean_net_pnl=round(float(net.mean()), 3),
                            total_net_pnl=round(float(net.sum()), 1),
                            win_rate=round(float((net > 0).mean()), 4),
                            sharpe=round(float(net.mean()/net.std()), 4) if net.std() > 0 else 0,
                            details="hold=300s",
                        ))

                    # ══════════════════════════════════════════════════════
                    #  HONEST per-fold OOS for ALL four strategy families.
                    #  The df_grid/df_time/df_trail/df_combined above are
                    #  IN-SAMPLE and only descriptive. This is the result.
                    # ══════════════════════════════════════════════════════
                    print(f"\n    === HONEST per-fold OOS (all strategies, Sharpe/Taker) ===")

                    # parameter grid per family (as a list of dicts)
                    # Consistent with Phase A: a tight grid derived beforehand from MFE/MAE
                    # derived grid (instead of 1377) — less selection variance,
                    # more defensible for the thesis. Consequence unchanged.
                    tp_sl_grid   = [dict(tp_bps=tp, sl_bps=sl)
                                    for tp in tp_grid for sl in sl_grid]
                    time_grid    = [dict(timeout_s=to) for to in TIMEOUTS]
                    trail_grid   = [dict(trail_offset_bps=tr, tp_bps=(tp if tp > 0 else None),
                                         timeout_s=(to if to > 0 else None))
                                    for tr in TRAIL_OFFSETS
                                    for tp in [0] + TP_COARSE
                                    for to in [0] + TIMEOUTS]

                    honest_families = [
                        ("tp_sl",    simulate_tp_sl,        tp_sl_grid),
                        ("time",     simulate_time_exit,    time_grid),
                        ("trailing", simulate_trailing_stop, trail_grid),
                    ]
                    honest_all_rows = []
                    for fam_label, fam_sim, fam_grid in honest_families:
                        res = honest_strategy_cv(
                            paths=paths, clusters=clusters,
                            event_indices=event_indices,
                            param_grid=fam_grid, sim_fn=fam_sim,
                            taker_cost=taker_cost, maker_cost=maker_cost,
                            n_folds=N_FOLDS, label=fam_label,
                        )
                        if res is None:
                            print(f"      {fam_label}: too few trades — skipped")
                            continue
                        g = res["global_"]; gt = g["oos_taker"]; gm = g["oos_maker"]
                        nuniq = len(set(map(str, g["chosen_per_fold"])))
                        print(f"      [{fam_label}] GLOBAL OOS: taker mean={gt['mean_net_pnl']:+.2f} "
                              f"sharpe={gt['sharpe']:.3f} WR={gt['win_rate']:.1%} n={gt['n']} "
                              f"| maker mean={gm['mean_net_pnl']:+.2f} "
                              f"| {'STABLE' if nuniq <= 2 else 'UNSTABLE'} ({nuniq} choices/{len(g['chosen_per_fold'])} folds)")
                        honest_all_rows.append(dict(
                            strategy=fam_label, scope="GLOBAL", cluster=-1,
                            **gt, sharpe_maker=gm["sharpe"],
                            mean_net_pnl_maker=gm["mean_net_pnl"],
                            chosen=str(g["chosen_per_fold"]),
                        ))
                        for cl, r in sorted(res["per_cluster"].items()):
                            ct = r["oos_taker"]; cm = r["oos_maker"]
                            print(f"        [{fam_label}] Cl {cl}: taker mean={ct['mean_net_pnl']:+.2f} "
                                  f"sharpe={ct['sharpe']:.3f} n={ct['n']}")
                            honest_all_rows.append(dict(
                                strategy=fam_label, scope="CLUSTER", cluster=cl,
                                **ct, sharpe_maker=cm["sharpe"],
                                mean_net_pnl_maker=cm["mean_net_pnl"],
                                chosen=str(r["chosen_per_fold"]),
                            ))

                    if honest_all_rows:
                        df_honest = pd.DataFrame(honest_all_rows)
                        df_honest.to_csv(out_dir / f"honest_oos_all_strategies_{tag}.csv", index=False)
                        # best honest GLOBAL result as the verdict
                        gdf = df_honest[df_honest["scope"] == "GLOBAL"]
                        if len(gdf):
                            bestrow = gdf.loc[gdf["sharpe"].idxmax()]
                            verdict = "POSITIVE" if bestrow["mean_net_pnl"] > 0 else "NO EDGE after costs"
                            print(f"\n      ── HONEST VERDICT: best family '{bestrow['strategy']}' "
                                  f"OOS taker mean={bestrow['mean_net_pnl']:+.2f} bps "
                                  f"sharpe={bestrow['sharpe']:.3f} → {verdict} ──")

                    # ── In-sample 'optimal' table (descriptive ONLY, clearly labelled) ──
                    print(f"\n      [IN-SAMPLE optimal table — descriptive, optimistically biased:]")
                    # Best static TP/SL
                    for cost_type in ["taker", "maker"]:
                        col = f"mean_net_pnl_{cost_type}"
                        best = df_grid.nlargest(1, col).iloc[0]
                        optimal_rows.append(dict(
                            strategy_label=f"[IS] Best TP/SL ({cost_type})",
                            cost_type=cost_type,
                            mean_net_pnl=best[col],
                            total_net_pnl=best[f"total_net_pnl_{cost_type}"],
                            win_rate=best[f"win_rate_{cost_type}"],
                            sharpe=best[f"sharpe_{cost_type}"],
                            details=f"TP={int(best['tp_bps'])} SL={int(best['sl_bps'])}",
                        ))

                    # Best time exit
                    for cost_type in ["taker", "maker"]:
                        best = df_time[df_time["cost_type"]==cost_type].nlargest(1, "mean_net_pnl").iloc[0]
                        optimal_rows.append(dict(
                            strategy_label=f"[IS] Best time exit",
                            cost_type=cost_type,
                            mean_net_pnl=best["mean_net_pnl"],
                            total_net_pnl=best["total_net_pnl"],
                            win_rate=best["win_rate"],
                            sharpe=best["sharpe"],
                            details=f"timeout={int(best['timeout_s'])}s",
                        ))

                    # Best trailing
                    for cost_type in ["taker", "maker"]:
                        best = df_trail[df_trail["cost_type"]==cost_type].nlargest(1, "mean_net_pnl").iloc[0]
                        tp_str = f"TP={int(best['tp_bps'])}" if best["tp_bps"] > 0 else "no TP"
                        optimal_rows.append(dict(
                            strategy_label=f"[IS] Best trailing stop",
                            cost_type=cost_type,
                            mean_net_pnl=best["mean_net_pnl"],
                            total_net_pnl=best["total_net_pnl"],
                            win_rate=best["win_rate"],
                            sharpe=best["sharpe"],
                            details=f"trail={int(best['trail_offset'])} {tp_str} t/o={int(best['timeout_s'])}s",
                        ))

                    # Best overall combined
                    for cost_type in ["taker", "maker"]:
                        best = df_combined[df_combined["cost_type"]==cost_type].nlargest(1, "mean_net_pnl").iloc[0]
                        tp_str = f"TP={int(best['tp_bps'])}" if best.get("tp_bps", 0) > 0 else ""
                        trail_str = f"trail={int(best['trail_offset'])}" if best.get("trail_offset", 0) > 0 else ""
                        timeout_str = f"t/o={int(best['timeout_s'])}s" if best.get("timeout_s", 0) > 0 else ""
                        optimal_rows.append(dict(
                            strategy_label=f"[IS] Best combined",
                            cost_type=cost_type,
                            mean_net_pnl=best["mean_net_pnl"],
                            total_net_pnl=best["total_net_pnl"],
                            win_rate=best["win_rate"],
                            sharpe=best["sharpe"],
                            details=f"{tp_str} {trail_str} {timeout_str}".strip(),
                        ))

                    df_optimal = pd.DataFrame(optimal_rows)
                    df_optimal.to_csv(out_dir / f"optimal_strategy_INSAMPLE_{tag}.csv", index=False)

                    # Plot comparison
                    # plot_strategy_comparison(df_optimal, tag, plot_dir)

                    # ── Console summary ───────────────────────────────────
                    print(f"\n    ╔{'═'*66}╗")
                    print(f"    ║  WS3 SUMMARY  {tag:>50} ║")
                    print(f"    ╠{'═'*66}╣")
                    for cost_type in ["taker", "maker"]:
                        ct_rows = df_optimal[df_optimal["cost_type"] == cost_type]
                        print(f"    ║  {cost_type.upper():>6} cost ({taker_cost if cost_type=='taker' else maker_cost} bps):")
                        print(f"    ║  {'Strategy':<30} {'PnL':>8} {'WR':>7} {'Sharpe':>8} {'Details':<20} ║")
                        print(f"    ╠{'─'*66}╣")
                        for _, r in ct_rows.iterrows():
                            print(f"    ║  {r['strategy_label']:<30} "
                                  f"{r['mean_net_pnl']:>+7.2f} "
                                  f"{r['win_rate']:>6.1%} "
                                  f"{r['sharpe']:>7.3f} "
                                  f"{str(r['details']):<20} ║")
                        print(f"    ╠{'═'*66}╣")
                    print(f"    ╚{'═'*66}╝")

                    print(f"\n    Files saved to: {out_dir}/")
                    print(f"      tp_sl_grid_{tag}.csv")
                    print(f"      tp_sl_grid_by_cluster_{tag}.csv")
                    print(f"      time_exits_{tag}.csv")
                    print(f"      trailing_stop_{tag}.csv")
                    print(f"      combined_exits_{tag}.csv")
                    print(f"      optimal_strategy_{tag}.csv")
                    print(f"      sub_cluster_analysis_{tag}.csv  (Phase D)")

                    # ══════════════════════════════════════════════════════
                    #  PHASE D: SUB-CLUSTER HETEROGENEITY ANALYSIS
                    # ══════════════════════════════════════════════════════
                    # Goal: detect whether a cluster contains sub-populations
                    # with meaningfully different optimal TP/SL dynamics.
                    # If sub-groups differ in optimal TP by >= SUBCLUSTER_TP_DIVERGENCE_BPS,
                    # the cluster is flagged for further feature-based investigation.
                    #
                    # Sub-groups are defined by:
                    #   - MAE tercile: tight (mae > p33), medium, loose (mae < p67)
                    #   - MFE speed: fast (mfe_time < median), slow (mfe_time >= median)
                    # Resulting in up to 6 sub-groups per cluster.
                    # TP/SL sweep uses TP_COARSE x SL_COARSE (104 combos) for speed.
                    # ──────────────────────────────────────────────────────
                    print(f"\n    === Phase D: Sub-cluster heterogeneity analysis ===")

                    # Require these columns from the cluster_trades CSV
                    has_mfe_time = "mfe_time_s" in df_trades.columns
                    has_mae      = "mae_bps" in df_trades.columns

                    subcluster_rows = []
                    flagged_clusters = []

                    for cl in unique_cl:
                        cl_mask     = clusters == cl
                        cl_paths    = paths[cl_mask]
                        cl_df       = df_trades[cl_mask].reset_index(drop=True)
                        n_cl        = int(cl_mask.sum())

                        if n_cl < SUBCLUSTER_MIN_N * 2:
                            print(f"      Cl {cl}: skipped (N={n_cl} < min)")
                            continue

                        # ── Define sub-group labels ───────────────────────
                        # MAE tercile
                        if has_mae:
                            mae_vals = cl_df["mae_bps"].values.astype(float)
                            p33, p67 = np.percentile(mae_vals, [33, 67])
                            mae_lbl = np.where(
                                mae_vals >= p67, "loose",
                                np.where(mae_vals >= p33, "medium", "tight")
                            )
                        else:
                            mae_lbl = np.full(n_cl, "all")

                        # MFE speed (fast vs slow peak)
                        if has_mfe_time:
                            mfe_t = cl_df["mfe_time_s"].values.astype(float)
                            med_t = np.median(mfe_t)
                            spd_lbl = np.where(mfe_t < med_t, "fast", "slow")
                        else:
                            spd_lbl = np.full(n_cl, "all")

                        sg_labels = np.array([f"{s}_{m}" for s, m in
                                              zip(spd_lbl, mae_lbl)])
                        unique_sgs = np.unique(sg_labels)

                        # ── Sweep each sub-group ──────────────────────────
                        sg_optima = {}   # sg_name -> (best_tp, best_sl, best_pnl)

                        for sg in unique_sgs:
                            sg_mask  = sg_labels == sg
                            n_sg     = int(sg_mask.sum())
                            if n_sg < SUBCLUSTER_MIN_N:
                                continue
                            sg_paths = cl_paths[sg_mask]
                            sg_sub   = cl_df[sg_mask].reset_index(drop=True)

                            best_pnl = -999.0
                            best_tp  = 0
                            best_sl  = 0
                            best_wr  = 0.0

                            for tp in TP_COARSE:
                                for sl in SL_COARSE:
                                    gpnl, outcomes, etimes = simulate_tp_sl(
                                        sg_paths, tp, sl)
                                    m = compute_strategy_metrics(
                                        gpnl, outcomes, etimes, maker_cost)
                                    if m["mean_net_pnl"] > best_pnl:
                                        best_pnl = m["mean_net_pnl"]
                                        best_tp  = tp
                                        best_sl  = sl
                                        best_wr  = m["win_rate"]

                            sg_optima[sg] = (best_tp, best_sl, best_pnl)
                            subcluster_rows.append(dict(
                                cluster=int(cl),
                                subgroup=sg,
                                n=n_sg,
                                n_cluster=n_cl,
                                pct_of_cluster=round(n_sg / n_cl, 3),
                                best_tp=best_tp,
                                best_sl=best_sl,
                                best_pnl_maker=round(best_pnl, 3),
                                best_wr=round(best_wr, 4),
                                mae_p50=round(float(sg_sub["mae_bps"].median()), 2)
                                        if has_mae else None,
                                mfe_time_p50=round(float(sg_sub["mfe_time_s"].median()), 1)
                                             if has_mfe_time else None,
                            ))

                        # ── Check heterogeneity ───────────────────────────
                        if len(sg_optima) >= 2:
                            opt_tps = [v[0] for v in sg_optima.values()]
                            opt_sls = [v[1] for v in sg_optima.values()]
                            tp_spread = max(opt_tps) - min(opt_tps)
                            sl_spread = max(opt_sls) - min(opt_sls)

                            is_heterogeneous = (
                                tp_spread >= SUBCLUSTER_TP_DIVERGENCE_BPS or
                                sl_spread >= SUBCLUSTER_SL_DIVERGENCE_BPS
                            )
                            flag = "*** HETEROGENEOUS ***" if is_heterogeneous else "homogeneous"
                            print(f"      Cl {cl} (N={n_cl}): {flag}  "
                                  f"TP spread={tp_spread} bps  SL spread={sl_spread} bps")
                            for sg, (btp, bsl, bpnl) in sg_optima.items():
                                print(f"        [{sg}]  TP={btp}  SL={bsl}  "
                                      f"mean_net={bpnl:+.2f} bps")
                            if is_heterogeneous:
                                flagged_clusters.append(cl)
                        else:
                            print(f"      Cl {cl}: only {len(sg_optima)} valid sub-group(s), skipping")

                    df_sub = pd.DataFrame(subcluster_rows)
                    if len(df_sub) > 0:
                        df_sub.to_csv(out_dir / f"sub_cluster_analysis_{tag}.csv",
                                      index=False)

                    if flagged_clusters:
                        print(f"\n    *** Heterogeneous clusters: {flagged_clusters}")
                        print(f"    *** These clusters may benefit from sub-group-specific")
                        print(f"    *** TP/SL in live trading. Investigate raw features")
                        print(f"    *** to find what distinguishes sub-groups at entry time.")
                    else:
                        print(f"    All clusters appear homogeneous within Phase D resolution.")

                    # Explicit cleanup between lookback iterations
                    del paths; gc.collect()

            elapsed = time.time() - t0
            print(f"\n  ━━ {asset.upper()}/{hz} done in {elapsed:.0f}s ━━")
            # Free y_1s now that all thresholds/lookbacks for this hz are done
            try:
                del y_1s
            except NameError:
                pass
            gc.collect()
            check_ram(label="after hz complete")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="WS3: TP/SL grid + dynamic exits on cluster-filtered trades")
    p.add_argument("--asset", choices=["btc","eth","both"], default="btc")
    p.add_argument("--hz", nargs="+", default=["15s"])
    p.add_argument("--thresholds", nargs="+", type=int, default=THRESHOLDS)
    p.add_argument("--lookbacks", nargs="+", type=int, default=LOOKBACKS)
    p.add_argument("--strategy", type=str, default=None,
                   help="Strategy name for cluster validation (optional).")
    p.add_argument("--trades-file", type=str, default=None,
                   help="Direct path to cluster_trades_*.csv (overrides the tag search).")
    p.add_argument("--log-level", default="INFO")
    a = p.parse_args()
    logging.basicConfig(
        level=getattr(logging, a.log_level),
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S", stream=sys.stdout,
    )
    assets = ("btc","eth") if a.asset == "both" else (a.asset,)
    run_ws3(
        assets=assets, horizons=tuple(a.hz),
        thresholds=a.thresholds, lookbacks=a.lookbacks,
        trades_file=a.trades_file,
    )


if __name__ == "__main__":
    main()