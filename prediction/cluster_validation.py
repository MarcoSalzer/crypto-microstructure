# prediction/cluster_validation.py
# ==============================================================================
# WS4c: Rolling Cluster Validation
# ==============================================================================
#
# PURPOSE:
#   The KMeans clusters in our pipeline are fitted on the FULL dataset
#   (~3 months, Feb–May). This is an in-sample concern: do the same cluster
#   structures emerge when we fit on different time windows? If a cluster only
#   appears in one specific block, it's not structural — it's noise.
#
#   This script answers three questions:
#   1. STABILITY: Do similar clusters emerge when KMeans is fitted
#      independently on 5 separate equal-size event blocks?
#   2. CARRY-OVER: If we fit clusters on the first 60% of events and predict
#      on the last 40%, do the profitable clusters carry over?
#   3. PROFILE CONSISTENCY: Do the feature profiles (what makes the target
#      cluster special) remain consistent across time windows?
#
# TESTS PERFORMED:
#
#   Test 1 — Block-wise cluster stability
#   ──────────────────────────────────────
#   Split events into 5 equal-size blocks (NOT fixed calendar days — block
#   size = n_events // 5, so it scales with the dataset). Fit KMeans
#   independently on each block; compare each block's labels against the
#   full-dataset labels via Adjusted Rand Index (ARI). High ARI = consistent.
#
#   Test 2 — Temporal carry-over (train/test split)
#   ────────────────────────────────────────────────
#   Fit clusters on the first 60% of events, assign the last 40% with the
#   trained KMeans. Check: are the "profitable" training clusters still
#   profitable in the test period? Compute PnL, DA, MFE per cluster in test.
#
#   Test 3 — Rolling window cluster re-estimation
#   ──────────────────────────────────────────────
#   Rolling 10-day window, 2-day step. Day length is derived from the real
#   sample count (n_samples / 86400 at 1s resolution), NOT a hardcoded 25.
#   At each step: fit KMeans, identify profitable clusters, compute OOS PnL
#   on the next step. Track whether profitability persists across windows.
#
#   Test 4 — Feature profile stability
#   ───────────────────────────────────
#   For EVERY profitable full-dataset cluster (not just the largest): compute
#   the top-50 defining-feature z-scores in each block separately, with their
#   magnitude (mean_z / std_z), not only the sign. Same features elevated in
#   all blocks = a real microstructure regime. Output carries a `cluster`
#   column so each good cluster's signature is recoverable; family attribution
#   is done downstream by feature_family_analysis.py via Annex Part B.
#
# OUTPUTS (in RESULTS_DIR/cluster_validation/):
#   block_stability_{tag}.csv        ARI scores between blocks
#   carryover_test_{tag}.csv         Train/test cluster PnL comparison
#   rolling_window_{tag}.csv         Rolling window profitability
#   feature_stability_{tag}.csv      Top-50 z-scores per block, ALL good clusters
#   cluster_validation_summary_{tag}.csv  Overall validation verdict
#   ws4c_plots/ws4c_*.png            Visualization
#
# USAGE:
#   python cluster_validation.py --asset eth --lookbacks 5
#   python cluster_validation.py --asset both
#
# RUNTIME: ~15-20 min per asset (multiple KMeans fits + LightGBM classifiers)
# ==============================================================================
from __future__ import annotations
# Optional deterministic mode (opt-in via WS4_DETERMINISTIC=1); default OFF so
# the full clustering matches cluster_final. See cluster_engine for the rationale.
import os as _os
if _os.environ.get("WS4_DETERMINISTIC") == "1":
    for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
               "NUMEXPR_NUM_THREADS"):
        _os.environ[_v] = "1"
import argparse, gc, logging, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# DEFAULTS (2026-06): aligned to the viable configs carried forward from
# the clustering grid (Section 4.4, Table 4.Z). The primary config is
# BTC/5s/15bps, pca600, k=6. These are the values used if no CLI flag is
# given — set so an argument-less run validates a REAL config rather than
# the default placeholder. Override per-config via CLI.
THRESHOLDS = [15]          # was [20] (compat). Viable thresholds: 10/15/40.
LOOKBACKS  = [1]           # was [1,2,5,10]. Lead time fixed at 1s (Sec 4.4).
DEFAULT_PCA_DIM = 600      # primary config pca dim
DEFAULT_K  = 6             # all viable configs use k=6
# Compat per-threshold k map, kept only as a fallback when --k is omitted
# AND the threshold is not the primary one. Prefer always passing --k.
K_CLUSTERS = {10: 6, 15: 6, 20: 7, 40: 6}
MFE_LOOKAHEAD = 300

# Test 4 reporting depth: how many top defining features to track per cluster.
TOP_N_FEATURES = 50


def _setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.grid": True, "grid.alpha": 0.3, "font.size": 11,
    })
    return plt


def fit_clusters(X_events, event_directions, y_events, k, taker_cost, pca_dim=30):
    """
    Fit PCA+KMeans on event features, return labels + cluster stats.
    Identical logic to the cluster pipeline / cluster_engine.py.

    NOTE on the `da` field: this is an IN-SAMPLE one-sidedness used only as an
    internal screen for which clusters to *track* in the validation tests. The
    reported predictive numbers come from the OOS tests (carry-over / rolling),
    which fix the direction on train and measure hit-rate on held-out data via
    compute_cluster_pnl. The honest directed-OOS selection that feeds the main
    results tables lives in cluster_engine (cluster_eval.select_good_clusters_oos).
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans

    nan_frac = np.isnan(X_events).mean(axis=0)
    valid_cols = np.where(nan_frac < 0.95)[0]
    X_clean = X_events[:, valid_cols]

    imputer = SimpleImputer(strategy="median")
    scaler  = StandardScaler()
    X_clean = imputer.fit_transform(X_clean)
    X_scaled = scaler.fit_transform(X_clean)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0)

    n_comp = min(pca_dim, X_scaled.shape[1], len(X_events) - 1)
    _svd = "full" if _os.environ.get("WS4_DETERMINISTIC") == "1" else "auto"
    pca = PCA(n_components=n_comp, random_state=42, svd_solver=_svd)
    X_pca = pca.fit_transform(X_scaled)

    km = KMeans(n_clusters=k, n_init=20, random_state=42)
    labels = km.fit_predict(X_pca)

    # Compute cluster stats
    stats = {}
    good_clusters = []
    for c in range(k):
        c_mask = labels == c
        if c_mask.sum() < 10:
            continue
        # Continuation-frame return r_cont = breakout_dir * forward_move
        # (identical to cluster_engine). DA = one-sidedness of r_cont; majority = +1
        # continuation / -1 reversal. Directional gate only (DA>0.55),
        # consistent with the ws4 selection; profitability is Section 4.5.
        rc = event_directions[c_mask] * y_events[c_mask]
        up_ratio = (rc > 0).mean()
        da = max(up_ratio, 1 - up_ratio)
        majority = 1 if up_ratio > 0.5 else -1
        mean_abs_ret = np.abs(y_events[c_mask]).mean() * 10_000

        stats[c] = dict(
            n=int(c_mask.sum()), da=round(float(da), 4), majority=majority,
            mean_abs_ret=round(mean_abs_ret, 2),
            up_ratio=round(float(up_ratio), 3),
        )
        if da > 0.55:
            good_clusters.append(c)

    return labels, stats, good_clusters, km, pca, imputer, scaler, valid_cols


def assign_clusters(X_events, km, pca, imputer, scaler, valid_cols):
    """Assign cluster labels using pre-fitted pipeline."""
    X_clean = X_events[:, valid_cols]
    X_clean = imputer.transform(X_clean)
    X_scaled = scaler.transform(X_clean)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0)
    X_pca = pca.transform(X_scaled)
    return km.predict(X_pca)


def compute_cluster_pnl(labels, event_directions, y_events, cluster_stats, maker_cost):
    """Directional OOS accuracy per cluster in the continuation frame.
    Trade return = majority * r_cont = majority * breakout_dir * forward_move,
    with majority fixed on the training data. `da` is the directional accuracy;
    net PnL is retained for reference only (Section 4.5 does costed tradability)."""
    results = {}
    for c in np.unique(labels):
        c_mask = labels == c
        if c_mask.sum() < 5:
            continue
        if c not in cluster_stats:
            continue
        majority = cluster_stats[c]["majority"]
        returns_bps = (event_directions[c_mask] * y_events[c_mask]) * majority * 10_000
        net = returns_bps - maker_cost
        results[c] = dict(
            n=int(c_mask.sum()),
            da=round(float((returns_bps > 0).mean()), 4),
            mean_gross=round(float(returns_bps.mean()), 2),
            mean_net=round(float(net.mean()), 2),
            total_net=round(float(net.sum()), 1),
            win_rate=round(float((net > 0).mean()), 4),
            sharpe=round(float(net.mean() / net.std()), 4) if net.std() > 0 else 0,
        )
    return results


def _feature_signature(X_events, full_labels, target_cl, feat_names,
                       n_blocks, block_size, n_events, top_n=TOP_N_FEATURES):
    """Top-`top_n` defining features of `target_cl` + per-block z-scores.

    Returns a list of row dicts (one per feature) carrying full_z, the per-block
    z-scores, mean_z/std_z (magnitude, not only sign) and consistent_sign.
    """
    cl_mask_full = full_labels == target_cl
    other_mask_full = ~cl_mask_full

    top_features = []
    for fi, fname in enumerate(feat_names):
        cl_vals = X_events[cl_mask_full, fi]
        ot_vals = X_events[other_mask_full, fi]
        cl_clean = cl_vals[~np.isnan(cl_vals)]
        ot_clean = ot_vals[~np.isnan(ot_vals)]
        if len(cl_clean) < 10 or len(ot_clean) < 10:
            continue
        z = (cl_clean.mean() - ot_clean.mean()) / ot_clean.std() if ot_clean.std() > 0 else 0
        top_features.append((fi, fname, abs(z), z))

    top_features.sort(key=lambda x: -x[2])
    top_sel = top_features[:top_n]

    rows = []
    for fi, fname, _, full_z in top_sel:
        block_zscores = []
        for b in range(n_blocks):
            b_start = b * block_size
            b_end = (b + 1) * block_size if b < n_blocks - 1 else n_events

            block_cl_mask = full_labels[b_start:b_end] == target_cl
            block_other_mask = ~block_cl_mask

            cl_vals = X_events[b_start:b_end][block_cl_mask, fi]
            ot_vals = X_events[b_start:b_end][block_other_mask, fi]
            cl_clean = cl_vals[~np.isnan(cl_vals)]
            ot_clean = ot_vals[~np.isnan(ot_vals)]

            if len(cl_clean) < 5 or len(ot_clean) < 5:
                block_zscores.append(np.nan)
                continue

            z = (cl_clean.mean() - ot_clean.mean()) / ot_clean.std() if ot_clean.std() > 0 else 0
            block_zscores.append(round(z, 3))

        valid_z = [z for z in block_zscores if not np.isnan(z)]
        if len(valid_z) >= 3:
            same_sign = all(z > 0 for z in valid_z) or all(z < 0 for z in valid_z)
            mean_z = float(np.mean(valid_z))
            std_z = float(np.std(valid_z))
        else:
            same_sign = False
            mean_z = 0.0
            std_z = 0.0

        rows.append(dict(
            cluster=int(target_cl),
            feature=fname,
            full_z=round(full_z, 3),
            block1_z=block_zscores[0],
            block2_z=block_zscores[1],
            block3_z=block_zscores[2],
            block4_z=block_zscores[3],
            block5_z=block_zscores[4] if len(block_zscores) > 4 else np.nan,
            mean_z=round(mean_z, 3),
            std_z=round(std_z, 3),
            consistent_sign=same_sign,
        ))
    return rows


def run_ws4c(
    assets=("eth",),
    horizons=("15s",),
    thresholds=None,
    lookbacks=None,
    pca_dim=DEFAULT_PCA_DIM,
    max_hours=None,
    k_override=DEFAULT_K,
):
    from common.data_loader import load_dataset
    from common.config import RESULTS_DIR, SPREAD_BPS, MAKER_COST_BPS
    from sklearn.metrics import adjusted_rand_score

    if thresholds is None:
        thresholds = THRESHOLDS
    if lookbacks is None:
        lookbacks = LOOKBACKS

    out_dir = RESULTS_DIR / "cluster_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = out_dir / "ws4c_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for asset in assets:
        # Fallbacks match config v6 (round-trip taker 10 bps, maker 4 bps).
        # Only used if asset is absent from the dicts; btc/eth are present.
        taker_cost = SPREAD_BPS.get(asset, {}).get("fut", 10.0)
        maker_cost = MAKER_COST_BPS.get(asset, {}).get("fut", 4.0)

        for hz in horizons:
            t0 = time.time()
            print(f"\n{'━'*70}")
            print(f"  WS4c Rolling Cluster Validation — {asset.upper()}/{hz}")
            print(f"{'━'*70}")

            try:
                X, y, info, feat_names = load_dataset(target=hz, asset=asset, profile="cluster", max_hours=max_hours)
            except Exception as e:
                logger.error("Load fail: %s", e)
                continue

            # Match ws4 EXACTLY: truncate to the common length with the 1s target
            # so the trailing-move events land on the same rows as the main run.
            if hz != "1s":
                try:
                    _, y_1s, _, _ = load_dataset(
                        target="1s", asset=asset, max_hours=max_hours,
                        target_only=True)
                    n_min = min(len(X), len(y_1s))
                    X, y = X[:n_min], y[:n_min]
                except Exception as e:
                    logger.error("1s load fail: %s", e)

            n = len(X)
            print(f"  Data: {n:,} samples, {X.shape[1]} features")

            hz_sec = int(hz[:-1])   # "30s" -> 30
            for thr_bps in thresholds:
                thr_dec = thr_bps / 10_000
                # TRAILING move (breakout completed at T), identical to ws4:
                # trailing_move[t] = forward return that started hz_sec ago.
                trailing_move = np.full_like(y, np.nan)
                trailing_move[hz_sec:] = y[:len(y) - hz_sec]
                event_mask = np.abs(trailing_move) > thr_dec
                event_indices = np.where(event_mask)[0]
                n_events = len(event_indices)
                event_directions = np.sign(trailing_move[event_indices])
                X_events = X[event_indices]
                y_events = y[event_indices]
                # k: CLI override takes precedence over the per-threshold dict.
                # This way the winner configuration found in the grid can be used
                # (e.g. k=6 at 15bps) directly, without editing K_CLUSTERS.
                k = k_override if k_override is not None else K_CLUSTERS.get(thr_bps, 7)

                if n_events < 500:
                    print(f"  {thr_bps} bps: {n_events} events — skipping")
                    continue

                print(f"\n  ── {thr_bps} bps: {n_events} events, k={k} ──")
                # cfg suffix with pca+k so runs of different pca/k combos
                # do NOT overwrite the same output files (the tag contained
                # previously only asset/hz/thr/lb).
                cfg = f"pca{pca_dim}_k{k}"
                tag = f"{asset}_{hz}_{thr_bps}bps_{cfg}"   # cluster-level tag (lookback-independent)

                # ══════════════════════════════════════════════════════════
                #  FULL-DATASET BASELINE
                # ══════════════════════════════════════════════════════════
                print(f"\n    Fitting full-dataset clusters (baseline)...")
                full_labels, full_stats, full_good, full_km, full_pca, \
                    full_imp, full_scaler, full_vcols = fit_clusters(
                        X_events, event_directions, y_events, k, taker_cost, pca_dim=pca_dim)

                print(f"    Good clusters (full): {full_good}")
                for c, s in sorted(full_stats.items()):
                    mark = " *" if c in full_good else ""
                    print(f"      Cl {c}: N={s['n']:>5}, DA={s['da']:.3f}, "
                          f"maj={s['majority']:+d}{mark}")

                # ══════════════════════════════════════════════════════════
                #  TEST 1: BLOCK-WISE STABILITY
                # ══════════════════════════════════════════════════════════
                print(f"\n    === Test 1: Block-wise stability (5 blocks) ===")

                n_blocks = 5
                block_size = n_events // n_blocks
                block_labels = []
                block_stats_list = []
                block_good_list = []

                for b in range(n_blocks):
                    b_start = b * block_size
                    b_end = (b + 1) * block_size if b < n_blocks - 1 else n_events

                    X_block = X_events[b_start:b_end]
                    d_block = event_directions[b_start:b_end]
                    y_block = y_events[b_start:b_end]

                    bl, bs, bg, _, _, _, _, _ = fit_clusters(
                        X_block, d_block, y_block, k, taker_cost, pca_dim=pca_dim)

                    # Pad to full length for ARI comparison
                    full_len_labels = np.full(n_events, -1, dtype=int)
                    full_len_labels[b_start:b_end] = bl
                    block_labels.append(full_len_labels)
                    block_stats_list.append(bs)
                    block_good_list.append(bg)

                    n_good = len(bg)
                    good_pct = sum(bs[c]["n"] for c in bg) / (b_end - b_start) * 100 if bg else 0
                    print(f"      Block {b+1} ({b_end-b_start} events): "
                          f"{n_good} good clusters, {good_pct:.0f}% of events in good clusters")

                # ARI between each block's labels and full-dataset labels
                ari_rows = []
                for b in range(n_blocks):
                    b_start = b * block_size
                    b_end = (b + 1) * block_size if b < n_blocks - 1 else n_events
                    block_fl = full_labels[b_start:b_end]
                    block_bl = block_labels[b][b_start:b_end]
                    ari = adjusted_rand_score(block_fl, block_bl)
                    ari_rows.append(dict(block=b+1, ari_vs_full=round(ari, 4),
                                         n_events=b_end-b_start,
                                         n_good_clusters=len(block_good_list[b])))
                    print(f"      Block {b+1} ARI vs full-dataset: {ari:.3f}")

                # ── HONEST stability test: block b vs. block b+1 ──────────
                # idea: train KMeans ONLY on block b, apply it to block b+1
                # (km.predict → assignment_from_b). Train KMeans separately on
                # block b+1 (native_b1). ARI(assignment_from_b, native_b1) measures
                # whether the geometry learned in block b still produces the same
                # partition is produced. NO full dataset involved → not circular.
                #
                # Note: ARI is permutation-invariant, so the label
                # matching between the two KMeans fits is not a problem.
                consec_rows = []
                for b in range(n_blocks - 1):
                    b1_s = b * block_size
                    b1_e = (b + 1) * block_size
                    b2_s = (b + 1) * block_size
                    b2_e = (b + 2) * block_size if b + 1 < n_blocks - 1 else n_events

                    # KMeans pipeline from block b (already fitted above would be possible,
                    # but we need the estimators → refit here, cleanly encapsulated)
                    _, _, _, km_b, pca_b, imp_b, scl_b, vc_b = fit_clusters(
                        X_events[b1_s:b1_e], event_directions[b1_s:b1_e],
                        y_events[b1_s:b1_e], k, taker_cost, pca_dim=pca_dim)
                    # Block b+1: native partition
                    lab_b1_native, _, _, _, _, _, _, _ = fit_clusters(
                        X_events[b2_s:b2_e], event_directions[b2_s:b2_e],
                        y_events[b2_s:b2_e], k, taker_cost, pca_dim=pca_dim)
                    # block b+1 assigned with the block-b pipeline (out of sample)
                    lab_b1_from_b = assign_clusters(
                        X_events[b2_s:b2_e], km_b, pca_b, imp_b, scl_b, vc_b)

                    ari_consec = adjusted_rand_score(lab_b1_native, lab_b1_from_b)
                    consec_rows.append(dict(
                        block_pair=f"{b+1}->{b+2}",
                        ari_consecutive=round(float(ari_consec), 4),
                        n_test=int(b2_e - b2_s),
                    ))
                    print(f"      Block {b+1}→{b+2} ARI (out-of-sample): {ari_consec:.3f}")

                df_ari = pd.DataFrame(ari_rows)
                df_consec = pd.DataFrame(consec_rows)
                if len(df_consec):
                    mean_consec = df_consec["ari_consecutive"].mean()
                    print(f"      ── Mean consecutive-block ARI: {mean_consec:.3f} "
                          f"({'STABLE' if mean_consec > 0.5 else 'UNSTABLE'}) ──")
                    df_consec.to_csv(out_dir / f"block_ari_consecutive_{tag}.csv", index=False)

                # ══════════════════════════════════════════════════════════
                #  TEST 2: TEMPORAL CARRY-OVER (60/40 SPLIT)
                # ══════════════════════════════════════════════════════════
                print(f"\n    === Test 2: Temporal carry-over (60/40 split) ===")

                split_idx = int(n_events * 0.6)
                train_X = X_events[:split_idx]
                train_dirs = event_directions[:split_idx]
                train_y = y_events[:split_idx]
                test_X = X_events[split_idx:]
                test_dirs = event_directions[split_idx:]
                test_y = y_events[split_idx:]

                print(f"      Train: {split_idx} events, Test: {n_events - split_idx} events")

                # Fit on train
                train_labels, train_stats, train_good, train_km, train_pca, \
                    train_imp, train_scaler, train_vcols = fit_clusters(
                        train_X, train_dirs, train_y, k, taker_cost, pca_dim=pca_dim)

                print(f"      Train good clusters: {train_good}")
                for c, s in sorted(train_stats.items()):
                    mark = " *" if c in train_good else ""
                    print(f"        Cl {c}: N={s['n']:>5}, DA={s['da']:.3f}, "
                          f"maj={s['majority']:+d}{mark}")

                # Assign test events using train KMeans
                test_labels = assign_clusters(test_X, train_km, train_pca,
                                              train_imp, train_scaler, train_vcols)

                # Compute PnL for each cluster in test period
                test_pnl = compute_cluster_pnl(test_labels, test_dirs, test_y,
                                               train_stats, maker_cost)

                carryover_rows = []
                print(f"\n      Test period results (using TRAIN cluster definitions):")
                for c in sorted(test_pnl.keys()):
                    tp = test_pnl[c]
                    in_train_good = c in train_good
                    mark = " * CARRY-OVER" if in_train_good and tp["da"] > 0.55 else ""
                    mark = " ! DEGRADED" if in_train_good and tp["da"] <= 0.55 else mark
                    print(f"        Cl {c}: N={tp['n']:>5}, DA={tp['da']:.3f}, "
                          f"maker PnL={tp['mean_net']:+.2f}, "
                          f"WR={tp['win_rate']:.1%}{mark}")
                    carryover_rows.append(dict(
                        cluster=c, train_good=in_train_good,
                        train_da=train_stats.get(c, {}).get("da", 0),
                        test_n=tp["n"], test_da=tp["da"],
                        test_mean_net=tp["mean_net"],
                        test_total_net=tp["total_net"],
                        test_win_rate=tp["win_rate"],
                        test_sharpe=tp["sharpe"],
                        carries_over=in_train_good and tp["da"] > 0.55,
                    ))

                df_carry = pd.DataFrame(carryover_rows)
                n_carry = df_carry[df_carry["carries_over"]].shape[0]
                n_train_good = len(train_good)
                print(f"\n      Carry-over rate: {n_carry}/{n_train_good} "
                      f"train-profitable clusters remain profitable in test")

                # ══════════════════════════════════════════════════════════
                #  TEST 3: ROLLING WINDOW
                # ══════════════════════════════════════════════════════════
                print(f"\n    === Test 3: Rolling window (10-day train, 2-day step) ===")

                # Convert event indices to approximate day numbers.
                # FIX (2026-06): the divisor was hardcoded to 25 (compat 25-day
                # dataset). With ~3 months of data this mislabels the window as
                # "10 days" while actually covering ~11% of events. Derive the
                # real day count from the sample count: load_dataset(target=hz)
                # returns 1s-resolution rows, so n_samples == seconds of data.
                # NOTE: this counts *available* seconds, not calendar span — if
                # the feed has gaps, n_days < calendar days. That is the correct
                # basis here, since windows should scale with available events.
                n_days = max(n / 86_400.0, 1.0)   # n = total 1s samples (run_ws4c scope)
                events_per_day = n_events / n_days
                window_events = int(10 * events_per_day)  # 10 days
                step_events = int(2 * events_per_day)  # 2 days

                # FIX (2026-06): for sparse configs (e.g. BTC/1s/10bps, ~17
                # events/day) a 2-day step yields test windows of ~34 events,
                # which the >=50-trade filter below rejects for EVERY window.
                # rolling_rows then stays empty and df_rolling has no
                # 'profitable' column → KeyError. We require the test window to
                # hold at least MIN_TEST_TRADES events; if the natural step is
                # too small, widen the step (and proportionally the window) so
                # that windows are actually evaluated. This trades temporal
                # resolution for evaluability on thin configs and is reported.
                MIN_TEST_TRADES = 50
                if step_events < MIN_TEST_TRADES:
                    scale = int(np.ceil(MIN_TEST_TRADES / max(step_events, 1)))
                    step_events *= scale
                    window_events = max(window_events, 5 * step_events)
                    print(f"      [sparse-config adjust] step widened ×{scale} "
                          f"→ window={window_events}, step={step_events} "
                          f"(to satisfy >= {MIN_TEST_TRADES} test trades/window)")

                print(f"      Data spans ~{n_days:.1f} days "
                      f"({events_per_day:.0f} events/day); "
                      f"window={window_events} events, step={step_events} events")

                rolling_rows = []
                window_num = 0

                for w_start in range(0, max(n_events - window_events, 1), step_events):
                    w_end = w_start + window_events
                    test_start = w_end
                    test_end = min(w_end + step_events, n_events)

                    if test_end - test_start < MIN_TEST_TRADES:
                        continue

                    window_num += 1
                    w_X = X_events[w_start:w_end]
                    w_dirs = event_directions[w_start:w_end]
                    w_y = y_events[w_start:w_end]

                    # Fit on window
                    w_labels, w_stats, w_good, w_km, w_pca, w_imp, w_sc, w_vc = \
                        fit_clusters(w_X, w_dirs, w_y, k, taker_cost, pca_dim=pca_dim)

                    # Predict on next step
                    t_X = X_events[test_start:test_end]
                    t_dirs = event_directions[test_start:test_end]
                    t_y = y_events[test_start:test_end]

                    t_labels = assign_clusters(t_X, w_km, w_pca, w_imp, w_sc, w_vc)
                    t_pnl = compute_cluster_pnl(t_labels, t_dirs, t_y, w_stats, maker_cost)

                    # Pooled directional accuracy of good clusters on the next
                    # window (continuation frame, cost-free). Persists if DA>0.55.
                    good_trade_rets = []
                    for c in w_good:
                        if c in t_pnl:
                            majority = w_stats[c]["majority"]
                            c_mask = t_labels == c
                            rc = (t_dirs[c_mask] * t_y[c_mask]) * majority
                            good_trade_rets.extend(rc.tolist())

                    n_good_test = len(good_trade_rets)
                    if n_good_test > 0:
                        da_next = round(float((np.array(good_trade_rets) > 0).mean()), 4)
                    else:
                        da_next = 0.0

                    rolling_rows.append(dict(
                        window=window_num,
                        train_start_evt=w_start,
                        train_end_evt=w_end,
                        test_start_evt=test_start,
                        test_end_evt=test_end,
                        n_train=w_end - w_start,
                        n_test=test_end - test_start,
                        n_good_clusters=len(w_good),
                        n_good_test_trades=n_good_test,
                        da_next=da_next,
                        profitable=da_next > 0.55,
                    ))

                    status = "+" if da_next > 0.55 else "-"
                    print(f"      Window {window_num}: train [{w_start}-{w_end}], "
                          f"test [{test_start}-{test_end}], "
                          f"good_cl={len(w_good)}, "
                          f"test_trades={n_good_test}, "
                          f"DA_next={da_next:.3f} [{status}]")

                df_rolling = pd.DataFrame(rolling_rows)
                if df_rolling.empty:
                    print("\n      Rolling test: no evaluable windows "
                          f"(n_events={n_events}, window={window_events}, "
                          f"step={step_events}). Too few events for this "
                          "config — rolling profitability not assessed.")
                    n_profitable_windows = 0
                    n_total_windows = 0
                    rolling_pct = float("nan")
                else:
                    n_profitable_windows = int(df_rolling["profitable"].sum())
                    n_total_windows = len(df_rolling)
                    rolling_pct = n_profitable_windows / n_total_windows
                    print(f"\n      Profitable windows: {n_profitable_windows}/{n_total_windows} "
                          f"({rolling_pct*100:.0f}%)")

                # ══════════════════════════════════════════════════════════
                #  TEST 4: FEATURE PROFILE STABILITY  (ALL good clusters, top-50)
                # ══════════════════════════════════════════════════════════
                print(f"\n    === Test 4: Feature profile stability "
                      f"(top-{TOP_N_FEATURES}, all good clusters) ===")

                # Track EVERY profitable full-dataset cluster, not just the
                # largest. Fallback to cluster 0 if none are flagged good.
                clusters_to_profile = full_good if full_good else [0]
                primary_cl = max(clusters_to_profile,
                                 key=lambda c: full_stats.get(c, {}).get("n", 0))

                feat_stability_rows = []   # ALL good clusters (saved to CSV)
                primary_rows = []          # primary cluster only (plot + summary)

                for target_cl in clusters_to_profile:
                    rows = _feature_signature(
                        X_events, full_labels, target_cl, feat_names,
                        n_blocks, block_size, n_events, top_n=TOP_N_FEATURES)
                    feat_stability_rows.extend(rows)
                    if target_cl == primary_cl:
                        primary_rows = rows
                    n_cons_c = sum(1 for r in rows if r["consistent_sign"])
                    print(f"      Cluster {target_cl}: {len(rows)} features tracked, "
                          f"{n_cons_c}/{max(len(rows),1)} sign-consistent")

                df_feat_stab = pd.DataFrame(feat_stability_rows)
                df_feat_stab_primary = pd.DataFrame(primary_rows)

                # Headline consistency is reported on the PRIMARY cluster so the
                # summary stays comparable to the single-cluster version; the CSV
                # holds every good cluster (with a `cluster` column).
                n_primary_feats = max(len(df_feat_stab_primary), 1)
                n_consistent = int(df_feat_stab_primary["consistent_sign"].sum()) \
                    if len(df_feat_stab_primary) else 0
                print(f"\n      [primary Cl {primary_cl}] consistent features: "
                      f"{n_consistent}/{n_primary_feats} "
                      f"({n_consistent/n_primary_feats*100:.0f}%) keep sign across blocks")

                # ══════════════════════════════════════════════════════════
                #  SAVE RESULTS
                # ══════════════════════════════════════════════════════════
                for lb in lookbacks:
                    tag = f"{asset}_{hz}_{thr_bps}bps_{cfg}_lb{lb}"

                    df_ari.to_csv(out_dir / f"block_stability_{tag}.csv", index=False)
                    df_carry.to_csv(out_dir / f"carryover_test_{tag}.csv", index=False)
                    df_rolling.to_csv(out_dir / f"rolling_window_{tag}.csv", index=False)
                    df_feat_stab.to_csv(out_dir / f"feature_stability_{tag}.csv", index=False)

                # Summary
                summary_rows = []
                summary_rows.append(dict(
                    test="Block ARI (mean)",
                    result=round(df_ari["ari_vs_full"].mean(), 3),
                    interpretation="higher = more stable (0=random, 1=identical)",
                ))
                summary_rows.append(dict(
                    test="Carry-over rate",
                    result=f"{n_carry}/{n_train_good}",
                    interpretation="train-profitable clusters still profitable in test",
                ))
                summary_rows.append(dict(
                    test="Rolling window profitability",
                    result=f"{n_profitable_windows}/{n_total_windows} ({n_profitable_windows/max(n_total_windows,1)*100:.0f}%)",
                    interpretation="% of rolling windows where good clusters are profitable OOS",
                ))
                summary_rows.append(dict(
                    test=f"Feature consistency (primary Cl {primary_cl})",
                    result=f"{n_consistent}/{n_primary_feats} ({n_consistent/n_primary_feats*100:.0f}%)",
                    interpretation="% of top features maintaining same sign across all blocks",
                ))

                df_summary = pd.DataFrame(summary_rows)
                for lb in lookbacks:
                    tag = f"{asset}_{hz}_{thr_bps}bps_{cfg}_lb{lb}"
                    df_summary.to_csv(out_dir / f"cluster_validation_summary_{tag}.csv", index=False)

                # ══════════════════════════════════════════════════════════
                #  PLOTS
                # ══════════════════════════════════════════════════════════
                print(f"\n    Generating plots...")
                plt = _setup_mpl()

                # Plot 1: Rolling window PnL
                fig, axes = plt.subplots(1, 2, figsize=(14, 6))

                ax = axes[0]
                if not df_rolling.empty and "profitable" in df_rolling.columns:
                    colors = ["#1D9E75" if p else "#D85A30" for p in df_rolling["profitable"]]
                    ax.bar(df_rolling["window"], df_rolling["da_next"],
                           color=colors, alpha=0.85, edgecolor="white")
                    ax.axhline(y=0.5, color="red", linestyle="--", linewidth=1.0)
                else:
                    ax.text(0.5, 0.5, "Rolling test not evaluable\n(too few events)",
                            ha="center", va="center", transform=ax.transAxes)
                ax.set_xlabel("Rolling window")
                ax.set_ylabel("Next-window DA (good clusters)")
                ax.set_title("Rolling window OOS directional persistence")

                # Plot 2: Feature stability heatmap (primary cluster, top-25 for legibility)
                ax = axes[1]
                if len(df_feat_stab_primary):
                    df_plot = df_feat_stab_primary.head(25)
                    z_cols = [c for c in df_plot.columns if c.startswith("block") and c.endswith("_z")]
                    z_matrix = df_plot[z_cols].values
                    feat_labels_short = [f[:35] for f in df_plot["feature"]]

                    finite = z_matrix[np.isfinite(z_matrix)]
                    vmax = max(abs(finite.min()), abs(finite.max())) if finite.size else 1.0
                    im = ax.imshow(z_matrix, cmap="RdYlGn", aspect="auto",
                                   vmin=-vmax, vmax=vmax)
                    ax.set_xticks(range(len(z_cols)))
                    ax.set_xticklabels([f"Blk {i+1}" for i in range(len(z_cols))], fontsize=9)
                    ax.set_yticks(range(len(feat_labels_short)))
                    ax.set_yticklabels(feat_labels_short, fontsize=7)
                    ax.set_title(f"Cluster {primary_cl} feature z-scores per block (top 25)")
                    plt.colorbar(im, ax=ax, shrink=0.8)
                else:
                    ax.text(0.5, 0.5, "No feature signature", ha="center",
                            va="center", transform=ax.transAxes)

                tag0 = f"{asset}_{hz}_{thr_bps}bps_{cfg}"
                fig.suptitle(f"Cluster validation — {tag0}", fontsize=13, y=1.02)
                plt.tight_layout()
                path = plot_dir / f"ws4c_validation_{tag0}.png"
                fig.savefig(path, dpi=150, bbox_inches="tight")
                plt.close()
                print(f"      Plot: {path.name}")

                # ══════════════════════════════════════════════════════════
                #  CONSOLE SUMMARY
                # ══════════════════════════════════════════════════════════
                print(f"\n    ╔{'═'*60}╗")
                print(f"    ║  VALIDATION SUMMARY — {asset.upper()} {thr_bps}bps{' ':>30} ║")
                print(f"    ╠{'═'*60}╣")
                for _, r in df_summary.iterrows():
                    print(f"    ║  {r['test']:<35} {str(r['result']):>22} ║")
                print(f"    ╚{'═'*60}╝")

                # Verdict
                mean_ari = df_ari["ari_vs_full"].mean()
                carry_pct = n_carry / max(n_train_good, 1)
                roll_pct = n_profitable_windows / max(n_total_windows, 1)
                feat_pct = n_consistent / n_primary_feats

                if mean_ari > 0.3 and carry_pct >= 0.5 and roll_pct >= 0.6 and feat_pct >= 0.6:
                    verdict = "STRONG — Clusters appear structurally stable"
                elif mean_ari > 0.15 and carry_pct >= 0.3 and roll_pct >= 0.4:
                    verdict = "MODERATE — Some stability, needs more data to confirm"
                else:
                    verdict = "WEAK — Clusters may be unstable, interpret results with caution"

                print(f"\n    VERDICT: {verdict}")

            elapsed = time.time() - t0
            print(f"\n  ━━ Done in {elapsed:.0f}s ━━")
            del X, y; gc.collect()


def main():
    p = argparse.ArgumentParser(description="WS4c: Rolling cluster validation")
    p.add_argument("--asset", choices=["btc","eth","both"], default="eth")
    p.add_argument("--hz", nargs="+", default=["15s"])
    p.add_argument("--thresholds", nargs="+", type=int, default=THRESHOLDS)
    p.add_argument("--lookbacks", nargs="+", type=int, default=LOOKBACKS)
    p.add_argument("--max-hours", type=int, default=None,
                   help="File cap like in WS4 (e.g. 2000); None=all")
    p.add_argument("--pca-dim", type=int, default=DEFAULT_PCA_DIM,
                   help="PCA dimension matching the WS4 winner config "
                        "(50/150/600 for the viable configs; default 600).")
    p.add_argument("--k", type=int, default=DEFAULT_K,
                   help="Cluster count k. Overrides the per-threshold "
                        "K_CLUSTERS dict. All viable configs use k=6 (default).")
    p.add_argument("--log-level", default="INFO")
    a = p.parse_args()
    logging.basicConfig(level=getattr(logging, a.log_level),
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S", stream=sys.stdout)
    assets = ("btc","eth") if a.asset == "both" else (a.asset,)
    run_ws4c(assets=assets, horizons=tuple(a.hz),
             thresholds=a.thresholds, lookbacks=a.lookbacks, pca_dim=a.pca_dim,
             max_hours=a.max_hours, k_override=a.k)


if __name__ == "__main__":
    main()