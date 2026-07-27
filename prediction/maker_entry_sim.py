#!/usr/bin/env python3
# prediction/maker_entry_sim.py
# ==============================================================================
# WS6 — Maker-Entry Cost-Reduction Simulation
# ==============================================================================
#
# MOTIVATION:
#   All previous tests turned the ENTRY-EDGE knob (cluster, features, exit) —
#   which is ~+3 bps and unstable. This test turns the COST knob, which
#   is much larger (~10 bps taker round-trip):
#
#   On cluster activation NO immediate taker entry is made; instead
#   a maker limit order is placed X bps BELOW the signal price (for long).
#     - price dips → fill X bps cheaper + maker fee (effectively +10-15 bps)
#     - price runs away → no fill, no loss
#
#   NOTE ADVERSE SELECTION: a limit order below the price fills
#   preferentially when the price moves AGAINST the position. Which trades
#   get filled and whether the filled ones run worse than the unfilled ones
#   is an EMPIRICAL question — exactly what this test measures.
#
# TRAIN-ONLY / CAUSAL: the fill decision uses only the path AFTER the signal (the
#   1s return path tells exactly whether the price ever fell X bps). No
#   forward information, no train/test split needed (pure simulation on
#   already-selected trades or all breakouts).
#
# COMPARISON:
#   (A) cluster-filtered trades (from cluster_trades_{tag}.csv)
#   (B) ALL breakouts (fresh from y_1s)
#   → shows whether the cluster filter adds anything beyond the maker lever.
#
# PARAMETER MATRIX (all from ONE path pass):
#   Offsets:    0 / -2.5 / -5 / -7.5 / -10 bps
#   wait time:  30 / 60 / 120 / unbounded(=lookahead) s
#   Exit:       Taker-Exit vs Maker-Exit
#
# USAGE:
#   python maker_entry_sim.py --asset btc --hz 15s --thresholds 15 \
#       --cluster-tag kmeans_pca25 --max-hours 2000
# ==============================================================================

from __future__ import annotations
import argparse, glob, logging, sys
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MFE_LOOKAHEAD = 300
OFFSETS_BPS   = [0.0, 2.5, 5.0, 7.5, 10.0]   # how far BELOW the signal price (magnitude)
WAIT_TIMES_S  = [30, 60, 120, MFE_LOOKAHEAD] # last = "unbounded" (reference)
TP_FRAC       = 0.60                          # fixed TP at 60% of the MFE-from-fill
MIN_HOLD_S    = 3


def _ts(): return datetime.now().strftime("%H:%M:%S")
def tprint(m=""): print(f"{_ts()}  {m}", flush=True)


def find_breakouts(y_1s, threshold_bps, window_s):
    """All breakouts: |cumulative return over window_s| >= threshold, non-overlapping."""
    n = len(y_1s)
    # Vectorised rolling sum. Was a Python loop over n rows
    # (np.sum per i) -> O(n*window) and hangs at ~7M rows (2000h).
    # sum(y[i:i+window_s]) == csum[i+window_s] - csum[i].
    csum = np.concatenate(([0.0], np.cumsum(y_1s, dtype=np.float64)))
    cum = ((csum[window_s:] - csum[:-window_s]) * 10_000.0)[:n - window_s]
    events, dirs, last = [], [], -10**9
    for i in range(len(cum)):
        if abs(cum[i]) >= threshold_bps and i - last >= window_s:
            events.append(i); dirs.append(1 if cum[i] > 0 else -1); last = i
    return np.array(events, int), np.array(dirs, int)


def extract_paths(y_1s, event_indices, directions, max_lookahead=MFE_LOOKAHEAD):
    """Direction-adjusted cumulative return paths in bps (positive = favourable)."""
    n = len(event_indices)
    paths = np.full((n, max_lookahead), np.nan, dtype=np.float32)
    for i in range(n):
        idx, d = int(event_indices[i]), float(directions[i])
        end = min(idx + max_lookahead + 1, len(y_1s))
        if end <= idx + 1:
            continue
        cum = np.cumsum(y_1s[idx+1:end]) * d * 10_000
        paths[i, :len(cum)] = cum
    return paths


def simulate_maker_entry(paths, offset_bps, wait_s, exit_mode,
                          taker_cost, maker_cost, tp_bps=20.0, sl_bps=15.0,
                          min_hold=MIN_HOLD_S, lookahead=MFE_LOOKAHEAD):
    """
    Simulates a maker limit entry at -offset_bps for a set of paths.

    Fill logic (conservative): a limit buy at -offset fills as soon as the
    direction-adjusted path reaches cum[t] <= -offset (the price first moves
    AGAINST the position) within [0, wait_s). offset=0 → immediate fill at t=0.

    Exit: FIXED absolute TP/SL (tp_bps / sl_bps), NO look-ahead onto the
    own MFE. The first touch decides; otherwise timeout at the path end.
      - 'taker': cost = maker(entry) + taker(exit)
      - 'maker': TP as a limit → maker(exit) only on a TP hit; SL/timeout taker.
    """
    n = len(paths)
    filled_pnl, unfilled_fav = [], []
    n_filled = 0
    eff_wait = min(wait_s, lookahead)
    # config gives ROUND-TRIP cost -> one side = /2.
    taker_ow, maker_ow = taker_cost / 2.0, maker_cost / 2.0
    # Entry: offset==0 = immediate market entry = TAKER; offset>0 with limit fill = MAKER.
    entry_cost = taker_ow if offset_bps == 0.0 else maker_ow

    for i in range(n):
        path = paths[i]
        valid = path[~np.isnan(path)]
        if len(valid) < min_hold + 1:
            continue

        # ── Fill determination ───────────────────────────────────────────────
        if offset_bps == 0.0:
            fill_t = 0
            entry_adj = 0.0           # fill at the signal price
        else:
            # first second in [0, eff_wait) with cum <= -offset
            window = valid[:eff_wait]
            hit = np.where(window <= -offset_bps)[0]
            if len(hit) == 0:
                # unfilled — hypothetical favourable path (for adverse selection)
                unfilled_fav.append(float(np.max(valid)))
                continue
            fill_t = int(hit[0])
            entry_adj = offset_bps    # entry offset bps better → +offset on all subsequent returns

        # ── Path from fill (relative to the fill price) ─────────────────────────
        post = valid[fill_t+1:]
        if len(post) < min_hold:
            # too soon after fill — flat exit at the fill price, cost as taker
            n_filled += 1
            filled_pnl.append(entry_adj - entry_cost - taker_ow)
            continue
        rel = post + entry_adj        # returns relative to the (better) fill price, in bps

        # ── Exit: FIXED absolute TP/SL (NO look-ahead onto the own MFE!) ──
        # Sequential pass: first touch of TP (win) or SL (loss)
        # decides. If neither is touched, exit at the path end (timeout).
        exit_kind = "timeout"
        realized = float(rel[-1])
        for k in range(min_hold, len(rel)):
            v = rel[k-1]
            if v >= tp_bps:
                realized = float(tp_bps); exit_kind = "tp"; break
            if v <= -sl_bps:
                realized = float(-sl_bps); exit_kind = "sl"; break

        if exit_mode == "taker":
            exit_cost = taker_ow
        else:  # maker-exit: TP as a limit -> maker; SL/timeout as market -> taker
            exit_cost = maker_ow if exit_kind == "tp" else taker_ow
        cost = entry_cost + exit_cost

        n_filled += 1
        filled_pnl.append(realized - cost)

    fill_rate = n_filled / n if n else 0.0
    fp = np.array(filled_pnl) if filled_pnl else np.array([0.0])
    return dict(
        offset_bps=offset_bps, wait_s=wait_s if wait_s < lookahead else -1,
        exit_mode=exit_mode,
        n_total=n, n_filled=n_filled, fill_rate=round(fill_rate, 4),
        pnl_mean=round(float(fp.mean()), 3),
        pnl_median=round(float(np.median(fp)), 3),
        win_rate=round(float((fp > 0).mean()), 4),
        pnl_x_fill=round(float(fp.mean() * fill_rate), 4),  # expected value per signal
        mean_fav_unfilled=round(float(np.mean(unfilled_fav)), 3) if unfilled_fav else None,
    )


def run_arm(label, paths, taker, maker, out_rows, asset, hz, thr, source_tag,
            tp_bps=20.0, sl_bps=15.0):
    """One parameter matrix over a path set (cluster-filtered OR all)."""
    tprint(f"  ── Arm: {label} ({len(paths)} Trades) ──")
    best = None
    for off in OFFSETS_BPS:
        for wait in WAIT_TIMES_S:
            for mode in ("taker", "maker"):
                r = simulate_maker_entry(paths, off, wait, mode, taker, maker,
                                         tp_bps=tp_bps, sl_bps=sl_bps)
                r.update(dict(arm=label, asset=asset, hz=hz, thr=thr, tag=source_tag))
                out_rows.append(r)
                if best is None or r["pnl_x_fill"] > best["pnl_x_fill"]:
                    best = r
    if best:
        w = "unbounded" if best["wait_s"] == -1 else f"{best['wait_s']}s"
        tprint(f"    best: offset=-{best['offset_bps']}bps wait={w} exit={best['exit_mode']} "
               f"→ PnL={best['pnl_mean']:+.2f} bps × fill={best['fill_rate']:.1%} "
               f"= {best['pnl_x_fill']:+.3f} bps/Signal (WR={best['win_rate']:.1%})")
    return best


def run_ws6(assets=("btc",), horizons=("15s",), thresholds=(15,),
            cluster_tag="kmeans_pca150_k6", lookback=1, window_map=None,
            max_hours=None, results_subdir="cluster_mfe",
            tp_bps=20.0, sl_bps=15.0):
    from common.data_loader import load_dataset
    from common.config import RESULTS_DIR, SPREAD_BPS, MAKER_COST_BPS

    ws4_dir = RESULTS_DIR / results_subdir
    out_dir = RESULTS_DIR / "maker_entry"; out_dir.mkdir(parents=True, exist_ok=True)
    window_map = window_map or {"5s": 5, "15s": 15}
    tprint(f"Cluster source: {ws4_dir}")

    for asset in assets:
        taker = SPREAD_BPS.get(asset, {}).get("fut", 9.0)
        maker = MAKER_COST_BPS.get(asset, {}).get("fut", 4.0)
        tprint(f"━━ WS6 {asset.upper()}  taker={taker} maker={maker} bps ━━")
        for hz in horizons:
            win = window_map.get(hz, 15)
            try:
                _, y_1s, _, _ = load_dataset(target="1s", asset=asset,
                                             target_only=True, max_hours=max_hours)
                y_1s = y_1s.astype(np.float32)
            except Exception as e:
                logger.error("1s load fail: %s", e); continue

            for thr in thresholds:
                tag = f"{cluster_tag}_{asset}_{hz}_{thr}bps_lb{lookback}"
                rows = []

                # ── Arm B: ALL breakouts (fresh) ───────────────────────
                ev_all, dir_all = find_breakouts(y_1s, thr, win)
                tprint(f"  {hz}/{thr}bps: {len(ev_all)} breakouts total")
                paths_all = extract_paths(y_1s, ev_all, dir_all)
                best_all = run_arm("all_breakouts", paths_all, taker, maker,
                                   rows, asset, hz, thr, tag, tp_bps, sl_bps)

                # ── Arm A: cluster-filtered trades ─────────────────────
                tpath = ws4_dir / f"cluster_trades_{tag}.csv"
                if tpath.exists():
                    dft = pd.read_csv(tpath)
                    ev_f = dft["event_index"].values.astype(int)
                    dir_f = dft["direction"].values.astype(float)
                    if len(ev_f) > len(ev_all):
                        tprint(f"  WARNING: filtered ({len(ev_f)}) > all_breakouts "
                               f"({len(ev_all)}) — universes inconsistent. Probably "
                               f"max-hours mismatch versus cluster_trades OR differing "
                               f"breakout definition. Results of this arm are unreliable.")
                    if ev_f.max() >= len(y_1s):
                        tprint(f"  event_index max ({ev_f.max()}) >= y_1s len "
                               f"({len(y_1s)}) — indices do not match this load.")
                    paths_f = extract_paths(y_1s, ev_f, dir_f)
                    # DECISIVE alignment check: ws4 has mfe_bps correct
                    # (its own aligned load). If our event_index matches
                    # THIS y_1s, the MFE recomputed from paths_f must match
                    # dft['mfe_bps']. If not -> the indices point
                    # to wrong rows and all path figures (incl. ws3) are garbage.
                    if "mfe_bps" in dft.columns:
                        rec = np.nanmax(paths_f, axis=1)
                        sto = dft["mfe_bps"].to_numpy(float)
                        msk = np.isfinite(rec) & np.isfinite(sto)
                        corr = float(np.corrcoef(rec[msk], sto[msk])[0, 1]) if msk.sum() > 2 else float("nan")
                        mad  = float(np.nanmean(np.abs(rec[msk] - sto[msk]))) if msk.any() else float("nan")
                        ok = corr > 0.95
                        tprint(f"  ► ALIGNMENT-CHECK (recomputed MFE vs cluster_trades.mfe_bps): "
                               f"corr={corr:.3f}  mean|Δ|={mad:.2f} bps  → "
                               f"{'OK — event_index matches this load' if ok else 'MISALIGNED — event_index does NOT match, path numbers invalid'}")
                    best_flt = run_arm("cluster_filtered", paths_f, taker, maker,
                                       rows, asset, hz, thr, tag, tp_bps, sl_bps)
                else:
                    tprint(f"  cluster_trades not found: {tpath.name} — all-arm only")
                    best_flt = None

                # ── Comparison ────────────────────────────────────────────
                if best_all and best_flt:
                    delta = best_flt["pnl_x_fill"] - best_all["pnl_x_fill"]
                    tprint(f"  ► cluster contribution (best filtered − best all): "
                           f"{delta:+.3f} bps/Signal")
                    verdict = ("MAKER+CLUSTER profitable" if best_flt["pnl_x_fill"] > 0
                               and delta > 0.5 else
                               "MAKER alone suffices" if best_all["pnl_x_fill"] > 0
                               else "no profitable maker lever")
                    tprint(f"  → {verdict}")

                pd.DataFrame(rows).to_csv(out_dir / f"maker_entry_{tag}.csv", index=False)
                tprint(f"  saved: maker_entry_{tag}.csv ({len(rows)} combinations)")
            del y_1s


def main():
    p = argparse.ArgumentParser(description="WS6 Maker-Entry cost-reduction simulation")
    p.add_argument("--asset", nargs="+", default=["btc"])
    p.add_argument("--hz", nargs="+", default=["15s"])
    p.add_argument("--thresholds", nargs="+", type=int, default=[15])
    p.add_argument("--cluster-tag", default="kmeans_pca150_k6",
                   help="Cluster-tag prefix of the cluster_trades CSV (e.g. kmeans_pca150_k6)")
    p.add_argument("--lookback", type=int, default=1)
    p.add_argument("--tp-bps", type=float, default=20.0,
                   help="Fixed take-profit in bps (no look-ahead)")
    p.add_argument("--sl-bps", type=float, default=15.0,
                   help="Fixed stop-loss in bps")
    p.add_argument("--results-subdir", default="cluster_mfe",
                   help="Subfolder in RESULTS_DIR with the cluster_trades CSVs. "
                        "For structure clusters (with edge): cluster_mfe_structure_v1")
    p.add_argument("--max-hours", type=int, default=None)
    p.add_argument("--log-level", default="INFO")
    a = p.parse_args()
    logging.basicConfig(level=getattr(logging, a.log_level),
        format="%(asctime)s  %(levelname)s  %(message)s", datefmt="%H:%M:%S")
    run_ws6(assets=tuple(a.asset), horizons=tuple(a.hz),
            thresholds=tuple(a.thresholds), cluster_tag=a.cluster_tag,
            lookback=a.lookback, max_hours=a.max_hours,
            results_subdir=a.results_subdir, tp_bps=a.tp_bps, sl_bps=a.sl_bps)


if __name__ == "__main__":
    main()