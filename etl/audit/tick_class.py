# -----------------------------------------------------------------------------
# etl/audit/tick_class.py
# Tick-size classification of the four raw LOB streams (BTC/ETH x spot/fut),
# following Briola et al. (2024): it infers the exchange tick from the
# best-bid/ask price grid and measures the spread in ticks.
#
# Produces the Section 3.1.1 in-text numbers (share of observations at a
# 1-tick spread, median spread in ticks and bps, and the Large-Tick verdict) --
# e.g. "the spread equals a single tick in 99.9% ... all four streams are
# therefore Large-Tick assets ... median spread 0.001 / 0.014 / 0.048 bps".
# NOTE: this is the Section 3.1.1 tick classification, printed to the console --
# NOT Table 5 (breakout counts, produced by prediction/breakout_counts.py).
#
# EXTERNAL DATA (standalone QA tool): reads the external, uncommitted raw LOB
#   parquet store (data_archive/*/raw_data/lobdeep_*.parquet), resolved via
#   common.paths.DATA_ROOT (env THESIS_DATA_ROOT or configs/paths.yaml). It does
#   NOT run inside the repo without that store, and is not wired into etl.run_all.
# START:  python -m etl.audit.tick_class
# -----------------------------------------------------------------------------
from __future__ import annotations

import glob

import numpy as np
import pandas as pd

from common.paths import DATA_ROOT

# data_archive/<date-range>/raw_data/lobdeep_<stream>_*.parquet
BASE = str(DATA_ROOT / "data_archive") + "/*/raw_data"
MAXF = 300
rng  = np.random.default_rng(42)


def infer_tick(p):
    u = np.unique(np.round(p[np.isfinite(p)], 8))
    d = np.diff(u); d = d[d > 1e-12]
    if d.size == 0:
        return float("nan")
    small = d[d <= np.percentile(d, 5)]
    return float(np.median(small)) if small.size else float(np.min(d))


def main() -> None:
    rows = []
    for stream in ("btc_spot", "btc_fut", "eth_spot", "eth_fut"):
        files = sorted(glob.glob(f"{BASE}/lobdeep_{stream}_*.parquet"))
        if len(files) > MAXF:
            idx = rng.choice(len(files), MAXF, replace=False)
            files = [files[i] for i in sorted(idx)]
        b, a = [], []
        for f in files:
            df = pd.read_parquet(f, columns=["best_bid", "best_ask"])
            b.append(df.best_bid.to_numpy(float)); a.append(df.best_ask.to_numpy(float))
        bid, ask = np.concatenate(b), np.concatenate(a)
        ok = np.isfinite(bid) & np.isfinite(ask) & (ask > bid)
        bid, ask = bid[ok], ask[ok]

        tick   = infer_tick(np.concatenate([bid, ask]))
        spread = ask - bid
        st     = np.round(spread / tick).astype(int)
        mid    = (ask + bid) / 2.0
        bps    = spread / mid * 1e4

        s1 = float(np.mean(st == 1))
        cls = ("Large-Tick (tick-constrained)" if s1 >= 0.90
               else "boundary of Large-Tick behaviour" if s1 >= 0.50
               else "Small-Tick")
        print(f"\n=== {stream.upper()} ===  ({len(files)} hours, {bid.size:,} observations)")
        print(f"  tick size            : {tick:.8g}")
        print(f"  spread == 1 tick     : {s1*100:6.2f} %")
        print(f"  spread <= 2 ticks    : {float(np.mean(st<=2))*100:6.2f} %")
        print(f"  median spread        : {np.median(st):.0f} ticks / {np.median(bps):.3f} bps")
        print(f"  mean   spread        : {np.mean(spread/tick):.3f} ticks / {np.mean(bps):.3f} bps")
        print(f"  -> {cls}")
        rows.append((stream, tick, s1, np.median(bps), cls))

    print("\n\n=== SUMMARY ===")
    for s, t, s1, mb, c in rows:
        print(f"  {s:9s} tick={t:<10g} 1-tick: {s1*100:5.1f} %  median {mb:5.3f} bps  -> {c}")


if __name__ == "__main__":
    main()
