# Minimal calibration: breakout events per (asset, window=horizon, threshold).
# Loads ONLY the target column per horizon, no feature matrix.
from __future__ import annotations
import numpy as np, pandas as pd
import pyarrow.parquet as pq
from common.data_loader import discover_files
from common.config import target_col, ML_FEATURES

WINDOWS        = ["1s", "5s", "15s", "30s"]
THRESHOLDS_BPS = [10, 15, 20, 30, 40]

def load_target_bps(asset: str, hz: str) -> np.ndarray:
    col = target_col(hz, asset)
    files = discover_files(None, None, None)
    if not files:
        raise FileNotFoundError(f"No ml_features files in {ML_FEATURES}")
    parts = []
    for f in files:
        if col in set(pq.read_schema(f).names):
            parts.append(pq.read_table(f, columns=[col]).column(0).to_numpy(zero_copy_only=False))
    y = np.concatenate(parts)
    y = y[np.isfinite(y)]                 # Discard NaN targets (forward move undefined at the sample end)
    if np.nanmean(np.abs(y)) < 0.01:      # Units guard: fraction -> bps
        y = y * 10_000
    return y

def main():
    rows = []
    for asset in ("btc", "eth"):
        for hz in WINDOWS:
            y = load_target_bps(asset, hz)
            n_valid = len(y)
            for thr in THRESHOLDS_BPS:
                n_up, n_down = int((y > thr).sum()), int((y < -thr).sum())
                rows.append({"asset": asset, "window": hz, "threshold_bps": thr,
                             "n_up": n_up, "n_down": n_down, "n_any": n_up + n_down,
                             "n_valid": n_valid,
                             "rate_any": round((n_up + n_down) / n_valid, 6)})
            print(f"{asset} {hz}: n_valid={n_valid:,}")
    pd.DataFrame(rows).to_csv("breakout_counts.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))

if __name__ == "__main__":
    main()