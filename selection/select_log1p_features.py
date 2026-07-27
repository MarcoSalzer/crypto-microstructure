#!/usr/bin/env python3
"""
select_log1p_features.py

Purpose: scans ALL numeric feature columns of the ML features and classifies
each into LOG1P_RECOMMENDED / NO / BORDERLINE based on data (not just names).

Background / criteria (all data-based, name only as a helper signal):
  log1p only makes sense if ALL three hold:
    1. non-negative (otherwise log1p -> NaN)
    2. strongly right-skewed (heavy tail; otherwise log adds nothing)
    3. quantity-/level-like (multiplicatively scaling), NOT already normalised

DISQUALIFICATION (-> NO):
  - negative values present (signed: signed_, imbalance, div, deviation, slope)
  - already normalised (name: bps, pct, _z_/z_, range_pos, _imb, ratio, frac, corr)
  - bounded to ~0..1 or ~-1..1
  - binary / very few unique values (state)
  - low skewness (symmetric enough)

IMPORTANT: This is a SUGGESTION. Check the BORDERLINE rows manually before
should be transformed. The script transforms NOTHING; it only writes a CSV.

Usage:
  python select_log1p_features.py -- <files...>
  python select_log1p_features.py --skew-min 3 -- data_storage/ml_features/ml_features_2026-02-*.parquet

Reliability: use many files over a wide time span for stable statistics.
"""

import sys
from common.paths import DATA_ROOT
import os
import glob
import argparse
import numpy as np
import pandas as pd

DEFAULT_DIR = str(DATA_ROOT / "ml_features")

# thresholds (configurable via CLI)
SKEW_LOG = 3.0          # Skewness >= -> heavy tail, log candidate
SKEW_BORDER = 1.5       # between BORDER and LOG -> borderline (BORDERLINE)
NEG_FRAC_TOL = 0.001    # allowed fraction of negative values (numerical noise)
MIN_UNIQUE = 10         # below this -> probably categorical/binary -> NO
BOUNDED_ABS_MAX = 1.5   # |values| almost all <= 1.5 -> bounded (ratio) -> NO

# Name markers that indicate already-normalised / signed features.
# Helper signal: nudges borderline cases toward NO, but does NOT override the
# hard data criteria (sign, skewness).
NORM_MARKERS = (
    "_bps", "_pct", "z_", "_z_", "range_pos", "_imb", "imbalance",
    "ratio", "frac", "_corr", "skew", "kurt", "_dev", "deviation",
    "slope", "_div_", "signed_", "_vs_", "above_", "below_", "broke_",
    "reclaim", "_pos_", "asymmetry", "pctl", "rank", "entropy", "_per_signed",
)
# Markers that positively indicate quantity/level character (helper signal for log1p)
QUANTITY_MARKERS = (
    "volume", "notional", "liq_sum", "depth_notional", "depth_sum",
    "absorb", "absorption", "refill", "count", "n_trades", "size",
    "qty", "turnover", "liq_", "depth_",
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--skew-min", type=float, default=SKEW_LOG,
                   help=f"Skewness threshold for the LOG1P_RECOMMENDED category (default {SKEW_LOG})")
    p.add_argument("--skew-border", type=float, default=SKEW_BORDER,
                   help=f"lower skewness for BORDERLINE (default {SKEW_BORDER})")
    p.add_argument("--out", default="log1p_selection.csv")
    p.add_argument("files", nargs="*")
    return p.parse_args()


def find_files(files):
    if files:
        out = []
        for a in files:
            out.extend(sorted(glob.glob(a)))
        return out
    return sorted(glob.glob(os.path.join(DEFAULT_DIR, "*.parquet")))


def has_norm_marker(name):
    return any(m in name for m in NORM_MARKERS)


def has_quantity_marker(name):
    return any(m in name for m in QUANTITY_MARKERS)


def classify(name, s, skew_min, skew_border):
    """Returns (category, reason)."""
    sv = s.dropna()
    n = len(sv)
    if n < 100:
        return "SKIP", f"too little data (n={n})"

    mn = float(sv.min())
    mx = float(sv.max())
    neg_frac = float((sv < 0).mean())
    nuniq = int(sv.nunique())
    skew = float(sv.skew())
    abs_p99 = float(sv.abs().quantile(0.99))

    # 1) negative -> NO (hard criterion)
    if neg_frac > NEG_FRAC_TOL:
        return "NO", f"negative (neg_frac={neg_frac:.3f}, min={mn:.4g}) -> log1p impossible"

    # 2) binary/categorical -> NO
    if nuniq < MIN_UNIQUE:
        return "NO", f"few unique values ({nuniq}) -> state/categorical"

    # 3) bounded to ~0..1 -> NO (ratio/share)
    if abs_p99 <= BOUNDED_ABS_MAX and mx <= 2.0:
        return "NO", f"bounded (p99={abs_p99:.3g}, max={mx:.3g}) -> ratio/normalised"

    # 4) Name suggests already normalised -> NO category (only if not clearly a quantity)
    if has_norm_marker(name) and not has_quantity_marker(name):
        return "NO", f"name marker normalised/signed (skew={skew:.2f})"

    # 5) Skewness-based classification (hard main criterion)
    if skew >= skew_min:
        tag = "LOG1P" if has_quantity_marker(name) else "LOG1P?"
        note = "quantity-like" if has_quantity_marker(name) else "no quantity marker, but heavy-tail"
        return ("LOG1P_RECOMMENDED" if tag == "LOG1P" else "BORDERLINE",
                f"non-neg, skew={skew:.2f}>= {skew_min}, {note}")

    if skew >= skew_border:
        return "BORDERLINE", f"moderate skew={skew:.2f} (between {skew_border} and {skew_min})"

    return "NO", f"skew={skew:.2f} < {skew_border} -> symmetric enough, log adds nothing"


def main():
    args = parse_args()
    files = find_files(args.files)
    if not files:
        print(f"ERROR: no files. Default dir: {DEFAULT_DIR}")
        sys.exit(1)

    print(f"Scanning {len(files)} file(s)")
    print(f"  first: {os.path.basename(files[0])}")
    print(f"  last: {os.path.basename(files[-1])}")

    # Schema from the first readable file
    schema = None
    for f in files:
        try:
            schema = pd.read_parquet(f).columns
            break
        except Exception as e:
            print(f"  WARN {f}: {e}")
    if schema is None:
        print("ERROR: no readable file.")
        sys.exit(1)

    # Numeric columns only; roughly exclude targets/meta
    EXCLUDE_PREFIX = ("ret_", "mfe_fwd_", "mae_fwd_", "rv_fwd_", "tbl_", "barrier_",
                      "label_", "data_", "health_", "usability_", "session_",
                      "us_holiday", "us_rth", "l2_coverage", "lob50_health",
                      "trades_coverage")
    probe = pd.read_parquet(files[0])
    num_cols = [
        c for c in probe.columns
        if pd.api.types.is_numeric_dtype(probe[c])
        and not any(c.startswith(px) for px in EXCLUDE_PREFIX)
    ]
    del probe
    print(f"Numeric feature columns (excluding targets/meta): {len(num_cols)}")

    # Collect data across all files (feature columns only)
    frames = []
    for f in files:
        try:
            frames.append(pd.read_parquet(f, columns=num_cols))
        except Exception as e:
            print(f"  WARN {f}: {e}")
    df = pd.concat(frames, ignore_index=True)
    print(f"Combined rows: {len(df):,}")
    print("=" * 90)

    rows = []
    for c in num_cols:
        cat, reason = classify(c, df[c], args.skew_min, args.skew_border)
        sv = df[c].dropna()
        rows.append({
            "column": c,
            "category": cat,
            "min": float(sv.min()) if len(sv) else np.nan,
            "max": float(sv.max()) if len(sv) else np.nan,
            "mean": float(sv.mean()) if len(sv) else np.nan,
            "skew": float(sv.skew()) if len(sv) else np.nan,
            "neg_frac": float((sv < 0).mean()) if len(sv) else np.nan,
            "n_unique": int(sv.nunique()) if len(sv) else 0,
            "reason": reason,
        })

    res = pd.DataFrame(rows)
    order = {"LOG1P_RECOMMENDED": 0, "BORDERLINE": 1, "NO": 2, "SKIP": 3}
    res["_o"] = res["category"].map(order).fillna(9)
    res = res.sort_values(["_o", "skew"], ascending=[True, False]).drop(columns="_o")
    res.to_csv(args.out, index=False)

    print("\nDISTRIBUTION:")
    for cat in ["LOG1P_RECOMMENDED", "BORDERLINE", "NO", "SKIP"]:
        print(f"  {cat:18s}: {(res['category']==cat).sum()}")

    print("\n=== LOG1P_RECOMMENDED ===")
    for _, r in res[res.category == "LOG1P_RECOMMENDED"].iterrows():
        print(f"  {r['column']:42s} skew={r['skew']:8.2f} min={r['min']:.4g} max={r['max']:.4g}")

    print("\n=== BORDERLINE (please check manually) ===")
    for _, r in res[res.category == "BORDERLINE"].iterrows():
        print(f"  {r['column']:42s} skew={r['skew']:8.2f} min={r['min']:.4g} max={r['max']:.4g}  | {r['reason']}")

    print(f"\nFull classification in: {args.out}")
    print("NEXT STEP: check LOG1P_RECOMMENDED + the desired borderline cases,")
    print("then use this final column list for the transform script.")


if __name__ == "__main__":
    main()