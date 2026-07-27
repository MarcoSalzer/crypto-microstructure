#!/usr/bin/env python3
# ==============================================================================
# Batch Runner for S0 Context Build
#
# PURPOSE:
#   Discover available raw data hours, run the S0 context builder for each
#   (asset, date, hour) combination, and produce a terminal summary report.
#
# ARCHITECTURE CONTEXT:
#   Pipeline: Binance-only, multi-asset (BTC + ETH).
#   Input file naming convention:
#     trades_{asset}_spot_YYYY-MM-DD_HH.parquet   (e.g. trades_btc_spot_2026-02-16_14.parquet)
#     trades_{asset}_fut_YYYY-MM-DD_HH.parquet
#     lobdeep_{asset}_spot_YYYY-MM-DD_HH.parquet
#     lobdeep_{asset}_fut_YYYY-MM-DD_HH.parquet
#
#   Output file naming convention:
#     s0_context_{asset}_YYYY-MM-DD_HH.parquet
#
# DISCOVERY LOGIC:
#   Scans the data directory for trades_{asset}_spot_*.parquet files.
#   Each match yields a (asset, date, hour) tuple. Then verifies that all
#   required companion files exist before running the context build.
#
# ASSET FILTERING:
#   --asset btc    -> only process BTC hours
#   --asset eth    -> only process ETH hours
#   --asset all    -> process all discovered assets (default)
#
#               KEY_RE captures asset from filename. HourKey includes asset.
#               Removed lob20 from required files. Removed --allow-missing-lob-deep.
#               Added --asset filter. Module path: etl.engine.s0_context_batch.
#               Summary cleaned of lob20 references.
# ==============================================================================

from __future__ import annotations
from common.paths import DATA_ROOT

import argparse
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

# Known assets in the pipeline (extend when adding new assets)
KNOWN_ASSETS = ["btc", "eth", "bnb"]

# ==============================================================================
# Auto-detect project paths (same pattern as collector.py)
# ==============================================================================
# this file is etl/<name>.py; imports resolve from the repo root
_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_RAW_DIR = DATA_ROOT / "raw_data"  # data_storage/raw_data
_DEFAULT_OUT_DIR = DATA_ROOT / "s0_context" # data_storage/s0_context

# Regex to discover available (asset, date, hour) combinations from trade files.
# Matches: trades_btc_spot_2026-02-16_14.parquet
#          trades_eth_spot_2026-02-16_00.parquet
KEY_RE = re.compile(r"^trades_([a-z]+)_spot_(\d{4}-\d{2}-\d{2})_(\d{2})\.parquet$")


@dataclass(frozen=True)
class HourKey:
    """Unique key for one context build unit: (asset, date, hour)."""
    asset: str
    date: str
    hour: int

    @property
    def hour_str(self) -> str:
        return f"{self.hour:02d}"

    @property
    def suffix(self) -> str:
        """Date-hour portion of filename: YYYY-MM-DD_HH.parquet"""
        return f"{self.date}_{self.hour_str}.parquet"


def discover_hours(data_dir: Path, asset_filter: str = "all") -> list[HourKey]:
    """
    Scan data_dir for trades_{asset}_spot_*.parquet and extract (asset, date, hour) keys.

    Args:
        data_dir:      Directory containing raw Parquet files
        asset_filter:  "btc", "eth", "bnb", or "all" (default)

    Returns:
        Sorted list of HourKey objects.
    """
    keys: list[HourKey] = []
    for p in data_dir.iterdir():
        m = KEY_RE.match(p.name)
        if not m:
            continue
        asset = m.group(1).lower()
        if asset_filter != "all" and asset != asset_filter.lower():
            continue
        keys.append(HourKey(asset=asset, date=m.group(2), hour=int(m.group(3))))

    keys.sort(key=lambda k: (k.asset, k.date, k.hour))
    return keys


def required_raw_files_exist(data_dir: Path, key: HourKey) -> tuple[bool, list[str]]:
    """
    Check that all required raw files exist for this (asset, date, hour).

    Required files (Binance-only pipeline, no lob20):
      - trades_{asset}_spot_{suffix}
      - trades_{asset}_fut_{suffix}
      - lobdeep_{asset}_spot_{suffix}
      - lobdeep_{asset}_fut_{suffix}
    """
    a = key.asset
    s = key.suffix
    needed = [
        f"trades_{a}_spot_{s}",
        f"trades_{a}_fut_{s}",
        f"lobdeep_{a}_spot_{s}",
        f"lobdeep_{a}_fut_{s}",
    ]
    missing = [n for n in needed if not (data_dir / n).exists()]
    return (len(missing) == 0, missing)


def summarize_context(context_path: Path) -> dict:
    """Read a completed context file and compute summary statistics."""
    df = pq.read_table(str(context_path)).to_pandas()

    out = {
        "file": context_path.name,
        "rows": int(len(df)),
        "t_min": str(df["bucket_dt_utc"].min()) if "bucket_dt_utc" in df.columns and len(df) else None,
        "t_max": str(df["bucket_dt_utc"].max()) if "bucket_dt_utc" in df.columns and len(df) else None,
    }

    def frac(col: str) -> float | None:
        if col not in df.columns or len(df) == 0:
            return None
        s = pd.to_numeric(df[col], errors="coerce").fillna(0)
        return float((s != 0).mean())

    out["health_ratio"] = frac("data_health_flag")
    out["usability_ratio"] = frac("data_usability_flag")
    out["l2_coverage_ratio"] = frac("l2_coverage_flag")
    out["trades_coverage_ratio"] = frac("trades_coverage_flag")
    out["lob50_health_ratio"] = frac("lob50_health_flag")

    # Depth stats (lobdeep only; lob20 removed from pipeline)
    for c in ["depth_availability", "depth_lobdeep_global"]:
        if c in df.columns and len(df):
            s = pd.to_numeric(df[c], errors="coerce")
            out[f"{c}_mean"] = float(s.mean())
            out[f"{c}_min"] = float(s.min())
            out[f"{c}_max"] = float(s.max())
        else:
            out[f"{c}_mean"] = None
            out[f"{c}_min"] = None
            out[f"{c}_max"] = None

    return out


def _print_summary(rep: pd.DataFrame, top_n: int = 10, show_all: bool = False) -> None:
    """Print terminal summary of context build results."""
    if rep.empty:
        print("[CTX] SUMMARY: no rows")
        return

    rep = rep.copy()

    # Derive usable minutes estimate (works for 1s bucket grids)
    if "usability_ratio" in rep.columns and "rows" in rep.columns:
        rep["usable_seconds_est"] = rep["rows"] * rep["usability_ratio"]
        rep["usable_minutes_est"] = rep["usable_seconds_est"] / 60.0
    else:
        rep["usable_minutes_est"] = 0.0

    cols = [
        "file", "rows",
        "usability_ratio", "health_ratio",
        "l2_coverage_ratio", "trades_coverage_ratio",
        "usable_minutes_est",
    ]
    cols = [c for c in cols if c in rep.columns]

    pd.set_option("display.max_columns", 200)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 120)

    print("\n" + "=" * 90)
    print("[CTX] SUMMARY (key metrics)")
    print("=" * 90)

    worst = rep.sort_values("usability_ratio", ascending=True).head(top_n)
    best = rep.sort_values("usability_ratio", ascending=False).head(top_n)

    print(f"\nWorst {top_n} by usability_ratio:")
    print(worst[cols].to_string(index=False))

    print(f"\nBest {top_n} by usability_ratio:")
    print(best[cols].to_string(index=False))

    total_usable = float(rep["usable_minutes_est"].sum())
    total_rows = int(rep["rows"].sum())
    mean_use = float(rep["usability_ratio"].mean())
    mean_health = float(rep["health_ratio"].mean()) if "health_ratio" in rep.columns else float("nan")

    print("\nTotals:")
    print(f"  hours:               {len(rep)}")
    print(f"  total_rows:          {total_rows}")
    print(f"  total_usable_minutes:{total_usable:.2f}")
    print(f"  mean_usability:      {mean_use:.4f}")
    print(f"  mean_health:         {mean_health:.4f}")

    # Per-asset breakdown if multiple assets
    if "file" in rep.columns:
        assets_seen = set()
        for f in rep["file"]:
            # Extract asset from filename: s0_context_btc_... or s0_context_eth_...
            parts = str(f).replace("s0_context_", "").split("_")
            if parts:
                assets_seen.add(parts[0])
        if len(assets_seen) > 1:
            print("\n  Per-asset breakdown:")
            for asset in sorted(assets_seen):
                mask = rep["file"].str.contains(f"s0_context_{asset}_")
                subset = rep[mask]
                if not subset.empty:
                    a_use = float(subset["usability_ratio"].mean())
                    a_health = float(subset["health_ratio"].mean()) if "health_ratio" in subset.columns else float("nan")
                    a_hours = len(subset)
                    print(f"    {asset.upper()}: hours={a_hours} mean_usability={a_use:.4f} mean_health={a_health:.4f}")

    if show_all:
        print("\nAll hours (sorted by file):")
        rep2 = rep.sort_values("file")[cols]
        print(rep2.to_string(index=False))

    print("=" * 90 + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch-build S0 context files + terminal summary.")
    ap.add_argument("--data-dir", type=str, default=str(_DEFAULT_RAW_DIR),
                    help=f"Raw parquet folder (default: {_DEFAULT_RAW_DIR}).")
    ap.add_argument("--out-dir", type=str, default=str(_DEFAULT_OUT_DIR),
                    help=f"Context output folder (default: {_DEFAULT_OUT_DIR}).")
    ap.add_argument("--module", type=str,
                    default="etl.engine.s0_context_batch",
                    help="Python module path for the context builder CLI.")
    ap.add_argument("--asset", type=str, default="all",
                    choices=KNOWN_ASSETS + ["all"],
                    help="Asset filter: btc, eth, or all (default: all).")
    ap.add_argument("--resample", type=str, default="1s")
    ap.add_argument("--usability-window", type=int, default=60)
    ap.add_argument("--usability-min-ratio", type=float, default=0.95)
    ap.add_argument("--usability-max-bad-streak", type=int, default=5)
    ap.add_argument("--soft-missing-budget", type=int, default=1,
                    help="B1a soft health: max missing L2 combos (default: 1 out of 2).")
    ap.add_argument("--dry-run", action="store_true")

    # Reporting behavior
    ap.add_argument("--top", type=int, default=10,
                    help="How many best/worst rows to print.")
    ap.add_argument("--show-all", action="store_true",
                    help="Print the full per-hour table.")
    ap.add_argument("--write-report", action="store_true",
                    help="Write s0_context_report.csv to out_dir.")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    keys = discover_hours(data_dir, asset_filter=args.asset)
    if not keys:
        asset_hint = f" for asset={args.asset}" if args.asset != "all" else ""
        raise SystemExit(
            f"No hours discovered in {data_dir}{asset_hint}. "
            f"Expected files like trades_btc_spot_YYYY-MM-DD_HH.parquet."
        )

    print(f"[CTX] Discovered {len(keys)} hour-keys (asset_filter={args.asset})")

    ran = 0
    skipped = 0
    failures = 0
    report_rows: list[dict] = []

    for k in keys:
        ok, missing = required_raw_files_exist(data_dir, k)
        if not ok:
            skipped += 1
            print(f"[CTX] SKIP {k.asset} {k.date} {k.hour_str}: missing {missing}")
            continue

        out_path = out_dir / f"s0_context_{k.asset}_{k.suffix}"
        if out_path.exists():
            print(f"[CTX] EXISTS {out_path.name} (skip)")
            report_rows.append(summarize_context(out_path))
            continue

        cmd = [
            "python", "-m", args.module,
            "--l0-dir", str(data_dir),
            "--date", k.date,
            "--hour", str(k.hour),
            "--asset", k.asset,
            "--output-dir", str(out_dir),
            "--resample", args.resample,
            "--usability-window", str(args.usability_window),
            "--usability-min-ratio", str(args.usability_min_ratio),
            "--usability-max-bad-streak", str(args.usability_max_bad_streak),
            "--soft-missing-budget", str(args.soft_missing_budget),
        ]

        print(f"[CTX] RUN {k.asset} {k.date} {k.hour_str} -> {out_path.name}")
        if args.dry_run:
            print("      ", " ".join(cmd))
            continue

        try:
            subprocess.run(cmd, check=True)
            ran += 1
            if out_path.exists():
                report_rows.append(summarize_context(out_path))
            else:
                failures += 1
                print(f"[CTX] FAIL {k.asset} {k.date} {k.hour_str}: output not found: {out_path}")
        except subprocess.CalledProcessError as e:
            failures += 1
            print(f"[CTX] FAIL {k.asset} {k.date} {k.hour_str}: {e}")

    # Terminal summary
    if report_rows:
        rep = pd.DataFrame(report_rows).sort_values(["file"])
        _print_summary(rep, top_n=int(args.top), show_all=bool(args.show_all))

        if args.write_report:
            rep_path = out_dir / "s0_context_report.csv"
            rep.to_csv(rep_path, index=False)
            print(f"[CTX] REPORT -> {rep_path}")

    print(f"[CTX] DONE ran={ran} skipped={skipped} failures={failures}")


if __name__ == "__main__":
    main()