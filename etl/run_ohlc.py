#!/usr/bin/env python3
# ==============================================================================
# OHLC Stage Runner — Discovers complete S0 days and generates OHLC parquets.
#
# PURPOSE:
#   Wraps generate_ohlc.py for pipeline orchestration. Scans the S0 feature
#   directory for dates where all 24 hourly files are present, then generates
#   OHLC parquets for those dates. Incomplete days (e.g. the current streaming
#   day) are silently skipped — this is the "complete-day guard".
#
# COMPLETE-DAY GUARD:
#   Daily OHLC requires max(mid) and min(mid) over all 24 hours. A partially
#   collected day would produce a false high/low (artificially compressed range)
#   that would corrupt range-position features for every second of that day.
#   The guard ensures only finished days enter the OHLC store.
#
# USAGE:
#   python -m etl.run_ohlc
#   python -m etl.run_ohlc --asset btc
#   python -m etl.run_ohlc --date 2026-03-10
#   python -m etl.run_ohlc --skip-existing   (default behaviour)
#   python -m etl.run_ohlc --dry-run
# ==============================================================================

from __future__ import annotations
from common.paths import DATA_ROOT

import argparse
import glob
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
_SCRIPT_DIR   = Path(__file__).resolve().parent          # etl/
_OHLC_GEN     = _SCRIPT_DIR / "ohlc" / "generate_ohlc.py"

_DEFAULT_S0_DIR  = DATA_ROOT / "s0_features"
_DEFAULT_OUT_DIR = DATA_ROOT / "ohlc"


# =============================================================================
# Discovery
# =============================================================================

def discover_complete_days(
    s0_dir: str,
    asset_filter: str | None,
    date_filter: str | None,
) -> list[tuple[str, str]]:
    """
    Scan s0_dir for (asset, date) pairs where all 24 hourly S0 files exist.

    Returns list of (asset, date_str) tuples, sorted chronologically.
    """
    pattern = os.path.join(s0_dir, "s0_features_*.parquet")
    all_files = sorted(glob.glob(pattern))

    # Group files by (asset, date): s0_features_{asset}_{date}_{hh}.parquet
    groups: dict[tuple[str, str], set[int]] = defaultdict(set)
    for fpath in all_files:
        stem = Path(fpath).stem          # s0_features_btc_2026-03-10_14
        parts = stem.split("_")
        # Expected: ['s0', 'features', asset, date, hh]
        if len(parts) < 5:
            continue
        asset   = parts[2]                     # btc / eth
        date_str = parts[3]                    # 2026-03-10
        try:
            hh = int(parts[4])                 # 14
        except ValueError:
            continue

        if asset_filter and asset != asset_filter.lower():
            continue
        if date_filter and date_str != date_filter:
            continue

        groups[(asset, date_str)].add(hh)

    complete = [
        (asset, date_str)
        for (asset, date_str), hours in sorted(groups.items())
        if len(hours) == 24 and set(hours) == set(range(24))
    ]
    return complete


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "OHLC stage runner: discover complete S0 days and generate OHLC parquets."
        )
    )
    ap.add_argument("--s0-dir",  type=str, default=str(_DEFAULT_S0_DIR))
    ap.add_argument("--out-dir", type=str, default=str(_DEFAULT_OUT_DIR))
    ap.add_argument("--asset",   type=str, default=None, choices=["btc", "eth", "bnb"],
                    help="Filter to single asset (default: both).")
    ap.add_argument("--date",    type=str, default=None, metavar="YYYY-MM-DD",
                    help="Filter to single date (default: all complete days).")
    ap.add_argument("--skip-existing", action="store_true", default=True,
                    help="Skip dates whose OHLC output already exists (default: True).")
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print what would be done but do not write any files.")
    ap.add_argument("--quiet", "-q", action="store_true")

    args = ap.parse_args()
    verbose = not args.quiet

    t0 = time.time()

    if verbose:
        print(f"[run_ohlc] Scanning S0 dir: {args.s0_dir}")

    complete_days = discover_complete_days(args.s0_dir, args.asset, args.date)

    if verbose:
        print(f"[run_ohlc] Complete days found: {len(complete_days)}")

    if not complete_days:
        print("[run_ohlc] No complete days found — nothing to do.")
        sys.exit(0)

    # Filter out existing outputs if skip_existing
    to_process = []
    skipped    = 0
    out_dir    = Path(args.out_dir)

    for asset, date_str in complete_days:
        out_path = out_dir / f"ohlc_{asset}_{date_str}.parquet"
        if args.skip_existing and out_path.exists():
            skipped += 1
            continue
        to_process.append((asset, date_str))

    if verbose:
        print(f"[run_ohlc] To process: {len(to_process)}  |  Skipped (exists): {skipped}")

    if args.dry_run:
        for asset, date_str in to_process:
            print(f"  [DRY-RUN] Would generate: ohlc_{asset}_{date_str}.parquet")
        print("[run_ohlc] Dry run complete — no files written.")
        sys.exit(0)

    # Import and call generator directly (same process — avoids subprocess overhead)
    try:
        from etl.ohlc.generate_ohlc import generate_ohlc_for_day
    except ImportError:
        # Fallback: direct file import for standalone usage
        import importlib.util
        spec = importlib.util.spec_from_file_location("generate_ohlc", str(_OHLC_GEN))
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        generate_ohlc_for_day = mod.generate_ohlc_for_day

    errors = 0
    for i, (asset, date_str) in enumerate(to_process, 1):
        if verbose:
            print(f"[run_ohlc] [{i}/{len(to_process)}] {asset} {date_str}")
        try:
            generate_ohlc_for_day(
                s0_dir=args.s0_dir,
                out_dir=args.out_dir,
                asset=asset,
                date_str=date_str,
                require_complete=True,
                skip_existing=args.skip_existing,
                verbose=verbose,
            )
        except Exception as e:
            print(f"[run_ohlc] ERROR: {asset} {date_str}: {e}")
            errors += 1

    elapsed = time.time() - t0
    status  = "DONE" if errors == 0 else f"DONE WITH {errors} ERROR(S)"
    print(f"[run_ohlc] {status} — {len(to_process)} processed in {elapsed:.1f}s")
    sys.exit(0 if errors == 0 else 1)


if __name__ == "__main__":
    main()