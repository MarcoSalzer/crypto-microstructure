#!/usr/bin/env python3
# ==============================================================================
# Run Script — S0 Feature Engine
#
# Orchestrates the S0 feature pipeline for one or more asset-hours:
#   1. Scans raw_data/ and s0_context/ to discover ready jobs
#   2. Validates that all required input files exist
#   3. Runs the S0 feature engine (raw data + context -> features)
#   4. Archives consumed raw data + context files
#   5. Prints a structured execution summary
#
# USAGE:
#   # Auto-discover: process everything that's ready
#   python -m etl.run_s0_features
#
#   # Filter by asset:
#   python -m etl.run_s0_features --asset btc
#
#   # Filter by date:
#   python -m etl.run_s0_features --date 2026-02-16
#
#   # Filter by asset + date + hour:
#   python -m etl.run_s0_features --asset btc --date 2026-02-16 --hour 3
#
#   # Dry run (validate inputs, show plan, don't execute):
#   python -m etl.run_s0_features --dry-run
#
#   # Skip archiving:
#   python -m etl.run_s0_features --no-archive
# ==============================================================================

from __future__ import annotations
from common.paths import DATA_ROOT

import argparse
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Path setup — resolve btc project root so imports work regardless of cwd
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent          # etl/

# The package is imported directly (repo root is on sys.path when run as a module).

from etl.engine.s0_feature_engine import (
    build_s0_features_for_hour,
    _paths_for_hour,
)

# ==============================================================================
# Constants
# ==============================================================================

_DEFAULT_RAW_DIR = DATA_ROOT / "raw_data"
_DEFAULT_CTX_DIR = DATA_ROOT / "s0_context"
_DEFAULT_OUT_DIR = DATA_ROOT / "s0_features"
_DEFAULT_ARCHIVE_DIR = DATA_ROOT / "data_archive"

ASSETS = ["btc", "eth", "bnb"]
HOURS = list(range(24))


# ==============================================================================
# Display helpers
# ==============================================================================

_BLUE = "\033[94m"
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_RED = "\033[91m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_RESET = "\033[0m"


def _ts() -> str:
    """Compact UTC timestamp for log lines."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def _header(title: str) -> None:
    width = 72
    print()
    print(f"{_BOLD}{'=' * width}{_RESET}")
    print(f"{_BOLD}  {title}{_RESET}")
    print(f"{_BOLD}{'=' * width}{_RESET}")


def _section(title: str) -> None:
    print(f"\n{_BLUE}{_BOLD}▸ {title}{_RESET}")


def _info(msg: str) -> None:
    print(f"  {_DIM}[{_ts()}]{_RESET} {msg}")


def _ok(msg: str) -> None:
    print(f"  {_GREEN}{msg}{_RESET}")


def _warn(msg: str) -> None:
    print(f"  {_YELLOW}{msg}{_RESET}")


def _fail(msg: str) -> None:
    print(f"  {_RED}{msg}{_RESET}")


def _kv(key: str, value: str, indent: int = 4) -> None:
    pad = " " * indent
    print(f"{pad}{_DIM}{key}:{_RESET} {value}")


# ==============================================================================
# Job definition
# ==============================================================================

@dataclass
class Job:
    asset: str
    date_str: str
    hour: int
    raw_dir: str
    ctx_dir: str
    out_dir: str
    archive_dir: Optional[str]


@dataclass
class JobResult:
    job: Job
    success: bool
    rows: int = 0
    cols: int = 0
    size_mb: float = 0.0
    elapsed_s: float = 0.0
    error: str = ""


# ==============================================================================
# Auto-discovery
# ==============================================================================

# Pattern: trades_btc_spot_2026-02-16_03.parquet  ->  (btc, 2026-02-16, 03)
_RAW_PATTERN = re.compile(
    r"^(?:trades|lobdeep)_(\w+)_(?:spot|fut)_(\d{4}-\d{2}-\d{2})_(\d{2})\.parquet$"
)

# Pattern: s0_context_btc_2026-02-16_03.parquet  ->  (btc, 2026-02-16, 03)
_CTX_PATTERN = re.compile(
    r"^s0_context_(\w+)_(\d{4}-\d{2}-\d{2})_(\d{2})\.parquet$"
)


def _discover_all_jobs(
    raw_dir: str,
    ctx_dir: str,
    out_dir: str,
    archive_dir: Optional[str],
    asset_filter: Optional[str] = None,
    date_filter: Optional[str] = None,
    hour_filter: Optional[int] = None,
) -> List[Job]:
    """
    Scan raw_data/ and s0_context/ to find all asset-date-hour combinations
    where all 5 input files exist (4 raw + 1 context).

    Returns a sorted list of Jobs ready to execute.
    """
    raw_path = Path(raw_dir)
    ctx_path = Path(ctx_dir)

    if not raw_path.exists():
        return []
    if not ctx_path.exists():
        return []

    # Step 1: Parse raw filenames to find (asset, date, hour) tuples and
    #         count how many of the 4 raw files exist per tuple.
    raw_counts: Dict[Tuple[str, str, int], int] = {}
    for f in raw_path.iterdir():
        if not f.is_file():
            continue
        m = _RAW_PATTERN.match(f.name)
        if not m:
            continue
        asset, date_str, hour_str = m.group(1), m.group(2), int(m.group(3))
        key = (asset, date_str, hour_str)
        raw_counts[key] = raw_counts.get(key, 0) + 1

    # Step 2: Parse context filenames
    ctx_available: Set[Tuple[str, str, int]] = set()
    for f in ctx_path.iterdir():
        if not f.is_file():
            continue
        m = _CTX_PATTERN.match(f.name)
        if not m:
            continue
        asset, date_str, hour_str = m.group(1), m.group(2), int(m.group(3))
        ctx_available.add((asset, date_str, hour_str))

    # Step 3: Build jobs where all 4 raw files + 1 context exist
    jobs: List[Job] = []
    for key, count in sorted(raw_counts.items()):
        asset, date_str, hour = key

        # Need exactly 4 raw files (trades_spot, trades_fut, lobdeep_spot, lobdeep_fut)
        if count < 4:
            continue

        # Need matching context
        if key not in ctx_available:
            continue

        # Apply filters
        if asset_filter and asset != asset_filter:
            continue
        if date_filter and date_str != date_filter:
            continue
        if hour_filter is not None and hour != hour_filter:
            continue

        jobs.append(Job(
            asset=asset,
            date_str=date_str,
            hour=hour,
            raw_dir=raw_dir,
            ctx_dir=ctx_dir,
            out_dir=out_dir,
            archive_dir=archive_dir,
        ))

    return jobs


# ==============================================================================
# Input validation
# ==============================================================================

def _check_inputs(job: Job) -> List[str]:
    """Return list of missing input files for a job."""
    raw_paths, ctx_path, _ = _paths_for_hour(
        job.raw_dir, job.ctx_dir, job.out_dir,
        job.asset, job.date_str, job.hour,
    )
    missing = []
    for p in raw_paths.all_paths():
        if not Path(p).exists():
            missing.append(p)
    if not ctx_path.exists():
        missing.append(str(ctx_path))
    return missing


def _output_exists(job: Job) -> bool:
    _, _, out_path = _paths_for_hour(
        job.raw_dir, job.ctx_dir, job.out_dir,
        job.asset, job.date_str, job.hour,
    )
    return out_path.exists()


# ==============================================================================
# Execution
# ==============================================================================

def _run_job(job: Job, verbose: bool = True) -> JobResult:
    """Run a single asset-hour through the S0 feature engine."""
    t0 = time.time()
    try:
        df = build_s0_features_for_hour(
            raw_dir=job.raw_dir,
            ctx_dir=job.ctx_dir,
            out_dir=job.out_dir,
            asset=job.asset,
            date_str=job.date_str,
            hour=job.hour,
            archive_dir=job.archive_dir,
            verbose=verbose,
        )
        _, _, out_path = _paths_for_hour(
            job.raw_dir, job.ctx_dir, job.out_dir,
            job.asset, job.date_str, job.hour,
        )
        size_mb = out_path.stat().st_size / (1024 * 1024) if out_path.exists() else 0.0
        return JobResult(
            job=job,
            success=True,
            rows=len(df),
            cols=len(df.columns),
            size_mb=size_mb,
            elapsed_s=time.time() - t0,
        )
    except Exception as e:
        return JobResult(
            job=job,
            success=False,
            elapsed_s=time.time() - t0,
            error=str(e),
        )


# ==============================================================================
# Summary
# ==============================================================================

def _print_summary(results: List[JobResult], total_elapsed: float) -> None:
    _header("Execution Summary")

    succeeded = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    _kv("Total jobs", str(len(results)), indent=2)
    _kv("Succeeded", f"{_GREEN}{len(succeeded)}{_RESET}", indent=2)
    if failed:
        _kv("Failed", f"{_RED}{len(failed)}{_RESET}", indent=2)
    _kv("Total time", f"{total_elapsed:.1f}s", indent=2)

    if succeeded:
        _section("Completed")
        for r in succeeded:
            j = r.job
            print(
                f"    {j.asset.upper()} {j.date_str} H{j.hour:02d}"
                f"  │  {r.rows:,} rows × {r.cols} cols"
                f"  │  {r.size_mb:.2f} MB"
                f"  │  {r.elapsed_s:.1f}s"
            )

    if failed:
        _section("Failed")
        for r in failed:
            j = r.job
            print(
                f"    {j.asset.upper()} {j.date_str} H{j.hour:02d}"
                f"  │  {r.error}"
            )

    print()


# ==============================================================================
# Main
# ==============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Run the S0 feature engine.\n\n"
            "Without arguments: auto-discovers all ready asset-date-hour\n"
            "combinations by scanning raw_data/ and s0_context/."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  %(prog)s                                    # process everything ready\n"
            "  %(prog)s --asset btc                        # only BTC, all dates/hours\n"
            "  %(prog)s --date 2026-02-16                  # only this date, both assets\n"
            "  %(prog)s --asset btc --date 2026-02-16      # BTC on this date\n"
            "  %(prog)s --asset btc --date 2026-02-16 --hour 3\n"
            "  %(prog)s --dry-run                          # show plan, don't execute\n"
        ),
    )

    # All optional — defaults to auto-discovery
    ap.add_argument("--asset", type=str, default=None, choices=["btc", "eth", "bnb"],
                     help="Filter by asset. Default: all assets.")
    ap.add_argument("--date", type=str, default=None,
                     help="Filter by date (YYYY-MM-DD). Default: all dates.")
    ap.add_argument("--hour", type=int, default=None,
                     help="Filter by hour (0-23). Default: all hours.")

    ap.add_argument("--raw-dir", type=str, default=str(_DEFAULT_RAW_DIR))
    ap.add_argument("--ctx-dir", type=str, default=str(_DEFAULT_CTX_DIR))
    ap.add_argument("--out-dir", type=str, default=str(_DEFAULT_OUT_DIR))
    ap.add_argument("--archive-dir", type=str, default=str(_DEFAULT_ARCHIVE_DIR))
    ap.add_argument("--no-archive", action="store_true",
                     help="Skip archiving consumed files.")

    ap.add_argument("--dry-run", action="store_true",
                     help="Validate inputs and show execution plan without running.")
    ap.add_argument("--skip-existing", action="store_true",
                     help="Skip jobs where the output file already exists.")
    ap.add_argument("--quiet", "-q", action="store_true",
                     help="Suppress engine-internal logs (summary still prints).")

    args = ap.parse_args()

    # -----------------------------------------------------------------
    # Validate optional date format
    # -----------------------------------------------------------------
    if args.date and not re.match(r"^\d{4}-\d{2}-\d{2}$", args.date):
        print(f"{_RED}Error:{_RESET} --date must be YYYY-MM-DD format, got: {args.date}")
        sys.exit(1)

    if args.hour is not None and (args.hour < 0 or args.hour > 23):
        print(f"{_RED}Error:{_RESET} --hour must be 0-23, got: {args.hour}")
        sys.exit(1)

    archive_dir = None if args.no_archive else args.archive_dir

    # -----------------------------------------------------------------
    # Describe mode
    # -----------------------------------------------------------------
    filter_parts = []
    if args.asset:
        filter_parts.append(f"asset={args.asset.upper()}")
    if args.date:
        filter_parts.append(f"date={args.date}")
    if args.hour is not None:
        filter_parts.append(f"hour={args.hour:02d}")
    mode_label = ", ".join(filter_parts) if filter_parts else "all assets, all dates, all hours"

    _header("S0 Feature Pipeline")

    _section("Configuration")
    _kv("Mode", f"Auto-discovery ({mode_label})")
    _kv("Raw dir", args.raw_dir)
    _kv("Context dir", args.ctx_dir)
    _kv("Output dir", args.out_dir)
    _kv("Archive dir", archive_dir or "(disabled)")

    # -----------------------------------------------------------------
    # Discover jobs
    # -----------------------------------------------------------------
    _section("Scanning for ready jobs")

    jobs = _discover_all_jobs(
        raw_dir=args.raw_dir,
        ctx_dir=args.ctx_dir,
        out_dir=args.out_dir,
        archive_dir=archive_dir,
        asset_filter=args.asset,
        date_filter=args.date,
        hour_filter=args.hour,
    )

    if not jobs:
        _warn("No complete input sets found (need 4 raw + 1 context per asset-hour).")
        _info(f"Scanned: {args.raw_dir}")
        _info(f"         {args.ctx_dir}")
        sys.exit(0)

    # Group for display
    by_asset_date: Dict[Tuple[str, str], List[int]] = {}
    for j in jobs:
        key = (j.asset, j.date_str)
        by_asset_date.setdefault(key, []).append(j.hour)

    for (asset, date_str), hours in sorted(by_asset_date.items()):
        _ok(f"{asset.upper()} {date_str}: {len(hours)} hours "
            f"[{', '.join(f'{h:02d}' for h in sorted(hours))}]")

    _info(f"Total: {len(jobs)} jobs discovered")

    # -----------------------------------------------------------------
    # Filter out existing outputs
    # -----------------------------------------------------------------
    valid_jobs: List[Job] = []
    skipped = 0

    if args.skip_existing:
        _section("Checking existing outputs")

    for job in jobs:
        label = f"{job.asset.upper()} {job.date_str} H{job.hour:02d}"

        if args.skip_existing and _output_exists(job):
            _warn(f"{label} — output already exists, skipping")
            skipped += 1
            continue

        valid_jobs.append(job)

    if args.skip_existing and skipped:
        _info(f"Skipped {skipped} existing, {len(valid_jobs)} remaining")

    if not valid_jobs:
        _warn("All outputs already exist. Nothing to do.")
        sys.exit(0)

    # -----------------------------------------------------------------
    # Validate inputs (double-check file existence)
    # -----------------------------------------------------------------
    _section(f"Input Validation ({len(valid_jobs)} jobs)")
    ready_jobs: List[Job] = []

    for job in valid_jobs:
        label = f"{job.asset.upper()} {job.date_str} H{job.hour:02d}"
        missing = _check_inputs(job)
        if missing:
            _fail(f"{label} — missing {len(missing)} file(s):")
            for m in missing:
                print(f"        {_DIM}{m}{_RESET}")
        else:
            _ok(f"{label} — all inputs present")
            ready_jobs.append(job)

    if not ready_jobs:
        _warn("No valid jobs after input check. Exiting.")
        sys.exit(1)

    # -----------------------------------------------------------------
    # Dry run: stop here
    # -----------------------------------------------------------------
    if args.dry_run:
        _section("Dry Run — Execution Plan")
        for job in ready_jobs:
            _, ctx_path, out_path = _paths_for_hour(
                job.raw_dir, job.ctx_dir, job.out_dir,
                job.asset, job.date_str, job.hour,
            )
            print(f"    {job.asset.upper()} {job.date_str} H{job.hour:02d}:")
            _kv("Context", str(ctx_path), indent=6)
            _kv("Output", str(out_path), indent=6)
            _kv("Archive", archive_dir or "(disabled)", indent=6)
        print(f"\n  {_DIM}Re-run without --dry-run to execute.{_RESET}\n")
        return

    # -----------------------------------------------------------------
    # Execute
    # -----------------------------------------------------------------
    _section(f"Execution ({len(ready_jobs)} jobs)")
    results: List[JobResult] = []
    total_t0 = time.time()

    for i, job in enumerate(ready_jobs, 1):
        label = f"{job.asset.upper()} {job.date_str} H{job.hour:02d}"
        _info(f"[{i}/{len(ready_jobs)}] {_BOLD}{label}{_RESET}")

        result = _run_job(job, verbose=not args.quiet)
        results.append(result)

        if result.success:
            _ok(f"{label} — {result.rows:,} rows, {result.size_mb:.2f} MB, {result.elapsed_s:.1f}s")
        else:
            _fail(f"{label} — {result.error}")

    total_elapsed = time.time() - total_t0

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------
    _print_summary(results, total_elapsed)

    # Exit code: 0 if all succeeded, 1 if any failed
    if any(not r.success for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()