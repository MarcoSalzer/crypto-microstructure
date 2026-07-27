#!/usr/bin/env python3
# ==============================================================================
# Run Script — S2 Feature Engine
#
# Orchestrates the S2 feature pipeline for one or more asset-hours:
#   1. Scans s1_features/ to discover ready jobs
#   2. Validates that all required input files exist
#   3. Runs the S2 feature engine (S1 features -> S2 features)
#   4. Prints a structured execution summary
#
# The S2 output retains all previous-stage columns (S0–S1) plus new S2
# columns. No archiving of intermediate feature files — raw data is archived
# once during S0 creation, which is sufficient to recreate all stages.
#
# USAGE:
#   python -m etl.run_s2_features
#   python -m etl.run_s2_features --asset btc
#   python -m etl.run_s2_features --date 2026-02-16
#   python -m etl.run_s2_features --asset btc --date 2026-02-16 --hour 3
#   python -m etl.run_s2_features --dry-run
# ==============================================================================

from __future__ import annotations
from common.paths import DATA_ROOT

import argparse
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent


from etl.engine.s2_feature_engine import (
    build_s2_features_for_hour,
    _paths_for_hour,
)

# ==============================================================================
# Constants
# ==============================================================================

_DEFAULT_INPUT_DIR = DATA_ROOT / "s1_features"
_DEFAULT_OUT_DIR   = DATA_ROOT / "s2_features"

ASSETS = ["btc", "eth", "bnb"]

# ==============================================================================
# Display helpers
# ==============================================================================

_BLUE   = "\033[94m"
_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_RED    = "\033[91m"
_BOLD   = "\033[1m"
_DIM    = "\033[2m"
_RESET  = "\033[0m"


def _ts() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def _header(title: str) -> None:
    w = 72
    print()
    print(f"{_BOLD}{'=' * w}{_RESET}")
    print(f"{_BOLD}  {title}{_RESET}")
    print(f"{_BOLD}{'=' * w}{_RESET}")


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
    input_dir: str
    out_dir: str


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

# Pattern: s1_features_btc_2026-02-16_03.parquet
_INPUT_PATTERN = re.compile(
    r"^s1_features_(\w+)_(\d{4}-\d{2}-\d{2})_(\d{2})\.parquet$"
)


def _discover_jobs(
    input_dir: str,
    out_dir: str,
    asset_filter: Optional[str] = None,
    date_filter: Optional[str] = None,
    hour_filter: Optional[int] = None,
) -> List[Job]:
    """Scan input directory for available S1 feature files."""
    in_path = Path(input_dir)
    if not in_path.exists():
        return []

    jobs: List[Job] = []
    for f in sorted(in_path.iterdir()):
        if not f.is_file():
            continue
        m = _INPUT_PATTERN.match(f.name)
        if not m:
            continue
        asset, date_str, hour_str = m.group(1), m.group(2), int(m.group(3))

        if asset_filter and asset != asset_filter:
            continue
        if date_filter and date_str != date_filter:
            continue
        if hour_filter is not None and int(hour_str) != hour_filter:
            continue

        jobs.append(Job(
            asset=asset,
            date_str=date_str,
            hour=int(hour_str),
            input_dir=input_dir,
            out_dir=out_dir,
        ))

    return jobs


# ==============================================================================
# Input validation
# ==============================================================================

def _check_inputs(job: Job) -> List[str]:
    in_path, _ = _paths_for_hour(job.input_dir, job.out_dir, job.asset, job.date_str, job.hour)
    if not in_path.exists():
        return [str(in_path)]
    return []


def _output_exists(job: Job) -> bool:
    _, out_path = _paths_for_hour(job.input_dir, job.out_dir, job.asset, job.date_str, job.hour)
    return out_path.exists()


# ==============================================================================
# Execution
# ==============================================================================

def _run_job(job: Job, verbose: bool = True) -> JobResult:
    t0 = time.time()
    try:
        df = build_s2_features_for_hour(
            s1_dir=job.input_dir,
            out_dir=job.out_dir,
            asset=job.asset,
            date_str=job.date_str,
            hour=job.hour,
            archive_dir=None,
            verbose=verbose,
        )
        _, out_path = _paths_for_hour(job.input_dir, job.out_dir, job.asset, job.date_str, job.hour)
        size_mb = out_path.stat().st_size / (1024 * 1024) if out_path.exists() else 0.0
        return JobResult(
            job=job, success=True,
            rows=len(df), cols=len(df.columns),
            size_mb=size_mb, elapsed_s=time.time() - t0,
        )
    except Exception as e:
        return JobResult(job=job, success=False, elapsed_s=time.time() - t0, error=str(e))


# ==============================================================================
# Summary
# ==============================================================================

def _print_summary(results: List[JobResult], total_elapsed: float) -> None:
    _header("Execution Summary")
    succeeded = [r for r in results if r.success]
    failed    = [r for r in results if not r.success]

    _kv("Total jobs", str(len(results)), indent=2)
    _kv("Succeeded",  f"{_GREEN}{len(succeeded)}{_RESET}", indent=2)
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
            print(f"    {j.asset.upper()} {j.date_str} H{j.hour:02d}  │  {r.error}")

    print()


# ==============================================================================
# Main
# ==============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Run the S2 feature engine.\n\n"
            "Without arguments: auto-discovers all ready asset-date-hour\n"
            "combinations by scanning s1_features/."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  %(prog)s                                    # process everything ready\n"
            "  %(prog)s --asset btc                        # only BTC\n"
            "  %(prog)s --date 2026-02-16                  # only this date\n"
            "  %(prog)s --asset btc --date 2026-02-16 --hour 3\n"
            "  %(prog)s --dry-run                          # show plan, don't execute\n"
        ),
    )

    ap.add_argument("--asset", type=str, default=None, choices=["btc", "eth", "bnb"])
    ap.add_argument("--date",  type=str, default=None, help="YYYY-MM-DD")
    ap.add_argument("--hour",  type=int, default=None, help="0-23")

    ap.add_argument("--input-dir", type=str, default=str(_DEFAULT_INPUT_DIR))
    ap.add_argument("--out-dir",   type=str, default=str(_DEFAULT_OUT_DIR))

    ap.add_argument("--dry-run",       action="store_true")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--quiet", "-q",   action="store_true")

    args = ap.parse_args()

    if args.date and not re.match(r"^\d{4}-\d{2}-\d{2}$", args.date):
        print(f"{_RED}Error:{_RESET} --date must be YYYY-MM-DD, got: {args.date}")
        sys.exit(1)
    if args.hour is not None and (args.hour < 0 or args.hour > 23):
        print(f"{_RED}Error:{_RESET} --hour must be 0-23, got: {args.hour}")
        sys.exit(1)

    filter_parts = []
    if args.asset: filter_parts.append(f"asset={args.asset.upper()}")
    if args.date:  filter_parts.append(f"date={args.date}")
    if args.hour is not None: filter_parts.append(f"hour={args.hour:02d}")
    mode_label = ", ".join(filter_parts) if filter_parts else "all assets, all dates, all hours"

    _header("S2 Feature Pipeline")

    _section("Configuration")
    _kv("Mode",       f"Auto-discovery ({mode_label})")
    _kv("Input dir",  args.input_dir)
    _kv("Output dir", args.out_dir)

    # -- Discover --
    _section("Scanning for ready jobs")
    jobs = _discover_jobs(
        input_dir=args.input_dir, out_dir=args.out_dir,
        asset_filter=args.asset, date_filter=args.date, hour_filter=args.hour,
    )

    if not jobs:
        _warn("No S1 feature files found.")
        _info(f"Scanned: {args.input_dir}")
        sys.exit(0)

    by_asset_date: Dict[Tuple[str, str], List[int]] = {}
    for j in jobs:
        by_asset_date.setdefault((j.asset, j.date_str), []).append(j.hour)
    for (asset, date_str), hours in sorted(by_asset_date.items()):
        _ok(f"{asset.upper()} {date_str}: {len(hours)} hours "
            f"[{', '.join(f'{h:02d}' for h in sorted(hours))}]")
    _info(f"Total: {len(jobs)} jobs discovered")

    # -- Skip existing --
    valid_jobs: List[Job] = []
    skipped = 0
    if args.skip_existing:
        _section("Checking existing outputs")
    for job in jobs:
        if args.skip_existing and _output_exists(job):
            _warn(f"{job.asset.upper()} {job.date_str} H{job.hour:02d} — output exists, skipping")
            skipped += 1
            continue
        valid_jobs.append(job)
    if args.skip_existing and skipped:
        _info(f"Skipped {skipped} existing, {len(valid_jobs)} remaining")
    if not valid_jobs:
        _warn("All outputs already exist. Nothing to do.")
        sys.exit(0)

    # -- Validate --
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

    # -- Dry run --
    if args.dry_run:
        _section("Dry Run — Execution Plan")
        for job in ready_jobs:
            in_path, out_path = _paths_for_hour(job.input_dir, job.out_dir, job.asset, job.date_str, job.hour)
            print(f"    {job.asset.upper()} {job.date_str} H{job.hour:02d}:")
            _kv("Input",  str(in_path),  indent=6)
            _kv("Output", str(out_path), indent=6)
        print(f"\n  {_DIM}Re-run without --dry-run to execute.{_RESET}\n")
        return

    # -- Execute --
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
    _print_summary(results, total_elapsed)

    if any(not r.success for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()