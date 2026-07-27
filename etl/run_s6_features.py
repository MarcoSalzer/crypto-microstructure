#!/usr/bin/env python3
# ==============================================================================
# Run Script — S6 Cross-Asset Feature Engine  (BTC ↔ ETH ↔ BNB)
#
# Orchestrates the S6 cross-asset feature pipeline for one or more date-hours:
#   1. Scans s5_features/ to discover jobs with ≥2 assets (same date/hour)
#   2. Validates that at least BTC+ETH S5 input files exist before queuing
#   3. Runs the S6 feature engine  (S5_btc + S5_eth [+ S5_bnb] → S6)
#   4. Prints a structured execution summary
#
# Unlike S1–S5 (per-asset), S6 is cross-asset: each job consumes 2–3 asset
# S5 files and produces a single output file. The output filename reflects
# the actually available assets: btceth or btcethbnb.
# Jobs are keyed by (date, hour) — not (asset, date, hour).
#
# STORAGE LAYOUT  (default base-dir = data_storage):
#   Input  BTC: data_storage/s5_features/s5_features_btc_{date}_{hour:02d}.parquet
#   Input  ETH: data_storage/s5_features/s5_features_eth_{date}_{hour:02d}.parquet
#   Input  BNB: data_storage/s5_features/s5_features_bnb_{date}_{hour:02d}.parquet  (optional)
#   Output (always):      data_storage/s6_features_btceth/s6_features_btceth_{date}_{hour:02d}.parquet
#   Output (BNB present): data_storage/s6_features_btcethbnb/s6_features_btcethbnb_{date}_{hour:02d}.parquet
#
# USAGE:
#   python -m etl.run_s6_features
#   python -m etl.run_s6_features --date 2026-02-16
#   python -m etl.run_s6_features --date 2026-02-16 --hour 3
#   python -m etl.run_s6_features --dry-run
#   python -m etl.run_s6_features --skip-existing
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
_SCRIPT_DIR   = Path(__file__).resolve().parent


from etl.engine.s6_feature_engine import S6FeatureEngine

# ==============================================================================
# Constants
# ==============================================================================

_DEFAULT_BASE_DIR  = DATA_ROOT
_DEFAULT_S5_DIR    = _DEFAULT_BASE_DIR / "s5_features"
_DEFAULT_S6_DIR_BTCETH    = _DEFAULT_BASE_DIR / "s6_features_btceth"
_DEFAULT_S6_DIR_BTCETHBNB = _DEFAULT_BASE_DIR / "s6_features_btcethbnb"

_REQUIRED_ASSETS = {"btc", "eth"}           # minimum for a valid S6 job
_ALL_ASSETS      = ("btc", "eth", "bnb")    # full asset set

# Pattern for S5 input files (used in discovery)
_S5_PAT = re.compile(
    r"^s5_features_(btc|eth|bnb)_(\d{4}-\d{2}-\d{2})_(\d{2})\.parquet$"
)

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
    """A single S6 computation unit: one date/hour pair → one cross-asset output."""
    date_str: str
    hour: int
    base_dir: str
    assets: List[str]   # sorted list of available assets, e.g. ["btc","eth"] or ["bnb","btc","eth"]


@dataclass
class JobResult:
    job: Job
    success: bool
    outputs: List[Path] = None     # paths actually written by the engine
    rows: int = 0                  # rows of the first (btceth) output
    cols: int = 0                  # cols of the first (btceth) output
    size_mb: float = 0.0           # total size across all outputs
    elapsed_s: float = 0.0
    error: str = ""

    def __post_init__(self):
        if self.outputs is None:
            self.outputs = []


# ==============================================================================
# Path helpers
# ==============================================================================

def _s5_path(base_dir: str, asset: str, date_str: str, hour: int) -> Path:
    return (
        Path(base_dir)
        / "s5_features"
        / f"s5_features_{asset}_{date_str}_{hour:02d}.parquet"
    )


def _s6_path(base_dir: str, date_str: str, hour: int, assets: List[str]) -> Path:
    """Mirrors the engine's output path: per-asset-set subdirectory."""
    tag = "".join(sorted(assets))
    return (
        Path(base_dir)
        / f"s6_features_{tag}"
        / f"s6_features_{tag}_{date_str}_{hour:02d}.parquet"
    )


def _expected_outputs(job: "Job") -> List[Path]:
    """
    Returns the list of output Paths that this job is expected to produce.
    A job with only BTC+ETH produces 1 output; with BNB it produces 2.
    """
    outputs = [_s6_path(job.base_dir, job.date_str, job.hour, ["btc", "eth"])]
    if "bnb" in job.assets:
        outputs.append(
            _s6_path(job.base_dir, job.date_str, job.hour, ["bnb", "btc", "eth"])
        )
    return outputs


# ==============================================================================
# Discovery
# ==============================================================================

def _discover_jobs(
    base_dir: str,
    date_filter: Optional[str] = None,
    hour_filter: Optional[int] = None,
) -> List[Job]:
    """
    Scan s5_features/ and return (date, hour) jobs where at least BTC and ETH
    S5 files are present.  BNB is optional but included when available.
    """
    s5_dir = Path(base_dir) / "s5_features"
    if not s5_dir.exists():
        return []

    # Index which (date, hour) have which assets
    present: Dict[Tuple[str, int], set] = {}
    for f in sorted(s5_dir.iterdir()):
        if not f.is_file():
            continue
        m = _S5_PAT.match(f.name)
        if not m:
            continue
        asset, date_str, hour = m.group(1), m.group(2), int(m.group(3))

        if date_filter and date_str != date_filter:
            continue
        if hour_filter is not None and hour != hour_filter:
            continue

        present.setdefault((date_str, hour), set()).add(asset)

    # Require at least BTC+ETH; include BNB when available
    jobs: List[Job] = []
    for (date_str, hour), assets in sorted(present.items()):
        if _REQUIRED_ASSETS.issubset(assets):
            available = sorted(a for a in assets if a in _ALL_ASSETS)
            jobs.append(Job(date_str=date_str, hour=hour, base_dir=base_dir, assets=available))

    return jobs


def _check_inputs(job: Job) -> List[str]:
    """Return list of missing S5 input file paths."""
    missing = []
    for asset in job.assets:
        p = _s5_path(job.base_dir, asset, job.date_str, job.hour)
        if not p.exists():
            missing.append(str(p))
    return missing


def _output_exists(job: Job) -> bool:
    """True only when ALL expected outputs for this job already exist."""
    return all(p.exists() for p in _expected_outputs(job))


# ==============================================================================
# Execution
# ==============================================================================

def _run_job(job: Job, overwrite: bool = True) -> JobResult:
    t0 = time.time()
    try:
        import pandas as pd

        engine   = S6FeatureEngine(base_dir=job.base_dir, overwrite=overwrite)
        # engine.run() now always returns a list (0, 1 or 2 Paths)
        written  = engine.run(date=job.date_str, hour=job.hour)

        if not written:
            return JobResult(
                job=job, success=False,
                elapsed_s=time.time() - t0,
                error="engine.run() produced no outputs — check logs",
            )

        # Use the first output (btceth) for row/col reporting
        df_first = pd.read_parquet(written[0])
        total_mb = sum(p.stat().st_size for p in written) / (1024 * 1024)

        return JobResult(
            job=job, success=True,
            outputs=written,
            rows=len(df_first), cols=len(df_first.columns),
            size_mb=total_mb, elapsed_s=time.time() - t0,
        )
    except Exception as exc:
        return JobResult(
            job=job, success=False,
            elapsed_s=time.time() - t0,
            error=str(exc),
        )


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
            n_out = len(r.outputs)
            out_tag = " + ".join(
                p.parent.name for p in r.outputs
            )
            print(
                f"    {j.date_str} H{j.hour:02d}"
                f"  │  {r.rows:,} rows × {r.cols} cols"
                f"  │  {r.size_mb:.2f} MB total"
                f"  │  {r.elapsed_s:.1f}s"
                f"  │  {n_out} output(s): {out_tag}"
            )

    if failed:
        _section("Failed")
        for r in failed:
            j = r.job
            print(f"    {j.date_str} H{j.hour:02d}  │  {r.error}")

    print()


# ==============================================================================
# Main
# ==============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Run the S6 cross-asset feature engine (BTC ↔ ETH ↔ BNB).\n\n"
            "Auto-discovers S5 files and produces one cross-asset output\n"
            "per date/hour. BTC+ETH required, BNB optional.\n"
            "Output filename reflects available assets: btceth or btcethbnb."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  %(prog)s                              # process all ready pairs\n"
            "  %(prog)s --date 2026-02-16            # only this date\n"
            "  %(prog)s --date 2026-02-16 --hour 3  # single hour\n"
            "  %(prog)s --dry-run                    # show plan, don't execute\n"
            "  %(prog)s --skip-existing              # skip already-written hours\n"
        ),
    )

    ap.add_argument("--date",     type=str, default=None, help="YYYY-MM-DD")
    ap.add_argument("--hour",     type=int, default=None, help="0-23")
    ap.add_argument("--base-dir", type=str, default=str(_DEFAULT_BASE_DIR),
                    help="Shared data storage root (default: data_storage)")
    ap.add_argument("--dry-run",        action="store_true")
    ap.add_argument("--skip-existing",  action="store_true")
    ap.add_argument("--quiet", "-q",    action="store_true")

    args = ap.parse_args()

    if args.date and not re.match(r"^\d{4}-\d{2}-\d{2}$", args.date):
        print(f"{_RED}Error:{_RESET} --date must be YYYY-MM-DD, got: {args.date}")
        sys.exit(1)
    if args.hour is not None and not (0 <= args.hour <= 23):
        print(f"{_RED}Error:{_RESET} --hour must be 0–23, got: {args.hour}")
        sys.exit(1)

    filter_parts = []
    if args.date: filter_parts.append(f"date={args.date}")
    if args.hour is not None: filter_parts.append(f"hour={args.hour:02d}")
    mode_label = ", ".join(filter_parts) if filter_parts else "all dates, all hours"

    _header("S6 Cross-Asset Feature Pipeline  (BTC ↔ ETH ↔ BNB)")

    _section("Configuration")
    _kv("Mode",        f"Auto-discovery ({mode_label})")
    _kv("Base dir",    args.base_dir)
    _kv("S5 input",    str(Path(args.base_dir) / "s5_features"))
    _kv("S6 btceth",      str(Path(args.base_dir) / "s6_features_btceth"))
    _kv("S6 btcethbnb",   str(Path(args.base_dir) / "s6_features_btcethbnb"))

    # ── Discovery ──────────────────────────────────────────────────────────────
    _section("Scanning for S5 jobs (BTC+ETH required, BNB optional)")
    jobs = _discover_jobs(
        base_dir=args.base_dir,
        date_filter=args.date,
        hour_filter=args.hour,
    )

    if not jobs:
        _warn("No valid S5 job sets found (need at least BTC+ETH).")
        _info(f"Scanned: {Path(args.base_dir) / 's5_features'}")
        _info("Ensure S5 has been run for at least BTC and ETH before running S6.")
        sys.exit(0)

    by_date: Dict[str, List[Job]] = {}
    for j in jobs:
        by_date.setdefault(j.date_str, []).append(j)
    for date_str, date_jobs in sorted(by_date.items()):
        n_full = sum(1 for j in date_jobs if len(j.assets) == len(_ALL_ASSETS))
        n_partial = len(date_jobs) - n_full
        hours = sorted(j.hour for j in date_jobs)
        tag = f"{n_full} full" + (f", {n_partial} partial (no BNB)" if n_partial else "")
        _ok(f"{date_str}: {len(date_jobs)} hour(s) [{tag}]  "
            f"[{', '.join(f'{h:02d}' for h in hours)}]")
    _info(f"Total: {len(jobs)} jobs discovered")

    # ── Skip existing ──────────────────────────────────────────────────────────
    valid_jobs: List[Job] = []
    skipped = 0
    if args.skip_existing:
        _section("Checking existing outputs")
    for job in jobs:
        if args.skip_existing and _output_exists(job):
            _warn(f"{job.date_str} H{job.hour:02d} — output exists, skipping")
            skipped += 1
            continue
        valid_jobs.append(job)
    if args.skip_existing and skipped:
        _info(f"Skipped {skipped} existing, {len(valid_jobs)} remaining")
    if not valid_jobs:
        _warn("All outputs already exist. Nothing to do.")
        sys.exit(0)

    # ── Validate inputs ────────────────────────────────────────────────────────
    _section(f"Input Validation ({len(valid_jobs)} jobs)")
    ready_jobs: List[Job] = []
    for job in valid_jobs:
        label  = f"{job.date_str} H{job.hour:02d}"
        missing = _check_inputs(job)
        if missing:
            _fail(f"{label} — missing {len(missing)} file(s):")
            for m in missing:
                print(f"        {_DIM}{m}{_RESET}")
        else:
            _ok(f"{label} — {' + '.join(a.upper() for a in job.assets)} S5 present")
            ready_jobs.append(job)

    if not ready_jobs:
        _warn("No valid jobs after input check. Exiting.")
        sys.exit(1)

    # ── Dry run ────────────────────────────────────────────────────────────────
    if args.dry_run:
        _section("Dry Run — Execution Plan")
        for job in ready_jobs:
            print(f"    {job.date_str} H{job.hour:02d}:")
            for asset in job.assets:
                _kv(f"Input {asset.upper()}", str(_s5_path(job.base_dir, asset, job.date_str, job.hour)), indent=6)
            for out_p in _expected_outputs(job):
                _kv("Output", str(out_p), indent=6)
        print(f"\n  {_DIM}Re-run without --dry-run to execute.{_RESET}\n")
        return

    # ── Execute ────────────────────────────────────────────────────────────────
    _section(f"Execution ({len(ready_jobs)} jobs)")
    results: List[JobResult] = []
    total_t0 = time.time()

    for i, job in enumerate(ready_jobs, 1):
        label = f"{job.date_str} H{job.hour:02d}"
        _info(f"[{i}/{len(ready_jobs)}] {_BOLD}{label}{_RESET}")
        result = _run_job(job, overwrite=not args.skip_existing)
        results.append(result)
        if result.success:
            n_out    = len(result.outputs)
            dirs_str = ", ".join(p.parent.name for p in result.outputs)
            _ok(f"{label} — {result.rows:,} rows, {result.size_mb:.2f} MB total, "
                f"{result.elapsed_s:.1f}s, {n_out} output(s) [{dirs_str}]")
        else:
            _fail(f"{label} — {result.error}")

    _print_summary(results, time.time() - total_t0)

    if any(not r.success for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()