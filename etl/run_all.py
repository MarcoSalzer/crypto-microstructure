#!/usr/bin/env python3
# ==============================================================================
# Pipeline Orchestrator: raw_data → S0 context → S0 features → S1 → … → S5
#
# OVERVIEW
# --------
# Runs the full feature pipeline sequentially, stage by stage.  Each stage
# delegates to its own run script (run_s0_context_batch.py, run_s0_features.py,
# run_s1_features.py … run_s5_features.py).  Supports BTC, ETH, and BNB as
# assets.  BNB data is processed when available but its absence never blocks
# the pipeline — hours with only BTC + ETH data are handled normally.
#
# STREAMING SAFETY
# ----------------
# The live collector writes Parquet files atomically (tmpfile → os.replace) and
# only finalises the current hour's files at rotation.  Until that moment the
# current-hour files do not exist in raw_data/ as complete files, so the S0
# discovery logic (which requires all 4 raw files to be present) will naturally
# skip the in-progress hour.
#
# Consequence: you can run this script continuously alongside live collection
# with no special locking.  The current streaming hour is simply never picked up.
#
# ARCHIVING
# ---------
# run_s0_features.py archives consumed raw + context files to
# data_storage/data_archive/<YYYY-MM-DD>/ after successful S0 processing.
# The higher stages operate on the already-archived S0 output and never touch
# raw_data again.
#
# CROSS-DAY DATA
# --------------
# Stage engines read prior-stage Parquets.  Because we always process
# chronologically (sorted by date+hour within each stage), previous-day outputs
# are present before the next day's hours are attempted.
#
# IDEMPOTENCY
# -----------
# Every stage script skips jobs whose output already exists (--skip-existing
# is passed automatically by this orchestrator).  Re-running the pipeline after
# a partial failure is safe.
#
# USAGE
# -----
#   # Full pipeline, all assets, all discoverable hours:
#   python -m etl.run_all
#
#   # Single date:
#   python -m etl.run_all --date 2026-03-01
#
#   # Single asset:
#   python -m etl.run_all --asset btc
#
#   # Only run certain stages (comma-separated, no spaces):
#   python -m etl.run_all --stages s0_context,s0_features,s1,s2
#
#   # Dry-run (passes --dry-run to every stage script):
#   python -m etl.run_all --dry-run
#
#   # Don't stop on stage failure (run all stages regardless):
#   python -m etl.run_all --continue-on-error
#
#   # Verbose: show full subprocess output (default: show summary only):
#   python -m etl.run_all --verbose
#
# EXIT CODES
# ----------
#   0  all stages succeeded (or dry-run)
#   1  one or more stages failed
# ==============================================================================

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

# ---------------------------------------------------------------------------
# Path layout
#   etl/run_all.py             ← this file
#   …
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent   # etl/
_REPO_ROOT   = _SCRIPT_DIR.parent               # repo root
_LOG_DIR    = _REPO_ROOT / "logs"
_LOG_FILE   = _LOG_DIR / "pipeline.log"
_PID_FILE   = _LOG_DIR / "pipeline.pid"


# ==============================================================================
# ANSI colours (skip gracefully if terminal doesn't support them)
# ==============================================================================

_BOLD  = "\033[1m"
_DIM   = "\033[2m"
_GREEN = "\033[92m"
_YELLOW= "\033[93m"
_RED   = "\033[91m"
_CYAN  = "\033[96m"
_RESET = "\033[0m"


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def _hr(char: str = "─", width: int = 72) -> str:
    return char * width


def _header(title: str) -> None:
    print()
    print(f"{_BOLD}{_hr('═')}{_RESET}")
    print(f"{_BOLD}  {title}{_RESET}")
    print(f"{_BOLD}{_hr('═')}{_RESET}")


def _stage_header(label: str) -> None:
    print()
    print(f"{_CYAN}{_BOLD}{_hr('─')}{_RESET}")
    print(f"{_CYAN}{_BOLD}  STAGE: {label}{_RESET}")
    print(f"{_CYAN}{_hr('─')}{_RESET}")


def _ok(msg: str)   -> None: print(f"  {_GREEN}{msg}{_RESET}")
def _warn(msg: str) -> None: print(f"  {_YELLOW}{msg}{_RESET}")
def _fail(msg: str) -> None: print(f"  {_RED}{msg}{_RESET}")
def _info(msg: str) -> None: print(f"  {_DIM}[{_ts()}]{_RESET} {msg}")


# ==============================================================================
# Stage definition
# ==============================================================================

@dataclass
class StageResult:
    stage_id:    str
    label:       str
    script:      Path
    returncode:  int          = -1
    elapsed_s:   float        = 0.0
    skipped:     bool         = False   # stage not in --stages filter
    script_missing: bool      = False


@dataclass
class StageSpec:
    """One pipeline stage."""
    stage_id:    str            # e.g. "s0_context", "s1"
    label:       str            # human-readable
    script_name: str            # filename in _SCRIPT_DIR

    # Which CLI args to forward from the orchestrator (subset of FORWARD_ARGS)
    forward_asset:   bool = True
    forward_date:    bool = True
    forward_dry_run: bool = True

    # Stage-specific extra args always appended (e.g. --skip-existing)
    extra_args:  List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Canonical stage sequence
# ---------------------------------------------------------------------------
ALL_STAGES: List[StageSpec] = [
    StageSpec(
        stage_id    = "s0_context",
        label       = "S0 Context (health + calendar + usability)",
        script_name = "run_s0_context_batch.py",
        # context batch doesn't support --date or --skip-existing as CLI args;
        # it operates on all available data and checks existence internally.
        forward_date = False,
    ),
    StageSpec(
        stage_id    = "s0_features",
        label       = "S0 Features (price / activity / aggression / bookshape / imbalance)",
        script_name = "run_s0_features.py",
        # s0_features doesn't accept --date either; it processes all hours
        # using --skip-existing to avoid re-computing.
        forward_date = False,
        extra_args  = ["--skip-existing"],
    ),
    StageSpec(
        stage_id     = "ohlc",
        label        = "OHLC (daily high/low/open/close — complete days only)",
        script_name  = "run_ohlc.py",
        forward_date = True,
        forward_asset= True,
        extra_args   = ["--skip-existing"],
        # Note: run_ohlc.py enforces its own complete-day guard internally.
        # Partial days (e.g. the current streaming day) are silently skipped.
    ),
    StageSpec(
        stage_id     = "levels",
        label        = "Levels (weekly / monthly / volume-profile reference levels for S1)",
        script_name  = "precompute_levels.py",
        forward_date = True,
        forward_asset= True,
        # --skip-ohlc: the `ohlc` stage above already produces the running-OHLC
        # files; here we orchestrate only weekly / monthly / volume-profile.
        extra_args   = ["--skip-ohlc"],
    ),
    StageSpec(
        stage_id    = "s1",
        label       = "S1 Features",
        script_name = "run_s1_features.py",
        extra_args  = ["--skip-existing"],
    ),
    StageSpec(
        stage_id    = "s2",
        label       = "S2 Features (rolling aggregations)",
        script_name = "run_s2_features.py",
        extra_args  = ["--skip-existing"],
    ),
    StageSpec(
        stage_id    = "s3",
        label       = "S3 Features (composite analytics)",
        script_name = "run_s3_features.py",
        extra_args  = ["--skip-existing"],
    ),
    StageSpec(
        stage_id    = "s4",
        label       = "S4 Features (advanced derived)",
        script_name = "run_s4_features.py",
        extra_args  = ["--skip-existing"],
    ),
    StageSpec(
        stage_id    = "s5",
        label       = "S5 Features",
        script_name = "run_s5_features.py",
        extra_args  = ["--skip-existing"],
    ),
    StageSpec(
        stage_id    = "s6",
        label       = "S6 Cross-Asset Features",
        script_name = "run_s6_features.py",
        extra_args  = ["--skip-existing"],
    ),
]

ALL_STAGE_IDS = [s.stage_id for s in ALL_STAGES]


# ==============================================================================
# Helpers
# ==============================================================================

def _parse_stages(raw: str) -> List[str]:
    """Parse comma-separated stage IDs, validate against ALL_STAGE_IDS."""
    requested = [s.strip() for s in raw.split(",") if s.strip()]
    unknown = [s for s in requested if s not in ALL_STAGE_IDS]
    if unknown:
        print(f"{_RED}Unknown stage(s): {', '.join(unknown)}{_RESET}")
        print(f"Valid stages: {', '.join(ALL_STAGE_IDS)}")
        sys.exit(1)
    return requested


def _build_cmd(
    spec:         StageSpec,
    asset_filter: Optional[str],
    date_filter:  Optional[str],
    dry_run:      bool,
) -> List[str]:
    """Build the subprocess command for one stage."""
    # Per-stage runners live under etl/ (run_ohlc.py included).
    script = _SCRIPT_DIR / spec.script_name
    cmd = [sys.executable, str(script)]

    if spec.forward_asset and asset_filter:
        cmd += ["--asset", asset_filter]

    if spec.forward_date and date_filter:
        cmd += ["--date", date_filter]

    if spec.forward_dry_run and dry_run:
        cmd.append("--dry-run")

    cmd.extend(spec.extra_args)
    return cmd


def _run_stage(
    spec:         StageSpec,
    asset_filter: Optional[str],
    date_filter:  Optional[str],
    dry_run:      bool,
    verbose:      bool,
) -> StageResult:
    """Execute a single stage subprocess; return StageResult."""
    result = StageResult(
        stage_id = spec.stage_id,
        label    = spec.label,
        script   = _SCRIPT_DIR / spec.script_name,
    )

    if not result.script.exists():
        _warn(f"Script not found: {result.script} — skipping stage")
        result.script_missing = True
        result.returncode = 0   # treat as non-fatal (stage may not exist yet)
        return result

    cmd = _build_cmd(spec, asset_filter, date_filter, dry_run)
    _info(f"cmd: {' '.join(cmd)}")

    t0 = time.time()
    try:
        if verbose:
            # Stream subprocess output directly to our stdout/stderr
            proc = subprocess.run(cmd, check=False)
        else:
            # Capture and only show on failure
            proc = subprocess.run(
                cmd,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            if proc.returncode != 0:
                # Print captured output so user sees what went wrong
                print(proc.stdout or "")

        result.returncode = proc.returncode
    except Exception as exc:
        result.returncode = 1
        _fail(f"Failed to launch {spec.script_name}: {exc}")

    result.elapsed_s = time.time() - t0
    return result


# ==============================================================================
# Summary
# ==============================================================================

def _print_summary(results: List[StageResult], total_elapsed: float) -> None:
    print()
    print(f"{_BOLD}{_hr('═')}{_RESET}")
    print(f"{_BOLD}  PIPELINE SUMMARY{_RESET}")
    print(f"{_BOLD}{_hr('═')}{_RESET}")
    print()

    col_w = max(len(r.label) for r in results) + 2
    for r in results:
        if r.skipped:
            status = f"{_DIM}SKIP{_RESET}"
            extra  = ""
        elif r.script_missing:
            status = f"{_YELLOW}MISSING{_RESET}"
            extra  = f"  script not found"
        elif r.returncode == 0:
            status = f"{_GREEN}OK{_RESET}"
            extra  = f"  {r.elapsed_s:.1f}s"
        else:
            status = f"{_RED}FAIL (rc={r.returncode}){_RESET}"
            extra  = f"  {r.elapsed_s:.1f}s"

        label_padded = r.label.ljust(col_w)
        print(f"  {label_padded}  {status}{extra}")

    print()
    print(f"  Total wall time: {total_elapsed:.1f}s")
    print(f"{_BOLD}{_hr('═')}{_RESET}")
    print()


# ==============================================================================
# Background / status helpers
# ==============================================================================

def _launch_background(args: argparse.Namespace) -> None:
    """Re-launch this script under nohup, sans --background flag."""
    _LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Reconstruct argv without --background / -b
    skip = {"--background", "-b"}
    forward_argv = [a for a in sys.argv[1:] if a not in skip]

    cmd = ["nohup", sys.executable, "-u", str(Path(__file__).resolve())] + forward_argv

    # Inherit current environment and ensure project root is on PYTHONPATH
    # so subprocesses can import the project's top-level packages regardless of working directory.
    env = os.environ.copy()
    project_root = str(_REPO_ROOT)   # repo root
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{project_root}:{existing}" if existing else project_root

    log_handle = open(_LOG_FILE, "a")
    proc = subprocess.Popen(
        cmd,
        stdout=log_handle,
        stderr=log_handle,          # 2>&1 — errors go to same log
        start_new_session=True,     # detach from this terminal session
        close_fds=True,
        env=env,
        cwd=str(_REPO_ROOT),  # repo root so the runners' package imports resolve
    )
    _PID_FILE.write_text(str(proc.pid))

    print(f"{_GREEN}{_BOLD}Pipeline started in background.{_RESET}")
    print(f"  PID : {proc.pid}  (saved to {_PID_FILE})")
    print(f"  Log : {_LOG_FILE}")
    print()
    print(f"  Follow output:  tail -f {_LOG_FILE}")
    print(f"  Check status:   python {Path(__file__).name} --status")
    print()


def _show_status() -> None:
    """Print background job status + last 30 lines of log."""
    print(f"{_BOLD}Pipeline status{_RESET}")
    print()

    # Check PID
    if _PID_FILE.exists():
        pid = int(_PID_FILE.read_text().strip())
        alive = _pid_running(pid)
        if alive:
            print(f"  {_GREEN}● RUNNING{_RESET}  (PID {pid})")
        else:
            print(f"  {_DIM}● NOT RUNNING{_RESET}  (PID {pid} — process finished or was killed)")
    else:
        print(f"  {_DIM}No PID file found — pipeline not started with --background yet.{_RESET}")

    print()

    # Tail log
    if _LOG_FILE.exists():
        size_mb = _LOG_FILE.stat().st_size / 1_048_576
        print(f"  Log: {_LOG_FILE}  ({size_mb:.2f} MB)")
        print(f"  {_DIM}Last 40 lines:{_RESET}")
        print(f"  {_DIM}{'─' * 68}{_RESET}")
        lines = _LOG_FILE.read_text(errors="replace").splitlines()
        for line in lines[-40:]:
            print(f"  {line}")
        print(f"  {_DIM}{'─' * 68}{_RESET}")
        print()
        print(f"  Live tail:  tail -f {_LOG_FILE}")
    else:
        print(f"  {_DIM}No log file yet: {_LOG_FILE}{_RESET}")
    print()


def _pid_running(pid: int) -> bool:
    """Return True if process with given PID is currently running."""
    try:
        os.kill(pid, 0)   # signal 0 = existence check, no actual signal sent
        return True
    except (ProcessLookupError, PermissionError):
        return False


# ==============================================================================
# Main
# ==============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Full pipeline orchestrator: raw_data → S0 → S1 → … → S6.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  %(prog)s                                    # everything ready\n"
            "  %(prog)s --asset btc                        # BTC only\n"
            "  %(prog)s --date 2026-03-01                  # single date\n"
            "  %(prog)s --stages s0_context,s0_features    # only S0 stages\n"
            "  %(prog)s --from-stage s2                    # S2 and beyond\n"
            "  %(prog)s --dry-run                          # plan only, no writes\n"
            "  %(prog)s --continue-on-error                # run all stages even if one fails\n"
        ),
    )

    ap.add_argument(
        "--asset", type=str, default=None, choices=["btc", "eth", "bnb"],
        help="Filter by asset. Default: all assets.",
    )
    ap.add_argument(
        "--date", type=str, default=None, metavar="YYYY-MM-DD",
        help="Filter by date. Default: all available dates.",
    )
    ap.add_argument(
        "--stages", type=str, default=None, metavar="ID[,ID,...]",
        help=(
            f"Comma-separated list of stages to run. "
            f"Valid: {', '.join(ALL_STAGE_IDS)}. "
            f"Default: all stages."
        ),
    )
    ap.add_argument(
        "--from-stage", type=str, default=None, metavar="STAGE_ID",
        help="Start from this stage (inclusive) and run all subsequent stages.",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Pass --dry-run to every stage script (no files written).",
    )
    ap.add_argument(
        "--continue-on-error", action="store_true",
        help="Run all stages even if an earlier stage fails. Default: stop on first failure.",
    )
    ap.add_argument(
        "--verbose", "-v", action="store_true",
        help="Stream full subprocess output. Default: suppress and show summary only.",
    )
    ap.add_argument(
        "--background", "-b", action="store_true",
        help=(
            f"Detach and run in background via nohup. "
            f"Output → {_LOG_FILE}  PID → {_PID_FILE}"
        ),
    )
    ap.add_argument(
        "--status", action="store_true",
        help="Check whether a background pipeline is running and tail the log.",
    )

    args = ap.parse_args()

    # ── --status: check a running background job ───────────────────────────
    if args.status:
        _show_status()
        return

    # ── --background: re-launch self under nohup and exit ─────────────────
    if args.background:
        _launch_background(args)
        return

    # ---- Resolve stage selection ----
    if args.stages and args.from_stage:
        print(f"{_RED}Error: --stages and --from-stage are mutually exclusive.{_RESET}")
        sys.exit(1)

    if args.stages:
        enabled_ids = _parse_stages(args.stages)
    elif args.from_stage:
        if args.from_stage not in ALL_STAGE_IDS:
            print(f"{_RED}Unknown stage: {args.from_stage}{_RESET}")
            print(f"Valid: {', '.join(ALL_STAGE_IDS)}")
            sys.exit(1)
        idx = ALL_STAGE_IDS.index(args.from_stage)
        enabled_ids = ALL_STAGE_IDS[idx:]
    else:
        enabled_ids = ALL_STAGE_IDS

    # ---- Header ----
    _header("BTC/ETH/BNB Feature Pipeline")

    now_utc = datetime.now(timezone.utc)
    _info(f"Started at {now_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    _info(f"Asset filter : {args.asset or 'all'}")
    _info(f"Date filter  : {args.date or 'all'}")
    _info(f"Stages       : {', '.join(enabled_ids)}")
    _info(f"Dry run      : {args.dry_run}")
    _info(f"Stop on fail : {not args.continue_on_error}")

    # ---- Note on streaming safety ----
    print()
    print(f"  {_DIM}Streaming safety: the current UTC hour ({now_utc.strftime('%H:xx')}) is")
    print(f"  automatically skipped because raw Parquet files for it are not yet")
    print(f"  fully written.  All completed prior hours are safe to process.{_RESET}")

    # ---- Execute stages ----
    results:       List[StageResult] = []
    total_t0:      float             = time.time()
    pipeline_ok:   bool              = True

    for spec in ALL_STAGES:
        if spec.stage_id not in enabled_ids:
            results.append(StageResult(
                stage_id = spec.stage_id,
                label    = spec.label,
                script   = _SCRIPT_DIR / spec.script_name,
                skipped  = True,
            ))
            continue

        _stage_header(spec.label)

        if not pipeline_ok and not args.continue_on_error:
            _warn("Skipping (prior stage failed and --continue-on-error not set)")
            results.append(StageResult(
                stage_id = spec.stage_id,
                label    = spec.label,
                script   = _SCRIPT_DIR / spec.script_name,
                skipped  = True,
            ))
            continue

        result = _run_stage(
            spec         = spec,
            asset_filter = args.asset,
            date_filter  = args.date,
            dry_run      = args.dry_run,
            verbose      = args.verbose,
        )
        results.append(result)

        if result.script_missing:
            _warn(f"Script missing → {spec.script_name}  (treating as skipped)")
        elif result.returncode == 0:
            _ok(f"Stage {spec.stage_id.upper()} completed in {result.elapsed_s:.1f}s")
        else:
            _fail(f"Stage {spec.stage_id.upper()} failed (rc={result.returncode})")
            pipeline_ok = False

    # ---- Summary ----
    total_elapsed = time.time() - total_t0
    _print_summary(results, total_elapsed)

    # ---- Exit ----
    if pipeline_ok or args.dry_run:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()