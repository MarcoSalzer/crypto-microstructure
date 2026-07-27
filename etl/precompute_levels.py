#!/usr/bin/env python3
# etl/precompute_levels.py
# ==============================================================================
# Level Precompute Orchestrator — run the reference-level generators for a
# target calendar day / ISO week / calendar month.
#
# PURPOSE:
#   The S1 feature engine reads these artefact parquets to build the
#   distance-to-level, range-position and volume-profile features (week_*,
#   monday_*, prev_week_*, month_*, prev_month_*, POC/VAH/VAL, price_vs_va,
#   fibonacci levels, ...). They are NOT produced by the S0/S1 engines
#   themselves, so this orchestrator drives the four generators:
#     - ohlc_running_{asset}_{YYYY-MM-DD}_{HH}.parquet   (running day OHLC; also
#           produced by the dedicated `ohlc` stage — see --skip-ohlc)
#     - weekly_{asset}_{iso_year}_{iso_week:02d}.parquet  (week_*, prev_week_*)
#     - monthly_{asset}_{year}_{month:02d}.parquet        (month_*, prev_month_*)
#     - vp_{asset}_{YYYY-MM-DD}.parquet                   (POC/VAH/VAL + migration)
#
#   It runs the generators for a specified date (default: yesterday UTC). As a
#   standalone cron it is meant to be called just after midnight UTC so the
#   previous day's artefacts exist; for backfill pass --date or --backfill-days.
#
# WHY YESTERDAY AND NOT TODAY?
#   The OHLC / VP scripts need the day's hourly S0 parquets, so a day can only
#   be summarised once it is complete. Weekly + monthly artefacts carry the
#   CURRENT week/month's expanding high/low up to the moment of computation;
#   running this daily keeps them fresh up to yesterday.
#
# RELATION TO run_all:
#   run_all wires this in as the `levels` stage between `ohlc` and `s1`, and
#   passes --skip-ohlc so the OHLC step is left to the dedicated `ohlc` stage
#   (the two would otherwise produce the same ohlc_running_* files). The stage
#   is date-targeted: `run_all --date X` precomputes X's levels; a full run
#   without --date falls back to this script's default (yesterday). To backfill
#   the whole history run `python -m etl.precompute_levels --asset btc eth
#   --backfill-days <N>` directly.
#
# USAGE:
#   # run for yesterday (cron mode) — all four generators
#   python -m etl.precompute_levels --asset btc
#
#   # a specific date
#   python -m etl.precompute_levels --asset btc --date 2026-03-10
#
#   # backfill the last 7 days for both assets
#   python -m etl.precompute_levels --asset btc eth --backfill-days 7
#
#   # weekly / monthly / VP only (OHLC handled by the ohlc stage)
#   python -m etl.precompute_levels --asset btc --date 2026-03-10 --skip-ohlc
#
# SCHEDULING:
#   Suggested cron (00:15 UTC daily for BTC + ETH):
#     15 0 * * *  /path/to/venv/bin/python -m etl.precompute_levels --asset btc eth >> /var/log/precompute.log 2>&1
#
# GENERATORS (all under etl/ohlc/, resolved via common.paths.DATA_ROOT):
#   - etl.ohlc.generate_ohlc
#   - etl.ohlc.generate_weekly_levels
#   - etl.ohlc.generate_monthly_levels
#   - etl.ohlc.generate_volume_profile
#
# EXIT CODES:
#   0 = all targeted artefacts exist or were produced successfully
#   1 = at least one precompute step failed (see stderr)
# ==============================================================================

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import date, datetime, timedelta, timezone
from typing import List, Tuple

# The four generators are invoked as modules so imports resolve against the
# flat package layout regardless of the working directory.
_OHLC_MODULE    = "etl.ohlc.generate_ohlc"
_WEEKLY_MODULE  = "etl.ohlc.generate_weekly_levels"
_MONTHLY_MODULE = "etl.ohlc.generate_monthly_levels"
_VP_MODULE      = "etl.ohlc.generate_volume_profile"


def _log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{ts}] [PRECOMPUTE] {msg}")


def _run(cmd: List[str], label: str, dry_run: bool = False) -> bool:
    """Run a subprocess, returning True on success. Streams stdout/stderr."""
    _log(f"▶ {label}  |  {' '.join(cmd)}")
    if dry_run:
        _log(f"  [dry-run] not executed")
        return True
    try:
        r = subprocess.run(cmd, capture_output=False)
        ok = r.returncode == 0
        _log(f"{label}  (exit={r.returncode})")
        return ok
    except Exception as e:
        _log(f"{label}  — exception: {e}")
        return False


def _dates_to_run(target_date: date, backfill_days: int) -> List[date]:
    """Return the list of dates to process, newest-first."""
    return [target_date - timedelta(days=i) for i in range(backfill_days + 1)]


def _iso_year_week(d: date) -> Tuple[int, int]:
    iso = d.isocalendar()
    return int(iso.year), int(iso.week)


def run_precompute_for_date(
    asset: str,
    d: date,
    python_exec: str = sys.executable,
    no_skip_existing: bool = False,
    no_require_complete: bool = False,
    skip_ohlc: bool = False,
    dry_run: bool = False,
) -> bool:
    """
    Run the level generators for one (asset, date). Returns True if ALL run
    steps succeed. A single failure returns False but the remaining steps still
    run so one missing artefact does not block the others.
    """
    date_str = d.strftime("%Y-%m-%d")
    iy, iw   = _iso_year_week(d)
    yr, mo   = d.year, d.month

    # generate_ohlc.py supports only --no-skip-existing; the level generators
    # additionally support --no-require-complete. Keep the two apart so the OHLC
    # step never receives a flag it does not define.
    ohlc_extra: List[str] = []
    level_extra: List[str] = []
    if no_skip_existing:
        ohlc_extra.append("--no-skip-existing")
        level_extra.append("--no-skip-existing")
    if no_require_complete:
        level_extra.append("--no-require-complete")

    results: List[bool] = []

    # 1) OHLC — needs the day's hourly S0 parquets. Skipped when the dedicated
    #    `ohlc` stage already produces the ohlc_running_* files.
    if not skip_ohlc:
        results.append(_run(
            [python_exec, "-m", _OHLC_MODULE,
             "--asset", asset, "--date", date_str] + ohlc_extra,
            f"ohlc     {asset} {date_str}", dry_run,
        ))

    # 2) Weekly — ISO week containing that date
    results.append(_run(
        [python_exec, "-m", _WEEKLY_MODULE,
         "--asset", asset, "--iso-week", f"{iy}-W{iw:02d}"] + level_extra,
        f"weekly   {asset} {iy}-W{iw:02d}", dry_run,
    ))

    # 3) Monthly — calendar month containing that date
    results.append(_run(
        [python_exec, "-m", _MONTHLY_MODULE,
         "--asset", asset, "--year", str(yr), "--month", str(mo)] + level_extra,
        f"monthly  {asset} {yr}-{mo:02d}", dry_run,
    ))

    # 4) Volume Profile
    results.append(_run(
        [python_exec, "-m", _VP_MODULE,
         "--asset", asset, "--date", date_str] + level_extra,
        f"vp       {asset} {date_str}", dry_run,
    ))

    return all(results)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run the level generators (OHLC, weekly, monthly, VP) for a "
                    "target date and asset."
    )
    ap.add_argument(
        "--asset", nargs="+", default=["btc", "eth"], choices=["btc", "eth"],
        help="One or more assets (default: both btc eth). run_all forwards a single\n"
             "asset only when it is itself invoked with --asset.",
    )
    ap.add_argument(
        "--date", type=str, default=None,
        help="Target date YYYY-MM-DD. Default: yesterday UTC.",
    )
    ap.add_argument(
        "--backfill-days", type=int, default=0,
        help="Also run for this many days before --date (default: 0 = just one).",
    )
    ap.add_argument(
        "--skip-ohlc", action="store_true",
        help="Skip the OHLC step (weekly/monthly/VP only). Used by the run_all "
             "`levels` stage, where the dedicated `ohlc` stage produces OHLC.",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Print the generator commands without executing them.",
    )
    ap.add_argument(
        "--no-skip-existing", action="store_true",
        help="Force re-computation even if output parquets already exist.",
    )
    ap.add_argument(
        "--no-require-complete", action="store_true",
        help="Allow missing hourly S0 parquets in the level generators (debugging).",
    )
    ap.add_argument(
        "--python", type=str, default=sys.executable,
        help="Python interpreter to use for subprocesses (default: current).",
    )
    args = ap.parse_args()

    # Resolve target date (default: yesterday UTC)
    if args.date:
        target_date = datetime.strptime(args.date, "%Y-%m-%d").date()
    else:
        target_date = (datetime.now(timezone.utc) - timedelta(days=1)).date()

    dates = _dates_to_run(target_date, args.backfill_days)

    _log(f"Target assets: {args.asset}")
    _log(f"Dates to run:  {len(dates)}  ({dates[-1]} .. {dates[0]})")
    if args.skip_ohlc:
        _log("OHLC step skipped (--skip-ohlc): weekly / monthly / VP only.")

    all_ok = True
    for d in dates:
        for asset in args.asset:
            ok = run_precompute_for_date(
                asset=asset, d=d,
                python_exec=args.python,
                no_skip_existing=args.no_skip_existing,
                no_require_complete=args.no_require_complete,
                skip_ohlc=args.skip_ohlc,
                dry_run=args.dry_run,
            )
            if not ok:
                all_ok = False

    _log(f"DONE  |  overall status: {'OK' if all_ok else 'FAILURES (see stderr)'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
