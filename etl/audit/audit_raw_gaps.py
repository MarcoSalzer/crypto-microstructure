#!/usr/bin/env python3
# etl/audit/audit_raw_gaps.py
# ==============================================================================
# Raw Data Gap Auditor
#
# Scans <DATA_ROOT>/raw_data/ and reports:
#   - Which (asset, file_type) series are present
#   - Gaps between hours (missing hours in a series)
#   - Incomplete hours (fewer than all 4 required files)
#   - Continuous runs (longest uninterrupted sequences)
#
# USAGE:
#   python -m etl.audit.audit_raw_gaps
#   python -m etl.audit.audit_raw_gaps --asset btc
#   python -m etl.audit.audit_raw_gaps --raw-dir /custom/path
#   python -m etl.audit.audit_raw_gaps --csv gaps.csv
# ==============================================================================

# -----------------------------------------------------------------------------
# etl/audit/audit_raw_gaps.py
# Raw-stream completeness/continuity audit (Thesis 3.1.4) over raw_data/ - file/hour
#   level only, never opens a feature value. Complementary input-side layer to 3.3.
#
# EXTERNAL DATA (standalone QA tool): reads the external, uncommitted ~94 GB
#   feature/data store, resolved via common.paths.DATA_ROOT (env THESIS_DATA_ROOT
#   or configs/paths.yaml). It does NOT run inside the repo without that store,
#   and is intentionally NOT wired into etl.run_all.
# START:  python -m etl.audit.audit_raw_gaps --help
# -----------------------------------------------------------------------------

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from common.paths import DATA_ROOT

_DEFAULT_RAW_DIR = DATA_ROOT / "raw_data"

# Matches: trades_btc_spot_2026-02-16_14.parquet
#          lobdeep_eth_fut_2026-03-01_00.parquet
_FILE_RE = re.compile(
    r"^(trades|lobdeep)_(\w+)_(spot|fut)_(\d{4}-\d{2}-\d{2})_(\d{2})\.parquet$"
)

# ANSI
_BOLD  = "\033[1m"; _DIM = "\033[2m"; _RST = "\033[0m"
_GRN   = "\033[92m"; _YLW = "\033[93m"; _RED = "\033[91m"; _CYN = "\033[96m"

# Required file types per asset-hour (complete set = 4 files)
_REQUIRED_TYPES = [
    ("trades",  "spot"),
    ("trades",  "fut"),
    ("lobdeep", "spot"),
    ("lobdeep", "fut"),
]


@dataclass(frozen=True, order=True)
class HourStamp:
    date_str: str
    hour: int

    @property
    def dt(self) -> datetime:
        return datetime(
            int(self.date_str[:4]),
            int(self.date_str[5:7]),
            int(self.date_str[8:10]),
            self.hour,
            tzinfo=timezone.utc,
        )

    def __str__(self) -> str:
        return f"{self.date_str} {self.hour:02d}:00"

    def next(self) -> "HourStamp":
        nxt = self.dt + timedelta(hours=1)
        return HourStamp(nxt.strftime("%Y-%m-%d"), nxt.hour)


def scan(raw_dir: Path, asset_filter: Optional[str] = None) -> Dict[str, Dict[HourStamp, List[str]]]:
    """
    Scan raw_dir and return:
        { asset: { HourStamp: [present_file_types] } }
    where file_type = "trades_spot" | "trades_fut" | "lobdeep_spot" | "lobdeep_fut"
    """
    result: Dict[str, Dict[HourStamp, List[str]]] = defaultdict(lambda: defaultdict(list))

    if not raw_dir.exists():
        print(f"{_RED}ERROR: raw_dir not found: {raw_dir}{_RST}")
        sys.exit(1)

    for f in raw_dir.iterdir():
        if not f.is_file():
            continue
        m = _FILE_RE.match(f.name)
        if not m:
            continue
        ftype, asset, market, date_str, hour_str = m.groups()
        if asset_filter and asset != asset_filter:
            continue
        hs = HourStamp(date_str, int(hour_str))
        result[asset][hs].append(f"{ftype}_{market}")

    return result


def find_gaps(hours: List[HourStamp]) -> List[Tuple[HourStamp, HourStamp, int]]:
    """
    Given a sorted list of present hours, return gaps:
        [(last_before_gap, first_after_gap, missing_count), ...]
    """
    gaps = []
    for i in range(len(hours) - 1):
        expected = hours[i].next()
        if expected != hours[i + 1]:
            # count missing hours
            cur = expected
            count = 0
            while cur != hours[i + 1]:
                count += 1
                cur = cur.next()
            gaps.append((hours[i], hours[i + 1], count))
    return gaps


def continuous_runs(hours: List[HourStamp]) -> List[Tuple[HourStamp, HourStamp, int]]:
    """
    Return list of (start, end, length) for each uninterrupted run.
    """
    if not hours:
        return []
    runs = []
    run_start = hours[0]
    run_end   = hours[0]
    for h in hours[1:]:
        if h == run_end.next():
            run_end = h
        else:
            runs.append((run_start, run_end, _hours_between(run_start, run_end)))
            run_start = h
            run_end   = h
    runs.append((run_start, run_end, _hours_between(run_start, run_end)))
    return runs


def _hours_between(a: HourStamp, b: HourStamp) -> int:
    delta = b.dt - a.dt
    return int(delta.total_seconds() // 3600) + 1


def _bar(n: int, total: int, width: int = 30) -> str:
    filled = round(width * n / max(total, 1))
    return f"[{'█' * filled}{'░' * (width - filled)}]"


def audit_asset(asset: str, hour_map: Dict[HourStamp, List[str]]) -> dict:
    all_hours   = sorted(hour_map.keys())
    complete    = [h for h in all_hours if len(hour_map[h]) == 4]
    incomplete  = [(h, hour_map[h]) for h in all_hours if len(hour_map[h]) < 4]
    gaps        = find_gaps(complete)
    runs        = continuous_runs(complete)
    longest_run = max(runs, key=lambda r: r[2]) if runs else None

    return {
        "asset":       asset,
        "all_hours":   all_hours,
        "complete":    complete,
        "incomplete":  incomplete,
        "gaps":        gaps,
        "runs":        runs,
        "longest_run": longest_run,
    }


def print_report(audit: dict) -> None:
    asset   = audit["asset"].upper()
    total   = len(audit["all_hours"])
    n_ok    = len(audit["complete"])
    n_inc   = len(audit["incomplete"])
    n_gaps  = len(audit["gaps"])
    runs    = audit["runs"]
    lr      = audit["longest_run"]

    print()
    print(f"{_BOLD}{_CYN}{'─'*72}{_RST}")
    print(f"{_BOLD}{_CYN}  Asset: {asset}{_RST}")
    print(f"{_BOLD}{_CYN}{'─'*72}{_RST}")
    print()

    if not total:
        print(f"  {_RED}No files found.{_RST}")
        return

    # -- Date range --
    first = audit["all_hours"][0]
    last  = audit["all_hours"][-1]
    span_h = _hours_between(first, last)
    print(f"  Range       : {first}  →  {last}")
    print(f"  Span        : {span_h} hours ({span_h/24:.1f} days)")
    print(f"  Present     : {n_ok} complete hours  {_bar(n_ok, span_h)}")
    coverage = 100 * n_ok / span_h
    col = _GRN if coverage >= 99 else _YLW if coverage >= 95 else _RED
    print(f"  Coverage    : {col}{coverage:.1f}%{_RST}")
    print()

    # -- Gaps --
    gaps = audit["gaps"]
    if not gaps:
        print(f"  {_GRN}No gaps — series is fully continuous.{_RST}")
    else:
        missing_total = sum(g[2] for g in gaps)
        print(f"  {_RED}{n_gaps} gap(s) found — {missing_total} missing hour(s):{_RST}")
        print()
        print(f"  {'#':>3}  {'Last present':<22}  {'First after gap':<22}  {'Missing':>7}")
        print(f"  {'─'*3}  {'─'*22}  {'─'*22}  {'─'*7}")
        for i, (before, after, count) in enumerate(gaps, 1):
            # List missing hours explicitly
            missing_list = []
            cur = before.next()
            while cur != after:
                missing_list.append(str(cur))
                cur = cur.next()
            flag = _YLW if count == 1 else _RED
            print(f"  {i:>3}  {str(before):<22}  {str(after):<22}  {flag}{count:>7}{_RST}")
            for mh in missing_list:
                print(f"       {_DIM}→ missing: {mh}{_RST}")
        print()

    # -- Incomplete hours --
    incomplete = audit["incomplete"]
    if incomplete:
        print(f"  {_YLW}{n_inc} incomplete hour(s) (< 4 raw files):{_RST}")
        req_set = {f"{ft}_{mk}" for ft, mk in _REQUIRED_TYPES}
        for hs, present in sorted(incomplete):
            missing_types = sorted(req_set - set(present))
            print(f"    {str(hs)}  has {len(present)}/4 files  "
                  f"{_DIM}missing: {', '.join(missing_types)}{_RST}")
        print()
    else:
        print(f"  {_GRN}All present hours have all 4 required files.{_RST}")

    # -- Continuous runs --
    print()
    print(f"  Continuous runs ({len(runs)} total):")
    runs_sorted = sorted(runs, key=lambda r: -r[2])  # longest first
    for i, (start, end, length) in enumerate(runs_sorted[:10]):
        marker = f"  {_GRN}★ LONGEST{_RST}" if i == 0 else ""
        print(f"    {str(start)}  →  {str(end)}  [{length:>5} hrs = {length/24:.1f} days]{marker}")
    if len(runs) > 10:
        print(f"    {_DIM}… {len(runs) - 10} more shorter runs not shown{_RST}")

    if lr:
        print()
        print(f"  {_BOLD}Longest continuous run: {lr[2]} hours ({lr[2]/24:.1f} days){_RST}")
        print(f"    {str(lr[0])}  →  {str(lr[1])}")


def write_csv(audits: List[dict], csv_path: Path) -> None:
    import csv
    rows = []
    for audit in audits:
        asset = audit["asset"]
        # Mark each hour
        all_set      = set(audit["all_hours"])
        complete_set = set(audit["complete"])
        incomplete_d = {h: files for h, files in audit["incomplete"]}

        # Build dense series from first to last
        if not audit["all_hours"]:
            continue
        cur  = audit["all_hours"][0]
        last = audit["all_hours"][-1]
        while cur <= last:
            status = "missing"
            files  = ""
            if cur in complete_set:
                status = "complete"
                files  = "4/4"
            elif cur in incomplete_d:
                n = len(incomplete_d[cur])
                status = f"incomplete_{n}/4"
                files  = ",".join(sorted(incomplete_d[cur]))
            rows.append({
                "asset":    asset,
                "date":     cur.date_str,
                "hour":     cur.hour,
                "datetime": str(cur),
                "status":   status,
                "files":    files,
            })
            cur = cur.next()

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["asset", "date", "hour", "datetime", "status", "files"])
        w.writeheader()
        w.writerows(rows)

    print(f"\n{_GRN}CSV written → {csv_path}{_RST}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Audit raw_data/ for hourly gaps and incomplete file sets."
    )
    ap.add_argument("--raw-dir", type=str, default=str(_DEFAULT_RAW_DIR),
                    help=f"Raw parquet folder (default: {_DEFAULT_RAW_DIR})")
    ap.add_argument("--asset", type=str, default=None, choices=["btc", "eth", "bnb"],
                    help="Filter by asset. Default: all.")
    ap.add_argument("--csv", type=str, default=None, metavar="PATH",
                    help="Write detailed per-hour CSV to this path.")
    ap.add_argument("--gaps-only", action="store_true",
                    help="Only print gap information, skip continuous runs detail.")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)

    print(f"{_BOLD}Raw Data Gap Audit{_RST}")
    print(f"  Directory : {raw_dir}")
    print(f"  Asset     : {args.asset or 'all'}")

    hour_maps = scan(raw_dir, asset_filter=args.asset)

    if not hour_maps:
        print(f"{_RED}No matching files found.{_RST}")
        sys.exit(1)

    audits = []
    for asset in sorted(hour_maps.keys()):
        a = audit_asset(asset, hour_maps[asset])
        audits.append(a)
        print_report(a)

    # Cross-asset alignment check (supports 2+ assets)
    if len(audits) >= 2:
        asset_hours = {a["asset"]: set(a["complete"]) for a in audits}
        all_assets  = sorted(asset_hours.keys())
        all_hours   = set().union(*asset_hours.values())

        # Hours where ALL discovered assets are complete
        common = all_hours.copy()
        for s in asset_hours.values():
            common &= s

        print()
        print(f"{_BOLD}{'─'*72}{_RST}")
        print(f"{_BOLD}  Cross-asset alignment{_RST}")
        print(f"{_BOLD}{'─'*72}{_RST}")
        names = " + ".join(a.upper() for a in all_assets)
        print(f"  All ({names}) complete : {_GRN}{len(common)}{_RST} hours")

        # Per-asset: hours present for this asset but missing for at least one other
        any_exclusive = False
        for asset in all_assets:
            exclusive = asset_hours[asset] - common
            if not exclusive:
                continue
            any_exclusive = True
            print(f"  {asset.upper()} without full overlap : {_YLW}{len(exclusive)}{_RST}")
            for h in sorted(exclusive)[:15]:
                present = [a.upper() for a in all_assets if h in asset_hours[a]]
                print(f"    {_DIM}{h}  [present: {', '.join(present)}]{_RST}")
            if len(exclusive) > 15:
                print(f"    {_DIM}… {len(exclusive)-15} more{_RST}")

        if not any_exclusive:
            print(f"  {_GRN}{names} are perfectly aligned.{_RST}")

    if args.csv:
        write_csv(audits, Path(args.csv))

    # Final verdict
    total_gaps = sum(len(a["gaps"]) for a in audits)
    print()
    if total_gaps == 0:
        print(f"{_GRN}{_BOLD}All series are gap-free. Safe to run full feature pipeline.{_RST}")
    else:
        print(f"{_YLW}{_BOLD}{total_gaps} gap(s) found. Review above before running pipeline.{_RST}")
        print(f"{_DIM}  Use --csv gaps.csv for a full per-hour breakdown.{_RST}")
    print()


if __name__ == "__main__":
    main()