#!/usr/bin/env python3
# etl/audit/audit_s6_features.py
# ==============================================================================
# Audit Script — S6 Cross-Asset Features  (dual-output edition)
#
# Reflects the dual-output architecture introduced 2026-05:
#   s6_features_btceth/     — BTC↔ETH pair only (all hours from BTC/ETH start)
#   s6_features_bnbbtceth/  — BTC↔ETH↔BNB (all 3 pairs, from BNB start onward)
#
# WHAT IS CHECKED:
#   COVERAGE    — Which date/hours have btceth-only vs both outputs.
#                 Flags any gap where bnbbtceth is unexpectedly missing once
#                 the first bnbbtceth file has been written.
#   STRUCTURAL  — Row count (~3600/hour), duplicate timestamps, Inf values,
#                 NaN rates per feature group, minimum column count.
#   SEMANTIC    — lag_corr ∈ [-1.01, 1.01]
#                 rolling_beta clipped ±5.0 (engine contract)
#                 regime flags ∈ {0, 1}
#                 bps_spread intermediaries ≥ 0
#                 cross_diff extreme values (|x| > 100 → likely Inf leak)
#   CROSS-COL   — Regime consistency: xor + align ≤ 1 per row per pair
#                 Lead-lag symmetry: a→b and b→a lags both present
#                 Regime complement pairs: xor ↔ align both present
#   QUALITY     — All-NaN columns, constant columns
#
# ASSET-SET AWARENESS:
#   btceth files   → only btceth pair features expected (~36 primary + 8 interm)
#   bnbbtceth files → all 3-pair features expected (~108 primary + 12 interm)
#   Minimum column floors and pair-tag checks adapt accordingly.
#
# FLAGS:
#   --mode btceth|bnbbtceth|both  Limit audit to one asset set (default: both)
#   --date YYYY-MM-DD             Audit only this date
#   --expand / -e                 Show per-file error/warning details inline
#   --stats  / -s                 Show aggregated statistics table per asset set
#   --file DATE HOUR [--set btceth|bnbbtceth]  Deep audit on a single file
#   --quick / -q                  With --file: checks only (no schema, no stats)
#   --coverage-only               Print coverage table and exit
# ==============================================================================

# -----------------------------------------------------------------------------
# etl/audit/audit_s6_features.py
# Sole S6 (cross-asset) Feature Corpus Audit (Thesis 3.3), on s6_features_btceth/
#   and s6_features_bnbbtceth/. audit_all does NOT cover S6. Runs independently.
#
# EXTERNAL DATA (standalone QA tool): reads the external, uncommitted ~94 GB
#   feature/data store, resolved via common.paths.DATA_ROOT (env THESIS_DATA_ROOT
#   or configs/paths.yaml). It does NOT run inside the repo without that store,
#   and is intentionally NOT wired into etl.run_all.
# START:  python -m etl.audit.audit_s6_features --help
# -----------------------------------------------------------------------------

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from common.paths import DATA_ROOT

# ---------------------------------------------------------------------------

_DATA_DIR                 = DATA_ROOT
_DEFAULT_DIR_BTCETH       = _DATA_DIR / "s6_features_btceth"
_DEFAULT_DIR_BTCETHBNB    = _DATA_DIR / "s6_features_bnbbtceth"
_DEFAULT_DATA_GAPS        = _DATA_DIR / "data_gaps.csv"

# Display helpers
_B, _G, _Y, _R, _D, _RST = "\033[1m", "\033[92m", "\033[93m", "\033[91m", "\033[2m", "\033[0m"
_C = "\033[96m"   # cyan for coverage info

def _header(t):  print(f"\n{_B}{'=' * 76}{_RST}\n{_B}  {t}{_RST}\n{_B}{'=' * 76}{_RST}")
def _section(t): print(f"\n{_B}▸ {t}{_RST}")
def _ok(m):      print(f"  {_G}{m}{_RST}")
def _warn(m):    print(f"  {_Y}{m}{_RST}")
def _fail(m):    print(f"  {_R}{m}{_RST}")
def _info(m):    print(f"  {_C}·{_RST} {m}")


# ==============================================================================
# Data gaps loader
# ==============================================================================

def _load_data_gaps(path: Path) -> Set[Tuple[str, int]]:
    """Load data_gaps.csv → set of (date_str, hour) tuples in any outage window."""
    if not path.exists():
        return set()
    try:
        df = pd.read_csv(path)
        gaps: Set[Tuple[str, int]] = set()
        for _, row in df.iterrows():
            for h in range(int(row["hour_start"]), int(row["hour_end"]) + 1):
                gaps.add((str(row["date"]), h))
        return gaps
    except Exception as exc:
        print(f"  {_Y}data_gaps.csv load failed: {exc}{_RST}")
        return set()


# ==============================================================================
# Asset-set metadata
# ==============================================================================

# Pair tags present in each asset set
_PAIRS_BY_SET: Dict[str, List[str]] = {
    "btceth":    ["btceth"],
    "bnbbtceth": ["btceth", "btcbnb", "ethbnb"],
}

# Minimum expected column count per asset set (generous floor for sanity check)
_MIN_COLS: Dict[str, int] = {
    "btceth":    30,    # ~36 primary btceth features + 8 btc/eth intermediaries
    "bnbbtceth": 80,    # ~108 primary (3 pairs) + 12 intermediaries
}


# ==============================================================================
# Feature classification
# ==============================================================================

def _classify(name: str) -> str:
    if not name.startswith("ca_"):
        return "other"
    if name.startswith("ca_lag_corr_"):
        return "lag_corr"
    if name.startswith("ca_rolling_beta_"):
        return "rolling_beta"
    if name.startswith("ca_residual_ret_"):
        return "beta_residual"
    if name.startswith("ca_z_residual_ret_"):
        return "residual_z"
    if name.startswith(("ca_regime_xor_", "ca_regime_align_")):
        return "regime_flag"
    if name.startswith(("ca_bps_mid_dev_", "ca_bps_spread_",
                         "ca_z_trade_count_", "ca_z_avg_trade_size_")):
        return "intermediary"
    if "microprice_dev_" in name and "_spread_" not in name:
        return "intermediary"
    return "cross_diff"


# NaN thresholds per semantic group (max acceptable NaN fraction)
_NAN_THRESH: Dict[str, float] = {
    "cross_diff":    0.15,
    "lag_corr":      0.10,
    "rolling_beta":  0.10,
    "beta_residual": 0.10,
    "residual_z":    0.15,
    "regime_flag":   0.20,   # 300s regime flag: structural ~18% NaN from rolling warmup
    "intermediary":  0.27,  # z_avg_trade_size has 900s window → ~25% warmup NaN
}

# Per-feature NaN threshold overrides (col prefix → max NaN fraction)
_NAN_THRESH_FEATURE_PATTERNS: Dict[str, float] = {
    "ca_z_avg_trade_size_spot_900s_":         0.55,   # 900s window on sparse trade data
    "ca_activity_avg_trade_size_spot_900s_":  0.55,
    "ca_basis_vwap_sf_spread_1s_":            0.40,   # instantaneous VWAP is sparse
    "ca_basis_vwap_sf_spread_60s_":           1.01,   # removed from spec; stale files only
    "ca_z_refill_vs_pull_":                   1.01,   # removed from spec; stale files only
    "ca_absorb_refill_":                      1.01,   # removed from spec; stale files only
}

# cross_diff features exempt from the ±100 extreme-value warning.
# Includes BPS-valued features (legitimately large) and per-asset intermediaries
# (ca_z_depth_*) whose extreme check is less meaningful since they feed into diffs.
_EXTREME_EXEMPT_PREFIXES = (
    "ca_dist_to_day_high_bps_",
    "ca_dist_to_day_low_bps_",
    "ca_day_range_bps_",
    "ca_range_pct_",
    "ca_basis_vwap_sf_",
    "ca_activity_",
    "ca_z_depth_",       # per-asset z-score intermediaries; clipped ±10 in operator
    "ca_z_refill_",      # not produced anymore; exempt to avoid stale-file warnings
)

_GROUP_ORDER = [
    "cross_diff", "lag_corr", "rolling_beta",
    "beta_residual", "residual_z", "regime_flag", "intermediary", "other",
]

_GROUP_LABELS = {
    "cross_diff":    "Cross-Asset Diffs  (z-diff / shock-diff / persist-diff / depth-diff)",
    "lag_corr":      "Lead-Lag Correlations  ∈ [-1, 1]",
    "rolling_beta":  "Rolling OLS Beta  (clipped ±5.0)",
    "beta_residual": "Beta Residual",
    "residual_z":    "Residual Robust Z-Score  (clipped ±20)",
    "regime_flag":   "Regime Flags  ∈ {0, 1}  (xor / align)",
    "intermediary":  "Intermediary Columns  (bps_mid_dev / bps_spread / activity-z)",
    "other":         "Non-S6 Columns  (unexpected)",
}


# ==============================================================================
# File discovery
# ==============================================================================

_PAT = re.compile(r"^s6_features_(btceth|bnbbtceth)_(\d{4}-\d{2}-\d{2})_(\d{2})\.parquet$")


@dataclass
class S6File:
    asset_set: str      # "btceth" or "bnbbtceth"
    date_str:  str
    hour:      int
    path:      Path


def _discover(
    base_dir:   Path,
    asset_set:  Optional[str] = None,   # None = both
    date_filter: Optional[str] = None,
    hour_filter: Optional[int] = None,
) -> List[S6File]:
    """
    Scan s6_features_btceth/ and s6_features_bnbbtceth/ under base_dir.
    Returns S6File objects sorted by (date, hour, asset_set).
    """
    dirs_to_scan: List[Tuple[str, Path]] = []
    if asset_set in (None, "btceth"):
        dirs_to_scan.append(("btceth",    base_dir / "s6_features_btceth"))
    if asset_set in (None, "bnbbtceth"):
        dirs_to_scan.append(("bnbbtceth", base_dir / "s6_features_bnbbtceth"))

    found: List[S6File] = []
    for aset, d in dirs_to_scan:
        if not d.exists():
            continue
        for f in d.iterdir():
            m = _PAT.match(f.name)
            if not m:
                continue
            file_set, date_str, hour = m.group(1), m.group(2), int(m.group(3))
            if file_set != aset:
                continue
            if date_filter and date_str != date_filter:
                continue
            if hour_filter is not None and hour != hour_filter:
                continue
            found.append(S6File(asset_set=aset, date_str=date_str, hour=hour, path=f))

    return sorted(found, key=lambda x: (x.date_str, x.hour, x.asset_set))


# ==============================================================================
# Coverage analysis
# ==============================================================================

def _print_coverage(files: List[S6File], base_dir: Path) -> None:
    """
    Show which date/hours have btceth-only vs both outputs.
    Flags gaps where bnbbtceth is unexpectedly missing after BNB start.
    """
    _section("Coverage")

    btceth_slots:    Set[Tuple[str, int]] = set()
    bnbbtceth_slots: Set[Tuple[str, int]] = set()

    for f in files:
        key = (f.date_str, f.hour)
        if f.asset_set == "btceth":
            btceth_slots.add(key)
        else:
            bnbbtceth_slots.add(key)

    all_slots = btceth_slots | bnbbtceth_slots
    if not all_slots:
        _warn("No S6 files found.")
        return

    both_slots      = btceth_slots & bnbbtceth_slots
    btceth_only     = btceth_slots - bnbbtceth_slots
    bnbbtceth_only  = bnbbtceth_slots - btceth_slots   # shouldn't happen

    # Gap detection: for every date that has at least one bnbbtceth file,
    # every btceth slot on that same date should also have a bnbbtceth file.
    bnb_dates = {s[0] for s in bnbbtceth_slots}
    bnb_gap_slots = {
        s for s in btceth_slots
        if s[0] in bnb_dates and s not in bnbbtceth_slots
    }

    total = len(all_slots)
    _ok(f"btceth files:            {len(btceth_slots):>5d}  ({len(btceth_slots)/total*100:.1f}%)")
    _ok(f"bnbbtceth files:         {len(bnbbtceth_slots):>5d}  ({len(bnbbtceth_slots)/total*100:.1f}%)")
    _info(f"both outputs present:    {len(both_slots):>5d} date/hours")
    _info(f"btceth-only (pre-BNB):   {len(btceth_only - bnb_gap_slots):>5d} date/hours")

    if bnbbtceth_only:
        _warn(f"bnbbtceth without matching btceth: {len(bnbbtceth_only)} slot(s)")

    if bnb_gap_slots:
        bnb_start = min(bnbbtceth_slots)
        bnb_end   = max(bnbbtceth_slots)
        _warn(
            f"bnbbtceth unexpectedly MISSING for {len(bnb_gap_slots)} slot(s) "
            f"within BNB date range [{bnb_start[0]} – {bnb_end[0]}]:"
        )
        for slot in sorted(bnb_gap_slots)[:20]:
            print(f"      {_Y}{slot[0]} H{slot[1]:02d}{_RST}")
        if len(bnb_gap_slots) > 20:
            print(f"      {_D}... and {len(bnb_gap_slots) - 20} more{_RST}")
    else:
        _ok("No unexpected gaps in bnbbtceth output window")

    # Date range summary
    dates = sorted({s[0] for s in all_slots})
    if dates:
        _info(f"Date range: {dates[0]} → {dates[-1]}  ({len(dates)} distinct dates)")

    # Missing hours check: for each date where files exist, which hours are absent?
    by_date_aset: Dict[Tuple[str, str], Set[int]] = defaultdict(set)
    for f in files:
        by_date_aset[(f.date_str, f.asset_set)].add(f.hour)

    missing_hours: List[str] = []
    for (date_str, aset), hours in sorted(by_date_aset.items()):
        missing = sorted(set(range(24)) - hours)
        if missing:
            missing_hours.append(
                f"{date_str} [{aset}]: missing H{', H'.join(f'{h:02d}' for h in missing)}"
            )

    if missing_hours:
        _warn(f"Incomplete days ({len(missing_hours)} date/set combinations):")
        for line in missing_hours[:15]:
            print(f"      {_Y}{line}{_RST}")
        if len(missing_hours) > 15:
            print(f"      {_D}... and {len(missing_hours) - 15} more{_RST}")
    else:
        _ok("All discovered dates have 24 hours of output")


# ==============================================================================
# Per-file audit
# ==============================================================================

@dataclass
class FileAudit:
    file:      S6File
    rows:      int   = 0
    cols:      int   = 0
    size_mb:   float = 0.0
    errors:    List[str] = field(default_factory=list)
    warnings:  List[str] = field(default_factory=list)
    ok:        bool  = True


def _audit_one_file(f: S6File, data_gap_hours: Set[Tuple[str, int]]) -> FileAudit:
    r      = FileAudit(file=f)
    r.size_mb = f.path.stat().st_size / (1024 * 1024)

    try:
        df = pq.read_table(str(f.path)).to_pandas()
    except Exception as exc:
        r.errors.append(f"read_error:{exc}"); r.ok = False; return r

    r.rows, r.cols, n = len(df), len(df.columns), len(df)

    if n == 0:
        r.errors.append("empty_df"); r.ok = False; return r
    if n < 3590:
        r.warnings.append(f"low_rows:{n}")

    # ── Column count sanity (asset-set-aware) ─────────────────────────────────
    min_cols = _MIN_COLS.get(f.asset_set, 30)
    if r.cols < min_cols:
        r.warnings.append(
            f"low_col_count:{r.cols} (expected ≥{min_cols} for {f.asset_set})"
        )

    # ── Timestamps ────────────────────────────────────────────────────────────
    ts = pd.to_datetime(df.index)
    nd = int(pd.Series(ts).duplicated().sum())
    if nd:
        r.errors.append(f"dup_timestamps:{nd}")

    s6_cols = [c for c in df.columns if c.startswith("ca_")]

    # ── Non-S6 columns should not be in the output ────────────────────────────
    non_s6 = [c for c in df.columns if not c.startswith("ca_")]
    if non_s6:
        r.warnings.append(
            f"unexpected_non_s6_cols:{len(non_s6)} "
            f"(upstream S5 cols should not be written to S6 output)"
        )

    # ── Per-column: Inf + NaN ─────────────────────────────────────────────────
    for col in s6_cols:
        g       = _classify(col)
        s       = pd.to_numeric(df[col], errors="coerce")
        nan_pct = s.isna().mean()
        inf_ct  = int(np.isinf(s.dropna()).sum())
        if inf_ct:
            r.errors.append(f"inf:{col}:{inf_ct}")
        feature_thresh = next(
            (v for pat, v in _NAN_THRESH_FEATURE_PATTERNS.items() if col.startswith(pat)),
            None,
        )
        thresh = feature_thresh if feature_thresh is not None else _NAN_THRESH.get(g, 0.15)
        if nan_pct > thresh:
            r.warnings.append(f"nan:{col}:{nan_pct:.1%}")

    # ==========================================================================
    # SEMANTIC CHECKS
    # ==========================================================================

    for col in s6_cols:
        g  = _classify(col)
        s  = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s) == 0:
            continue
        sf = s[np.isfinite(s)]
        if len(sf) == 0:
            continue
        vmin, vmax = float(sf.min()), float(sf.max())

        if g == "lag_corr":
            if vmin < -1.01 or vmax > 1.01:
                r.errors.append(f"lag_corr_oob:{col}:[{vmin:.4f},{vmax:.4f}]")
            elif vmin < -0.98 and vmax > 0.98:
                r.warnings.append(f"lag_corr_saturated:{col}")

        elif g == "rolling_beta":
            if abs(vmin) > 5.01 or abs(vmax) > 5.01:
                r.errors.append(f"beta_clip_violated:{col}:[{vmin:.4f},{vmax:.4f}]")

        elif g == "residual_z":
            if abs(vmin) > 20.01 or abs(vmax) > 20.01:
                r.errors.append(f"residual_z_clip_violated:{col}:[{vmin:.4f},{vmax:.4f}]")
            elif abs(vmin) > 15 or abs(vmax) > 15:
                r.warnings.append(f"residual_z_near_clip:{col}:[{vmin:.2f},{vmax:.2f}]")

        elif g == "regime_flag":
            non_binary = len(sf[~sf.isin([0.0, 1.0])])
            if non_binary:
                r.errors.append(f"regime_flag_non_binary:{col}:{non_binary}")

        elif g == "intermediary" and "bps_spread" in col:
            neg = int((sf < -1e-9).sum())
            if neg:
                r.errors.append(f"bps_spread_negative:{col}:{neg}")

        elif g == "cross_diff":
            # BPS-valued features (day ranges, distances, basis) are legitimately
            # larger than ±100 bps — exempt them from the extreme-value check.
            if not any(col.startswith(p) for p in _EXTREME_EXEMPT_PREFIXES):
                if abs(vmin) > 100 or abs(vmax) > 100:
                    r.warnings.append(f"cross_diff_extreme:{col}:[{vmin:.1f},{vmax:.1f}]")

    # ==========================================================================
    # CROSS-COLUMN CHECKS  (only for pairs actually present in this asset set)
    # ==========================================================================

    pair_tags = _PAIRS_BY_SET.get(f.asset_set, ["btceth"])

    # 1. Regime consistency: xor + align ≤ 1 at every row
    for win in ("60s", "300s"):
        for ptag in pair_tags:
            xor_col   = f"ca_regime_xor_{win}_{ptag}"
            align_col = f"ca_regime_align_{win}_{ptag}"
            if xor_col in df.columns and align_col in df.columns:
                xor_s   = pd.to_numeric(df[xor_col],   errors="coerce").fillna(0)
                align_s = pd.to_numeric(df[align_col], errors="coerce").fillna(0)
                bad     = int(((xor_s + align_s) > 1.001).sum())
                if bad:
                    r.errors.append(
                        f"regime_consistency:{xor_col}+{align_col}>1 at {bad} rows"
                    )

    # 2. Lead-lag symmetry: a→b and b→a both present for each pair
    for a, b in [t.split("") for t in []]:   # built dynamically below
        pass

    _LAG_PAIRS = [("btc", "eth")]
    if f.asset_set == "bnbbtceth":
        _LAG_PAIRS += [("btc", "bnb"), ("eth", "bnb")]

    for a, b in _LAG_PAIRS:
        lags_ab = {
            int(re.search(r"_(\d+)s$", c).group(1))
            for c in s6_cols
            if c.startswith(f"ca_lag_corr_{a}_taker_lead_{b}_ret_")
        }
        lags_ba = {
            int(re.search(r"_(\d+)s$", c).group(1))
            for c in s6_cols
            if c.startswith(f"ca_lag_corr_{b}_taker_lead_{a}_ret_")
        }
        if not lags_ab and not lags_ba:
            continue
        if lags_ba - lags_ab:
            r.warnings.append(
                f"cross_check:missing_{a}_lead_{b}_lags:{sorted(lags_ba - lags_ab)}"
            )
        if lags_ab - lags_ba:
            r.warnings.append(
                f"cross_check:missing_{b}_lead_{a}_lags:{sorted(lags_ab - lags_ba)}"
            )

    # 3. Regime complement pairs: xor ↔ align both present
    for win in ("60s", "300s"):
        for ptag in pair_tags:
            xor_col   = f"ca_regime_xor_{win}_{ptag}"
            align_col = f"ca_regime_align_{win}_{ptag}"
            if xor_col in df.columns and align_col not in df.columns:
                r.warnings.append(f"cross_check:missing_align_complement:{align_col}")
            if align_col in df.columns and xor_col not in df.columns:
                r.warnings.append(f"cross_check:missing_xor_complement:{xor_col}")

    # ==========================================================================
    # QUALITY CHECKS
    # ==========================================================================

    is_data_gap = (f.date_str, f.hour) in data_gap_hours

    all_nan_cols = [
        c for c in s6_cols
        if pd.to_numeric(df[c], errors="coerce").isna().all()
    ]
    if all_nan_cols:
        if is_data_gap:
            by_group: Dict[str, int] = {}
            for c in all_nan_cols:
                g = _classify(c)
                by_group[g] = by_group.get(g, 0) + 1
            summary = ", ".join(f"{g}×{n}" for g, n in sorted(by_group.items()))
            r.warnings.append(
                f"data_gap:all_nan:{len(all_nan_cols)} [{summary}] "
                f"(collector outage — NaN propagation expected)"
            )
        else:
            regime_nan = [c for c in all_nan_cols if _classify(c) == "regime_flag"]
            # Columns with known high structural NaN (basis_vwap_sf_1s, activity-z)
            # are downgraded to warning when all-NaN — can occur in low-activity hours.
            known_sparse = [
                c for c in all_nan_cols
                if c not in regime_nan and
                any(c.startswith(p) for p in _NAN_THRESH_FEATURE_PATTERNS)
            ]
            other_nan = [
                c for c in all_nan_cols
                if c not in regime_nan and c not in known_sparse
            ]
            if regime_nan:
                r.errors.append(f"all_nan_regime_cols:{len(regime_nan)}")
            if known_sparse:
                r.warnings.append(
                    f"all_nan_known_sparse:{len(known_sparse)} "
                    f"({', '.join(known_sparse[:3])}{'...' if len(known_sparse) > 3 else ''})"
                )
            if other_nan:
                r.errors.append(f"all_nan_s6_cols:{len(other_nan)}")

    constant = sum(
        1 for c in s6_cols
        if len(pd.to_numeric(df[c], errors="coerce").dropna()) > 100
        and pd.to_numeric(df[c], errors="coerce").dropna().std() < 1e-15
    )
    if constant > 3:
        r.warnings.append(f"constant_cols:{constant}")

    r.ok = len(r.errors) == 0
    return r


# ==============================================================================
# NaN Diagnosis
# ==============================================================================

def _diagnose_nan_reason(col: str) -> str:
    g = _classify(col)
    m = re.search(r"_(\d+)s", col)
    w = m.group(1) if m else "?"

    if g == "cross_diff":
        return (
            f"cross-asset diff — inherits NaN from upstream z-scored/shocked inputs "
            f"(rolling warmup ≤ {w}s upstream window)"
        )
    if g == "lag_corr":
        lag_m = re.search(r"_(\d+)s$", col)
        lag   = lag_m.group(1) if lag_m else "?"
        return f"rolling cross-correlation (window=60s, lag={lag}s) — first 60+{lag} rows NaN"
    if g == "rolling_beta":
        return f"rolling OLS beta (window={w}s) — first {w} rows NaN"
    if g == "beta_residual":
        return f"beta residual — NaN where parent beta is NaN (inherits {w}s warmup)"
    if g == "residual_z":
        return f"robust z-score of residual (window={w}s) — beta warmup + z warmup stacked"
    if g == "regime_flag":
        return "binary regime flag — NaN only if both upstream asset flags were NaN"
    if g == "intermediary":
        if "z_avg_trade_size" in col:
            return "activity z-score (window=900s) — first ~25% of hour rows NaN (expected)"
        if "z_trade_count" in col:
            return "activity z-score (window=300s) — first ~8% of hour rows NaN (expected)"
        return "intermediary — NaN where mid=0 or input NaN"
    return "unknown — non-S6 column"


def _nan_diagnosis(files: List[S6File], label: str) -> None:
    frames = []
    for f in files:
        try:
            frames.append(pq.read_table(str(f.path)).to_pandas())
        except Exception:
            pass
    if not frames:
        return

    big     = pd.concat(frames, ignore_index=True)
    n       = len(big)
    n_files = len(frames)

    nan_info = []
    for col in big.columns:
        if not col.startswith("ca_"):
            continue
        s       = pd.to_numeric(big[col], errors="coerce")
        nan_pct = s.isna().mean() * 100
        if nan_pct < 0.1:
            continue
        nan_info.append((col, int(s.isna().sum()), nan_pct, _diagnose_nan_reason(col)))

    if not nan_info:
        _section(f"NaN Diagnosis [{label}]: no significant NaN")
        return

    reason_groups: Dict[str, List[Tuple[str, int, float]]] = defaultdict(list)
    for col, ct, pct, reason in nan_info:
        reason_groups[reason].append((col, ct, pct))

    sorted_groups = sorted(
        reason_groups.items(),
        key=lambda x: sum(ct for _, ct, _ in x[1]),
        reverse=True,
    )

    _section(f"NaN Diagnosis [{label}]  ({n:,} rows × {n_files} files)")
    print(f"  {len(nan_info)} S6 columns with >0.1% NaN\n")

    for reason, cols in sorted_groups:
        total_nan = sum(ct for _, ct, _ in cols)
        print(
            f"  {_B}{reason}{_RST}  "
            f"({len(cols)} feature{'s' if len(cols) > 1 else ''}, "
            f"{total_nan:,} total NaN)"
        )
        for i, (col, ct, pct) in enumerate(sorted(cols, key=lambda x: -x[2])):
            if i < 5:
                short = col if len(col) <= 52 else col[:49] + "..."
                print(f"    {_D}│{_RST} {short:<54s} {pct:>6.1f}%  ({ct:>9,} NaN)")
            elif i == 5:
                print(f"    {_D}│ ... and {len(cols) - 5} more{_RST}")
                break
        print()


# ==============================================================================
# Aggregated error/warning summary
# ==============================================================================

def _aggregate_issues(results: List[FileAudit], label: str) -> None:
    err_counter:  Counter = Counter()
    warn_counter: Counter = Counter()

    for r in results:
        for e in r.errors:
            key = ":".join(e.split(":")[:2]) if ":" in e else e
            err_counter[key] += 1
        for w in r.warnings:
            if w.startswith("nan:"):
                continue
            key = ":".join(w.split(":")[:2]) if ":" in w else w
            warn_counter[key] += 1

    if err_counter:
        _section(f"Errors [{label}] — grouped ({sum(err_counter.values())} total)")
        for issue, count in err_counter.most_common(30):
            print(f"    {_R}{issue}  ({count} file{'s' if count > 1 else ''}){_RST}")

    if warn_counter:
        _section(f"Warnings [{label}] — non-NaN ({sum(warn_counter.values())} total)")
        for issue, count in warn_counter.most_common(30):
            print(f"    {_Y}{issue}  ({count} file{'s' if count > 1 else ''}){_RST}")


# ==============================================================================
# Stats table
# ==============================================================================

def _print_stats(files: List[S6File], label: str) -> None:
    frames = []
    for f in files:
        try:
            frames.append(pq.read_table(str(f.path)).to_pandas())
        except Exception:
            pass
    if not frames:
        return

    big = pd.concat(frames, ignore_index=True)
    n   = len(big)

    primary_cols = [
        c for c in big.columns
        if c.startswith("ca_") and _classify(c) != "intermediary"
    ]
    if not primary_cols:
        return

    _section(f"Stats [{label}] — primary S6 columns ({n:,} rows, {len(frames)} files)")
    hdr = f"    {'feature':<54s} {'nan%':>5s}  {'min':>12s}  {'median':>12s}  {'max':>12s}  {'std':>12s}"
    print(hdr)
    print(f"    {'─' * 106}")

    grouped: Dict[str, List[str]] = {}
    for col in primary_cols:
        grouped.setdefault(_classify(col), []).append(col)

    for g in _GROUP_ORDER:
        if g in ("intermediary", "other") or g not in grouped:
            continue
        print(f"\n    {_B}{_GROUP_LABELS.get(g, g)}{_RST}")
        for col in sorted(grouped[g]):
            s       = pd.to_numeric(big[col], errors="coerce")
            nan_pct = s.isna().mean() * 100
            sc      = s.dropna()
            short   = col if len(col) <= 52 else col[:49] + "..."
            if len(sc) == 0:
                print(f"    {short:<54s} {nan_pct:>5.1f}  {'(all NaN)':>12s}")
                continue
            vmin, vmed = float(sc.min()), float(sc.median())
            vmax, vstd = float(sc.max()), float(sc.std())
            fmt = ".4f" if abs(vmax) <= 10 else ".2f"
            print(
                f"    {short:<54s} {nan_pct:>5.1f}"
                f"  {vmin:>12{fmt}}  {vmed:>12{fmt}}  {vmax:>12{fmt}}  {vstd:>12{fmt}}"
            )


# ==============================================================================
# Schema printer
# ==============================================================================

def _print_schema(df: pd.DataFrame, label: str) -> None:
    _section(f"Schema — {label} ({len(df.columns)} columns)")
    grouped: Dict[str, List[str]] = {}
    for col in df.columns:
        grouped.setdefault(_classify(col), []).append(col)
    for g in _GROUP_ORDER:
        if g not in grouped:
            continue
        glabel = _GROUP_LABELS.get(g, g)
        print(f"\n    {_B}{glabel}{_RST} ({len(grouped[g])})")
        for col in sorted(grouped[g]):
            print(f"      {_D}│{_RST} {col:<66s} {_D}{df[col].dtype}{_RST}")


# ==============================================================================
# Multi-file audit runner (per asset set)
# ==============================================================================

def _run_audit_set(
    files:          List[S6File],
    label:          str,
    data_gap_hours: Set[Tuple[str, int]],
    expand:         bool,
    show_stats:     bool,
) -> Tuple[int, int, int]:
    """Audit all files in one asset set. Returns (n_ok, n_warn, n_fail)."""
    if not files:
        _warn(f"No files found for {label}")
        return 0, 0, 0

    _section(f"{label}  ({len(files)} files)")

    results: List[FileAudit] = []
    n_ok = n_warn = n_fail = 0

    for f in files:
        r     = _audit_one_file(f, data_gap_hours)
        results.append(r)
        label_ = f"{f.date_str} H{f.hour:02d}"

        if r.ok and not r.warnings:
            n_ok += 1
        elif r.ok:
            n_warn += 1
        else:
            n_fail += 1

        if expand:
            if not r.ok:
                print(f"  {_R}{label_}  │  {len(r.errors)}e {len(r.warnings)}w{_RST}")
                for e in r.errors:   print(f"      {_R}{e}{_RST}")
                for w in r.warnings: print(f"      {_Y}{w}{_RST}")
            elif r.warnings:
                print(f"  {_Y}{label_}  │  {len(r.warnings)}w{_RST}")
                for w in r.warnings: print(f"      {_Y}{w}{_RST}")
            else:
                print(f"  {_G}{label_}{_RST}")
        elif not r.ok:
            print(f"  {_R}{label_}  │  {len(r.errors)}e {len(r.warnings)}w{_RST}")

    print(
        f"\n  {_G}{n_ok} passed{_RST}  "
        f"{_Y}{n_warn} warnings-only{_RST}  "
        f"{_R}{n_fail} errors{_RST}  "
        f"(of {len(results)} files)"
    )

    _aggregate_issues(results, label)
    _nan_diagnosis(files, label)

    if show_stats:
        _print_stats(files, label)

    return n_ok, n_warn, n_fail


# ==============================================================================
# Main
# ==============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Audit S6 cross-asset features — dual-output edition (btceth + bnbbtceth).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  %(prog)s                                      # audit all files (both sets)\n"
            "  %(prog)s --mode bnbbtceth                     # only bnbbtceth files\n"
            "  %(prog)s --date 2026-03-20                    # only this date\n"
            "  %(prog)s --coverage-only                      # coverage table and exit\n"
            "  %(prog)s --expand                             # per-file details\n"
            "  %(prog)s --stats                              # full stats table\n"
            "  %(prog)s --file 2026-03-20 14                 # deep audit one file (both sets)\n"
            "  %(prog)s --file 2026-03-20 14 --set btceth    # deep audit btceth only\n"
            "  %(prog)s --file 2026-03-20 14 --quick         # checks only\n"
        ),
    )
    ap.add_argument("--base-dir", type=str, default=str(_DATA_DIR),
                    help="Root data directory (default: <DATA_ROOT>)")
    ap.add_argument("--data-gaps-file", type=str, default=str(_DEFAULT_DATA_GAPS))
    ap.add_argument("--mode", choices=["btceth", "bnbbtceth", "both"], default="both",
                    help="Which asset set to audit (default: both)")
    ap.add_argument("--date", type=str, default=None, help="YYYY-MM-DD")
    ap.add_argument("--expand",   "-e", action="store_true")
    ap.add_argument("--stats",    "-s", action="store_true")
    ap.add_argument("--coverage-only", action="store_true",
                    help="Print coverage table and exit immediately")
    ap.add_argument("--file", nargs=2, metavar=("DATE", "HOUR"),
                    help="Deep audit on a single date/hour")
    ap.add_argument("--set", dest="asset_set_filter",
                    choices=["btceth", "bnbbtceth"],
                    help="With --file: which asset set to inspect (default: both)")
    ap.add_argument("--quick", "-q", action="store_true",
                    help="With --file: checks only (no schema, no stats)")

    args       = ap.parse_args()
    base_dir   = Path(args.base_dir)
    asset_set  = None if args.mode == "both" else args.mode
    gap_hours  = _load_data_gaps(Path(args.data_gaps_file))

    _header("S6 Cross-Asset Feature Audit  (btceth + bnbbtceth)")
    print(f"  {_D}Base dir:{_RST} {base_dir}")
    print(f"  {_D}Mode:    {_RST} {args.mode}")

    # ── Single-file deep audit ─────────────────────────────────────────────────
    if args.file:
        d_arg    = args.file[0]
        h_arg    = int(args.file[1])
        aset_flt = args.asset_set_filter  # None = both
        files    = _discover(base_dir, asset_set=aset_flt,
                             date_filter=d_arg, hour_filter=h_arg)
        if not files:
            _fail(f"No S6 files found for {d_arg} H{h_arg:02d} (mode={aset_flt or 'both'})")
            sys.exit(1)

        for f in files:
            _header(
                f"{'Quick' if args.quick else 'Deep'} Audit — "
                f"S6 [{f.asset_set}] — {f.date_str} H{f.hour:02d}"
            )
            r = _audit_one_file(f, gap_hours)
            status = f"{r.rows:,} rows × {r.cols} cols  │  {r.size_mb:.2f} MB"

            if args.quick:
                if r.ok and not r.warnings:
                    _ok(f"{status}  │  clean")
                elif r.ok:
                    _warn(f"{status}  │  {len(r.warnings)} warnings")
                else:
                    _fail(f"{status}  │  {len(r.errors)} errors  {len(r.warnings)} warnings")
                for e in r.errors: _fail(f"  {e}")
                non_nan_w = [w for w in r.warnings if not w.startswith("nan:")]
                for w in non_nan_w[:15]: _warn(f"  {w}")
            else:
                print(f"  {status}")
                if r.ok and not r.warnings:
                    _ok("All checks passed")
                for e in r.errors:   _fail(e)
                for w in r.warnings: _warn(w)
                df = pq.read_table(str(f.path)).to_pandas()
                _print_schema(df, f"S6 [{f.asset_set}] {f.date_str} H{f.hour:02d}")
                _print_stats([f], f"S6 [{f.asset_set}] {f.date_str} H{f.hour:02d}")
                _nan_diagnosis([f], f"S6 [{f.asset_set}] {f.date_str} H{f.hour:02d}")
        return

    # ── Multi-file audit ───────────────────────────────────────────────────────
    all_files = _discover(base_dir, asset_set=asset_set, date_filter=args.date)

    if not all_files:
        _warn("No S6 files found. Check that the pipeline has been run.")
        sys.exit(0)

    print(f"  {_D}Total files:{_RST} {len(all_files)}")

    # Coverage (uses all files regardless of --mode for the overview)
    all_files_for_cov = _discover(base_dir, asset_set=None, date_filter=args.date)
    _print_coverage(all_files_for_cov, base_dir)

    if args.coverage_only:
        return

    # Split by asset set
    btceth_files    = [f for f in all_files if f.asset_set == "btceth"]
    bnbbtceth_files = [f for f in all_files if f.asset_set == "bnbbtceth"]

    total_ok = total_warn = total_fail = 0

    if btceth_files and asset_set in (None, "btceth"):
        ok, wn, fl = _run_audit_set(
            btceth_files, "btceth", gap_hours, args.expand, args.stats
        )
        total_ok += ok; total_warn += wn; total_fail += fl

    if bnbbtceth_files and asset_set in (None, "bnbbtceth"):
        ok, wn, fl = _run_audit_set(
            bnbbtceth_files, "bnbbtceth", gap_hours, args.expand, args.stats
        )
        total_ok += ok; total_warn += wn; total_fail += fl

    # ── Overall summary ────────────────────────────────────────────────────────
    _section("Overall Summary")
    total_rows = 0
    total_mb   = 0.0
    for f in all_files:
        total_mb += f.path.stat().st_size / (1024 * 1024)
    print(f"  {len(all_files)} files │ {total_mb:.1f} MB")
    print(
        f"  {_G}{total_ok} passed{_RST}  "
        f"{_Y}{total_warn} warnings-only{_RST}  "
        f"{_R}{total_fail} errors{_RST}"
    )

    if total_fail == 0 and total_warn == 0:
        print(f"  {_G}{_B}All checks passed{_RST}\n")
    elif total_fail == 0:
        print(f"  {_Y}No errors. Warnings are structural "
              f"(NaN warmup or sparse upstream data).{_RST}\n")
    else:
        print(f"  {_R}{_B}{total_fail} file(s) with errors — see above.{_RST}\n")

    sys.exit(1 if total_fail else 0)


if __name__ == "__main__":
    main()