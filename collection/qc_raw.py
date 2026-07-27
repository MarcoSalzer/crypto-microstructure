# ==============================================================================
# Quality Control & Health Segment Detection — Binance Multi-Asset
#
# PURPOSE:
#   Post-hoc analysis of raw Parquet files produced by the unified pipeline.
#   Validates data quality, detects gaps, computes latency/spread stats,
#   and segments files into healthy/unhealthy time windows.
#
# SUPPORTED DATA TYPES:
#   - trades (spot/fut, per asset): trades_btc_spot_*.parquet, etc.
#   - deep L2 orderbook (spot/fut, per asset): lobdeep_btc_spot_*.parquet, etc.
#
# FILE NAMING CONVENTION (from collector.py):
#   {type}_{asset}_{market}_{date}_{hour}.parquet
#   Examples:
#     trades_btc_spot_2026-02-16_14.parquet
#     trades_eth_fut_2026-02-16_14.parquet
#     lobdeep_btc_spot_2026-02-16_14.parquet
#     lobdeep_eth_fut_2026-02-16_14.parquet
#
# CLI USAGE:
#   python -m collection.qc_raw --type all --asset all --market all
#   python -m collection.qc_raw --type trades --asset btc --market spot
#   python -m collection.qc_raw --type deep --asset eth --segments
#   python -m collection.qc_raw --file /path/to/specific.parquet
#   python -m collection.qc_raw --type deep --asset btc --segments --json
#
# HEALTH SEGMENT DETECTION:
#   Splits each file into fixed-duration windows (default 120s) and checks:
#   - Gap frequency (ts_ms discontinuities > MAX_GAP_MS)
#   - Spread magnitude (orderbook only, in basis points)
#   - Latency (ts_ms - exch_ts_ms)
#   - Crossed books (best_bid >= best_ask)
#   - Depth fulfillment (depth_actual vs depth_target)
#   - Reconnect events within the segment
#
# KEY DESIGN DECISIONS:
#   1. Uses ts_ms (receive-time) as the canonical time axis for gaps/segments.
#      This matches the adapter convention of receive-time as ground truth.
#   2. Latency computed as ts_ms - exch_ts_ms, bounded to avoid pollution.
#   3. Single-venue pipeline (Binance only): per-venue breakdowns still work
#      but will always show just "Binance". The --asset flag provides the
#      meaningful filtering dimension in the current architecture.
#
# ==============================================================================

from __future__ import annotations

import argparse
import json
import math
import statistics as stats
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

try:
    import pyarrow.parquet as pq
except ImportError:
    print("ERROR: pyarrow not installed. Run: pip install pyarrow")
    raise SystemExit(1)

# ==============================================================================
# Paths
# ==============================================================================
# ROOT is the repository root (one level above collection/).
# DATA_DIR is where the collector writes hourly Parquet files.

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data_storage" / "raw_data"

# ==============================================================================
# Asset Configuration
# ==============================================================================
# Must match the ASSETS dict in collector.py. Used to auto-generate
# file glob patterns when --asset=all.

KNOWN_ASSETS = ["btc", "eth", "bnb"]

# ==============================================================================
# Health Thresholds
# ==============================================================================
# Tuned for live Binance data. Adjust if adding more assets or venues.

MAX_GAP_MS = 3_000           # gaps longer than this are flagged
MAX_SPREAD_BP = 50           # 0.5% in basis points (orderbook only)
MAX_LATENCY_MS = 1_000       # segments with avg latency above this are flagged
MIN_DEPTH_RATIO = 0.5        # depth_actual / depth_target below this is flagged
SEGMENT_MIN_ROWS = 50        # segments with fewer rows are skipped

# Latency sanity bounds (avoid polluted stats from bad timestamps)
LAT_MIN_MS = -5_000
LAT_MAX_MS = 60_000


# ==============================================================================
# Health Segment Data Class
# ==============================================================================

@dataclass
class HealthSegment:
    """
    One fixed-duration window of data with aggregated health metrics.

    Attributes:
        start_ts_ms:    window start (epoch ms)
        end_ts_ms:      window end (epoch ms)
        rows:           total rows in this window
        venues:         unique venues seen (always ["Binance"] in current pipeline)
        avg_spread_bp:  mean bid-ask spread in basis points (orderbook only)
        avg_latency_ms: mean ts_ms - exch_ts_ms (None if no valid samples)
        issues:         list of detected problems (empty = healthy)
    """
    start_ts_ms: int
    end_ts_ms: int
    rows: int
    venues: List[str]
    avg_spread_bp: float
    avg_latency_ms: Optional[float]
    issues: List[str] = field(default_factory=list)

    @property
    def duration_sec(self) -> float:
        return (self.end_ts_ms - self.start_ts_ms) / 1000.0

    @property
    def is_healthy(self) -> bool:
        return len(self.issues) == 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_ts_ms": int(self.start_ts_ms),
            "end_ts_ms": int(self.end_ts_ms),
            "duration_sec": round(self.duration_sec, 1),
            "rows": int(self.rows),
            "venues": self.venues,
            "avg_spread_bp": round(float(self.avg_spread_bp), 3),
            "avg_latency_ms": round(float(self.avg_latency_ms), 1) if self.avg_latency_ms is not None else None,
            "is_healthy": self.is_healthy,
            "issues": self.issues,
        }


# ==============================================================================
# Utility Functions
# ==============================================================================

def latest_file(pattern: str) -> Optional[Path]:
    """Find the most recent file matching a glob pattern in DATA_DIR."""
    if not DATA_DIR.exists():
        return None
    files = sorted(DATA_DIR.glob(pattern))
    return files[-1] if files else None


def safe_len(x: Any) -> int:
    """Safe length for list columns (some rows may have None instead of list)."""
    return len(x) if isinstance(x, list) else 0


def percentile(arr: List[float], p: int) -> Optional[float]:
    """Compute percentile, returning None if insufficient samples."""
    arr2 = [x for x in arr if isinstance(x, (int, float)) and not math.isnan(x)]
    if len(arr2) < 10:
        return None
    return float(stats.quantiles(arr2, n=100)[p - 1])


def detect_data_type(columns: Set[str]) -> str:
    """Infer data type from Parquet column names."""
    if "trade_id" in columns and "price" in columns and "qty" in columns:
        return "trades"
    if "bids_px" in columns and "asks_px" in columns and "best_bid" in columns and "best_ask" in columns:
        return "book"  # deep L2 orderbook
    return "unknown"


def _iso(ts_ms: int) -> str:
    """Convert epoch milliseconds to ISO 8601 string."""
    return datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc).isoformat()


# ==============================================================================
# File-Level Analysis
# ==============================================================================

def analyze_file(path: Path, verbose: bool = True) -> Dict[str, Any]:
    """
    Analyze a single Parquet file and return a dict of quality metrics.

    Computes:
      - Row count, time range, duration
      - Per-venue and per-market breakdowns
      - Reconnect count
      - Gap detection (ts_ms discontinuities > MAX_GAP_MS)
      - Rows/sec total and per-venue
      - Latency stats (mean, p50, p90, p99)
      - Type-specific metrics (trades: sides, price, volume;
        book: spread, crossed, depth)
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"File: {path.name}")
        print(f"{'='*70}")

    try:
        t = pq.read_table(path)
    except Exception as e:
        print(f"ERROR reading file: {e}")
        return {"file": path.name, "error": str(e)}

    cols = {c.name for c in t.schema}
    data_type = detect_data_type(cols)

    def col(name: str) -> List[Any]:
        return t.column(name).to_pylist() if name in cols else []

    ts_ms = col("ts_ms")
    exch_ts_ms = col("exch_ts_ms")
    venue = col("venue")
    market_type = col("market_type")
    symbol = col("symbol")
    reconnect_flag = col("reconnect_flag")

    n_rows = len(ts_ms)
    if n_rows == 0:
        if verbose:
            print("  (empty file)")
        return {"file": path.name, "data_type": data_type, "rows": 0}

    venue_counts = Counter([v for v in venue if v is not None])
    market_counts = Counter([m for m in market_type if m is not None])
    symbol_counts = Counter([s for s in symbol if s is not None])
    reconnects = sum(1 for r in reconnect_flag if r == 1)

    ts_valid = [x for x in ts_ms if isinstance(x, int) and x > 0]
    ts_min = min(ts_valid) if ts_valid else 0
    ts_max = max(ts_valid) if ts_valid else 0
    duration_sec = ((ts_max - ts_min) / 1000.0) if ts_min and ts_max else 0.0

    # Latency: ts_ms - exch_ts_ms, bounded for sanity
    latencies: List[float] = []
    for i in range(n_rows):
        ts = ts_ms[i] if i < len(ts_ms) else None
        ex = exch_ts_ms[i] if i < len(exch_ts_ms) else None
        if isinstance(ts, int) and isinstance(ex, int) and ex > 0:
            lat = ts - ex
            if LAT_MIN_MS < lat < LAT_MAX_MS:
                latencies.append(float(lat))

    # Gap detection by ts_ms
    gaps: List[Tuple[int, int, int]] = []
    sorted_ts = sorted(ts_valid)
    for i in range(1, len(sorted_ts)):
        gap = sorted_ts[i] - sorted_ts[i - 1]
        if gap > MAX_GAP_MS:
            gaps.append((sorted_ts[i - 1], sorted_ts[i], int(gap)))

    # Throughput: rows/sec total + per venue
    rows_per_sec_total = (n_rows / duration_sec) if duration_sec > 0 else None
    rows_by_venue: Dict[str, int] = dict(venue_counts)
    rows_per_sec_by_venue: Dict[str, Optional[float]] = {}
    if duration_sec > 0:
        for v, c in rows_by_venue.items():
            rows_per_sec_by_venue[str(v)] = float(c) / duration_sec
    else:
        for v in rows_by_venue:
            rows_per_sec_by_venue[str(v)] = None

    result: Dict[str, Any] = {
        "file": path.name,
        "path": str(path),
        "data_type": data_type,
        "rows": int(n_rows),
        "duration_sec": round(duration_sec, 1),
        "time_range": {"start": _iso(ts_min) if ts_min else None, "end": _iso(ts_max) if ts_max else None},
        "venues": dict(venue_counts),
        "markets": dict(market_counts),
        "symbols": dict(symbol_counts),
        "reconnects": int(reconnects),
        "gaps_over_max_gap": int(len(gaps)),
        "max_gap_ms": int(MAX_GAP_MS),
        "rows_per_sec_total": round(rows_per_sec_total, 3) if rows_per_sec_total is not None else None,
        "rows_per_sec_by_venue": {k: (round(v, 3) if v is not None else None) for k, v in rows_per_sec_by_venue.items()},
    }

    if latencies:
        result["latency"] = {
            "mean": round(stats.fmean(latencies), 1),
            "p50": round(stats.median(latencies), 1),
            "p90": round(percentile(latencies, 90) or 0.0, 1),
            "p99": round(percentile(latencies, 99) or 0.0, 1),
            "samples": int(len(latencies)),
        }

    if data_type == "trades":
        result.update(analyze_trades(cols, col))
    elif data_type == "book":
        result.update(analyze_orderbook(cols, col))
    else:
        result["warn"] = "Unknown schema (missing expected columns)."

    if verbose:
        print_report(result)

    return result


# ==============================================================================
# Trade-Specific Analysis
# ==============================================================================

def analyze_trades(cols: Set[str], col) -> Dict[str, Any]:
    """Compute trade-specific metrics: side distribution, price range, volume."""
    price = col("price")
    qty = col("qty")
    side = col("side")

    side_counts = Counter([s for s in side if s is not None])

    notionals: List[float] = []
    for p, q in zip(price, qty):
        if isinstance(p, (int, float)) and isinstance(q, (int, float)) and p > 0 and q > 0:
            notionals.append(float(p) * float(q))

    pxs = [float(p) for p in price if isinstance(p, (int, float)) and p > 0]

    out: Dict[str, Any] = {"sides": dict(side_counts)}
    if pxs:
        out["price"] = {"min": round(min(pxs), 2), "max": round(max(pxs), 2)}
    if notionals:
        out["volume"] = {"total": round(sum(notionals), 2), "mean": round(stats.fmean(notionals), 4)}
    return out


# ==============================================================================
# Orderbook-Specific Analysis
# ==============================================================================

def analyze_orderbook(cols: Set[str], col) -> Dict[str, Any]:
    """Compute orderbook-specific metrics: spread, crossed books, depth levels."""
    best_bid = col("best_bid")
    best_ask = col("best_ask")
    bids_px = col("bids_px")
    asks_px = col("asks_px")
    depth_target = col("depth_target") if "depth_target" in cols else []
    depth_actual = col("depth_actual") if "depth_actual" in cols else []

    spreads_bp: List[float] = []
    crossed = 0

    n = min(len(best_bid), len(best_ask))
    for i in range(n):
        bb = best_bid[i]
        ba = best_ask[i]
        if isinstance(bb, (int, float)) and isinstance(ba, (int, float)) and bb > 0 and ba > 0:
            if bb >= ba:
                crossed += 1
            else:
                mid = 0.5 * (float(bb) + float(ba))
                spreads_bp.append((float(ba) - float(bb)) / mid * 1e4)

    bid_levels = [safe_len(bids_px[i]) for i in range(len(bids_px))]
    ask_levels = [safe_len(asks_px[i]) for i in range(len(asks_px))]

    depth_short = 0
    if depth_target and depth_actual:
        for i in range(min(len(depth_target), len(depth_actual))):
            dt = depth_target[i]
            da = depth_actual[i]
            if isinstance(dt, int) and isinstance(da, int) and dt > 0:
                if da < int(dt * MIN_DEPTH_RATIO):
                    depth_short += 1

    out: Dict[str, Any] = {"crossed_books": int(crossed)}
    if spreads_bp:
        out["spread_bp"] = {
            "mean": round(stats.fmean(spreads_bp), 3),
            "p50": round(stats.median(spreads_bp), 3),
            "p90": round(percentile(spreads_bp, 90) or 0.0, 3),
            "p99": round(percentile(spreads_bp, 99) or 0.0, 3),
        }
    if bid_levels:
        out["bid_levels"] = {"mean": round(stats.fmean(bid_levels), 1), "min": int(min(bid_levels)), "max": int(max(bid_levels))}
    if ask_levels:
        out["ask_levels"] = {"mean": round(stats.fmean(ask_levels), 1), "min": int(min(ask_levels)), "max": int(max(ask_levels))}
    if depth_target:
        out["depth_target"] = dict(Counter([d for d in depth_target if isinstance(d, int)]))
    if depth_short > 0:
        out["depth_short_count"] = int(depth_short)
    return out


# ==============================================================================
# Health Segment Detection
# ==============================================================================

def detect_health_segments(
    path: Path,
    segment_duration_sec: int = 120,
    required_venues: Optional[Set[str]] = None,
) -> List[HealthSegment]:
    """
    Split a Parquet file into fixed-duration windows and evaluate health.

    Each segment is checked for:
      - Missing required venues (if specified)
      - Reconnect events
      - Crossed book instances
      - High spread (> MAX_SPREAD_BP)
      - High latency (> MAX_LATENCY_MS)
      - Gaps in ts_ms (> MAX_GAP_MS)
      - Depth shortfalls (depth_actual < MIN_DEPTH_RATIO * depth_target)
      - Zero rows/sec for any required venue

    Args:
        path:                 Parquet file to analyze
        segment_duration_sec: Window size in seconds (default 120)
        required_venues:      Set of venue names that must appear in each segment.
                              In the current Binance-only pipeline, pass {"Binance"}
                              to flag segments where Binance data is missing.

    Returns:
        List of HealthSegment objects, one per non-empty window.
    """
    try:
        t = pq.read_table(path)
    except Exception as e:
        print(f"ERROR reading file: {e}")
        return []

    cols = {c.name for c in t.schema}

    def col(name: str) -> List[Any]:
        return t.column(name).to_pylist() if name in cols else []

    ts_ms = col("ts_ms")
    venue = col("venue")
    reconnect_flag = col("reconnect_flag")
    exch_ts_ms = col("exch_ts_ms")

    best_bid = col("best_bid") if "best_bid" in cols else []
    best_ask = col("best_ask") if "best_ask" in cols else []
    depth_target = col("depth_target") if "depth_target" in cols else []
    depth_actual = col("depth_actual") if "depth_actual" in cols else []

    ts_valid = [x for x in ts_ms if isinstance(x, int) and x > 0]
    if not ts_valid:
        return []

    indices = sorted(range(len(ts_ms)), key=lambda i: ts_ms[i] if isinstance(ts_ms[i], int) else 0)
    ts_min = int(ts_ms[indices[0]])
    ts_max = int(ts_ms[indices[-1]])

    seg_ms = int(segment_duration_sec * 1000)
    segments: List[HealthSegment] = []

    seg_start = ts_min
    while seg_start < ts_max:
        seg_end = seg_start + seg_ms

        seg_idx = [i for i in indices if isinstance(ts_ms[i], int) and seg_start <= ts_ms[i] < seg_end]
        if len(seg_idx) < SEGMENT_MIN_ROWS:
            seg_start = seg_end
            continue

        seg_venues_set = {str(venue[i]) for i in seg_idx if i < len(venue) and venue[i] is not None}
        seg_venues = sorted(seg_venues_set)

        seg_reconnects = sum(1 for i in seg_idx if i < len(reconnect_flag) and reconnect_flag[i] == 1)

        # Latency within segment
        seg_lat: List[float] = []
        for i in seg_idx:
            ts = ts_ms[i]
            ex = exch_ts_ms[i] if i < len(exch_ts_ms) else None
            if isinstance(ts, int) and isinstance(ex, int) and ex > 0:
                lat = ts - ex
                if 0 <= lat < 60_000:
                    seg_lat.append(float(lat))
        avg_latency = stats.fmean(seg_lat) if seg_lat else None

        # Spread and crossed books within segment
        seg_spreads: List[float] = []
        seg_crossed = 0
        if best_bid and best_ask:
            for i in seg_idx:
                if i < len(best_bid) and i < len(best_ask):
                    bb = best_bid[i]
                    ba = best_ask[i]
                    if isinstance(bb, (int, float)) and isinstance(ba, (int, float)) and bb > 0 and ba > 0:
                        if bb >= ba:
                            seg_crossed += 1
                        else:
                            mid = 0.5 * (float(bb) + float(ba))
                            seg_spreads.append((float(ba) - float(bb)) / mid * 1e4)
        avg_spread = stats.fmean(seg_spreads) if seg_spreads else 0.0

        # Depth fulfillment within segment
        depth_issues = 0
        if depth_target and depth_actual:
            for i in seg_idx:
                if i < len(depth_target) and i < len(depth_actual):
                    dt = depth_target[i]
                    da = depth_actual[i]
                    if isinstance(dt, int) and isinstance(da, int) and dt > 0:
                        if da < int(dt * MIN_DEPTH_RATIO):
                            depth_issues += 1

        # Gap detection within segment
        seg_ts_sorted = sorted(int(ts_ms[i]) for i in seg_idx if isinstance(ts_ms[i], int))
        seg_gaps = 0
        for j in range(1, len(seg_ts_sorted)):
            if seg_ts_sorted[j] - seg_ts_sorted[j - 1] > MAX_GAP_MS:
                seg_gaps += 1

        # Per-venue throughput within segment
        seg_duration = (seg_end - seg_start) / 1000.0
        seg_counts = Counter(str(venue[i]) for i in seg_idx if i < len(venue) and venue[i] is not None)
        seg_rows_per_sec_by_venue = {k: (v / seg_duration if seg_duration > 0 else None) for k, v in seg_counts.items()}

        # --- Issue Detection ---
        issues: List[str] = []

        if required_venues:
            missing = sorted(set(required_venues) - seg_venues_set)
            if missing:
                issues.append(f"missing_venues={','.join(missing)}")

        if seg_reconnects > 0:
            issues.append(f"reconnects={seg_reconnects}")
        if seg_crossed > 0:
            issues.append(f"crossed_books={seg_crossed}")
        if avg_spread > MAX_SPREAD_BP:
            issues.append(f"high_spread={avg_spread:.1f}bp")
        if avg_latency is not None and avg_latency > MAX_LATENCY_MS:
            issues.append(f"high_latency={avg_latency:.0f}ms")
        if seg_gaps > 0:
            issues.append(f"gaps={seg_gaps}")
        if depth_target and depth_actual and depth_issues > len(seg_idx) * 0.1:
            issues.append(f"depth_short={depth_issues}")

        # Check throughput for required venues
        if required_venues and seg_duration > 0:
            for rv in sorted(required_venues):
                rps = seg_rows_per_sec_by_venue.get(rv)
                if rps is None or rps <= 0:
                    issues.append(f"rps_zero={rv}")
                    break

        segments.append(
            HealthSegment(
                start_ts_ms=seg_start,
                end_ts_ms=seg_end,
                rows=len(seg_idx),
                venues=seg_venues,
                avg_spread_bp=float(avg_spread),
                avg_latency_ms=float(avg_latency) if avg_latency is not None else None,
                issues=issues,
            )
        )

        seg_start = seg_end

    return segments


# ==============================================================================
# Report Printing
# ==============================================================================

def print_report(result: Dict[str, Any]) -> None:
    """Print human-readable analysis results for a single file."""
    print(f"\nRows: {result.get('rows', 0):,}")
    print(f"Duration: {result.get('duration_sec', 0):.1f}s")
    tr = result.get("time_range") or {}
    print(f"Time: {tr.get('start')} to {tr.get('end')}")

    print(f"\nBy venue: {result.get('venues')}")
    print(f"By market: {result.get('markets')}")
    print(f"By symbol: {result.get('symbols')}")

    print(f"\nReconnects: {result.get('reconnects')}")
    print(f"Gaps >{result.get('max_gap_ms')}ms: {result.get('gaps_over_max_gap')}")

    rps_total = result.get("rows_per_sec_total")
    if rps_total is not None:
        print(f"\nRows/sec total: {rps_total}")
        rps_v = result.get("rows_per_sec_by_venue") or {}
        if rps_v:
            parts = [f"{k}={v}" for k, v in sorted(rps_v.items(), key=lambda x: x[0])]
            print("Rows/sec by venue: " + ", ".join(parts))

    if "latency" in result:
        lat = result["latency"]
        print(f"\nLatency: mean={lat['mean']}ms p50={lat['p50']}ms p90={lat['p90']}ms p99={lat['p99']}ms (n={lat['samples']})")

    if "spread_bp" in result:
        sp = result["spread_bp"]
        print(f"Spread: mean={sp['mean']}bp p50={sp['p50']}bp p90={sp['p90']}bp p99={sp['p99']}bp")

    if "crossed_books" in result:
        print(f"Crossed books: {result['crossed_books']}")

    if "bid_levels" in result:
        bl = result["bid_levels"]
        print(f"Bid levels: mean={bl['mean']} min={bl['min']} max={bl['max']}")

    if "ask_levels" in result:
        al = result["ask_levels"]
        print(f"Ask levels: mean={al['mean']} min={al['min']} max={al['max']}")

    if "depth_target" in result:
        print(f"Depth targets: {result['depth_target']}")

    if "depth_short_count" in result:
        print(f"Depth short: {result['depth_short_count']} rows")

    if "sides" in result:
        print(f"\nSide distribution: {result['sides']}")

    if "price" in result:
        pr = result["price"]
        print(f"Price range: {pr.get('min')} - {pr.get('max')}")

    if "volume" in result:
        vol = result["volume"]
        print(f"Volume: total={vol['total']:,.2f} mean={vol['mean']:.4f}")


def print_segments(segments: List[HealthSegment]) -> None:
    """Print formatted table of health segments."""
    healthy = [s for s in segments if s.is_healthy]
    unhealthy = [s for s in segments if not s.is_healthy]

    print(f"\n{'='*70}")
    print("HEALTH SEGMENTS")
    print(f"{'='*70}")
    print(f"Total segments: {len(segments)}")
    if segments:
        print(f"Healthy: {len(healthy)} ({len(healthy)/len(segments)*100:.1f}%)")
    print(f"Unhealthy: {len(unhealthy)}")

    if healthy:
        total_healthy_sec = sum(s.duration_sec for s in healthy)
        total_healthy_rows = sum(s.rows for s in healthy)
        print(f"\nHealthy coverage: {total_healthy_sec:.0f}s ({total_healthy_rows:,} rows)")

    print(f"\n{'Segment':<12} {'Duration':>10} {'Rows':>10} {'Spread':>10} {'Latency':>10} {'Status':<40}")
    print("-" * 105)

    for seg in segments:
        start_str = datetime.fromtimestamp(seg.start_ts_ms / 1000, tz=timezone.utc).strftime("%H:%M:%S")
        status = "HEALTHY" if seg.is_healthy else f"ISSUES: {', '.join(seg.issues[:4])}"
        lat_str = f"{seg.avg_latency_ms:.0f}ms" if seg.avg_latency_ms is not None else "n/a"
        print(
            f"{start_str:<12} {seg.duration_sec:>8.0f}s {seg.rows:>10,} "
            f"{seg.avg_spread_bp:>8.2f}bp {lat_str:>10} {status:<40}"
        )


# ==============================================================================
# CLI Helpers
# ==============================================================================

def _parse_required_venues(s: Optional[str]) -> Optional[Set[str]]:
    """Parse comma-separated venue names from CLI argument."""
    if not s:
        return None
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        return None
    return set(parts)


def _build_file_patterns(data_type: str, asset: str, market: str) -> List[str]:
    """
    Build glob patterns for finding Parquet files.

    File naming convention: {type}_{asset}_{market}_{date}_{hour}.parquet
    Examples:
      trades_btc_spot_2026-02-16_14.parquet
      lobdeep_eth_fut_2026-02-16_14.parquet

    Args:
        data_type: "trades", "deep", or "all"
        asset:     "btc", "eth", or "all"
        market:    "spot", "fut", or "all"

    Returns:
        List of glob pattern strings.
    """
    patterns: List[str] = []
    assets = KNOWN_ASSETS if asset == "all" else [asset.lower()]
    markets = ["spot", "fut"] if market == "all" else [market]

    if data_type in ("all", "trades"):
        for a in assets:
            for m in markets:
                patterns.append(f"trades_{a}_{m}_*.parquet")

    if data_type in ("all", "deep"):
        for a in assets:
            for m in markets:
                patterns.append(f"lobdeep_{a}_{m}_*.parquet")

    return patterns


# ==============================================================================
# Main Entry Point
# ==============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="QC — Quality Control & Health Segment Detection for Binance multi-asset pipeline"
    )
    parser.add_argument(
        "--type", choices=["trades", "deep", "all"], default="all",
        help="Data type to analyze: trades, deep (orderbook), or all (default: all)",
    )
    parser.add_argument(
        "--asset", choices=["btc", "eth", "bnb", "all"], default="all",
        help="Asset to analyze: btc, eth, or all (default: all)",
    )
    parser.add_argument(
        "--market", choices=["spot", "fut", "all"], default="all",
        help="Market type to analyze: spot, fut, or all (default: all)",
    )
    parser.add_argument(
        "--file", type=str,
        help="Specific file path to analyze (overrides --type/--asset/--market)",
    )
    parser.add_argument(
        "--segments", action="store_true",
        help="Show health segment analysis",
    )
    parser.add_argument(
        "--segment-duration", type=int, default=120,
        help="Segment duration in seconds (default: 120)",
    )
    parser.add_argument(
        "--venues", type=str, default=None,
        help="Required venues for segment coverage check, comma-separated. "
             "In current pipeline, use 'Binance' (must match parquet venue strings).",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output results as JSON to stdout",
    )
    args = parser.parse_args()

    req_venues = _parse_required_venues(args.venues)

    print("=" * 70)
    print("QC — Quality Control & Health Segment Detection")
    print("Pipeline: Binance-only, multi-asset (BTC + ETH)")
    print("=" * 70)
    print(f"Data directory: {DATA_DIR}")
    print(f"Assets: {', '.join(KNOWN_ASSETS)}")
    print(f"Thresholds: max_gap_ms={MAX_GAP_MS} segment_min_rows={SEGMENT_MIN_ROWS} segment_duration={args.segment_duration}s")

    if args.file:
        target = Path(args.file)
        if not target.exists():
            print(f"ERROR: file not found: {args.file}", file=sys.stderr)
            raise SystemExit(2)
        files = [target]
    else:
        patterns = _build_file_patterns(args.type, args.asset, args.market)

        files = []
        for p in patterns:
            f = latest_file(p)
            if f:
                files.append(f)

    if not files:
        print("\nNo files found to analyze.")
        if not args.file:
            print(f"Looked for patterns in: {DATA_DIR}")
            patterns = _build_file_patterns(args.type, args.asset, args.market)
            for p in patterns:
                print(f"  {p}")
        return

    all_results: List[Dict[str, Any]] = []
    all_segments: List[HealthSegment] = []

    for f in files:
        result = analyze_file(f, verbose=not args.json)
        all_results.append(result)

        if args.segments:
            segs = detect_health_segments(f, segment_duration_sec=args.segment_duration, required_venues=req_venues)
            all_segments.extend(segs)
            if not args.json:
                print_segments(segs)

    if args.json:
        out: Dict[str, Any] = {"files": all_results}
        if args.segments:
            out["segments"] = [s.to_dict() for s in all_segments]
        print(json.dumps(out, indent=2))

    if not args.json and len(files) > 1:
        print(f"\n{'='*70}")
        print(f"SUMMARY: Analyzed {len(files)} files")
        total_rows = sum(int(r.get("rows", 0) or 0) for r in all_results)
        total_reconnects = sum(int(r.get("reconnects", 0) or 0) for r in all_results)
        print(f"Total rows: {total_rows:,}")
        print(f"Total reconnects: {total_reconnects}")


if __name__ == "__main__":
    main()