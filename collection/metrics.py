# ==============================================================================
# Runtime Metrics Collection for Trade + Orderbook Adapters
#
# PURPOSE:
#   Lightweight per-stream counters and health signals during live collection.
#   Used by adapters to track: rows, reconnects, resyncs, invalid rows,
#   crossed books, latency, and mid-prices.
#
# ARCHITECTURE CONTEXT:
#   This module is imported by the Binance adapters (binance_trades.py,
#   binance_orderbook_deep.py) and the unified pipeline (collector.py).
#   The pipeline collects BTC/ETH/BNB data from Binance spot + futures.
#
# KEY FORMAT:
#   Key = (venue, market_type)
#     venue:       lowercase string (always "binance" in current pipeline)
#     market_type: "spot" | "fut" (canonical)
#
#   NOTE: BTC and ETH data is aggregated under the same key (e.g.
#   "binance:spot" covers both BTCUSDT and ETHUSDT spot). This is
#   intentional for high-level health monitoring. Per-asset granularity
#   is available via the Parquet files and QC tool.
#
# ADAPTER USAGE:
#   from collection import metrics
#   metrics.inc_rows("binance", "spot")
#   metrics.inc_reconnect("binance", "fut")
#   metrics.note_latency("binance", "spot", latency_ms)
#   metrics.note_mid("binance", "spot", mid)
#
# CONVENTIONS:
#   - "last message" uses ts_ms (receive-time) as ground truth, NOT exch_ts_ms.
#     Exchange timestamps can be missing, delayed, or inconsistent.
#   - Latency is computed only when both timestamps are sane and within bounds.
#   - Mid-price tracking stores the last written value per key. Since BTC and
#     ETH both write to the same key, this reflects whichever updated last.
#     The preflight check reports current mids and spot/fut basis for sanity.
#
# ==============================================================================

from __future__ import annotations

import asyncio
import statistics
import time
from collections import defaultdict, deque
from numbers import Integral, Real
from typing import Any, Deque, Dict, List, Optional, Tuple

# Key format: (venue, market_type)
Key = Tuple[str, str]
# Key3 format: (venue, market_type, stream)  — stream = "trades" | "depth"
Key3 = Tuple[str, str, str]

# ==============================================================================
# Internal State
# ==============================================================================
# All state is module-level. Thread safety is not needed (single event loop).
# Counters persist for the lifetime of the process.

_last_msg_ts_ms: Dict[Key, int] = defaultdict(int)  # recv-time ts_ms
_rows_window: Dict[Key, Deque[float]] = defaultdict(lambda: deque())
_rows_60s: Dict[Key, Deque[float]] = defaultdict(lambda: deque())     # 60s window for rows/s
_total_rows: Dict[Key, int] = defaultdict(int)

_resyncs: Dict[Key, int] = defaultdict(int)
_reconnects: Dict[Key, int] = defaultdict(int)
_invalid_rows: Dict[Key, int] = defaultdict(int)
_crossed_rows: Dict[Key, int] = defaultdict(int)

# Compat combined latency (emit_ts - exch_ts, kept for backward compat)
_latency_ms: Dict[Key, Deque[int]] = defaultdict(lambda: deque(maxlen=6000))
_latest_mid: Dict[Key, Optional[float]] = defaultdict(lambda: None)

# Stream-specific latency breakdown (Key3 = venue, market_type, stream)
_event_age_ms: Dict[Key3, Deque[int]] = defaultdict(lambda: deque(maxlen=6000))  # recv - exch
_pipe_ms: Dict[Key3, Deque[int]] = defaultdict(lambda: deque(maxlen=6000))       # emit - recv
_bad_exch_ts_count: Dict[Key3, int] = defaultdict(int)   # exch_ts was missing/invalid
_skew_count: Dict[Key3, int] = defaultdict(int)           # event_age < 0 (clock skew)

_QUEUE_MAX = 0
_WINDOW_SEC = 600  # 10-minute rolling window for rows/sec calculation
_WINDOW_60S = 60   # 60-second window for rows/s gauge

# Latency sanity bounds (strict to avoid pollution from bad timestamps)
_LAT_MIN_MS = 0
_LAT_MAX_MS = 60_000

# ==============================================================================
# Prometheus Integration (optional — graceful fallback if not installed)
# ==============================================================================
try:
    from prometheus_client import (
        Counter as _PCounter,
        Gauge as _PGauge,
        Histogram as _PHistogram,
        generate_latest as _prom_generate_latest,
        CONTENT_TYPE_LATEST as _PROM_CONTENT_TYPE,
    )
    _HAS_PROM = True

    _L2 = ["venue", "market"]   # 2-label metrics
    _L3 = ["venue", "market", "stream"]  # 3-label metrics

    _prom_rows_total       = _PCounter("ingest_rows_total",        "Total rows emitted",               _L2)
    _prom_reconnects_total = _PCounter("ingest_reconnects_total",  "Total reconnects",                 _L2)
    _prom_resyncs_total    = _PCounter("ingest_resyncs_total",     "Total resyncs (seq gap)",          _L2)
    _prom_invalid_total    = _PCounter("ingest_invalid_rows_total","Total invalid/dropped rows",       _L2)
    _prom_crossed_total    = _PCounter("ingest_crossed_rows_total","Total crossed-book rows",          _L2)
    _prom_bad_exch_total   = _PCounter("ingest_bad_exch_ts_total", "Total bad/missing exch timestamps",_L3)
    _prom_skew_total       = _PCounter("ingest_skew_total",        "Total clock-skew events (age<0)",  _L3)

    _prom_last_msg_age = _PGauge("ingest_last_msg_age_ms", "Age of last received message (ms)", _L2)
    _prom_mid          = _PGauge("ingest_mid",             "Latest mid price",                  _L2)
    _prom_rows_per_s   = _PGauge("ingest_rows_per_s",      "Rows per second (60s window)",      _L2)

    _BUCKETS_AGE  = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
    _BUCKETS_PIPE = [0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500]

    _prom_event_age = _PHistogram(
        "ingest_event_age_ms",
        "Event age ms (exchange→receive)",
        _L3,
        buckets=_BUCKETS_AGE,
    )
    _prom_pipe_ms_hist = _PHistogram(
        "ingest_pipe_ms",
        "Pipeline latency ms (receive→emit)",
        _L3,
        buckets=_BUCKETS_PIPE,
    )

except ImportError:
    _HAS_PROM = False


def _key3(venue: str, market_type: str, stream: str) -> Key3:
    """Create normalized 3-tuple key for stream-specific metrics."""
    v, m = _key(venue, market_type)
    s = (stream or "?").strip().lower()
    return (v, m, s)


def _key(venue: str, market_type: str) -> Key:
    """
    Create normalized key from venue and market_type.

    Normalization rules:
      - venue: lowercase (e.g. "binance")
      - market_type: canonical "spot" | "fut"
        Accepts aliases: fut/future/futures/perp/swap -> "fut"
        Fallback: lowercased unknown string
    """
    v = (venue or "?").strip().lower()

    m_raw = (market_type or "?").strip()
    m_low = m_raw.lower()

    if m_low == "spot":
        m = "spot"
    elif m_low in ("fut", "future", "futures", "perp", "swap"):
        m = "fut"
    else:
        m = m_low

    return (v, m)


# ==============================================================================
# Public API — Counter Functions
# ==============================================================================

def inc_rows(venue: str, market_type: str) -> None:
    """Increment row counter and update rolling 10-minute + 60-second windows."""
    k = _key(venue, market_type)
    _total_rows[k] += 1

    now = time.time()
    for dq, window in ((_rows_window[k], _WINDOW_SEC), (_rows_60s[k], _WINDOW_60S)):
        dq.append(now)
        cutoff = now - window
        while dq and dq[0] < cutoff:
            dq.popleft()

    if _HAS_PROM:
        try:
            _prom_rows_total.labels(venue=k[0], market=k[1]).inc()
        except Exception:
            pass


def inc_resync(venue: str, market_type: str) -> None:
    """Increment resync counter (sequence gap detected, book resyncing)."""
    k = _key(venue, market_type)
    _resyncs[k] += 1
    if _HAS_PROM:
        try:
            _prom_resyncs_total.labels(venue=k[0], market=k[1]).inc()
        except Exception:
            pass


def inc_reconnect(venue: str, market_type: str) -> None:
    """Increment reconnect counter (connection lost, reconnecting)."""
    k = _key(venue, market_type)
    _reconnects[k] += 1
    if _HAS_PROM:
        try:
            _prom_reconnects_total.labels(venue=k[0], market=k[1]).inc()
        except Exception:
            pass


def inc_invalid_rows(venue: str, market_type: str) -> None:
    """Increment invalid row counter (parse error, empty side, bad fields, etc.)."""
    k = _key(venue, market_type)
    _invalid_rows[k] += 1
    if _HAS_PROM:
        try:
            _prom_invalid_total.labels(venue=k[0], market=k[1]).inc()
        except Exception:
            pass


def inc_crossed_rows(venue: str, market_type: str) -> None:
    """Increment crossed book counter (best_bid >= best_ask)."""
    k = _key(venue, market_type)
    _crossed_rows[k] += 1
    if _HAS_PROM:
        try:
            _prom_crossed_total.labels(venue=k[0], market=k[1]).inc()
        except Exception:
            pass


def set_last_msg(venue: str, market_type: str, ts_ms: int) -> None:
    """
    Set last message timestamp for staleness detection.
    Convention: pass recv-time ts_ms (ground truth), not exch_ts_ms.
    """
    k = _key(venue, market_type)
    try:
        _last_msg_ts_ms[k] = int(ts_ms)
    except Exception:
        _last_msg_ts_ms[k] = 0


def note_latency(venue: str, market_type: str, latency_ms: int) -> None:
    """
    Record compat combined-latency sample (emit_ts - exch_ts).
    Kept for backward compat. Prefer note_event_age() / note_pipe_ms().
    """
    try:
        lat = int(latency_ms)
    except Exception:
        return
    if _LAT_MIN_MS <= lat < _LAT_MAX_MS:
        _latency_ms[_key(venue, market_type)].append(lat)


def note_event_age(
    venue: str,
    market_type: str,
    stream: str,
    age_ms: int,
    bad_ts: bool = False,
    skew: bool = False,
) -> None:
    """
    Record event-age sample: time from exchange event to our receive (recv_ts - exch_ts).

    Args:
        age_ms:   recv_ts_ms - exch_ts_ms (clamped to 0 if negative)
        bad_ts:   True when exch_ts was missing/invalid and we used recv_ts as fallback
        skew:     True when raw (recv_ts - exch_ts) was < 0 before clamping
    """
    k3 = _key3(venue, market_type, stream)
    try:
        a = int(age_ms)
    except Exception:
        return

    if bad_ts:
        _bad_exch_ts_count[k3] += 1
        if _HAS_PROM:
            try:
                _prom_bad_exch_total.labels(venue=k3[0], market=k3[1], stream=k3[2]).inc()
            except Exception:
                pass
        return  # don't record a fake 0 as latency sample

    if skew:
        _skew_count[k3] += 1
        if _HAS_PROM:
            try:
                _prom_skew_total.labels(venue=k3[0], market=k3[1], stream=k3[2]).inc()
            except Exception:
                pass

    if _LAT_MIN_MS <= a < _LAT_MAX_MS:
        _event_age_ms[k3].append(a)
        if _HAS_PROM:
            try:
                _prom_event_age.labels(venue=k3[0], market=k3[1], stream=k3[2]).observe(a)
            except Exception:
                pass


def note_pipe_ms(venue: str, market_type: str, stream: str, ms: int) -> None:
    """
    Record pipeline-latency sample: time from frame receive to emission (emit_ts - recv_ts).
    Only meaningful for streams with a separate emitter step (i.e. depth/orderbook).
    """
    k3 = _key3(venue, market_type, stream)
    try:
        p = int(ms)
    except Exception:
        return
    if _LAT_MIN_MS <= p < _LAT_MAX_MS:
        _pipe_ms[k3].append(p)
        if _HAS_PROM:
            try:
                _prom_pipe_ms_hist.labels(venue=k3[0], market=k3[1], stream=k3[2]).observe(p)
            except Exception:
                pass


def note_mid(venue: str, market_type: str, mid: float) -> None:
    """
    Record latest mid-price.

    NOTE: With multi-asset (BTC/ETH/BNB), this stores the last written mid
    per (venue, market_type). Since BTC and ETH alternate writes, the stored
    value reflects whichever asset updated most recently. This is acceptable
    for basic sanity checks but not for per-asset price tracking.
    """
    k = _key(venue, market_type)
    try:
        _latest_mid[k] = float(mid)
        if _HAS_PROM:
            try:
                _prom_mid.labels(venue=k[0], market=k[1]).set(float(mid))
            except Exception:
                pass
    except Exception:
        return


def note_row(row: Dict[str, Any]) -> None:
    """
    Convenience hook: call once per emitted row.
    Extracts venue + market_type and updates counters/latency/mid.

    IMPORTANT:
      - Uses row["ts_ms"] for staleness (recv-time ground truth).
      - Uses row["exch_ts_ms"] only for latency sampling.
    """
    venue = row.get("venue", "?")
    mtype = row.get("market_type", "?")

    inc_rows(venue, mtype)

    ts_ms = row.get("ts_ms")
    if isinstance(ts_ms, Integral) and int(ts_ms) > 0:
        set_last_msg(venue, mtype, int(ts_ms))

    ex_ms = row.get("exch_ts_ms")
    if isinstance(ts_ms, Integral) and isinstance(ex_ms, Integral) and int(ex_ms) > 0:
        lat = int(ts_ms) - int(ex_ms)
        if _LAT_MIN_MS <= lat < _LAT_MAX_MS:
            note_latency(venue, mtype, lat)

    bb = row.get("best_bid")
    ba = row.get("best_ask")
    if isinstance(bb, Real) and isinstance(ba, Real) and float(bb) > 0 and float(ba) > float(bb):
        note_mid(venue, mtype, 0.5 * (float(bb) + float(ba)))


# ==============================================================================
# Statistics Helpers
# ==============================================================================

def _p50(xs: List[int]) -> Optional[float]:
    if not xs:
        return None
    return float(statistics.median(xs))


def _p90(xs: List[int]) -> Optional[float]:
    if not xs or len(xs) < 10:
        return None
    return float(statistics.quantiles(xs, n=10)[8])


def _p99(xs: List[int]) -> Optional[float]:
    if not xs or len(xs) < 100:
        return None
    return float(statistics.quantiles(xs, n=100)[98])


def _latest(xs: List[int]) -> Optional[int]:
    return xs[-1] if xs else None


def _rows_per_s(k: Key, now: float) -> float:
    """Rows per second over last 60s window."""
    dq = _rows_60s.get(k)
    if not dq:
        return 0.0
    cutoff = now - _WINDOW_60S
    valid = sum(1 for t in dq if t >= cutoff)
    return valid / _WINDOW_60S


def get_stats() -> Dict[str, Any]:
    """
    Get current metrics snapshot as a dict.

    Top-level keys are "venue:market_type" (e.g. "binance:spot").
    Each entry has standard health counters + per-stream latency breakdown.

    Stream-specific metrics (event_age, pipe_ms) live under
    stats[key]["streams"][stream_name] where stream_name is "trades" or "depth".
    """
    now = time.time()
    now_ms = int(now * 1000)

    # Collect all 2-key and 3-key keys
    all_keys_2: set = (
        set(_total_rows.keys()) |
        set(_last_msg_ts_ms.keys()) |
        set(_rows_window.keys())
    )
    all_keys_3: set = (
        set(_event_age_ms.keys()) |
        set(_pipe_ms.keys()) |
        set(_bad_exch_ts_count.keys()) |
        set(_skew_count.keys())
    )

    # Update prometheus gauges that depend on current time
    if _HAS_PROM:
        for k in all_keys_2:
            try:
                last_ts = _last_msg_ts_ms.get(k, 0)
                age = (now_ms - last_ts) if last_ts else -1
                if age >= 0:
                    _prom_last_msg_age.labels(venue=k[0], market=k[1]).set(age)
                _prom_rows_per_s.labels(venue=k[0], market=k[1]).set(_rows_per_s(k, now))
            except Exception:
                pass

    stats: Dict[str, Any] = {}
    for k in sorted(all_keys_2):
        venue, mtype = k
        key_str = f"{venue}:{mtype}"

        total = _total_rows.get(k, 0)
        last_ts_ms = _last_msg_ts_ms.get(k, 0)
        age_ms = (now_ms - last_ts_ms) if last_ts_ms else -1
        last10m = len(_rows_window.get(k, ()))
        rps = _rows_per_s(k, now)

        lats = list(_latency_ms.get(k, ()))

        stats[key_str] = {
            "total_rows":        total,
            "rows_10m":          last10m,
            "rows_per_s":        round(rps, 2),
            "age_ms":            age_ms,
            "resyncs":           _resyncs.get(k, 0),
            "reconnects":        _reconnects.get(k, 0),
            "invalid_rows":      _invalid_rows.get(k, 0),
            "crossed_rows":      _crossed_rows.get(k, 0),
            # Compat combined latency (emit_ts - exch_ts), kept for compat
            "latency_p50":       _p50(lats),
            "latency_p90":       _p90(lats),
            "latency_p99":       _p99(lats),
            "latency_samples":   len(lats),
            "mid":               _latest_mid.get(k),
            "streams":           {},
        }

    # Attach per-stream breakdowns
    for k3 in sorted(all_keys_3):
        venue, mtype, stream = k3
        key_str = f"{venue}:{mtype}"
        if key_str not in stats:
            stats[key_str] = {
                "total_rows": 0, "rows_10m": 0, "rows_per_s": 0.0,
                "age_ms": -1, "resyncs": 0, "reconnects": 0,
                "invalid_rows": 0, "crossed_rows": 0,
                "latency_p50": None, "latency_p90": None, "latency_p99": None,
                "latency_samples": 0, "mid": None, "streams": {},
            }

        ages  = list(_event_age_ms.get(k3, []))
        pipes = list(_pipe_ms.get(k3, []))

        stats[key_str]["streams"][stream] = {
            "event_age_p50":      _p50(ages),
            "event_age_p90":      _p90(ages),
            "event_age_p99":      _p99(ages),
            "event_age_latest":   _latest(ages),
            "event_age_samples":  len(ages),
            "pipe_ms_p50":        _p50(pipes),
            "pipe_ms_p90":        _p90(pipes),
            "pipe_ms_p99":        _p99(pipes),
            "pipe_ms_latest":     _latest(pipes),
            "pipe_ms_samples":    len(pipes),
            "bad_exch_ts_count":  _bad_exch_ts_count.get(k3, 0),
            "skew_count":         _skew_count.get(k3, 0),
        }

    return stats


# ==============================================================================
# Prometheus Export
# ==============================================================================

def prometheus_generate_latest() -> bytes:
    """Return current metrics in Prometheus text exposition format."""
    if not _HAS_PROM:
        return b"# prometheus_client not installed\n"
    try:
        return _prom_generate_latest()
    except Exception as e:
        return f"# ERROR generating metrics: {e}\n".encode()


def prometheus_content_type() -> str:
    """Return correct Content-Type for Prometheus scrape responses."""
    if not _HAS_PROM:
        return "text/plain; charset=utf-8"
    return _PROM_CONTENT_TYPE


# ==============================================================================
# Async Health Ticker
# ==============================================================================

async def metrics_ticker(
    queue: Optional[asyncio.Queue] = None,
    interval_sec: int = 600,
    label: str = "METRICS",
) -> None:
    """
    Print health metrics every interval_sec (default: 10 minutes).

    Outputs per-stream row counts, staleness age, reconnect/resync counts,
    invalid/crossed counters, and latency percentiles.

    Args:
        queue:        Optional queue to monitor for pressure (qsize/max)
        interval_sec: Print interval in seconds
        label:        Label prefix for log output
    """
    global _QUEUE_MAX

    while True:
        await asyncio.sleep(interval_sec)

        now = time.time()
        now_ms = int(now * 1000)

        qsize = None
        qmax = None
        if queue is not None and hasattr(queue, "qsize"):
            qsize = int(queue.qsize())
            qmax = int(getattr(queue, "maxsize", 0) or 0)
            _QUEUE_MAX = max(_QUEUE_MAX, qsize)

        print(f"\n=== {label} health @ {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} UTC ===")

        all_keys = set(_total_rows.keys()) | set(_last_msg_ts_ms.keys()) | set(_rows_window.keys())

        for k in sorted(all_keys):
            venue, mtype = k
            total = _total_rows.get(k, 0)
            last_ts = _last_msg_ts_ms.get(k, 0)
            age_ms = (now_ms - last_ts) if last_ts else -1
            last10m = len(_rows_window.get(k, ()))

            lats = list(_latency_ms.get(k, ()))
            p50 = _p50(lats)
            avg = (sum(lats) / len(lats)) if lats else None

            invalid = _invalid_rows.get(k, 0)
            crossed = _crossed_rows.get(k, 0)

            print(
                f"  {venue:10s} {mtype:4s} | "
                f"10m={last10m:6d} total={total:10d} | "
                f"age={age_ms:7d}ms | "
                f"resync={_resyncs.get(k, 0):4d} reconn={_reconnects.get(k, 0):4d} | "
                f"invalid={invalid:6d} crossed={crossed:6d} | "
                f"lat_p50={p50 if p50 is not None else 'n/a':>6} "
                f"lat_avg={round(avg, 1) if avg is not None else 'n/a':>6}"
            )

        if qsize is not None:
            if qmax and qmax > 0:
                fill = qsize / qmax
                print(f"  queue_size={qsize} queue_max={_QUEUE_MAX} maxsize={qmax} fill={fill:.0%}")
            else:
                print(f"  queue_size={qsize} queue_max={_QUEUE_MAX}")
        print("=" * 80 + "\n")


# ==============================================================================
# Preflight Checks
# ==============================================================================

async def preflight_mid_check(tolerance_bps: float = 50.0) -> None:
    """
    Sanity-check mid-prices after a few seconds of streaming.

    With the Binance-only multi-asset pipeline, this checks:
      1. That we have mids for both spot and futures
      2. That the spot/fut basis (difference) is within tolerance
      3. Reports current mid values for operator awareness

    NOTE: Since BTC and ETH both write to the same (venue, market_type)
    key, the stored mid is whichever asset updated last. This check is
    a basic sanity gate, not a precise per-asset comparison. Use the QC
    tool (collection/qc_raw.py) for per-asset analysis on Parquet files.

    Args:
        tolerance_bps: Maximum allowed spot/fut mid deviation in basis points.
                       Default 50 bps (0.5%) is generous to accommodate the
                       natural basis between spot and perp futures.
    """
    spot_key = ("binance", "spot")
    fut_key = ("binance", "fut")

    spot_mid = _latest_mid.get(spot_key)
    fut_mid = _latest_mid.get(fut_key)

    print("[PRE-FLIGHT] Binance mid-price sanity check")

    if spot_mid is None and fut_mid is None:
        print("[PRE-FLIGHT] No mids available yet; skipping check.")
        return

    if spot_mid is not None:
        print(f"[PRE-FLIGHT] binance:spot  mid = {spot_mid:,.2f}")
    else:
        print("[PRE-FLIGHT][WARN] binance:spot  mid = (none)")

    if fut_mid is not None:
        print(f"[PRE-FLIGHT] binance:fut   mid = {fut_mid:,.2f}")
    else:
        print("[PRE-FLIGHT][WARN] binance:fut   mid = (none)")

    # Compare spot vs futures basis if both are available
    if spot_mid is not None and fut_mid is not None and spot_mid > 0:
        basis_bps = (fut_mid - spot_mid) / spot_mid * 1e4
        print(f"[PRE-FLIGHT] spot/fut basis = {basis_bps:+.1f} bps")
        if abs(basis_bps) > tolerance_bps:
            print(
                f"[PRE-FLIGHT][WARN] basis exceeds {tolerance_bps:.0f} bps tolerance "
                f"({abs(basis_bps):.1f} bps). This may indicate a data issue or "
                f"extreme market conditions."
            )
        else:
            print(f"[PRE-FLIGHT] basis within tolerance ({tolerance_bps:.0f} bps)")


# ==============================================================================
# Reset (for testing)
# ==============================================================================

def reset() -> None:
    """Reset all metrics state (useful for testing)."""
    global _QUEUE_MAX
    _last_msg_ts_ms.clear()
    _rows_window.clear()
    _rows_60s.clear()
    _total_rows.clear()
    _resyncs.clear()
    _reconnects.clear()
    _invalid_rows.clear()
    _crossed_rows.clear()
    _latency_ms.clear()
    _latest_mid.clear()
    _event_age_ms.clear()
    _pipe_ms.clear()
    _bad_exch_ts_count.clear()
    _skew_count.clear()
    _QUEUE_MAX = 0