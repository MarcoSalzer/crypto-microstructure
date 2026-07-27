# ==============================================================================
# Binance Trade Stream Adapter (spot + USDT-M futures)
#
# PURPOSE:
#   Connects to a single Binance WebSocket trade stream for one symbol
#   (e.g. BTCUSDT or ETHUSDT) and one market type (spot or fut).
#   Parses each raw trade message into a normalized row dict and pushes it
#   into a shared asyncio queue consumed by the unified pipeline.
#
# SYMBOL-PARAMETRIC DESIGN:
#   The same code handles any Binance symbol. The caller (collector.py)
#   spawns separate instances for BTC and ETH:
#     binance_trade_consumer("BTCUSDT", "spot", queue)
#     binance_trade_consumer("ETHUSDT", "fut", queue)
#
# DATA FLOW:
#   Binance WS -> parse JSON -> validate -> dedupe -> normalize -> queue.put()
#
# RECONNECT STRATEGY:
#   The consumer runs in an infinite while-True loop. On any connection loss,
#   parse error burst, or Binance error payload, it reconnects with exponential
#   backoff (1s → 30s max, with random jitter to desync parallel adapters).
#   The outer supervised_task wrapper in collector.py adds crash recovery.
#
# KEY DESIGN DECISIONS:
#   1. RECEIVE-TIME AS GROUND TRUTH (ts_ms):
#      We capture wall-clock time the instant each WS frame arrives.
#      This is the primary alignment axis for downstream feature engineering.
#      Exchange timestamp (exch_ts_ms) is kept for latency diagnostics only.
#
#   2. ROLLING DEDUPE WINDOW:
#      Binance can replay trades during reconnects. We maintain a sliding
#      window of the last 4096 trade IDs. Any ID already seen is dropped.
#      Implementation: set + deque for O(1) lookup with bounded memory.
#      IMPORTANT: We do NOT use deque(maxlen=N) because its auto-eviction
#      doesn't update the companion set, causing silent memory growth.
#
#   3. OUT-OF-ORDER: LOG ONLY, NEVER DROP:
#      Trade IDs occasionally arrive non-monotonically (network jitter,
#      Binance internal sharding). We log these for diagnostics but never
#      drop or reconnect — OOO trades are still valid data.
#
#   4. UNKNOWN SIDE -> DROP:
#      If the "m" (is_buyer_maker) flag is missing or unrecognizable,
#      we cannot determine aggressor side. These trades are useless for
#      downstream features (trade flow, VPIN, etc.) and are dropped.
#
#   5. RECONNECT_FLAG:
#      First row after each reconnect carries reconnect_flag=1 so downstream
#      consumers can detect gaps and handle feature warm-up accordingly.
#
# ==============================================================================

from __future__ import annotations

import asyncio
import math
import random
from collections import deque
from datetime import datetime, timezone
from typing import Any, Optional, Tuple

import orjson
import websockets

from collection import metrics

# ==============================================================================
# Binance WebSocket Endpoints
# ==============================================================================
# Spot and futures use different base URLs. The stream name (e.g. "btcusdt@trade")
# is appended as a path segment.

SPOT_WS = "wss://stream.binance.com:9443/ws"
FUT_WS  = "wss://fstream.binance.com/ws"

# ==============================================================================
# Logging and Deduplication Constants
# ==============================================================================

# Rate-limit noisy log messages: only print every Nth drop
LOG_EVERY_N_DROPS = 500

# Rate-limit out-of-order log messages
OOO_LOG_INTERVAL = 1000

# Rolling window size for duplicate detection.
# 4096 covers ~10-30 seconds of BTC trades at peak throughput,
# which is more than enough to catch reconnect replays.
_DEDUPE_WINDOW = 4096

# ==============================================================================
# Timestamp Validation
# ==============================================================================
# Any timestamp before 2000-01-01 is treated as invalid/corrupt.
# This catches zero values, negative values, and obvious garbage.

MIN_VALID_TS_MS = 946684800000

# ==============================================================================
# Venue Identity Constants
# ==============================================================================
# These are injected into every output row for downstream routing and labeling.

VENUE = "binance"
VENUE_SCOPE = "Binance"
MARKET_SCOPE_CANON = {"spot": "Spot", "fut": "Futures"}


# ==============================================================================
# Utility Functions
# ==============================================================================

def _now_ms() -> int:
    """Current UTC time in milliseconds. Used as receive-time ground truth."""
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def _safe_parse_float(v: Any) -> Optional[float]:
    """
    Parse a value to float, returning None if invalid.
    Rejects NaN, Inf, zero, and negative values — a valid trade price/qty
    must be a finite positive number.
    """
    try:
        f = float(v)
        if math.isfinite(f) and f > 0:
            return f
        return None
    except (ValueError, TypeError):
        return None


def _safe_parse_int(v: Any, min_val: int = 0) -> Optional[int]:
    """Parse a value to int, returning None if below min_val or unparseable."""
    try:
        i = int(v)
        if i >= min_val:
            return i
        return None
    except (ValueError, TypeError):
        return None


def _safe_parse_trade_id_int(v: Any) -> Optional[int]:
    """
    Try to parse trade ID as integer for OOO detection.
    Returns None if the ID is non-numeric (some venues use string IDs).
    Binance trade IDs are always numeric integers.
    """
    if v is None:
        return None
    try:
        return int(v)
    except (ValueError, TypeError):
        return None


# ==============================================================================
# Aggressor Side Detection
# ==============================================================================

def _aggressor_side_from_is_buyer_maker(m_flag: Any) -> Tuple[str, str]:
    """
    Determine aggressor (taker) side from Binance's is_buyer_maker flag.

    Binance convention:
      m=True  -> the buyer was the maker -> the taker was SELLING
      m=False -> the buyer was the taker -> the taker was BUYING

    Returns:
      (side, source) — e.g. ("buy", "is_buyer_maker") or ("unknown", "unknown")
    """
    if m_flag is True:
        return "sell", "is_buyer_maker"
    if m_flag is False:
        return "buy", "is_buyer_maker"
    return "unknown", "unknown"


# ==============================================================================
# Main Consumer Coroutine
# ==============================================================================

async def binance_trade_consumer(symbol: str, market_type: str, queue: asyncio.Queue):
    """
    Connect to Binance trade stream and emit normalized rows into queue.

    Args:
        symbol:      Binance symbol like "BTCUSDT" or "ETHUSDT"
        market_type: "spot" for spot market, "fut" for USDT-M perpetual futures
        queue:       Shared asyncio.Queue consumed by the pipeline router

    Output row schema:
        ts_ms            int    — receive-time (wall clock, ground truth for alignment)
        exch_ts_ms       int    — exchange timestamp (for latency diagnostics)
        venue            str    — "binance"
        market_type      str    — "spot" or "fut"
        symbol           str    — "BTCUSDT" / "ETHUSDT"
        trade_id         str    — unique trade identifier
        price            float  — trade price
        qty              float  — trade quantity in base asset
        side             str    — "buy" or "sell" (aggressor/taker side)
        reconnect_flag   int    — 1 on first row after reconnect, else 0
        venue_scope      str    — "Binance" (display name)
        market_scope     str    — "Spot" or "Futures" (display name)
        side_src         str    — how side was determined ("is_buyer_maker")
    """
    if market_type not in ("spot", "fut"):
        raise ValueError(f"Invalid market_type: {market_type}. Must be 'spot' or 'fut'.")

    # Build WebSocket URL: e.g. wss://stream.binance.com:9443/ws/btcusdt@trade
    stream = f"{symbol.lower()}@trade"
    url = f"{SPOT_WS}/{stream}" if market_type == "spot" else f"{FUT_WS}/{stream}"
    sym_upper = symbol.upper()
    market_scope = MARKET_SCOPE_CANON[market_type]

    # Cumulative counter for all types of invalid/dropped messages (persists across reconnects)
    invalid_drop_count: int = 0
    backoff: float = 1.0

    # --------------------------------------------------------------------------
    # Outer reconnect loop: runs forever, reconnects on any failure
    # --------------------------------------------------------------------------
    while True:
        # reconnect_flag=1 marks the first row after each new connection
        reconnect_flag: int = 1
        reconnect_counted: bool = False

        # Out-of-order tracking (LOG ONLY — never causes drops or reconnects)
        last_trade_id_int: int = 0
        ooo_count: int = 0

        # Rolling dedupe window: set for O(1) lookup, deque for FIFO eviction.
        # We manually manage eviction instead of using deque(maxlen=N) because
        # maxlen auto-eviction doesn't remove the evicted ID from seen_ids.
        seen_ids = set()
        seen_fifo = deque()

        try:
            print(f"[BINANCE TRADES] {market_type} {sym_upper} connect attempt")
            async with websockets.connect(
                url,
                max_size=None,       # no frame size limit (Binance can send large frames)
                ping_interval=20,    # send ping every 20s to keep connection alive
                ping_timeout=20,     # wait up to 20s for pong response
                close_timeout=2,     # don't hang on close
                open_timeout=10,     # explicit handshake timeout
            ) as ws:
                print(f"[BINANCE TRADES] {market_type} {sym_upper} connected")
                backoff = 1.0  # reset backoff on successful connect

                should_reconnect = False

                # ------------------------------------------------------------------
                # Inner message loop: process each incoming WS frame
                # ------------------------------------------------------------------
                async for raw in ws:

                    # ---- Step 1: Capture receive-time immediately ----
                    # This is our ground truth for wall-clock alignment.
                    # All downstream time-series features align on ts_ms.
                    ts_ms = _now_ms()
                    if ts_ms <= 0:
                        invalid_drop_count += 1
                        metrics.inc_invalid_rows(VENUE, market_type)
                        continue

                    # ---- Step 2: Parse JSON ----
                    try:
                        d = orjson.loads(raw)
                    except Exception as parse_err:
                        invalid_drop_count += 1
                        metrics.inc_invalid_rows(VENUE, market_type)
                        if invalid_drop_count % LOG_EVERY_N_DROPS == 1:
                            print(
                                f"[BINANCE TRADES] {market_type} {sym_upper} "
                                f"parse error #{invalid_drop_count}: {parse_err}"
                            )
                        continue

                    # ---- Step 3: Check for Binance error payload ----
                    # Binance sends {"e":"error", ...} for stream-level errors
                    if d.get("e") == "error":
                        print(f"[BINANCE TRADES] {market_type} {sym_upper} error payload: {d}")
                        should_reconnect = True
                        break

                    # ---- Step 4: Extract trade ID for deduplication ----
                    raw_trade_id = d.get("t")
                    trade_id_str = str(raw_trade_id) if raw_trade_id is not None else ""

                    # ---- Step 5: Rolling duplicate detection ----
                    # Binance can replay recent trades during reconnects.
                    # We check against a sliding window of recent trade IDs.
                    if trade_id_str:
                        if trade_id_str in seen_ids:
                            invalid_drop_count += 1
                            metrics.inc_invalid_rows(VENUE, market_type)
                            if invalid_drop_count % LOG_EVERY_N_DROPS == 1:
                                print(
                                    f"[BINANCE TRADES] {market_type} {sym_upper} "
                                    f"duplicate trade_id drop #{invalid_drop_count}: {trade_id_str}"
                                )
                            continue
                        # Add to window and evict oldest if over capacity
                        seen_ids.add(trade_id_str)
                        seen_fifo.append(trade_id_str)
                        if len(seen_fifo) > _DEDUPE_WINDOW:
                            old = seen_fifo.popleft()
                            seen_ids.discard(old)

                    # ---- Step 6: Out-of-order detection (LOG ONLY) ----
                    # Trade IDs should generally be monotonically increasing.
                    # OOO events are rare but valid — we log them for diagnostics
                    # but never drop the trade or trigger a reconnect.
                    tid_int = _safe_parse_trade_id_int(raw_trade_id)
                    if tid_int is not None:
                        if last_trade_id_int > 0 and tid_int < last_trade_id_int:
                            ooo_count += 1
                            if ooo_count == 1 or (ooo_count % OOO_LOG_INTERVAL == 0):
                                print(
                                    f"[BINANCE TRADES] {market_type} {sym_upper} "
                                    f"out-of-order (NOT dropping): tid={tid_int} last={last_trade_id_int} "
                                    f"total_ooo={ooo_count}"
                                )
                        last_trade_id_int = max(last_trade_id_int, tid_int)

                    # ---- Step 7: Exchange timestamp with sanity guard ----
                    # Binance provides "T" (trade time in ms). If it's missing or
                    # corrupt (< year 2000), we fall back to our receive-time.
                    # We NEVER drop a trade just because the exchange timestamp is bad.
                    exch_ts_raw = _safe_parse_int(d.get("T"), min_val=0)
                    has_real_exch_ts = exch_ts_raw is not None and exch_ts_raw >= MIN_VALID_TS_MS
                    exch_ts_ms = exch_ts_raw if has_real_exch_ts else ts_ms

                    # Track event_age (exchange → receive).
                    # Case 1: "T" is a real exchange timestamp → measure event_age.
                    # Case 2: "T" missing/bad → count bad_ts, skip age sample.
                    raw_age = ts_ms - exch_ts_ms
                    _age_skew = raw_age < 0
                    _event_age = max(raw_age, 0)
                    metrics.note_event_age(
                        VENUE, market_type, "trades",
                        _event_age,
                        bad_ts=not has_real_exch_ts,
                        skew=_age_skew,
                    )

                    # ---- Step 8: Parse and validate price ----
                    price = _safe_parse_float(d.get("p"))
                    if price is None:
                        invalid_drop_count += 1
                        metrics.inc_invalid_rows(VENUE, market_type)
                        continue

                    # ---- Step 9: Parse and validate quantity ----
                    qty = _safe_parse_float(d.get("q"))
                    if qty is None:
                        invalid_drop_count += 1
                        metrics.inc_invalid_rows(VENUE, market_type)
                        continue

                    # ---- Step 10: Determine aggressor side ----
                    # Binance uses "m" (is_buyer_maker) to indicate trade direction.
                    # If we can't determine the side, the trade is useless for
                    # downstream features like trade flow imbalance and VPIN.
                    side, side_src = _aggressor_side_from_is_buyer_maker(d.get("m"))
                    if side == "unknown":
                        invalid_drop_count += 1
                        metrics.inc_invalid_rows(VENUE, market_type)
                        continue

                    # ---- Step 11: Build trade ID string ----
                    # Prefer Binance's native trade ID. If absent (shouldn't happen),
                    # construct a deterministic fallback from trade fields.
                    if raw_trade_id is not None:
                        trade_id = str(raw_trade_id)
                    else:
                        trade_id = f"{VENUE}:{market_type}:{sym_upper}:{exch_ts_ms}:{price}:{qty}:{side}:0"
                    if not trade_id:
                        invalid_drop_count += 1
                        metrics.inc_invalid_rows(VENUE, market_type)
                        continue

                    # ---- Step 12: Build normalized output row and enqueue ----
                    row = {
                        "ts_ms": ts_ms,
                        "exch_ts_ms": exch_ts_ms,
                        "venue": VENUE,
                        "market_type": market_type,
                        "symbol": sym_upper,
                        "trade_id": trade_id,
                        "price": price,
                        "qty": qty,
                        "side": side,
                        "reconnect_flag": reconnect_flag,
                        "venue_scope": VENUE_SCOPE,
                        "market_scope": market_scope,
                        "side_src": side_src,
                    }

                    await queue.put(row)

                    # After the first successful row, clear reconnect flag
                    reconnect_flag = 0
                    metrics.inc_rows(VENUE, market_type)
                    metrics.set_last_msg(VENUE, market_type, ts_ms)

                # If we broke out of the message loop due to an error payload,
                # count the reconnect for metrics
                if should_reconnect and not reconnect_counted:
                    metrics.inc_reconnect(VENUE, market_type)
                    reconnect_counted = True

        # ------------------------------------------------------------------
        # Connection-level error handling
        # ------------------------------------------------------------------
        except websockets.ConnectionClosed as ce:
            print(
                f"[BINANCE TRADES] {market_type} {sym_upper} "
                f"connection closed: code={getattr(ce, 'code', None)}"
            )
            if not reconnect_counted:
                metrics.inc_reconnect(VENUE, market_type)
                reconnect_counted = True

        except Exception as e:
            print(f"[BINANCE TRADES] {market_type} {sym_upper} error: {e}")
            if not reconnect_counted:
                metrics.inc_reconnect(VENUE, market_type)
                reconnect_counted = True

        # Exponential backoff before reconnect (jittered, per-adapter)
        sleep_time = min(backoff, 30) + random.uniform(0, 0.5)
        print(f"[BINANCE TRADES] {market_type} {sym_upper} reconnecting in {sleep_time:.1f}s")
        await asyncio.sleep(sleep_time)
        backoff = min(backoff * 2, 30)