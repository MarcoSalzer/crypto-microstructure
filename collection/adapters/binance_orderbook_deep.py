# ==============================================================================
# Binance Deep L2 Orderbook Adapter (spot + USDT-M futures)
#
# PURPOSE:
#   Maintains a local copy of the full Binance orderbook for one symbol
#   (e.g. BTCUSDT, ETHUSDT) by combining a REST snapshot with incremental
#   WebSocket diff updates. Emits normalized orderbook snapshots at a fixed
#   interval (default 100ms) into a shared asyncio queue.
#
# SYMBOL-PARAMETRIC DESIGN:
#   Same code handles any Binance symbol. The caller (collector.py)
#   spawns separate instances:
#     binance_deep_l2_consumer("BTCUSDT", "spot", queue, depth_limit=1000)
#     binance_deep_l2_consumer("ETHUSDT", "fut",  queue, depth_limit=1000)
#
# BOOK RECONSTRUCTION ALGORITHM:
#   Binance uses a sequence-ID-based diffing protocol. The process is:
#   1. Connect to the WebSocket depth stream
#   2. Buffer incoming diff events during connection setup
#   3. Fetch a REST snapshot (provides lastUpdateId + full book state)
#   4. Find the "linking event" — the first diff whose sequence range
#      overlaps with the snapshot's lastUpdateId
#   5. Apply that diff and all subsequent diffs to maintain the book
#   6. If a sequence gap is detected, the book is corrupt — resync
#
#   Spot and futures use slightly different linking/continuity rules
#   (documented in _links_spot/_links_fut and _contig_spot/_contig_fut).
#
# TWO-TASK ARCHITECTURE (receiver + emitter):
#   After successful linking, the adapter splits into two concurrent tasks:
#
#   RECEIVER TASK:
#     - Reads raw WS frames as fast as they arrive
#     - Applies diffs to the in-memory book (bids/asks dicts)
#     - Validates sequence continuity (gaps -> resync)
#     - Updates exch_ts_ms and last_data_recv_ms timestamps
#
#   EMITTER TASK:
#     - Wakes up every emit_every_ms (default 100ms)
#     - Extracts sorted top-N levels from the book
#     - Validates: non-empty sides, non-crossed book
#     - Builds normalized row dict and pushes to queue
#     - Runs three watchdog checks (detailed below)
#
# WATCHDOG / RELIABILITY MECHANISMS:
#
#   1. NO-DATA WATCHDOG (NO_DATA_RECONNECT_MS = 8000ms):
#      If no valid depth update arrives for 8 seconds, the WebSocket
#      connection is likely dead or stuck. Triggers reconnect.
#
#   2. VALIDITY GATE (VALIDITY_GATE_MS = 3000ms):
#      Monitors the output side: if we haven't produced a valid emission
#      for 3 seconds (because the book is empty-sided or crossed), something
#      is wrong with the book state. Triggers resync.
#      This catches the "alive but silent" failure mode where raw data
#      keeps arriving but the book is corrupt after a partial reconnect.
#
#   3. STALENESS GUARD (MAX_STALENESS_MS = 2000ms):
#      If the exchange timestamp lags more than 2s behind our emit time,
#      it's stale — we clamp exch_ts_ms to emit time. This prevents
#      downstream latency calculations from being poisoned by frozen
#      timestamps during slow recovery periods.
#
# OUTPUT ROW SCHEMA:
#   ts_ms          int         — emission time (wall clock, ground truth)
#   exch_ts_ms     int         — exchange event time (guarded, for latency)
#   venue          str         — "binance"
#   market_type    str         — "spot" or "fut"
#   symbol         str         — "BTCUSDT" / "ETHUSDT"
#   seq            int         — last applied update ID (sequence number)
#   depth_target   int         — requested depth limit
#   depth_actual   int         — min(len(bids), len(asks)) actually available
#   best_bid       float       — highest bid price
#   best_ask       float       — lowest ask price
#   bids_px        list[float] — bid prices, descending
#   bids_qty       list[float] — bid quantities, matching bids_px order
#   asks_px        list[float] — ask prices, ascending
#   asks_qty       list[float] — ask quantities, matching asks_px order
#   reconnect_flag int         — 1 on first emission after reconnect, else 0
#
# ==============================================================================

from __future__ import annotations

import asyncio
import math
import random
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import aiohttp
import orjson
import websockets

from collection import metrics

# ==============================================================================
# Binance API Endpoints
# ==============================================================================
# REST endpoints for fetching orderbook snapshots (different for spot vs futures).
# WS endpoints for streaming incremental depth updates.

SPOT_REST = "https://api.binance.com/api/v3/depth"
FUT_REST  = "https://fapi.binance.com/fapi/v1/depth"
SPOT_WS   = "wss://stream.binance.com:9443/ws"
FUT_WS    = "wss://fstream.binance.com/ws"

# ==============================================================================
# Reliability Constants
# ==============================================================================

# Rate-limit noisy log messages: only print every Nth invalid row
LOG_EVERY_N_DROPS = 500

# If exch_ts_ms lags behind emit time by more than this, clamp it.
# Prevents stale timestamps from poisoning latency metrics.
MAX_STALENESS_MS = 2000

# If no valid WS depth event arrives for this long, the connection is dead.
# Triggers full reconnect + resync.
NO_DATA_RECONNECT_MS = 8000

# If no valid orderbook emission (non-empty, non-crossed) for this long,
# the book state is corrupt. Triggers resync even though raw data may still flow.
# This is the "validity gate" that catches "alive but silent" failures.
VALIDITY_GATE_MS = 3000

# ==============================================================================
# Timestamp Validation
# ==============================================================================
# Any timestamp before 2000-01-01 is treated as invalid/corrupt.

MIN_VALID_TS_MS = 946684800000


# ==============================================================================
# Utility Functions
# ==============================================================================

def _now_ms() -> int:
    """Current UTC time in milliseconds."""
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def _is_valid_level(price: float, qty: float) -> bool:
    """
    Validate a single orderbook level.
    Both price and quantity must be finite positive numbers.
    Rejects NaN, Inf, zero, and negative values.
    """
    return math.isfinite(price) and math.isfinite(qty) and price > 0 and qty > 0


def _levels_from_map(book: Dict[float, float], reverse: bool, limit: int) -> List[Tuple[float, float]]:
    """
    Extract sorted price levels from the book dictionary.

    Args:
        book:    {price: qty} dictionary
        reverse: True for bids (descending by price), False for asks (ascending)
        limit:   maximum number of levels to return

    Returns:
        List of (price, qty) tuples, sorted and capped at limit.
        Invalid levels (NaN, zero, negative) are filtered out.
    """
    items = [(p, q) for p, q in book.items() if _is_valid_level(p, q)]
    items.sort(key=lambda x: x[0], reverse=reverse)
    return items[:limit]


# ==============================================================================
# REST Snapshot Fetcher
# ==============================================================================

async def _fetch_snapshot(
    session: aiohttp.ClientSession,
    symbol: str,
    market_type: str,
    limit: int,
) -> Dict[str, Any]:
    """
    Fetch full orderbook snapshot from Binance REST API.

    Returns dict with keys: lastUpdateId, bids, asks.
    Bids/asks are lists of [price_str, qty_str] pairs.
    """
    url = SPOT_REST if market_type == "spot" else FUT_REST
    params = {"symbol": symbol.upper(), "limit": str(limit)}
    async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as r:
        r.raise_for_status()
        return await r.json()


# ==============================================================================
# Book Update Application
# ==============================================================================

def _apply_diff(book: Dict[float, float], updates: List[List[str]]) -> None:
    """
    Apply incremental diff updates to the book.

    Each update is [price_str, qty_str]:
      - qty = 0  -> remove the price level
      - qty > 0  -> set/overwrite the level (absolute, not delta)

    Invalid entries (non-numeric, negative) are silently skipped.
    """
    for entry in updates:
        try:
            p = float(entry[0])
            q = float(entry[1])
            if q == 0.0:
                book.pop(p, None)
            elif _is_valid_level(p, q):
                book[p] = q
        except (ValueError, TypeError, IndexError):
            continue


# ==============================================================================
# Exchange Timestamp Extraction
# ==============================================================================

def _extract_event_ts(e: Dict[str, Any], fallback_ts: int) -> int:
    """
    Extract exchange event timestamp from a Binance depth update.

    Binance provides "E" (event time) as epoch milliseconds.
    If missing or invalid (< year 2000), fall back to the provided timestamp.
    """
    try:
        ts = int(e.get("E", 0))
        if ts >= MIN_VALID_TS_MS:
            return ts
    except (ValueError, TypeError):
        pass
    return int(fallback_ts)


# ==============================================================================
# Sequence Linking and Continuity Checks
# ==============================================================================
# These functions implement Binance's orderbook diff protocol.
#
# Each depth update carries:
#   U = first update ID in this event
#   u = last update ID in this event
#   pu = previous event's u (futures only, optional on spot)
#
# LINKING: Finding the first diff that bridges the snapshot's lastUpdateId.
#   This diff "links" the REST snapshot to the WS stream so all subsequent
#   diffs can be applied continuously.
#
# CONTINUITY: Verifying each subsequent diff follows without gaps.
#   If a gap is detected, the book is corrupt and must be resynced.
#
# Spot and futures have slightly different rules:

def _links_spot(e: Dict[str, Any], last_id: int) -> bool:
    """
    Check if event `e` is the linking event for a SPOT snapshot.
    Condition: U <= lastUpdateId+1 <= u (the event straddles the snapshot ID).
    If pu (previous update ID) is present, it must equal lastUpdateId.
    """
    U = int(e.get("U", -1))
    u = int(e.get("u", -1))
    if U <= last_id + 1 <= u:
        pu = e.get("pu")
        return (pu is None) or (int(pu) == last_id)
    return False


def _contig_spot(e: Dict[str, Any], last_id: int) -> bool:
    """
    Check if event `e` is contiguous with the current book state for SPOT.
    Either U == lastUpdateId+1, or pu explicitly equals lastUpdateId.
    """
    U = int(e.get("U", -1))
    pu = e.get("pu")
    return (U == last_id + 1) or (pu is not None and int(pu) == last_id)


def _links_fut(e: Dict[str, Any], last_id: int) -> bool:
    """
    Check if event `e` is the linking event for a FUTURES snapshot.
    Slightly more permissive: U <= lastUpdateId <= u (not +1).
    Also accepts pu == lastUpdateId as a valid link.
    """
    U = int(e.get("U", -1))
    u = int(e.get("u", -1))
    pu = e.get("pu")
    return (U <= last_id <= u) or (pu is not None and int(pu) == last_id)


def _contig_fut(e: Dict[str, Any], last_id: int) -> bool:
    """
    Check continuity for FUTURES.
    Accepts: U == lastUpdateId+1, or pu == lastUpdateId,
    or U <= lastUpdateId+1 <= u (range overlap).
    """
    U = int(e.get("U", -1))
    u = int(e.get("u", -1))
    pu = e.get("pu")
    if (U == last_id + 1) or (pu is not None and int(pu) == last_id):
        return True
    return U <= (last_id + 1) <= u


# ==============================================================================
# Frame Parsing Helpers
# ==============================================================================

def _unwrap(d: Any) -> Any:
    """
    Unwrap combined-stream envelope if present.
    Some Binance infrastructure wraps payloads as {"stream": "...", "data": {...}}.
    This extracts the inner "data" dict; if not wrapped, returns as-is.
    """
    if isinstance(d, dict) and "data" in d and isinstance(d["data"], dict):
        return d["data"]
    return d


def _looks_like_depthupdate(d: Any) -> bool:
    """
    Quick check if a parsed frame looks like a depth update event.
    Must be a dict with "U" and "u" keys (update ID range).
    If "e" (event type) is present, it must be "depthUpdate".
    """
    if not isinstance(d, dict):
        return False
    d2 = _unwrap(d)
    if not isinstance(d2, dict):
        return False
    if d2.get("e") and d2.get("e") != "depthUpdate":
        return False
    return ("U" in d2) and ("u" in d2)


def _dbg_frame_preview(x: Any) -> str:
    """Short debug preview of a frame for troubleshooting link failures."""
    try:
        if isinstance(x, dict):
            keys = list(x.keys())[:12]
            return f"dict keys={keys}"
        if isinstance(x, list):
            return f"list len={len(x)}"
        s = str(x)
        return s[:160]
    except Exception:
        return "<unprintable>"


# ==============================================================================
# Main Consumer Coroutine
# ==============================================================================

async def binance_deep_l2_consumer(
    symbol: str,
    market_type: str,
    queue: asyncio.Queue,
    depth_limit: int = 1000,
    emit_every_ms: int = 100,
    ws_interval_ms: int = 100,
):
    """
    Maintain Binance deep L2 orderbook and emit snapshots into queue.

    Args:
        symbol:          Binance symbol like "BTCUSDT" or "ETHUSDT"
        market_type:     "spot" or "fut"
        queue:           Output queue for normalized orderbook rows
        depth_limit:     Max price levels per side (default 1000)
        emit_every_ms:   Emission interval in ms (default 100 = 10 snapshots/sec)
        ws_interval_ms:  Binance WS update frequency (100 or 250ms)
    """
    if market_type not in ("spot", "fut"):
        raise ValueError(f"Invalid market_type: {market_type}. Must be 'spot' or 'fut'.")

    # Build WS URL: e.g. wss://stream.binance.com:9443/ws/btcusdt@depth@100ms
    stream = f"{symbol.lower()}@depth@{ws_interval_ms}ms"
    ws_url = f"{SPOT_WS}/{stream}" if market_type == "spot" else f"{FUT_WS}/{stream}"
    sym_upper = symbol.upper()

    invalid_drop_count: int = 0
    backoff: float = 1.0

    # ==========================================================================
    # Outer reconnect loop: runs forever, full resync on any failure
    # ==========================================================================
    while True:
        bids: Dict[float, float] = {}
        asks: Dict[float, float] = {}
        last_update_id: int = 0
        reconnect_flag: int = 1
        reconnect_counted: bool = False

        try:
            async with aiohttp.ClientSession() as session:

                # ==============================================================
                # SPOT PATH: buffer-first strategy
                # ==============================================================
                # Binance spot docs require: "Buffer the events you receive
                # from the stream. Get a depth snapshot. Drop any event where
                # u is <= lastUpdateId in the snapshot."
                #
                # We open the WS first, buffer ~0.5s of events, then fetch the
                # REST snapshot, then find the linking event in the buffer.

                if market_type == "spot":
                    print(f"[BINANCE DEEP] spot {sym_upper} connect attempt")
                    async with websockets.connect(
                        ws_url,
                        max_size=None,
                        ping_interval=20,
                        ping_timeout=20,
                        close_timeout=2,
                        open_timeout=10,
                    ) as ws:
                        print(f"[BINANCE DEEP] spot {sym_upper} connected (buffering)")

                        # -- Phase 1: Pre-buffer WS events --
                        # Collect depth updates for ~0.5s while WS is fresh.
                        # These may contain the linking event we need.
                        prebuf: List[Dict[str, Any]] = []
                        end_t = asyncio.get_event_loop().time() + 0.5
                        while asyncio.get_event_loop().time() < end_t:
                            try:
                                raw = await asyncio.wait_for(ws.recv(), timeout=0.05)
                                d = orjson.loads(raw)
                                if _looks_like_depthupdate(d):
                                    prebuf.append(_unwrap(d))
                            except asyncio.TimeoutError:
                                pass
                            except Exception:
                                pass

                        # -- Phase 2: REST snapshot --
                        # Fetch the full book state. This gives us lastUpdateId
                        # which we need to link the buffered WS events.
                        snap = await _fetch_snapshot(session, symbol, market_type, depth_limit)
                        last_update_id = int(snap["lastUpdateId"])

                        bids.clear()
                        asks.clear()
                        for p, q in snap.get("bids", []):
                            bids[float(p)] = float(q)
                        for p, q in snap.get("asks", []):
                            asks[float(p)] = float(q)

                        print(
                            f"[BINANCE DEEP] spot {sym_upper} snapshot lastUpdateId={last_update_id} "
                            f"buffered={len(prebuf)}"
                        )

                        # -- Phase 3: Drop stale buffered events --
                        # Events with u <= lastUpdateId are already reflected in the snapshot.
                        prebuf = [e for e in prebuf if int(e.get("u", -1)) > last_update_id]

                        # -- Phase 4: Find the linking event in buffer --
                        # The linking event is the first diff whose update ID range
                        # straddles the snapshot's lastUpdateId.
                        linked = False
                        for e in prebuf:
                            if _links_spot(e, last_update_id):
                                _apply_diff(bids, e.get("b", []))
                                _apply_diff(asks, e.get("a", []))
                                last_update_id = int(e["u"])
                                linked = True
                                break

                        # -- Phase 5: If buffer didn't have it, try live stream --
                        # The linking event may not have arrived during our 0.5s buffer window.
                        # Wait up to 3 more seconds for it from the live stream.
                        if not linked:
                            deadline = asyncio.get_event_loop().time() + 3.0
                            while asyncio.get_event_loop().time() < deadline and not linked:
                                try:
                                    raw = await asyncio.wait_for(ws.recv(), timeout=0.2)
                                    d = orjson.loads(raw)
                                    if not _looks_like_depthupdate(d):
                                        continue
                                    e = _unwrap(d)
                                    if int(e.get("u", -1)) <= last_update_id:
                                        continue
                                    if _links_spot(e, last_update_id):
                                        _apply_diff(bids, e.get("b", []))
                                        _apply_diff(asks, e.get("a", []))
                                        last_update_id = int(e["u"])
                                        linked = True
                                except asyncio.TimeoutError:
                                    pass

                        if not linked:
                            sleep_time = min(backoff, 30) + random.uniform(0, 0.5)
                            print(f"[BINANCE DEEP] spot {sym_upper} link failed, reconnecting in {sleep_time:.1f}s")
                            metrics.inc_resync("binance", market_type)
                            await asyncio.sleep(sleep_time)
                            backoff = min(backoff * 2, 30)
                            continue

                        print(f"[BINANCE DEEP] spot {sym_upper} linked, streaming")
                        backoff = 1.0

                        # -- Phase 6: Concurrent receiver + emitter --
                        stop = asyncio.Event()
                        exch_ts_ms: int = _now_ms()
                        last_data_recv_ms: int = exch_ts_ms
                        last_valid_emit_ms: int = exch_ts_ms

                        async def receiver():
                            """Read WS frames, apply diffs, detect sequence gaps."""
                            nonlocal last_update_id, invalid_drop_count, exch_ts_ms, last_data_recv_ms
                            try:
                                async for raw in ws:
                                    recv_ts = _now_ms()
                                    try:
                                        d = orjson.loads(raw)
                                    except Exception:
                                        invalid_drop_count += 1
                                        continue

                                    if not _looks_like_depthupdate(d):
                                        continue
                                    e = _unwrap(d)

                                    exch_ts_ms = _extract_event_ts(e, recv_ts)
                                    last_data_recv_ms = recv_ts

                                    # Track event_age (exchange → receive) for spot depth.
                                    # "E" is the Binance event time field.
                                    _raw_e = e.get("E", 0)
                                    try:
                                        _has_real_exch = int(_raw_e) >= MIN_VALID_TS_MS
                                    except (ValueError, TypeError):
                                        _has_real_exch = False
                                    _raw_age = recv_ts - exch_ts_ms
                                    _age_skew = _raw_age < 0
                                    metrics.note_event_age(
                                        "binance", market_type, "depth",
                                        max(_raw_age, 0),
                                        bad_ts=not _has_real_exch,
                                        skew=_age_skew,
                                    )

                                    # Check sequence continuity
                                    if _contig_spot(e, last_update_id):
                                        _apply_diff(bids, e.get("b", []))
                                        _apply_diff(asks, e.get("a", []))
                                        last_update_id = int(e["u"])
                                    else:
                                        # Sequence gap = book is corrupt, must resync
                                        print(f"[BINANCE DEEP] spot {sym_upper} gap detected, resyncing")
                                        metrics.inc_resync("binance", market_type)
                                        stop.set()
                                        return
                            except websockets.ConnectionClosed:
                                stop.set()
                            except Exception as err:
                                print(f"[BINANCE DEEP] spot {sym_upper} receiver error: {err}")
                                stop.set()

                        async def emitter():
                            """Periodically extract, validate, and emit book snapshots."""
                            nonlocal reconnect_flag, invalid_drop_count, exch_ts_ms, last_valid_emit_ms
                            try:
                                interval = emit_every_ms / 1000.0
                                while not stop.is_set():
                                    await asyncio.sleep(interval)

                                    emit_ts = _now_ms()

                                    # --- Watchdog 1: No-data check ---
                                    # If no WS depth update for 5s, connection is dead
                                    if (emit_ts - last_data_recv_ms) > NO_DATA_RECONNECT_MS:
                                        print(
                                            f"[BINANCE DEEP] spot {sym_upper} no data for "
                                            f"{(emit_ts - last_data_recv_ms)}ms, reconnecting"
                                        )
                                        metrics.inc_resync("binance", market_type)
                                        stop.set()
                                        return

                                    # Extract sorted levels from book
                                    tb = _levels_from_map(bids, True, depth_limit)
                                    ta = _levels_from_map(asks, False, depth_limit)

                                    # --- Watchdog 2a: Empty side check ---
                                    if not tb or not ta:
                                        invalid_drop_count += 1
                                        metrics.inc_invalid_rows("binance", market_type)
                                        if invalid_drop_count % LOG_EVERY_N_DROPS == 1:
                                            print(f"[BINANCE DEEP] spot {sym_upper} empty side #{invalid_drop_count}")
                                        # Validity gate: if no valid emission for too long, resync
                                        if (emit_ts - last_valid_emit_ms) > VALIDITY_GATE_MS:
                                            print(f"[BINANCE DEEP] spot {sym_upper} no valid emit for {emit_ts - last_valid_emit_ms}ms, resyncing")
                                            metrics.inc_resync("binance", market_type)
                                            stop.set()
                                            return
                                        continue

                                    best_bid = tb[0][0]
                                    best_ask = ta[0][0]

                                    # --- Watchdog 2b: Crossed book check ---
                                    # best_bid >= best_ask means book state is inconsistent
                                    if best_bid >= best_ask:
                                        metrics.inc_crossed_rows("binance", market_type)
                                        if (emit_ts - last_valid_emit_ms) > VALIDITY_GATE_MS:
                                            print(f"[BINANCE DEEP] spot {sym_upper} crossed book for {emit_ts - last_valid_emit_ms}ms, resyncing")
                                            metrics.inc_resync("binance", market_type)
                                            stop.set()
                                            return
                                        continue

                                    depth_actual = min(len(tb), len(ta))

                                    # --- Pipe latency: receive → emit (spot) ---
                                    _pipe = emit_ts - last_data_recv_ms
                                    if 0 <= _pipe < 5000:
                                        metrics.note_pipe_ms("binance", market_type, "depth", _pipe)

                                    # --- Staleness guard ---
                                    # Clamp exch_ts if it lags too far behind emit time
                                    final_exch_ts = exch_ts_ms if exch_ts_ms >= MIN_VALID_TS_MS else emit_ts
                                    if (emit_ts - final_exch_ts) > MAX_STALENESS_MS:
                                        final_exch_ts = emit_ts

                                    # Build and emit normalized row
                                    row = {
                                        "ts_ms": emit_ts,
                                        "exch_ts_ms": final_exch_ts,
                                        "venue": "binance",
                                        "market_type": market_type,
                                        "symbol": sym_upper,
                                        "seq": last_update_id,
                                        "depth_target": depth_limit,
                                        "depth_actual": depth_actual,
                                        "best_bid": best_bid,
                                        "best_ask": best_ask,
                                        "bids_px": [p for p, _ in tb],
                                        "bids_qty": [q for _, q in tb],
                                        "asks_px": [p for p, _ in ta],
                                        "asks_qty": [q for _, q in ta],
                                        "reconnect_flag": reconnect_flag,
                                    }
                                    await queue.put(row)

                                    reconnect_flag = 0
                                    last_valid_emit_ms = emit_ts
                                    metrics.inc_rows("binance", market_type)
                                    metrics.set_last_msg("binance", market_type, row["ts_ms"])
                                    metrics.note_mid("binance", market_type, 0.5 * (best_bid + best_ask))

                                    lat = row["ts_ms"] - final_exch_ts
                                    if 0 <= lat < 60000:
                                        metrics.note_latency("binance", market_type, lat)

                            except asyncio.CancelledError:
                                pass

                        rt = asyncio.create_task(receiver())
                        et = asyncio.create_task(emitter())
                        await stop.wait()
                        rt.cancel()
                        et.cancel()
                        await asyncio.gather(rt, et, return_exceptions=True)

                # ==============================================================
                # FUTURES PATH: buffer-first (same strategy as spot)
                # ==============================================================
                # Futures use the same buffer-first approach but with different
                # sequence linking rules (_links_fut / _contig_fut).

                else:
                    print(f"[BINANCE DEEP] fut {sym_upper} connect attempt")
                    async with websockets.connect(
                        ws_url,
                        max_size=None,
                        ping_interval=20,
                        ping_timeout=20,
                        close_timeout=2,
                        open_timeout=10,
                    ) as ws:
                        print(f"[BINANCE DEEP] fut {sym_upper} connected (buffering)")

                        # -- Phase 1: Pre-buffer ~0.5s of WS frames --
                        prebuf: List[Dict[str, Any]] = []
                        end_t = asyncio.get_event_loop().time() + 0.5
                        while asyncio.get_event_loop().time() < end_t:
                            try:
                                raw = await asyncio.wait_for(ws.recv(), timeout=0.05)
                                d = orjson.loads(raw)
                                if _looks_like_depthupdate(d):
                                    prebuf.append(_unwrap(d))
                            except asyncio.TimeoutError:
                                pass
                            except Exception:
                                pass

                        # -- Phase 2: REST snapshot --
                        snap = await _fetch_snapshot(session, symbol, market_type, depth_limit)
                        last_update_id = int(snap["lastUpdateId"])

                        bids.clear()
                        asks.clear()
                        for p, q in snap.get("bids", []):
                            bids[float(p)] = float(q)
                        for p, q in snap.get("asks", []):
                            asks[float(p)] = float(q)

                        print(
                            f"[BINANCE DEEP] fut {sym_upper} snapshot lastUpdateId={last_update_id} "
                            f"buffered={len(prebuf)}"
                        )

                        # -- Phase 3: Drop stale buffered events --
                        prebuf = [e for e in prebuf if int(e.get("u", -1)) > last_update_id]

                        # -- Phase 4: Find linking event in buffer --
                        linked = False
                        for e in prebuf:
                            if _links_fut(e, last_update_id):
                                _apply_diff(bids, e.get("b", []))
                                _apply_diff(asks, e.get("a", []))
                                last_update_id = int(e["u"])
                                linked = True
                                break

                        # -- Phase 5: Try live stream if buffer didn't have it --
                        if not linked:
                            deadline = asyncio.get_event_loop().time() + 3.0
                            while asyncio.get_event_loop().time() < deadline and not linked:
                                try:
                                    raw = await asyncio.wait_for(ws.recv(), timeout=0.2)
                                    d = orjson.loads(raw)
                                    if not _looks_like_depthupdate(d):
                                        continue
                                    e = _unwrap(d)
                                    if int(e.get("u", -1)) <= last_update_id:
                                        continue
                                    if _links_fut(e, last_update_id):
                                        _apply_diff(bids, e.get("b", []))
                                        _apply_diff(asks, e.get("a", []))
                                        last_update_id = int(e["u"])
                                        linked = True
                                except asyncio.TimeoutError:
                                    pass

                        if not linked:
                            sleep_time = min(backoff, 30) + random.uniform(0, 0.5)
                            print(f"[BINANCE DEEP] fut {sym_upper} link failed, reconnecting in {sleep_time:.1f}s")
                            metrics.inc_resync("binance", market_type)
                            await asyncio.sleep(sleep_time)
                            backoff = min(backoff * 2, 30)
                            continue

                        print(f"[BINANCE DEEP] fut {sym_upper} linked, streaming")
                        backoff = 1.0

                        # -- Phase 6: Concurrent receiver + emitter --
                        stop = asyncio.Event()
                        exch_ts_ms: int = _now_ms()
                        last_data_recv_ms: int = exch_ts_ms
                        last_valid_emit_ms: int = exch_ts_ms

                        async def receiver_fut():
                            """Read WS frames, apply diffs, detect sequence gaps (futures rules)."""
                            nonlocal last_update_id, invalid_drop_count, exch_ts_ms, last_data_recv_ms
                            try:
                                async for raw in ws:
                                    recv_ts = _now_ms()
                                    try:
                                        d = orjson.loads(raw)
                                    except Exception:
                                        invalid_drop_count += 1
                                        continue

                                    if not _looks_like_depthupdate(d):
                                        continue

                                    e = _unwrap(d)

                                    exch_ts_ms = _extract_event_ts(e, recv_ts)
                                    last_data_recv_ms = recv_ts

                                    # Track event_age (exchange → receive) for futures depth.
                                    _raw_e_fut = e.get("E", 0)
                                    try:
                                        _has_real_exch_fut = int(_raw_e_fut) >= MIN_VALID_TS_MS
                                    except (ValueError, TypeError):
                                        _has_real_exch_fut = False
                                    _raw_age_fut = recv_ts - exch_ts_ms
                                    _age_skew_fut = _raw_age_fut < 0
                                    metrics.note_event_age(
                                        "binance", market_type, "depth",
                                        max(_raw_age_fut, 0),
                                        bad_ts=not _has_real_exch_fut,
                                        skew=_age_skew_fut,
                                    )

                                    if _contig_fut(e, last_update_id):
                                        _apply_diff(bids, e.get("b", []))
                                        _apply_diff(asks, e.get("a", []))
                                        last_update_id = int(e["u"])
                                    else:
                                        print(f"[BINANCE DEEP] fut {sym_upper} gap detected, resyncing")
                                        metrics.inc_resync("binance", market_type)
                                        stop.set()
                                        return
                            except websockets.ConnectionClosed:
                                stop.set()
                            except Exception as err:
                                print(f"[BINANCE DEEP] fut {sym_upper} receiver error: {err}")
                                stop.set()

                        async def emitter_fut():
                            """Periodically extract, validate, and emit book snapshots (futures)."""
                            nonlocal reconnect_flag, invalid_drop_count, exch_ts_ms, last_valid_emit_ms
                            try:
                                interval = emit_every_ms / 1000.0
                                while not stop.is_set():
                                    await asyncio.sleep(interval)

                                    emit_ts = _now_ms()

                                    # --- Watchdog 1: No-data check ---
                                    if (emit_ts - last_data_recv_ms) > NO_DATA_RECONNECT_MS:
                                        print(
                                            f"[BINANCE DEEP] fut {sym_upper} no data for "
                                            f"{(emit_ts - last_data_recv_ms)}ms, reconnecting"
                                        )
                                        metrics.inc_resync("binance", market_type)
                                        stop.set()
                                        return

                                    tb = _levels_from_map(bids, True, depth_limit)
                                    ta = _levels_from_map(asks, False, depth_limit)

                                    # --- Watchdog 2a: Empty side + validity gate ---
                                    if not tb or not ta:
                                        invalid_drop_count += 1
                                        metrics.inc_invalid_rows("binance", market_type)
                                        if invalid_drop_count % LOG_EVERY_N_DROPS == 1:
                                            print(f"[BINANCE DEEP] fut {sym_upper} empty side #{invalid_drop_count}")
                                        if (emit_ts - last_valid_emit_ms) > VALIDITY_GATE_MS:
                                            print(f"[BINANCE DEEP] fut {sym_upper} no valid emit for {emit_ts - last_valid_emit_ms}ms, resyncing")
                                            metrics.inc_resync("binance", market_type)
                                            stop.set()
                                            return
                                        continue

                                    best_bid = tb[0][0]
                                    best_ask = ta[0][0]

                                    # --- Watchdog 2b: Crossed book + validity gate ---
                                    if best_bid >= best_ask:
                                        metrics.inc_crossed_rows("binance", market_type)
                                        if (emit_ts - last_valid_emit_ms) > VALIDITY_GATE_MS:
                                            print(f"[BINANCE DEEP] fut {sym_upper} crossed book for {emit_ts - last_valid_emit_ms}ms, resyncing")
                                            metrics.inc_resync("binance", market_type)
                                            stop.set()
                                            return
                                        continue

                                    depth_actual = min(len(tb), len(ta))

                                    # --- Pipe latency: receive → emit (fut) ---
                                    _pipe_fut = emit_ts - last_data_recv_ms
                                    if 0 <= _pipe_fut < 5000:
                                        metrics.note_pipe_ms("binance", market_type, "depth", _pipe_fut)

                                    # --- Staleness guard ---
                                    final_exch_ts = exch_ts_ms if exch_ts_ms >= MIN_VALID_TS_MS else emit_ts
                                    if (emit_ts - final_exch_ts) > MAX_STALENESS_MS:
                                        final_exch_ts = emit_ts

                                    row = {
                                        "ts_ms": emit_ts,
                                        "exch_ts_ms": final_exch_ts,
                                        "venue": "binance",
                                        "market_type": market_type,
                                        "symbol": sym_upper,
                                        "seq": last_update_id,
                                        "depth_target": depth_limit,
                                        "depth_actual": depth_actual,
                                        "best_bid": best_bid,
                                        "best_ask": best_ask,
                                        "bids_px": [p for p, _ in tb],
                                        "bids_qty": [q for _, q in tb],
                                        "asks_px": [p for p, _ in ta],
                                        "asks_qty": [q for _, q in ta],
                                        "reconnect_flag": reconnect_flag,
                                    }
                                    await queue.put(row)

                                    reconnect_flag = 0
                                    last_valid_emit_ms = emit_ts
                                    metrics.inc_rows("binance", market_type)
                                    metrics.set_last_msg("binance", market_type, row["ts_ms"])
                                    metrics.note_mid("binance", market_type, 0.5 * (best_bid + best_ask))

                                    lat = row["ts_ms"] - final_exch_ts
                                    if 0 <= lat < 60000:
                                        metrics.note_latency("binance", market_type, lat)

                            except asyncio.CancelledError:
                                pass

                        rt = asyncio.create_task(receiver_fut())
                        et = asyncio.create_task(emitter_fut())
                        await stop.wait()
                        rt.cancel()
                        et.cancel()
                        await asyncio.gather(rt, et, return_exceptions=True)

        # ==================================================================
        # Top-level error handling: catch-all for unexpected failures
        # ==================================================================
        except Exception as e:
            print(f"[BINANCE DEEP] {market_type} {sym_upper} error: {e}")
            if not reconnect_counted:
                metrics.inc_reconnect("binance", market_type)
                reconnect_counted = True

        # Exponential backoff before reconnect (jittered, per-adapter)
        sleep_time = min(backoff, 30) + random.uniform(0, 0.5)
        print(f"[BINANCE DEEP] {market_type} {sym_upper} reconnecting in {sleep_time:.1f}s")
        await asyncio.sleep(sleep_time)
        backoff = min(backoff * 2, 30)