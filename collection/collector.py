# collection/collector.py
# ==============================================================================
# Unified Live Pipeline — Binance-only, Multi-Asset (BTC/ETH/BNB)
#
# PURPOSE:
#   Central orchestrator that connects all adapter tasks, routes data by
#   (asset, market_type), writes per-asset Parquet files, and serves a
#   live WebSocket dashboard. This is the single entry point for the
#   entire data collection system.
#
# ARCHITECTURE OVERVIEW:
#
#   ┌─────────────────────────────────────────────────────────────────┐
#   │                    ADAPTER LAYER (12 tasks: 3 assets x 2 markets x 2 streams)                      │
#   │                                                                 │
#   │  BTC spot trades ──┐                    ┌── BTC spot deep L2    │
#   │  BTC fut  trades ──┤  trades_main queue │── BTC fut  deep L2    │
#   │  ETH spot trades ──┤  ───────────────►  │── ETH spot deep L2    │
#   │  ETH fut  trades ──┘                    └── ETH fut  deep L2    │
#   │                          deep_main queue ───────────────►       │
#   └─────────────────┬───────────────────────────────┬───────────────┘
#                     │                               │
#   ┌─────────────────▼───────────────────────────────▼───────────────┐
#   │                    ROUTER LAYER (2 tasks)                       │
#   │                                                                 │
#   │  trades router: reads trades_main, injects alias fields,        │
#   │    routes to btc_spot / btc_fut / eth_spot / eth_fut queues     │
#   │                                                                 │
#   │  deep router: same logic for orderbook data                     │
#   │                                                                 │
#   │  Both routers also fan out to dashboard queues (non-blocking)   │
#   └─────────────────┬───────────────────────────────┬───────────────┘
#                     │                               │
#   ┌─────────────────▼───────────────────────────────▼───────────────┐
#   │                   WRITER LAYER (12 tasks: 3 assets x 2 markets x 2 streams)                        │
#   │                                                                 │
#   │  trades_btc_spot → trades_btc_spot_2026-02-16_14.parquet        │
#   │  trades_btc_fut  → trades_btc_fut_2026-02-16_14.parquet         │
#   │  trades_eth_spot → trades_eth_spot_2026-02-16_14.parquet        │
#   │  trades_eth_fut  → trades_eth_fut_2026-02-16_14.parquet         │
#   │  lobdeep_btc_spot → lobdeep_btc_spot_2026-02-16_14.parquet     │
#   │  lobdeep_btc_fut  → lobdeep_btc_fut_2026-02-16_14.parquet      │
#   │  lobdeep_eth_spot → lobdeep_eth_spot_2026-02-16_14.parquet     │
#   │  lobdeep_eth_fut  → lobdeep_eth_fut_2026-02-16_14.parquet      │
#   │                                                                 │
#   │  Each writer rotates files at hour boundaries and uses           │
#   │  atomic tmp→final rename for crash safety.                      │
#   └─────────────────────────────────────────────────────────────────┘
#
#   ┌─────────────────────────────────────────────────────────────────┐
#   │                  DASHBOARD LAYER (3 tasks)                      │
#   │                                                                 │
#   │  dashboard-trades: reads trade dashboard queue → server buffer  │
#   │  dashboard-deep:   reads deep dashboard queue  → server buffer  │
#   │  dashboard-server: WS server broadcasting to connected clients  │
#   └─────────────────────────────────────────────────────────────────┘
#
# ASSET CONFIGURATION:
#   All assets are defined in the ASSETS dict at the top of the file.
#   To add a new asset (e.g. SOL), just add one entry — the pipeline
#   auto-creates all necessary tasks, queues, and writers.
#
# ROUTING LOGIC:
#   The router extracts the base asset from each row's symbol field
#   (e.g. "BTCUSDT" → base="BTC") and combines it with market_type
#   to form a routing key like "btc_spot" or "eth_fut".
#   This key selects the destination queue and Parquet writer.
#
# CANONICALIZATION:
#   Every row passes through _inject_alias_fields() which adds:
#   - venue / venue_scope: always "Binance"
#   - market_scope: "Spot" or "Futures"
#   - base / quote: extracted from symbol (e.g. "BTC" / "USDT")
#   - symbol_canon: base asset name (e.g. "BTC")
#   - instrument_canon: standardized pair (e.g. "BTC-USDT")
#
# RELIABILITY:
#   - supervised_task wrapper: auto-restarts crashed adapter tasks
#   - Signal handling: SIGINT/SIGTERM trigger graceful shutdown,
#     SIGHUP is ignored (tmux-safe)
#   - Graceful shutdown: shielded flush + finalize of all Parquet writers
#   - Queue pressure warnings: logged when any queue exceeds 80% capacity
#
# ==============================================================================

import os
import signal
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List, Set, Tuple

import pyarrow as pa
import pyarrow.parquet as pq

try:
    from aiohttp import web as _aiohttp_web
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False
    print("[WARNING] aiohttp not installed - Prometheus HTTP endpoint disabled")

# ==============================================================================
# Project Root and Environment
# ==============================================================================
# REPO_ROOT points to the repository root (1 level above this file's dir).
# .env is loaded for optional configuration overrides.

REPO_ROOT = Path(__file__).resolve().parents[1]

try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except Exception:
    pass

# ==============================================================================
# Adapter Imports
# ==============================================================================
# Both adapters are symbol-parametric: same code handles BTCUSDT and ETHUSDT.
# The caller (main()) spawns separate instances per (asset, market_type).

from collection.adapters.binance_trades import binance_trade_consumer
from collection.adapters.binance_orderbook_deep import binance_deep_l2_consumer

from collection import metrics

# ==============================================================================
# Optional WebSocket Server for Dashboard
# ==============================================================================
# Dashboard is a nice-to-have; pipeline works without it.

try:
    import websockets  # noqa: F401
    from websockets.asyncio.server import serve as ws_serve
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False
    print("[WARNING] websockets not installed - dashboard server disabled")

# ==============================================================================
# Output Configuration
# ==============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Output directory for Parquet files. Override with RAW_OUTPUT_DIR env var.
RAW_OUTPUT_DIR = os.getenv("RAW_OUTPUT_DIR", "")
if RAW_OUTPUT_DIR.strip():
    OUTPUT_DIR = Path(RAW_OUTPUT_DIR).expanduser().resolve()
else:
    OUTPUT_DIR = PROJECT_ROOT / "data_storage" / "raw_data"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Dashboard WebSocket server settings
DASHBOARD_HOST = "0.0.0.0"
DASHBOARD_PORT = int(os.getenv("DASHBOARD_PORT", "8765"))
PROMETHEUS_PORT = int(os.getenv("PROMETHEUS_PORT", "8766"))

# Parquet compression (zstd is the best balance of speed + ratio)
PARQUET_COMP = os.getenv("PARQUET_COMP", "zstd")

# ==============================================================================
# Timestamp Validation
# ==============================================================================
# Any timestamp before 2000-01-01 is treated as invalid/corrupt.

MIN_VALID_TS_MS = 946684800000

# ==============================================================================
# Asset Configuration
# ==============================================================================
# Central config for all traded assets. Each entry defines:
#   - binance_symbol:     exact symbol string for Binance APIs
#   - depth_limit:        max orderbook levels per side
#   - fut_ws_interval_ms: WS update frequency for futures depth stream
#
# To add a new asset, just add an entry here. The pipeline auto-creates
# all tasks, queues, writers, and routing for the new asset.

ASSETS = {
    "BTC": {
        "binance_symbol": "BTCUSDT",
        "depth_limit": 1000,
        "fut_ws_interval_ms": 100,
    },
    "ETH": {
        "binance_symbol": "ETHUSDT",
        "depth_limit": 1000,
        "fut_ws_interval_ms": 100,
    },
    "BNB": {
        "binance_symbol": "BNBUSDT",
        "depth_limit": 1000,
        "fut_ws_interval_ms": 100,
    },
}

# Venue identity (always Binance in this pipeline)
VENUE = "Binance"
VENUE_LOWER = "binance"

# ==============================================================================
# Canonicalization Helpers
# ==============================================================================
# These functions normalize raw adapter output into a standardized schema
# with consistent venue names, market labels, and symbol decomposition.

MARKET_SCOPE_MAP = {
    "spot": "Spot",
    "fut": "Futures",
    "futures": "Futures",
}

# Ordered by specificity: USDT before USD to avoid "BTC" + "USD" + "T" mismatch
KNOWN_QUOTES = [
    "USDT", "USDC", "FDUSD", "TUSD", "BUSD",
    "USD", "EUR", "GBP", "JPY",
    "BTC", "ETH",
]


def _now_ms() -> int:
    """Current UTC time in milliseconds."""
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def _canon_market_scope(market_type: Any) -> str:
    """Convert market_type to display label: 'fut' -> 'Futures', 'spot' -> 'Spot'."""
    if market_type is None:
        return "Spot"
    s = str(market_type).strip().lower()
    return MARKET_SCOPE_MAP.get(s, "Spot")


def _split_symbol(symbol: Any) -> Tuple[str, str]:
    """
    Split a Binance symbol into (base, quote) components.

    Examples:
        "BTCUSDT" → ("BTC", "USDT")
        "ETHUSDT" → ("ETH", "USDT")
        "SOLUSDT" → ("SOL", "USDT")

    Uses suffix matching against KNOWN_QUOTES for robustness.
    Returns ("", "") if the symbol can't be decomposed.
    """
    if not symbol:
        return "", ""
    s = str(symbol).strip().upper()
    s = s.replace("-", "").replace("/", "").replace("_", "")
    for q in KNOWN_QUOTES:
        if s.endswith(q) and len(s) > len(q):
            return s[:-len(q)], q
    return "", ""


def _ensure_timestamps(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enforce valid timestamps on every row.

    Rules:
      - ts_ms (receive time) must be >= 2000-01-01; else set to now
      - exch_ts_ms (exchange time) must be >= 2000-01-01; else fallback to ts_ms

    We NEVER drop rows due to bad timestamps — we fix them.
    This ensures downstream features always have valid time axes.
    """
    ts = row.get("ts_ms")
    try:
        ts_i = int(ts)
    except Exception:
        ts_i = 0
    if ts_i < MIN_VALID_TS_MS:
        ts_i = _now_ms()
    row["ts_ms"] = ts_i

    ex = row.get("exch_ts_ms")
    try:
        ex_i = int(ex)
    except Exception:
        ex_i = 0
    if ex_i < MIN_VALID_TS_MS:
        ex_i = ts_i
    row["exch_ts_ms"] = ex_i
    return row


def _inject_alias_fields(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enrich a raw adapter row with canonical fields needed by downstream consumers.

    Added fields:
      - venue / venue_scope: "Binance"
      - market_scope: "Spot" or "Futures"
      - base: extracted base asset ("BTC", "ETH")
      - quote: extracted quote asset ("USDT")
      - symbol_canon: just the base asset name
      - instrument_canon: standardized pair ("BTC-USDT", "ETH-USDT")
    """
    row = _ensure_timestamps(row)

    # Venue is always Binance in this pipeline
    row["venue"] = VENUE
    row["venue_scope"] = VENUE
    row["market_scope"] = _canon_market_scope(row.get("market_type"))

    base, quote = _split_symbol(row.get("symbol"))
    if not row.get("base"):
        row["base"] = base
    if not row.get("quote"):
        row["quote"] = quote

    row["symbol_canon"] = base if base else row.get("symbol", "")

    if base and quote:
        row["instrument_canon"] = f"{base}-{quote}"
    else:
        row["instrument_canon"] = row.get("symbol", "")

    return row


# ==============================================================================
# Arrow Schemas
# ==============================================================================
# Define the exact column layout for Parquet files. Every row written to
# Parquet is filtered to only these columns. Extra adapter debug fields
# (like side_src, qty_src) are silently dropped.

TRADE_SCHEMA = pa.schema([
    ("ts_ms",          pa.int64()),      # receive-time (alignment ground truth)
    ("exch_ts_ms",     pa.int64()),      # exchange event time (latency diagnostics)
    ("venue",          pa.string()),     # "Binance"
    ("market_type",    pa.string()),     # "spot" or "fut"
    ("symbol",         pa.string()),     # "BTCUSDT" or "ETHUSDT"
    ("trade_id",       pa.string()),     # unique trade identifier
    ("price",          pa.float64()),    # trade price
    ("qty",            pa.float64()),    # trade quantity in base asset
    ("side",           pa.string()),     # "buy" or "sell" (aggressor/taker side)
    ("reconnect_flag", pa.int64()),      # 1 on first row after reconnect
    ("venue_scope",    pa.string()),     # "Binance" (display name)
    ("market_scope",   pa.string()),     # "Spot" or "Futures"
    ("base",           pa.string()),     # "BTC" or "ETH"
    ("quote",          pa.string()),     # "USDT"
    ("symbol_canon",   pa.string()),     # "BTC" or "ETH"
    ("instrument_canon", pa.string()),   # "BTC-USDT" or "ETH-USDT"
])

DEEP_SCHEMA = pa.schema([
    ("ts_ms",          pa.int64()),      # emission time (alignment ground truth)
    ("exch_ts_ms",     pa.int64()),      # exchange event time (latency diagnostics)
    ("venue",          pa.string()),     # "Binance"
    ("market_type",    pa.string()),     # "spot" or "fut"
    ("symbol",         pa.string()),     # "BTCUSDT" or "ETHUSDT"
    ("seq",            pa.int64()),      # Binance update ID (sequence number)
    ("depth_target",   pa.int64()),      # requested depth limit
    ("depth_actual",   pa.int64()),      # actual levels available
    ("best_bid",       pa.float64()),    # top-of-book bid price
    ("best_ask",       pa.float64()),    # top-of-book ask price
    ("bids_px",        pa.list_(pa.float64())),   # bid prices (descending)
    ("bids_qty",       pa.list_(pa.float64())),   # bid quantities
    ("asks_px",        pa.list_(pa.float64())),   # ask prices (ascending)
    ("asks_qty",       pa.list_(pa.float64())),   # ask quantities
    ("reconnect_flag", pa.int64()),      # 1 on first emission after reconnect
    ("venue_scope",    pa.string()),     # "Binance"
    ("market_scope",   pa.string()),     # "Spot" or "Futures"
    ("base",           pa.string()),     # "BTC" or "ETH"
    ("quote",          pa.string()),     # "USDT"
    ("symbol_canon",   pa.string()),     # "BTC" or "ETH"
    ("instrument_canon", pa.string()),   # "BTC-USDT" or "ETH-USDT"
])

# ==============================================================================
# File Path Helpers
# ==============================================================================

def _ts() -> str:
    """Human-readable UTC timestamp for log messages."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _hour_key(dt: Optional[datetime] = None) -> str:
    """
    Generate hour-based partition key for Parquet file rotation.
    Format: "2026-02-16_14" (date + hour).
    """
    dt = (dt or datetime.now(timezone.utc)).replace(minute=0, second=0, microsecond=0)
    return dt.strftime("%Y-%m-%d_%H")


def _final_path(base_dir: Path, prefix: str, hour_key: str) -> Path:
    """Final Parquet file path: e.g. trades_btc_spot_2026-02-16_14.parquet"""
    return base_dir / f"{prefix}_{hour_key}.parquet"


def _tmp_path(final_path: Path, pid: int) -> Path:
    """
    Temporary file path used during writing.
    The .tmp.{pid} suffix prevents collisions if multiple processes write
    to the same directory. On completion, this is atomically renamed to final.
    """
    return final_path.with_suffix(final_path.suffix + f".tmp.{pid}")


# ==============================================================================
# Supervised Task Wrapper
# ==============================================================================

async def supervised_task(coro_factory, name: str, restart_delay: float = 5.0):
    """
    Run a coroutine in a crash-recovery loop.

    If the coroutine raises any exception (except CancelledError for clean
    shutdown), log the error and restart after restart_delay seconds.
    This ensures individual adapter crashes don't bring down the pipeline.
    """
    while True:
        try:
            print(f"[{_ts()}] [SUPERVISOR] Starting {name}")
            await coro_factory()
        except asyncio.CancelledError:
            print(f"[{_ts()}] [SUPERVISOR] {name} cancelled (clean shutdown)")
            raise
        except Exception as e:
            print(f"[{_ts()}] [SUPERVISOR] {name} crashed: {type(e).__name__}: {e}")
            print(f"[{_ts()}] [SUPERVISOR] Restarting {name} in {restart_delay}s...")
            await asyncio.sleep(restart_delay)


# ==============================================================================
# Rotating Parquet Writer
# ==============================================================================

class RotatingParquetWriter:
    """
    Writes rows to hourly-rotated Parquet files with crash safety.

    Behavior:
      - Buffers rows in memory until flush_every threshold is reached
      - Writes buffered rows as an Arrow table (efficient columnar batch)
      - Rotates to a new file at each hour boundary
      - Uses tmp file + atomic rename for crash safety:
        if the process dies mid-write, no corrupt final file is left behind
      - On flush failure, the batch is restored to the buffer (no data loss)

    Thread safety: All mutations are protected by an asyncio Lock.
    """

    def __init__(
        self,
        base_dir: Path,
        file_prefix: str,
        schema: pa.Schema,
        flush_every: int = 2000,
        compression: str = "zstd",
    ):
        self.base_dir = base_dir
        self.file_prefix = file_prefix
        self.schema = schema
        self.flush_every = flush_every
        self.compression = compression
        self.pid = os.getpid()

        self.cur_hour: Optional[str] = None
        self.writer: Optional[pq.ParquetWriter] = None
        self.tmp_path: Optional[Path] = None
        self.final_path: Optional[Path] = None

        self.buffer: List[Dict[str, Any]] = []
        self.rows_written: int = 0

        self._lock = asyncio.Lock()
        # Pre-compute schema field names for efficient row filtering
        self._schema_fields = set(schema.names)

    def _open_for(self, hour_key: str) -> None:
        """Open a new Parquet writer for the given hour."""
        final_p = _final_path(self.base_dir, self.file_prefix, hour_key)
        tmp_p = _tmp_path(final_p, self.pid)
        tmp_p.parent.mkdir(parents=True, exist_ok=True)

        self.writer = pq.ParquetWriter(tmp_p, self.schema, compression=self.compression)
        self.tmp_path = tmp_p
        self.final_path = final_p
        self.cur_hour = hour_key
        self.buffer.clear()
        self.rows_written = 0
        print(f"[{_ts()}] [PARQUET {self.file_prefix}] opened hour={hour_key}")

    def _write_batch_sync(self, rows: List[Dict[str, Any]]) -> int:
        """Convert row dicts to Arrow table and write to Parquet. Returns row count."""
        if not rows or self.writer is None:
            return 0
        tbl = pa.Table.from_pylist(rows, schema=self.schema)
        self.writer.write_table(tbl)
        return int(tbl.num_rows)

    async def _flush_batch_async(self, rows_to_write: List[Dict[str, Any]]) -> int:
        """Async wrapper for batch write (currently synchronous, extensible)."""
        return self._write_batch_sync(rows_to_write)

    async def flush_async(self) -> None:
        """Flush buffered rows to disk. Restores buffer on failure."""
        async with self._lock:
            if self.writer is None or not self.buffer:
                return
            batch = self.buffer
            self.buffer = []

        # Filter each row to only schema columns (drops debug fields)
        rows_to_write = [{k: row.get(k) for k in self._schema_fields} for row in batch]

        try:
            n = await self._flush_batch_async(rows_to_write)
        except Exception as e:
            # On failure, put the batch back so no data is lost
            async with self._lock:
                self.buffer = batch + self.buffer
            print(f"[{_ts()}] [PARQUET {self.file_prefix}] flush ERROR (batch restored): {e}")
            return

        async with self._lock:
            self.rows_written += int(n)

    async def finalize_async(self) -> None:
        """Close current writer and atomically rename tmp → final."""
        async with self._lock:
            if self.writer is None:
                return
            writer = self.writer
            tmp_p = self.tmp_path
            final_p = self.final_path
            hour = self.cur_hour

        await self.flush_async()

        try:
            writer.close()
        except Exception:
            pass

        async with self._lock:
            rows = self.rows_written
            self.writer = None
            self.tmp_path = None
            self.final_path = None
            self.cur_hour = None
            self.rows_written = 0

        if tmp_p and final_p:
            try:
                os.replace(tmp_p, final_p)
                print(f"[{_ts()}] [PARQUET {self.file_prefix}] finalized {final_p.name} rows={rows} hour={hour}")
            except Exception as e:
                print(f"[{_ts()}] [PARQUET {self.file_prefix}] ERROR: rename failed: {e}")

    async def append_async(self, row: Dict[str, Any]) -> None:
        """Add a row to the buffer. Triggers rollover or flush if needed."""
        need_rollover = False
        need_flush = False

        async with self._lock:
            hour = _hour_key()
            if self.writer is None:
                self._open_for(hour)
            elif self.cur_hour != hour:
                # Hour changed — need to finalize current file and start new one
                need_rollover = True

            if not need_rollover:
                self.buffer.append(row)
                need_flush = (len(self.buffer) >= self.flush_every)

        if need_rollover:
            await self.finalize_async()
            async with self._lock:
                hour = _hour_key()
                if self.writer is None:
                    self._open_for(hour)
                self.buffer.append(row)
                need_flush = (len(self.buffer) >= self.flush_every)

        if need_flush:
            await self.flush_async()

    def close(self) -> None:
        """Synchronous close for final cleanup (called in finally blocks)."""
        try:
            if self.writer is None:
                return
            if self.buffer:
                filtered = [{k: row.get(k) for k in self._schema_fields} for row in self.buffer]
                tbl = pa.Table.from_pylist(filtered, schema=self.schema)
                self.writer.write_table(tbl)
                self.rows_written += int(tbl.num_rows)
                self.buffer.clear()
            self.writer.close()
            if self.tmp_path and self.final_path:
                os.replace(self.tmp_path, self.final_path)
                print(f"[{_ts()}] [PARQUET {self.file_prefix}] finalized {self.final_path.name} rows={self.rows_written}")
        except Exception as e:
            print(f"[{_ts()}] [PARQUET {self.file_prefix}] close error: {e}")


# ==============================================================================
# Asset-Market Router
# ==============================================================================

async def asset_market_router(
    name: str,
    source: asyncio.Queue,
    output_queues: Dict[str, asyncio.Queue],
    dashboard_queue: asyncio.Queue,
    hot_queue: Optional[asyncio.Queue] = None,
    stats_interval: int = 60,
):
    """
    Central routing task: reads from a shared source queue, enriches each row
    with canonical fields, and routes it to the correct per-(asset, market) queue.

    Routing logic:
      1. Read row from source queue (all assets mixed together)
      2. Inject alias fields (venue_scope, base, quote, etc.)
      3. Extract routing key: "{base}_{market}" e.g. "btc_spot", "eth_fut"
      4. Push to the matching output queue (which feeds a Parquet writer)
      5. Fan out to dashboard queue (non-blocking, drops on overflow)
      6. Fan out to hot-path queue (non-blocking, drops on overflow)

    Args:
        name:            Router name for logging ("trades" or "deep")
        source:          Shared input queue (all adapters feed into this)
        output_queues:   Dict of routing key → asyncio.Queue
                         Keys: "btc_spot", "btc_fut", "eth_spot", "eth_fut"
        dashboard_queue: Non-blocking fan-out queue for live dashboard
        hot_queue:       Non-blocking fan-out queue for hot-path pipeline (optional)
        stats_interval:  Seconds between periodic stats log messages
    """
    row_counts: Dict[str, int] = {k: 0 for k in output_queues}
    unrouted = 0
    dashboard_drops = 0
    hot_drops = 0
    last_stats = asyncio.get_event_loop().time()
    last_queue_warn = 0.0

    try:
        while True:
            row = await source.get()
            row = _inject_alias_fields(row)

            # Build routing key from base asset + market type
            base = (row.get("base") or "").lower()
            market = row.get("market_type", "spot")
            key = f"{base}_{market}"

            q = output_queues.get(key)
            if q is not None:
                await q.put(row)
                row_counts[key] = row_counts.get(key, 0) + 1
            else:
                # Unroutable row: symbol couldn't be decomposed or unknown asset.
                # Should be rare; log first few occurrences for debugging.
                unrouted += 1
                if unrouted <= 5 or unrouted % 1000 == 0:
                    print(
                        f"[{_ts()}] [ROUTER {name}] unrouted row #{unrouted}: "
                        f"base={base} market={market} symbol={row.get('symbol')}"
                    )

            # Dashboard fan-out: non-blocking to prevent backpressure
            try:
                dashboard_queue.put_nowait(row)
            except asyncio.QueueFull:
                dashboard_drops += 1

            # Hot-path fan-out: non-blocking to prevent backpressure
            if hot_queue is not None:
                try:
                    hot_queue.put_nowait(row)
                except asyncio.QueueFull:
                    hot_drops += 1

            # Queue pressure monitoring: warn if any output queue > 80% full
            now_mono = asyncio.get_event_loop().time()
            if now_mono - last_queue_warn > 30:
                for qk, qq in output_queues.items():
                    if qq.maxsize > 0 and qq.qsize() / qq.maxsize > 0.8:
                        print(f"[{_ts()}] [ROUTER {name}] queue pressure: {qk}={qq.qsize()}/{qq.maxsize}")
                        last_queue_warn = now_mono
                        break
                # Also check hot queue pressure
                if hot_queue is not None and hot_queue.maxsize > 0:
                    fill = hot_queue.qsize() / hot_queue.maxsize
                    if fill > 0.8:
                        print(f"[{_ts()}] [ROUTER {name}] HOT queue pressure: {hot_queue.qsize()}/{hot_queue.maxsize}")
                        last_queue_warn = now_mono

            # Periodic throughput stats
            if now_mono - last_stats >= stats_interval:
                total = sum(row_counts.values())
                parts = " ".join(f"{k}={v}" for k, v in sorted(row_counts.items()))
                drop_pct = (dashboard_drops / total * 100) if total > 0 else 0
                hot_str = f" hot_drops={hot_drops}" if hot_queue is not None else ""
                print(
                    f"[{_ts()}] [ROUTER {name}] {parts} total={total} "
                    f"unrouted={unrouted} drops={dashboard_drops} ({drop_pct:.1f}%){hot_str}"
                )
                last_stats = now_mono

    except asyncio.CancelledError:
        total = sum(row_counts.values())
        parts = " ".join(f"{k}={v}" for k, v in sorted(row_counts.items()))
        hot_str = f" hot_drops={hot_drops}" if hot_queue is not None else ""
        print(f"[{_ts()}] [ROUTER {name}] shutdown: {parts} total={total} unrouted={unrouted} drops={dashboard_drops}{hot_str}")
        raise


# ==============================================================================
# Parquet Writer Task
# ==============================================================================

async def parquet_writer_task(name: str, prefix: str, schema: pa.Schema, queue: asyncio.Queue):
    """
    Consume rows from a queue and write them to hourly-rotated Parquet files.

    Uses a 2-second timeout on queue reads to trigger periodic flushes even
    during quiet periods. This ensures data is written to disk promptly.

    On cancellation (shutdown), performs a shielded flush + finalize to ensure
    all buffered data is persisted before the process exits.
    """
    writer = RotatingParquetWriter(OUTPUT_DIR, prefix, schema, flush_every=2000, compression=PARQUET_COMP)
    rows_total = 0
    rows_invalid = 0

    try:
        while True:
            try:
                row = await asyncio.wait_for(queue.get(), timeout=2.0)
                try:
                    await writer.append_async(row)
                    rows_total += 1
                except Exception as e:
                    rows_invalid += 1
                    if rows_invalid <= 10 or rows_invalid % 1000 == 0:
                        print(f"[{_ts()}] [PARQUET {name}] invalid row #{rows_invalid}: {e}")
            except asyncio.TimeoutError:
                # No data for 2s — flush whatever we have to disk
                await writer.flush_async()
    except asyncio.CancelledError:
        # Graceful shutdown: shield flush + finalize from further cancellation
        try:
            await asyncio.shield(writer.flush_async())
            await asyncio.shield(writer.finalize_async())
            print(f"[{_ts()}] [PARQUET {name}] graceful shutdown complete, total={rows_total} invalid={rows_invalid}")
        except Exception as e:
            print(f"[{_ts()}] [PARQUET {name}] shutdown flush/finalize failed: {e}")
        raise
    finally:
        writer.close()


# ==============================================================================
# Depth BPS Bucket Computation
# ==============================================================================
# Converts raw bid/ask price arrays (up to 1000 levels) into a compact
# fixed-window BPS representation for the React dashboard.
#
# Window: [-15 .. -1] bps for bids, [+1 .. +15] bps for asks (relative to mid).
# Each bucket: [bps_offset, qty_sum, notional_sum]
# Missing buckets → [bps, 0.0, 0.0]  (all 30 slots always present)

_DEPTH_BPS_WINDOW = 15   # ±15 bps around mid
_DEPTH_BPS_BUCKET = 1    # 1 bps per bucket


def _compute_depth_bps(
    bids_px: List[float],
    bids_qty: List[float],
    asks_px: List[float],
    asks_qty: List[float],
    mid: float,
) -> Tuple[List[List], List[List]]:
    """
    Aggregate orderbook levels into fixed BPS buckets relative to mid.

    Returns:
        bids: list of [bps, qty_sum, notional_sum] for buckets -15 to -1
        asks: list of [bps, qty_sum, notional_sum] for buckets +1 to +15
    """
    if mid <= 0:
        empty_bids = [[-b, 0.0, 0.0] for b in range(1, _DEPTH_BPS_WINDOW + 1)]
        empty_asks = [[ a, 0.0, 0.0] for a in range(1, _DEPTH_BPS_WINDOW + 1)]
        return empty_bids, empty_asks

    bid_buckets: Dict[int, List[float]] = {}  # bucket → [qty_sum, notional_sum]
    ask_buckets: Dict[int, List[float]] = {}

    for px, qty in zip(bids_px, bids_qty):
        bps = int(round((px - mid) / mid * 10_000))
        if -_DEPTH_BPS_WINDOW <= bps <= -1:
            if bps not in bid_buckets:
                bid_buckets[bps] = [0.0, 0.0]
            bid_buckets[bps][0] += qty
            bid_buckets[bps][1] += qty * px

    for px, qty in zip(asks_px, asks_qty):
        bps = int(round((px - mid) / mid * 10_000))
        if 1 <= bps <= _DEPTH_BPS_WINDOW:
            if bps not in ask_buckets:
                ask_buckets[bps] = [0.0, 0.0]
            ask_buckets[bps][0] += qty
            ask_buckets[bps][1] += qty * px

    # Always emit complete windows — missing buckets get 0
    bids = [
        [b, bid_buckets[b][0] if b in bid_buckets else 0.0,
            bid_buckets[b][1] if b in bid_buckets else 0.0]
        for b in range(-_DEPTH_BPS_WINDOW, 0)
    ]
    asks = [
        [a, ask_buckets[a][0] if a in ask_buckets else 0.0,
            ask_buckets[a][1] if a in ask_buckets else 0.0]
        for a in range(1, _DEPTH_BPS_WINDOW + 1)
    ]
    return bids, asks


# ==============================================================================
# Dashboard WebSocket Server
# ==============================================================================

class DashboardServer:
    """
    WebSocket server for live monitoring of the data pipeline.

    Accumulates the latest trade and orderbook state per (asset, market) key.
    Broadcasts aggregated snapshots to all connected clients at a fixed interval.

    The server is non-critical: dashboard queue drops are expected under load
    and don't affect data collection or storage.
    """

    def __init__(self, host: str, port: int, broadcast_interval: float = 0.1):
        self.host = host
        self.port = port
        self.broadcast_interval = broadcast_interval
        self.clients: Set = set()

        # Buffers hold the latest state per routing key, overwritten each interval
        self.trade_buffer: Dict[str, Dict[str, Any]] = {}
        self.deep_buffer: Dict[str, Dict[str, Any]] = {}
        # depth_bps holds latest BPS snapshot per key — NOT cleared between broadcasts
        # (we always want last known state, replaced when new data arrives)
        self.depth_bps_buffer: Dict[str, Dict[str, Any]] = {}
        self.lock = asyncio.Lock()

        # Timing for lower-frequency broadcast types
        self._last_depth_bps_ts: float = 0.0   # 1 Hz
        self._last_metrics_ts: float = 0.0      # every 5s

    async def register(self, websocket, path=None):
        """Handle new WebSocket client connection."""
        self.clients.add(websocket)
        print(f"[{_ts()}] [DASHBOARD] client connected, total={len(self.clients)}")
        try:
            await websocket.wait_closed()
        finally:
            self.clients.discard(websocket)
            print(f"[{_ts()}] [DASHBOARD] client disconnected, total={len(self.clients)}")

    async def add_trade(self, row: Dict[str, Any]):
        """Buffer a trade row. Accumulates trade_count and volume per key."""
        key = f"{row.get('base','?')}:{row['market_type']}:{row['symbol']}"
        async with self.lock:
            if key not in self.trade_buffer:
                self.trade_buffer[key] = {
                    "venue": row["venue"],
                    "market_type": row["market_type"],
                    "market_scope": row.get("market_scope"),
                    "venue_scope": row.get("venue_scope"),
                    "symbol": row["symbol"],
                    "base": row.get("base"),
                    "quote": row.get("quote"),
                    "symbol_canon": row.get("symbol_canon"),
                    "instrument_canon": row.get("instrument_canon"),
                    "price": row["price"],
                    "qty": row["qty"],
                    "side": row["side"],
                    "ts_ms": row["ts_ms"],
                    "trade_count": 1,
                    "volume": row["qty"],
                }
            else:
                buf = self.trade_buffer[key]
                buf["price"] = row["price"]
                buf["qty"] = row["qty"]
                buf["side"] = row["side"]
                buf["ts_ms"] = row["ts_ms"]
                buf["trade_count"] += 1
                buf["volume"] += row["qty"]

    async def add_deep(self, row: Dict[str, Any]):
        """Buffer an orderbook row and compute BPS buckets for dashboard."""
        key = f"{row.get('base','?')}:{row['market_type']}:{row['symbol']}"
        best_bid = row["best_bid"]
        best_ask = row["best_ask"]
        mid = 0.5 * (best_bid + best_ask)

        # Compute BPS buckets from full arrays (available before router strips them)
        bps_bids, bps_asks = _compute_depth_bps(
            row.get("bids_px", []),
            row.get("bids_qty", []),
            row.get("asks_px", []),
            row.get("asks_qty", []),
            mid,
        )

        async with self.lock:
            self.deep_buffer[key] = {
                "venue":           row["venue"],
                "market_type":     row["market_type"],
                "market_scope":    row.get("market_scope"),
                "venue_scope":     row.get("venue_scope"),
                "symbol":          row["symbol"],
                "base":            row.get("base"),
                "quote":           row.get("quote"),
                "symbol_canon":    row.get("symbol_canon"),
                "instrument_canon":row.get("instrument_canon"),
                "best_bid":        best_bid,
                "best_ask":        best_ask,
                "spread":          best_ask - best_bid,
                "mid":             mid,
                "depth_target":    row["depth_target"],
                "depth_actual":    row["depth_actual"],
                "ts_ms":           row["ts_ms"],
            }
            self.depth_bps_buffer[key] = {
                "venue":           row["venue"],
                "market_type":     row["market_type"],
                "symbol":          row["symbol"],
                "base":            row.get("base"),
                "mid":             mid,
                "bucket_size_bps": _DEPTH_BPS_BUCKET,
                "window_bps":      _DEPTH_BPS_WINDOW,
                "bids":            bps_bids,
                "asks":            bps_asks,
                "ts_ms":           row["ts_ms"],
            }

    async def broadcast_loop(self):
        """Periodically send accumulated state to all connected clients."""
        while True:
            await asyncio.sleep(self.broadcast_interval)
            now_mono = asyncio.get_event_loop().time()

            # --- Cadence flags ---
            send_depth_bps = (now_mono - self._last_depth_bps_ts) >= 1.0
            send_metrics   = (now_mono - self._last_metrics_ts)   >= 5.0

            # If nobody is listening, clear volatile buffers and skip
            if not self.clients:
                async with self.lock:
                    self.trade_buffer.clear()
                    self.deep_buffer.clear()
                    # depth_bps_buffer is NOT cleared — keep last known state
                if send_metrics:
                    self._last_metrics_ts = now_mono
                if send_depth_bps:
                    self._last_depth_bps_ts = now_mono
                continue

            # Snapshot and clear trade/deep buffers atomically
            async with self.lock:
                trade_data = list(self.trade_buffer.values()) if self.trade_buffer else None
                deep_data  = list(self.deep_buffer.values())  if self.deep_buffer  else None
                self.trade_buffer.clear()
                self.deep_buffer.clear()
                # Snapshot depth_bps without clearing (always serve latest state)
                bps_data = list(self.depth_bps_buffer.values()) if self.depth_bps_buffer else None

            # Build JSON messages
            messages = []
            ts = _now_ms()

            if trade_data:
                messages.append(json.dumps({"type": "trade", "ts": ts, "data": trade_data}))
            if deep_data:
                messages.append(json.dumps({"type": "deep",  "ts": ts, "data": deep_data}))

            # depth_bps: 1 Hz
            if send_depth_bps:
                if bps_data:
                    messages.append(json.dumps({
                        "type": "depth_bps",
                        "ts":   ts,
                        "data": bps_data,
                    }))
                self._last_depth_bps_ts = now_mono

            # metrics: every 5s
            if send_metrics:
                messages.append(json.dumps({
                    "type": "metrics",
                    "ts":   ts,
                    "data": metrics.get_stats(),
                }))
                self._last_metrics_ts = now_mono

            if not messages:
                continue

            # Send to all clients, track dead connections
            dead_clients = []
            for client in self.clients.copy():
                for msg in messages:
                    try:
                        await client.send(msg)
                    except Exception:
                        dead_clients.append(client)
                        break

            for client in dead_clients:
                self.clients.discard(client)

    async def start(self):
        """Start the WebSocket server and broadcast loop."""
        if not HAS_WEBSOCKETS:
            print(f"[{_ts()}] [DASHBOARD] websockets not available, server disabled")
            while True:
                await asyncio.sleep(3600)

        broadcast_task = asyncio.create_task(self.broadcast_loop())

        try:
            async with ws_serve(self.register, self.host, self.port):
                print(f"[{_ts()}] [DASHBOARD] WebSocket server started on ws://{self.host}:{self.port}")
                await asyncio.Future()  # run forever
        except asyncio.CancelledError:
            broadcast_task.cancel()
            try:
                await broadcast_task
            except asyncio.CancelledError:
                pass
            print(f"[{_ts()}] [DASHBOARD] server shutdown")


async def dashboard_consumer(queue: asyncio.Queue, server: DashboardServer, data_type: str):
    """Bridge between router dashboard queue and the DashboardServer buffers."""
    try:
        while True:
            row = await queue.get()
            if data_type == "trade":
                await server.add_trade(row)
            elif data_type == "deep":
                await server.add_deep(row)
    except asyncio.CancelledError:
        pass


async def prometheus_server(host: str = "0.0.0.0", port: int = 8766):
    """
    Serve Prometheus metrics on GET /metrics (port 8766 by default).

    Uses aiohttp so it runs inside the same event loop — no extra threads.
    Grafana scrapes this endpoint; configure scrape_interval ~15s.

    Requires: pip install aiohttp prometheus_client
    """
    if not HAS_AIOHTTP:
        print(f"[{_ts()}] [PROMETHEUS] aiohttp not available, endpoint disabled")
        while True:
            await asyncio.sleep(3600)

    async def handle_metrics(request: "_aiohttp_web.Request") -> "_aiohttp_web.Response":
        body = metrics.prometheus_generate_latest()
        ctype = metrics.prometheus_content_type()
        return _aiohttp_web.Response(
            body=body,
            headers={"Content-Type": ctype},
        )

    async def handle_health(request: "_aiohttp_web.Request") -> "_aiohttp_web.Response":
        return _aiohttp_web.Response(text="ok")

    app = _aiohttp_web.Application()
    app.router.add_get("/metrics", handle_metrics)
    app.router.add_get("/health",  handle_health)

    runner = _aiohttp_web.AppRunner(app)
    await runner.setup()
    site = _aiohttp_web.TCPSite(runner, host, port)
    await site.start()
    print(f"[{_ts()}] [PROMETHEUS] HTTP endpoint: http://{host}:{port}/metrics")

    try:
        await asyncio.Future()   # run until cancelled
    except asyncio.CancelledError:
        pass
    finally:
        await runner.cleanup()
        print(f"[{_ts()}] [PROMETHEUS] server shutdown")


# ==============================================================================
# Main Entry Point
# ==============================================================================

async def main():
    """
    Orchestrate the entire data collection pipeline.

    Steps:
      1. Create shared input queues (all adapters write here)
      2. Create per-(asset, market) output queues (routers write, writers read)
      3. Spawn adapter tasks from ASSETS config (auto-scales with new assets)
      4. Spawn routers that split data by asset + market
      5. Spawn Parquet writers for each queue
      6. Spawn dashboard server
      7. Wait for shutdown signal (SIGINT/SIGTERM)
      8. Cancel all tasks and flush remaining data
    """

    # ------------------------------------------------------------------
    # Step 0: Pre-flight sanity check — spot/futures basis
    # ------------------------------------------------------------------
    # Verifies that spot and futures mid prices are reachable and that the
    # basis (in basis points) is within tolerance before connecting WS
    # streams. Soft check: anomalies are logged as warnings; a failure
    # here never aborts startup. See metrics.preflight_mid_check for the
    # tolerance semantics (default 50 bps).
    try:
        await metrics.preflight_mid_check(tolerance_bps=50.0)
    except AttributeError:
        # metrics module present but preflight_mid_check not defined —
        # tolerate older metrics builds without breaking startup.
        print(f"[{_ts()}] [PRE-FLIGHT] preflight_mid_check unavailable in metrics module — skipping")
    except Exception as e:
        # Network/API hiccup or unexpected response shape: log and continue.
        print(f"[{_ts()}] [PRE-FLIGHT] check failed (non-fatal): {e}")

    # ------------------------------------------------------------------
    # Step 1: Shared input queues
    # ------------------------------------------------------------------
    # All trade adapters feed into trades_main, all deep adapters into deep_main.
    # The routers then split these by (asset, market).
    trades_main = asyncio.Queue(maxsize=50000)
    deep_main = asyncio.Queue(maxsize=20000)

    # ------------------------------------------------------------------
    # Step 2: Per-(asset, market) output queues
    # ------------------------------------------------------------------
    # Created dynamically from ASSETS config.
    # Keys: per asset x market, e.g. "btc_spot", "btc_fut", ..., "bnb_spot", "bnb_fut"
    trades_queues: Dict[str, asyncio.Queue] = {}
    deep_queues: Dict[str, asyncio.Queue] = {}

    for asset in ASSETS:
        for market in ("spot", "fut"):
            key = f"{asset.lower()}_{market}"
            trades_queues[key] = asyncio.Queue(maxsize=15000)
            deep_queues[key] = asyncio.Queue(maxsize=5000)

    # Dashboard fan-out queues (non-blocking, drops on overflow)
    trades_dashboard = asyncio.Queue(maxsize=5000)
    deep_dashboard = asyncio.Queue(maxsize=2000)

    dashboard_server = DashboardServer(DASHBOARD_HOST, DASHBOARD_PORT, broadcast_interval=0.1)

    # ------------------------------------------------------------------
    # Step 3: Spawn adapter tasks
    # ------------------------------------------------------------------
    # 3 assets × 2 markets × 2 data types = 12 adapter tasks.
    # Each runs inside supervised_task for crash recovery.
    #
    # IMPORTANT: Lambda default args (s=sym, d=depth, w=fut_ws_ms) are
    # required to avoid Python's late-binding closure gotcha. Without them,
    # all lambdas would capture the last loop iteration's values.
    trade_adapters = []
    deep_adapters = []

    for asset, cfg in ASSETS.items():
        sym = cfg["binance_symbol"]
        depth = cfg["depth_limit"]
        fut_ws_ms = cfg["fut_ws_interval_ms"]

        # Trade adapters: spot + futures
        trade_adapters.append(
            asyncio.create_task(
                supervised_task(
                    lambda s=sym: binance_trade_consumer(s, "spot", trades_main),
                    f"trades-{asset.lower()}-spot",
                ),
                name=f"trades-{asset.lower()}-spot",
            )
        )
        trade_adapters.append(
            asyncio.create_task(
                supervised_task(
                    lambda s=sym: binance_trade_consumer(s, "fut", trades_main),
                    f"trades-{asset.lower()}-fut",
                ),
                name=f"trades-{asset.lower()}-fut",
            )
        )

        # Deep L2 adapters: spot + futures
        deep_adapters.append(
            asyncio.create_task(
                supervised_task(
                    lambda s=sym, d=depth: binance_deep_l2_consumer(s, "spot", deep_main, depth_limit=d),
                    f"deep-{asset.lower()}-spot",
                ),
                name=f"deep-{asset.lower()}-spot",
            )
        )
        deep_adapters.append(
            asyncio.create_task(
                supervised_task(
                    lambda s=sym, d=depth, w=fut_ws_ms: binance_deep_l2_consumer(
                        s, "fut", deep_main, depth_limit=d, ws_interval_ms=w,
                    ),
                    f"deep-{asset.lower()}-fut",
                ),
                name=f"deep-{asset.lower()}-fut",
            )
        )

    # ------------------------------------------------------------------
    # Step 4: Spawn routers
    # ------------------------------------------------------------------
    # Two routers: one for trades, one for deep L2.
    # Each reads from a shared queue and routes to per-(asset, market) queues.
    # Hot-path queues are passed as optional fan-out targets.
    routers = [
        asyncio.create_task(
            asset_market_router("trades", trades_main, trades_queues, trades_dashboard),
            name="router-trades",
        ),
        asyncio.create_task(
            asset_market_router("deep", deep_main, deep_queues, deep_dashboard),
            name="router-deep",
        ),
    ]

    # ------------------------------------------------------------------
    # Step 5: Spawn Parquet writers
    # ------------------------------------------------------------------
    # 12 writers: 3 assets × 2 markets × 2 data types.
    # Each writes to hourly-rotated Parquet files.
    writers = []
    for asset in ASSETS:
        for market in ("spot", "fut"):
            key = f"{asset.lower()}_{market}"

            writers.append(
                asyncio.create_task(
                    parquet_writer_task(
                        f"trades_{key}", f"trades_{key}", TRADE_SCHEMA, trades_queues[key],
                    ),
                    name=f"writer-trades-{key}",
                )
            )
            writers.append(
                asyncio.create_task(
                    parquet_writer_task(
                        f"lobdeep_{key}", f"lobdeep_{key}", DEEP_SCHEMA, deep_queues[key],
                    ),
                    name=f"writer-deep-{key}",
                )
            )

    # ------------------------------------------------------------------
    # Step 6: Spawn dashboard tasks
    # ------------------------------------------------------------------
    dashboard_tasks = [
        asyncio.create_task(dashboard_consumer(trades_dashboard, dashboard_server, "trade"), name="dashboard-trades"),
        asyncio.create_task(dashboard_consumer(deep_dashboard, dashboard_server, "deep"), name="dashboard-deep"),
        asyncio.create_task(dashboard_server.start(), name="dashboard-server"),
        asyncio.create_task(prometheus_server(DASHBOARD_HOST, PROMETHEUS_PORT), name="prometheus-server"),
    ]

    hot_tasks = []

    all_tasks = trade_adapters + deep_adapters + routers + writers + dashboard_tasks + hot_tasks

    # ------------------------------------------------------------------
    # Step 7: Signal handling
    # ------------------------------------------------------------------
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()

    def signal_handler():
        print(f"\n[{_ts()}] [MAIN] shutdown signal received")
        stop.set()

    # SIGINT (Ctrl+C) and SIGTERM trigger graceful shutdown
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, signal_handler)
        except NotImplementedError:
            pass

    # SIGHUP is ignored so detaching from tmux doesn't kill the pipeline
    try:
        loop.add_signal_handler(signal.SIGHUP, lambda: print(f"[{_ts()}] [MAIN] SIGHUP received, ignoring"))
    except NotImplementedError:
        pass

    # ------------------------------------------------------------------
    # Startup banner
    # ------------------------------------------------------------------
    asset_list = ", ".join(ASSETS.keys())
    queue_keys = sorted(trades_queues.keys())

    print("=" * 70)
    print("UNIFIED LIVE PIPELINE - BINANCE ONLY")
    print("=" * 70)
    print(f"Assets:            {asset_list}")
    print(f"Trade adapters:    {len(trade_adapters)}  ({', '.join(t.get_name() for t in trade_adapters)})")
    print(f"Deep L2 adapters:  {len(deep_adapters)}  ({', '.join(t.get_name() for t in deep_adapters)})")
    print(f"Routers:           {len(routers)}")
    print(f"Parquet writers:   {len(writers)}  (queues: {', '.join(queue_keys)})")
    print(f"Parquet output:    {OUTPUT_DIR}")
    print(f"Dashboard:         ws://{DASHBOARD_HOST}:{DASHBOARD_PORT}")
    print(f"Prometheus:        http://{DASHBOARD_HOST}:{PROMETHEUS_PORT}/metrics")
    print(f"Auto-reconnect:    enabled (5s delay)")
    print(f"Periodic flush:    enabled (2s timeout)")
    print(f"SIGHUP handling:   ignored (tmux-safe)")
    print(f"Graceful shutdown: enabled (shielded flush)")
    print("=" * 70)

    for asset, cfg in ASSETS.items():
        print(f"  {asset}: symbol={cfg['binance_symbol']}  depth={cfg['depth_limit']}  fut_ws_ms={cfg['fut_ws_interval_ms']}")

    print("=" * 70)
    print(f"[{_ts()}] Starting all tasks...")

    # ------------------------------------------------------------------
    # Step 8: Run until shutdown, then clean up
    # ------------------------------------------------------------------
    try:
        await stop.wait()
    finally:
        print(f"[{_ts()}] [MAIN] cancelling tasks...")
        for t in all_tasks:
            t.cancel()

        results = await asyncio.gather(*all_tasks, return_exceptions=True)

        errors = 0
        for t, result in zip(all_tasks, results):
            if isinstance(result, Exception) and not isinstance(result, asyncio.CancelledError):
                print(f"[{_ts()}] [MAIN] task {t.get_name()} error: {result}")
                errors += 1

        print(f"[{_ts()}] [MAIN] shutdown complete (errors: {errors})")


if __name__ == "__main__":
    asyncio.run(main())