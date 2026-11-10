"""
Deribit BTC options collector (public endpoints):
- Streams best bid/ask for ALL BTC options over WebSocket (quote.*)
- Snapshots once per hour into Parquet files under per-strike folders
- Uses REST summaries for mark/underlying/OI/etc., overrides bid/ask from WS cache
- Carry-forward (LOCF) with max age + provenance flag per row
- WebSocket watchdog (auto-reconnect on stall)
- Round-robin REST refresher to prevent coverage decay
- **Skips expired instruments** (after a small grace)
- Stores as: datahourly/BTC/expiry=YYYY-MM-DD/type=call|put/STRIKEPRICE/YYYY-MM-DD-type-strike-UNIXTIMESTAMP.parquet

Minor changes vs data_collection.py
"""


import os
import math
import time
import json
import errno
import shutil
import random
import signal
import pathlib
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime, timezone
from typing import Dict, Any, Tuple, Optional, List

import platform
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from dateutil import tz

# --- WebSocket sidecar ---
import asyncio
import threading
import websockets

# =========================
# Config (env-tunable)
# =========================
CURRENCIES = [s.strip().upper() for s in os.getenv("COLLECT_SYMBOLS", "BTC").split(",") if s.strip()]
DATA_ROOT = pathlib.Path(os.getenv("DATA_ROOT", "datahourly"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FILE = os.getenv("LOG_FILE", "collector.log")
MIN_FREE_BYTES = int(os.getenv("MIN_FREE_BYTES", str(1_000_000_000)))  # 1GB
RPC_TIMEOUT = float(os.getenv("RPC_TIMEOUT", "10"))
MAX_CONSEC_FAIL = int(os.getenv("MAX_CONSEC_FAIL", "5"))
FAIL_BACKOFF_SECS = float(os.getenv("FAIL_BACKOFF_SECS", "120"))
PARQUET_TS_UNIT = os.getenv("PARQUET_TS_UNIT", "us")  # 'us' widely compatible

# Partitioning helper (still used for daily refresh / snapshots)
PARTITION_TZ = os.getenv("PARTITION_TZ", "UTC").upper()  # "UTC" or "LOCAL"

# Expiry handling
EXPIRY_GRACE_SEC = int(os.getenv("EXPIRY_GRACE_SEC", "300"))  # ignore instruments this many seconds after expiry

# Carry-forward freshness controls
WS_FRESH_MS  = int(os.getenv("WS_FRESH_MS", "90000"))                 # quote freshness gate (ms)
CARRY_FORWARD_MAX_AGE_SEC = int(os.getenv("CARRY_FORWARD_MAX_AGE_SEC", "7200"))  # default 30 minutes

# WebSocket robustness
WS_STALL_SECS = int(os.getenv("WS_STALL_SECS", "30"))      # if no msg for this many seconds, reconnect
PING_INTERVAL_SECS = int(os.getenv("PING_INTERVAL_SECS", "20"))

# Round-robin refresher (small, steady REST load to re-seed stale quotes)
ROUND_ROBIN_REFRESH_PER_MIN = int(os.getenv("ROUND_ROBIN_REFRESH_PER_MIN", "200"))

# REST fallback to fill instruments lacking a usable bid/ask this minute
MAX_FALLBACKS_PER_MIN = int(os.getenv("MAX_FALLBACKS_PER_MIN", "50"))
FALLBACK_TTL_SECS = float(os.getenv("FALLBACK_TTL_SECS", "20"))

# WebSocket settings
DERIBIT_WS_URL = os.getenv("DERIBIT_WS_URL", "wss://www.deribit.com/ws/api/v2")
WS_SUB_BATCH = int(os.getenv("WS_SUB_BATCH", "450"))       # channels per subscribe batch

DERIBIT_API = "https://www.deribit.com/api/v2/"

# =========================
# Logging & lifecycle
# =========================
def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = RotatingFileHandler(LOG_FILE, maxBytes=5_000_000, backupCount=5)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

_lock_handle = None
def acquire_lock(lockfile: str = "collector.lock"):
    global _lock_handle
    lock_path = pathlib.Path(lockfile)
    _lock_handle = open(lock_path, "a+")
    try:
        if platform.system() == "Windows":
            import msvcrt
            msvcrt.locking(_lock_handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(_lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except Exception:
        raise SystemExit("Another collector instance appears to be running (lock busy).")

_SHOULD_RUN = True
def _stop(*_):
    global _SHOULD_RUN
    _SHOULD_RUN = False

# =========================
# HTTP / JSON-RPC
# =========================
def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"Content-Type": "application/json"})
    retry = Retry(
        total=8, connect=8, read=8, backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("POST",),
        raise_on_status=False, respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=16, pool_maxsize=32)
    s.mount("https://", adapter); s.mount("http://", adapter)
    return s

def _sleep_with_jitter(base_seconds: float, cap: float) -> None:
    time.sleep(min(cap, random.uniform(0, base_seconds)))

def rpc(session: requests.Session, method: str, params: Dict[str, Any], *,
        attempts: int = 5, base_backoff: float = 0.6, cap: float = 5.0) -> Any:
    payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
    for i in range(attempts):
        try:
            resp = session.post(DERIBIT_API, json=payload, timeout=RPC_TIMEOUT)
            resp.raise_for_status()
            j = resp.json()
        except (requests.RequestException, ValueError):
            if i < attempts - 1:
                _sleep_with_jitter(base_backoff * (2 ** i), cap); continue
            raise
        if "error" in j and j["error"]:
            code = j["error"].get("code"); msg = j["error"].get("message", "")
            if code in (-10009, -32002) or "temporarily" in msg.lower():
                if i < attempts - 1:
                    _sleep_with_jitter(base_backoff * (2 ** i), cap); continue
            raise RuntimeError(f"RPC {method} error {code}: {msg}")
        return j["result"]
    raise RuntimeError(f"RPC {method} failed after {attempts} attempts")

# =========================
# Utils
# =========================
def have_enough_space(path: pathlib.Path) -> bool:
    try:
        total, used, free = shutil.disk_usage(path.resolve().anchor)
        return free >= MIN_FREE_BYTES
    except Exception:
        return True

def write_heartbeat(currency: str, written: int, last_error: Optional[str] = None):
    hb = {
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "currency": currency,
        "rows_written": written,
        "last_error": last_error,
        "pid": os.getpid(),
    }
    pathlib.Path("heartbeat.json").write_text(json.dumps(hb, indent=2))

def _nn(x):
    if x is None: return None
    if isinstance(x, (int, float)):
        if math.isnan(x) or math.isinf(x): return None
    return x

def _none_if_zero(x):
    if x is None: return None
    if isinstance(x, (int, float)) and x == 0.0: return None
    return x

def _partition_date_from_ts(ts: pd.Timestamp) -> str:
    """
    Return YYYY-MM-DD for partitioning / daily refresh based on PARTITION_TZ.
    Accepts naive or tz-aware timestamps; naive are assumed UTC.
    """
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    if PARTITION_TZ == "UTC":
        return ts.tz_convert("UTC").date().isoformat()
    return ts.tz_convert(tz.tzlocal()).date().isoformat()

# =========================
# Metadata & summaries (REST)
# =========================
def fetch_instruments(session: requests.Session, currency: str) -> pd.DataFrame:
    result = rpc(session, "public/get_instruments", {
        "currency": currency, "kind": "option", "expired": False
    })
    rows = []
    for ins in result:
        rows.append({
            "instrument_name": ins["instrument_name"],
            "expiration_timestamp": ins["expiration_timestamp"],  # ms since epoch (UTC)
            "expiration_date": datetime.fromtimestamp(ins["expiration_timestamp"]/1000, tz=timezone.utc).date().isoformat(),
            "strike": ins["strike"],
            "option_type": "call" if ins.get("option_type","call")=="call" else "put",
        })
    return pd.DataFrame(rows).sort_values(["expiration_date","option_type","strike"]).reset_index(drop=True)

def snapshot_instruments(df: pd.DataFrame, currency: str, local_tz):
    snap_dir = DATA_ROOT / currency / "instrument_snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    fname = snap_dir / (datetime.now(local_tz).date().isoformat() + ".parquet")
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(
        table, fname, compression="zstd", use_dictionary=False,
        write_statistics=True, coerce_timestamps=PARQUET_TS_UNIT
    )

def fetch_book_summaries(session: requests.Session, currency: str) -> pd.DataFrame:
    result = rpc(session, "public/get_book_summary_by_currency", {
        "currency": currency, "kind": "option"
    })
    now_utc = datetime.now(timezone.utc)
    rows = []
    for r in result:
        bid = _none_if_zero(_nn(r.get("best_bid_price")))
        ask = _none_if_zero(_nn(r.get("best_ask_price")))
        rows.append({
            "instrument_name": r["instrument_name"],
            "timestamp": now_utc,
            "underlying_price": _nn(r.get("underlying_price")),
            "estimated_delivery_price": _nn(r.get("estimated_delivery_price")),
            "best_bid_price": bid,
            "best_ask_price": ask,
            "mark_price": _nn(r.get("mark_price")),
            "last": _nn(r.get("last")),
            "open_interest": _nn(r.get("open_interest")),
            "volume": _nn(r.get("volume")),
            "price_change": _nn(r.get("price_change")),
            "mid_price": ((bid + ask)/2.0) if (bid is not None and ask is not None) else None,
        })
    return pd.DataFrame(rows)

# =========================
# Order-book fallback (REST, bounded)
# =========================
_FALLBACK_CACHE: Dict[str, Tuple[float, Optional[float], Optional[float]]] = {}
def _get_cached_fallback(name: str):
    ent = _FALLBACK_CACHE.get(name)
    if not ent: return None, None
    ts, bid, ask = ent
    if time.monotonic() - ts > FALLBACK_TTL_SECS: return None, None
    return bid, ask
def _set_cached_fallback(name: str, bid, ask):
    _FALLBACK_CACHE[name] = (time.monotonic(), bid, ask)
def fetch_top_of_book(session, instrument_name: str):
    try:
        ob = rpc(session, "public/get_order_book", {"instrument_name": instrument_name, "depth": 1})
        bid = ob["bids"][0][0] if ob.get("bids") else None
        ask = ob["asks"][0][0] if ob.get("asks") else None
        return bid, ask
    except Exception:
        return None, None

# =========================
# Fixed schema for part files
# =========================
TARGET_SCHEMA = pa.schema([
    pa.field("timestamp", pa.timestamp(PARQUET_TS_UNIT, tz="UTC")),
    pa.field("instrument_name", pa.string()),
    pa.field("underlying_price", pa.float64()),
    pa.field("estimated_delivery_price", pa.float64()),
    pa.field("best_bid_price", pa.float64()),
    pa.field("best_ask_price", pa.float64()),
    pa.field("mark_price", pa.float64()),
    pa.field("last", pa.float64()),
    pa.field("open_interest", pa.float64()),
    pa.field("volume", pa.float64()),
    pa.field("price_change", pa.float64()),
    pa.field("mid_price", pa.float64()),
    pa.field("expiry", pa.string()),
    pa.field("type", pa.string()),
    pa.field("strike", pa.float64()),
    pa.field("bid_ask_source", pa.string()),  # ws_fresh | ws_stale | rest | fallback | missing
])

def _coerce_df_to_target_schema(df: pd.DataFrame) -> pd.DataFrame:
    for f in TARGET_SCHEMA.names:
        if f not in df.columns:
            df[f] = pd.Series([None] * len(df))
    df = df[TARGET_SCHEMA.names]
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    for col in ("instrument_name", "expiry", "type", "bid_ask_source"):
        df[col] = df[col].astype("string")
    for col in ("underlying_price","estimated_delivery_price","best_bid_price",
                "best_ask_price","mark_price","last","open_interest","volume",
                "price_change","mid_price","strike"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def _atomic_write_part(table: pa.Table, target_path: pathlib.Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
    pq.write_table(
        table, tmp_path,
        compression="zstd",
        use_dictionary=False,
        write_statistics=True,
        coerce_timestamps=PARQUET_TS_UNIT
    )
    tmp_path.replace(target_path)

def write_part_file(part_path: pathlib.Path, row_df: pd.DataFrame) -> None:
    try:
        df = _coerce_df_to_target_schema(row_df)
        table = pa.Table.from_pandas(df, schema=TARGET_SCHEMA, preserve_index=False)
        _atomic_write_part(table, part_path)
        logging.debug("wrote %s", part_path)
    except OSError as e:
        if e.errno in (errno.ENOSPC, errno.EDQUOT):
            logging.error("Disk full writing %s; skipping.", part_path)
        elif e.errno in (errno.EACCES, errno.EPERM):
            logging.error("Permission error writing %s; skipping.", part_path)
        else:
            logging.error("OS error writing %s: %s", part_path, e)
    except Exception as e:
        logging.error("Unexpected error writing %s: %s", part_path, e)

# =========================
# WebSocket quote sidecar
# =========================
_quote_cache: Dict[str, Dict[str, Optional[float]]] = {}
_quote_lock = threading.Lock()
_current_subs = set()
_subs_lock = threading.Lock()

_last_ws_msg_ts = 0.0  # monotonic seconds since boot (for watchdog)

def ws_get_quote(instrument: str):
    with _quote_lock:
        ent = _quote_cache.get(instrument)
        if not ent: return None
        return ent

def ws_prime_subscriptions(instruments: List[str]) -> List[List[str]]:
    channels = [f"quote.{name}" for name in instruments]
    return [channels[i:i+WS_SUB_BATCH] for i in range(0, len(channels), WS_SUB_BATCH)]

async def _ws_send(ws, msg): await ws.send(json.dumps(msg))

async def _subscribe_batches(ws, channels_batches: List[List[str]]):
    for channels in channels_batches:
        if not channels: continue
        with _subs_lock:
            needed = [ch for ch in channels if ch not in _current_subs]
            _current_subs.update(needed)
        if not needed: continue
        await _ws_send(ws, {"jsonrpc":"2.0","id":random.randint(1,1_000_000),
                            "method":"public/subscribe","params":{"channels":needed}})

async def _subscribe_instrument_state(ws, currency="BTC"):
    await _ws_send(ws, {"jsonrpc":"2.0","id":random.randint(1,1_000_000),
                        "method":"public/subscribe",
                        "params":{"channels":[f"instrument.state.option.{currency}"]}})

async def _set_heartbeat(ws, interval_ms=30000):
    await _ws_send(ws, {"jsonrpc":"2.0","id":1,
                        "method":"public/set_heartbeat",
                        "params":{"interval": interval_ms}})

async def ws_quote_loop(get_instruments_callable):
    backoff = 1.0
    while _SHOULD_RUN:
        try:
            async with websockets.connect(DERIBIT_WS_URL, max_size=None, ping_interval=None) as ws:
                await _ws_send(ws, {"jsonrpc":"2.0","id":42,"method":"public/hello",
                                    "params":{"client_name":"options-collector","client_version":"1.0"}})
                await _set_heartbeat(ws, 30000)

                inst_names = get_instruments_callable()
                batches = ws_prime_subscriptions(inst_names)
                await _subscribe_batches(ws, batches)
                await _subscribe_instrument_state(ws, "BTC")

                logging.info("WS connected; subscribed to %d instruments in %d batches.",
                             len(inst_names), len(batches))
                backoff = 1.0
                global _last_ws_msg_ts
                _last_ws_msg_ts = time.monotonic()
                last_ping = time.monotonic()

                while _SHOULD_RUN:
                    nowm = time.monotonic()
                    if nowm - last_ping >= PING_INTERVAL_SECS:
                        try:
                            await _ws_send(ws, {"jsonrpc":"2.0","id":0,"method":"public/ping","params":{}})
                        except Exception:
                            raise
                        last_ping = nowm

                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=15.0)
                    except asyncio.TimeoutError:
                        if time.monotonic() - _last_ws_msg_ts > WS_STALL_SECS:
                            raise RuntimeError("WS stalled (no messages)")
                        else:
                            continue

                    msg = json.loads(raw)
                    _last_ws_msg_ts = time.monotonic()

                    if msg.get("method") == "heartbeat" and msg.get("params", {}).get("type") == "test_request":
                        await _ws_send(ws, {"jsonrpc":"2.0","id":0,"method":"public/test","params":{}})
                        continue

                    if msg.get("method") != "subscription":
                        continue

                    ch = msg["params"]["channel"]
                    data = msg["params"]["data"]

                    if ch.startswith("instrument.state.option."):
                        name = data.get("instrument_name"); state = data.get("state")
                        if name and state in ("created","started"):
                            await _subscribe_batches(ws, [[f"quote.{name}"]])
                        continue

                    if ch.startswith("quote."):
                        ins = data.get("instrument_name")
                        if not ins: continue
                        with _quote_lock:
                            _quote_cache[ins] = {
                                "bid": data.get("best_bid_price"),
                                "ask": data.get("best_ask_price"),
                                "bid_sz": data.get("best_bid_amount"),
                                "ask_sz": data.get("best_ask_amount"),
                                "ts_ms": data.get("timestamp") or int(time.time()*1000),
                            }
        except Exception as e:
            logging.warning("WS disconnected (%s). Reconnecting in %.1fs...", e, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff*2.0, 30.0)

def start_ws_thread(get_instruments_callable):
    loop = asyncio.new_event_loop()
    t = threading.Thread(
        target=lambda: loop.run_until_complete(ws_quote_loop(get_instruments_callable)),
        name="DeribitWS", daemon=True
    )
    t.start()
    return loop, t

# =========================
# Hour scheduler
# =========================
def hour_scheduler():
    # Align to the top of the next hour
    start_wall = time.time()
    first_target = math.floor(start_wall / 3600) * 3600 + 3600
    time.sleep(max(0, first_target - start_wall))
    next_tick = time.monotonic() + 3600.0
    while True:
        yield
        now = time.monotonic()
        time.sleep(max(0.0, next_tick - now))
        next_tick += 3600.0


# =========================
# Main per-currency loop
# =========================
def run_currency(session: requests.Session, currency: str):
    local_tz = tz.tzlocal()
    today_partition = None

    # Instruments
    for i in range(3):
        try:
            instruments = fetch_instruments(session, currency)
            break
        except Exception as e:
            logging.warning("[%s] Initial instrument fetch failed (%s). Retrying...", currency, e)
            _sleep_with_jitter(0.8 * (2 ** i), 4.0)
    else:
        raise SystemExit(f"[{currency}] Failed to fetch instruments on startup after multiple attempts.")
    logging.info("[%s] Fetched %d active option instruments.", currency, len(instruments))

    meta_by_name: Dict[str, Tuple[int, str, float, str]] = {
        row.instrument_name: (row.expiration_timestamp, row.expiration_date, row.strike, row.option_type)
        for row in instruments.itertuples(index=False)
    }
    snapshot_instruments(instruments, currency, local_tz)

    rr_names = instruments["instrument_name"].tolist()
    rr_idx = 0

    def _get_instr_names(): return instruments["instrument_name"].tolist()
    start_ws_thread(_get_instr_names)

    sched = hour_scheduler()
    consec_fail = 0

    for _ in sched:
        if not _SHOULD_RUN: break
        loop_start = time.time()

        # Midnight refresh (by chosen partition tz)
        now_partition_date = _partition_date_from_ts(pd.Timestamp.now(tz="UTC"))
        if now_partition_date != today_partition:
            try:
                instruments = fetch_instruments(session, currency)
                meta_by_name = {
                    row.instrument_name: (row.expiration_timestamp, row.expiration_date, row.strike, row.option_type)
                    for row in instruments.itertuples(index=False)
                }
                rr_names = instruments["instrument_name"].tolist()
                rr_idx = 0
                today_partition = now_partition_date
                snapshot_instruments(instruments, currency, local_tz)
                logging.info("[%s] Rolled date; instruments refreshed: %d active.", currency, len(instruments))
            except Exception as e:
                logging.warning("[%s] Midnight instrument refresh failed: %s (will retry).", currency, e)

        # Pull summaries
        unknown_seen = False
        try:
            quotes = fetch_book_summaries(session, currency)
        except Exception as e:
            consec_fail += 1
            logging.error("[%s] Error fetching summaries (%d consec): %s", currency, consec_fail, e)
            write_heartbeat(currency, written=0, last_error=str(e))
            if consec_fail >= MAX_CONSEC_FAIL:
                logging.warning("[%s] Backing off for %.0fs after repeated failures.", currency, FAIL_BACKOFF_SECS)
                time.sleep(FAIL_BACKOFF_SECS)
            continue
        else:
            if consec_fail:
                logging.info("[%s] Recovered after %d consecutive failures.", currency, consec_fail)
            consec_fail = 0

        now_ms = int(time.time() * 1000)
        records = quotes.to_dict("records")

        def enrich_row(r):
            nonlocal unknown_seen
            meta = meta_by_name.get(r["instrument_name"])
            if meta is None:
                unknown_seen = True
                return None
            exp_ts_ms, expiry_iso, strike, opt_type = meta

            # Skip expired (expiry + grace)
            if now_ms >= (exp_ts_ms + EXPIRY_GRACE_SEC * 1000):
                return None

            bid = r["best_bid_price"]
            ask = r["best_ask_price"]
            src = "missing"
            used_ws = False

            q = ws_get_quote(r["instrument_name"])
            if q:
                age_ms = now_ms - (q.get("ts_ms") or now_ms)
                if (q.get("bid") is not None) or (q.get("ask") is not None):
                    if age_ms <= WS_FRESH_MS:
                        bid = q.get("bid") if q.get("bid") is not None else bid
                        ask = q.get("ask") if q.get("ask") is not None else ask
                        src = "ws_fresh"; used_ws = True
                    elif age_ms <= CARRY_FORWARD_MAX_AGE_SEC * 1000:
                        bid = q.get("bid") if q.get("bid") is not None else bid
                        ask = q.get("ask") if q.get("ask") is not None else ask
                        src = "ws_stale"; used_ws = True

            if not used_ws and ((bid is not None) or (ask is not None)):
                src = "rest"

            return {
                "timestamp": r["timestamp"],
                "instrument_name": r["instrument_name"],
                "expiry": expiry_iso,
                "type": opt_type,
                "strike": strike,
                "underlying_price": r["underlying_price"],
                "estimated_delivery_price": r["estimated_delivery_price"],
                "best_bid_price": bid,
                "best_ask_price": ask,
                "mark_price": r["mark_price"],
                "last": r["last"],
                "open_interest": r["open_interest"],
                "volume": r["volume"],
                "price_change": r["price_change"],
                "mid_price": ((bid + ask)/2.0) if (bid is not None and ask is not None) else r.get("mid_price"),
                "bid_ask_source": src,
            }

        enriched = [e for e in (enrich_row(rec) for rec in records) if e is not None]
        df = pd.DataFrame(enriched)

        if unknown_seen:
            try:
                instruments = fetch_instruments(session, currency)
                meta_by_name = {
                    row.instrument_name: (row.expiration_timestamp, row.expiration_date, row.strike, row.option_type)
                    for row in instruments.itertuples(index=False)
                }
                rr_names = instruments["instrument_name"].tolist()
                rr_idx = 0
                snapshot_instruments(instruments, currency, local_tz)
                enriched = [e for e in (enrich_row(rec) for rec in records) if e is not None]
                df = pd.DataFrame(enriched)
                logging.info("[%s] Instrument set changed intraday; metadata refreshed.", currency)
            except Exception as e:
                logging.warning("[%s] Failed to refresh instruments after unknown sightings: %s", currency, e)

        if not df.empty:
            total = len(df)
            none_ba = ((df["best_bid_price"].isna()) & (df["best_ask_price"].isna())).sum()
            ws_lag = time.monotonic() - globals().get("_last_ws_msg_ts", 0.0)
            logging.info("[%s] WS last msg %.1fs ago; missing (post-WS+carry) %d/%d (%.1f%%).",
                         currency, ws_lag if ws_lag > 0 else -1.0, none_ba, total, 100*none_ba/max(1,total))

        # Round-robin refresh
        if not df.empty and ROUND_ROBIN_REFRESH_PER_MIN > 0:
            names = rr_names if rr_names else df["instrument_name"].tolist()
            end = min(rr_idx + ROUND_ROBIN_REFRESH_PER_MIN, len(names))
            slice_names = names[rr_idx:end]
            rr_idx = 0 if end >= len(names) else end

            refreshed = 0
            now_ms_local = int(time.time()*1000)
            for name in slice_names:
                meta = meta_by_name.get(name)
                if not meta: continue
                exp_ts_ms = meta[0]
                if now_ms_local >= (exp_ts_ms + EXPIRY_GRACE_SEC * 1000):
                    continue

                row_sel = (df["instrument_name"] == name)
                if not row_sel.any():
                    continue
                row = df[row_sel].iloc[0]
                if row.get("bid_ask_source") == "ws_fresh":
                    continue
                bid, ask = fetch_top_of_book(session, name)
                if bid is None and ask is None:
                    continue
                if bid is not None:
                    df.loc[row_sel, "best_bid_price"] = bid
                if ask is not None:
                    df.loc[row_sel, "best_ask_price"] = ask
                both = df.loc[row_sel, ["best_bid_price","best_ask_price"]].iloc[0]
                if pd.notna(both["best_bid_price"]) and pd.notna(both["best_ask_price"]):
                    df.loc[row_sel, "mid_price"] = (both["best_bid_price"] + both["best_ask_price"]) / 2.0
                if df.loc[row_sel, "bid_ask_source"].iloc[0] != "ws_fresh":
                    df.loc[row_sel, "bid_ask_source"] = "rest"
                with _quote_lock:
                    _quote_cache[name] = {
                        "bid": bid, "ask": ask,
                        "bid_sz": None, "ask_sz": None,
                        "ts_ms": int(time.time()*1000),
                    }
                refreshed += 1

            if refreshed:
                logging.info("[%s] RR refresh updated %d instruments this hour.", currency, refreshed)

        # REST fallback
        fallbacks_used = 0
        if MAX_FALLBACKS_PER_MIN > 0 and not df.empty:
            need = df[(df["best_bid_price"].isna()) & (df["best_ask_price"].isna())]
            for name in need["instrument_name"].head(MAX_FALLBACKS_PER_MIN).tolist():
                meta = meta_by_name.get(name)
                if meta and now_ms >= (meta[0] + EXPIRY_GRACE_SEC * 1000):
                    continue
                bid, ask = _get_cached_fallback(name)
                if bid is None and ask is None:
                    if fallbacks_used >= MAX_FALLBACKS_PER_MIN: break
                    bid, ask = fetch_top_of_book(session, name)
                    _set_cached_fallback(name, bid, ask)
                    fallbacks_used += 1
                if bid is not None or ask is not None:
                    sel = (df["instrument_name"] == name)
                    if bid is not None: df.loc[sel, "best_bid_price"] = bid
                    if ask is not None: df.loc[sel, "best_ask_price"] = ask
                    df.loc[sel, "bid_ask_source"] = "fallback"
                    both = df.loc[sel, ["best_bid_price","best_ask_price"]].iloc[0]
                    if pd.notna(both["best_bid_price"]) and pd.notna(both["best_ask_price"]):
                        df.loc[sel, "mid_price"] = (both["best_bid_price"] + both["best_ask_price"]) / 2.0
        if fallbacks_used:
            logging.info("[%s] REST fallback filled top-of-book for %d instruments this hour.", currency, fallbacks_used)

        if df.empty:
            logging.warning("[%s] No quotes to write this hour.", currency)
            write_heartbeat(currency, written=0, last_error=None)
            continue

        # Disk check once per minute (no day-dir anymore)
        base_dir = DATA_ROOT / currency
        if not have_enough_space(base_dir):
            logging.error("[%s] Low disk space; skipping writes this hour.", currency)
            write_heartbeat(currency, written=0, last_error="low_disk")
            continue

        # --- Write one file per strike per minute ---
        written = 0
        for row in df.itertuples(index=False):
            ts = pd.to_datetime(row.timestamp, utc=True)
            date_str = ts.tz_convert("UTC").date().isoformat()
            unix_ts = int(ts.timestamp())

            # Strike directory: expiry=YYYY-MM-DD/type=call|put/STRIKEPRICE/
            strike_str = (str(int(row.strike)) if float(row.strike).is_integer()
                          else str(row.strike).rstrip('0').rstrip('.'))
            strike_dir = (DATA_ROOT / currency /
                          f"expiry={row.expiry}" /
                          f"type={row.type}" /
                          f"strike={strike_str}")

            filename = f"{date_str}-{row.type}-{strike_str}-{unix_ts}.parquet"
            part_path = strike_dir / filename

            row_df = pd.DataFrame([{
                "timestamp": ts,
                "instrument_name": row.instrument_name,
                "underlying_price": row.underlying_price,
                "estimated_delivery_price": row.estimated_delivery_price,
                "best_bid_price": row.best_bid_price,
                "best_ask_price": row.best_ask_price,
                "mark_price": row.mark_price,
                "last": row.last,
                "open_interest": row.open_interest,
                "volume": row.volume,
                "price_change": row.price_change,
                "mid_price": getattr(row, "mid_price", None),
                "expiry": row.expiry,
                "type": row.type,
                "strike": row.strike,
                "bid_ask_source": getattr(row, "bid_ask_source", None),
            }])
            write_part_file(part_path, row_df)
            written += 1

        elapsed = time.time() - loop_start
        if elapsed > 55:
            logging.warning("[%s] Slow cycle (%.2fs); wrote %d parts; realigning.", currency, elapsed, written)
        else:
            logging.info("[%s] Wrote %d parts in %.2fs.", currency, written, elapsed)

        write_heartbeat(currency, written=written, last_error=None)

# =========================
# Entrypoint
# =========================
def main():
    setup_logging()
    acquire_lock()
    signal.signal(signal.SIGINT, _stop); signal.signal(signal.SIGTERM, _stop)

    session = make_session()
    logging.info("Using Deribit public endpoints (no auth).")

    for cur in CURRENCIES:
        (DATA_ROOT / cur).mkdir(parents=True, exist_ok=True)

    try:
        for cur in CURRENCIES:
            run_currency(session, cur)
    finally:
        logging.info("Shutting down.")

if __name__ == "__main__":
    main()