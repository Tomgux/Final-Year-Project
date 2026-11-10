"""
Combines multiple parquet files into one full.parquet per folder path

Majority AI generated.
"""

import os
import pathlib
import logging
from logging.handlers import RotatingFileHandler
from typing import List, Optional

import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd

# =========================
# Config
# =========================
DATA_ROOT = pathlib.Path(os.getenv("DATA_ROOT", "data2"))
CURRENCIES = [s.strip().upper() for s in os.getenv("COLLECT_SYMBOLS", "BTC").split(",") if s.strip()]
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FILE = os.getenv("COMPACTOR_LOG_FILE", "compactor.log")

PARQUET_TS_UNIT = os.getenv("PARQUET_TS_UNIT", "us")

# Core schema from collector + unix_timestamp added last
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
    pa.field("bid_ask_source", pa.string()),
    pa.field("unix_timestamp", pa.int64()),
])

# =========================
# Logging
# =========================
def setup_logging():
    logger = logging.getLogger()
    # use LOG_LEVEL as a *name* on the logging module
    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = RotatingFileHandler(LOG_FILE, maxBytes=2_000_000, backupCount=3)
    fh.setFormatter(fmt)
    logger.addHandler(fh)


# =========================
# Helpers
# =========================
def list_strike_dirs(currency: str) -> List[pathlib.Path]:
    root = DATA_ROOT / currency
    if not root.exists():
        return []
    strike_dirs = []
    for expiry_dir in root.glob("expiry=*"):
        for type_dir in expiry_dir.glob("type=*"):
            for strike_dir in type_dir.glob("strike=*"):
                if strike_dir.is_dir():
                    strike_dirs.append(strike_dir)
    return strike_dirs

def extract_unix_from_name(path: pathlib.Path) -> Optional[int]:
    """
    For per-minute files named like:
      YYYY-MM-DD-call-STRIKE-UNIxtimestamp.parquet
    take the last '-' part before the extension.
    """
    if path.name == "full.parquet":
        return None
    stem = path.stem  # drop .parquet
    try:
        unix_part = stem.rsplit("-", 1)[-1]
        return int(unix_part)
    except Exception:
        return None

def read_table_loose(path: pathlib.Path) -> Optional[pa.Table]:
    """
    Very tolerant reader:
    - Reads all row-groups into pandas.
    - Ensures all expected columns exist and are properly typed.
    - Adds 'unix_timestamp' column:
        * per-minute files: from filename
        * full.parquet: from 'timestamp' column.
    - Returns a pyarrow.Table with TARGET_SCHEMA.
    """
    try:
        pf = pq.ParquetFile(path)
    except Exception as e:
        logging.error("Loose read: cannot open ParquetFile %s: %s", path, e)
        return None

    dfs = []
    for i in range(pf.num_row_groups):
        try:
            rg = pf.read_row_group(i)
            df = rg.to_pandas()
            dfs.append(df)
        except Exception as e:
            logging.error(
                "Loose read: failed reading row-group %d of %s: %s",
                i, path, e
            )

    if not dfs:
        return None

    df_all = pd.concat(dfs, ignore_index=True)

    # Add unix_timestamp based on file name or timestamp
    if path.name == "full.parquet":
        # derive from timestamp per row
        if "timestamp" in df_all.columns:
            ts = pd.to_datetime(df_all["timestamp"], utc=True, errors="coerce")
            df_all["timestamp"] = ts
            # ns -> s
            df_all["unix_timestamp"] = (ts.view("int64") // 10**9).astype("int64")
        else:
            df_all["unix_timestamp"] = pd.NA
    else:
        unix_ts = extract_unix_from_name(path)
        if unix_ts is not None:
            df_all["unix_timestamp"] = int(unix_ts)
        else:
            # Fallback: derive from timestamp if filename is unexpected
            if "timestamp" in df_all.columns:
                ts = pd.to_datetime(df_all["timestamp"], utc=True, errors="coerce")
                df_all["timestamp"] = ts
                df_all["unix_timestamp"] = (ts.view("int64") // 10**9).astype("int64")
            else:
                df_all["unix_timestamp"] = pd.NA

    # Ensure all expected columns exist
    for name in TARGET_SCHEMA.names:
        if name not in df_all.columns:
            df_all[name] = pd.NA

    # Type coercions mirroring the collector
    df_all["timestamp"] = pd.to_datetime(df_all["timestamp"], utc=True, errors="coerce")

    for col in ("instrument_name", "expiry", "type", "bid_ask_source"):
        df_all[col] = df_all[col].astype("string")

    for col in (
        "underlying_price", "estimated_delivery_price", "best_bid_price",
        "best_ask_price", "mark_price", "last", "open_interest", "volume",
        "price_change", "mid_price", "strike",
    ):
        df_all[col] = pd.to_numeric(df_all[col], errors="coerce")

    df_all["unix_timestamp"] = pd.to_numeric(df_all["unix_timestamp"], errors="coerce").astype("Int64")

    # Keep only the schema columns, in order
    df_all = df_all[TARGET_SCHEMA.names]

    try:
        table = pa.Table.from_pandas(df_all, schema=TARGET_SCHEMA, preserve_index=False)
        return table
    except Exception as e:
        logging.error("Loose read: failed building Arrow table from pandas for %s: %s", path, e)
        return None

# =========================
# Core
# =========================
def compact_strike_dir(strike_dir: pathlib.Path) -> None:
    """
    For a given strike directory, read existing full.parquet (if any)
    plus all *.parquet minute files, write a new full.parquet, and
    delete the minute files that were merged.
    """
    parquet_files = sorted(strike_dir.glob("*.parquet"))
    if not parquet_files:
        return

    full_path = strike_dir / "full.parquet"

    # Separate base "full.parquet" from minute files
    base = full_path if full_path.exists() else None
    minute_files = [f for f in parquet_files if f.name != "full.parquet"]

    # If there is nothing new besides full.parquet, skip
    if base and not minute_files:
        logging.info("Strike %s: nothing new to compact (full.parquet only).", strike_dir)
        return

    files_to_read: List[pathlib.Path] = []
    if base:
        files_to_read.append(base)
    files_to_read.extend(minute_files)

    logging.info(
        "Strike %s: compacting %d file(s) into full.parquet (base=%s, new=%d).",
        strike_dir, len(files_to_read), "yes" if base else "no", len(minute_files)
    )

    tables = []
    for f in files_to_read:
        t = read_table_loose(f)
        if t is None:
            logging.error("Strike %s: failed reading %s (skipping this file).", strike_dir, f)
            continue
        tables.append(t)

    if not tables:
        logging.warning("Strike %s: no readable tables, skipping.", strike_dir)
        return

    try:
        combined = pa.concat_tables(tables, promote=True)
    except Exception as e:
        logging.error("Strike %s: error concatenating tables: %s", strike_dir, e)
        return

    tmp_path = full_path.with_suffix(".parquet.tmp")
    try:
        pq.write_table(
            combined,
            tmp_path,
            compression="zstd",
            use_dictionary=False,
            write_statistics=True,
            coerce_timestamps=PARQUET_TS_UNIT,
        )
        tmp_path.replace(full_path)
        logging.info("Strike %s: wrote %s (rows=%d).",
                     strike_dir, full_path.name, combined.num_rows)
    except Exception as e:
        logging.error("Strike %s: failed writing %s: %s", strike_dir, full_path, e)
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass
        return

    # Delete the minute files we just compacted (but keep the new full.parquet)
    for f in minute_files:
        try:
            f.unlink()
        except Exception as e:
            logging.warning("Strike %s: failed to delete %s: %s", strike_dir, f, e)

def main():
    setup_logging()
    logging.info("Starting strike-level compactor. DATA_ROOT=%s, CURRENCIES=%s",
                 DATA_ROOT, ",".join(CURRENCIES))

    for cur in CURRENCIES:
        strike_dirs = list_strike_dirs(cur)
        logging.info("[%s] Found %d strike directories.", cur, len(strike_dirs))
        for d in strike_dirs:
            compact_strike_dir(d)

    logging.info("Compaction finished.")

if __name__ == "__main__":
    main()

