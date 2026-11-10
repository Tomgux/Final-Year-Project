"""
Clean BTC options CSV files.

Input:
    A folder full of CSVs named like:
        YYYY-MM-DD-type-strikeprice.csv

Each CSV row has columns:
    timestamp,instrument_name,underlying_price,estimated_delivery_price,
    best_bid_price,best_ask_price,mark_price,last,open_interest,volume,
    price_change,mid_price,expiry,type,strike,bid_ask_source,unix_timestamp

Output:
    Cleaned CSVs in a separate output folder (by default: <input>/cleaned),
    does the following:
1. coerce dtypes
2. drop bad timestamp rows (none in the end)
3. reconcile instrument columns (none in the end)
4. sort rows by timestamp
5. drop duplicates (none in the end)
6. check for time gaps (none in the end)
7. fix unix timestamps (7538 fixed)
8. checks underlying/strike for crazy values (none fixed)
9. Clean bid/ask and set mid
10. Arb check (upper and lower bound on mid prices relative to strike + underlying)

→This means that the mid prices that remain are going to follow to more of an extent the rules you’d expect prices to follow. It removes obviously silly mid prices. Doesn’t change bid or ask at all. (upper = 482 < 0.02%, lower = 10953 < 0.3%). 

→Upper bounds used are obvious (1BTC for a call and Strike price for a put)

→Lower bounds are not as obvious. As the options are European, we calculate intrinsic lower bound of the value, then check if the price falls below 0.75*intrinsic. This leads to only catching obviously silly mid prices, but allows prices up to 25% below intrinsic (since prices are based off futures so we need to allow room here)

11. Add abs + rel spread column
12. Adds expiry time in unix + moneyness + log moneyness
13. Ensure volume and open interest non negative (no issues in the end).
14. Organises columns nicely in output csv
15. Computes liquidity stats for summary

Skeleton AI generated, details in functions were me.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd


# --------- Configuration constants --------- #

NUMERIC_COLS = [
    "underlying_price",
    "estimated_delivery_price",
    "best_bid_price",
    "best_ask_price",
    "mark_price",
    "last",
    "open_interest",
    "volume",
    "price_change",
    "mid_price",
    "strike",
    "unix_timestamp",
]

PRICE_COLS = [
    "best_bid_price",
    "best_ask_price",
    "mark_price",
    "last",
    "mid_price",
]

PRIMARY_KEY_COLS = [
    "instrument_name",
    "timestamp",
]

# We’ll fix unix_timestamp if it differs from the datetime by more than this (in seconds)
UNIX_TS_TOLERANCE_SECONDS = 1

# No-arbitrage tolerances 
NOARB_UPPER_TOL = 0.06  
NOARB_LOWER_TOL = 0.06


@dataclass
class FileStats:
    filename: str
    rows_before: int
    rows_after: int

    dropped_duplicates: int
    dropped_bad_timestamps: int
    fixed_bid_ask_swaps: int
    fixed_unix_timestamp: int
    instrument_type_fixes: int
    instrument_strike_fixes: int
    instrument_expiry_fixes: int

    expected_minutes: int = 0
    missing_minutes: int = 0
    noarb_upper_violations: int = 0
    noarb_lower_violations: int = 0

    quote_minutes: int = 0
    core_quote_minutes: int = 0
    negative_open_interest: int = 0
    negative_volume: int = 0

    bad_underlying_prices: int = 0
    bad_strike_prices: int = 0


# --------- Core cleaning helpers --------- #


import pandas as pd

def coerce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Coerce timestamps, expiry, and numeric columns.
    SPECIFICALLY: Sets expiry to Deribit standard (08:00 UTC).
    """
    
    # 1. Define Numeric Columns (Global or local scope)
    NUMERIC_COLS = [
        "underlying_price", "estimated_delivery_price", "best_bid_price", 
        "best_ask_price", "mark_price", "last", "open_interest", 
        "volume", "price_change", "mid_price", "strike", "unix_timestamp"
    ]

    # 2. Timestamp: Parse and keep as timezone-aware UTC
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)

    # 3. Expiry: Handle Deribit 08:00 UTC logic
    if "expiry" in df.columns:
        # First, convert string "2025-11-10" to Timestamp "2025-11-10 00:00:00+00:00"
        df["expiry"] = pd.to_datetime(df["expiry"], errors="coerce", utc=True)
        
        # Then, ADD 8 hours to shift from Midnight to 8 AM
        df["expiry"] = df["expiry"] + pd.Timedelta(hours=8)

    # 4. Numeric columns (Sanitize strings like "**222.4" -> 222.4)
    for col in NUMERIC_COLS:
        if col in df.columns:
            if df[col].dtype == 'object':
                 # Remove any characters that aren't digits, decimals, or negative signs
                df[col] = df[col].astype(str).str.replace(r'[^\d.-]', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


import pandas as pd

def drop_bad_timestamps(df: pd.DataFrame, stats: FileStats) -> pd.DataFrame:
    """
    Tries to convert timestamps to datetime objects. 
    Drops rows where the timestamp is missing OR unparseable.
    """
    if "timestamp" not in df.columns:
        return df

    before = len(df)
    
    # 1. Coerce the column to datetime. 
    # Invalid strings become NaT (Not a Time).
    # This handles the complex format: 2025-11-08 12:50:00.109271+00:00
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors='coerce')

    # 2. Filter out NaT (which includes original NaNs and newly failed parses)
    df = df[df["timestamp"].notna()].copy()
    
    after = len(df)
    stats.dropped_bad_timestamps = before - after
    
    return df


def sort_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Sort rows by instrument and timestamp (if available)."""
    sort_cols = []
    for col in ["instrument_name", "timestamp"]:
        if col in df.columns:
            sort_cols.append(col)

    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)
    return df


def drop_duplicates(df: pd.DataFrame, stats: FileStats) -> pd.DataFrame:
    """Drop duplicate rows based on a logical primary key."""
    key_cols = [c for c in PRIMARY_KEY_COLS if c in df.columns]
    if not key_cols:
        return df

    before = len(df)
    df = df.drop_duplicates(subset=key_cols, keep="last").reset_index(drop=True)
    after = len(df)
    stats.dropped_duplicates = before - after
    return df

def sanity_check_underlying_and_strike(df: pd.DataFrame, stats: FileStats) -> pd.DataFrame:
    # Underlying must be > 0 and not absurdly huge
    if "underlying_price" in df.columns:
        up = df["underlying_price"]
        # Create mask for invalid prices
        mask_bad_up = (up <= 0) | (up > 1e7)
        
        # Record stats
        stats.bad_underlying_prices = int(mask_bad_up.sum())
        
        # Null the values
        df.loc[mask_bad_up, "underlying_price"] = np.nan

    # Strike must be > 0 and not absurdly huge
    if "strike" in df.columns:
        k = df["strike"]
        # Create mask for invalid strikes
        mask_bad_k = (k <= 0) | (k > 1e7)
        
        # Record stats
        stats.bad_strike_prices = int(mask_bad_k.sum())
        
        # Null the values
        df.loc[mask_bad_k, "strike"] = np.nan

    return df


def add_spread(df: pd.DataFrame) -> pd.DataFrame:
    required = {"best_bid_price", "best_ask_price", "mid_price"}
    
    # 1. Early exit if columns are missing
    if not required.issubset(df.columns):
        df["relative_spread"] = np.nan
        df["absolute_spread"] = np.nan
        return df

    bid = df["best_bid_price"]
    ask = df["best_ask_price"]
    mid = df["mid_price"]

    # 2. Initialize both series with NaN
    rel = pd.Series(np.nan, index=df.index, dtype="float64")
    abso = pd.Series(np.nan, index=df.index, dtype="float64")

    # 3. Create Mask
    # Checks that Bid/Ask/Mid exist, and Mid is not 0 (to avoid division error)
    mask = (
        bid.notna()
        & ask.notna()
        & mid.notna()
        & (mid > 0)
    )

    # 4. Calculate
    # We calculate the difference (absolute) first, then use it for relative
    diff = ask[mask] - bid[mask]
    
    abso.loc[mask] = diff
    rel.loc[mask] = diff / mid[mask]

    # 5. Assign to DataFrame
    df["absolute_spread"] = abso
    df["relative_spread"] = rel
    
    return df

def add_additional_columns(df):
    """
    Adds a few columns I wanted
    """
    # 1. Convert Expiry string to Unix Timestamp (Seconds)
    # This assumes expiry is at 08:00:00 UTC of the given date.
    df['expiry_dt'] = pd.to_datetime(df['expiry'])
    df['expiry_unix'] = (df['expiry_dt'].astype('int64') // 10**9) + 28800
    
    # Drop the temporary datetime column
    df.drop(columns=['expiry_dt'], inplace=True)

    # 2. Calculate Moneyness
    # Formula: Underlying Price / Strike Price
    df['moneyness'] = df['underlying_price'] / df['strike']

    # 3. Calculate Log Moneyness
    # Formula: ln(Moneyness)
    df['log_moneyness'] = np.log(df['moneyness'])

    # 4. Calculate Time to Expiry (Annualized)
    # Formula: (Expiry Unix - Current Unix) / Seconds in a Year
    seconds_per_year = 365 * 24 * 60 * 60  # 31,536,000
    df['time_to_expiry'] = (df['expiry_unix'] - df['unix_timestamp']) / seconds_per_year

    return df

def reorder_final_columns(df):
    """
    Reorders columns logically: ID -> Time -> Price -> Spreads -> Analysis -> Others
    """
    # 1. Define your ideal order
    desired_order = [
        # --- IDENTITY ---
        'instrument_name', 
        'expiry', 
        'type', 
        'strike',
        
        # --- TIME ---
        'timestamp', 
        'unix_timestamp', 
        'expiry_unix', 
        'time_to_expiry',
        
        # --- MARKET PRICES ---
        'best_bid_price', 
        'best_ask_price', 
        'mid_price', 
        'mark_price', 
        'last',
        
        # --- SPREADS (New) ---
        'absolute_spread', 
        'relative_spread',
        
        # --- ANALYSIS / UNDERLYING ---
        'estimated_delivery_price',
        'underlying_price', 
        'moneyness', 
        'log_moneyness',
        
        # --- METADATA ---
        'open_interest', 
        'volume'
    ]
    
    # 2. Filter the desired list to only include columns that actually exist in df
    # (This prevents errors if a column is missing)
    cols_to_keep = [c for c in desired_order if c in df.columns]
    
    # 3. Find any columns in df that are NOT in our desired list
    # (This ensures we don't accidentally delete data we forgot to list)
    remaining_cols = [c for c in df.columns if c not in cols_to_keep]
    
    # 4. Combine and reorder
    new_order = cols_to_keep + remaining_cols
    
    return df[new_order]

def analyze_time_gaps(df: pd.DataFrame, stats: FileStats) -> None:
    """
    Check minute coverage per instrument: how many minutes we expect vs how many we actually have.
    This is a *check only* – it does not modify df, just updates stats.
    """
    if "timestamp" not in df.columns or df.empty:
        return

    # Floor timestamps to minute (since you expect minute data)
    ts_floor = df["timestamp"].dt.floor("min")

    if "instrument_name" in df.columns:
        groups = df.groupby("instrument_name")
    else:
        # Treat the whole file as one instrument if no column
        groups = [(None, df)]

    for _, g in groups:
        g_ts = ts_floor.loc[g.index]
        if g_ts.isna().all():
            continue

        t_min = g_ts.min()
        t_max = g_ts.max()
        # Expected minute grid, inclusive
        expected = pd.date_range(t_min, t_max, freq="1min")
        expected_n = len(expected)
        actual_n = g_ts.nunique()

        stats.expected_minutes += expected_n
        stats.missing_minutes += max(expected_n - actual_n, 0)


def clean_bid_ask_mid(df: pd.DataFrame, stats: FileStats) -> pd.DataFrame:
    """Sanity checks on bid/ask + mid; handle negatives, swaps, and missing mids."""
    if "best_bid_price" not in df.columns or "best_ask_price" not in df.columns:
        return df

    # 1) Negative prices -> set to NaN
    for col in PRICE_COLS:
        if col in df.columns:
            df.loc[df[col] < 0, col] = np.nan

    # 2) Fix bid > ask where both are present by swapping them
    mask_swap = (
        df["best_bid_price"].notna()
        & df["best_ask_price"].notna()
        & (df["best_bid_price"] > df["best_ask_price"])
    )
    stats.fixed_bid_ask_swaps = int(mask_swap.sum())

    if stats.fixed_bid_ask_swaps > 0:
        bid_tmp = df.loc[mask_swap, "best_bid_price"].copy()
        df.loc[mask_swap, "best_bid_price"] = df.loc[mask_swap, "best_ask_price"]
        df.loc[mask_swap, "best_ask_price"] = bid_tmp

    # 3) Compute mid price where missing
    if "mid_price" in df.columns:
        needs_mid = (
            df["mid_price"].isna()
            & df["best_bid_price"].notna()
            & df["best_ask_price"].notna()
        )
        df.loc[needs_mid, "mid_price"] = (
            df.loc[needs_mid, "best_bid_price"]
            + df.loc[needs_mid, "best_ask_price"]
        ) / 2.0

    return df

def apply_noarbitrage_checks(df: pd.DataFrame, stats: FileStats) -> pd.DataFrame:

    LOWER_BOUND_FACTOR = 0.75

    required_cols = {"underlying_price", "strike", "type", "instrument_name"}
    if not required_cols.issubset(df.columns):
        return df

    # Build price series
    price = pd.Series(np.nan, index=df.index)
    if "mid_price" in df.columns: price = df["mid_price"].copy()
    if "mark_price" in df.columns: price = price.fillna(df["mark_price"])   

    S = df["underlying_price"]
    K = df["strike"]
    typ = df["type"].astype(str).str.lower().str.strip()
    
    avg_price = price.mean()
    avg_underlying = S.mean()

    is_quoted_in_coin = True

    mask_valid = price.notna() & S.notna() & K.notna() & typ.notna()

    # --- PUTS ---
    mask_put = mask_valid & (typ == "put")
    if mask_put.any():
        S_p = S[mask_put]
        K_p = K[mask_put]
        p_p = price[mask_put]

        upper_bound = (K_p / S_p)
            
        upper_violation_p = p_p > (upper_bound * (1.0 + NOARB_UPPER_TOL))

        idx_upper = p_p[upper_violation_p].index
        stats.noarb_upper_violations += len(idx_upper)
        if "mid_price" in df.columns:
            df.loc[idx_upper, "mid_price"] = np.nan

        # Soft Lower Bound Check
        # Intrinsic: (K/S) - 1
        intrinsic_p = (K_p / S_p) - 1.0
        
        # Only check if intrinsic is significantly positive
        mask_itm = intrinsic_p > 0.01 
        
        # VIOLATION IF: Price < (Intrinsic * Safety Factor)
        lower_violation_p = mask_itm & (p_p < (intrinsic_p * LOWER_BOUND_FACTOR))

        idx_lower = p_p[lower_violation_p].index
        stats.noarb_lower_violations += len(idx_lower) 
        if "mid_price" in df.columns:
            df.loc[idx_lower, "mid_price"] = np.nan

    # --- CALLS ---
    mask_call = mask_valid & (typ == "call")
    if mask_call.any():
        S_c = S[mask_call]
        K_c = K[mask_call]
        p_c = price[mask_call]
        
        upper_bound = 1.0
            
        upper_violation_c = p_c > upper_bound * (1.0 + NOARB_UPPER_TOL)
        idx_upper = p_c[upper_violation_c].index
        stats.noarb_upper_violations += len(idx_upper)
        if "mid_price" in df.columns:
            df.loc[idx_upper, "mid_price"] = np.nan
        
        # Soft Lower Bound Check
        # Intrinsic: 1 - (K/S)
        intrinsic_c = 1.0 - (K_c / S_c)
        
        mask_itm = intrinsic_c > 0.01
        
        # VIOLATION IF: Price < (Intrinsic * Safety Factor)
        lower_violation_c = mask_itm & (p_c < (intrinsic_c * LOWER_BOUND_FACTOR))

        idx_lower = p_c[lower_violation_c].index
        stats.noarb_lower_violations += len(idx_lower)
        if "mid_price" in df.columns:
            df.loc[idx_lower, "mid_price"] = np.nan

    return df


def sanitize_volume_oi(df: pd.DataFrame, stats: FileStats) -> pd.DataFrame:
    """Ensure volume and open_interest are non-negative; set negatives to NaN and count them."""
    if "open_interest" in df.columns:
        mask_neg_oi = df["open_interest"] < 0
        stats.negative_open_interest = int(mask_neg_oi.sum())
        df.loc[mask_neg_oi, "open_interest"] = np.nan

    if "volume" in df.columns:
        mask_neg_vol = df["volume"] < 0
        stats.negative_volume = int(mask_neg_vol.sum())
        df.loc[mask_neg_vol, "volume"] = np.nan

    return df

def compute_liquidity_stats(df: pd.DataFrame, stats: FileStats) -> None:
    """
    Compute simple per-file liquidity stats:
        - quote_minutes: number of distinct minutes with a valid bid+ask
        - core_quote_minutes: number of distinct minutes with bid+ask,
          and mid_price available.
    """
    if "timestamp" not in df.columns:
        return
    if "best_bid_price" not in df.columns or "best_ask_price" not in df.columns:
        return
    if df.empty:
        return

    # Floor timestamps to minute for per-minute stats
    ts_floor = df["timestamp"].dt.floor("min")

    has_quote = df["best_bid_price"].notna() & df["best_ask_price"].notna()
    if has_quote.any():
        stats.quote_minutes = int(ts_floor[has_quote].nunique())

    core_mask = has_quote
    if "mid_price" in df.columns:
        core_mask = core_mask & df["mid_price"].notna()

    if core_mask.any():
        stats.core_quote_minutes = int(ts_floor[core_mask].nunique())



def fix_unix_timestamp(df: pd.DataFrame, stats: FileStats) -> pd.DataFrame:
    """Ensure unix_timestamp matches timestamp (seconds); fix when clearly different."""
    if "timestamp" not in df.columns or "unix_timestamp" not in df.columns:
        return df

    # Compute unix seconds from timestamp
    # pandas datetime64[ns, UTC] -> int64 ns; divide by 1e9
    ts_unix = (df["timestamp"].astype("int64") // 10**9).astype("Int64")

    diff = (df["unix_timestamp"] - ts_unix).abs()
    # diff can be NaN; treat as mismatches only when > tolerance
    mismatch_mask = diff > UNIX_TS_TOLERANCE_SECONDS

    stats.fixed_unix_timestamp = int(mismatch_mask.sum())
    df.loc[mismatch_mask, "unix_timestamp"] = ts_unix[mismatch_mask]
    return df


def parse_instrument_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    From instrument_name like 'BTC-9NOV25-92000-C',
    derive expiry (date), strike, and type ('call'/'put').
    """
    if "instrument_name" not in df.columns:
        return df

    # splits from right: Type, Strike, Expiry, <Rest is Underlying>
    parts = df["instrument_name"].astype(str).str.rsplit("-", n=3, expand=True)
    
    # Verify we actually got 4 columns (Indices: 0, 1, 2, 3)
    if parts.shape[1] < 4:
        # If splitting failed (e.g. name is "BTC-PERPETUAL"), return gracefully
        return df

    # Because we rsplit, index 0 is underlying, 1 expiry, 2 strike, 3 cp
    df["_instr_expiry_code"] = parts[1]
    df["_instr_strike_code"] = parts[2]
    df["_instr_cp_code"] = parts[3]

    df["_instr_expiry"] = pd.to_datetime(
        df["_instr_expiry_code"].str.upper(),
        format="%d%b%y",
        errors="coerce",
    )

    df["_instr_strike"] = pd.to_numeric(
        df["_instr_strike_code"], errors="coerce"
    )

    cp = df["_instr_cp_code"].astype(str).str.upper().str.strip()
    type_map = {"C": "call", "P": "put"}
    df["_instr_type"] = cp.map(type_map)

    return df


def reconcile_instrument_columns(df: pd.DataFrame, stats: FileStats) -> pd.DataFrame:
    """
    Use parsed instrument metadata to fix type/strike/expiry columns when they disagree.
    """
    df = parse_instrument_metadata(df)

    # Fix type
    if "type" in df.columns and "_instr_type" in df.columns:
        df["type"] = df["type"].fillna(df["_instr_type"])
        mask = df["_instr_type"].notna() & (df["type"] != df["_instr_type"])
        stats.instrument_type_fixes = int(mask.sum())
        df.loc[mask, "type"] = df.loc[mask, "_instr_type"]

    # Fix strike
    if "strike" in df.columns and "_instr_strike" in df.columns:
        df["strike"] = df["strike"].fillna(df["_instr_strike"])
        diff = (df["strike"] - df["_instr_strike"]).abs()
        mask = df["_instr_strike"].notna() & (diff > 1e-6)
        stats.instrument_strike_fixes = int(mask.sum())
        df.loc[mask, "strike"] = df.loc[mask, "_instr_strike"]

    # Fix expiry
    if "expiry" in df.columns and "_instr_expiry" in df.columns:
        expiry_dates = df["expiry"].dt.date
        instr_exp_dates = df["_instr_expiry"].dt.date
        mask = df["_instr_expiry"].notna() & (expiry_dates != instr_exp_dates)
        stats.instrument_expiry_fixes = int(mask.sum())
        df["_instr_expiry"] = pd.to_datetime(df["_instr_expiry"], utc=True) + pd.Timedelta(hours=8)
        df.loc[mask, "expiry"] = df.loc[mask, "_instr_expiry"]
        pass

    # Drop helper columns
    helper_cols = [
        "_instr_underlying",
        "_instr_expiry_code",
        "_instr_strike_code",
        "_instr_cp_code",
        "_instr_expiry",
        "_instr_strike",
        "_instr_type",
    ]
    existing_helpers = [c for c in helper_cols if c in df.columns]
    if existing_helpers:
        df = df.drop(columns=existing_helpers)

    return df


# --------- Per-file clean pipeline --------- #


def clean_single_file(path: Path, output_dir: Path) -> FileStats:
    """Load, clean, and save a single CSV file; return stats."""
    df = pd.read_csv(path)

    stats = FileStats(
        filename=path.name,
        rows_before=len(df),
        rows_after=0, dropped_duplicates=0, dropped_bad_timestamps=0,
        fixed_bid_ask_swaps=0, fixed_unix_timestamp=0, instrument_type_fixes=0,
        instrument_strike_fixes=0, instrument_expiry_fixes=0, expected_minutes=0,
        missing_minutes=0,
        noarb_upper_violations=0, noarb_lower_violations=0, quote_minutes=0,
        core_quote_minutes=0, negative_open_interest=0, negative_volume=0,
    )

    # 1. Basics
    df = coerce_dtypes(df)
    df = drop_bad_timestamps(df, stats)
    
    # 2. Reconcile Metadata immediately
    df = reconcile_instrument_columns(df, stats)
    
    # 3. Sort and remove dupes
    df = sort_rows(df)
    df = drop_duplicates(df, stats)

    # 4. Time checks
    analyze_time_gaps(df, stats)
    df = fix_unix_timestamp(df, stats)

    # 5. Price Cleaning
    df = sanity_check_underlying_and_strike(df, stats) 
    df = clean_bid_ask_mid(df, stats)
    
    # 6. No-arbitrage checks
    df = apply_noarbitrage_checks(df, stats)

    # 6.5 Spread + additional columns 
    df = add_spread(df)
    df = add_additional_columns(df)

    # 7. Volume / OI
    df = sanitize_volume_oi(df, stats)

    # Nice final ordering
    df = reorder_final_columns(df)

    # Extra stats
    compute_liquidity_stats(df, stats)

    stats.rows_after = len(df)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / path.name
    df.to_csv(out_path, index=False)

    return stats



import pandas as pd
import traceback

def process_directory(input_dir: Path, output_dir: Optional[Path] = None) -> List[FileStats]:
    """Process all .csv files in the input directory."""
    if output_dir is None:
        output_dir = input_dir / "cleaned"

    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return []

    all_stats: List[FileStats] = []

    print(f"Found {len(csv_files)} files. Starting processing...")

    try:
        for path in csv_files:
            try:
                print(f"Cleaning {path.name} ...")
                
                stats = clean_single_file(path, output_dir)
                all_stats.append(stats)
                
            except Exception as e:
                # If a specific file fails, print error but CONTINUE the loop
                print(f"!!! CRITICAL ERROR processing {path.name}: {e}")
                # Optional: print full traceback if you need deep debugging
                # traceback.print_exc()
                # ^ never needed to date, but useful for future debugging

    except KeyboardInterrupt:
        # Allows you to hit Ctrl+C to stop processing but still see stats
        print("\n\nProcessing interrupted by user. Generating summary for processed files...")

    # --- The Summary Block ---
    # This now runs even if errors occurred or you stopped it early.
    
    if not all_stats:
        print("No files were successfully processed.")
        return []

    print("\n" + "="*40)
    print(f"SUMMARY REPORT ({len(all_stats)} files processed)")
    print("="*40)

    summary_df = pd.DataFrame([vars(s) for s in all_stats])
    
    # Identify numeric columns for aggregation
    numeric_cols = summary_df.select_dtypes(include=['number']).columns
    
    # 1. Print Grand Totals
    print("\n--- GRAND TOTALS ---")
    print(summary_df[numeric_cols].sum())

    # 2. Print Per-File details (if not too many)
    if len(summary_df) < 50:
        print("\n--- PER FILE BREAKDOWN ---")
        # Set pandas options to ensure columns don't get hidden
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print(summary_df)
    else:
        # If huge number of files, save the stats to a CSV instead of printing wall of text
        stats_path = output_dir / "processing_summary.csv"
        summary_df.to_csv(stats_path, index=False)
        print(f"\nDetailed file stats saved to: {stats_path}")

    return all_stats


# --------- CLI entrypoint --------- #


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean BTC options CSV files in a folder."
    )
    parser.add_argument(
        "input_dir",
        help="Path to the folder containing raw CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        help="Path to output folder for cleaned CSVs (default: <input>/cleaned).",
        default=None,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else None
    )
    
    process_directory(input_dir, output_dir)

if __name__ == "__main__":
    main()

