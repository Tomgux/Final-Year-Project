"""
Options Data Validator — part-file strike folders

Validates an options dataset stored like:

 data/
  └── BTC/
       ├── 2025-10-08/
       │     ├── expiry=2025-10-25/
       │     │     ├── type=call/
       │     │     │     ├── strike=40000/
       │     │     │     │    ├── part-20251019T093900.parquet
       │     │     │     │    └── part-20251019T094000.parquet
       │     │     │     ├── strike=41000/
       │     │     │     └── strike=42000/
       │     │     └── type=put/
       │     │           ├── strike=40000/
       │     │           └── strike=41000/
       │     └── expiry=2025-11-01/
       │           └── ...
       └── 2025-10-09/
             └── ...

Checks:
  1) Directory structure (underlying/day/expiry/type/strike) for each *part* file.
  2) File readability & essential columns.
  3) Timestamp parse, monotonicity, duplicates, and (optional) day alignment.
  4) Logical bounds/relationships (bid<=ask, IV/delta ranges, non-negatives).
  5) Path vs content consistency (strike matches folder; delta vs call/put).
  6) Statistical outliers (robust z + IQR) per numeric column.

Outputs:
  - CSV summary (one row per part-file issue) via --report
  - Optional JSONL per-file detailed report via --jsonl
  - Exit code 1 if any ERRORs, else 0

Usage:
  python options_data_validator.py \
      --root data \
      --report issues.csv \
      --jsonl issues.jsonl \
      --max-rows 200000 \
      --allow-cross-day

      Use with python Current_Project/data_validator.py --root data --report issues.csv --jsonl issues.jsonl --max-rows 200000 --allow-cross-day

Notes:
  • --root can be the dataset root (e.g. 'data') or 'data/BTC'; underlying will be inferred if present.

Original data validator script to check the parquet files being collected from data_collection.py at the very start. 
Just to ensure the data being collected was valid.

Majority AI generated.
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, date
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype


# ---------------- Path extraction (robust to --root = data or data/BTC) ----------------

@dataclass
class PathInfo:
    path: Path              # the part file
    underlying: Optional[str]
    day: Optional[str]      # YYYY-MM-DD
    expiry: Optional[str]   # YYYY-MM-DD (from 'expiry=...')
    otype: Optional[str]    # 'call' or 'put'
    strike: Optional[float] # from 'strike=...'


def extract_path_info(root: Path, part_file: Path) -> PathInfo:
    underlying = None
    day = None
    expiry = None
    otype = None
    strike_val: Optional[float] = None

    for parent in part_file.parents:
        # stop above root
        try:
            parent.relative_to(root)
        except ValueError:
            break

        name = parent.name
        if name.startswith("strike=") and strike_val is None:
            try:
                strike_val = float(name.split("=", 1)[1])
            except Exception:
                strike_val = None
        elif name.startswith("type=") and otype is None:
            maybe = name.split("=", 1)[1]
            if maybe in ("call", "put"):
                otype = maybe
        elif name.startswith("expiry=") and expiry is None:
            exp = name.split("=", 1)[1]
            if re.match(r"^\d{4}-\d{2}-\d{2}$", exp):
                expiry = exp
        elif re.match(r"^\d{4}-\d{2}-\d{2}$", name) and day is None:
            day = name

        # underlying = first-level child of root
        try:
            rel = parent.relative_to(root)
            if len(rel.parts) == 1 and underlying is None:
                underlying = rel.parts[0]
        except ValueError:
            pass

    return PathInfo(
        path=part_file, underlying=underlying, day=day,
        expiry=expiry, otype=otype, strike=strike_val
    )


# -------------------------------- Reporting structures --------------------------------

@dataclass
class Issue:
    level: str  # 'ERROR' | 'WARN' | 'INFO'
    code: str
    message: str
    row_count: Optional[int] = None

@dataclass
class FileReport:
    info: PathInfo
    issues: List[Issue] = field(default_factory=list)
    n_rows: int = 0
    n_cols: int = 0
    columns: List[str] = field(default_factory=list)

    def add(self, level: str, code: str, message: str, row_count: Optional[int] = None):
        self.issues.append(Issue(level, code, message, row_count))

    @property
    def has_errors(self) -> bool:
        return any(i.level == "ERROR" for i in self.issues)


# -------------------------------- Rules & helpers --------------------------------

ESSENTIAL_COLUMNS = ["timestamp"]

OPTIONAL_COLUMNS_RULES = {
    "price": {"min": 0},
    "mark_price": {"min": 0},
    "best_bid_price": {"min": 0},
    "best_ask_price": {"min": 0},
    "bid_iv": {"min": 0, "max": 10},
    "ask_iv": {"min": 0, "max": 10},
    "mark_iv": {"min": 0, "max": 10},
    "delta": {"min": -1.25, "max": 1.25},
    "gamma": {"min": 0},
    "vega": {"min": 0},
    "theta": {},
    "rho": {},
    "underlying_price": {"min": 0},
    "open_interest": {"min": 0},
    "volume": {"min": 0},
    "strike": {"min": 0},
    "best_bid_amount": {"min": 0},
    "best_ask_amount": {"min": 0},
}

def robust_outlier_mask(x: np.ndarray, z_thresh: float = 6.0) -> np.ndarray:
    x = x.astype(float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if not np.isfinite(mad) or mad == 0:
        return np.zeros_like(x, dtype=bool)
    z = 0.6745 * (x - med) / mad
    return np.abs(z) > z_thresh

def iqr_outlier_mask(x: np.ndarray, k: float = 3.0) -> np.ndarray:
    q1 = np.nanpercentile(x, 25)
    q3 = np.nanpercentile(x, 75)
    iqr = q3 - q1
    if not np.isfinite(iqr) or iqr <= 0:
        return np.zeros_like(x, dtype=bool)
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return (x < lower) | (x > upper)

def parse_iso_day(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except Exception:
        return None


# -------------------------------- Per-file checks --------------------------------

def check_part_file(root: Path, fpath: Path, allow_cross_day: bool, max_rows: Optional[int]) -> FileReport:
    pinfo = extract_path_info(root, fpath)
    rep = FileReport(info=pinfo)

    # path completeness
    if pinfo.day is None:
        rep.add("ERROR", "DAY_FOLDER", "Missing or invalid day folder (YYYY-MM-DD)")
    if pinfo.expiry is None:
        rep.add("ERROR", "EXPIRY_FOLDER", "Missing or invalid expiry folder 'expiry=YYYY-MM-DD'")
    if pinfo.otype not in ("call", "put"):
        rep.add("ERROR", "TYPE_FOLDER", "Missing or invalid type folder 'type=call|put'")
    if pinfo.strike is None:
        rep.add("ERROR", "STRIKE_FOLDER", "Missing or invalid strike folder 'strike=<number>'")
    if rep.has_errors:
        return rep

    # read parquet
    try:
        df = pd.read_parquet(fpath)
        if max_rows and len(df) > max_rows:
            df = df.head(max_rows)
    except Exception as e:
        rep.add("ERROR", "PARQUET_READ", f"Failed to read parquet: {e}")
        return rep

    rep.n_rows, rep.n_cols = df.shape
    rep.columns = list(df.columns)

    # essential columns
    for col in ESSENTIAL_COLUMNS:
        if col not in df.columns:
            rep.add("ERROR", "MISSING_COLUMN", f"Missing required column '{col}'")
    if rep.has_errors:
        return rep

    # timestamp checks
    ts = df["timestamp"]
    if not is_datetime64_any_dtype(ts):
        ts = pd.to_datetime(ts, errors="coerce", utc=True)

    null_ts = int(ts.isna().sum())
    if null_ts > 0:
        rep.add("ERROR", "TIMESTAMP_NA", f"{null_ts} null/unparseable timestamps")
    else:
        if not ts.is_monotonic_increasing:
            rep.add("WARN", "TIMESTAMP_ORDER", "Timestamps are not monotonically non-decreasing")
        dup = int(ts.duplicated().sum())
        if dup > 0:
            rep.add("WARN", "TIMESTAMP_DUP", f"{dup} duplicate timestamps")

        folder_day = parse_iso_day(pinfo.day)
        if folder_day and not allow_cross_day:
            if getattr(ts.dt, "tz", None) is not None:
                day_only = ts.dt.tz_convert("UTC").dt.date
            else:
                day_only = ts.dt.date
            bad_day = int(np.sum(day_only != folder_day))
            if bad_day > 0:
                rep.add(
                    "WARN", "DAY_MISMATCH",
                    f"{bad_day} rows' date != folder day {pinfo.day} (use --allow-cross-day to suppress)"
                )

    # path vs content: strike
    if "strike" in df.columns and pinfo.strike is not None:
        try:
            svals = pd.to_numeric(df["strike"], errors="coerce")
            mism = int((np.abs(svals - pinfo.strike) > max(1e-6, 1e-8 * max(1.0, abs(pinfo.strike)))).sum())
            if mism > 0:
                rep.add("ERROR", "STRIKE_MISMATCH",
                        f"{mism} rows strike != folder strike {pinfo.strike}")
        except Exception:
            rep.add("WARN", "STRIKE_PARSE", "Could not parse 'strike' column as numeric for matching")

    # delta sanity vs type
    if "delta" in df.columns and pinfo.otype in ("call", "put"):
        d = pd.to_numeric(df["delta"], errors="coerce")
        if pinfo.otype == "call":
            bad = int(((d < -0.1) | (d > 1.1)).sum())
            if bad > 0:
                rep.add("WARN", "DELTA_RANGE_CALL", f"{bad} delta values unusual for calls (<-0.1 or >1.1)")
        else:
            bad = int(((d < -1.1) | (d > 0.1)).sum())
            if bad > 0:
                rep.add("WARN", "DELTA_RANGE_PUT", f"{bad} delta values unusual for puts (<-1.1 or >0.1)")

    # bid/ask relationships
    if "best_bid_price" in df.columns and "best_ask_price" in df.columns:
        bb = pd.to_numeric(df["best_bid_price"], errors="coerce")
        ba = pd.to_numeric(df["best_ask_price"], errors="coerce")
        inversions = int((bb > ba).sum())
        if inversions > 0:
            rep.add("WARN", "BID_GT_ASK", f"{inversions} rows where best_bid_price > best_ask_price")

        for mark_name in ("mark_price", "price"):
            if mark_name in df.columns:
                mk = pd.to_numeric(df[mark_name], errors="coerce")
                below_bid = int((mk + 1e-12 < bb).sum())
                above_ask = int((mk - 1e-12 > ba).sum())
                if below_bid + above_ask > 0:
                    rep.add("INFO", "MARK_OUTSIDE_SPREAD",
                            f"{below_bid} below bid, {above_ask} above ask for '{mark_name}'")

    # numeric rules + outliers
    for col, rules in OPTIONAL_COLUMNS_RULES.items():
        if col not in df.columns:
            continue
        s = df[col]
        if not pd.api.types.is_numeric_dtype(s):
            s = pd.to_numeric(s, errors="coerce")

        n_nan = int(s.isna().sum())
        if n_nan > 0:
            rep.add("WARN", "NAN_VALUES", f"{n_nan} NaN in '{col}'")

        finite_mask = np.isfinite(s.astype(float))
        n_non_finite = int((~finite_mask).sum())
        if n_non_finite > 0:
            rep.add("ERROR", "NON_FINITE", f"{n_non_finite} non-finite values in '{col}'")

        if "min" in rules:
            below = int((s < rules["min"]).sum())
            if below > 0:
                rep.add("ERROR", "MIN_BREACH", f"{below} values in '{col}' < {rules['min']}")
        if "max" in rules:
            above = int((s > rules["max"]).sum())
            if above > 0:
                rep.add("ERROR", "MAX_BREACH", f"{above} values in '{col}' > {rules['max']}")

        vals = s.astype(float).to_numpy()
        n_out = int(np.nansum(robust_outlier_mask(vals) | iqr_outlier_mask(vals)))
        if n_out > 0:
            rep.add("INFO", "OUTLIERS", f"{n_out} statistical outliers in '{col}' (robust/IQR)")

    # expiry consistency (soft)
    if "expiry" in df.columns and pinfo.expiry is not None:
        try:
            exp_str = df["expiry"].astype(str)
            mism = int((exp_str != str(pinfo.expiry)).sum())
            if mism > 0:
                rep.add("WARN", "EXPIRY_MISMATCH",
                        f"{mism} rows expiry != folder expiry {pinfo.expiry}")
        except Exception:
            rep.add("INFO", "EXPIRY_TYPE",
                    "Expiry column present with non-standard dtype (skipped strict match)")

    return rep


# -------------------------------- Runner --------------------------------

def find_part_files(root: Path) -> List[Path]:
    return [p for p in root.rglob("*.parquet") if p.is_file()]

def main():
    ap = argparse.ArgumentParser(description="Validate options part-file tree & data sanity")
    ap.add_argument("--root", type=str, required=True, help="Root folder, e.g. data or data/BTC")
    ap.add_argument("--report", type=str, default="issues.csv", help="CSV summary output path")
    ap.add_argument("--jsonl", type=str, default=None, help="Optional JSONL detailed output path")
    ap.add_argument("--max-rows", type=int, default=None, help="Limit rows read per file (for speed)")
    ap.add_argument("--allow-cross-day", action="store_true",
                    help="Don't warn if row dates differ from folder day")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        print(f"ERROR: root folder not found: {root}")
        raise SystemExit(2)

    files = find_part_files(root)
    if not files:
        print("No parquet files found under", root)
        raise SystemExit(2)

    reports: List[FileReport] = []
    for fp in sorted(files):
        rep = check_part_file(root, fp, allow_cross_day=args.allow_cross_day, max_rows=args.max_rows)
        reports.append(rep)

    # build CSV rows
    rows = []
    n_error_files = 0
    for rep in reports:
        if rep.has_errors:
            n_error_files += 1
        if rep.issues:
            for it in rep.issues:
                rows.append({
                    "file": str(rep.info.path),
                    "underlying": rep.info.underlying or "?",
                    "day": rep.info.day or "?",
                    "expiry": rep.info.expiry or "?",
                    "type": rep.info.otype or "?",
                    "strike": rep.info.strike,
                    "level": it.level,
                    "code": it.code,
                    "message": it.message,
                    "row_count": it.row_count,
                    "n_rows": rep.n_rows,
                    "n_cols": rep.n_cols,
                })
        else:
            rows.append({
                "file": str(rep.info.path),
                "underlying": rep.info.underlying or "?",
                "day": rep.info.day or "?",
                "expiry": rep.info.expiry or "?",
                "type": rep.info.otype or "?",
                "strike": rep.info.strike,
                "level": "OK",
                "code": "OK",
                "message": "No issues",
                "row_count": None,
                "n_rows": rep.n_rows,
                "n_cols": rep.n_cols,
            })

    out_csv = Path(args.report)
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    if args.jsonl:
        out_jsonl = Path(args.jsonl)
        with out_jsonl.open("w", encoding="utf-8") as f:
            for rep in reports:
                obj = {
                    "file": str(rep.info.path),
                    "underlying": rep.info.underlying,
                    "day": rep.info.day,
                    "expiry": rep.info.expiry,
                    "type": rep.info.otype,
                    "strike": rep.info.strike,
                    "n_rows": rep.n_rows,
                    "n_cols": rep.n_cols,
                    "columns": rep.columns,
                    "issues": [it.__dict__ for it in rep.issues],
                }
                f.write(json.dumps(obj) + "\n")

    total = len(reports)
    print(f"Checked {total} part files. Summary written to {out_csv}.")
    if args.jsonl:
        print(f"Detailed JSONL written to {args.jsonl}.")
    if n_error_files > 0:
        print(f"Files with ERRORs: {n_error_files}/{total}")
        raise SystemExit(1)

if __name__ == "__main__":
    main()
