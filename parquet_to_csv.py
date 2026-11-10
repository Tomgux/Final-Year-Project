# Converts all full.parquet files in DATA/data_full (from hetzner) to CSVs in data_full_csv (which I actually work with)

# Majority AI generated

import pandas as pd
from pathlib import Path

# Root folders
PARQUET_ROOT = Path("DATA/data_full")
CSV_ROOT = Path("finalminutedata_csv2")

def extract_metadata(parquet_path: Path):
    """
    Given a path like:
    data_full/BTC/expiry=YYYY-MM-DD/type=call/strike=ZZZZZZ/full.parquet

    return ("YYYY-MM-DD", "call", "ZZZZZZ")
    """
    expiry = opt_type = strike = None

    for part in parquet_path.parents:
        name = part.name
        if name.startswith("expiry="):
            expiry = name.split("=", 1)[1]
        elif name.startswith("type="):
            opt_type = name.split("=", 1)[1]
        elif name.startswith("strike="):
            strike = name.split("=", 1)[1]

    if not (expiry and opt_type and strike):
        raise ValueError(f"Could not extract metadata from path: {parquet_path}")

    return expiry, opt_type, strike


def main():
    # Make sure output folder exists
    CSV_ROOT.mkdir(exist_ok=True)

    parquet_files = list(PARQUET_ROOT.rglob("full.parquet"))
    print(f"Found {len(parquet_files)} full.parquet files")

    for i, pq in enumerate(parquet_files, start=1):
        try:
            expiry, opt_type, strike = extract_metadata(pq)
            out_name = f"{expiry}-{opt_type}-{strike}.csv"
            out_path = CSV_ROOT / out_name

            print(f"[{i}/{len(parquet_files)}] Converting {pq} -> {out_path}")

            df = pd.read_parquet(pq)
            df.to_csv(out_path, index=False)

        except Exception as e:
            # You can choose to raise here instead if you don't want to skip any failures
            print(f"ERROR processing {pq}: {e}")

if __name__ == "__main__":
    main()
