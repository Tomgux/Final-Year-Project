# This script takes all the collected minute data and picks only X rows per day, and then calculates IV (and vega but it isn't used) for each row using the Black-76 model.
# Output is a "trainingdata.csv" with only a few rows per day, and relevant columns + IV, to be used to learn Q parameters for the Bates model.

import pandas as pd
import glob
import os
import numpy as np
from scipy.stats import norm

def black_76_price(F, K, T, r, sigma, option_type):
    """ Calculates the Black-76 price for a European option. F is the forward price, K is the strike, T is time to expiry, r is risk-free rate, sigma is volatility, and option_type is 'call' or 'put'. """
    if sigma <= 0 or T <= 0:
        return 0.0
        
    # Standard formula
    d1 = (np.log(F / K) + (0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    discount_factor = np.exp(-r * T)
    
    # Returns price dependant on option type
    if option_type == 'call':
        return discount_factor * (F * norm.cdf(d1) - K * norm.cdf(d2))
    else: 
        return discount_factor * (K * norm.cdf(-d2) - F * norm.cdf(-d1))

def solve_iv(row):
    """Calculates IV for some row"""
    # Relevant data from the row
    price = row['mark_price'] * row['estimated_delivery_price']
    F = row['underlying_price']
    K = row['strike']
    T = row['time_to_expiry']
    
    # Finds option type
    otype_str = str(row['type']).lower().strip()
    option_type = 'call' if 'c' in otype_str else 'put'
    
    r = 0.04 # Assuming 4% risk-free rate
    
    if T <= 0.001: return np.nan # Ignores tiny T
    if price <= 0 or F <= 0 or K <= 0: return np.nan

    # Solver (bisection)
    def objective(sigma):
        return black_76_price(F, K, T, r, sigma, option_type) - price

    # Runs 100 times for every option to ensure accurate estimate.
    low = 1e-12
    high = 500.0
    tol = 0
    max_iter = 100

    f_low = objective(low)
    f_high = objective(high)

    # No bracket -> no valid IV in this range
    if np.isnan(f_low) or np.isnan(f_high) or (f_low * f_high > 0):
        return np.nan

    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        f_mid = objective(mid)

        if np.isnan(f_mid):
            return np.nan

        if abs(f_mid) < tol or (high - low) < tol:
            return mid

        if f_low * f_mid <= 0:
            high = mid
            f_high = f_mid
        else:
            low = mid
            f_low = f_mid

    return 0.5 * (low + high)

# Main function to process all data and save to trainingdata.csv
def process_options_data(input_folder, output_file='DATA/trainingdata.csv'):
    # Defines columns to keep and their mapping (currently it's 1:1)
    column_mapping = {
        'strike': 'strike',
        'type': 'type',
        'timestamp': 'timestamp', 
        'unix_timestamp': 'unix_timestamp',
        'time_to_expiry': 'time_to_expiry',
        'mark_price': 'mark_price',
        'estimated_delivery_price': 'estimated_delivery_price',
        'underlying_price': 'underlying_price', 
        'moneyness': 'moneyness',
        'instrument_name': 'instrument_name',
        'best_bid_price': 'bid',
        'best_ask_price': 'ask'
    }

    print(f"Scanning folder: {input_folder}...")
    all_files = glob.glob(os.path.join(input_folder, "*.csv"))
    
    if not all_files:
        print("No CSV files found in the specified folder.")
        return

    # day_seconds determines how many snapshots are taken per day, with 86400 being 1 per day, 43200 being 2 per day, etc.
    anchor_ts = 1765080000 # Sun Dec 07 2025 04:00:00 GMT+0000
    day_seconds = 86400/24 # every hour
    tolerance_seconds = 20

    df_list = []
    total_raw_rows = 0 
    print(f"Found {len(all_files)} files. Loading and filtering data...")
    
    for filename in all_files:
        try:
            df = pd.read_csv(filename, usecols=column_mapping.keys())
            total_raw_rows += len(df) 

            # Removes options expiring too soon
            if 'time_to_expiry' in df.columns:
                df = df[df['time_to_expiry'] >= (0.5/365)]

            # Filter according to timestamp 
            if 'unix_timestamp' in df.columns:
                remainder_86400 = (df['unix_timestamp'] - anchor_ts) % 86400
                target_seconds = [i * 3600 for i in [19, 20, 21]] # 19:00, 20:00, 21:00
                mask = False
                for target in target_seconds:
                    mask |= (abs(remainder_86400 - target) <= tolerance_seconds)
                df = df[mask]

            # Only append if data remains after both filters
            if not df.empty:
                df_list.append(df)
                
        except ValueError as e:
            print(f"Skipping {filename}: Missing required columns. ({e})")

    if not df_list:
        print("No valid data loaded (or all data filtered out).")
        return

    # Create one dataframe
    daily_df = pd.concat(df_list, ignore_index=True)

    # Rename columns according to mapping
    rename_map = {k: v for k, v in column_mapping.items() if k != v}
    daily_df.rename(columns=rename_map, inplace=True)
    
    if daily_df.empty:
        print("Warning: No rows matched the specific timestamp criteria.")
        return

    daily_df.reset_index(drop=True, inplace=True)

    # Calculate IV for each row
    print(f"Calculating Implied Volatility (IV) for {len(daily_df)} filtered rows...")
    daily_df['IV'] = daily_df.apply(solve_iv, axis=1)

    print("Calculating vega for each row...")

    # Vectorized Vega calculation
    valid_mask = (daily_df['IV'] > 0) & (daily_df['time_to_expiry'] > 0) & daily_df['IV'].notna()

    F = daily_df.loc[valid_mask, 'underlying_price']
    K = daily_df.loc[valid_mask, 'strike']
    T = daily_df.loc[valid_mask, 'time_to_expiry']
    sigma = daily_df.loc[valid_mask, 'IV']
    r = 0.04

    # Calculate d1 vectorized
    d1 = (np.log(F / K) + (0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    # Calculate vega vectorized: V = F * N'(d1) * sqrt(T) * e^{-rT}
    vega_values = F * norm.pdf(d1) * np.sqrt(T) * np.exp(-r * T)

    # Assign back to dataframe, leaving NaNs where the mask was False
    daily_df['vega'] = np.nan
    daily_df.loc[valid_mask, 'vega'] = vega_values

    # add section column
    remainder_86400 = (daily_df['unix_timestamp'] - anchor_ts) % 86400
    for i in range(19, 22):
        daily_df.loc[abs(remainder_86400 - i*86400/24) <= tolerance_seconds, 'section'] = i 

    # Filter for final output columns
    final_columns = [
        'strike', 
        'type',
        'timestamp', 
        'unix_timestamp',
        'time_to_expiry', 
        'mark_price', 
        'estimated_delivery_price', 
        'underlying_price', 
        'bid',
        'ask',
        'moneyness',
        'IV',
        'section',
        'vega',
    ]
    
    # Reorder columns
    available_cols = [c for c in final_columns if c in daily_df.columns]
    final_output = daily_df[available_cols]

    # Save to CSV
    final_output.to_csv(output_file, index=False)
    print(f"Success! Processed {total_raw_rows} raw rows into {len(final_output)} target samples.")
    print(f"Saved to: {output_file}")

# Actual execution
folder_path = 'DATA/finalminutedata_csv2/cleaned' 

if __name__ == "__main__":
    process_options_data(folder_path)