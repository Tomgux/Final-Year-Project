"""
Validate Bates pricing model implementation against QuantLib.
Tests ranges match GA: moneyness [0.4, 3.0], TTM [1/365, 1 year].

Majority AI written (The QuantLib part), just as a final additional check !
"""

import QuantLib as ql
import pandas as pd
import numpy as np
from numba import njit
import math

print("QuantLib version:", ql.__version__)
print("Has BatesModel:", hasattr(ql, "BatesModel"))
print("Has BatesEngine:", hasattr(ql, "BatesEngine"))
print("Has BatesProcess:", hasattr(ql, "BatesProcess"))


# ============================================================================
# CONFIGURATION - matches svjGA3day ranges
# ============================================================================

# Gauss-Legendre Quadrature setup
DEG = 64
X_NODES, WEIGHTS = np.polynomial.legendre.leggauss(DEG)
UPPER_BOUND = 200
GL_NODES = 0.5 * UPPER_BOUND * X_NODES + 0.5 * UPPER_BOUND
GL_WEIGHTS = 0.5 * UPPER_BOUND * WEIGHTS

# Test parameters - from svjGA3day
S0 = 75000.0
r = 0.04
reference_params = (8.0, 0.3, 0.4, 1.8, -0.25, 5, -0.01, 0.25)

# Ranges matching svjGA3day exactly
TTM_MIN, TTM_MAX = 1/365, 365/365           # Full year
MONEYNESS_MIN, MONEYNESS_MAX = 0.4, 3.0     # K/F ratio
N_TTM_TESTS = 100                             # Sample 100 points across TTM
N_MONEYNESS_TESTS = 100                     # Sample moneyness grid
PRICE_FILTER_THRESHOLD = 1.0              # Skip prices below this threshold


# ============================================================================
# QUANTLIB INTERFACE
# ============================================================================

def ql_bates_call_price(S0, K, T, r, q, v0, kappa, theta, sigma, rho, lamb, muJ, delta):
    day_count = ql.Actual365Fixed()
    today = ql.Date(1, 1, 2025)
    ql.Settings.instance().evaluationDate = today
    n_days = int(round(T * 365))
    maturity = today + n_days
    actual_T = day_count.yearFraction(today, maturity)

    spot = ql.QuoteHandle(ql.SimpleQuote(S0))
    r_ts  = ql.YieldTermStructureHandle(ql.FlatForward(today, r, day_count))
    q_ts  = ql.YieldTermStructureHandle(ql.FlatForward(today, q, day_count))

    # Some builds use BatesProcess, others construct BatesModel off a HestonProcess + jump params.
    if hasattr(ql, "BatesProcess"):
        process = ql.BatesProcess(r_ts, q_ts, spot, v0, kappa, theta, sigma, rho, lamb, muJ, delta)
        model = ql.BatesModel(process)
    else:
        process = ql.HestonProcess(r_ts, q_ts, spot, v0, kappa, theta, sigma, rho)
        model = ql.BatesModel(process, lamb, muJ, delta)

    payoff = ql.PlainVanillaPayoff(ql.Option.Call, K)
    exercise = ql.EuropeanExercise(maturity)
    option = ql.VanillaOption(payoff, exercise)

    if hasattr(ql, "BatesEngine"):
        engine = ql.BatesEngine(model)
    else:
        raise RuntimeError("No BatesEngine in this QuantLib build.")

    option.setPricingEngine(engine)
    return option.NPV(), actual_T


# ============================================================================
# My implementation
# ============================================================================

@njit(fastmath=True)
def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

@njit(fastmath=True)
def bs_price(F, K, T, r, sigma):
    if sigma <= 1e-5: return max(np.exp(-r*T) * (F - K), 0.0)
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return np.exp(-r * T) * (F * norm_cdf(d1) - K * norm_cdf(d2))

@njit(fastmath=True)
def bs_put_price(F, K, T, r, sigma):
    if sigma <= 1e-5: return max(np.exp(-r*T) * (K - F), 0.0)
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return np.exp(-r * T) * (K * norm_cdf(-d2) - F * norm_cdf(-d1))

@njit
def bates_char_func_numba(phi, T, r, v0, kappa, theta, sigma, rho, lamb, muJ, delta, P_num):

    i = 1j
    phi_c = i * phi

    if P_num == 1:
        u = 0.5
        b = kappa - rho * sigma
    else:
        u = -0.5
        b = kappa

    alpha = kappa * theta
    sig2 = sigma * sigma

    khat = np.exp(muJ + 0.5 * delta * delta) - 1.0

    beta = b - rho * sigma * phi_c
    d_sq = beta * beta - 2.0 * sig2 * (u * phi_c + 0.5 * phi_c * phi_c)
    d = np.sqrt(d_sq)
    if d.real < 0.0:
        d = -d

    g = (beta - d) / (beta + d)
    exp_neg_dT = np.exp(-d * T)

    D = ((beta - d) / sig2) * ((1.0 - exp_neg_dT) / (1.0 - g * exp_neg_dT))

    log_term = np.log(1.0 - g * exp_neg_dT) - np.log(1.0 - g)

    C = (r - lamb * khat) * phi_c * T + (alpha / sig2) * ((beta - d) * T - 2.0 * log_term)

    jump_exp_component = phi_c * np.log(1.0 + khat) + (delta * delta) * (u * phi_c + 0.5 * phi_c * phi_c)
    
    jump_term = lamb * T * ((1.0 + khat) ** (u + 0.5)) * (np.exp(jump_exp_component) - 1.0)

    return np.exp(C + D * v0 + jump_term)


@njit(fastmath=True)
def compute_price_numba(S, F, K, T, r_riskfree, params, nodes, weights):
    kappa, v0, theta, sigma, rho, lamb, muJ, delta = params
    
    if sigma <= 1e-30 and lamb <= 1e-30:
        return bs_price(F, K, T, r_riskfree, np.sqrt(v0))

    r_drift = np.log(F / S) / T 
    
    int_P1 = 0.0
    int_P2 = 0.0
    
    for i in range(len(nodes)):
        phi = nodes[i]
        w = weights[i]
        
        if phi == 0: continue

        cf1 = bates_char_func_numba(phi, T, r_drift, v0, kappa, theta, sigma, rho, lamb, muJ, delta, 1)
        val1 = (np.exp(-1j * phi * np.log(K / S)) * cf1) / (1j * phi)
        int_P1 += w * val1.real

        cf2 = bates_char_func_numba(phi, T, r_drift, v0, kappa, theta, sigma, rho, lamb, muJ, delta, 2)
        val2 = (np.exp(-1j * phi * np.log(K / S)) * cf2) / (1j * phi)
        int_P2 += w * val2.real
        
    P1 = 0.5 + (1/np.pi) * int_P1
    P2 = 0.5 + (1/np.pi) * int_P2

    CallPrice = np.exp(-r_riskfree * T) * (F * P1 - K * P2)

    return max(CallPrice, 0)


# ============================================================================
# VALIDATION TEST
# ============================================================================

def generate_test_grid():
    """Generate test points matching svjGA3day data ranges."""
    # Time to expiry: log-spaced for better coverage of short-term options
    ttm_vals = np.logspace(np.log10(TTM_MIN), np.log10(TTM_MAX), N_TTM_TESTS)
    
    # Moneyness: linear grid from 0.4 to 3.0
    moneyness_vals = np.linspace(MONEYNESS_MIN, MONEYNESS_MAX, N_MONEYNESS_TESTS)
    
    return ttm_vals, moneyness_vals


def run_validation():
    """Compare implementation against QuantLib."""
    ttm_vals, moneyness_vals = generate_test_grid()
    
    results = {
        'ttm': [], 'moneyness': [], 'ql_price': [], 'impl_price': [],
        'abs_diff': [], 'rel_diff': [], 'ql_price_pct': []
    }
    
    kappa, v0, theta, sigma, rho, lamb, muJ, delta = reference_params
    q = 0.0  # Zero dividend yield
    
    n_tests = len(ttm_vals) * len(moneyness_vals)
    count = 0
    
    print(f"\nValidating across {n_tests} test points...\n")
    print(f"{'TTM':>8} {'K/F':>8} {'QL Price':>12} {'Impl Price':>12} {'Abs Diff':>12} {'Rel Diff':>10}")
    print("-" * 72)
    
    for T in ttm_vals:
        
        for moneyness in moneyness_vals:
            F_approx = S0 * np.exp(r * T)
            K = F_approx * moneyness
            
            # Calculate QuantLib price (returns actual T used due to day rounding)
            try:
                call_price_ql, actual_T = ql_bates_call_price(S0, K, T, r, q, v0, kappa, theta, sigma, rho, lamb, muJ, delta)
            except Exception as e:
                print(f"QL error at T={T:.4f}, K/F={moneyness:.2f}: {e}")
                continue

            # Now filter based on the actual QL price
            if call_price_ql < PRICE_FILTER_THRESHOLD:
                continue
            
            # Use the exact same T that QuantLib used
            F_T = S0 * np.exp(r * actual_T)
            
            # Custom implementation
            call_price_impl = compute_price_numba(S0, F_T, K, actual_T, r, reference_params, GL_NODES, GL_WEIGHTS)
            
            # Metrics
            abs_diff = abs(call_price_impl - call_price_ql)
            rel_diff = abs_diff / max(call_price_ql, 1e-8)
            
            results['ttm'].append(T)
            results['moneyness'].append(moneyness)
            results['ql_price'].append(call_price_ql)
            results['impl_price'].append(call_price_impl)
            results['abs_diff'].append(abs_diff)
            results['rel_diff'].append(rel_diff)
            results['ql_price_pct'].append(call_price_ql / S0 * 100)
            
            count += 1
            if count % 50 == 0 or count == 1:
                print(f"{T:8.4f} {moneyness:8.2f} {call_price_ql:12.6f} {call_price_impl:12.6f} {abs_diff:12.6f} {rel_diff:10.4%}")
    
    return pd.DataFrame(results)


def summarize_results(df):
    """Print validation summary statistics."""
    print("\n" + "=" * 72)
    print("VALIDATION SUMMARY")
    print("=" * 72)
    
    print(f"\nTests completed: {len(df)}")
    print(f"TTM range: {df['ttm'].min():.4f} to {df['ttm'].max():.4f} years")
    print(f"Moneyness range: {df['moneyness'].min():.2f} to {df['moneyness'].max():.2f}")
    
    print(f"\nPrice range (QL): ${df['ql_price'].min():.6f} to ${df['ql_price'].max():.6f}")
    print(f"Absolute difference:")
    print(f"  Max: ${df['abs_diff'].max():.6f}")
    print(f"  Mean: ${df['abs_diff'].mean():.6f}")
    print(f"  Median: ${df['abs_diff'].median():.6f}")
    print(f"  Std: ${df['abs_diff'].std():.6f}")
    
    print(f"\nRelative difference:")
    print(f"  Max: {df['rel_diff'].max():.4%}")
    print(f"  Mean: {df['rel_diff'].mean():.4%}")
    print(f"  Median: {df['rel_diff'].median():.4%}")
    print(f"  Std: {df['rel_diff'].std():.4%}")
    
    print(f"\n% tests with rel_diff < 0.1%: {(df['rel_diff'] < 0.001).sum() / len(df) * 100:.1f}%")
    print(f"% tests with rel_diff < 1.0%: {(df['rel_diff'] < 0.01).sum() / len(df) * 100:.1f}%")
    print(f"% tests with rel_diff < 5.0%: {(df['rel_diff'] < 0.05).sum() / len(df) * 100:.1f}%")
    
    # Breakdown by moneyness bucket
    print(f"\nBy moneyness bucket:")
    for bucket_low in [0.4, 1.0, 1.5, 2.0, 2.5]:
        bucket_high = min(bucket_low + 0.5, MONEYNESS_MAX)
        mask = (df['moneyness'] >= bucket_low) & (df['moneyness'] < bucket_high)
        if mask.sum() > 0:
            subset = df[mask]
            print(f"  [{bucket_low:.1f}, {bucket_high:.1f}): "
                  f"Mean rel_diff = {subset['rel_diff'].mean():.4%}, "
                  f"Max rel_diff = {subset['rel_diff'].max():.4%}")
    
    print("\n" + "=" * 72)
    return df

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("Bates Model Validation Against QuantLib")
    print("Reference params:", reference_params)
    print(f"Sample size: {N_TTM_TESTS} TTM points × {N_MONEYNESS_TESTS} moneyness points")
    
    df_results = run_validation()
    df_results = summarize_results(df_results)