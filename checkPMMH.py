# Checks the ability of the algorithm to recover likely parameters from synthetic returns data

import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import time

# Config
# mu, kappa, theta, sigma, rho, lamb, muJ, delta
TRUE_PARAMS = np.array([0.12, 8.0, 0.32, 1.8, -0.05, 18.0, 0.003, 0.06], dtype=np.float64)
FALSE_PARAMS = np.array([-0.08, 2.5, 0.12, 0.9, 0.45, 3.0, -0.04, 0.18], dtype=np.float64)
NUM_YEARS = 5.0
DAYS_PER_YEAR = 365
NUM_DAYS = int(NUM_YEARS * DAYS_PER_YEAR)
INITIAL_GUESSES = [0, 2, 0.3, 1.0, 0, 5.0, 0.0, 0.1]
ITERATIONS = 16000
PARTICLES = 4000
BURN_IN = 2000
DAYSOFRETURNSIGNORED = 20

# Variable bounds, in order we have mu, kappa, theta, sigma, rho, lamb, muJ, delta

BOUNDS = np.array([
    [-0.5, 0.5],    
    [ 0.1,  10.0],    
    [0.1, 1.0],
	[0.1, 3.0],
	[-0.5, 0.5],
	[0.1, 20.0],
	[-0.2, 0.2],
	[0.01, 0.5],
], dtype=float)

def log_prior(params):
    mu, kappa, theta, sigma, rho, lamb, muJ, delta = params
    
    if not (-0.5 <= mu <= 0.5): return -np.inf
    if not (0.1 <= kappa <= 10.0): return -np.inf 
    if not (0.1 <= theta <= 1.0): return -np.inf 
    if not (0.1 <= sigma <= 3.0): return -np.inf
    if not (-0.5 <= rho <= 0.5): return -np.inf 
    if not (0.1 <= lamb <= 20): return -np.inf   
    if not (-0.2 <= muJ <= 0.2): return -np.inf
    if not (0.01 <= delta <= 0.5): return -np.inf
    
    log_prob = 0.0

    return log_prob

# Below is the bulk of the code, main functions and helper functions

@njit
def simulate_svj_log_returns(params, num_days, burn_in_days=50):
    """Simulate one synthetic log-return path."""
    # unpack parameters
    mu, kappa, theta, sigma, rho, lamb, muJ, delta = params

    # time step and root time step
    dt = 1.0 / 365.0
    sqrt_dt = np.sqrt(dt)

    # initilizing particles using distribution recommended by Gemini. in reality, this is not that important !
    alpha_gamma = (2.0 * kappa * theta) / (sigma ** 2)
    beta_gamma = (sigma ** 2) / (2.0 * kappa)
    alpha_gamma = max(alpha_gamma, 1e-1)
    beta_gamma = max(beta_gamma, 1e-8)
    v_curr = np.random.gamma(shape=alpha_gamma, scale=beta_gamma)

    # results list
    total_steps = num_days + burn_in_days
    out = np.zeros(num_days)

    # simulate day by day, ignoring the first burn_in_days for the output to allow for burn in of the volatility process
    for t in range(total_steps):
        v_prev = max(v_curr, 1e-5)
        z1 = np.random.normal(0.0, 1.0)
        z2 = np.random.normal(0.0, 1.0)

        # simulate the next variance using Euler-Maruyama discretization 
        drift_v = kappa * (theta - v_prev) * dt
        diff_v = sigma * np.sqrt(v_prev) * sqrt_dt * z1
        v_curr = max(v_prev + drift_v + diff_v, 1e-5)

        rho2 = rho * rho

        # the following are found using the SVJ SDE + ito's lemma + correlation structure shown in John C. Hull's book "Options, Futures and Other Derivatives" (11th edition, 2022) 

        # log return from vol
        mean_diff = (mu - 0.5 * v_prev) * dt + np.sqrt(v_prev) * sqrt_dt * rho * z1
        std_diff = np.sqrt(v_prev) * sqrt_dt * np.sqrt(1.0 - rho2)
        diffusion_ret = mean_diff + std_diff * z2

        # log return from jumps
        n_jumps = np.random.poisson(lamb * dt)
        jump_ret = 0.0
        if n_jumps > 0:
            jump_ret = n_jumps * muJ + np.sqrt(n_jumps) * delta * np.random.normal(0.0, 1.0)

        # record log return if past burn in period
        if t >= burn_in_days:
            out[t - burn_in_days] = diffusion_ret + jump_ret

    return out

def sigmoid(z):
    """standard sigmoid function"""
    z = np.clip(z, -40, 40)
    return 1.0 / (1.0 + np.exp(-z))
 
def to_unconstrained(x, bounds=BOUNDS, eps=1e-12):
    """converts original parameter to an unconstrained space ([a, b] -> (-inf, inf)) using a logit transform"""
    a = bounds[:, 0]
    b = bounds[:, 1]
    u = (x - a) / (b - a)
    u = np.clip(u, eps, 1 - eps)
    return np.log(u) - np.log1p(-u)

def from_unconstrained(z, bounds=BOUNDS):
    """reverts to normal parameter space from the unconstrained space, using the inverse of the logit transform (sigmoid)"""
    a = bounds[:, 0]
    b = bounds[:, 1]
    s = sigmoid(z)
    return a + (b - a) * s

def log_jacobian(z, bounds=BOUNDS, eps=1e-12):
    """required for the acceptance ratio calculation to account for the change of variables from the constrained space to the unconstrained space"""
    a = bounds[:, 0]
    b = bounds[:, 1]
    s = sigmoid(z)
    s = np.clip(s, eps, 1 - eps)
    return np.sum(np.log(b - a) + np.log(s) + np.log1p(-s))

@njit(fastmath=True)
def fast_norm_pdf(x, mean, std):
    """standard normal pdf"""
    var = std**2
    denom = np.sqrt(2 * np.pi * var)
    num = np.exp(-0.5 * ((x - mean)**2) / var)
    return num / denom

@njit
def multinomial_resample_numba(weights):
    """resamples indices based on the given weights using multinomial resampling."""
    N = len(weights)
    cumulative_sum = np.cumsum(weights)
    cumulative_sum[-1] = 1.0  # ensure numerical stability 
    
    indices = np.zeros(N, dtype=np.int32)
    
    for i in range(N):
        u = np.random.random()
        # find the first index where the cumulative sum exceeds u
        indices[i] = np.searchsorted(cumulative_sum, u)
        
    return indices

# Runs the particle filter over the set of returns and calculates total log likelihood, and minimum ESS (for monitoring)
@njit(fastmath=True)
def run_particle_filter(params, returns, num_particles):
    """Runs a particle filter to estimate the log likelihood of the observed returns given the SVJ parameters, 
    and also tracks the minimum ESS across time steps for monitoring particle degeneracy."""
    # unpack parameters
    mu, kappa, theta, sigma, rho, lamb, muJ, delta = params
    
    # time step (1 day) and root time step
    T_step = 1/365.0
    N = len(returns)
    
    # Initialize particles for the variance process using the stationary distribution of the CIR process (gamma distribution)
    alpha_gamma = (2 * kappa * theta) / (sigma**2)
    beta_gamma = (sigma**2) / (2 * kappa)  
    alpha_gamma = max(alpha_gamma, 1e-1)
    beta_gamma  = min(max(beta_gamma, 1e-8), 1e3)
    particles_v = np.random.gamma(shape=alpha_gamma, scale=beta_gamma, size=num_particles)
    
    log_likelihood = 0.0
    min_ess = num_particles
    
    sqrt_dt = np.sqrt(T_step)

    # calculate the likelihood of the observed returns given the particles, and resample at each time step
    for t in range(N):
        rt = returns[t]
        
        v_prev = particles_v
        v_prev_plus = np.maximum(v_prev, 1e-5) 
        
        zv1 = np.random.normal(0,1,num_particles)

        drift_v = kappa * (theta - v_prev_plus) * T_step
        diff_v = sigma * np.sqrt(v_prev_plus) * sqrt_dt * zv1
        v_curr = np.maximum(v_prev + drift_v + diff_v, 1e-5)

        mean_diff = (mu - 0.5 * v_prev_plus) * T_step + np.sqrt(v_prev_plus) * sqrt_dt * rho * zv1
        rho2 = rho*rho
       
        std_diff = np.sqrt(v_prev_plus) * sqrt_dt * np.sqrt(1.0 - rho2)

        # probability of the observed return given no jump vs given a jump, using the normal distribution for the diffusion part and the jump part 
        pdf_diff = fast_norm_pdf(rt, mean_diff, std_diff + 1e-4)
        
        var_total = std_diff**2 + delta**2
        pdf_jump = fast_norm_pdf(rt, mean_diff + muJ, np.sqrt(var_total + 1e-9))
        
        prob_jump = 1.0 - np.exp(-lamb * T_step)
        prob_jump = min(prob_jump, 1.0)
        
        weights = (1 - prob_jump) * pdf_diff + prob_jump * pdf_jump 
        
        weights = np.maximum(weights, 1e-300)
        
        mean_weight = np.mean(weights)
        if t > (DAYSOFRETURNSIGNORED - 1):
            log_likelihood += np.log(mean_weight)

        norm_weights = weights / np.sum(weights)
        current_ess = 1.0 / np.sum(norm_weights**2)

        if current_ess < min_ess:
            min_ess = current_ess

        indices = multinomial_resample_numba(norm_weights)
        particles_v = v_curr[indices]
    
        particles_v = np.maximum(particles_v, 1e-300)

    return log_likelihood, min_ess

def run_adaptive_pmmh(iterations, initial_params, returns, particles):
    """Runs the adaptive Particle Marginal Metropolis Hastings algorithm"""
    
    D = len(initial_params)

    current_z = to_unconstrained(np.array(initial_params, dtype=float))
    current_x = from_unconstrained(current_z)

    current_log_lik, current_ess = run_particle_filter(current_x, returns, num_particles=particles)

    current_log_pr = log_prior(current_x)
    if not np.isfinite(current_log_pr):
        raise ValueError("Initial params are outside the prior support.")

    trace = np.zeros((iterations, D))
    trace_ess = np.zeros(iterations)
    accepted = 0
    
    m = current_z.copy()
    S = np.zeros((D, D))  # Tracks sum of squared deviations for exact covariance
    C = np.eye(D) * 0.01  # Initial covariance for the non-adaptive phase
    sd = (2.4**2) / D     # Optimal scaling factor per Haario et al. (2001)
    adapt_t = 0

    current_log_post = current_log_lik + current_log_pr + log_jacobian(current_z)

    start = time.time()

    # runs algorithm for the specified number of iterations
    for i in range(iterations):
      
        # After 1000 iterations, start using the empirical covariance of the samples for the proposal distribution
        if i > 1000: 
            proposal_cov = sd * (C + 1e-3 * np.eye(D)) 
            prop_z = np.random.multivariate_normal(current_z, proposal_cov)
        else:
            # else uses simple scales
            scales = np.array([0.01, 0.5, 0.03, 0.05, 0.05, 0.10, 0.1, 0.05])
            prop_z = current_z + np.random.normal(0, scales, D)

        prop_x = from_unconstrained(prop_z)

        prop_log_pr = log_prior(prop_x)

        if not np.isfinite(prop_log_pr):
            log_ratio = -np.inf
        else:
            prop_log_lik, prop_ess = run_particle_filter(prop_x, returns, particles)
            prop_log_post = prop_log_lik + prop_log_pr + log_jacobian(prop_z)
            log_ratio = prop_log_post - current_log_post

        if np.log(np.random.rand()) < log_ratio:
            current_z = prop_z
            current_x = prop_x
            current_log_lik = prop_log_lik
            current_ess = prop_ess
            current_log_post = prop_log_post
            accepted += 1
            
        trace[i] = current_x
        trace_ess[i] = current_ess
        
        if 100 < i < BURN_IN: 
            adapt_t += 1
            # Haario et al. (2001) adaptive covariance update
            dm = current_z - m
            m = m + dm / adapt_t
            # Welford's update for the sum of squared deviations
            S = S + np.outer(dm, current_z - m)
            # Calculate the unbiased empirical covariance
            if adapt_t > 1:
                C = S / (adapt_t - 1)

        if (i % 100) == 0:
            end = time.time()
            length = end - start
            time_remaining = ((length/60) / (i + 1)) * (iterations - (i + 1))
            print(f"ITER {i} | Acc: {accepted/(i+1):.2%} | LL: {current_log_lik:.2f} | Minimum ESS: {current_ess:.1f} | Estimated time remaining: {time_remaining:.2f}m")
            np.set_printoptions(suppress=True, precision=4)
            print(f"Current params: {current_x}") 
        
    return np.array(trace), np.array(trace_ess)

# Data

log_returns_array = simulate_svj_log_returns(TRUE_PARAMS, NUM_DAYS)

# Main running

print(f"Trained on {len(log_returns_array) / 365:.1f} year(s) of simulated returns, ignoring the first {DAYSOFRETURNSIGNORED} days for likelihood calculation. {ITERATIONS} iterations, {PARTICLES} particles, {BURN_IN} burn in.")
mcmc_trace, mcmc_trace_ess = run_adaptive_pmmh(iterations=ITERATIONS, initial_params=INITIAL_GUESSES, returns=log_returns_array, particles=PARTICLES)

param_names = ["mu", "kappa", "theta", "sigma", "rho", "lamb", "muJ", "delta"]

fig, axes = plt.subplots(len(param_names), 2, figsize=(12, 20))

for i, name in enumerate(param_names):
   
    axes[i, 0].plot(mcmc_trace[:, i])
    axes[i, 0].set_title(f"{name} Trace")
    axes[i, 0].axvline(BURN_IN, color='r', linestyle='--')
    
    axes[i, 1].hist(mcmc_trace[BURN_IN:, i], bins=30, density=True, alpha=0.7)
    axes[i, 1].set_title(f"{name} Density")

plt.tight_layout()
plt.show()

print("Posterior Means:")
means = np.mean(mcmc_trace[BURN_IN:], axis=0)
medians = np.median(mcmc_trace[BURN_IN:], axis=0)
for name, val in zip(param_names, means):
    print(f"{name}: {val:.4f}")

print("Posterior Medians:")
for name, val in zip(param_names, medians):
    print(f"{name}: {val:.4f}")

print("\n--- Posterior Estimates (Mean & 95% Credible Intervals) ---")
samples = mcmc_trace[BURN_IN:]
for i, name in enumerate(param_names):
    param_samples = samples[:, i]
    mean_val = np.mean(param_samples)
    median_val = np.median(param_samples)
    lower_bound = np.percentile(param_samples, 2.5)
    upper_bound = np.percentile(param_samples, 97.5)
    print(
        f"{name:>6}: Mean = {mean_val:6.4f} | Median = {median_val:6.4f} | "
        f"95% CrI = [{lower_bound:6.4f}, {upper_bound:6.4f}]"
    )

print("\nTrue vs Estimated (posterior mean):")
for i, name in enumerate(param_names):
    true_val = TRUE_PARAMS[i]
    est_val = means[i]
    abs_err = abs(est_val - true_val)
    rel_err_pct = np.nan
    if abs(true_val) > 1e-12:
        rel_err_pct = 100.0 * abs_err / abs(true_val)
    print(
        f"{name:>6}: true={true_val: .4f} | est={est_val: .4f} | "
        f"abs_err={abs_err: .4f} | rel_err={rel_err_pct: .2f}%"
    )


#  For equal sample sizes, this is the mean absolute difference between sorted samples
def wasserstein_distance_1d(sample_a, sample_b):
    """Calculates the 1D Wasserstein distance (Earth Mover's Distance) between two samples by sorting them and taking the mean absolute difference."""
    a = np.sort(sample_a)
    b = np.sort(sample_b)
    n = min(len(a), len(b))
    if n == 0:
        return np.nan
    return np.mean(np.abs(a[:n] - b[:n]))

num_ll_runs = 100

true_ll_runs = np.zeros(num_ll_runs)
true_ess_runs = np.zeros(num_ll_runs)
est_ll_runs = np.zeros(num_ll_runs)
est_ess_runs = np.zeros(num_ll_runs)
false_ll_runs = np.zeros(num_ll_runs)
false_ess_runs = np.zeros(num_ll_runs)

for j in range(num_ll_runs):
    true_ll_runs[j], true_ess_runs[j] = run_particle_filter(TRUE_PARAMS, log_returns_array, PARTICLES)
    est_ll_runs[j], est_ess_runs[j] = run_particle_filter(means.astype(np.float64), log_returns_array, PARTICLES)
    false_ll_runs[j], false_ess_runs[j] = run_particle_filter(FALSE_PARAMS, log_returns_array, PARTICLES)

true_ll_avg = np.mean(true_ll_runs)
est_ll_avg = np.mean(est_ll_runs)
false_ll_avg = np.mean(false_ll_runs)
true_ess_avg = np.mean(true_ess_runs)
est_ess_avg = np.mean(est_ess_runs)
false_ess_avg = np.mean(false_ess_runs)

print(f"\nTotal log-likelihood on data ({num_ll_runs} runs each):")
print(f"True params LL mean      : {true_ll_avg:.4f}")
print(f"Estimated params LL mean : {est_ll_avg:.4f}")
print(f"False params LL mean     : {false_ll_avg:.4f}")
print(f"LL mean difference (est - true): {est_ll_avg - true_ll_avg:.4f}")
print(f"LL mean difference (false - true): {false_ll_avg - true_ll_avg:.4f}")

num_wasserstein_runs = 100
emd_est_runs = np.zeros(num_wasserstein_runs)
emd_false_runs = np.zeros(num_wasserstein_runs)
emd_true_runs = np.zeros(num_wasserstein_runs)
estimated_path_for_plot = None

for j in range(num_wasserstein_runs):
    estimated_path = simulate_svj_log_returns(means.astype(np.float64), len(log_returns_array))
    false_path = simulate_svj_log_returns(FALSE_PARAMS, len(log_returns_array))
    true_path2 = simulate_svj_log_returns(TRUE_PARAMS, len(log_returns_array))

    if estimated_path_for_plot is None:
        estimated_path_for_plot = estimated_path

    emd_est_runs[j] = wasserstein_distance_1d(
        100.0 * np.expm1(log_returns_array),
        100.0 * np.expm1(estimated_path),
    )
    emd_false_runs[j] = wasserstein_distance_1d(
        100.0 * np.expm1(log_returns_array),
        100.0 * np.expm1(false_path),
    )
    emd_true_runs[j] = wasserstein_distance_1d(
        100.0 * np.expm1(log_returns_array),
        100.0 * np.expm1(true_path2),
    )

emd_pct_mean = np.mean(emd_est_runs)
emd_pct_std = np.std(emd_est_runs)
print(
    f"\nWasserstein distance (EMD) - Estimated params vs actual (return-% units): "
    f"mean={emd_pct_mean:.6f}, std={emd_pct_std:.6f}"
)

emd_false_vs_true_pct_mean = np.mean(emd_false_runs)
emd_false_vs_true_pct_std = np.std(emd_false_runs)
print(
    f"Wasserstein distance - False params vs actual (return-% units): "
    f"mean={emd_false_vs_true_pct_mean:.6f}, std={emd_false_vs_true_pct_std:.6f}"
)

emd_true_vs_true_pct_mean = np.mean(emd_true_runs)
emd_true_vs_true_pct_std = np.std(emd_true_runs)
print(
    f"Wasserstein distance - True params vs actual (return-% units): "
    f"mean={emd_true_vs_true_pct_mean:.6f}, std={emd_true_vs_true_pct_std:.6f}"
)