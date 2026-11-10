# GA to fit model to observed IV surface

import pandas as pd
from numba import njit, prange
import math
import matplotlib.pyplot as plt
import numpy as np
import time

# CONFIG

# Models
forceheston = False
forceblack76 = False
# Information for Gauss-Legendre Quadrature
DEG = 64
X_NODES, WEIGHTS = np.polynomial.legendre.leggauss(DEG)
UPPER_BOUND = 200
GL_NODES = 0.5 * UPPER_BOUND * X_NODES + 0.5 * UPPER_BOUND
GL_WEIGHTS = 0.5 * UPPER_BOUND * WEIGHTS
# Running information
r = 0.04 
file_path = 'DATA/trainingdata.csv'
POPULATION_SIZE = 100
PARENT_SIZE = 25
ELITE_NUM = 3
OFFSPRING_SIZE = 75-ELITE_NUM
MAX_GENERATIONS = 1000
ISLAND_COUNT = 12
TOURNAMENT_SIZE = 2
EARLY_STOP_COUNT = 20
MUTATION_RATE = 0.5

# BOUNDS (kappa, v0, theta, sigma, rho, lamb, muJ, delta)

BOUNDS = np.array([
        [0.1, 25.0],
		[0.1, 1.0],
		[0.1, 1.0],
		[0.1, 3.0],
		[-0.5, 0.5],
		[0.1, 20.0],
		[-0.2, 0.2],
		[0.01, 0.5],
    ], dtype=float)

# DATA

def get_data_by_date(file_path, month, day, section):
    """Gets data from csv for desired day and month"""
    if month == 13:
        month = 1
    if month == 14:
        month = 2
    try:
        df = pd.read_csv(file_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        filtered_df = df[
            (df['timestamp'].dt.month == month) & 
            (df['timestamp'].dt.day == day) &
            (df['type'] == 'call') &
            (df['IV'] > 0.4) &
            (df['section'] == section) &
            (df['vega'] > 0.01)
        ]
        return filtered_df
    except Exception as e:
        print(f"Error: {e}")
        return None

T_E = []
for i in range(365):
    T_E.append((i+1)/365)
K_E = []
for i in range(52):
    K_E.append(0.4 + 0.05*i)

def correctdistribution(df, t_col='T', k_col='K'):
    """Splits data into evenly spaced points across surface"""
    T_edges = T_E
    K_edges = K_E
    df['T_bin'] = pd.cut(df[t_col], bins=T_edges)
    df['K_bin'] = pd.cut(df[k_col], bins=K_edges)
    df_filtered = df.dropna(subset=['T_bin', 'K_bin']).copy()
    result = df_filtered.groupby(['T_bin', 'K_bin'], observed=True).head(1)
    return result.drop(columns=['T_bin', 'K_bin'])

def first_valid_timestamp(day_sections):
    """Used for analysis/debugging"""
    for section_df in day_sections:
        if section_df is not None and not section_df.empty and 'timestamp' in section_df.columns:
            timestamps = pd.to_datetime(section_df['timestamp'], errors='coerce').dropna()
            if not timestamps.empty:
                return timestamps.iloc[0].normalize()
    return pd.NaT

# MAIN FUNCTIONS

@njit
def bates_char_func_numba(phi, T, r, v0, kappa, theta, sigma, rho, lamb, muJ, delta, P_num):
    """Standard bates characteristic function"""
    i = 1j
    phi_c = i * phi

    if P_num == 1:
        u = 0.5
        b = kappa - rho * sigma
    else:
        u = -0.5
        b = kappa

    alpha = kappa * theta
    sig2 = max(sigma * sigma, 1e-12)

    khat = np.exp(muJ + 0.5 * delta * delta) - 1.0

    beta = b - rho * sigma * phi_c
    d_sq = beta * beta - 2.0 * sig2 * (u * phi_c + 0.5 * phi_c * phi_c)
    d = np.sqrt(d_sq)
    if d.real < 0.0:
        d = -d

    g = (beta - d) / (beta + d + 1e-14)
    exp_neg_dT = np.exp(-d * T)

    D = ((beta - d) / sig2) * ((1.0 - exp_neg_dT) / (1.0 - g * exp_neg_dT + 1e-14))

    log_term = np.log(1.0 - g * exp_neg_dT + 1e-14) - np.log(1.0 - g + 1e-14)

    C = (r - lamb * khat) * phi_c * T + (alpha / sig2) * ((beta - d) * T - 2.0 * log_term)

    jump_exp_component = phi_c * np.log(1.0 + khat) + (delta * delta) * (u * phi_c + 0.5 * phi_c * phi_c)
    
    jump_term = lamb * T * ((1.0 + khat) ** (u + 0.5)) * (np.exp(jump_exp_component) - 1.0)

    return np.exp(C + D * v0 + jump_term)

@njit(fastmath=True)
def compute_price_numba(S, F, K, T, r_riskfree, params, nodes, weights):
    """Compute call price using Gauss-Legendre quadrature and the characteristic function"""
    kappa, v0, theta, sigma, rho, lamb, muJ, delta = params
    
    # b76 shortcut
    if sigma <= 1e-30 and lamb <= 1e-30:
        return b76_price(F, K, T, r_riskfree, np.sqrt(v0))

    # drift implied by forward price (to ensure discounted martingale)
    r_drift = np.log(F / S) / T 
    
    int_P1 = 0.0
    int_P2 = 0.0
    
    # integral calculations using G-L Quad
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

@njit(fastmath=True)
def norm_cdf(x):
    """standard norm cdf"""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

@njit(fastmath=True)
def b76_price(F, K, T, r, sigma):
    """standard b76 call price"""
    if sigma <= 1e-5: return max(np.exp(-r*T) * (F - K), 0.0)
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return np.exp(-r * T) * (F * norm_cdf(d1) - K * norm_cdf(d2))

@njit(fastmath=True)
def b76_put_price(F, K, T, r, sigma):
    """standard b76 put price"""
    if sigma <= 1e-5: return max(np.exp(-r*T) * (K - F), 0.0)
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return np.exp(-r * T) * (K * norm_cdf(-d2) - F * norm_cdf(-d1))

@njit(fastmath=True)
def implied_vol_numba(target_price, F, K, T, r):
    """finds IV using bisection method"""
    is_call = True

    if F > K:
        is_call = False
        intrinsic = np.exp(-r * T) * (F - K)
        target_price = target_price - intrinsic
        if target_price <= 0:
            return 1e-5

    def objective(sigma):
        return b76_price(F, K, T, r, sigma) - target_price if is_call else b76_put_price(F, K, T, r, sigma) - target_price

    low_vol = 1e-5
    high_vol = 10.0
    for _ in range(60):
        mid_vol = 0.5 * (low_vol + high_vol)
        val = objective(mid_vol)
        
        if val > 0:
            high_vol = mid_vol
        else:
            low_vol = mid_vol
            
    return mid_vol

@njit(parallel=True)
def evaluate_population_fast(population, S_arr, F_arr, K_arr, T_arr, market_prices, bids, asks, market_ivs, market_vegas, r_riskfree, nodes, weights):
    """evaluate fitness of entire population according to relative IV error, additionally returns various other metrics"""
    n_pop = population.shape[0]
    n_data = len(S_arr)
    fitness_scores = np.zeros(n_pop)
    mae_scores = np.zeros(n_pop)
    abs_scores = np.zeros(n_pop)
    rel_scores = np.zeros(n_pop)

    withinspread_scores = np.zeros(n_pop)
    
    # Loop over population 
    for i in prange(n_pop):
        params = population[i]
        sse = 0.0
        sae = 0.0
        total_abs = 0.0
        total_rel = 0.0

        total_withinspread = 0.0
        valid_spread_count = 0.0

        # Loop over data points
        for j in range(n_data):
            # get model price
            model_price = compute_price_numba(
                S_arr[j], F_arr[j], K_arr[j], T_arr[j], 
                r_riskfree, params, nodes, weights
            )
            
            # get model IV from price
            model_iv = implied_vol_numba(model_price, F_arr[j], K_arr[j], T_arr[j], r_riskfree)
            
            # find error between IV's and sum squared errors (relative IV error)
            diff = model_iv - market_ivs[j]
            wdiff = abs(diff) / (market_ivs[j] + 1e-8)  # avoid division by zero
            sse += (wdiff * wdiff)
            
            # find abs, rel, mae
            total_abs += abs(model_price - market_prices[j])
            if total_abs < 10: # ignore rel issues for price errors < 10
                total_rel += 0
            else: # ignore tiny prices
                if market_prices[j] < 10:
                    total_rel += 0
                else:
                    total_rel += abs(model_price - market_prices[j]) / (market_prices[j] + 1e-8)
            sae += abs(diff)

            # check if within bid-ask spread (skip NaN bids/asks)
            bid_valid = not np.isnan(bids[j])
            ask_valid = not np.isnan(asks[j])
            if bid_valid and ask_valid:
                valid_spread_count += 1.0
                if model_price >= bids[j] and model_price <= asks[j]:
                    total_withinspread += 1
            
        # record fitness and other metrics for this parameter set
        mse = sse / n_data
        mae = sae / n_data
        fitness_scores[i] = 1.0 / (mse + 1e-8)
        mae_scores[i] = mae
        abs_scores[i] = total_abs / n_data
        rel_scores[i] = total_rel / n_data
        if valid_spread_count > 0:
            withinspread_scores[i] = total_withinspread / valid_spread_count
        else:
            withinspread_scores[i] = 0.0

    return fitness_scores, mae_scores, abs_scores, rel_scores, withinspread_scores

def initialpopulation():
    """Creates random initial population"""
    pop_size = POPULATION_SIZE
    rng = np.random.default_rng()
    pop = rng.random((pop_size, BOUNDS.shape[0]))
    return BOUNDS[:, 0] + (pop * (BOUNDS[:, 1] - BOUNDS[:, 0]))

def select_parents_tournament(population, fitness, num_parents=PARENT_SIZE, tournament_size=TOURNAMENT_SIZE):
    """Tournament selection to select parents and maintain diversity"""
    selected_indices = []
    pop_size = len(population)
    
    for _ in range(num_parents):
        contestants = np.random.randint(0, pop_size, tournament_size)
        winner_idx = contestants[np.argmax(fitness[contestants])]
        selected_indices.append(winner_idx)
        
    return population[np.array(selected_indices)]

def select_offspring(parents, num_offspring=OFFSPRING_SIZE):
    """Randomly take parameters from either parent"""
    # Randomly select indices for the first and second parents
    idx1 = np.random.choice(len(parents), size=num_offspring, replace=True)
    idx2 = np.random.choice(len(parents), size=num_offspring, replace=True)
    
    p1 = parents[idx1]
    p2 = parents[idx2]
    
    mask = np.random.rand(num_offspring, p1.shape[1]) > 0.5
    
    # np.where uses the mask: if True, take from p1; if False, take from p2
    offspring = np.where(mask, p1, p2)
    
    return offspring

def mutate(offspring, mutation_rate):
    """Apply scaled mutations at a probability of mutation_rate per parameter"""
    scales = np.array([0.5, 0.03, 0.03, 0.05, 0.05, 0.1, 0.05, 0.05]) 
    for i in range(len(offspring)):
        if np.random.rand() < mutation_rate:
            p_idx = np.random.randint(len(offspring[i]))
            offspring[i][p_idx] += np.random.uniform(-1, 1) * scales[p_idx]
    return offspring

def check_bounds(population):
    """Enforces bounds per model"""
    for i in range(len(population)):
        population[i] = np.clip(population[i], BOUNDS[:, 0], BOUNDS[:, 1])
    if forceheston:
        population[:, 5] = 0.0
        population[:, 6] = 0.0
        population[:, 7] = 0.0
    if forceblack76:
        population[:, 0] = 10.0
        population[:, 2] = population[:, 1]
        population[:, 3] = 0.0
        population[:, 4] = 0.0
        population[:, 5] = 0.0
        population[:, 6] = 0.0
        population[:, 7] = 0.0
    return population

# MAIN EXECUTION

# Setup Data
df_list = []
day_dates = []

# Collect data into nice dataframes
for i in range(6, 80): 
    df = []
    for j in range(19,22):
        dfj = get_data_by_date(file_path, 12+np.floor(i/31), 1+(i%31), j)
        dfj = correctdistribution(dfj, 'time_to_expiry', 'moneyness')
        df.append(dfj)
    df_list.append(df)
    day_dates.append(first_valid_timestamp(df))

plot_dates = day_dates[:len(df_list)-2]

# Gives data information for debugging/informative purposes
for i in range(len(df_list)):
    if df_list[i] is not None:
        for j in range(3):
            if df_list[i][j] is not None:
                print(f"Day {i+1} | Section {j+1} | Data points: {len(df_list[i][j])} | Avg IV: {np.mean(df_list[i][j]['IV']):.4f}")
            else:
                print(f"Day {i+1} | Section {j+1} | No data available.")
    else:
        print(f"Day {i+1} | No data available.")

totalrmse = []
totalmae = []
totalabserror = []
totalrelerror = []
bestparamslist = []
percentwithinspread = []

totalrmseH = []
totalmaeH = []
totalabserrorH = []
totalrelerrorH = []
bestparamslistH = []
percentwithinspreadH = []

totalrmseBS = []
totalmaeBS = []
totalabserrorBS = []
totalrelerrorBS = []
bestparamslistBS = []
percentwithinspreadBS = []

initialtime = time.time()
totalgenerations = 0
dayselapsed = 0

lowesttotalMAE = 100.0

findingbestdayrmse = 100.0
findingbestdayindex = -1

for model in range(3):
    if model==0:    
        print("Running GA with full Bates model...")
        forceheston = False
        forceblack76 = False
    elif model==1:
        print("Running GA with Heston model (no jumps)...")
        forceheston = True
        forceblack76 = False
    else:
        print("Running GA with Black-76 model...")
        forceheston = True
        forceblack76 = True

    print("Starting GA algorithm across all days...")
    for i in range(len(df_list)-2):

        if (df_list[i] is not None and len(df_list[i]) == 3 and df_list[i][0] is not None and df_list[i][1] is not None and df_list[i][2] is not None):
            df = df_list[i][0]
            S_vals = df['estimated_delivery_price'].values.astype(float)
            F_vals = df['underlying_price'].values.astype(float)
            K_vals = df['strike'].values.astype(float)
            T_vals = df['time_to_expiry'].values.astype(float)
            Market_Prices = df['mark_price'].values.astype(float) * F_vals
            Market_IVs = df['IV'].values.astype(float)
            Market_Vegas = df['vega'].values.astype(float)
            bids = df['bid'].values.astype(float) * F_vals
            asks = df['ask'].values.astype(float) * F_vals

            df2 = df_list[i][1]
            S_vals2 = df2['estimated_delivery_price'].values.astype(float)
            F_vals2 = df2['underlying_price'].values.astype(float)
            K_vals2 = df2['strike'].values.astype(float)
            T_vals2 = df2['time_to_expiry'].values.astype(float)
            Market_Prices2 = df2['mark_price'].values.astype(float) * F_vals2
            Market_IVs2 = df2['IV'].values.astype(float)
            Market_Vegas2 = df2['vega'].values.astype(float)
            bids2 = df2['bid'].values.astype(float) * F_vals2
            asks2 = df2['ask'].values.astype(float) * F_vals2

            df3 = df_list[i][2]
            S_vals3 = df3['estimated_delivery_price'].values.astype(float)
            F_vals3 = df3['underlying_price'].values.astype(float)
            K_vals3 = df3['strike'].values.astype(float)
            T_vals3 = df3['time_to_expiry'].values.astype(float)
            Market_Prices3 = df3['mark_price'].values.astype(float) * F_vals3
            Market_IVs3 = df3['IV'].values.astype(float)
            Market_Vegas3 = df3['vega'].values.astype(float)
            bids3 = df3['bid'].values.astype(float) * F_vals3
            asks3 = df3['ask'].values.astype(float) * F_vals3

            overalldaybestrmse = 1e300
            overalldaybestparams = []
            overallbestoosiv = 1e300
            overallbestofmae = 1e300
            overallbestofabs = 1e300
            overallbestofrel = 1e300
            overallbestofspread = 1e300

            for island in range(ISLAND_COUNT):
                lowestiverrortotal = 1e300
                bestoosiv = 1e300
                bestofmae = 1e300
                bestofabs = 1e300
                bestofrel = 1e300
                bestofspread = 1e300
                bestmaeparams = []
                lengthwithoutimprovement = 0
                population = check_bounds(initialpopulation())
                print(f"Island {island} | Starting GA for day {i+1}. Min IV: {np.min(Market_IVs):.4f} Max IV: {np.max(Market_IVs):.4f} | Average IV: {np.mean(Market_IVs):.4f} | Data points: {len(Market_IVs)}")
                #print(f"Island {island} | Min IV 2: {np.min(Market_IVs2):.4f} Max IV 2: {np.max(Market_IVs2):.4f} | Average IV 2: {np.mean(Market_IVs2):.4f} | Data points 2: {len(Market_IVs2)}")
                #print(f"Island {island} | Min IV 3: {np.min(Market_IVs3):.4f} Max IV 3: {np.max(Market_IVs3):.4f} | Average IV 3: {np.mean(Market_IVs3):.4f} | Data points 3: {len(Market_IVs3)}")
                for gen in range(MAX_GENERATIONS):

                    # Fitness evaluation
                    fitness, mae, abs, rel, withinspread = evaluate_population_fast(
                        population, S_vals, F_vals, K_vals, T_vals, 
                        Market_Prices, bids, asks, Market_IVs, Market_Vegas, r, GL_NODES, GL_WEIGHTS
                    )

                    num_elites = ELITE_NUM
                    elite_indices = np.argsort(fitness)[-num_elites:]
                    elites = population[elite_indices].copy()

                    fit2_arr, mae2_arr, abs2_arr, rel2_arr, withinspread2 = evaluate_population_fast(
                        elites, S_vals2, F_vals2, K_vals2, T_vals2, Market_Prices2, bids2, asks2, Market_IVs2, Market_Vegas2, r, GL_NODES, GL_WEIGHTS
                    )
                    fit2_best = np.argmax(fit2_arr)
                    mae2_best = mae2_arr[fit2_best]
                    abs2_best = abs2_arr[fit2_best]
                    rel2_best = rel2_arr[fit2_best]
                    best_params = elites[fit2_best : fit2_best+1]

                    fit3_arr, mae3_arr, abs3_arr, rel3_arr, withinspread3 = evaluate_population_fast(
                        best_params, S_vals3, F_vals3, K_vals3, T_vals3, Market_Prices3, bids3, asks3, Market_IVs3, Market_Vegas3, r, GL_NODES, GL_WEIGHTS
                    )
                    mae3_best = mae3_arr[0]
                    abs3_best = abs3_arr[0]
                    rel3_best = rel3_arr[0]

                    best_idx = np.argmax(fitness)
                    best_err1 = np.sqrt(1 / (fitness[best_idx] + 1e-8))
                    best_err2 = np.sqrt(1 / (fit2_arr[fit2_best] + 1e-8))
                    best_err3 = np.sqrt(1 / (fit3_arr[0] + 1e-8))

                    if (gen+1) % 50 == 0:
                        print(f"Island {island} | Gen {gen+1} | Best Fit: {np.max(fitness):.4f} | Best Pms1 RMSE: {best_err1:.4f} | Best Pms2 RMSE 2: {best_err2:.4f} | Best Pms MAE 3: {mae3_best:.4f}")

                    if best_err2 < lowestiverrortotal*0.999:
                        lowestiverrortotal = best_err2
                        bestoosiv = best_err3
                        bestofmae = mae3_best
                        bestofabs = abs3_best
                        bestofrel = rel3_best
                        bestofspread = withinspread3[0]
                        bestmaeparams = elites[fit2_best : fit2_best+1]
                        lengthwithoutimprovement = 0
                        #print(f"New lowest RMSE 2: {lowestiverrortotal:.4f} | oos MAE: {mae3_best:.4f} | Params: {bestmaeparams}")
                    else:
                        lengthwithoutimprovement += 1
                        if lengthwithoutimprovement > EARLY_STOP_COUNT:
                            print(f"Stopping GA for day {i+1}, island {island} due to no improvement in {EARLY_STOP_COUNT} generations.")
                            break

                    parents = select_parents_tournament(population, fitness)
                    offspring = select_offspring(parents)
                    fitness_std = np.std(fitness)
                    mut_offspring = mutate(offspring, mutation_rate=MUTATION_RATE)
                    mut_offspring = check_bounds(mut_offspring)
                    population = np.vstack((parents, mut_offspring, elites))

                # finished an island
                if lowestiverrortotal < overalldaybestrmse:
                    overalldaybestrmse = lowestiverrortotal
                    overalldaybestparams = bestmaeparams
                    overallbestoosiv = bestoosiv
                    overallbestofmae = bestofmae
                    overallbestofabs = bestofabs
                    overallbestofrel = bestofrel
                    overallbestofspread = bestofspread

        # finished all islands for a day
        if model==0:
            totalrmse.append(overallbestoosiv)
            totalmae.append(overallbestofmae)
            totalabserror.append(overallbestofabs)
            totalrelerror.append(overallbestofrel)
            bestparamslist.append(overalldaybestparams[0])
            percentwithinspread.append(overallbestofspread)
            if (overallbestoosiv < findingbestdayrmse) & (np.max(Market_IVs3) > 1.4):
                findingbestdayrmse = overallbestoosiv
                findingbestdayindex = i
        elif model==1:
            totalrmseH.append(overallbestoosiv)
            totalmaeH.append(overallbestofmae)
            totalabserrorH.append(overallbestofabs)
            totalrelerrorH.append(overallbestofrel)
            bestparamslistH.append(overalldaybestparams[0])
            percentwithinspreadH.append(overallbestofspread)
        else:
            totalrmseBS.append(overallbestoosiv)
            totalmaeBS.append(overallbestofmae)
            totalabserrorBS.append(overallbestofabs)
            totalrelerrorBS.append(overallbestofrel)
            bestparamslistBS.append(overalldaybestparams[0])
            percentwithinspreadBS.append(overallbestofspread)

        print(f"Finished GA for day {i+1}, island {island}")
        #print(f"Best GA parameters for this day (judged off MAE3):{overalldaybestparams}")
        #print(f"Best param set MAE for this day: {overallbestofmae:.4f}")
        print(f"Current mean MAE {np.mean(totalmae)}")
        print(f"Current average within spread {np.mean(percentwithinspread)}")
        if model > 0:
            print(f"Current mean MAEH {np.mean(totalmaeH)}")
        if model > 1:
            print(f"Current mean MAEBS {np.mean(totalmaeBS)}")
        timenow = time.time()
        currenttime = timenow - initialtime
        totalgenerations = totalgenerations + gen
        dayselapsed = dayselapsed + 1
        print(f"Estimated time remaining: { ((currenttime/totalgenerations) * ( (len(df_list)-2-(i+1)+((2-model)*(len(df_list)-2))) * (totalgenerations/(dayselapsed)) ))/60:.4f} minutes")
    print(f"Average oos RMSE across all days for best param sets: {np.mean(totalrmse):.4f}")
    print(f"Average oos MAE across all days for best param sets: {np.mean(totalmae):.4f}")
    print(f"Median oos MAE across all days for best param sets: {np.median(totalmae):.4f}")
    print(f"Average oos AbsError across all days for best param sets: {np.mean(totalabserror):.4f}")
    print(f"Median oos AbsError across all days for best param sets: {np.median(totalabserror):.4f}")
    print(f"Average oos RelError across all days for best param sets: {np.mean(totalrelerror):.4f}")
    print(f"Median oos RelError across all days for best param sets: {np.median(totalrelerror):.4f}")
    print(f"Average percentage of predictions within bid-ask spread: {np.mean(percentwithinspread):.4f}")
    print("======================================================================================")
    print(f"Average oos RMSE across all days for best param sets (H): {np.mean(totalrmseH):.4f}")
    print(f"Average oos MAE across all days for best param sets (H): {np.mean(totalmaeH):.4f}")
    print(f"Median oos MAE across all days for best param sets (H): {np.median(totalmaeH):.4f}")
    print(f"Average oos AbsError across all days for best param sets (H): {np.mean(totalabserrorH):.4f}")
    print(f"Median oos AbsError across all days for best param sets (H): {np.median(totalabserrorH):.4f}")
    print(f"Average oos RelError across all days for best param sets (H): {np.mean(totalrelerrorH):.4f}")
    print(f"Median oos RelError across all days for best param sets (H): {np.median(totalrelerrorH):.4f}")
    print(f"Average percentage of predictions within bid-ask spread (H): {np.mean(percentwithinspreadH):.4f}")
    print("======================================================================================")
    print(f"Average oos RMSE across all days for best param sets (BS): {np.mean(totalrmseBS):.4f}")
    print(f"Average oos MAE across all days for best param sets (BS): {np.mean(totalmaeBS):.4f}")
    print(f"Median oos MAE across all days for best param sets (BS): {np.median(totalmaeBS):.4f}")
    print(f"Average oos AbsError across all days for best param sets (BS): {np.mean(totalabserrorBS):.4f}")
    print(f"Median oos AbsError across all days for best param sets (BS): {np.median(totalabserrorBS):.4f}")
    print(f"Average oos RelError across all days for best param sets (BS): {np.mean(totalrelerrorBS):.4f}")
    print(f"Median oos RelError across all days for best param sets (BS): {np.median(totalrelerrorBS):.4f}")
    print(f"Average percentage of predictions within bid-ask spread (BS): {np.mean(percentwithinspreadBS):.4f}")

print("Total time taken: {:.2f} minutes".format((time.time() - initialtime) / 60))

# PLOTTING (Plotting was heavily AI assisted)

def apply_date_axis(ax, dates, xlabel, max_ticks=12):
    if len(dates) == 0:
        ax.set_xlabel(xlabel)
        return

    tick_count = min(max_ticks, len(dates))
    tick_positions = np.linspace(0, len(dates) - 1, tick_count, dtype=int)
    tick_positions = np.unique(tick_positions)
    tick_labels = []
    for idx in tick_positions:
        tick_date = dates[idx]
        if pd.isna(tick_date):
            tick_labels.append(f'Day {idx + 1}')
        else:
            tick_labels.append(tick_date.strftime('%Y-%m-%d'))

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha='right')
    ax.set_xlabel(xlabel)

# MA window
window = 10

# RMSE MAs
rmse_ma = pd.Series(totalrmse).rolling(window=window, min_periods=1).mean()
rmseH_ma = pd.Series(totalrmseH).rolling(window=window, min_periods=1).mean()
rmseBS_ma = pd.Series(totalrmseBS).rolling(window=window, min_periods=1).mean()

# MAE MAs
mae_ma = pd.Series(totalmae).rolling(window=window, min_periods=1).mean()
maeH_ma = pd.Series(totalmaeH).rolling(window=window, min_periods=1).mean()
maeBS_ma = pd.Series(totalmaeBS).rolling(window=window, min_periods=1).mean()

# Absolute Error MAs
abs_ma = pd.Series(totalabserror).rolling(window=window, min_periods=1).mean()
absH_ma = pd.Series(totalabserrorH).rolling(window=window, min_periods=1).mean()
absBS_ma = pd.Series(totalabserrorBS).rolling(window=window, min_periods=1).mean()

# Relative Error MAs
rel_ma = pd.Series(totalrelerror).rolling(window=window, min_periods=1).mean()
relH_ma = pd.Series(totalrelerrorH).rolling(window=window, min_periods=1).mean()
relBS_ma = pd.Series(totalrelerrorBS).rolling(window=window, min_periods=1).mean()

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(14, 20), sharex=True)

ms = 3      
alpha = 0.3 

# rmse

ax1.plot(totalrmse, color='tab:blue', marker='o', markersize=ms, alpha=alpha, label='_nolegend_')
ax1.plot(rmse_ma, color='tab:blue', linewidth=2, label='Bates Model')

ax1.plot(totalrmseH, color='tab:orange', marker='o', markersize=ms, alpha=alpha, label='_nolegend_')
ax1.plot(rmseH_ma, color='tab:orange', linewidth=2, label='Heston')

ax1.plot(totalrmseBS, color='tab:green', marker='o', markersize=ms, alpha=alpha, label='_nolegend_')
ax1.plot(rmseBS_ma, color='tab:green', linewidth=2, label='Black-76')

ax1.set_yscale('log')
ax1.set_title('Root Mean Square Error (RMSE) Over Time')
ax1.set_ylabel('RMSE (Log Scale)')
ax1.grid(True, which="both", ls="-", alpha=0.5)
ax1.legend(loc='upper left')

# mae 

ax2.plot(totalmae, color='tab:blue', marker='s', markersize=ms, alpha=alpha, label='_nolegend_')
ax2.plot(mae_ma, color='tab:blue', linewidth=2, label='Bates Model')

ax2.plot(totalmaeH, color='tab:orange', marker='s', markersize=ms, alpha=alpha, label='_nolegend_')
ax2.plot(maeH_ma, color='tab:orange', linewidth=2, label='Heston')

ax2.plot(totalmaeBS, color='tab:green', marker='s', markersize=ms, alpha=alpha, label='_nolegend_')
ax2.plot(maeBS_ma, color='tab:green', linewidth=2, label='Black-76')

ax2.set_yscale('log')
ax2.set_title('Mean Absolute Error (MAE) Over Time')
ax2.set_ylabel('MAE (Log Scale)')
ax2.grid(True, which="both", ls="-", alpha=0.5)
ax2.legend(loc='upper left')

# abs error

ax3.plot(totalabserror, color='tab:blue', marker='^', markersize=ms, alpha=alpha, label='_nolegend_')
ax3.plot(abs_ma, color='tab:blue', linewidth=2, label='Bates Model')

ax3.plot(totalabserrorH, color='tab:orange', marker='^', markersize=ms, alpha=alpha, label='_nolegend_')
ax3.plot(absH_ma, color='tab:orange', linewidth=2, label='Heston')

ax3.plot(totalabserrorBS, color='tab:green', marker='^', markersize=ms, alpha=alpha, label='_nolegend_')
ax3.plot(absBS_ma, color='tab:green', linewidth=2, label='Black-76')

ax3.set_yscale('log')
ax3.set_title('Absolute Error Over Time')
ax3.set_ylabel('Abs Error (Log Scale)')
ax3.grid(True, which="both", ls="-", alpha=0.5)
ax3.legend(loc='upper left')

# rel error

ax4.plot(totalrelerror, color='tab:blue', marker='D', markersize=ms, alpha=alpha, label='_nolegend_')
ax4.plot(rel_ma, color='tab:blue', linewidth=2, label='Bates Model')

ax4.plot(totalrelerrorH, color='tab:orange', marker='D', markersize=ms, alpha=alpha, label='_nolegend_')
ax4.plot(relH_ma, color='tab:orange', linewidth=2, label='Heston')

ax4.plot(totalrelerrorBS, color='tab:green', marker='D', markersize=ms, alpha=alpha, label='_nolegend_')
ax4.plot(relBS_ma, color='tab:green', linewidth=2, label='Black-76')

ax4.set_yscale('log')
ax4.set_title('Relative Error Over Time')
ax4.set_ylabel('Rel Error (Log Scale)')
ax4.grid(True, which="both", ls="-", alpha=0.5)
ax4.legend(loc='upper left')

apply_date_axis(ax4, plot_dates, 'Date')

plt.tight_layout() 
plt.show()

# Plot IV surfaces for random day with large max IV
i = 10
trained_params = bestparamslist[i]
trained_paramsH = bestparamslistH[i]
trained_paramsBS = bestparamslistBS[i]
df = get_data_by_date(file_path, 12+np.floor(i/31), 1+(i%31), 21)
T_E = []
for i in range(365):
    T_E.append((i+1)/365)
K_E = []
for i in range(52):
    K_E.append(0.4 + 0.05*i)
df = correctdistribution(df, 'time_to_expiry', 'moneyness')
bins = [5/365, 10/365, 20/365, 30/365, 40/365, 50/365, 60/365, 90/365, 120/365, 150/365, 180/365, 210/365, 240/365, 270/365, 300/365, 330/365, 360/365]
labels = ['5-10 days', '10-20 days', '20-30 days', '30-40 days', '40-50 days', '50-60 days', '60-90 days', '90-120 days', '120-150 days', '150-180 days', '180-210 days', '210-240 days', '240-270 days', '270-300 days', '300-330 days', '330-360 days']
df['bucket'] = pd.cut(df['time_to_expiry'], bins=bins, labels=labels, right=False)
def plot_market_vs_model_iv(data, params, r_riskfree, nodes, weights, title_suffix):
    for label in labels:
        subset = data[(data['bucket'] == label) & (data['type'] == 'call')].copy()
        
        if subset.empty:
            continue
        
        plt.figure(figsize=(12, 7))
        
        # Market data as scatter
        plt.scatter(subset['strike'], subset['IV'], alpha=0.6, s=20, color='tab:blue',
                    label='Market', zorder=3)
        
        # Model IVs at each market strike
        model_ivs = []
        for _, row in subset.iterrows():
            S_i = float(row['estimated_delivery_price'])
            F_i = float(row['underlying_price'])
            K_i = float(row['strike'])
            T_i = float(row['time_to_expiry'])
            model_price = compute_price_numba(S_i, F_i, K_i, T_i, r_riskfree, params, nodes, weights)
            model_iv = implied_vol_numba(model_price, F_i, K_i, T_i, r_riskfree)
            model_ivs.append(model_iv)
        
        plt.scatter(subset['strike'], model_ivs, alpha=0.6, s=20, color='tab:red',
                    label='Model', zorder=2)

        plt.xlabel('Strike Price')
        plt.ylabel('Implied Volatility (IV)')
        plt.ylim(0, 2.0)
        plt.title(f'IV Smile: {title_suffix} — {label}')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()

def plot_param_progression_against_baseline(best_params_array, baseline_params, baseline_name, plot_dates, ma_window=10):
    param_names = ['kappa', 'v0', 'theta', 'sigma', 'rho', 'lambda', 'muJ', 'delta']
    best_params_array = np.asarray(best_params_array, dtype=float)
    baseline_params = np.asarray(baseline_params, dtype=float)

    if best_params_array.ndim != 2 or best_params_array.shape[1] != 8:
        print("Skipping progression plot: bestparamslist must be shaped (n_days, 8).")
        return

    if len(plot_dates) != best_params_array.shape[0]:
        print("Skipping progression plot: date labels do not match parameter history length.")
        return

    days = np.arange(1, best_params_array.shape[0] + 1)
    fig, axes = plt.subplots(4, 2, figsize=(16, 18), sharex=True)
    axes = axes.flatten()

    for idx in range(8):
        ax = axes[idx]
        series = best_params_array[:, idx]
        baseline_val = baseline_params[idx]
        ma = pd.Series(series).rolling(window=ma_window, min_periods=1).mean().values

        ax.plot(days, series, color='tab:blue', linestyle='--', linewidth=1.0, alpha=0.35, label='Best Params by Day')
        ax.plot(days, ma, color='tab:blue', linewidth=1.0, label=f'{ma_window}-day MA')
        ax.axhline(y=baseline_val, color='tab:red', linestyle='--', linewidth=1.5,
                   label=f'{baseline_name} Baseline')

        ax.set_title(f"{param_names[idx]} progression")
        ax.set_ylabel('Value')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='best', fontsize=8)

    apply_date_axis(axes[-1], plot_dates, 'Date')
    apply_date_axis(axes[-2], plot_dates, 'Date')
    fig.suptitle(f"SVJ Parameter Progression vs {baseline_name} Baseline", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()

    lambda_series = best_params_array[:, 5]
    muj_series = best_params_array[:, 6]
    delta_series = best_params_array[:, 7]

    lambda_muj_series = lambda_series * muj_series
    lambda_muj_baseline = baseline_params[5] * baseline_params[6]
    lambda_muj_ma = pd.Series(lambda_muj_series).rolling(window=ma_window, min_periods=1).mean().values

    fig_lm, ax_lm = plt.subplots(figsize=(14, 6))
    ax_lm.plot(days, lambda_muj_series, color='tab:blue', linestyle='--', linewidth=1.0, alpha=0.35,
               label='Market-implied by day')
    ax_lm.plot(days, lambda_muj_ma, color='tab:blue', linewidth=1.8, label=f'{ma_window}-day MA')
    ax_lm.axhline(y=lambda_muj_baseline, color='tab:red', linestyle='--', linewidth=1.5,
                  label=f'{baseline_name} Historical Baseline')
    ax_lm.set_title(f"Market-implied lambda*muJ vs {baseline_name} Historical")
    ax_lm.set_ylabel('lambda*muJ')
    ax_lm.grid(True, linestyle='--', alpha=0.5)
    ax_lm.legend(loc='best')
    apply_date_axis(ax_lm, plot_dates, 'Date')
    plt.tight_layout()
    plt.show()

    lambda_delta2_series = lambda_series * (delta_series ** 2)
    lambda_delta2_baseline = baseline_params[5] * (baseline_params[7] ** 2)
    lambda_delta2_ma = pd.Series(lambda_delta2_series).rolling(window=ma_window, min_periods=1).mean().values

    fig_ld2, ax_ld2 = plt.subplots(figsize=(14, 6))
    ax_ld2.plot(days, lambda_delta2_series, color='tab:blue', linestyle='--', linewidth=1.0, alpha=0.35,
                label='Market-implied by day')
    ax_ld2.plot(days, lambda_delta2_ma, color='tab:blue', linewidth=1.8, label=f'{ma_window}-day MA')
    ax_ld2.axhline(y=lambda_delta2_baseline, color='tab:red', linestyle='--', linewidth=1.5,
                   label=f'{baseline_name} Historical Baseline')
    ax_ld2.set_title(f"Market-implied lambda*delta^2 vs {baseline_name} Historical")
    ax_ld2.set_ylabel('lambda*delta^2')
    ax_ld2.grid(True, linestyle='--', alpha=0.5)
    ax_ld2.legend(loc='best')
    apply_date_axis(ax_ld2, plot_dates, 'Date')
    plt.tight_layout()
    plt.show()

def plot_market_vs_historical_iv_diff(best_params_array, historical_params, historical_name, day_data, plot_dates, r_riskfree, nodes, weights, section_idx=2):
    # Use the OOS section by default (section_idx=2) and compute average IV gap per day.
    n_days = min(len(best_params_array), len(plot_dates), len(day_data))
    diff_pct = []

    for day_idx in range(n_days):
        day_sections = day_data[day_idx]
        if day_sections is None or len(day_sections) <= section_idx or day_sections[section_idx] is None or day_sections[section_idx].empty:
            diff_pct.append(np.nan)
            continue

        df_day = day_sections[section_idx]
        market_params = np.asarray(best_params_array[day_idx], dtype=float)
        hist_params = np.asarray(historical_params, dtype=float)
        hist_params[1] = market_params[1]  # Align v0 to market-implied for fair IV comparison

        day_diffs = []
        for _, row in df_day.iterrows():
            S_i = float(row['estimated_delivery_price'])
            F_i = float(row['underlying_price'])
            K_i = float(row['strike'])
            T_i = float(row['time_to_expiry'])

            market_price = compute_price_numba(S_i, F_i, K_i, T_i, r_riskfree, market_params, nodes, weights)
            market_iv = implied_vol_numba(market_price, F_i, K_i, T_i, r_riskfree)

            hist_price = compute_price_numba(S_i, F_i, K_i, T_i, r_riskfree, hist_params, nodes, weights)
            hist_iv = implied_vol_numba(hist_price, F_i, K_i, T_i, r_riskfree)

            day_diffs.append((market_iv - hist_iv))

        if len(day_diffs) == 0:
            diff_pct.append(np.nan)
        else:
            diff_pct.append(100.0 * float(np.mean(day_diffs)))

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(diff_pct, color='tab:blue', marker='o', markersize=3, linewidth=1.5,
            label='Market-implied IV - Historical-implied IV')
    ax.axhline(y=0.0, color='tab:red', linestyle='--', linewidth=1.2, label='0% baseline')
    ax.set_title(f'IV Difference vs {historical_name} Historical Params')
    ax.set_ylabel('IV Difference (%)')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='best')
    apply_date_axis(ax, plot_dates[:n_days], 'Date')
    plt.tight_layout()
    plt.show()

P_5y_params = [17.6523, 0.2676, 0.2676, 2.4189, -0.2120, 17.6679, 0.0017, 0.0574] # final
P_6m_params = [14.212, 0.3198, 0.3198, 2.1199, -0.259, 6.957, -0.0325, 0.1801] # final

plot_param_progression_against_baseline(bestparamslist, P_5y_params, 'P_5y', plot_dates)
plot_param_progression_against_baseline(bestparamslist, P_6m_params, 'P_6m', plot_dates)

plot_market_vs_historical_iv_diff(bestparamslist, P_6m_params, '6m', df_list, plot_dates, r, GL_NODES, GL_WEIGHTS)
plot_market_vs_historical_iv_diff(bestparamslist, P_5y_params, '5y', df_list, plot_dates, r, GL_NODES, GL_WEIGHTS)

plot_market_vs_model_iv(df, trained_params, r, GL_NODES, GL_WEIGHTS, "Market vs Bates Model")
plot_market_vs_model_iv(df, trained_paramsH, r, GL_NODES, GL_WEIGHTS, "Market vs Heston Model")
plot_market_vs_model_iv(df, trained_paramsBS, r, GL_NODES, GL_WEIGHTS, "Market vs Black-76 Model")