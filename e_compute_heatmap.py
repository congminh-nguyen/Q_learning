# e_heatmap_compute.py (UPDATED with CS / (q_H - c) metric)
import os
import json
import time
import numpy as np
import pandas as pd

# Assuming a_init_q.py, b_qlearning.py, and main.equilibrium.py are available
from a_init_q import ModelFixedBB
from b_qlearning import simulate_game
from main.equilibrium import calculate_competitive_prices, calculate_monopoly_prices
from typing import List, Dict, Tuple, Any

# --- Re-use existing utility functions ---
def convert_to_native_python_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        if np.isnan(obj) or np.isinf(obj):
            return str(obj)
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {str(key): convert_to_native_python_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_python_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_native_python_types(item) for item in obj)
    else:
        return obj

def calculate_benchmark_metrics(params: dict, temp_game_for_config: ModelFixedBB):
    """Calculate BN equilibrium and monopoly benchmark metrics using equilibrium.py"""
    bn_eq_results = calculate_competitive_prices(
        theta=params['theta'], q_H=params['q_H'], q_L=params['q_L'],
        Delta=params['Delta'], c=params['c'], N=params['N'], K=params['K'], m=params['m']
    )
    m_eq_results = calculate_monopoly_prices(
        theta=params['theta'], q_H=params['q_H'], q_L=params['q_L'],
        Delta=params['Delta'], c=params['c'], N=params['N'], K=params['K'], m=params['m']
    )

    num_agents = temp_game_for_config.N
    bn_profits_agent = np.full(num_agents, np.nan)
    monopoly_profits_agent = np.full(num_agents, np.nan)
    bn_prices_agent = np.full(num_agents, np.nan)
    monopoly_prices_agent = np.full(num_agents, np.nan)

    for i, agent_config in enumerate(temp_game_for_config.agent_types):
        q_type = agent_config['quality_type']
        position = temp_game_for_config.agent_positions[agent_config['id']]
        pos_prefix = 'tilde' if position == 'in' else 'hat'

        bn_profits_agent[i] = bn_eq_results.get(f"{pos_prefix}_profit_{q_type}", np.nan)
        monopoly_profits_agent[i] = m_eq_results.get(f"{pos_prefix}_monopoly_profit_{q_type}_indiv", np.nan)
        bn_prices_agent[i] = bn_eq_results.get(f"{pos_prefix}_P_{q_type}", np.nan)
        monopoly_prices_agent[i] = m_eq_results.get(f"{pos_prefix}_P_M_{q_type}", np.nan)

    return {
        "bn_profits_per_agent": bn_profits_agent, "monopoly_profits_per_agent": monopoly_profits_agent,
        "bn_prices_per_agent": bn_prices_agent, "monopoly_prices_per_agent": monopoly_prices_agent,
        "cs_bn": bn_eq_results.get('consumer_welfare', np.nan), "cs_m": m_eq_results.get('consumer_welfare_monopoly', np.nan)
    }

def run_single_simulation_for_heatmap(params: dict, session_id: int):
    """
    Run a single simulation session and return convergence results.
    After convergence, compute payoffs and all relevant metrics by averaging over the last
    target period horizon (e.g., 100,000 periods), as described in the provided instructions.
    """
    game = None
    try:
        game = ModelFixedBB(**params)
    except Exception as e:
        print(f"❌ Error during ModelFixedBB initialization for session {session_id}: {e}")
        return None

    sim_results = simulate_game(game)
    raw_price_history_tuples = sim_results.get('price_history')

    if not raw_price_history_tuples:
        return None

    prices_over_time = np.array([item[1] for item in raw_price_history_tuples], dtype=float)
    if prices_over_time.size == 0:
        return None

    history_log_interval = sim_results.get('history_log_interval', 1)
    if history_log_interval is None or history_log_interval <= 0:
        history_log_interval = 1

    target_averaging_period_horizon = params.get('tstable', 100_000)
    num_logged_points_for_horizon = max(1, int(round(target_averaging_period_horizon / history_log_interval)))
    actual_avg_window_logged_points = min(num_logged_points_for_horizon, prices_over_time.shape[0])

    if actual_avg_window_logged_points < 1:
        return None

    stable_prices = prices_over_time[-actual_avg_window_logged_points:]

    if stable_prices.shape[0] == 0:
        return None

    try:
        profits_over_time = np.array([game.compute_profits(prices) for prices in stable_prices])
        avg_profits = np.nanmean(profits_over_time, axis=0)

        cs_over_time = []
        for prices_in_window in stable_prices:
            E_T_stab, max_util_ot = game.get_utility_components_for_cs(prices_in_window)
            cs_val = np.nan
            if E_T_stab > 1e-100 and not (np.isinf(max_util_ot) and max_util_ot < 0):
                cs_val = game.theta * (np.log(E_T_stab) + max_util_ot)
            elif max_util_ot == -np.inf:
                cs_val = -np.inf
            cs_over_time.append(cs_val)
        avg_cs = np.nanmean(np.array(cs_over_time))

        converged = not ( (avg_profits is not None and np.isnan(avg_profits).any()) or np.isnan(avg_cs))
        
        benchmark_metrics = calculate_benchmark_metrics(params, game)
        bn_profits_agent = benchmark_metrics["bn_profits_per_agent"]
        monopoly_profits_agent = benchmark_metrics["monopoly_profits_per_agent"]
        cs_bn = benchmark_metrics["cs_bn"]
        cs_m = benchmark_metrics["cs_m"]

        total_profit = np.nansum(avg_profits)
        total_bn_profit = np.nansum(bn_profits_agent)
        total_monopoly_profit = np.nansum(monopoly_profits_agent)
        
        total_profit_ratio = np.nan
        total_denominator = total_monopoly_profit - total_bn_profit
        if abs(total_denominator) > 1e-9:
            total_profit_ratio = (total_profit - total_bn_profit) / total_denominator
        
        # Clip profit ratio to be between 0 and 1
        total_profit_ratio_clipped = np.clip(total_profit_ratio, 0, 1)

        cs_welfare_ratio = np.nan
        denominator_cs = cs_bn - cs_m
        if abs(denominator_cs) > 1e-9:
            cs_welfare_ratio = (cs_bn - avg_cs) / denominator_cs
        
        # Clip CS welfare ratio to be between 0 and 1
        cs_welfare_ratio_clipped = np.clip(cs_welfare_ratio, 0, 1)

        # NEW METRIC: Consumer Surplus divided by (q_H - c)
        q_H = params['q_H']
        c = params['c']
        cs_normalized_by_quality_cost = np.nan
        if (q_H - c) > 1e-9: # Avoid division by zero or very small numbers
            cs_normalized_by_quality_cost = avg_cs / (q_H - c)


        return {
            'converged': converged,
            'consumer_surplus_raw': avg_cs, # Also return raw CS for completeness
            'consumer_surplus_per_quality_cost': cs_normalized_by_quality_cost, # New metric
            'consumer_surplus_normalized': cs_welfare_ratio_clipped,
            'total_profit_normalized': total_profit_ratio_clipped
        }
    except Exception as e:
        print(f"❌ Error calculating metrics for session {session_id}: {e}")
        return None

def run_heatmap_sweep_and_save(base_params: dict, m_values: List[int], alpha_vals: List[float], beta_vals: List[float], num_sessions_per_point: int = 1, output_dir: str = "heatmap_data"):
    """
    Runs simulations across varying alpha and beta, collecting normalized welfare and profit data,
    then saves the results to a CSV file.
    Averages over `num_sessions_per_point` for each (alpha, beta, m) combination.
    """
    print(f"\n=== Running Heatmap Sweep for m values: {m_values} ===")
    all_sweep_results = []
    
    param_grid_template = []
    for m_val in m_values:
        for alpha_val in alpha_vals:
            for beta_val in beta_vals:
                param_grid_template.append({
                    'm': m_val,
                    'alpha_initial': alpha_val,
                    'beta_exploration': beta_val
                })

    total_runs = len(param_grid_template) * num_sessions_per_point
    print(f"Total simulation runs: {total_runs} (across {len(param_grid_template)} parameter combinations)")

    session_counter = 0
    for params_set in param_grid_template:
        session_results_for_point = []
        for i in range(num_sessions_per_point):
            current_params = base_params.copy()
            current_params.update(params_set)
            
            current_params['seed'] = (base_params.get('seed', 555) + 
                                      params_set['m'] * 1000000 + 
                                      int(params_set['alpha_initial'] * 10000) + # Scale alpha for seed
                                      int(params_set['beta_exploration'] * 1e7) + # Scale beta for seed
                                      i) 
            
            result = run_single_simulation_for_heatmap(current_params, session_counter)
            if result and result['converged']:
                session_results_for_point.append(result)
            session_counter += 1
            if session_counter % 50 == 0:
                print(f"  Completed {session_counter}/{total_runs} runs.")
        
        # Average results for the current (m, alpha, beta) point
        if session_results_for_point:
            avg_cs_raw = np.nanmean([r['consumer_surplus_raw'] for r in session_results_for_point])
            avg_cs_per_quality_cost = np.nanmean([r['consumer_surplus_per_quality_cost'] for r in session_results_for_point])
            avg_cs_normalized = np.nanmean([r['consumer_surplus_normalized'] for r in session_results_for_point])
            avg_profit_normalized = np.nanmean([r['total_profit_normalized'] for r in session_results_for_point])
            
            all_sweep_results.append({
                'm': params_set['m'],
                'alpha_initial': params_set['alpha_initial'],
                'beta_exploration': params_set['beta_exploration'],
                'consumer_surplus_raw': avg_cs_raw, # Include raw CS
                'consumer_surplus_per_quality_cost': avg_cs_per_quality_cost, # New metric
                'consumer_surplus_normalized': avg_cs_normalized, 
                'total_profit_normalized': avg_profit_normalized
            })
        else:
            all_sweep_results.append({
                'm': params_set['m'],
                'alpha_initial': params_set['alpha_initial'],
                'beta_exploration': params_set['beta_exploration'],
                'consumer_surplus_raw': np.nan,
                'consumer_surplus_per_quality_cost': np.nan, # Ensure NaN for new metric if no convergence
                'consumer_surplus_normalized': np.nan,
                'total_profit_normalized': np.nan
            })

    print(f"Completed all {total_runs} runs.")
    heatmap_df = pd.DataFrame(all_sweep_results)
    
    # Reorder columns to place new metric before normalized ones
    cols = ['m', 'alpha_initial', 'beta_exploration', 
            'consumer_surplus_raw', 'consumer_surplus_per_quality_cost', 
            'consumer_surplus_normalized', 'total_profit_normalized']
    # Ensure all original columns are still present and in case of missing new column
    # (e.g. if the above logic changes), prevent error.
    heatmap_df = heatmap_df[cols] 

    # Save raw DataFrame to CSV
    df_save_path = os.path.join(output_dir, "heatmap_data.csv")
    heatmap_df.to_csv(df_save_path, index=False)
    print(f"\nRaw heatmap data saved to: {df_save_path}")
    return df_save_path # Return path to saved file


def main_compute_data():
    # Base simulation parameters - these remain fixed
    base_simulation_params = dict(
        N=3,
        K=1,
        c=5,
        q_H=50.0,
        q_L=30.0,
        theta=5,
        Delta=0.75,
        delta_rl=0.7,
        grid_size=10,
        tstable=100_000,
        tmax=10_000_000,
        seed=5
    )

    # Define the ranges and resolution for alpha and beta
    grid_resolution = 5 # 5x5 grid for initial exploration
    alpha_values = np.linspace(0.05, 0.25, grid_resolution).tolist()
    beta_values = np.linspace(0.5e-5, 1.5e-5, grid_resolution).tolist()
    
    # Define m values to test
    m_values_to_test = [1, 2, 3] # Example m values

    # Number of sessions per (alpha, beta, m) point
    num_sessions_per_point = 25 

    output_dir = f"heatmap_alpha_beta_data_N{base_simulation_params['N']}_m{'_'.join(map(str, m_values_to_test))}_{time.strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Starting computation for heatmap analysis with parameters: {base_simulation_params}")
    print(f"Alpha values (linspace over {grid_resolution} points): {alpha_values}")
    print(f"Beta values (linspace over {grid_resolution} points): {[f'{b:.1e}' for b in beta_values]}")
    print(f"M values: {m_values_to_test}")
    print(f"Sessions per (alpha, beta, m) point: {num_sessions_per_point}")

    # Run the sweep and save results
    run_heatmap_sweep_and_save(base_simulation_params, m_values_to_test, alpha_values, beta_values, num_sessions_per_point, output_dir)

    print("\nComputation complete. Data saved to CSV.")

if __name__ == "__main__":
    main_compute_data()