import os
import json
import time
import numpy as np
import pandas as pd
from a_init_q import ModelFixedBB
from typing import List, Dict, Tuple, Any
from b_qlearning import simulate_game
from main.equilibrium import calculate_competitive_prices, calculate_monopoly_prices
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Patch

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

def run_single_simulation(params: dict, session_id: int, results_base_dir: str = "simulation_runs_output"):
    """
    Run a single simulation session and return convergence results.
    After convergence, compute payoffs and all relevant metrics by averaging over the last
    target period horizon (e.g., 100,000 periods), as described in the provided instructions.
    """
    game = None
    try:
        game = ModelFixedBB(**params)
    except Exception as e:
        print(f"❌ Error during ModelFixedBB initialization for session {session_id} (m={params.get('m')}): {e}")
        return None

    sim_results = simulate_game(game)
    raw_price_history_tuples = sim_results.get('price_history')

    if not raw_price_history_tuples:
        print(f"Warning: No price history returned for session {session_id} (m={params.get('m')}).")
        return None

    prices_over_time = np.array([item[1] for item in raw_price_history_tuples], dtype=float)
    if prices_over_time.size == 0:
        print(f"Warning: Price history was empty after extraction for session {session_id} (m={params.get('m')}).")
        return None

    history_log_interval = sim_results.get('history_log_interval', 1)
    if history_log_interval is None or history_log_interval <= 0:
        print(f"⚠️ Warning: 'history_log_interval' is invalid. Assuming 1.")
        history_log_interval = 1

    target_averaging_period_horizon = params.get('tstable', 100_000)
    num_logged_points_for_horizon = max(1, int(round(target_averaging_period_horizon / history_log_interval)))
    actual_avg_window_logged_points = min(num_logged_points_for_horizon, prices_over_time.shape[0])

    if actual_avg_window_logged_points < 1:
        print(f"Warning: Not enough logged data to average over for target horizon in session {session_id}.")
        return None

    print(f"ℹ️ Session {session_id}: Averaging metrics over the last {actual_avg_window_logged_points} logged data points.")
    stable_prices = prices_over_time[-actual_avg_window_logged_points:]

    if stable_prices.shape[0] == 0:
        print(f"Error: 'stable_prices' window is empty for session {session_id}.")
        return None

    try:
        profits_over_time = np.array([game.compute_profits(prices) for prices in stable_prices])
        avg_profits = np.nanmean(profits_over_time, axis=0)
        final_prices = np.nanmean(stable_prices, axis=0)

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

        converged = not (np.isnan(final_prices).any() or (avg_profits is not None and np.isnan(avg_profits).any()) or np.isnan(avg_cs))
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

        cs_welfare_ratio = np.nan
        denominator_cs = cs_bn - cs_m
        if abs(denominator_cs) > 1e-9:
            cs_welfare_ratio = (cs_bn - avg_cs) / denominator_cs

        return {
            'converged': converged, 'final_prices': final_prices, 'profits': avg_profits,
            'consumer_surplus': avg_cs, 'session_id': session_id, 'm_value': params.get('m'),
            'total_profit': total_profit, 'total_profit_ratio': total_profit_ratio,
            'cs_welfare_ratio': cs_welfare_ratio, 'benchmark_metrics': benchmark_metrics, 'game_instance': game
        }
    except Exception as e:
        print(f"❌ Error calculating metrics for session {session_id} (m={params.get('m')}): {e}")
        return None

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

def run_multiple_sessions(base_params: dict, num_sessions: int = 100):
    """Run multiple simulation sessions and aggregate results"""
    m_val = base_params.get('m')
    print(f"\n--- Running {num_sessions} sessions for m={m_val} ---")
    session_results_list = []
    base_seed = base_params.get('seed', 555)
    np.random.seed(base_seed)

    for session_idx in range(num_sessions):
        session_params = base_params.copy()
        session_params['seed'] = base_seed + (m_val if m_val else 1) * 10000 + session_idx
        np.random.seed(session_params['seed'])
        result = run_single_simulation(session_params, session_idx, "temp_sessions_output")
        if result:
            session_results_list.append(result)
            print(f"Session {session_idx} (m={m_val}): {'Converged' if result['converged'] else 'Did not converge'}")

    convergent_results = [r for r in session_results_list if r['converged']]
    convergence_rate = len(convergent_results) / len(session_results_list) if session_results_list else 0.0
    print(f"  m={m_val}: Convergence rate: {convergence_rate:.2%} ({len(convergent_results)}/{len(session_results_list)})")

    num_agents = base_params.get('N', 1)
    if not convergent_results:
        return {
            'm': m_val, 'convergence_rate': convergence_rate, 'num_convergent_sessions': 0,
            'avg_prices': [np.nan] * num_agents, 'avg_profits_per_agent': [np.nan] * num_agents,
            'avg_consumer_surplus': np.nan, 'avg_total_profit': np.nan,
            'avg_total_profit_ratio': np.nan, 'avg_cs_welfare_ratio': np.nan,
            'all_convergent_prices_per_agent': [[] for _ in range(num_agents)]
        }

    # Calculate averages and collect all prices from convergent results
    avg_prices = np.nanmean(np.array([r['final_prices'] for r in convergent_results]), axis=0)
    avg_profits_per_agent = np.nanmean(np.array([r['profits'] for r in convergent_results]), axis=0)
    
    valid_cs = [r['consumer_surplus'] for r in convergent_results if np.isfinite(r['consumer_surplus'])]
    avg_consumer_surplus = np.nanmean(valid_cs) if valid_cs else np.nan

    valid_profits = [r['total_profit'] for r in convergent_results if np.isfinite(r['total_profit'])]
    avg_total_profit = np.nanmean(valid_profits) if valid_profits else np.nan

    valid_profit_ratios = [r['total_profit_ratio'] for r in convergent_results if np.isfinite(r['total_profit_ratio'])]
    avg_total_profit_ratio = np.nanmean(valid_profit_ratios) if valid_profit_ratios else np.nan

    valid_cs_ratios = [r['cs_welfare_ratio'] for r in convergent_results if np.isfinite(r['cs_welfare_ratio'])]
    avg_cs_welfare_ratio = np.nanmean(valid_cs_ratios) if valid_cs_ratios else np.nan
    
    # Collect all prices from convergent sessions for each agent
    all_prices_raw = [r['final_prices'] for r in convergent_results]
    all_prices_per_agent_T = np.array(all_prices_raw).T

    calc_mode_params = {**base_params, 'calculator_mode': True}
    temp_game = ModelFixedBB(**calc_mode_params)
    benchmark_metrics = convergent_results[0]['benchmark_metrics']

    # Add 'label' to agent_info for each agent
    agent_info = []
    for i, ac in enumerate(temp_game.agent_types):
        agent_id = ac['id']
        quality_type = ac['quality_type']
        position = temp_game.agent_positions[agent_id]
        label = f"Agent {i} ({quality_type}-Quality)"
        agent_info.append({
            'agent_idx': i,
            'quality_type': quality_type,
            'position': position,
            'label': label
        })

    return {
        'm': m_val, 'convergence_rate': convergence_rate, 'num_convergent_sessions': len(convergent_results),
        'avg_prices': avg_prices.tolist(), 'avg_profits_per_agent': avg_profits_per_agent.tolist(),
        'avg_consumer_surplus': avg_consumer_surplus, 'avg_total_profit': avg_total_profit,
        'avg_total_profit_ratio': avg_total_profit_ratio, 'avg_cs_welfare_ratio': avg_cs_welfare_ratio,
        'benchmark_bn_profits_per_agent': benchmark_metrics["bn_profits_per_agent"].tolist(),
        'benchmark_monopoly_profits_per_agent': benchmark_metrics["monopoly_profits_per_agent"].tolist(),
        'benchmark_bn_prices_per_agent': benchmark_metrics["bn_prices_per_agent"].tolist(),
        'benchmark_monopoly_prices_per_agent': benchmark_metrics["monopoly_prices_per_agent"].tolist(),
        'benchmark_cs_bn': benchmark_metrics["cs_bn"], 'benchmark_cs_m': benchmark_metrics["cs_m"],
        'agent_info': agent_info,
        'all_convergent_prices_per_agent': all_prices_per_agent_T.tolist() if all_prices_per_agent_T.size > 0 else [[] for _ in range(num_agents)]
    }

def run_buybox_analysis(base_params: dict, m_values: list, num_sessions: int = 100):
    """Run analysis varying buy box size m"""
    print(f"\n=== Buy Box Size Analysis (m values: {m_values}) ===")
    all_m_results = []
    original_seed = base_params.get('seed', 555)
    np.random.seed(original_seed)

    for m_val_loop in m_values:
        current_m_params = base_params.copy()
        current_m_params['m'] = m_val_loop
        current_m_params['seed'] = original_seed
        all_m_results.append(run_multiple_sessions(current_m_params, num_sessions))

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = f"buybox_N{base_params['N']}_theta{base_params['theta']}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "results.json"), 'w') as f:
        json.dump(convert_to_native_python_types(all_m_results), f, indent=4)
    with open(os.path.join(output_dir, "parameters.json"), 'w') as f:
        json.dump(convert_to_native_python_types(base_params), f, indent=4)

    generate_buybox_plots(all_m_results, base_params, m_values, output_dir)
    print(f"\n✅ Buy box analysis complete. Results saved to: {output_dir}")
    return all_m_results

def generate_buybox_plots(results_by_m: list, base_params_info: dict, m_plot_axis: list, output_dir: str):
    """
    Generate comprehensive plots for buybox analysis with violin plots for price distribution.
    """

    if not any(r.get('num_convergent_sessions', 0) > 0 for r in results_by_m):
        print("❌ No valid convergent session data found. Cannot generate plots.")
        return

    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 11, 'font.family': 'serif', 'axes.labelsize': 12, 'axes.titlesize': 13,
        'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 10,
        'figure.titlesize': 14, 'lines.linewidth': 2.5, 'lines.markersize': 8, 'axes.grid': False
    })

    colors = {
        'simulated': '#2E86AB', 'competitive': '#A23B72', 'monopoly': '#F18F01',
        'consumer_surplus': '#C73E1D', 'total_profit': '#4A5D23', 'reference_line': '#6C757D'
    }

    agent_box_info = {}
    if any(r.get('agent_info') for r in results_by_m):
        for r in results_by_m:
            for agent_info in r.get('agent_info', []):
                idx = agent_info['agent_idx']
                if idx not in agent_box_info:
                    agent_box_info[idx] = {'quality_type': agent_info['quality_type'], 'entry_m': None, 'always_out': True}
                if agent_info['position'] == 'in':
                    agent_box_info[idx]['always_out'] = False
                    if agent_box_info[idx]['entry_m'] is None:
                        agent_box_info[idx]['entry_m'] = r['m']

    for agent_idx, box_info in agent_box_info.items():
        fig, ax = plt.subplots(1, 1, figsize=(12, 7))
        quality_type = box_info['quality_type']

        # --- Data Extraction ---
        mean_prices, comp_prices, mono_prices = [], [], []
        all_session_prices = []

        for r in results_by_m:
            avg_prices = r.get('avg_prices', [])
            mean_prices.append(avg_prices[agent_idx] if len(avg_prices) > agent_idx else np.nan)
            comp_prices_list = r.get('benchmark_bn_prices_per_agent', [])
            comp_prices.append(comp_prices_list[agent_idx] if len(comp_prices_list) > agent_idx else np.nan)
            mono_prices_list = r.get('benchmark_monopoly_prices_per_agent', [])
            mono_prices.append(mono_prices_list[agent_idx] if len(mono_prices_list) > agent_idx else np.nan)
            session_prices_list = r.get('all_convergent_prices_per_agent', [])
            session_prices = session_prices_list[agent_idx] if len(session_prices_list) > agent_idx else []
            all_session_prices.append(session_prices)

        # --- Plotting ---
        if box_info['always_out']:
            position_status = "Out of Box"
        elif box_info['entry_m']:
            position_status = f"Enters Buy Box from m={box_info['entry_m']}"
        else:
            position_status = "In Box"
        fig.suptitle(f'Agent {agent_idx} ({quality_type}-Quality) - {position_status}', fontsize=14, fontweight='bold', y=0.98)

        # Add density violins using PolyCollection to show price distribution at each 'm'.
        verts = []
        for i, m in enumerate(m_plot_axis):
            prices = all_session_prices[i]
            if len(prices) > 5:
                counts, bin_edges = np.histogram(prices, bins=15, density=True)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                max_count = counts.max() if counts.max() > 0 else 1
                scaled_density = (counts / max_count) * 0.3
                verts.append(list(zip([m + sd for sd in scaled_density], bin_centers)) +
                             list(zip([m - sd for sd in scaled_density][::-1], bin_centers[::-1])))
        if verts:
            poly_collection = PolyCollection(verts, facecolors=colors['simulated'], edgecolors=colors['simulated'], alpha=0.5)
            ax.add_collection(poly_collection)

        # Plot mean and benchmark lines on top of the distributions
        ax.plot(m_plot_axis, mean_prices, 'o-', color=colors['simulated'], linewidth=3, markersize=8, label='Simulated (Mean)')
        ax.plot(m_plot_axis, comp_prices, 's--', color=colors['competitive'], linewidth=2.5, markersize=8, alpha=0.9, label='Competitive')
        ax.plot(m_plot_axis, mono_prices, '^--', color=colors['monopoly'], linewidth=2.5, markersize=8, alpha=0.9, label='Fully Collusive')

        ax.set_xlabel('Buy Box Size (m)', fontweight='bold')
        ax.set_ylabel('Price', fontweight='bold')
        ax.set_xticks(m_plot_axis)

        legend_elements = [
            Patch(facecolor=colors['simulated'], alpha=0.5, label='Price Distribution Density'),
            plt.Line2D([0], [0], color=colors['simulated'], lw=3, marker='o', label='Simulated (Mean)'),
            plt.Line2D([0], [0], color=colors['competitive'], lw=2.5, ls='--', marker='s', label='Competitive'),
            plt.Line2D([0], [0], color=colors['monopoly'], lw=2.5, ls='--', marker='^', label='Fully Collusive')
        ]
        ax.legend(handles=legend_elements, loc='best', fontsize=10, frameon=True, fancybox=True, shadow=True)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plot_path = os.path.join(output_dir, f"agent_{agent_idx}_{quality_type}_prices_dist.png")
        try:
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"📊 Agent {agent_idx} distribution plot saved to: {plot_path}")
        except Exception as e:
            print(f"❌ Error saving plot for Agent {agent_idx}: {e}")
        plt.close(fig)

    # --- Create combined welfare plots ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"Welfare Metrics (N={base_params_info['N']}, K={base_params_info['K']})", fontsize=14, fontweight='bold', y=0.96)

    cs_sim = [r['avg_consumer_surplus'] if np.isfinite(r['avg_consumer_surplus']) else np.nan for r in results_by_m]
    total_profits = [r['avg_total_profit'] if np.isfinite(r['avg_total_profit']) else np.nan for r in results_by_m]

    ax1_twin = ax1.twinx()
    ax1.plot(m_plot_axis, cs_sim, 'o-', color=colors['consumer_surplus'], label='Consumer Surplus')
    ax1_twin.plot(m_plot_axis, total_profits, 'D-', color=colors['total_profit'], label='Total Profit')
    ax1.set(xlabel='Buy Box Size (m)', ylabel='Consumer Surplus', title='Consumer Surplus & Total Profit', xticks=m_plot_axis)
    ax1_twin.set_ylabel('Total Profit', color=colors['total_profit'])
    ax1.yaxis.label.set_color(colors['consumer_surplus'])
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

    cs_ratios = [r['avg_cs_welfare_ratio'] if np.isfinite(r['avg_cs_welfare_ratio']) else np.nan for r in results_by_m]
    profit_ratios = [r['avg_total_profit_ratio'] if np.isfinite(r['avg_total_profit_ratio']) else np.nan for r in results_by_m]
    ax2.plot(m_plot_axis, cs_ratios, 'o-', color=colors['consumer_surplus'], label='CS Welfare Ratio')
    ax2.plot(m_plot_axis, profit_ratios, 'D-', color=colors['total_profit'], label='Total Profit Ratio')
    ax2.axhline(y=0, color=colors['competitive'], linestyle='--', alpha=0.7, label='Competitive')
    ax2.axhline(y=1, color=colors['monopoly'], linestyle='--', alpha=0.7, label='Monopoly')
    ax2.set(xlabel='Buy Box Size (m)', ylabel='Ratio', title='Normalized Welfare Measure', xticks=m_plot_axis)
    ax2.legend(loc='best')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    welfare_plot_path = os.path.join(output_dir, "buybox_welfare_analysis.png")
    try:
        plt.savefig(welfare_plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Welfare analysis plot saved to: {welfare_plot_path}")
    except Exception as e:
        print(f"❌ Error saving welfare plot: {e}")
    plt.close(fig)