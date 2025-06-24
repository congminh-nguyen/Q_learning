# main1.py
import numpy as np
import os
import json
from c_analysis import run_buybox_analysis, convert_to_native_python_types

def main():
    """
    Main function to run the buybox analysis with reproducible results.
    This will also generate and save the relevant plots as part of the analysis.
    """
    # Fixed parameters for reproducibility
    # We run for theta = 1.5 and 15; while N increases from 3 to 4 with K = 1.
    base_simulation_params = dict(
        N=3,
        K=1, 
        c=5, 
        q_H=50.0, 
        q_L=30.0,
        theta=5, 
        Delta=0.75, 
        alpha_initial=0.15,
        beta_exploration=1e-5,
        delta_rl=0.7, 
        grid_size=10, 
        tstable=100_000, 
        tmax=10_000_000,
        seed=5
    )
    
    # Set m values to test
    m_values_to_test = [1, 2, 3] 
    
    # Number of sessions per m value
    num_sessions_per_m = 100
    
    # Create output directory for results
    output_dir = "simulation_runs_output"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Running analysis with parameters: {base_simulation_params}")
    print(f"Testing m values: {m_values_to_test}")
    print(f"Sessions per m value: {num_sessions_per_m}")
    
    # Run the analysis (this will also generate and save plots via c_analysis)
    results_summary = run_buybox_analysis(base_simulation_params, m_values_to_test, num_sessions_per_m)
    # NOTE: run_buybox_analysis (see c_analysis.py) generates and saves all relevant plots
    # including price distribution plots and welfare plots, in a timestamped output directory.
    # The output directory and plot file paths are printed by that function.
    
    # Save results to JSON file in our own output directory for convenience
    results_file = os.path.join(output_dir, "simulation_results.json")
    with open(results_file, 'w') as f:
        json.dump(convert_to_native_python_types(results_summary), f, indent=2)
    print(f"\nResults saved to: {results_file}")
    
    # Print summary results
    print(f"\n=== OVERALL SUMMARY ===")
    for result_m in results_summary:
        print(f"\nFor m = {result_m['m']}:")
        print(f"  Convergence Rate: {result_m['convergence_rate']:.2%}")
        if result_m['num_convergent_sessions'] > 0:
            print(f"  Agent positions:")
            for agent_info in result_m['agent_info']:
                print(f"    {agent_info['label']}: {agent_info['position']}")
            print(f"  Avg Prices: {[f'{p:.2f}' for p in result_m['avg_prices']]}")
            print(f"  Avg Profits/Agent: {[f'{p:.2f}' for p in result_m['avg_profits_per_agent']]}")
            
            cs_value = result_m['avg_consumer_surplus']
            if not np.isnan(cs_value):
                print(f"  Avg Consumer Surplus: {cs_value:.3f}")
            else:
                print(f"  Avg Consumer Surplus: N/A")
                
            total_profit_value = result_m['avg_total_profit']
            if not np.isnan(total_profit_value):
                print(f"  Avg Total Profit: {total_profit_value:.3f}")
            else:
                print(f"  Avg Total Profit: N/A")
                
            total_pr_value = result_m['avg_total_profit_ratio']
            if not np.isnan(total_pr_value):
                print(f"  Avg Total Profit Ratio: {total_pr_value:.3f}")
            else:
                print(f"  Avg Total Profit Ratio: N/A")
                
            cs_wr_value = result_m['avg_cs_welfare_ratio']
            if not np.isnan(cs_wr_value):
                print(f"  Avg CS Welfare Ratio: {cs_wr_value:.3f}")
            else:
                print(f"  Avg CS Welfare Ratio: N/A")
        else:
            print("  No convergent sessions for this m value.")

    print("\nNOTE: All relevant plots (price distributions, welfare metrics) are generated and saved automatically by run_buybox_analysis (see c_analysis.py).")

if __name__ == "__main__":
    main()