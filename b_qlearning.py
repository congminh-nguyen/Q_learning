# qlearning_fixed_bb.py
import numpy as np
from typing import Dict, List, Tuple, Any

from a_init_q import ModelFixedBB 

def pick_strategies(
    game: ModelFixedBB,
    current_q_state_tuple: Tuple[int, ...], 
    t: int
) -> np.ndarray:
    """
    Determines actions for each agent based on epsilon-greedy strategy.
    State tuple is (price_idx_agent0, ..., price_idx_agentN-1).
    """
    chosen_action_indices = np.zeros(game.N, dtype=int)
    pr_explore = np.exp(-game.beta_exploration * t)
    explore_flags = (np.random.rand(game.N) < pr_explore)

    for agent_loop_idx in range(game.N):
        agent_id = game.agent_types[agent_loop_idx]['id']
        agent_action_set = game.A_n.get(agent_id)

        if agent_action_set is None or not agent_action_set.size:
            # This should ideally be prevented by robust A_n initialization in ModelFixedBB
            chosen_action_indices[agent_loop_idx] = 0 # Default action
            # Log much less frequently to avoid flooding console
            if t % 100000 == 0: 
                 print(f"⚠️ pick_strategies (t={t}): Agent {agent_id} has no actions. Defaulting to 0.")
            continue 
        
        num_available_actions = len(agent_action_set)

        if explore_flags[agent_loop_idx]:  # Explore
            chosen_action_indices[agent_loop_idx] = np.random.randint(0, num_available_actions)
        else:  # Exploit
            # Q-table for agent_id, indexed by the current state tuple
            q_values_for_state = game.Q[agent_id][current_q_state_tuple]
            # Consider only Q-values for actions available to this agent
            q_values_for_available_actions = q_values_for_state[:num_available_actions]
            
            if not q_values_for_available_actions.size:
                 # Fallback if, for some reason, there are no Q-values for available actions
                 chosen_action_indices[agent_loop_idx] = np.random.randint(0, num_available_actions)
                 if t % 100000 == 0:
                      print(f"⚠️ pick_strategies (t={t}): Agent {agent_id} no Q-vals for exploit. Defaulting to random.")
                 continue
            chosen_action_indices[agent_loop_idx] = np.argmax(q_values_for_available_actions)
                
    return chosen_action_indices

def update_q_and_policy_stability(
    game: ModelFixedBB,
    s_tuple: Tuple[int, ...],            # Current state (price_indices_agents_t-1)
    action_indices_chosen_t: np.ndarray, # Joint action (price_indices_agents_t) just taken
    s_prime_tuple: Tuple[int, ...],      # Next state (price_indices_agents_t)
    rewards_t: np.ndarray,               # Rewards for actions_chosen_t
    t: int                               # Current global timestep
) -> Tuple[float, bool]: # Returns (max_q_change_this_step, all_policies_stable_this_q_update_step)
    """
    Updates Q-values for all agents based on the experience (s, a, r, s').
    Tracks if any agent's greedy policy for state 's_tuple' changed due to this Q-update.
    Returns the maximum Q-value change in this step and a boolean indicating if all
    agents' policies (for s_tuple) were stable during this update.
    """
    max_q_change_this_step = 0.0
    current_alpha = game.alpha_initial  # Use constant alpha
    all_policies_stable_this_q_update_step = True # Assume stability until a flip is detected

    for agent_loop_idx in range(game.N):
        agent_id = game.agent_types[agent_loop_idx]['id']
        action_idx_agent_n_took = action_indices_chosen_t[agent_loop_idx] # Agent's own action index

        # Q-table index for the specific state-action pair Q_n(s_t, a_n_t)
        q_table_index_s_a = s_tuple + (action_idx_agent_n_took,)
        
        agent_action_set_s = game.A_n.get(agent_id) # Actions available to this agent
        if agent_action_set_s is None or not agent_action_set_s.size:
            # Agent has no actions, skip (should be rare if A_n is well-defined)
            continue
        num_actions_in_s = len(agent_action_set_s)

        try:
            # Greedy action for agent_id in s_tuple BEFORE this specific Q-value update
            old_argmax = np.argmax(game.Q[agent_id][s_tuple][:num_actions_in_s])
            old_q_value = game.Q[agent_id][q_table_index_s_a]
        except IndexError:
            # Only print errors very rarely to avoid performance impact
            if t % 500000 == 0:
                print(f"ERROR: Q-table index out of bounds for old_q_value/old_argmax. Agent {agent_id}, action_idx {action_idx_agent_n_took}.")
                print(f"   s_tuple: {s_tuple}, Q-table shape for state: {game.Q[agent_id][s_tuple].shape}, num_actions_in_s: {num_actions_in_s}")
            all_policies_stable_this_q_update_step = False # Mark instability
            continue

        # Determine max_a' Q_n(s_t+1, a')
        agent_action_set_s_prime = game.A_n.get(agent_id) # Actions for s_prime (same as for s in fixed action set model)
        max_next_q_value = -np.inf 

        if agent_action_set_s_prime is not None and agent_action_set_s_prime.size:
            num_actions_in_s_prime = len(agent_action_set_s_prime)
            q_values_s_prime_all_grid = game.Q[agent_id][s_prime_tuple] # Q-values for all grid_size actions
            max_next_q_value = np.max(q_values_s_prime_all_grid[:num_actions_in_s_prime]) # Max over available actions
        
        # Handle -np.inf for robust update
        discounted_future_value = game.delta_rl * max_next_q_value
        if np.isneginf(discounted_future_value) and game.delta_rl > 0 : 
             discounted_future_value = -1e12 # Large negative to prevent nan with finite rewards
        elif np.isneginf(max_next_q_value) and game.delta_rl == 0 :
             discounted_future_value = 0

        # Q-update
        reward_agent_n = rewards_t[agent_loop_idx]
        # Prevent inf - inf = nan or -inf + inf = nan
        if np.isneginf(reward_agent_n) and np.isposinf(discounted_future_value): new_q_value = -1e12 
        elif np.isposinf(reward_agent_n) and np.isneginf(discounted_future_value): new_q_value = 1e12 
        else:
            new_q_value = (1 - current_alpha) * old_q_value + \
                          current_alpha * (reward_agent_n + discounted_future_value)
        
        game.Q[agent_id][q_table_index_s_a] = new_q_value
        
        q_change = abs(new_q_value - old_q_value)
        if not np.isnan(q_change):
             if q_change > max_q_change_this_step: max_q_change_this_step = q_change
        elif not (np.isnan(new_q_value) and np.isnan(old_q_value)): # if one is nan and other isn't
             max_q_change_this_step = max(max_q_change_this_step, 1e12) # Arbitrary large change

        # Greedy action for agent_id in s_tuple AFTER this specific Q-value update
        new_argmax = np.argmax(game.Q[agent_id][s_tuple][:num_actions_in_s])
        
        if old_argmax != new_argmax:
            all_policies_stable_this_q_update_step = False # If any agent's policy for s_tuple flips, this step is unstable
            
    return max_q_change_this_step, all_policies_stable_this_q_update_step

def check_convergence(
    game: ModelFixedBB, t: int,
    global_policy_stable_steps: int, # Now a single integer counter
    q_value_deltas_history: List[float],
    convergence_window: int = 100_000, # How many recent Q-deltas to average
    q_delta_threshold: float = 1e-7   # Threshold for avg Q-delta
) -> bool:
    """
    Checks if the game has converged based on global policy stability and Q-value stability.
    """
    # Minimum number of iterations before checking convergence
    # game.tstable here is used as the target count for global_policy_stable_steps
    if t < game.tstable : # Avoid checking too early if tstable is also used as a minimum iteration count
        return False

    # Policy stability: global counter must exceed game.tstable (which is the target stability count)
    policy_has_stabilized = (global_policy_stable_steps >= game.tstable)

    # Q-value stability: average of recent max Q-value changes must be below threshold
    q_values_have_stabilized = False
    mean_recent_q_delta = np.nan
    if len(q_value_deltas_history) >= convergence_window:
        mean_recent_q_delta = np.mean(q_value_deltas_history[-convergence_window:])
        if not np.isnan(mean_recent_q_delta): # Ensure mean is a valid number
            q_values_have_stabilized = mean_recent_q_delta < q_delta_threshold
    
    # Logging much less frequently for performance
    log_frequency = max(100000, int(game.tmax // 50)) if game.tmax > 0 else 100000 
    if t > 0 and (t + 1) % log_frequency == 0:
        mean_q_delta_log = f"{mean_recent_q_delta:.3e}" if not np.isnan(mean_recent_q_delta) else "gathering_data"
        current_alpha_log = game.alpha_initial  # Use constant alpha
        current_epsilon_log = np.exp(-game.beta_exploration * t)

        print(f"\nIter {t/1e6:.2f}M/{game.tmax/1e6:.2f}M ({(t/game.tmax)*100:.1f}%): α={current_alpha_log:.2e}, ε={current_epsilon_log:.2e}")
        print(f"  Avg Q-Δ (last {convergence_window/1e3:.0f}k): {mean_q_delta_log} (Thr: {q_delta_threshold:.1e})")
        print(f"  Global Policy Stable Steps: {global_policy_stable_steps}, Target_Count: {game.tstable/1e3:.0f}k")
    
    return policy_has_stabilized and q_values_have_stabilized


def simulate_game(game: ModelFixedBB) -> Dict[str, Any]:
    """
    Simulates the Q-learning game.
    """
    # Initial state: (price_idx_agent0, ..., price_idx_agentN-1)
    current_q_state_tuple: Tuple[int, ...] = tuple(game.s0.astype(int)) 
    
    # Single global counter for policy stability, as in the original paper's logic
    global_policy_stable_steps = 0 
    
    q_value_deltas_history: List[float] = [] # Stores max Q change for each step
    
    # History logging parameters - reduce frequency for performance
    num_history_logs = 500 # Reduced from 1000 for performance
    history_interval = max(1, int(game.tmax // num_history_logs)) if game.tmax > 0 else 1
    price_history: List[Tuple[int, np.ndarray]] = [] # (timestep, actual_prices_N_agents)
    profit_history: List[Tuple[int, np.ndarray]] = [] # (timestep, profits_N_agents)
    avg_q_value_history : List[Tuple[int, float]] = [] # (timestep, avg_Q_val_across_tables)

    max_q_change_at_convergence = np.nan 
    iterations_done = 0

    print(f"🚀 Starting Q-learning (Fixed Buy Box, Simpler Global Stability): N={game.N}, K={game.K}, m={game.m}, grid_size={game.grid_size}")
    print(f"   tmax={game.tmax/1e6:.2f}M, tstable_target_count={game.tstable/1e3:.0f}k, RL_discount={game.delta_rl}")
    print(f"   alpha={game.alpha_initial:.2f}")
    print(f"   beta_expl={game.beta_exploration:.1e} (for exp decay of epsilon: e^(-beta*t))")
    print("-" * 30)

    for t in range(int(game.tmax)):
        iterations_done = t + 1
        
        # Agents pick strategies based on current_q_state_tuple and time t (for epsilon)
        chosen_action_indices_t = pick_strategies(game, current_q_state_tuple, t)
        
        # Simulate the period: get profits and the next state definition
        # In ModelFixedBB, get_period_profits_and_next_state expects only action indices
        period_outcomes = game.get_period_profits_and_next_state(chosen_action_indices_t)
        rewards_t = period_outcomes['period_profits'] 
        # For fixed buy box, next state is defined by the actions just taken
        s_prime_tuple = tuple(period_outcomes['next_q_state_definition']) 
        
        # Update Q-values and check if policies were stable during this specific Q-update step
        max_q_change_at_step, all_policies_stable_this_q_update = update_q_and_policy_stability(
            game, current_q_state_tuple, chosen_action_indices_t, s_prime_tuple,
            rewards_t, t
        )
        q_value_deltas_history.append(max_q_change_at_step)

        # Update global policy stability counter
        if all_policies_stable_this_q_update:
            global_policy_stable_steps += 1
        else:
            global_policy_stable_steps = 0 # Reset if any agent's policy flipped
        
        # Log history much less frequently for performance
        if t == 0 or (t + 1) % history_interval == 0 or t == int(game.tmax) - 1:
            actual_prices_t = np.full(game.N, np.nan)
            for agent_idx_loop in range(game.N):
                agent_id = game.agent_types[agent_idx_loop]['id']
                action_idx = chosen_action_indices_t[agent_idx_loop]
                agent_action_set = game.A_n.get(agent_id)
                if agent_action_set is not None and agent_action_set.size > action_idx :
                     actual_prices_t[agent_idx_loop] = agent_action_set[action_idx]
            price_history.append((t, actual_prices_t))
            profit_history.append((t, rewards_t))
            
            # Calculate average Q value much less frequently as it can be very slow
            log_avg_q_interval = history_interval * 50  # Increased from 10 for performance
            if t == 0 or (t+1) % log_avg_q_interval == 0 or t == int(game.tmax) -1:
                current_avg_q = 0
                total_q_entries = 0
                for agent_id_q_calc in game.Q.keys():
                    q_table = game.Q[agent_id_q_calc]
                    current_avg_q += np.nansum(q_table) # Sum of non-NaN Q values
                    total_q_entries += np.sum(~np.isnan(q_table)) # Count of non-NaN Q values
                if total_q_entries > 0:
                    avg_q_value_history.append((t, current_avg_q / total_q_entries))

        # Transition to the next state
        current_q_state_tuple = s_prime_tuple
        
        # Check for convergence much less frequently for performance
        check_conv_interval = max(10000, history_interval // 5) if history_interval > 50000 else 50000
        if t > game.tstable and (t+1) % check_conv_interval == 0 : # Start checking after min iterations
            if check_convergence(game, t, global_policy_stable_steps, q_value_deltas_history):
                print(f"\n🎉 Converged after {iterations_done / 1e6:.2f}M iterations!")
                max_q_change_at_convergence = max_q_change_at_step
                break
    else: # Loop finished without breaking (i.e., tmax reached)
        print(f"\n🏁 Simulation finished after {iterations_done / 1e6:.2f}M iterations (max iterations reached).")
        if q_value_deltas_history: max_q_change_at_convergence = q_value_deltas_history[-1]

    # Final summary prints
    final_alpha = game.alpha_initial  # Use constant alpha
    final_epsilon = np.exp(-game.beta_exploration * (iterations_done-1))
    print(f"Final effective alpha_t={final_alpha:.3e}, epsilon_t={final_epsilon:.3e}")
    if not np.isnan(max_q_change_at_convergence):
        print(f"Final max Q-value change: {max_q_change_at_convergence:.3e}")
    else:
        print("Final max Q-value change: N/A")
    
    return {
        "Q_tables": game.Q, # Note: Q_tables can be very large
        "iterations_completed": iterations_done,
        "q_value_deltas_history": q_value_deltas_history, 
        "global_policy_stable_steps_at_end": global_policy_stable_steps, # Return the final count
        "max_q_change_at_end": max_q_change_at_convergence,
        "final_q_state_tuple": current_q_state_tuple,
        "price_history": price_history,
        "profit_history": profit_history,
        "avg_q_value_history": avg_q_value_history,
        "history_log_interval": history_interval 
    }