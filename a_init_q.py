import numpy as np
from itertools import product, combinations  # product is used, combinations is not but kept for compatibility
from typing import List, Dict, Tuple, Any, Literal
from main.equilibrium import calculate_monopoly_prices

AgentQualityLiteral = Literal['H', 'L']
AgentPositionLiteral = Literal['in', 'out']

class ModelFixedBB:
    def __init__(self, **kwargs: Any):
        # --- Core Model Parameters with Defaults ---
        self.N: int = kwargs.get('N', 3)  # Number of agents
        self.K: int = kwargs.get('K', 1)  # Number of high-quality agents
        self.m: int = kwargs.get('m', 2)  # Number of buy box slots
        self.c: float = kwargs.get('c', 15.0)  # Cost for high-quality agents
        self.q_H: float = kwargs.get('q_H', 100.0)  # Value for high-quality agents
        self.q_L: float = kwargs.get('q_L', 80.0)   # Value for low-quality agents
        self.theta: float = kwargs.get('theta', 5.0)  # Logit parameter
        self.Delta: float = kwargs.get('Delta', 0.7)  # Buy box advantage parameter

        # --- Control Flags and Overrides ---
        self.calculator_mode: bool = kwargs.get('calculator_mode', False)
        self.agent_types_override: List[Dict[str, Any]] | None = kwargs.get('agent_types_override', None)
        self.positions_override: Tuple[List[int], Dict[int, AgentPositionLiteral]] | None = kwargs.get('positions_override', None)

        # --- Q-Learning Parameters ---
        self.alpha_initial: float = kwargs.get('alpha_initial', 0.5)  # Initial learning rate
        self.alpha_decay_rate: float = kwargs.get('alpha_decay_rate', 1e-7)  # Learning rate decay
        self.alpha_min: float = kwargs.get('alpha_min', 0.01)  # Minimum learning rate
        self.beta_exploration: float = kwargs.get('beta_exploration', 5e-7)  # Exploration parameter
        self.delta_rl: float = kwargs.get('delta_rl', 0.95)  # Discount factor

        # --- Simulation & Discretization Parameters ---
        self.grid_size: int = kwargs.get('grid_size', 5)  # Number of price points per agent
        self.tstable: int = kwargs.get('tstable', 1_000_000)  # Stabilization period
        self.tmax: int = kwargs.get('tmax', 10_000_000)  # Maximum simulation time

        # --- Random Seed for Reproducibility ---
        self.seed: int | None = kwargs.get('seed', None)
        if self.seed is not None:
            np.random.seed(self.seed)

        # --- Parameter Validation ---
        if self.N < 2:
            raise ValueError("N must be >= 2.")
        if not (1 <= self.K < self.N):
            raise ValueError(f"K ({self.K}) must be >= 1 and < N ({self.N}).")
        if not (1 <= self.m <= self.N):
            raise ValueError(f"m ({self.m}) must be >= 1 and <= N ({self.N}).")

        # Validate price bounds for high and low quality agents
        if self.K > 0:
            if not (self.c < self.q_H):
                raise ValueError(f"Cost c ({self.c}) must be less than high quality value q_H ({self.q_H})")
            if not (self.q_H - self.c > self.q_L):
                raise ValueError(f"Net value of high quality (q_H - c = {self.q_H - self.c:.2f}) must be greater than low quality value q_L ({self.q_L:.2f})")

        # Additional validations for learning mode only
        if not self.calculator_mode:
            if not (0 < self.alpha_initial <= 1 and 0 < self.alpha_min <= self.alpha_initial):
                raise ValueError(f"Invalid alpha parameters: initial={self.alpha_initial}, min={self.alpha_min}")
            if self.beta_exploration <= 0:
                raise ValueError("beta_exploration must be positive.")
            if not (0 < self.delta_rl < 1):
                raise ValueError("RL discount factor delta_rl must be in (0,1).")

        # --- Initialize Model Components ---
        # Agent types: override or default
        if self.calculator_mode and self.agent_types_override is not None:
            self.agent_types = self.agent_types_override
        else:
            self.agent_types = self._init_agent_types()

        # Buy box positions: override or random assignment
        if self.calculator_mode and self.positions_override is not None:
            self.fixed_buy_box_agent_ids, self.agent_positions = self.positions_override
        else:
            self.fixed_buy_box_agent_ids, self.agent_positions = self._determine_fixed_buy_box_and_positions()

        # Price bounds for each agent
        self.p_bounds_per_agent: Dict[int, Tuple[float, float]] = self._compute_price_bounds_for_agents()

        # --- Simulation-Specific Components ---
        if not self.calculator_mode:
            print(f"🚀 [ModelFixedBB SIM MODE] N={self.N}, K={self.K}, m={self.m}")
            print(f"📦 Fixed Buy Box Agent IDs: {self.fixed_buy_box_agent_ids}")

            # Action spaces for each agent
            self.A_n: Dict[int, np.ndarray] = self._init_individual_actions()
            # State space dimensions and initial state
            self.sdim, self.s0 = self._init_state_space()

            state_space_configs = np.prod(self.sdim, dtype=np.int64) if self.sdim else 0
            q_table_entries_per_agent = state_space_configs * self.grid_size
            print(f"🧩 State space definitions (grid_size={self.grid_size}, sdim={self.sdim}):")
            print(f"  State configurations = {state_space_configs}, Entries per Q-table = {q_table_entries_per_agent}")

            print("💰 Initializing Profit Array (PI)...")
            self.PI: np.ndarray = self._init_profit_array()
            print(f"  Profit Array (PI) shape {self.PI.shape}.")

            print("🧠 Initializing Q-tables...")
            self.Q: Dict[int, np.ndarray] = self._init_q_tables()
            if self.N > 0 and self.agent_types and 0 in self.Q and self.Q[0].size > 0:
                print(f"  Q-tables initialized. Shape (e.g., agent 0): {self.Q[0].shape}")
            else:
                print(f"  Q-tables appear empty or not fully initialized (N={self.N}, agent_types: {bool(self.agent_types)}, Q[0] valid: {0 in self.Q and self.Q[0].size > 0}).")
            print("-" * 30)
        else:
            print(f"🛠️ [ModelFixedBB CALC MODE] N={self.N}, K={self.K}, m={self.m}")
            # Minimal attributes for calculator mode
            self.A_n, self.sdim, self.s0, self.PI, self.Q = {}, tuple(), np.array([]), np.array([]), {}

    def _compute_price_bounds_for_agents(self) -> Dict[int, Tuple[float, float]]:
        """
        Compute price bounds for each agent based on their type and position.
        Returns a dictionary mapping agent IDs to (min_price, max_price) tuples.
        """
        price_bounds: Dict[int, Tuple[float, float]] = {}
        
        # Calculate monopoly prices for each type and position
        monopoly_prices = calculate_monopoly_prices(self.theta, self.q_H, self.q_L, self.Delta, self.c, self.N, self.K, self.m)
        p_H_in_monopoly = monopoly_prices['tilde_P_M_H']
        p_H_out_monopoly = monopoly_prices['hat_P_M_H']
        p_L_in_monopoly = monopoly_prices['tilde_P_M_L']
        p_L_out_monopoly = monopoly_prices['hat_P_M_L']

        print(f"Monopoly prices: H_in={p_H_in_monopoly}, H_out={p_H_out_monopoly}, L_in={p_L_in_monopoly}, L_out={p_L_out_monopoly}")
        
        for agent in self.agent_types:
            agent_id = agent['id']
            quality_type = agent['quality_type']
            cost = agent['cost']
            position = self.agent_positions[agent_id]

            # Set price bounds based on quality type and position
            if quality_type == 'H':
                min_price = cost
                if position == 'in':
                    max_price = 1.1 * p_H_in_monopoly
                else:  # 'out'
                    max_price = 1.1 * p_H_out_monopoly
            else:  # 'L'
                min_price = 0.0
                if position == 'in':
                    max_price = 1.1 * p_L_in_monopoly
                else:  # 'out'
                    max_price = 1.1 * p_L_out_monopoly

            price_bounds[agent_id] = (min_price, max_price)

            price_bounds[agent_id] = (min_price, max_price)
        return price_bounds

    def _init_agent_types(self) -> List[Dict[str, Any]]:
        """
        Initialize agent types: first K agents are high-quality, rest are low-quality.
        """
        agents: List[Dict[str, Any]] = []
        for i in range(self.N):
            if i < self.K:
                agents.append({'id': i, 'quality_type': 'H', 'cost': self.c, 'quality_value': self.q_H})
            else:
                agents.append({'id': i, 'quality_type': 'L', 'cost': 0.0, 'quality_value': self.q_L})
        return agents

    def _determine_fixed_buy_box_and_positions(self) -> Tuple[List[int], Dict[int, AgentPositionLiteral]]:
        """
        Randomly assign m agents to fixed buy box positions.
        """
        fixed_bb_ids: List[int] = []
        high_q_agent_ids: List[int] = [a['id'] for a in self.agent_types if a['quality_type'] == 'H']
        low_q_agent_ids: List[int] = [a['id'] for a in self.agent_types if a['quality_type'] == 'L']

        if self.m == 0:
            pass  # No agents in buy box
        elif self.m < self.K:
            # Select subset of high-quality agents
            if len(high_q_agent_ids) < self.m:
                if not self.calculator_mode:
                    print(f"⚠️ Warning: Requested m ({self.m}) HQs, but only {len(high_q_agent_ids)} available. Taking all HQs.")
                fixed_bb_ids = high_q_agent_ids[:]
            elif high_q_agent_ids:
                chosen_hq_indices = np.random.choice(len(high_q_agent_ids), size=self.m, replace=False)
                fixed_bb_ids = [high_q_agent_ids[i] for i in chosen_hq_indices]
        else:
            # Include all high-quality agents and some low-quality agents
            fixed_bb_ids.extend(high_q_agent_ids)
            num_low_q_needed = self.m - self.K
            if num_low_q_needed > 0:
                if len(low_q_agent_ids) <= num_low_q_needed:
                    fixed_bb_ids.extend(low_q_agent_ids)
                elif low_q_agent_ids:
                    chosen_lq_indices = np.random.choice(len(low_q_agent_ids), size=num_low_q_needed, replace=False)
                    fixed_bb_ids.extend([low_q_agent_ids[i] for i in chosen_lq_indices])

        fixed_bb_ids = sorted(set(fixed_bb_ids))

        # Assign positions based on buy box membership
        positions: Dict[int, AgentPositionLiteral] = {}
        for agent in self.agent_types:
            positions[agent['id']] = 'in' if agent['id'] in fixed_bb_ids else 'out'

        return fixed_bb_ids, positions

    def _get_stabilized_exp_utilities_and_max_val(self, prices: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Compute numerically stable exp((quality_value - price) / theta) for each agent.
        Returns the stabilized exponential utilities and the max utility value used for stabilization.
        """
        qualities = np.array([a['quality_value'] for a in self.agent_types])
        utilities_over_theta = (qualities - prices) / self.theta

        max_utility_val_ot = 0.0
        if utilities_over_theta.size > 0:
            current_max = np.max(utilities_over_theta)
            if not np.isinf(current_max):
                max_utility_val_ot = current_max
            elif current_max == -np.inf:
                max_utility_val_ot = -np.inf
            # If current_max is +inf, keep as 0.0

        # Stabilize utilities
        if np.isinf(max_utility_val_ot) and max_utility_val_ot > 0:
            stable_utilities = utilities_over_theta
        elif max_utility_val_ot == -np.inf:
            stable_utilities = np.full_like(utilities_over_theta, -np.inf)
        elif utilities_over_theta.size > 0:
            stable_utilities = utilities_over_theta - max_utility_val_ot
        else:
            stable_utilities = np.array([])

        exp_utilities = np.exp(stable_utilities)  # np.exp(-inf) = 0.0
        return exp_utilities, max_utility_val_ot

    def get_utility_components_for_cs(self, prices: np.ndarray) -> Tuple[float, float]:
        exp_utilities_stabilized, max_utility_val_ot = self._get_stabilized_exp_utilities_and_max_val(prices)
        E_T_stabilized: float = 0.0

        # === CORRECTED LOGIC START ===
        if self.m == self.N:
            # When all sellers are in the box, there is no Delta advantage. Weight is 1/N.
            E_T_stabilized = (1 / self.N) * np.sum(exp_utilities_stabilized)
        else:
            # === CORRECTED LOGIC END ===
            sellers_in_bb_mask = np.zeros(self.N, dtype=bool)
            if self.fixed_buy_box_agent_ids:
                valid_ids = [aid for aid in self.fixed_buy_box_agent_ids if 0 <= aid < self.N]
                if valid_ids:
                    sellers_in_bb_mask[valid_ids] = True

            num_agents_in_bb_mask = np.sum(sellers_in_bb_mask)
            if num_agents_in_bb_mask > 0:
                E_T_stabilized += (self.Delta / self.m) * np.sum(exp_utilities_stabilized[sellers_in_bb_mask])

            num_agents_out_bb_mask = self.N - num_agents_in_bb_mask
            slots_outside_bb = self.N - self.m
            if num_agents_out_bb_mask > 0 and slots_outside_bb > 0:
                E_T_stabilized += ((1 - self.Delta) / slots_outside_bb) * np.sum(exp_utilities_stabilized[~sellers_in_bb_mask])

        return E_T_stabilized, max_utility_val_ot

    def demand(self, prices: np.ndarray) -> np.ndarray:
        exp_utilities_stabilized, _ = self._get_stabilized_exp_utilities_and_max_val(prices)
        E_T: float = 0.0
        demands_calc: np.ndarray = np.zeros(self.N)

        # === CORRECTED LOGIC START ===
        if self.m == self.N:
            E_T = (1 / self.N) * np.sum(exp_utilities_stabilized)
            if E_T > 1e-100:
                demands_calc = ((1 / self.N) * exp_utilities_stabilized) / E_T
        else:
            # === CORRECTED LOGIC END ===
            sellers_in_bb_mask = np.zeros(self.N, dtype=bool)
            if self.fixed_buy_box_agent_ids:
                valid_ids = [aid for aid in self.fixed_buy_box_agent_ids if 0 <= aid < self.N]
                if valid_ids:
                    sellers_in_bb_mask[valid_ids] = True

            num_agents_in_bb_mask = np.sum(sellers_in_bb_mask)
            num_agents_out_bb_mask = self.N - num_agents_in_bb_mask
            slots_outside_bb = self.N - self.m

            if num_agents_in_bb_mask > 0:
                E_T += (self.Delta / self.m) * np.sum(exp_utilities_stabilized[sellers_in_bb_mask])
            if num_agents_out_bb_mask > 0 and slots_outside_bb > 0:
                E_T += ((1 - self.Delta) / slots_outside_bb) * np.sum(exp_utilities_stabilized[~sellers_in_bb_mask])

            if E_T < 1e-100:
                return demands_calc

            if num_agents_in_bb_mask > 0:
                demands_calc[sellers_in_bb_mask] = ((self.Delta / self.m) * exp_utilities_stabilized[sellers_in_bb_mask]) / E_T
            if num_agents_out_bb_mask > 0 and slots_outside_bb > 0:
                demands_calc[~sellers_in_bb_mask] = (((1 - self.Delta) / slots_outside_bb) * exp_utilities_stabilized[~sellers_in_bb_mask]) / E_T

        sum_demands = np.sum(demands_calc)
        if sum_demands > 1e-9 and not np.isclose(sum_demands, 1.0, atol=1e-8):
            demands_calc /= sum_demands
        return np.clip(demands_calc, 0, 1)

    def compute_profits(self, prices: np.ndarray) -> np.ndarray:
        """
        Compute profits for each agent given prices: (price - cost) * demand.
        """
        demands = self.demand(prices)
        costs = np.array([a['cost'] for a in self.agent_types])
        profits = (prices - costs) * demands
        return profits

    def _init_individual_actions(self) -> Dict[int, np.ndarray]:
        """
        Initialize discrete action spaces (price grids) for each agent.
        """
        A_n_actions: Dict[int, np.ndarray] = {}
        for agent_info in self.agent_types:
            agent_id = agent_info['id']
            p_min, p_max = self.p_bounds_per_agent[agent_id]
            if self.grid_size <= 1:
                A_n_actions[agent_id] = np.array([(p_min + p_max) / 2.0])
            else:
                A_n_actions[agent_id] = np.linspace(p_min, p_max, self.grid_size)
        return A_n_actions

    def _init_state_space(self) -> Tuple[Tuple[int, ...], np.ndarray]:
        """
        Initialize state space dimensions and random initial state.
        """
        sdim_q_table_state_part: Tuple[int, ...] = tuple([self.grid_size] * self.N)
        high_rand_bound = max(1, self.grid_size)
        initial_price_action_indices = np.random.randint(0, high_rand_bound, size=self.N)
        s0_indices: np.ndarray = initial_price_action_indices.astype(int)
        return sdim_q_table_state_part, s0_indices

    def _init_profit_array(self) -> np.ndarray:
        """
        Pre-compute profit array for all possible state-action combinations.
        Enables fast profit lookups during Q-learning.
        """
        profit_array_shape = self.sdim + (self.N,)
        PI_calc: np.ndarray = np.full(profit_array_shape, np.nan, dtype=np.float64)

        total_configs_for_pi = np.prod(self.sdim, dtype=np.int64) if self.sdim else 0
        if not self.sdim:
            return PI_calc

        print(f"  [PI Init] Iterating through {total_configs_for_pi} configurations...")
        for i, s_price_action_indices_tuple in enumerate(product(*[range(dim_size) for dim_size in self.sdim])):
            current_prices = np.zeros(self.N)
            actions_valid = True

            # Convert action indices to actual prices
            for agent_idx_loop in range(self.N):
                action_idx = s_price_action_indices_tuple[agent_idx_loop]
                agent_id = self.agent_types[agent_idx_loop]['id']  # Assumes ordered agent IDs 0..N-1

                if agent_id not in self.A_n or action_idx >= len(self.A_n[agent_id]):
                    actions_valid = False
                    break
                current_prices[agent_idx_loop] = self.A_n[agent_id][action_idx]

            if not actions_valid:
                PI_calc[s_price_action_indices_tuple] = np.full(self.N, -np.inf)
                continue

            try:
                PI_calc[s_price_action_indices_tuple] = self.compute_profits(current_prices)
            except Exception as e:
                print(f"ERROR PI state {s_price_action_indices_tuple}, prices {current_prices}: {e}")
                PI_calc[s_price_action_indices_tuple] = np.full(self.N, -np.inf)

            # Progress reporting for large state spaces
            if total_configs_for_pi > 1000 and (i + 1) % (max(1, total_configs_for_pi // 20)) == 0:
                print(f"    PI init: {(i + 1)}/{total_configs_for_pi} states processed.")

        if np.any(np.isnan(PI_calc)) and total_configs_for_pi > 0:
            print(f"⚠️ Warning: PI array has {np.sum(np.isnan(PI_calc))} NaN values.")
        return PI_calc

    def _init_q_tables(self) -> Dict[int, np.ndarray]:
        """
        Initializes Q-tables according to the specified procedure.

        The procedure is as follows:
        Fixing an agent and state, for each action available to that agent we derive
        the within-period payoff that would be expected if all other agents uniformly randomized
        their actions. We then divide this value by 1-δ so that the Q-matrix indeed contains an
        initial estimate of the total future payoffs of taking different actions today.
        """
        q_table_shape = self.sdim + (self.grid_size,)
        Q_tables_calc: Dict[int, np.ndarray] = {}
        all_agent_ids = [agent['id'] for agent in self.agent_types]

        # --- WARNING: This is computationally intensive ---
        # Calculation complexity is roughly: N * grid_size * (grid_size^(N-1))
        # This can be very slow for N > 3 or grid_size > 10.
        total_calcs = self.N * self.grid_size * (self.grid_size ** (self.N - 1))
        print(f"  [Q-Init] Calculating state-action specific initial Q-values...")
        print(f"  [Q-Init] This requires iterating through ~{total_calcs:.2e} scenarios. This may take some time.")

        # 1. For each agent 'n'
        for agent_info in self.agent_types:
            agent_id_n = agent_info['id']
            # The full Q-table for this agent, to be filled
            q_table_for_agent_n = np.zeros(q_table_shape, dtype=np.float64)

            # Get the actions (prices) for our agent 'n'
            actions_for_agent_n = self.A_n[agent_id_n]

            # Get the other agents and their action spaces
            other_agent_ids = [aid for aid in all_agent_ids if aid != agent_id_n]
            other_agents_actions_list = [self.A_n[aid] for aid in other_agent_ids]

            # 2. For each action 'a_n' available to agent 'n'
            for action_idx_n, price_n in enumerate(actions_for_agent_n):
                
                # 3. Derive the expected payoff if all other agents randomize
                profits_for_this_action = []
                
                # Use itertools.product to get every combination of other agents' actions
                # This simulates the "uniform randomization" by iterating through all possibilities
                for other_prices_tuple in product(*other_agents_actions_list):
                    
                    # Reconstruct the full price vector for this specific scenario
                    current_prices = np.zeros(self.N)
                    current_prices[agent_id_n] = price_n # Set price for our agent
                    
                    # Set prices for all other agents
                    for i, other_agent_id in enumerate(other_agent_ids):
                        current_prices[other_agent_id] = other_prices_tuple[i]
                        
                    # Calculate the profits for this joint action
                    all_profits = self.compute_profits(current_prices)
                    profit_for_agent_n = all_profits[agent_id_n]
                    profits_for_this_action.append(profit_for_agent_n)

                # The expected payoff is the mean of the profits calculated
                expected_payoff = np.mean(profits_for_this_action)

                # 4. Divide by (1 - δ) to get the initial Q-value
                if 0 < self.delta_rl < 1 and abs(1 - self.delta_rl) > 1e-9:
                    initial_q_val_for_action = expected_payoff / (1 - self.delta_rl)
                else:
                    initial_q_val_for_action = expected_payoff # No discounting

                # Since the expectation doesn't depend on the state 's', this Q-value
                # is the same for this action across all states.
                # We can fill the entire slice of the Q-table corresponding to this action.
                # The '...' means "all state dimensions".
                q_table_for_agent_n[..., action_idx_n] = initial_q_val_for_action
            
            # Store the fully calculated Q-table for the agent
            Q_tables_calc[agent_id_n] = q_table_for_agent_n
            
            print(f"  [Q-Init] Done initializing Q-table for Agent {agent_id_n}.")

        return Q_tables_calc

    def get_current_alpha(self, t: int) -> float:
        """
        Compute current learning rate with exponential decay.
        """
        alpha = self.alpha_initial * np.exp(-self.alpha_decay_rate * t)
        return max(alpha, self.alpha_min)

    def get_period_profits_and_next_state(self, joint_action_indices: np.ndarray) -> Dict[str, Any]:
        """
        Retrieve period profits and next state for given joint action indices.
        Used during Q-learning to get rewards and state transitions.
        """
        if len(joint_action_indices) != self.N:
            raise ValueError(f"joint_action_indices length mismatch N ({self.N}).")
        if not hasattr(self, 'PI') or self.PI.size == 0:
            raise AttributeError("Profit array PI is not initialized or is empty.")

        pi_lookup_tuple = tuple(joint_action_indices.astype(int))

        # Validate indices are within bounds
        for idx_val, dim_size in zip(pi_lookup_tuple, self.PI.shape[:-1]):
            if not (0 <= idx_val < dim_size):
                raise IndexError(f"PI lookup index {idx_val} for dim size {dim_size} is out of bounds.")

        # Retrieve profits and handle NaN values
        period_profits = self.PI[pi_lookup_tuple]
        if np.any(np.isnan(period_profits)):
            period_profits = np.nan_to_num(period_profits, nan=-1.0e9)

        return {
            'period_profits': period_profits,
            'next_q_state_definition': joint_action_indices.astype(int)
        }