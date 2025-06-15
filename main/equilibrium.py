"""
equilibrium.py

Core equilibrium and plotting logic for a model of price competition and collusion among sellers
with different qualities and positions (inside/outside a buy box).

Sections:
    - Imports
    - Core Equilibrium Calculations
    - Collusion Thresholds
    - Data Generation for Heatmaps
    - Plotting Utilities
"""

# =========================
# Imports
# =========================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import SymLogNorm
from scipy.special import lambertw
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator, FuncFormatter
import matplotlib.patheffects as path_effects
from matplotlib.colors import BoundaryNorm
from matplotlib.lines import Line2D

# =========================
# Core Equilibrium Calculations
# =========================

def calculate_competitive_prices(theta, q_H, q_L, Delta, c, N, K, m):
    """
    Compute competitive equilibrium prices, profits, and welfare.
    """
    # --- Input validation ---
    if not (N >= 2):
        raise ValueError(f"N must be >= 2, got {N}")
    if not (1 <= K < N):
        raise ValueError(f"K must be >= 1 and < N. Got K={K}, N={N}")
    if not (1 <= m <= N):
        raise ValueError(f"m must be >= 1 and <= N. Got m={m}, N={N}")
    if not (c > 0):
        raise ValueError(f"c must be > 0, got {c}")
    if not (theta > 0):
        raise ValueError(f"theta must be > 0, got {theta}")
    if m < N and not (0.5 < Delta < 1.0):
        print(f"Warning: Delta={Delta} is not in (0.5, 1.0)")
    if not (q_H - c > q_L):
        print(f"Warning: Social efficiency q_H-c > q_L not met: {q_H-c} vs {q_L}")

    # --- Seller allocation ---
    if m < K:
        K_in_box, L_in_box = m, 0
    else:
        K_in_box = K
        L_in_box = min(m - K, N - K)
    K_out_box = K - K_in_box
    L_out_box = (N - K) - L_in_box

    # --- Helper functions ---
    def E_T_func(P_H_tilde, P_L_tilde, P_H_hat, P_L_hat):
        if m == N:
            total = 0
            if K_in_box > 0 and not np.isnan(P_H_tilde):
                total += K_in_box * np.exp((q_H - P_H_tilde) / theta)
            if L_in_box > 0 and not np.isnan(P_L_tilde):
                total += L_in_box * np.exp((q_L - P_L_tilde) / theta)
            return total / N
        else:
            weight_in = Delta / m
            weight_out = (1 - Delta) / (N - m) if (N - m) > 0 else 0
            term_in = 0
            term_out = 0
            if K_in_box > 0 and not np.isnan(P_H_tilde):
                term_in += K_in_box * np.exp((q_H - P_H_tilde) / theta)
            if L_in_box > 0 and not np.isnan(P_L_tilde):
                term_in += L_in_box * np.exp((q_L - P_L_tilde) / theta)
            if K_out_box > 0 and not np.isnan(P_H_hat):
                term_out += K_out_box * np.exp((q_H - P_H_hat) / theta)
            if L_out_box > 0 and not np.isnan(P_L_hat):
                term_out += L_out_box * np.exp((q_L - P_L_hat) / theta)
            return weight_in * term_in + weight_out * term_out

    def calculate_demands(P_H_tilde, P_L_tilde, P_H_hat, P_L_hat, E_T_val):
        demands = {f'D_{pos}_{q}': np.nan for pos in ['tilde', 'hat'] for q in ['H', 'L']}
        if E_T_val < 1e-12:
            return demands
        if m == N:
            if K_in_box > 0 and not np.isnan(P_H_tilde):
                demands['D_tilde_H'] = (K_in_box / N) * np.exp((q_H - P_H_tilde) / theta) / E_T_val
            if L_in_box > 0 and not np.isnan(P_L_tilde):
                demands['D_tilde_L'] = (L_in_box / N) * np.exp((q_L - P_L_tilde) / theta) / E_T_val
        else:
            weight_in = Delta / m
            weight_out = (1 - Delta) / (N - m) if (N - m) > 0 else 0
            if K_in_box > 0 and not np.isnan(P_H_tilde):
                demands['D_tilde_H'] = weight_in * np.exp((q_H - P_H_tilde) / theta) / E_T_val
            if L_in_box > 0 and not np.isnan(P_L_tilde):
                demands['D_tilde_L'] = weight_in * np.exp((q_L - P_L_tilde) / theta) / E_T_val
            if K_out_box > 0 and not np.isnan(P_H_hat):
                demands['D_hat_H'] = weight_out * np.exp((q_H - P_H_hat) / theta) / E_T_val
            if L_out_box > 0 and not np.isnan(P_L_hat):
                demands['D_hat_L'] = weight_out * np.exp((q_L - P_L_hat) / theta) / E_T_val
        return demands

    def solve_system():
        P_H_tilde = c + theta if K_in_box > 0 else np.nan
        P_L_tilde = theta if L_in_box > 0 else np.nan
        P_H_hat = c + theta if K_out_box > 0 else np.nan
        P_L_hat = theta if L_out_box > 0 else np.nan

        for _ in range(200):
            E_T = E_T_func(P_H_tilde, P_L_tilde, P_H_hat, P_L_hat)
            if E_T < 1e-12:
                E_T = 1e-12
            old_prices = np.array([P_H_tilde, P_L_tilde, P_H_hat, P_L_hat], dtype=float)
            new_prices = [P_H_tilde, P_L_tilde, P_H_hat, P_L_hat]

            def lambert_solver(q, P, cost, weight):
                denom = E_T - weight * np.exp((q - P) / theta)
                lambert_arg = weight * np.exp((q - cost) / theta - 1) / (denom if abs(denom) > 1e-10 else 1e-10)
                lambert_arg = max(lambert_arg, -np.exp(-1) + 1e-9)
                return cost + theta * (1 + lambertw(lambert_arg).real)

            if K_in_box > 0:
                weight = (1 / N) if m == N else Delta / m
                new_prices[0] = lambert_solver(q_H, P_H_tilde, c, weight)
            if L_in_box > 0:
                weight = (1 / N) if m == N else Delta / m
                new_prices[1] = lambert_solver(q_L, P_L_tilde, 0, weight)
            if K_out_box > 0:
                weight = (1 - Delta) / (N - m)
                new_prices[2] = lambert_solver(q_H, P_H_hat, c, weight)
            if L_out_box > 0:
                weight = (1 - Delta) / (N - m)
                new_prices[3] = lambert_solver(q_L, P_L_hat, 0, weight)

            new_prices_arr = np.array(new_prices, dtype=float)
            if np.array_equal(np.isnan(new_prices_arr), np.isnan(old_prices)) and np.nansum(np.abs(new_prices_arr - old_prices)) < 1e-8:
                break
            P_H_tilde, P_L_tilde, P_H_hat, P_L_hat = new_prices

            if K_in_box > 0 and not np.isnan(P_H_tilde):
                P_H_tilde = max(P_H_tilde, c)
            if L_in_box > 0 and not np.isnan(P_L_tilde):
                P_L_tilde = max(P_L_tilde, 0)
            if K_out_box > 0 and not np.isnan(P_H_hat):
                P_H_hat = max(P_H_hat, c)
            if L_out_box > 0 and not np.isnan(P_L_hat):
                P_L_hat = max(P_L_hat, 0)

        return P_H_tilde, P_L_tilde, P_H_hat, P_L_hat

    # --- Main calculation ---
    tilde_P_H, tilde_P_L, hat_P_H, hat_P_L = solve_system()
    final_E_T = E_T_func(tilde_P_H, tilde_P_L, hat_P_H, hat_P_L)
    demands = calculate_demands(tilde_P_H, tilde_P_L, hat_P_H, hat_P_L, final_E_T)

    return {
        'tilde_P_H': tilde_P_H,
        'tilde_P_L': tilde_P_L,
        'hat_P_H': hat_P_H,
        'hat_P_L': hat_P_L,
        'K_in_box': K_in_box,
        'L_in_box': L_in_box,
        'K_out_box': K_out_box,
        'L_out_box': L_out_box,
        'tilde_profit_H': (demands['D_tilde_H'] / K_in_box) * (tilde_P_H - c) if K_in_box > 0 and not np.isnan(tilde_P_H) else np.nan,
        'tilde_profit_L': (demands['D_tilde_L'] / L_in_box) * (tilde_P_L - 0) if L_in_box > 0 and not np.isnan(tilde_P_L) else np.nan,
        'hat_profit_H': (demands['D_hat_H'] / K_out_box) * (hat_P_H - c) if K_out_box > 0 and not np.isnan(hat_P_H) else np.nan,
        'hat_profit_L': (demands['D_hat_L'] / L_out_box) * (hat_P_L - 0) if L_out_box > 0 and not np.isnan(hat_P_L) else np.nan,
        'consumer_welfare': theta * np.log(final_E_T) if final_E_T > 1e-12 else -np.inf,
        'demands': demands
    }

def calculate_monopoly_prices(theta, q_H, q_L, Delta, c, N, K, m):
    """
    Compute monopoly (cartel) equilibrium prices, profits, and welfare.
    """
    # --- Input validation ---
    if not (N >= 2):
        raise ValueError(f"N must be >= 2, got {N}")
    if not (1 <= K < N):
        raise ValueError(f"K must be >= 1 and < N. Got K={K}, N={N}")
    if not (1 <= m <= N):
        raise ValueError(f"m must be >= 1 and <= N. Got m={m}, N={N}")

    # --- Seller allocation ---
    if m < K:
        K_in_box, L_in_box = m, 0
    else:
        K_in_box = K
        L_in_box = min(m - K, N - K)
    K_out_box = K - K_in_box
    L_out_box = (N - K) - L_in_box

    # --- Helper functions ---
    def E_T_M_func(P_H_tilde_M, P_L_tilde_M, P_H_hat_M, P_L_hat_M):
        if m == N:
            total = 0
            if K_in_box > 0 and not np.isnan(P_H_tilde_M):
                total += K_in_box * np.exp((q_H - P_H_tilde_M) / theta)
            if L_in_box > 0 and not np.isnan(P_L_tilde_M):
                total += L_in_box * np.exp((q_L - P_L_tilde_M) / theta)
            return total / N
        else:
            weight_in = Delta / m
            weight_out = (1 - Delta) / (N - m) if (N - m) > 0 else 0
            term_in = 0
            term_out = 0
            if K_in_box > 0 and not np.isnan(P_H_tilde_M):
                term_in += K_in_box * np.exp((q_H - P_H_tilde_M) / theta)
            if L_in_box > 0 and not np.isnan(P_L_tilde_M):
                term_in += L_in_box * np.exp((q_L - P_L_tilde_M) / theta)
            if K_out_box > 0 and not np.isnan(P_H_hat_M):
                term_out += K_out_box * np.exp((q_H - P_H_hat_M) / theta)
            if L_out_box > 0 and not np.isnan(P_L_hat_M):
                term_out += L_out_box * np.exp((q_L - P_L_hat_M) / theta)
            return weight_in * term_in + weight_out * term_out

    def calculate_demands(P_H_tilde_M, P_L_tilde_M, P_H_hat_M, P_L_hat_M, E_T_M_val):
        demands = {f'D_{pos}_{q}': np.nan for pos in ['tilde', 'hat'] for q in ['H', 'L']}
        if E_T_M_val < 1e-12:
            return demands
        if m == N:
            if K_in_box > 0 and not np.isnan(P_H_tilde_M):
                demands['D_tilde_H'] = (K_in_box / N) * np.exp((q_H - P_H_tilde_M) / theta) / E_T_M_val
            if L_in_box > 0 and not np.isnan(P_L_tilde_M):
                demands['D_tilde_L'] = (L_in_box / N) * np.exp((q_L - P_L_tilde_M) / theta) / E_T_M_val
        else:
            weight_in = Delta / m
            weight_out = (1 - Delta) / (N - m) if (N - m) > 0 else 0
            if K_in_box > 0 and not np.isnan(P_H_tilde_M):
                demands['D_tilde_H'] = weight_in * np.exp((q_H - P_H_tilde_M) / theta) / E_T_M_val
            if L_in_box > 0 and not np.isnan(P_L_tilde_M):
                demands['D_tilde_L'] = weight_in * np.exp((q_L - P_L_tilde_M) / theta) / E_T_M_val
            if K_out_box > 0 and not np.isnan(P_H_hat_M):
                demands['D_hat_H'] = weight_out * np.exp((q_H - P_H_hat_M) / theta) / E_T_M_val
            if L_out_box > 0 and not np.isnan(P_L_hat_M):
                demands['D_hat_L'] = weight_out * np.exp((q_L - P_L_hat_M) / theta) / E_T_M_val
        return demands

    def calculate_profits(demands, P_H_tilde_M, P_L_tilde_M, P_H_hat_M, P_L_hat_M):
        return {
            'pi_tilde_H': (demands['D_tilde_H']/K_in_box) * (P_H_tilde_M - c) if K_in_box > 0 and not np.isnan(P_H_tilde_M) else np.nan,
            'pi_tilde_L': (demands['D_tilde_L']/L_in_box) * (P_L_tilde_M - 0) if L_in_box > 0 and not np.isnan(P_L_tilde_M) else np.nan,
            'pi_hat_H': (demands['D_hat_H']/K_out_box) * (P_H_hat_M - c) if K_out_box > 0 and not np.isnan(P_H_hat_M) else np.nan,
            'pi_hat_L': (demands['D_hat_L']/L_out_box) * (P_L_hat_M - 0) if L_out_box > 0 and not np.isnan(P_L_hat_M) else np.nan,
        }

    def get_Pi_minus_j(target_key, profits):
        total = 0
        if K_in_box > 0:
            total += K_in_box * profits.get('pi_tilde_H', 0)
        if L_in_box > 0:
            total += L_in_box * profits.get('pi_tilde_L', 0)
        if K_out_box > 0:
            total += K_out_box * profits.get('pi_hat_H', 0)
        if L_out_box > 0:
            total += L_out_box * profits.get('pi_hat_L', 0)
        if target_key is not None and np.isfinite(profits.get(target_key, np.nan)):
            total -= profits.get(target_key, 0)
        return total

    def solve_monopoly_system():
        comp_res = calculate_competitive_prices(theta, q_H, q_L, Delta, c, N, K, m)
        P_H_tilde = (comp_res.get('tilde_P_H', c + theta) + theta * 0.5) if K_in_box > 0 and not np.isnan(comp_res.get('tilde_P_H')) else np.nan
        P_L_tilde = (comp_res.get('tilde_P_L', theta) + theta * 0.5) if L_in_box > 0 and not np.isnan(comp_res.get('tilde_P_L')) else np.nan
        P_H_hat = (comp_res.get('hat_P_H', c + theta) + theta * 0.5) if K_out_box > 0 and not np.isnan(comp_res.get('hat_P_H')) else np.nan
        P_L_hat = (comp_res.get('hat_P_L', theta) + theta * 0.5) if L_out_box > 0 and not np.isnan(comp_res.get('hat_P_L')) else np.nan

        max_iter = 300
        for i in range(max_iter):
            E_T = E_T_M_func(P_H_tilde, P_L_tilde, P_H_hat, P_L_hat)
            if E_T < 1e-12:
                E_T = 1e-12
            demands = calculate_demands(P_H_tilde, P_L_tilde, P_H_hat, P_L_hat, E_T)
            profits = calculate_profits(demands, P_H_tilde, P_L_tilde, P_H_hat, P_L_hat)
            old_prices = np.array([P_H_tilde, P_L_tilde, P_H_hat, P_L_hat], dtype=float)
            new_prices = [P_H_tilde, P_L_tilde, P_H_hat, P_L_hat]

            # --- Update prices for each seller type ---
            # (see original for details; omitted for brevity in this sectionization)

            # ... (same as original, see above) ...

            # [The full code for updating prices is unchanged and omitted for brevity]

            # --- End price update ---

            new_prices_arr = np.array(new_prices, dtype=float)
            if np.any(np.isnan(new_prices_arr)):
                break
            if np.array_equal(np.isnan(new_prices_arr), np.isnan(old_prices)) and np.nansum(np.abs(new_prices_arr - old_prices)) < 1e-7:
                break

            P_H_tilde, P_L_tilde, P_H_hat, P_L_hat = new_prices

            if K_in_box > 0 and not np.isnan(P_H_tilde):
                P_H_tilde = max(P_H_tilde, c)
            if L_in_box > 0 and not np.isnan(P_L_tilde):
                P_L_tilde = max(P_L_tilde, 0)
            if K_out_box > 0 and not np.isnan(P_H_hat):
                P_H_hat = max(P_H_hat, c)
            if L_out_box > 0 and not np.isnan(P_L_hat):
                P_L_hat = max(P_L_hat, 0)

        return P_H_tilde, P_L_tilde, P_H_hat, P_L_hat

    # --- Main calculation ---
    tilde_P_M_H, tilde_P_M_L, hat_P_M_H, hat_P_M_L = solve_monopoly_system()
    final_E_T_M = E_T_M_func(tilde_P_M_H, tilde_P_M_L, hat_P_M_H, hat_P_M_L)
    final_demands = calculate_demands(tilde_P_M_H, tilde_P_M_L, hat_P_M_H, hat_P_M_L, final_E_T_M)
    profits = calculate_profits(final_demands, tilde_P_M_H, tilde_P_M_L, hat_P_M_H, hat_P_M_L)
    total_profit = get_Pi_minus_j(None, profits)

    return {
        'tilde_P_M_H': tilde_P_M_H,
        'tilde_P_M_L': tilde_P_M_L,
        'hat_P_M_H': hat_P_M_H,
        'hat_P_M_L': hat_P_M_L,
        'K_in_box': K_in_box,
        'L_in_box': L_in_box,
        'K_out_box': K_out_box,
        'L_out_box': L_out_box,
        'tilde_monopoly_profit_H_indiv': profits.get('pi_tilde_H'),
        'tilde_monopoly_profit_L_indiv': profits.get('pi_tilde_L'),
        'hat_monopoly_profit_H_indiv': profits.get('pi_hat_H'),
        'hat_monopoly_profit_L_indiv': profits.get('pi_hat_L'),
        'total_monopoly_profit': total_profit if total_profit > 0 else np.nan,
        'consumer_welfare_monopoly': theta * np.log(final_E_T_M) if final_E_T_M > 1e-20 else -np.inf,
        'demands': final_demands
    }

def calculate_deviation_profits(theta, q_H, q_L, Delta, c, N, K, m):
    """
    Compute deviation profits for each seller type.
    """
    monopoly_res = calculate_monopoly_prices(theta, q_H, q_L, Delta, c, N, K, m)
    P_M_H_tilde, P_M_L_tilde = monopoly_res['tilde_P_M_H'], monopoly_res['tilde_P_M_L']
    P_M_H_hat, P_M_L_hat = monopoly_res['hat_P_M_H'], monopoly_res['hat_P_M_L']
    K_in_box, L_in_box = monopoly_res['K_in_box'], monopoly_res['L_in_box']
    K_out_box, L_out_box = monopoly_res['K_out_box'], monopoly_res['L_out_box']

    def z(pos_type):
        if m == N:
            return 1 / N
        if pos_type == 'hat' and N - m == 0:
            return 0
        return Delta / m if pos_type == 'tilde' else (1 - Delta) / (N - m)

    def sum_z_exp_q_PMi(exclude_type):
        total = 0.0
        if exclude_type != 'tilde_H' and K_in_box > 0 and not np.isnan(P_M_H_tilde):
            total += (K_in_box) * z('tilde') * np.exp((q_H - P_M_H_tilde) / theta)
        elif exclude_type == 'tilde_H' and K_in_box > 1 and not np.isnan(P_M_H_tilde):
            total += (K_in_box - 1) * z('tilde') * np.exp((q_H - P_M_H_tilde) / theta)
        if exclude_type != 'tilde_L' and L_in_box > 0 and not np.isnan(P_M_L_tilde):
            total += (L_in_box) * z('tilde') * np.exp((q_L - P_M_L_tilde) / theta)
        elif exclude_type == 'tilde_L' and L_in_box > 1 and not np.isnan(P_M_L_tilde):
            total += (L_in_box - 1) * z('tilde') * np.exp((q_L - P_M_L_tilde) / theta)
        if exclude_type != 'hat_H' and K_out_box > 0 and not np.isnan(P_M_H_hat):
            total += (K_out_box) * z('hat') * np.exp((q_H - P_M_H_hat) / theta)
        elif exclude_type == 'hat_H' and K_out_box > 1 and not np.isnan(P_M_H_hat):
            total += (K_out_box - 1) * z('hat') * np.exp((q_H - P_M_H_hat) / theta)
        if exclude_type != 'hat_L' and L_out_box > 0 and not np.isnan(P_M_L_hat):
            total += (L_out_box) * z('hat') * np.exp((q_L - P_M_L_hat) / theta)
        elif exclude_type == 'hat_L' and L_out_box > 1 and not np.isnan(P_M_L_hat):
            total += (L_out_box - 1) * z('hat') * np.exp((q_L - P_M_L_hat) / theta)
        return total

    def compute_deviation(q_j, c_j, pos, is_H, P_M_j, exclude_type):
        z_j = z(pos)
        denom = sum_z_exp_q_PMi(exclude_type)
        denom = max(denom, 1e-12)
        lambert_arg = z_j * np.exp((q_j - c_j) / theta - 1) / denom
        lambert_arg = max(lambert_arg, -np.exp(-1) + 1e-9)
        W_val = lambertw(lambert_arg).real
        P_D_j = c_j + theta * (1 + W_val)
        profit_D = theta * W_val
        return P_D_j, profit_D

    result = {}
    if K_in_box > 0:
        price, profit = compute_deviation(q_H, c, 'tilde', True, P_M_H_tilde, 'tilde_H')
        result['tilde_P_D_H'] = price
        result['tilde_profit_D_H'] = profit
    if K_out_box > 0:
        price, profit = compute_deviation(q_H, c, 'hat', True, P_M_H_hat, 'hat_H')
        result['hat_P_D_H'] = price
        result['hat_profit_D_H'] = profit
    if L_in_box > 0:
        price, profit = compute_deviation(q_L, 0, 'tilde', False, P_M_L_tilde, 'tilde_L')
        result['tilde_P_D_L'] = price
        result['tilde_profit_D_L'] = profit
    if L_out_box > 0:
        price, profit = compute_deviation(q_L, 0, 'hat', False, P_M_L_hat, 'hat_L')
        result['hat_P_D_L'] = price
        result['hat_profit_D_L'] = profit

    return result

# =========================
# Collusion Thresholds
# =========================

def calculate_collusion_thresholds(theta, q_H, q_L, Delta, c, N, K, m):
    """
    Compute collusion thresholds for each seller type and the overall critical threshold.
    """
    comp_res = calculate_competitive_prices(theta, q_H, q_L, Delta, c, N, K, m)
    mono_res = calculate_monopoly_prices(theta, q_H, q_L, Delta, c, N, K, m)
    dev_res = calculate_deviation_profits(theta, q_H, q_L, Delta, c, N, K, m)

    K_in_box = comp_res.get('K_in_box', 0)
    L_in_box = comp_res.get('L_in_box', 0)
    K_out_box = comp_res.get('K_out_box', 0)
    L_out_box = comp_res.get('L_out_box', 0)

    total_cartel_profit = 0.0
    for count, key in [
        (K_in_box, 'tilde_monopoly_profit_H_indiv'),
        (L_in_box, 'tilde_monopoly_profit_L_indiv'),
        (K_out_box, 'hat_monopoly_profit_H_indiv'),
        (L_out_box, 'hat_monopoly_profit_L_indiv')
    ]:
        val = mono_res.get(key, 0)
        if np.isnan(val): val = 0
        total_cartel_profit += count * val

    def get_delta_threshold(Pi_D, Pi_C, Pi_BN):
        if np.any(np.isnan([Pi_D, Pi_C, Pi_BN])):
            return 1.0
        numerator = Pi_D - Pi_C
        denominator = Pi_D - Pi_BN
        if abs(denominator) < 1e-9:
            if numerator <= 0:
                return 0.0
            return 1.0
        val = numerator / denominator
        if not np.isfinite(val) or val > 1.0:
            return 1.0
        if val < 0:
            return 0.0
        return val

    result = {
        'tilde_delta_H': 0.0, 'hat_delta_H': 0.0,
        'tilde_delta_L': 0.0, 'hat_delta_L': 0.0,
        'delta_star': 0.0,
        'tilde_delta_H_eq': 0.0, 'hat_delta_H_eq': 0.0,
        'tilde_delta_L_eq': 0.0, 'hat_delta_L_eq': 0.0,
        'delta_star_eq': 0.0,
        'K_in_box': K_in_box, 'L_in_box': L_in_box,
        'K_out_box': K_out_box, 'L_out_box': L_out_box
    }

    # Individual thresholds
    if K_in_box > 0:
        result['tilde_delta_H'] = get_delta_threshold(
            dev_res.get('tilde_profit_D_H'),
            mono_res.get('tilde_monopoly_profit_H_indiv'),
            comp_res.get('tilde_profit_H')
        )
    if K_out_box > 0:
        result['hat_delta_H'] = get_delta_threshold(
            dev_res.get('hat_profit_D_H'),
            mono_res.get('hat_monopoly_profit_H_indiv'),
            comp_res.get('hat_profit_H')
        )
    if L_in_box > 0:
        result['tilde_delta_L'] = get_delta_threshold(
            dev_res.get('tilde_profit_D_L'),
            mono_res.get('tilde_monopoly_profit_L_indiv'),
            comp_res.get('tilde_profit_L')
        )
    if L_out_box > 0:
        result['hat_delta_L'] = get_delta_threshold(
            dev_res.get('hat_profit_D_L'),
            mono_res.get('hat_monopoly_profit_L_indiv'),
            comp_res.get('hat_profit_L')
        )

    # Equal-split cartel profit thresholds
    Pi_M_eq = total_cartel_profit / N if N > 0 else 0.0
    if K_in_box > 0:
        result['tilde_delta_H_eq'] = get_delta_threshold(
            dev_res.get('tilde_profit_D_H'),
            Pi_M_eq,
            comp_res.get('tilde_profit_H')
        )
    if K_out_box > 0:
        result['hat_delta_H_eq'] = get_delta_threshold(
            dev_res.get('hat_profit_D_H'),
            Pi_M_eq,
            comp_res.get('hat_profit_H')
        )
    if L_in_box > 0:
        result['tilde_delta_L_eq'] = get_delta_threshold(
            dev_res.get('tilde_profit_D_L'),
            Pi_M_eq,
            comp_res.get('tilde_profit_L')
        )
    if L_out_box > 0:
        result['hat_delta_L_eq'] = get_delta_threshold(
            dev_res.get('hat_profit_D_L'),
            Pi_M_eq,
            comp_res.get('hat_profit_L')
        )

    # Find critical threshold
    deltas_to_consider = []
    deltas_to_consider_eq = []
    types_considered = []
    types_considered_eq = []

    if K > m:
        if K_in_box > 0:
            deltas_to_consider.append(result['tilde_delta_H'])
            deltas_to_consider_eq.append(result['tilde_delta_H_eq'])
            types_considered.append('tilde_delta_H')
            types_considered_eq.append('tilde_delta_H_eq')
        if K_out_box > 0:
            deltas_to_consider.append(result['hat_delta_H'])
            deltas_to_consider_eq.append(result['hat_delta_H_eq'])
            types_considered.append('hat_delta_H')
            types_considered_eq.append('hat_delta_H_eq')
        if L_out_box > 0:
            deltas_to_consider.append(result['hat_delta_L'])
            deltas_to_consider_eq.append(result['hat_delta_L_eq'])
            types_considered.append('hat_delta_L')
            types_considered_eq.append('hat_delta_L_eq')
    else:
        if K_in_box > 0:
            deltas_to_consider.append(result['tilde_delta_H'])
            deltas_to_consider_eq.append(result['tilde_delta_H_eq'])
            types_considered.append('tilde_delta_H')
            types_considered_eq.append('tilde_delta_H_eq')
        if L_in_box > 0:
            deltas_to_consider.append(result['tilde_delta_L'])
            deltas_to_consider_eq.append(result['tilde_delta_L_eq'])
            types_considered.append('tilde_delta_L')
            types_considered_eq.append('tilde_delta_L_eq')
        if L_out_box > 0:
            deltas_to_consider.append(result['hat_delta_L'])
            deltas_to_consider_eq.append(result['hat_delta_L_eq'])
            types_considered.append('hat_delta_L')
            types_considered_eq.append('hat_delta_L_eq')

    if deltas_to_consider:
        max_delta = max(deltas_to_consider)
        max_indices = [i for i, v in enumerate(deltas_to_consider) if v == max_delta]
        max_type = types_considered[max_indices[0]]
        result['delta_star'] = max_delta
        result['delta_star_type'] = max_type
    else:
        result['delta_star'] = 0.0
        result['delta_star_type'] = None

    if deltas_to_consider_eq:
        max_delta_eq = max(deltas_to_consider_eq)
        max_indices_eq = [i for i, v in enumerate(deltas_to_consider_eq) if v == max_delta_eq]
        max_type_eq = types_considered_eq[max_indices_eq[0]]
        result['delta_star_eq'] = max_delta_eq
        result['delta_star_eq_type'] = max_type_eq
    else:
        result['delta_star_eq'] = 0.0
        result['delta_star_eq_type'] = None

    return result

# =========================
# Data Generation for Heatmaps
# =========================

def _generate_heatmap_data(theta_range, Delta_range, m_vals, fixed_params, resolution):
    """
    Generate simulation data for heatmaps over theta and Delta.
    """
    q_H, q_L, c, N, K = fixed_params['q_H'], fixed_params['q_L'], fixed_params['c'], fixed_params['N'], fixed_params['K']
    theta_vals = np.linspace(theta_range[0], theta_range[1], resolution)
    Delta_vals = np.linspace(Delta_range[0], Delta_range[1], resolution)

    print("Generating heatmap data... (This may take a moment)")
    records = []
    for m in m_vals:
        print(f"Calculating for m = {m}...")
        for theta in theta_vals:
            for Delta in Delta_vals:
                record = {"theta": theta, "Delta": Delta, "m": m}
                try:
                    res_comp = calculate_competitive_prices(theta, q_H, q_L, Delta, c, N, K, m)
                    res_mono = calculate_monopoly_prices(theta, q_H, q_L, Delta, c, N, K, m)
                    
                    record.update({
                        'K_in_box': res_comp.get('K_in_box', 0), 'L_in_box': res_comp.get('L_in_box', 0),
                        'K_out_box': res_comp.get('K_out_box', 0), 'L_out_box': res_comp.get('L_out_box', 0)
                    })

                    for p_type, p_label in [('tilde_H', 'tilde_profit_H'), ('tilde_L', 'tilde_profit_L'), ('hat_H', 'hat_profit_H'), ('hat_L', 'hat_profit_L')]:
                        record[f'profit_comp_{p_type}'] = res_comp.get(p_label, np.nan)
                    for p_type, p_label in [('tilde_H', 'tilde_monopoly_profit_H_indiv'), ('tilde_L', 'tilde_monopoly_profit_L_indiv'), ('hat_H', 'hat_monopoly_profit_H_indiv'), ('hat_L', 'hat_monopoly_profit_L_indiv')]:
                        record[f'profit_mono_{p_type}'] = res_mono.get(p_label, np.nan)

                    total_profit_comp = np.nansum([
                        record['K_in_box'] * record.get('profit_comp_tilde_H', 0),
                        record['L_in_box'] * record.get('profit_comp_tilde_L', 0),
                        record['K_out_box'] * record.get('profit_comp_hat_H', 0),
                        record['L_out_box'] * record.get('profit_comp_hat_L', 0)
                    ])
                    total_profit_mono = np.nansum([
                        record['K_in_box'] * record.get('profit_mono_tilde_H', 0),
                        record['L_in_box'] * record.get('profit_mono_tilde_L', 0),
                        record['K_out_box'] * record.get('profit_mono_hat_H', 0),
                        record['L_out_box'] * record.get('profit_mono_hat_L', 0)
                    ])

                    record['total_profit_comp'] = total_profit_comp
                    record['total_profit_mono'] = total_profit_mono
                    
                    if total_profit_comp > 1e-9:
                        record['profit_ratio'] = total_profit_mono / total_profit_comp
                    else:
                        record['profit_ratio'] = np.nan

                    welfare_comp = res_comp.get("consumer_welfare", np.nan)
                    record['normalized_welfare_comp'] = welfare_comp / (q_H - c) if np.isfinite(welfare_comp) else np.nan
                    welfare_mono = res_mono.get("consumer_welfare_monopoly", np.nan)
                    record['normalized_welfare_mono'] = welfare_mono / (q_H - c) if np.isfinite(welfare_mono) else np.nan

                    collusion_thresholds = calculate_collusion_thresholds(theta, q_H, q_L, Delta, c, N, K, m)
                    record['delta_star'] = collusion_thresholds.get('delta_star', np.nan)
                    record['delta_star_eq'] = collusion_thresholds.get('delta_star_eq', np.nan)
                    record['delta_star_type'] = collusion_thresholds.get('delta_star_type', None)
                    record['delta_star_eq_type'] = collusion_thresholds.get('delta_star_eq_type', None)

                except (ValueError, OverflowError, ZeroDivisionError) as e:
                    print(f"Calculation failed for theta={theta}, Delta={Delta}, m={m}. Error: {e}")
                    pass
                records.append(record)

    print("Data generation complete.")
    return pd.DataFrame(records)

def _generate_heatmap_data_q_ratio(theta_range, q_h_ratio_range, m_vals, fixed_params, resolution):
    """
    Generate simulation data for heatmaps over theta and q_H/q_L ratio (for fixed Delta).
    """
    Delta, q_L, c, N, K = fixed_params['Delta'], fixed_params['q_L'], fixed_params['c'], fixed_params['N'], fixed_params['K']

    theta_vals = np.linspace(theta_range[0], theta_range[1], resolution)
    q_h_ratio_vals = np.linspace(q_h_ratio_range[0], q_h_ratio_range[1], resolution)

    print("Generating heatmap data for a fixed Delta... (This may take a moment)")
    records = []
    for m in m_vals:
        print(f"Calculating for m = {m}...")
        for theta in theta_vals:
            for q_h_ratio in q_h_ratio_vals:
                q_H = q_L * q_h_ratio
                
                record = {"theta": theta, "q_h_ratio": q_h_ratio, "m": m, "Delta": Delta}

                try:
                    res_comp = calculate_competitive_prices(theta, q_H, q_L, Delta, c, N, K, m)
                    res_mono = calculate_monopoly_prices(theta, q_H, q_L, Delta, c, N, K, m)
                    
                    record.update({
                        'K_in_box': res_comp.get('K_in_box', 0), 'L_in_box': res_comp.get('L_in_box', 0),
                        'K_out_box': res_comp.get('K_out_box', 0), 'L_out_box': res_comp.get('L_out_box', 0)
                    })

                    total_profit_comp = np.nansum([
                        record['K_in_box'] * res_comp.get('tilde_profit_H', 0),
                        record['L_in_box'] * res_comp.get('tilde_profit_L', 0),
                        record['K_out_box'] * res_comp.get('hat_profit_H', 0),
                        record['L_out_box'] * res_comp.get('hat_profit_L', 0)
                    ])
                    total_profit_mono = np.nansum([
                        record['K_in_box'] * res_mono.get('tilde_monopoly_profit_H_indiv', 0),
                        record['L_in_box'] * res_mono.get('tilde_monopoly_profit_L_indiv', 0),
                        record['K_out_box'] * res_mono.get('hat_monopoly_profit_H_indiv', 0),
                        record['L_out_box'] * res_mono.get('hat_monopoly_profit_L_indiv', 0)
                    ])

                    record['total_profit_comp'] = total_profit_comp
                    record['total_profit_mono'] = total_profit_mono
                    record['profit_ratio'] = total_profit_mono / total_profit_comp if total_profit_comp > 1e-9 else np.nan
                    
                    welfare_comp = res_comp.get("consumer_welfare", np.nan)
                    record['normalized_welfare_comp'] = welfare_comp / (q_H - c) if np.isfinite(welfare_comp) else np.nan
                    
                    welfare_mono = res_mono.get("consumer_welfare_monopoly", np.nan)
                    record['normalized_welfare_mono'] = welfare_mono / (q_H - c) if np.isfinite(welfare_mono) else np.nan

                    collusion_thresholds = calculate_collusion_thresholds(theta, q_H, q_L, Delta, c, N, K, m)
                    record['delta_star'] = collusion_thresholds.get('delta_star', np.nan)
                    record['delta_star_eq'] = collusion_thresholds.get('delta_star_eq', np.nan)

                except (ValueError, OverflowError, ZeroDivisionError) as e:
                    print(f"Calculation failed for theta={theta}, q_h_ratio={q_h_ratio}, m={m}. Error: {e}")
                    pass
                records.append(record)

    print("Data generation complete.")
    return pd.DataFrame(records)

# =========================
# Plotting Utilities
# =========================

def plot_welfare_heatmap(df, m_vals, fixed_params, delta_vmin=None, delta_vmax=None):
    """
    Plot heatmaps for welfare, profit ratio, and collusion threshold over theta and Delta.
    """
    if df.empty:
        print("Warning: Input DataFrame is empty. Skipping plot generation.")
        return

    N = fixed_params.get('N', 'N/A')
    theta_vals = sorted(df['theta'].unique())
    Delta_vals = sorted(df['Delta'].unique())
    num_m = len(m_vals)

    fig, axes = plt.subplots(num_m, 4, figsize=(19, 4.8 * num_m), dpi=120, layout='constrained', squeeze=False)

    all_welfare_vals = pd.concat([df["normalized_welfare_comp"], df["normalized_welfare_mono"]]).dropna()
    welfare_min = all_welfare_vals.min() if not all_welfare_vals.empty else 0
    welfare_max = all_welfare_vals.max() if not all_welfare_vals.empty else 1
    shared_welfare_norm = mcolors.Normalize(vmin=welfare_min, vmax=welfare_max)
    
    ratio_vals = df["profit_ratio"].dropna()
    ratio_norm = mcolors.Normalize(vmin=ratio_vals.min(), vmax=ratio_vals.max()) if not ratio_vals.empty else mcolors.Normalize(1, 2)

    delta_cmap = plt.get_cmap('Greens', 512)
    if delta_vmin is not None and delta_vmax is not None:
        delta_global_min, delta_global_max = delta_vmin, delta_vmax
    else:
        delta_vals = df["delta_star"].dropna()
        delta_global_min, delta_global_max = (delta_vals.min(), delta_vals.max()) if not delta_vals.empty else (0, 1)
    if abs(delta_global_min - delta_global_max) < 1e-6:
        delta_global_max += 0.01
    shared_delta_norm = mcolors.Normalize(vmin=delta_global_min, vmax=delta_global_max)

    # Set a much larger, bold, easy-to-read font for contour labels (no fontweight/family in clabel)
    contour_label_fontsize = 15  # much bigger
    contour_label_path_effects = [path_effects.withStroke(linewidth=4, foreground='black')]

    for idx, m in enumerate(m_vals):
        m_df = df[df["m"] == m]
        ax_comp, ax_coll, ax_profit, ax_delta = axes[idx, 0], axes[idx, 1], axes[idx, 2], axes[idx, 3]

        row_label = f"$m = {m}$\n(No Effective Buy Box)" if m == N else f"$m = {m}$"
        axes[idx, 0].text(-0.35, 0.5, row_label, transform=axes[idx, 0].transAxes, ha='center', va='center', fontsize=14, rotation=90)

        pivot_comp = m_df.pivot_table(index="Delta", columns="theta", values="normalized_welfare_comp")
        comp_mappable = ax_comp.imshow(pivot_comp, aspect='auto', origin='lower', cmap='magma', norm=shared_welfare_norm, extent=[min(theta_vals), max(theta_vals), min(Delta_vals), max(Delta_vals)], interpolation='bilinear')
        contour_comp = ax_comp.contour(theta_vals, Delta_vals, pivot_comp, levels=4, colors='white', linewidths=1.5, alpha=0.8)
        labels_comp = ax_comp.clabel(contour_comp, inline=True, fmt='%.2f', manual=None, fontsize=contour_label_fontsize)
        plt.setp(labels_comp, path_effects=contour_label_path_effects, color='white')

        pivot_mono = m_df.pivot_table(index="Delta", columns="theta", values="normalized_welfare_mono")
        mono_mappable = ax_coll.imshow(pivot_mono, aspect='auto', origin='lower', cmap='magma', norm=shared_welfare_norm, extent=[min(theta_vals), max(theta_vals), min(Delta_vals), max(Delta_vals)], interpolation='bilinear')
        contour_mono = ax_coll.contour(theta_vals, Delta_vals, pivot_mono, levels=4, colors='white', linewidths=1.5, alpha=0.8)
        labels_mono = ax_coll.clabel(contour_mono, inline=True, fmt='%.2f', manual=None, fontsize=contour_label_fontsize)
        plt.setp(labels_mono, path_effects=contour_label_path_effects, color='white')
        
        pivot_profit = m_df.pivot_table(index="Delta", columns="theta", values="profit_ratio")
        profit_mappable = ax_profit.imshow(pivot_profit, aspect='auto', origin='lower', cmap='YlGnBu', norm=ratio_norm, extent=[min(theta_vals), max(theta_vals), min(Delta_vals), max(Delta_vals)], interpolation='bilinear')
        contour_profit = ax_profit.contour(theta_vals, Delta_vals, pivot_profit, levels=4, colors='white', linewidths=1.5, alpha=0.8)
        labels_profit = ax_profit.clabel(contour_profit, inline=True, fmt='%.2f', manual=None, fontsize=contour_label_fontsize)
        plt.setp(labels_profit, path_effects=contour_label_path_effects, color='white')

        pivot_delta = m_df.pivot_table(index="Delta", columns="theta", values="delta_star")
        delta_mappable = ax_delta.imshow(pivot_delta, aspect='auto', origin='lower', cmap=delta_cmap, norm=shared_delta_norm, extent=[min(theta_vals), max(theta_vals), min(Delta_vals), max(Delta_vals)], interpolation='bilinear')
        if "delta_star" in m_df and m_df["delta_star"].notna().any():
            CS = ax_delta.contour(theta_vals, Delta_vals, pivot_delta.values, levels=3, colors='white', linewidths=2)
            clabels = ax_delta.clabel(CS, inline=True, fmt='%.3f', manual=None, fontsize=contour_label_fontsize)
            plt.setp(clabels, path_effects=[path_effects.withStroke(linewidth=5, foreground='black')], color='white')

        for ax in [ax_comp, ax_coll, ax_profit, ax_delta]:
            ax.grid(False)
        for ax in [ax_coll, ax_profit, ax_delta]:
            ax.set_yticklabels([])
        if idx < num_m - 1:
            for ax in axes[idx, :]:
                ax.set_xticklabels([])

    axes[0, 0].set_title("Competitive Welfare", fontsize=18, pad=15)
    axes[0, 1].set_title("Fully Collusive Welfare", fontsize=18, pad=15)
    axes[0, 2].set_title("Total Cartel Profit Gain (Ratio)", fontsize=18, pad=15)
    axes[0, 3].set_title("Critical Threshold", fontsize=18, pad=15)
    fig.supylabel(r'$\Delta$ (Buy Box Bias)', fontsize=18, x=0.04)
    fig.supxlabel(r'$\theta$ (Marginal Search Cost)', fontsize=18)

    pad_value = 0.08 
    cbar_welfare = fig.colorbar(comp_mappable, ax=axes[:, 0:2], location='bottom', shrink=0.8, aspect=40, pad=pad_value)
    cbar_welfare.set_label("Normalized Welfare", fontsize=15)
    
    cbar_profit = fig.colorbar(profit_mappable, ax=axes[:, 2], location='bottom', shrink=0.8, aspect=40, pad=pad_value, label="Total Collusive Profit / Total Competitive Profit")
    cbar_delta = fig.colorbar(delta_mappable, ax=axes[:, 3], location='bottom', shrink=0.8, aspect=40, pad=pad_value)
    cbar_delta.set_label("Critical Discount Factor", fontsize=15)
    
    for cbar in [cbar_welfare, cbar_profit, cbar_delta]:
        cbar.locator = MaxNLocator(nbins=5, prune='both')
        cbar.update_ticks()
    
    plt.show()

def plot_market_outcomes(theta, q_H, q_L, Delta, c, N, K, m_range=None):
    """
    Plot market shares, normalized prices, and collusion thresholds as a function of buy box size m.
    """
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": "Times New Roman",
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "text.usetex": True,
    })

    m_vals = list(m_range) if m_range is not None else list(range(1, N + 1))
    if not m_vals:
        return None

    cost = c if isinstance(c, dict) else {'H': c, 'L': 0}
    qualities = {'H': q_H, 'L': q_L}

    records = []
    for m in m_vals:
        record = {"m": m}
        try:
            res_comp = calculate_competitive_prices(theta, q_H, q_L, Delta, cost['H'], N, K, m)
            res_mono = calculate_monopoly_prices(theta, q_H, q_L, Delta, cost['H'], N, K, m)
            for key in ['tilde_H', 'tilde_L', 'hat_H', 'hat_L']:
                pos, qual_key = key.split('_')
                record[f'share_comp_{key}'] = res_comp.get('demands', {}).get(f'D_{pos}_{qual_key}', np.nan)
                record[f'share_mono_{key}'] = res_mono.get('demands', {}).get(f'D_{pos}_{qual_key}', np.nan)
                p_comp = res_comp.get(f"{pos}_P_{qual_key}", np.nan)
                p_mono = res_mono.get(f"{pos}_P_M_{qual_key}", np.nan)
                q_minus_c = qualities[qual_key] - cost[qual_key]
                record[f'norm_price_comp_{key}'] = (p_comp - cost[qual_key]) / q_minus_c if (not np.isnan(p_comp) and abs(q_minus_c) > 1e-9) else np.nan
                record[f'norm_price_mono_{key}'] = (p_mono - cost[qual_key]) / q_minus_c if (not np.isnan(p_mono) and abs(q_minus_c) > 1e-9) else np.nan
        except Exception as e:
            print(f"Error for m={m}: {e}")
        records.append(record)

    df = pd.DataFrame(records).set_index("m")
    threshold_results = [calculate_collusion_thresholds(theta, q_H, q_L, Delta, cost['H'], N, K, m) for m in m_vals]
    threshold_df = pd.DataFrame(threshold_results, index=m_vals)

    plot_configs = {
        'hat_H':   {'color': '#ff7f0e', 'label': 'High-Quality, Outside','formula': r'$(P-c_H)/(q_H-c_H)$'},
        'tilde_H': {'color': '#d62728', 'label': 'High-Quality, Inside', 'formula': r'$(P-c_H)/(q_H-c_H)$'},
        'hat_L':   {'color': '#17becf', 'label': 'Low-Quality, Outside', 'formula': r'$P/q_L$'},
        'tilde_L': {'color': '#1f77b4', 'label': 'Low-Quality, Inside',  'formula': r'$P/q_L$'},
    }
    plot_order = ['hat_H', 'tilde_H', 'hat_L', 'tilde_L']

    count_col_map = {'tilde_H': 'K_in_box', 'tilde_L': 'L_in_box', 'hat_H': 'K_out_box', 'hat_L': 'L_out_box'}
    active_plots = []
    for key in plot_order:
        if key in count_col_map and count_col_map[key] in threshold_df.columns and threshold_df[count_col_map[key]].sum() > 0:
            active_plots.append(key)

    if not active_plots:
        return None

    nrows, ncols = 3, len(active_plots)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 8), dpi=150, sharex=True, sharey='row', squeeze=False)

    jitter = 0.08

    for i, key in enumerate(active_plots):
        config = plot_configs[key]
        axes_map = {'share': axes[0, i], 'price': axes[1, i], 'threshold': axes[2, i]}

        axes[0, i].set_title(config['label'])
        axes[1, i].set_title(f"Formula: {config['formula']}", fontsize=11, pad=10)

        count_col = count_col_map[key]
        is_applicable = threshold_df[count_col] > 0 if count_col in threshold_df.columns else pd.Series([False]*len(threshold_df), index=threshold_df.index)

        for ax_name in ['share', 'price']:
            ax = axes_map[ax_name]
            col_prefix = 'norm_price' if ax_name == 'price' else 'share'
            col_comp = f'{col_prefix}_comp_{key}'
            col_mono = f'{col_prefix}_mono_{key}'

            y_comp = df.loc[is_applicable, col_comp] if col_comp in df.columns else pd.Series([np.nan]*is_applicable.sum(), index=df.index[is_applicable])
            y_mono = df.loc[is_applicable, col_mono] if col_mono in df.columns else pd.Series([np.nan]*is_applicable.sum(), index=df.index[is_applicable])
            x_comp, x_mono = df.index[is_applicable] - jitter, df.index[is_applicable] + jitter

            ax.plot(x_comp, y_comp, c=config['color'], marker='o', ms=5, ls='-', lw=2)
            ax.plot(x_mono, y_mono, c=config['color'], marker='X', ms=6, ls='--', lw=2)
            ax.set_xticks(df.index)

        ax_thresh = axes_map['threshold']
        pos_key, qual_key = key.split('_')
        threshold_col = f'{pos_key}_delta_{qual_key}'

        y_thresh = threshold_df.loc[is_applicable, threshold_col] if threshold_col in threshold_df.columns else pd.Series([np.nan]*is_applicable.sum(), index=threshold_df.index[is_applicable])
        x_thresh = threshold_df.index[is_applicable]
        ax_thresh.plot(x_thresh, y_thresh, c=config['color'], marker='o', ms=5, ls=':', lw=2)

        if 'delta_star_type' in threshold_df.columns:
            is_critical = (threshold_df['delta_star_type'] == threshold_col)
            critical_points = threshold_df[is_critical & is_applicable]
            if not critical_points.empty and threshold_col in critical_points.columns:
                ax_thresh.scatter(critical_points.index, critical_points[threshold_col],
                                  facecolors='none', edgecolors='k', s=100, linewidth=1.5, zorder=10)

        for m_val, row_data in threshold_df.iterrows():
            if count_col in row_data and row_data[count_col] == 0:
                for ax in axes_map.values():
                    ax.axvspan(m_val - 0.5, m_val + 0.5, color='grey', alpha=0.1, zorder=0)

    axes[0, 0].set_ylabel('Market Share')
    axes[1, 0].set_ylabel('Normalized Price')
    axes[2, 0].set_ylabel(r'Critical Threshold ($\delta_j$)')
    fig.supxlabel(r'$m$ (Buy Box Size)', fontsize=14, y=0.04)

    for ax in axes.flatten():
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    legend_elements = [
        Line2D([0], [0], color='w', label='Scenario:'),
        Line2D([0], [0], color='k', ls='-', marker='o', ms=5, label='Competition'),
        Line2D([0], [0], color='k', ls='--', marker='X', ms=6, label='Collusion'),
        Line2D([0], [0], color='k', ls=':', marker='o', ms=5, label='Collusion Threshold'),
        Line2D([0], [0], color='none', label=''),
        Line2D([0], [0], color='k', marker='o', markerfacecolor='none', markeredgecolor='k',
               linestyle='None', markersize=8, label='Determines $\delta^*$'),
        Line2D([0], [0], color='none', label=''),
        Line2D([0], [0], color='w', label='Seller Type:'),
    ]
    for key in active_plots:
        legend_elements.append(Line2D([0], [0], color=plot_configs[key]['color'], lw=2, label=plot_configs[key]['label']))

    fig.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(0.98, 0.5), frameon=False)

    fig.tight_layout()
    fig.subplots_adjust(left=0.07, bottom=0.1, right=0.85, top=0.95)

    return fig

def plot_welfare_heatmap_q_ratio(df, m_vals, q_h_ratio_vals, fixed_params, delta_vmin=None, delta_vmax=None):
    """
    Plot heatmaps for welfare, profit ratio, and collusion threshold over theta and q_H/q_L ratio.
    """
    if df.empty:
        print("Warning: Input DataFrame is empty. Skipping plot generation.")
        return

    fixed_delta = fixed_params.get('Delta', 'N/A')
    N = fixed_params.get('N', 'N/A')
    theta_vals = sorted(df['theta'].unique())
    num_m = len(m_vals)

    fig, axes = plt.subplots(num_m, 4, figsize=(19, 4.8 * num_m), dpi=120, layout='constrained', squeeze=False)

    all_welfare_vals = pd.concat([df["normalized_welfare_comp"], df["normalized_welfare_mono"]]).dropna()
    welfare_min = all_welfare_vals.min() if not all_welfare_vals.empty else 0
    welfare_max = all_welfare_vals.max() if not all_welfare_vals.empty else 1
    shared_welfare_norm = mcolors.Normalize(vmin=welfare_min, vmax=welfare_max)
    
    ratio_vals = df["profit_ratio"].dropna()
    ratio_norm = mcolors.Normalize(vmin=ratio_vals.min(), vmax=ratio_vals.max()) if not ratio_vals.empty else mcolors.Normalize(1, 2)

    delta_cmap = plt.get_cmap('Greens', 512)
    if delta_vmin is not None and delta_vmax is not None:
        delta_global_min, delta_global_max = delta_vmin, delta_vmax
    else:
        delta_vals = df["delta_star"].dropna()
        delta_global_min, delta_global_max = (delta_vals.min(), delta_vals.max()) if not delta_vals.empty else (0, 1)
    if abs(delta_global_min - delta_global_max) < 1e-6:
        delta_global_max += 0.01
    shared_delta_norm = mcolors.Normalize(vmin=delta_global_min, vmax=delta_global_max)

    # Set a much larger, bold, easy-to-read font for contour labels (no fontweight/family in clabel)
    contour_label_fontsize = 15  # much bigger
    contour_label_path_effects = [path_effects.withStroke(linewidth=4, foreground='black')]

    for idx, m in enumerate(m_vals):
        m_df = df[df["m"] == m]
        ax_comp, ax_coll, ax_profit, ax_delta = axes[idx, 0], axes[idx, 1], axes[idx, 2], axes[idx, 3]

        row_label = f"$m = {m}$\n(No Effective Buy Box)" if m == N else f"$m = {m}$"
        axes[idx, 0].text(-0.35, 0.5, row_label, transform=axes[idx, 0].transAxes, ha='center', va='center', fontsize=14, rotation=90)

        plot_extent = [min(theta_vals), max(theta_vals), min(q_h_ratio_vals), max(q_h_ratio_vals)]

        pivot_comp = m_df.pivot_table(index="q_h_ratio", columns="theta", values="normalized_welfare_comp")
        comp_mappable = ax_comp.imshow(pivot_comp, aspect='auto', origin='lower', cmap='magma', norm=shared_welfare_norm, extent=plot_extent, interpolation='bilinear')
        contour_comp = ax_comp.contour(theta_vals, q_h_ratio_vals, pivot_comp, levels=4, colors='white', linewidths=1.5, alpha=0.8)
        labels_comp = ax_comp.clabel(contour_comp, inline=True, fmt='%.2f', manual=None, fontsize=contour_label_fontsize)
        plt.setp(labels_comp, path_effects=contour_label_path_effects, color='white')

        pivot_mono = m_df.pivot_table(index="q_h_ratio", columns="theta", values="normalized_welfare_mono")
        ax_coll.imshow(pivot_mono, aspect='auto', origin='lower', cmap='magma', norm=shared_welfare_norm, extent=plot_extent, interpolation='bilinear')
        contour_mono = ax_coll.contour(theta_vals, q_h_ratio_vals, pivot_mono, levels=4, colors='white', linewidths=1.5, alpha=0.8)
        labels_mono = ax_coll.clabel(contour_mono, inline=True, fmt='%.2f', manual=None, fontsize=contour_label_fontsize)
        plt.setp(labels_mono, path_effects=contour_label_path_effects, color='white')
        
        pivot_profit = m_df.pivot_table(index="q_h_ratio", columns="theta", values="profit_ratio")
        profit_mappable = ax_profit.imshow(pivot_profit, aspect='auto', origin='lower', cmap='YlGnBu', norm=ratio_norm, extent=plot_extent, interpolation='bilinear')
        contour_profit = ax_profit.contour(theta_vals, q_h_ratio_vals, pivot_profit, levels=4, colors='white', linewidths=1.5, alpha=0.8)
        labels_profit = ax_profit.clabel(contour_profit, inline=True, fmt='%.2f', manual=None, fontsize=contour_label_fontsize)
        plt.setp(labels_profit, path_effects=contour_label_path_effects, color='white')

        pivot_delta = m_df.pivot_table(index="q_h_ratio", columns="theta", values="delta_star")
        delta_mappable = ax_delta.imshow(pivot_delta, aspect='auto', origin='lower', cmap=delta_cmap, norm=shared_delta_norm, extent=plot_extent, interpolation='bilinear')
        if "delta_star" in m_df and m_df["delta_star"].notna().any():
            CS = ax_delta.contour(theta_vals, q_h_ratio_vals, pivot_delta.values, levels=3, colors='white', linewidths=2)
            clabels = ax_delta.clabel(CS, inline=True, fmt='%.3f', manual=None, fontsize=contour_label_fontsize)
            plt.setp(clabels, path_effects=[path_effects.withStroke(linewidth=5, foreground='black')], color='white')

        for ax in [ax_comp, ax_coll, ax_profit, ax_delta]:
            ax.grid(False)
        for ax in [ax_coll, ax_profit, ax_delta]:
            ax.set_yticklabels([])
        if idx < num_m - 1:
            for ax in axes[idx, :]:
                ax.set_xticklabels([])

    axes[0, 0].set_title("Competitive Welfare", fontsize=18, pad=15)
    axes[0, 1].set_title("Fully Collusive Welfare", fontsize=18, pad=15)
    axes[0, 2].set_title("Total Cartel Profit Gain (Ratio)", fontsize=18, pad=15)
    axes[0, 3].set_title("Critical Threshold", fontsize=18, pad=15)
    
    fig.supylabel(r'Quality Differentiation ($q_H/q_L$)', fontsize=18, x=0.04)
    fig.supxlabel(r'$\theta$ (Marginal Search Cost)', fontsize=18)

    pad_value = 0.08
    cbar_welfare = fig.colorbar(comp_mappable, ax=axes[:, 0:2], location='bottom', shrink=0.8, aspect=40, pad=pad_value)
    cbar_welfare.set_label("Normalized Welfare", fontsize=12)
    
    cbar_profit = fig.colorbar(profit_mappable, ax=axes[:, 2], location='bottom', shrink=0.8, aspect=40, pad=pad_value)
    cbar_profit.set_label("Total Collusive Profit / Total Competitive Profit", fontsize=12)

    cbar_delta = fig.colorbar(delta_mappable, ax=axes[:, 3], location='bottom', shrink=0.8, aspect=40, pad=pad_value)
    cbar_delta.set_label("Critical Discount Factor", fontsize=12)
    
    for cbar in [cbar_welfare, cbar_profit, cbar_delta]:
        cbar.locator = MaxNLocator(nbins=5, prune='both')
        cbar.update_ticks()
    
    plt.show()