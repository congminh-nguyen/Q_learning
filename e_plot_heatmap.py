# e_heatmap_plot.py (UPDATED: Fixes for layout, spacing, and legend text)
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
from matplotlib.ticker import MaxNLocator
from typing import List

def plot_alpha_beta_heatmaps(df: pd.DataFrame, m_vals: List[int], output_dir: str):
    """
    Generate two side-by-side heatmaps for each m:
      - Consumer Surplus divided by (q_H - c) [high is good]
      - Consumer Surplus normalized [low is good]
    across alpha (y-axis) and beta (x-axis).
    Uses a unified colorbar for each metric across all m for direct comparison.
    Includes explicit color legend notations with improved spacing.
    """
    if df.empty:
        print("Warning: Input DataFrame is empty. Skipping plot generation.")
        return

    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 11, 'font.family': 'serif', 'axes.labelsize': 12, 'axes.titlesize': 13,
        'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 10,
        'figure.titlesize': 14, 'lines.linewidth': 2.5, 'lines.markersize': 8, 'axes.grid': False
    })

    # Color maps and legend system
    cmap_cs_quality = 'YlGn'    # Yellow (low) to Green (high)
    cmap_cs_normalized = 'YlOrRd'  # Yellow (low) to Orange to Red (high)

    metrics = [
        ("consumer_surplus_per_quality_cost", r"Consumer Welfare / $(q_H - c)$", "Normalized Value", cmap_cs_quality, "high"),
        ("consumer_surplus_normalized", r"Consumer Welfare (normalized)", "0 = Competitive, 1 = Collusion", cmap_cs_normalized, "low"),
    ]

    alpha_vals = sorted(df['alpha_initial'].unique())
    beta_vals = sorted(df['beta_exploration'].unique())
    num_m = len(m_vals)

    # --- Compute global vmin/vmax for each metric for unified colorbar ---
    global_norms = []
    global_vmins = []
    global_vmaxs = []
    for metric, _, _, _, _ in metrics:
        all_vals = []
        for m in m_vals:
            m_df = df[df["m"] == m]
            if not m_df.empty:
                vals = m_df.pivot_table(index="alpha_initial", columns="beta_exploration", values=metric).values.flatten()
                all_vals.append(vals)
        
        if not all_vals:
            global_norms.append(mcolors.Normalize(vmin=0, vmax=1))
            global_vmins.append(0)
            global_vmaxs.append(1)
            continue

        all_vals = np.concatenate(all_vals)
        all_vals = all_vals[~np.isnan(all_vals)]

        if len(all_vals) == 0:
            vmin, vmax = 0, 1
        else:
            vmin = np.nanpercentile(all_vals, 1)
            vmax = np.nanpercentile(all_vals, 99)
            if np.isclose(vmin, vmax):
                vmin = np.nanmin(all_vals)
                vmax = np.nanmax(all_vals)
                if np.isclose(vmin, vmax): vmin, vmax = 0, 1
        global_norms.append(mcolors.Normalize(vmin=vmin, vmax=vmax))
        global_vmins.append(vmin)
        global_vmaxs.append(vmax)

    # --- Figure setup: Adjusted figsize for more compact layout ---
    fig, axes = plt.subplots(num_m, 2, figsize=(11, 4.5 * num_m), dpi=120, squeeze=False)
    im_handles = [None, None]

    for idx, m in enumerate(m_vals):
        m_df = df[df["m"] == m]
        if m_df.empty:
            axes[idx, 0].set_visible(False)
            axes[idx, 1].set_visible(False)
            continue
            
        for col, (metric, title, cbar_label, cmap, direction) in enumerate(metrics):
            pivot = m_df.pivot_table(index="alpha_initial", columns="beta_exploration", values=metric)
            pivot = pivot.reindex(index=alpha_vals, columns=beta_vals)

            im = axes[idx, col].imshow(
                pivot, aspect='auto', origin='lower', cmap=cmap, norm=global_norms[col],
                extent=[min(beta_vals), max(beta_vals), min(alpha_vals), max(alpha_vals)],
                interpolation='nearest'
            )

            if idx == 0:
                im_handles[col] = im
                axes[idx, col].set_title(title, fontsize=14, pad=15)

            if col == 0:
                axes[idx, col].set_ylabel(r'$\alpha$ (Learning Rate)', fontsize=12)
            else:
                axes[idx, col].set_yticklabels([])

            axes[idx, col].xaxis.set_major_locator(MaxNLocator(nbins=5, prune='both'))
            axes[idx, col].yaxis.set_major_locator(MaxNLocator(nbins=5, prune='both'))

            formatter_beta = ticker.ScalarFormatter(useMathText=True)
            formatter_beta.set_scientific(True)
            formatter_beta.set_powerlimits((0, 0))
            axes[idx, col].xaxis.set_major_formatter(formatter_beta)
            axes[idx, col].yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=False))

            plt.setp(axes[idx, col].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

            if idx < num_m - 1:
                axes[idx, col].set_xticklabels([])
            else:
                axes[idx, col].set_xlabel(r'$\beta$ (Experimentation Rate)', fontsize=12)

        row_label = f"$m = {m}$"
        axes[idx, 0].text(-0.3, 0.5, row_label, transform=axes[idx, 0].transAxes, ha='center', va='center', fontsize=14, rotation=90)

    # --- Adjust main layout to prevent overlaps and reduce whitespace ---
    fig.subplots_adjust(left=0.1, right=0.98, bottom=0.2, top=0.92, hspace=0.3, wspace=0.15)
    
    # --- Add unified colorbars with improved spacing ---
    cbar_y_position = 0.06
    cbar_height = 0.025
    cbar_ax1_pos = [0.12, cbar_y_position, 0.35, cbar_height]
    cbar_ax2_pos = [0.58, cbar_y_position, 0.35, cbar_height]
    
    # Y-coordinates for text relative to colorbar axes. >1 is above, <0 is below.
    text_y_above = 1.8  # Position for "Min/Max" text
    text_y_below = -2.8 # Position for descriptive color text

    if im_handles[0]:
        cbar_ax1 = fig.add_axes(cbar_ax1_pos)
        cbar1 = fig.colorbar(im_handles[0], cax=cbar_ax1, orientation='horizontal')
        cbar1.set_label(r"Consumer Welfare / $(q_H-c)$", fontsize=12, labelpad=10)
        cbar1.locator = MaxNLocator(nbins=5, prune='both')
        cbar1.update_ticks()
        
        cbar_ax1.text(0.0, text_y_above, f"Min={global_vmins[0]:.2g}", ha='left', va='bottom', fontsize=10, transform=cbar_ax1.transAxes)
        cbar_ax1.text(1.0, text_y_above, f"Max={global_vmaxs[0]:.2g}", ha='right', va='bottom', fontsize=10, transform=cbar_ax1.transAxes)
        
        # CORRECTED: Yellow (low) on left, Green (high) on right for 'YlGn'
        cbar_ax1.text(0.0, text_y_below, "Yellow = Lower Welfare", ha='left', va='center', fontsize=10, color='#B8860B', transform=cbar_ax1.transAxes)
        cbar_ax1.text(1.0, text_y_below, "Green = Higher Welfare", ha='right', va='center', fontsize=10, color='#228B22', transform=cbar_ax1.transAxes)

    if im_handles[1]:
        cbar_ax2 = fig.add_axes(cbar_ax2_pos)
        cbar2 = fig.colorbar(im_handles[1], cax=cbar_ax2, orientation='horizontal')
        cbar2.set_label("Normalized Welfare (0=Competitive, 1=Collusion)", fontsize=12, labelpad=10)
        cbar2.locator = MaxNLocator(nbins=5, prune='both')
        cbar2.update_ticks()

        cbar_ax2.text(0.0, text_y_above, f"Min={global_vmins[1]:.2g}", ha='left', va='bottom', fontsize=10, transform=cbar_ax2.transAxes)
        cbar_ax2.text(1.0, text_y_above, f"Max={global_vmaxs[1]:.2g}", ha='right', va='bottom', fontsize=10, transform=cbar_ax2.transAxes)
        
        cbar_ax2.text(0.0, text_y_below, "Yellow = Competitive", ha='left', va='center', fontsize=10, color='#B8860B', transform=cbar_ax2.transAxes)
        cbar_ax2.text(1.0, text_y_below, "Red = Collusive", ha='right', va='center', fontsize=10, color='#B22222', transform=cbar_ax2.transAxes)

    fig.suptitle('Consumer Surplus Metrics Across Learning and Experimentation Rates', fontsize=16, fontweight='bold')

    plot_path = os.path.join(output_dir, "alpha_beta_consumer_surplus_heatmaps_corrected.png")
    try:
        # Use bbox_inches='tight' to crop final whitespace
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Heatmaps saved to: {plot_path}")
    except Exception as e:
        print(f"❌ Error saving heatmaps: {e}")
    plt.close(fig)

def main_plot_data():
    # --- Configuration for loading data and plotting ---
    data_directory = "heatmap_alpha_beta_data_N3_m1_2_3_20250619_121635"
    csv_file_path = os.path.join(data_directory, "heatmap_data.csv")
    m_values_used = [1, 2, 3]

    if not os.path.exists(csv_file_path):
        print(f"Error: Data file not found at {csv_file_path}")
        # Create dummy data for demonstration if file not found
        print("Generating dummy data for demonstration purposes...")
        os.makedirs(data_directory, exist_ok=True)
        alpha_initial = np.linspace(0.01, 0.1, 10)
        beta_exploration = np.logspace(-5, -2, 10)
        data = []
        for m in m_values_used:
            for alpha in alpha_initial:
                for beta in beta_exploration:
                    base_welfare = (1 - (alpha * 10)) * (1 - (np.log10(beta) + 5) / 3) / m
                    cs_per_qc = np.clip(base_welfare * 5.0 + np.random.rand() * 0.1, 0, 5)
                    cs_norm = np.clip((1 - base_welfare) + np.random.rand() * 0.05, 0, 1)
                    data.append({
                        "m": m, "alpha_initial": alpha, "beta_exploration": beta,
                        "consumer_surplus_per_quality_cost": cs_per_qc,
                        "consumer_surplus_normalized": cs_norm
                    })
        dummy_df = pd.DataFrame(data)
        dummy_df.to_csv(csv_file_path, index=False)
        heatmap_df = dummy_df
    else:
        print(f"Loading data from: {csv_file_path}")
        heatmap_df = pd.read_csv(csv_file_path)

    print("Data loaded successfully. Generating plots...")
    plot_alpha_beta_heatmaps(heatmap_df, m_values_used, data_directory)
    print("\nPlotting complete.")

if __name__ == "__main__":
    main_plot_data()