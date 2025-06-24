# --- e_heatmap_plot.py (UPDATED for m vs. delta heatmap, dual metrics, improved color, and new colorbar label for normalized) ---
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker 
from matplotlib.ticker import MaxNLocator
from typing import List

def plot_m_delta_heatmaps(df: pd.DataFrame, m_vals: List[int], delta_vals: List[float], output_dir: str):
    """
    Generate two side-by-side heatmaps:
      - Consumer Surplus divided by (q_H - c) [high is good]
      - Consumer Surplus normalized [low is good]
    across m (x-axis) and delta (y-axis).
    """
    if df.empty:
        print("Warning: Input DataFrame is empty. Skipping plot generation.")
        return

    # Use the correct column name for delta (from your data: 'delta_rl')
    delta_col = 'delta_rl'

    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 11, 'font.family': 'serif', 'axes.labelsize': 12, 'axes.titlesize': 13,
        'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 10,
        'figure.titlesize': 14, 'lines.linewidth': 2.5, 'lines.markersize': 8, 'axes.grid': False
    })

    # Improved color maps for clarity:
    # - For CS/(q_H-c): high is good, use 'YlGn' (yellow=low, green=high, no red)
    # - For normalized: low is good, use 'YlOrRd' (yellow=low/good, red=high/bad)
    cmap_cs_quality = 'YlGn'
    cmap_cs_normalized = 'YlOrRd'

    metrics = [
        ("consumer_surplus_per_quality_cost", r"Consumer Welfare / $(q_H - c)$", "Normalized Value (Red=Worse, Green=Better)", cmap_cs_quality, "high"),
        ("consumer_surplus_normalized", r"Consumer Welfare (normalized)", "0 = Competitive (Blue), 1 = Collusion (Red)", cmap_cs_normalized, "low"),
    ]

    # Ensure delta_vals and m_vals are sorted for correct plotting order
    delta_vals = sorted(df[delta_col].unique())
    m_vals = sorted(df['m'].unique())

    # Figure setup: 1 row, 2 columns for the two metrics
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=120, sharey=True)
    if not isinstance(axes, np.ndarray):
        axes = [axes]

    for ax, (metric, title, cbar_label, cmap, direction) in zip(axes, metrics):
        # Pivot for the specific metric
        pivot = df.pivot_table(index=delta_col, columns="m", values=metric)
        pivot = pivot.reindex(index=delta_vals, columns=m_vals)

        # Robust normalization: use 1st and 99th percentiles to enhance color contrast
        vmin = np.nanpercentile(pivot.values, 1)
        vmax = np.nanpercentile(pivot.values, 99)
        if np.isclose(vmin, vmax):
            vmin = np.nanmin(pivot.values)
            vmax = np.nanmax(pivot.values)
            if np.isclose(vmin, vmax):
                vmin, vmax = 0, 1

        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        # Plot the heatmap
        im = ax.imshow(
            pivot, aspect='auto', origin='lower', cmap=cmap, norm=norm,
            extent=[min(m_vals)-0.5, max(m_vals)+0.5, min(delta_vals), max(delta_vals)],
            interpolation='nearest'
        )

        # Set title and labels
        ax.set_title(f"{title}", fontsize=14, pad=20)
        ax.set_xlabel(r'$m$ (Number of Sellers)', fontsize=12)
        if ax is axes[0]:
            ax.set_ylabel(r'$\delta$ (Discount Factor)', fontsize=12)

        # Set discrete ticks for m (x-axis)
        ax.set_xticks(m_vals)
        ax.set_xticklabels([str(int(val)) for val in m_vals])

        # Set discrete ticks for delta (y-axis)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=len(delta_vals), prune='both'))
        ax.set_yticks(delta_vals)
        ax.set_yticklabels([f'{val:.2f}' for val in delta_vals])

        # Add colorbar to each subplot
        cbar = fig.colorbar(im, ax=ax, orientation='vertical', pad=0.03, fraction=0.045)
        cbar.set_label(cbar_label, fontsize=12)
        cbar.locator = MaxNLocator(nbins=5, prune='both')
        cbar.update_ticks()

    plt.tight_layout(rect=[0.03, 0.05, 0.97, 0.95])

    plot_path = os.path.join(output_dir, "m_delta_consumer_surplus_dual_heatmap.png")
    try:
        plt.savefig(plot_path, dpi=300)
        print(f"📊 Dual heatmap saved to: {plot_path}")
    except Exception as e:
        print(f"❌ Error saving heatmap: {e}")
    plt.close(fig)


def main_plot_data():
    # --- Configuration for loading data and plotting ---
    data_directory = "heatmap_m_delta_rl_data_N3_m1_2_3_delta_rl01_02_03_04_05_06_07_08_09_20250621_104113"  # Update as needed
    csv_file_path = os.path.join(data_directory, "heatmap_data.csv")

    # Define the m and delta values that were used during computation
    m_values_used = [1, 2, 3]
    delta_values_used = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # Use all available for full heatmap

    if not os.path.exists(csv_file_path):
        print(f"Error: Data file not found at {csv_file_path}")
        print("Please run 'e_heatmap_compute.py' first to generate the data (with the new CS metric),")
        print("and update 'data_directory' in 'e_heatmap_plot.py' to point to the correct output folder.")
        return

    print(f"Loading data from: {csv_file_path}")
    heatmap_df = pd.read_csv(csv_file_path)
    print("Data loaded successfully. Generating plot...")

    # Generate the dual heatmap
    plot_m_delta_heatmaps(heatmap_df, m_values_used, delta_values_used, data_directory)

    print("\nPlotting complete.")

if __name__ == "__main__":
    main_plot_data()