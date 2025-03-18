#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

def escape_gloss(gloss):
    """
    Escape dollar signs in gloss names to avoid mathtext parsing errors.
    """
    return str(gloss).replace('$', r'\$')

def plot_aggregated_metrics(agg, output_dir):
    """
    For each metric (GD, IGT, OGT, TGT), create two separate plots:
    one for the average per gloss and one for the median per gloss.
    Each plot highlights and annotates the minimum and maximum values.
    """
    metrics = [
        ("gd", "GD (Gloss Duration)"),
        ("igt", "IGT (Into-Gloss Time)"),
        ("ogt", "OGT (Out-of-Gloss Time)"),
        ("tgt", "TGT (Total Gloss Time)")
    ]
    
    for metric, label in metrics:
        avg_col = f"avg_{metric}"
        median_col = f"median_{metric}"
        
        # --------------------------
        # Plot for Average Values
        # --------------------------
        sorted_avg_df = agg.sort_values(by=avg_col).reset_index(drop=True)
        x_avg = range(len(sorted_avg_df))
        plt.figure(figsize=(12, 6))
        plt.plot(x_avg, sorted_avg_df[avg_col], marker='o', linestyle='-', color='blue', label='Average')
        
        # Highlight min and max average values.
        min_idx_avg = sorted_avg_df[avg_col].idxmin()
        max_idx_avg = sorted_avg_df[avg_col].idxmax()
        plt.scatter([min_idx_avg], [sorted_avg_df.loc[min_idx_avg, avg_col]], color='red', s=100, zorder=5, label='Min Avg')
        plt.scatter([max_idx_avg], [sorted_avg_df.loc[max_idx_avg, avg_col]], color='green', s=100, zorder=5, label='Max Avg')
        
        min_gloss_avg = escape_gloss(sorted_avg_df.loc[min_idx_avg, 'gloss'])
        max_gloss_avg = escape_gloss(sorted_avg_df.loc[max_idx_avg, 'gloss'])
        
        plt.annotate(f"{min_gloss_avg} ({sorted_avg_df.loc[min_idx_avg, avg_col]:.1f})",
                     (min_idx_avg, sorted_avg_df.loc[min_idx_avg, avg_col]),
                     textcoords="offset points", xytext=(0, -15), ha='center', color='red')
        plt.annotate(f"{max_gloss_avg} ({sorted_avg_df.loc[max_idx_avg, avg_col]:.1f})",
                     (max_idx_avg, sorted_avg_df.loc[max_idx_avg, avg_col]),
                     textcoords="offset points", xytext=(0, 10), ha='center', color='green')
        
        plt.title(f"{label}: Average per Gloss")
        plt.xlabel("Gloss (sorted by average value)")
        plt.ylabel("Time (ms)")
        plt.legend()
        plt.tight_layout()
        avg_filename = os.path.join(output_dir, f"{metric}_aggregated_avg_plot.png")
        plt.savefig(avg_filename)
        print(f"Saved aggregated average plot for {label} to {avg_filename}")
        plt.close()
        
        # --------------------------
        # Plot for Median Values
        # --------------------------
        sorted_median_df = agg.sort_values(by=median_col).reset_index(drop=True)
        x_median = range(len(sorted_median_df))
        plt.figure(figsize=(12, 6))
        plt.plot(x_median, sorted_median_df[median_col], marker='s', linestyle='-', color='orange', label='Median')
        
        # Highlight min and max median values.
        min_idx_median = sorted_median_df[median_col].idxmin()
        max_idx_median = sorted_median_df[median_col].idxmax()
        plt.scatter([min_idx_median], [sorted_median_df.loc[min_idx_median, median_col]], color='red', s=100, zorder=5, label='Min Median')
        plt.scatter([max_idx_median], [sorted_median_df.loc[max_idx_median, median_col]], color='green', s=100, zorder=5, label='Max Median')
        
        min_gloss_median = escape_gloss(sorted_median_df.loc[min_idx_median, 'gloss'])
        max_gloss_median = escape_gloss(sorted_median_df.loc[max_idx_median, 'gloss'])
        
        plt.annotate(f"{min_gloss_median} ({sorted_median_df.loc[min_idx_median, median_col]:.1f})",
                     (min_idx_median, sorted_median_df.loc[min_idx_median, median_col]),
                     textcoords="offset points", xytext=(0, -15), ha='center', color='red')
        plt.annotate(f"{max_gloss_median} ({sorted_median_df.loc[max_idx_median, median_col]:.1f})",
                     (max_idx_median, sorted_median_df.loc[max_idx_median, median_col]),
                     textcoords="offset points", xytext=(0, 10), ha='center', color='green')
        
        plt.title(f"{label}: Median per Gloss")
        plt.xlabel("Gloss (sorted by median value)")
        plt.ylabel("Time (ms)")
        plt.legend()
        plt.tight_layout()
        median_filename = os.path.join(output_dir, f"{metric}_aggregated_median_plot.png")
        plt.savefig(median_filename)
        print(f"Saved aggregated median plot for {label} to {median_filename}")
        plt.close()

def plot_raw_metrics(raw_df, output_dir):
    """
    For each metric (GD, IGT, OGT, TGT), create a box plot overlaid with a strip plot
    to show the distribution of raw values. Horizontal lines mark the minimum and maximum values.
    """
    metrics = [
        ("gd", "GD (Gloss Duration)"),
        ("igt", "IGT (Into-Gloss Time)"),
        ("ogt", "OGT (Out-of-Gloss Time)"),
        ("tgt", "TGT (Total Gloss Time)")
    ]
    
    print("Generating raw metric plots...")
    for metric, label in tqdm(metrics, desc="Processing metrics"):
        plt.figure(figsize=(10, 6))
        
        # Box plot (fast)
        sns.boxplot(y=raw_df[metric], color='lightblue')

        # Strip plot (much faster than swarm plot)
        sns.stripplot(y=raw_df[metric], color='gray', alpha=0.6, jitter=True)

        # Draw horizontal lines for min and max values.
        min_val = raw_df[metric].min()
        max_val = raw_df[metric].max()
        plt.axhline(min_val, color='red', linestyle='--', label=f'Min: {min_val:.1f} ms')
        plt.axhline(max_val, color='green', linestyle='--', label=f'Max: {max_val:.1f} ms')
        
        plt.title(f"{label}: Distribution of Raw Values")
        plt.ylabel("Time (ms)")
        plt.legend()
        plt.tight_layout()
        
        filename = os.path.join(output_dir, f"{metric}_raw_distribution.png")
        plt.savefig(filename)
        print(f"Saved raw distribution plot for {label} to {filename}")
        plt.close()
        
def main():
    # Define the base path and CSV file locations.
    base_path = "/Volumes/IISY/DGSKorpus/dgs-gloss-times"
    eval_csv = os.path.join(base_path, "evaluated_gloss_metrics.csv")
    raw_csv = os.path.join(base_path, "raw_gloss_metrics.csv")
    
    # Read in the aggregated and raw metrics CSV files.
    agg = pd.read_csv(eval_csv)
    raw_df = pd.read_csv(raw_csv)
    
    # Define an output directory for the generated plots.
    output_dir = os.path.join(base_path, "plots")
    os.makedirs(output_dir, exist_ok=True)
    
    # Set a pleasant plotting style.
    sns.set(style="whitegrid")
    
    # Generate plots based on aggregated metrics.
    plot_aggregated_metrics(agg, output_dir)
    
    # Generate plots based on raw metric distributions.
    plot_raw_metrics(raw_df, output_dir)
    
    print("All plots created successfully.")

if __name__ == "__main__":
    main()
