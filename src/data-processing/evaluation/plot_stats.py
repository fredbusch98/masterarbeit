"""
Generates bar charts of sentence and gloss sequence length distributions for multiple datasets.
Creates individual plots per dataset and combined plots (regular and log-scale) with consistent x-axes.
Saves all figures to structured output directories for easy comparison.
"""
import csv
import matplotlib.pyplot as plt
import os
import numpy as np

# List of all datasets to process
DATASETS = ["bt-2", "og-preprocessed", "phoenix"]
BASE_STATS_DIR = "./dataset-stats"
BASE_PLOTS_DIR = "./dataset-plots"

# Fixed color for each dataset
DATASET_COLORS = {
    "bt-2": "tab:blue",
    "og-preprocessed": "tab:orange",
    "phoenix": "tab:green"
}

# List of CSV metrics to plot: (filename, x-axis label, output filename)
METRICS = [
    ("sentence_lengths.csv", "Sentence Length (words)", "sentence-length-distribution.png"),
    ("gloss_sequence_lengths.csv", "Gloss Sequence Length (glosses)", "gloss-sequence-length-distribution.png")
]

def load_length_data(csv_path):
    """
    Load lengths and their frequencies from a CSV file.
    Expects columns: length, count
    """
    lengths = []
    counts = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            lengths.append(int(row['length']))
            counts.append(int(row['count']))
    return lengths, counts


def annotate_bars(bars):
    """
    Annotate each bar with its height value.
    """
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f'{int(height)}',
            ha='center',
            va='bottom',
            fontsize=8
        )


def plot_distribution(lengths, counts, x_label, title, output_path, color=None, annotate=True, log_scale=False, x_ticks=None):
    """
    Plot a bar chart of length distribution with optional annotations and logarithmic scale.
    """
    plt.figure(figsize=(10, 6))
    bars = plt.bar(lengths, counts, color=color)

    if annotate and not log_scale:
        annotate_bars(bars)

    plt.xlabel(x_label)
    plt.ylabel("Number of Sentences")
    plt.title(title + (" (logarithmic scale)" if log_scale else ""))
    if log_scale:
        plt.yscale('log')
    if x_ticks is not None:
        plt.xticks(x_ticks)
        plt.xlim(min(x_ticks) - 0.5, max(x_ticks) + 0.5)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path)
        plt.close()
        print(f"Plot saved to {output_path}")
    else:
        plt.show()


def main():
    # Collect combined data for each metric
    combined_data = {metric[0]: {} for metric in METRICS}
    all_lengths = {metric[0]: set() for metric in METRICS}

    # Load all data first and record all unique lengths
    for ds in DATASETS:
        stats_dir = os.path.join(BASE_STATS_DIR, ds)

        for csv_file, x_label, output_filename in METRICS:
            csv_path = os.path.join(stats_dir, csv_file)
            lengths, counts = load_length_data(csv_path)

            # Save for combined
            combined_data[csv_file][ds] = dict(zip(lengths, counts))
            all_lengths[csv_file].update(lengths)

    # Determine global sorted lengths for consistent x-axis
    sorted_lengths_per_metric = {
        csv_file: sorted(all_lengths[csv_file]) for csv_file in all_lengths
    }

    # Generate individual plots with consistent x-axis
    for ds in DATASETS:
        stats_dir = os.path.join(BASE_STATS_DIR, ds)
        plots_dir = os.path.join(BASE_PLOTS_DIR, ds)

        for csv_file, x_label, output_filename in METRICS:
            csv_path = os.path.join(stats_dir, csv_file)
            data_dict = combined_data[csv_file][ds]
            sorted_lengths = sorted_lengths_per_metric[csv_file]

            # Ensure all lengths are represented (fill missing with 0)
            counts = [data_dict.get(l, 0) for l in sorted_lengths]

            # Plot individual with consistent x-axis
            title = f"{x_label} Distribution - {ds}"
            out_path = os.path.join(plots_dir, output_filename)
            color = DATASET_COLORS[ds]
            plot_distribution(sorted_lengths, counts, x_label, title, out_path, color=color, x_ticks=sorted_lengths)

    # Generate combined plots (regular and log-scale)
    for csv_file, x_label, output_filename in METRICS:
        combined_dir = os.path.join(BASE_PLOTS_DIR, "combined")
        os.makedirs(combined_dir, exist_ok=True)

        sorted_lengths = sorted_lengths_per_metric[csv_file]

        # Prepare x positions for grouped bars
        x = np.arange(len(sorted_lengths))
        width = 0.8 / len(DATASETS)

        # --- Regular combined (no annotations) ---
        out_standard = os.path.join(combined_dir, output_filename)
        plt.figure(figsize=(12, 7))
        for i, ds in enumerate(DATASETS):
            counts = [combined_data[csv_file][ds].get(l, 0) for l in sorted_lengths]
            plt.bar(x + i * width, counts, width=width, label=ds, color=DATASET_COLORS[ds])
        plt.xlabel(x_label)
        plt.ylabel("Number of Sentences")
        plt.title(f"{x_label} Distribution - All Datasets")
        plt.xticks(x + width * (len(DATASETS) - 1) / 2, sorted_lengths, rotation=45)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(out_standard)
        plt.close()
        print(f"Combined plot saved to {out_standard}")

        # --- Log-scale combined (no annotations) ---
        log_filename = output_filename.replace('.png', '-log.png')
        out_log = os.path.join(combined_dir, log_filename)
        plt.figure(figsize=(12, 7))
        for i, ds in enumerate(DATASETS):
            counts = [combined_data[csv_file][ds].get(l, 0) for l in sorted_lengths]
            plt.bar(x + i * width, counts, width=width, label=ds, color=DATASET_COLORS[ds])
        plt.xlabel(x_label)
        plt.ylabel("Number of Sentences")
        plt.title(f"{x_label} Distribution - All Datasets (logarithmic scale)")
        plt.yscale('log')
        plt.xticks(x + width * (len(DATASETS) - 1) / 2, sorted_lengths, rotation=45)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(out_log)
        plt.close()
        print(f"Log-scale combined plot saved to {out_log}")


if __name__ == "__main__":
    main()
