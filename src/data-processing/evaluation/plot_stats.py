import csv
import matplotlib.pyplot as plt
import os
import numpy as np

# List of all datasets to process
DATASETS = ["DGS-BT2", "DGS-OG", "PHOENIX"]
BASE_STATS_DIR = "./dataset-stats"
BASE_PLOTS_DIR = "./dataset-plots"


def load_sentence_length_data(csv_path):
    """
    Load sentence lengths and their frequencies from a CSV file.
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


def plot_sentence_lengths(lengths, counts, title, output_path=None):
    """
    Plot a bar chart of sentence length distribution with annotations.
    If output_path is provided, saves the figure; otherwise displays it.
    """
    plt.figure(figsize=(10, 6))
    bars = plt.bar(lengths, counts)
    annotate_bars(bars)

    plt.xlabel("Sentence Length (words)")
    plt.ylabel("Number of Sentences")
    plt.title(title)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()

    if output_path:
        os.makedirs(output_path, exist_ok=True)
        filename = os.path.join(output_path, "sentence-length-distribution.png")
        plt.savefig(filename)
        plt.close()
        print(f"Plot saved to {filename}")
    else:
        plt.show()


def main():
    # Store all data for combined plot
    all_data = {}
    all_lengths = set()

    # Generate individual plots
    for ds in DATASETS:
        stats_dir = os.path.join(BASE_STATS_DIR, ds)
        plots_dir = os.path.join(BASE_PLOTS_DIR, ds)

        # Load
        csv_path = os.path.join(stats_dir, "sentence_lengths.csv")
        lengths, counts = load_sentence_length_data(csv_path)

        # Save for combined
        all_data[ds] = dict(zip(lengths, counts))
        all_lengths.update(lengths)

        # Plot individual
        title = f"Sentence Length Distribution - {ds}"
        plot_sentence_lengths(lengths, counts, title, plots_dir)

    # Prepare combined plot
    combined_dir = os.path.join(BASE_PLOTS_DIR, "combined")
    os.makedirs(combined_dir, exist_ok=True)

    # Sorted unique lengths
    sorted_lengths = sorted(all_lengths)
    x = np.arange(len(sorted_lengths))
    width = 0.8 / len(DATASETS)

    plt.figure(figsize=(12, 7))

    # Plot each dataset and annotate
    for i, ds in enumerate(DATASETS):
        counts = [all_data[ds].get(l, 0) for l in sorted_lengths]
        bars = plt.bar(x + i * width, counts, width=width, label=ds)
        annotate_bars(bars)

    plt.xlabel("Sentence Length (words)")
    plt.ylabel("Number of Sentences")
    plt.title("Sentence Length Distribution - All Datasets")
    plt.xticks(x + width * (len(DATASETS) - 1) / 2, sorted_lengths, rotation=45)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()

    # Save combined
    combined_path = os.path.join(combined_dir, "sentence-length-distribution.png")
    plt.savefig(combined_path)
    plt.close()
    print(f"Combined plot saved to {combined_path}")


if __name__ == "__main__":
    main()
