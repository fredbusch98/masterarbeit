#!/usr/bin/env python3
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

# Provided statistics in milliseconds
metrics = {
    "GD": {"Avg": 248.51, "Median": 180.00, "Std": 217.10, "Q25": 120.00, "Q75": 300.00},
    "IGT": {"Avg": 168.99, "Median": 140.00, "Std": 159.67, "Q25": 60.00, "Q75": 240.00},
    "OGT": {"Avg": 172.31, "Median": 160.00, "Std": 157.14, "Q25": 80.00, "Q75": 240.00},
    "TGT": {"Avg": 589.80, "Median": 520.00, "Std": 323.95, "Q25": 380.00, "Q75": 720.00}
}

# Convert milliseconds to frames at 50 FPS (1 frame = 20 ms)
def ms_to_frames(ms):
    return ms * 50 / 1000.0

# Plot overall boxplot with Avg/Median/Std for metrics
def plot_metrics(metrics, convert_to_frames=False, save_name="metrics_boxplot.png"):
    fig, ax = plt.subplots(figsize=(8, 6))
    labels = []
    box_data = []

    for metric, values in metrics.items():
        if convert_to_frames:
            q1 = ms_to_frames(values["Q25"])
            median = ms_to_frames(values["Median"])
            q3 = ms_to_frames(values["Q75"])
            mean = ms_to_frames(values["Avg"])
            std = ms_to_frames(values["Std"])
        else:
            q1 = values["Q25"]
            median = values["Median"]
            q3 = values["Q75"]
            mean = values["Avg"]
            std = values["Std"]

        box_data.append({
            "q1": q1,
            "median": median,
            "q3": q3,
            "mean": mean,
            "std": std
        })
        labels.append(metric)

    positions = range(1, len(labels) + 1)

    for i, data in enumerate(box_data):
        # IQR bar
        ax.broken_barh([(data["q1"], data["q3"] - data["q1"])], (i + 0.75, 0.5),
                       facecolors='skyblue', edgecolors='black', label='Q25–Q75' if i == 0 else "")
        # Median (red dot)
        ax.plot(data["median"], i + 1, 'ro', label='Median' if i == 0 else "")
        # Mean (green dot)
        ax.plot(data["mean"], i + 1, 'go', label='Mean' if i == 0 else "")
        # Std (blue dot)
        ax.plot(data["std"], i + 1, 'bo', label='Std Dev' if i == 0 else "")

    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    unit = "Frames (50 FPS)" if convert_to_frames else "Milliseconds"
    ax.set_xlabel(unit)
    title_suffix = " (in Frames @50FPS)" if convert_to_frames else " (in Milliseconds)"
    ax.set_title("Overall Gloss Time Metrics" + title_suffix)
    ax.legend(loc="lower right")
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_name)
    print(f"Saved plot to {save_name}")
    plt.close()

# New: Plot per-gloss std dev distribution (boxplot + dots, no labels)
def plot_std_per_gloss(df, metric_base, output_dir):
    import matplotlib.patches as mpatches

    std_col = f"std_{metric_base}"

    # --- Plot in milliseconds ---
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[std_col], color='skyblue', showfliers=False)
    sns.stripplot(x=df[std_col], color='black', size=3, jitter=0.2)
    plt.title(f"{metric_base.upper()} Std Dev per Unique Gloss (ms)")
    plt.xlabel("Milliseconds")
    plt.yticks([])

    # Legend: Box = IQR, Dots = Gloss values
    box_patch = mpatches.Patch(color='skyblue', label='Q25–Q75')
    dot_patch = mpatches.Patch(color='black', label='Unique Gloss Std Dev')
    plt.legend(handles=[box_patch, dot_patch], loc='upper right')

    plt.tight_layout()
    ms_plot_path = os.path.join(output_dir, f"{metric_base}_std_per_gloss_ms.png")
    plt.savefig(ms_plot_path)
    print(f"Saved plot to {ms_plot_path}")
    plt.close()

    # --- Convert to frames ---
    df[f"{std_col}_frames"] = df[std_col] / 20.0

    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[f"{std_col}_frames"], color='skyblue', showfliers=False)
    sns.stripplot(x=df[f"{std_col}_frames"], color='black', size=3, jitter=0.2)
    plt.title(f"{metric_base.upper()} Std Dev per Unique Gloss (frames @50FPS)")
    plt.xlabel("Frames")
    plt.yticks([])

    # Legend: Box = IQR, Dots = Gloss values
    box_patch = mpatches.Patch(color='skyblue', label='Q25–Q75')
    dot_patch = mpatches.Patch(color='black', label='Unique Gloss Std Dev')
    plt.legend(handles=[box_patch, dot_patch], loc='upper right')

    plt.tight_layout()
    frames_plot_path = os.path.join(output_dir, f"{metric_base}_std_per_gloss_frames.png")
    plt.savefig(frames_plot_path)
    print(f"Saved plot to {frames_plot_path}")
    plt.close()

if __name__ == "__main__":
    # Plot overall statistics in ms and frames
    plot_metrics(metrics, convert_to_frames=False, save_name="metrics_boxplot_ms.png")
    plot_metrics(metrics, convert_to_frames=True, save_name="metrics_boxplot_frames.png")

    # New: Per-gloss std dev distribution plots
    base_path = "/Volumes/IISY/DGSKorpus/dgs-gloss-times"
    csv_file = os.path.join(base_path, "evaluated_gloss_metrics_with_std.csv")
    output_dir = os.path.join(base_path, "plots")
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_file)

    for metric in ["gd", "igt", "ogt", "tgt"]:
        plot_std_per_gloss(df, metric, output_dir)
