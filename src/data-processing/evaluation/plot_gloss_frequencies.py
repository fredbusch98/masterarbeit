"""
Generates a combined horizontal boxplot with jittered scatter points showing gloss occurrence distributions
for multiple datasets on a log scale. Saves the plot with customized colors, median, and interquartile range highlights. 
Designed for quick comparison of gloss frequency across datasets.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# Dataset info and colors
datasets = {
    "og-preprocessed": {
        "path": "./dataset-stats/og-preprocessed/gloss_frequencies.csv",
        "color": "orange"
    },
    "bt-2": {
        "path": "./dataset-stats/bt-2/gloss_frequencies.csv",
        "color": "blue"
    },
    "phoenix": {
        "path": "./dataset-stats/phoenix/gloss_frequencies.csv",
        "color": "green"
    }
}

output_dir = "./dataset-plots/gloss_distribution_plots"
os.makedirs(output_dir, exist_ok=True)

plt.figure(figsize=(10, 6))
ax = plt.gca()

# We will assign a separate y-position for each dataset row
dataset_names = list(datasets.keys())
n_datasets = len(dataset_names)

for i, dataset_name in enumerate(dataset_names):
    info = datasets[dataset_name]
    df = pd.read_csv(info["path"])
    if 'count' not in df.columns:
        raise ValueError(f"'count' column not found in {info['path']}")

    gloss_counts = df['count'].astype(int)

    # Calculate stats for boxplot
    q1 = np.percentile(gloss_counts, 25)
    median = np.median(gloss_counts)
    q3 = np.percentile(gloss_counts, 75)
    iqr = q3 - q1
    whisker_low = max(min(gloss_counts), q1 - 1.5*iqr)
    whisker_high = min(max(gloss_counts), q3 + 1.5*iqr)

    # Plot boxplot at y=i+1 horizontally
    boxprops = dict(facecolor='lightgrey', edgecolor='black', alpha=0.6)
    bp = ax.boxplot(gloss_counts, positions=[i + 1], widths=0.6, vert=False, patch_artist=True,
                    showfliers=False, boxprops=boxprops, medianprops=dict(color='red'))

    # Set box color
    for patch in bp['boxes']:
        patch.set_facecolor('lightgrey')

    # Scatter plot with vertical jitter around the y=i+1 position
    y_jitter = np.random.uniform(low=i + 0.85, high=i + 1.15, size=len(gloss_counts))
    ax.scatter(gloss_counts, y_jitter, color=info["color"], alpha=0.6, edgecolors='k', s=40, label=dataset_name)

# Set axis labels and formatting
ax.set_yticks(np.arange(1, n_datasets + 1))
ax.set_yticklabels(dataset_names)
ax.set_xscale('log')
ax.set_xlabel('Gloss Occurrence Count (log scale)')
ax.set_title('Gloss Occurrence Distribution Across Datasets')
ax.grid(axis='x', linestyle='--', alpha=0.7)

# Custom legend handles
custom_handles = [
    Patch(facecolor='lightgrey', edgecolor='black', label='Q25–Q75)'),
    Line2D([0], [0], color='red', lw=2, label='Median')
]

# Get scatter handles/labels and combine
scatter_handles, scatter_labels = ax.get_legend_handles_labels()
all_handles = custom_handles + scatter_handles
all_labels = [h.get_label() for h in all_handles]

# Add legend
ax.legend(all_handles, all_labels)

plt.tight_layout()
output_path = os.path.join(output_dir, "combined_gloss_occurrence_boxplot_log.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Saved combined plot to {output_path}")
