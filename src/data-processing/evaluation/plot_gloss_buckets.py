"""
Creates a log-scaled bar chart showing the distribution of unique glosses across different 
frequency buckets for multiple datasets, converting percentage data into actual counts. 
Saves the chart as a PNG file.
"""
import pandas as pd
import matplotlib.pyplot as plt

# Total unique glosses per dataset
total_unique = {
    "og-preprocessed": 9858,
    "bt-2": 9858,
    "phoenix": 1115
}

# Percentage data for frequency buckets (excluding avg/median)
percent_data = {
    "Category": [
        "> 1000", "100–1000", "< 100",
        "≤ Avg", "≤ Median", "≤ 10", "≤ 2", "= 1"
    ],
    "og-preprocessed": [0.23, 4.20, 95.57, 85.73, 54.83, 69.64, 34.64, 22.19],
    "bt-2": [0.93, 9.76, 89.31, 85.34, 51.10, 49.03, 11.67, 1.82],
    "phoenix": [0.81, 11.12, 88.06, 84.62, 51.02, 66.85, 43.42, 31.84]
}

# Create DataFrame
df_percent = pd.DataFrame(percent_data).set_index("Category")

# Convert percentages to counts
df_counts = df_percent.copy()
for col in df_counts.columns:
    df_counts[col] = (df_counts[col] / 100 * total_unique[col]).round().astype(int)

# Plotting
fig, ax = plt.subplots(figsize=(12, 6))
df_counts.T.plot(kind='bar', ax=ax)

# Set y-axis to log scale
ax.set_yscale('log')

ax.set_ylabel("Gloss Count (log scale)")
ax.set_xticklabels(df_counts.T.index, rotation=0)
ax.legend(title="Occurrence Bucket", bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Add long title (inside plot bounds)
fig.suptitle(
    "Distribution of Unique Glosses by Total Occurrence Frequency Across Datasets",
    fontsize=14
)

# Adjust layout and save (including title)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("gloss_occurrence_bar_chart_log_scale.png", dpi=300, bbox_inches='tight')
