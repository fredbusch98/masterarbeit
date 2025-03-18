#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def plot_top_glosses(counts_csv, output_dir):
    """
    Plot a horizontal bar chart of the top 20 most frequent glosses.
    """
    df = pd.read_csv(counts_csv)
    # Sort glosses by occurrence count in descending order
    df_sorted = df.sort_values(by="count", ascending=False)
    top_n = 20
    top_df = df_sorted.head(top_n)
    
    plt.figure(figsize=(12, 8))
    # Assign gloss as hue to map the palette; dodge=False to ensure one bar per gloss.
    barplot = sns.barplot(x="count", y="gloss", data=top_df, hue="gloss",
                           palette="viridis", dodge=False)
    
    # Remove the legend if it exists.
    legend = barplot.get_legend()
    if legend is not None:
        legend.remove()
    
    # Annotate each bar with its count.
    for index, row in top_df.iterrows():
        barplot.text(row["count"] + 5, index, f'{row["count"]}', color='black', va="center")
    
    plt.title("Top 20 Most Frequent Glosses")
    plt.xlabel("Occurrence Count")
    plt.ylabel("Gloss")
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "top_20_glosses.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Saved top 20 glosses plot to: {output_path}")

def plot_gloss_distribution(counts_csv, output_dir):
    """
    Plot a histogram (with KDE) showing the distribution of gloss occurrence counts,
    and include an inset focusing on the region where most glosses occur.
    """
    df = pd.read_csv(counts_csv)
    
    # Create figure and main axis.
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df["count"], bins=50, kde=True, color="skyblue", ax=ax)
    ax.set_title("Distribution of Gloss Occurrence Counts")
    ax.set_xlabel("Occurrence Count")
    ax.set_ylabel("Number of Glosses")
    
    # Create an inset axis for a zoomed-in view (e.g., 0-1000)
    axins = inset_axes(ax, width="40%", height="40%", loc='upper right')
    sns.histplot(df["count"], bins=50, kde=True, color="skyblue", ax=axins)
    axins.set_xlim(0, 1000)
    axins.set_title("Zoom: 0-1000", fontsize=10)
    axins.tick_params(axis='both', which='major', labelsize=8)
    
    # Instead of tight_layout (which causes issues with inset axes), manually adjust subplot margins.
    fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    
    output_path = os.path.join(output_dir, "gloss_count_distribution.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Saved gloss count distribution plot to: {output_path}")

def plot_lost_gloss_details(lost_csv, output_dir):
    """
    Create a pie chart showing the proportion of glosses that are only lost.
    """
    df = pd.read_csv(lost_csv)
    # Ensure "Only Lost" is interpreted as boolean
    df["Only Lost"] = df["Only Lost"].astype(bool)
    
    # Pie chart: Proportion of glosses that are only lost vs not
    only_lost_count = df["Only Lost"].sum()
    not_only_lost_count = len(df) - only_lost_count
    labels = ["Only Lost", "Not Only Lost"]
    sizes = [only_lost_count, not_only_lost_count]
    colors = ["red", "blue"]
    
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=140)
    plt.title("Proportion of Lost Glosses: Only Lost vs Not Only Lost")
    plt.tight_layout()
    
    output_path_pie = os.path.join(output_dir, "lost_gloss_pie.png")
    plt.savefig(output_path_pie)
    plt.close()
    print(f"Saved lost gloss pie chart to: {output_path_pie}")

def plot_overall_lost_vs_notlost(output_dir):
    """
    Create a pie chart comparing not-lost glosses vs. lost glosses using provided numbers:
      - Total lost glosses found: 63,994
      - Not-lost glosses: 294,093
    """
    lost_glosses = 63994
    not_lost_glosses = 294093
    labels = ["Lost Glosses", "Not Lost Glosses"]
    sizes = [lost_glosses, not_lost_glosses]
    colors = ["red", "green"]
    
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=140)
    plt.title("Overall Lost vs. Not Lost Glosses")
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "overall_lost_vs_notlost.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Saved overall lost vs not lost pie chart to: {output_path}")

def main():
    # Define the main folder path as used previously.
    main_folder = "/Volumes/IISY/DGSKorpus"
    counts_csv = os.path.join(main_folder, "all-gloss-counts-from-transcripts.csv")
    lost_csv = os.path.join(main_folder, "lost-gloss-details.csv")
    
    # Define an output directory for the generated plots.
    output_dir = os.path.join(main_folder, "plots_gloss_metrics")
    os.makedirs(output_dir, exist_ok=True)
    
    # Set a visually appealing style.
    sns.set(style="whitegrid", palette="muted", font_scale=1.2)
    
    print("Generating gloss frequency plots...\n")
    
    plot_top_glosses(counts_csv, output_dir)
    plot_gloss_distribution(counts_csv, output_dir)
    plot_lost_gloss_details(lost_csv, output_dir)
    plot_overall_lost_vs_notlost(output_dir)
    
    print("\nAll gloss metrics plots created successfully!")

if __name__ == "__main__":
    main()
