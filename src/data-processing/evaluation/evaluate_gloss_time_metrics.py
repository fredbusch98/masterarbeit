#!/usr/bin/env python3
import os
import pandas as pd

def read_raw_metrics(file_path):
    """
    Read the raw metrics CSV using pandas.
    """
    df = pd.read_csv(file_path)
    return df

def aggregate_metrics(df):
    """
    Group by gloss and compute average and median metrics and counts.
    """
    agg = df.groupby("gloss").agg(
        avg_gd=("gd", "mean"),
        median_gd=("gd", "median"),  # Added median
        avg_igt=("igt", "mean"),
        median_igt=("igt", "median"),  # Added median
        avg_ogt=("ogt", "mean"),
        median_ogt=("ogt", "median"),  # Added median
        avg_tgt=("tgt", "mean"),
        median_tgt=("tgt", "median"),  # Added median
        count=("gloss", "count")
    ).reset_index()
    return agg

def print_top_aggregated(metric, agg, label):
    """
    Print the top-5 lowest and highest aggregated averages per gloss, including medians.
    
    Parameters:
    - metric (str): The average metric to sort by (e.g., "avg_gd").
    - agg (DataFrame): Aggregated DataFrame with average, median, and count columns.
    - label (str): Descriptive label for the metric (e.g., "GD (Gloss Duration)").
    """
    sorted_df = agg.sort_values(by=metric)
    metric_name = metric.split('_')[1]  # Extract base metric name, e.g., "gd" from "avg_gd"
    median_metric = f"median_{metric_name}"  # Corresponding median column, e.g., "median_gd"
    
    print(f"\nTop 5 lowest {label} averages (with medians):")
    for _, row in sorted_df.head(5).iterrows():
        print(f"  {row['gloss']}: Avg = {row[metric]:.2f} ms, Median = {row[median_metric]:.2f} ms (n={int(row['count'])})")
    
    print(f"\nTop 5 highest {label} averages (with medians):")
    for _, row in sorted_df.tail(5).sort_values(by=metric, ascending=False).iterrows():
        print(f"  {row['gloss']}: Avg = {row[metric]:.2f} ms, Median = {row[median_metric]:.2f} ms (n={int(row['count'])})")

def print_top_occurrences(metric, label, index_field, df, n=5):
    """
    Print the top-n lowest and highest single-occurrence values of a metric,
    including the extra details (entry, index string, and speaker).
    """
    sorted_df = df.sort_values(by=metric)
    print(f"\nTop {n} lowest {label} occurrences:")
    for _, row in sorted_df.head(n).iterrows():
        print(f"  {row['gloss']}: {row[metric]:.0f} ms (entry: {row['entry']}, index: {row[index_field]}, {row['speaker']})")
    print(f"\nTop {n} highest {label} occurrences:")
    for _, row in sorted_df.tail(n).sort_values(by=metric, ascending=False).iterrows():
        print(f"  {row['gloss']}: {row[metric]:.0f} ms (entry: {row['entry']}, index: {row[index_field]}, {row['speaker']})")

import os

def find_and_save_metrics_threshold(df, metric, threshold, above=True, output_dir="/Volumes/IISY/DGSKorpus/dgs-gloss-times"):
    """
    Save all occurrences of a metric (e.g., "gd") that are above or below a given threshold to a CSV file.
    Results are ordered by the time metric value. Prints only the number of occurrences found.
    """
    os.makedirs(output_dir, exist_ok=True)
    if above:
        filtered = df[df[metric] > threshold]
        condition = "above"
    else:
        filtered = df[df[metric] < threshold]
        condition = "below"

    num_results = len(filtered)
    print(f"\nFound {num_results} occurrences of {metric.upper()} {condition} {threshold} ms.")

    if num_results > 0:
        # Order the results by the metric value (ascending order)
        filtered = filtered.sort_values(by=metric, ascending=False)
        
        index_field = f"{metric}_index" if metric in ["igt", "ogt", "tgt"] else "block_index"
        filtered = filtered[["gloss", "entry", index_field, "speaker", metric]]
        
        filename = f"{metric}_{condition}_{threshold}ms.csv"
        file_path = os.path.join(output_dir, filename)
        filtered.to_csv(file_path, index=False)
        print(f"Results saved to {file_path}")

def extract_filtered_columns(input_file, output_file):
    """
    Read the evaluated metrics CSV, extract only the specified columns,
    and save the result to a new CSV file.
    """
    df = pd.read_csv(input_file)
    filtered_df = df[['gloss', 'median_igt', 'median_ogt']]
    filtered_df.to_csv(output_file, index=False)
    print(f"Filtered CSV written to {output_file}")

def main():
    base_path = "/Volumes/IISY/DGSKorpus/"
    raw_csv = os.path.join(base_path, "dgs-gloss-times", "raw_gloss_metrics.csv")
    df = read_raw_metrics(raw_csv)
    
    # Compute overall averages and medians
    overall_avg_gd = df["gd"].mean()
    overall_median_gd = df["gd"].median()  # Added median
    overall_avg_igt = df["igt"].mean()
    overall_median_igt = df["igt"].median()  # Added median
    overall_avg_ogt = df["ogt"].mean()
    overall_median_ogt = df["ogt"].median()  # Added median
    overall_avg_tgt = df["tgt"].mean()
    overall_median_tgt = df["tgt"].median()  # Added median
    
    print("Overall Statistics:")
    print(f"  GD: Avg = {overall_avg_gd:.2f} ms, Median = {overall_median_gd:.2f} ms")
    print(f"  IGT: Avg = {overall_avg_igt:.2f} ms, Median = {overall_median_igt:.2f} ms")
    print(f"  OGT: Avg = {overall_avg_ogt:.2f} ms, Median = {overall_median_ogt:.2f} ms")
    print(f"  TGT: Avg = {overall_avg_tgt:.2f} ms, Median = {overall_median_tgt:.2f} ms")
    
    # Aggregate per-gloss averages, medians and counts
    agg = aggregate_metrics(df)
    
    # Write the aggregated evaluation summary to CSV
    eval_csv = os.path.join(base_path, "dgs-gloss-times", "evaluated_gloss_metrics.csv")
    agg.to_csv(eval_csv, index=False)
    print(f"\nEvaluation metrics written to {eval_csv}")

    # Extract and save filtered columns (gloss, median_igt, median_ogt)
    filtered_csv = os.path.join(base_path, "dgs-gloss-times", "evaluated_gloss_metrics_filtered.csv")
    extract_filtered_columns(eval_csv, filtered_csv)
    
    # Print top-5 aggregated averages per gloss
    print_top_aggregated("avg_gd", agg, "GD (Gloss Duration)")
    print_top_aggregated("avg_igt", agg, "IGT (Into-Gloss Time)")
    print_top_aggregated("avg_ogt", agg, "OGT (Out-of-Gloss Time)")
    print_top_aggregated("avg_tgt", agg, "TGT (Total Gloss Time)")
    
    # Print top-5 single-occurrence metrics with full details
    print_top_occurrences("igt", "IGT (Into-Gloss Time)", "igt_index", df, n=5)
    print_top_occurrences("ogt", "OGT (Out-of-Gloss Time)", "ogt_index", df, n=5)
    print_top_occurrences("gd", "GD (Gloss Duration)", "block_index", df, n=5)
    print_top_occurrences("tgt", "TGT (Total Gloss Time)", "tgt_index", df, n=5)
    
    # Example usage: print all GD occurrences above 2000 ms
    find_and_save_metrics_threshold(df, "gd", 2000, above=True)
    find_and_save_metrics_threshold(df, "gd", 1000, above=True)

if __name__ == "__main__":
    main()