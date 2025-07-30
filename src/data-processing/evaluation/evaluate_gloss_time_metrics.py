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
    Group by gloss and compute average, median, standard deviation metrics and counts.
    """
    agg = df.groupby("gloss").agg(
        avg_gd=("gd", "mean"),
        median_gd=("gd", "median"),
        std_gd=("gd", "std"),
        avg_igt=("igt", "mean"),
        median_igt=("igt", "median"),
        std_igt=("igt", "std"),
        avg_ogt=("ogt", "mean"),
        median_ogt=("ogt", "median"),
        std_ogt=("ogt", "std"),
        avg_tgt=("tgt", "mean"),
        median_tgt=("tgt", "median"),
        std_tgt=("tgt", "std"),
        count=("gloss", "count")
    ).reset_index()
    return agg

def print_top_aggregated(metric, agg, label):
    """
    Print the top-5 lowest and highest aggregated averages per gloss, including medians.
    """
    sorted_df = agg.sort_values(by=metric)
    metric_name = metric.split('_')[1]
    median_metric = f"median_{metric_name}"
    
    print(f"\nTop 10 lowest {label} averages (with medians):")
    for _, row in sorted_df.head(10).iterrows():
        print(f"  {row['gloss']}: Avg = {row[metric]:.2f} ms, Median = {row[median_metric]:.2f} ms (occurrence={int(row['count'])})")
    
    print(f"\nTop 10 highest {label} averages (with medians):")
    for _, row in sorted_df.tail(10).sort_values(by=metric, ascending=False).iterrows():
        print(f"  {row['gloss']}: Avg = {row[metric]:.2f} ms, Median = {row[median_metric]:.2f} ms (occurrence={int(row['count'])})")

def print_top_occurrences(metric, label, index_field, df, n=10):
    """
    Print the top-n lowest and highest single-occurrence values of a metric.
    """
    sorted_df = df.sort_values(by=metric)
    print(f"\nTop {n} lowest {label} occurrences:")
    for _, row in sorted_df.head(n).iterrows():
        print(f"  {row['gloss']}: {row[metric]:.0f} ms (entry: {row['entry']}, index: {row[index_field]}, {row['speaker']})")
    print(f"\nTop {n} highest {label} occurrences:")
    for _, row in sorted_df.tail(n).sort_values(by=metric, ascending=False).iterrows():
        print(f"  {row['gloss']}: {row[metric]:.0f} ms (entry: {row['entry']}, index: {row[index_field]}, {row['speaker']})")

def find_and_save_metrics_threshold(df, metric, threshold, above=True, output_dir="/Volumes/IISY/DGSKorpus/dgs-gloss-times"):
    """
    Save all occurrences of a metric that are above or below a given threshold to a CSV.
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
        filtered = filtered.sort_values(by=metric, ascending=False)
        index_field = f"{metric}_index" if metric in ["igt", "ogt", "tgt"] else "block_index"
        filtered = filtered[["gloss", "entry", index_field, "speaker", metric]]
        filename = f"{metric}_{condition}_{threshold}ms.csv"
        file_path = os.path.join(output_dir, filename)
        filtered.to_csv(file_path, index=False)
        print(f"Results saved to {file_path}")

def extract_filtered_columns(input_file, output_file):
    """
    Extract specific columns from the evaluated metrics CSV and save to a new CSV.
    """
    df = pd.read_csv(input_file)
    filtered_df = df[['gloss', 'median_igt', 'median_ogt', 'median_gd']]
    filtered_df.to_csv(output_file, index=False)
    print(f"Filtered CSV written to {output_file}")

def save_per_occurrence_with_aggregated_stats(df, output_path):
    """
    Save a CSV where each original row includes the average and median values of its gloss group.
    """
    gloss_stats = df.groupby("gloss").agg(
        avg_gd=("gd", "mean"),
        median_gd=("gd", "median"),
        avg_igt=("igt", "mean"),
        median_igt=("igt", "median"),
        avg_ogt=("ogt", "mean"),
        median_ogt=("ogt", "median"),
        avg_tgt=("tgt", "mean"),
        median_tgt=("tgt", "median"),
    ).reset_index()
    
    df_with_stats = df.merge(gloss_stats, on="gloss", how="left")
    df_with_stats.to_csv(output_path, index=False)
    print(f"Per-occurrence data with gloss-level averages/medians written to {output_path}")

def main():
    base_path = "/Volumes/IISY/DGSKorpus/"
    raw_csv = os.path.join(base_path, "dgs-gloss-times", "raw_gloss_metrics.csv")
    df = read_raw_metrics(raw_csv)
    
    # Overall statistics
    print("Overall Statistics:")
    for metric in ["gd", "igt", "ogt", "tgt"]:
        avg = df[metric].mean()
        med = df[metric].median()
        std = df[metric].std()
        q25 = df[metric].quantile(0.25)
        q75 = df[metric].quantile(0.75)
        print(f"  {metric.upper()}: Avg = {avg:.2f} ms, Median = {med:.2f} ms, Std = {std:.2f} ms, Q25 = {q25:.2f} ms, Q75 = {q75:.2f} ms")

    # Aggregate per gloss
    agg = aggregate_metrics(df)

    # Save aggregated evaluation metrics
    eval_csv = os.path.join(base_path, "dgs-gloss-times", "evaluated_gloss_metrics.csv")
    agg.to_csv(eval_csv, index=False)
    print(f"\nEvaluation metrics written to {eval_csv}")

    eval_csv_with_std = os.path.join(base_path, "dgs-gloss-times", "evaluated_gloss_metrics_with_std.csv")
    agg.to_csv(eval_csv_with_std, index=False)
    print(f"Evaluation metrics (with std) written to {eval_csv_with_std}")

    # Save filtered columns
    filtered_csv = os.path.join(base_path, "dgs-gloss-times", "evaluated_gloss_metrics_filtered.csv")
    extract_filtered_columns(eval_csv, filtered_csv)

    # Save per-occurrence data with gloss-level aggregates
    detailed_csv = os.path.join(base_path, "dgs-gloss-times", "raw_with_gloss_aggregates.csv")
    save_per_occurrence_with_aggregated_stats(df, detailed_csv)

    # Print top gloss-level aggregates
    print_top_aggregated("avg_gd", agg, "GD (Gloss Duration)")
    print_top_aggregated("avg_igt", agg, "IGT (Into-Gloss Time)")
    print_top_aggregated("avg_ogt", agg, "OGT (Out-of-Gloss Time)")
    print_top_aggregated("avg_tgt", agg, "TGT (Total Gloss Time)")

    # Print top single-occurrence values
    print_top_occurrences("igt", "IGT (Into-Gloss Time)", "igt_index", df, n=10)
    print_top_occurrences("ogt", "OGT (Out-of-Gloss Time)", "ogt_index", df, n=10)
    print_top_occurrences("gd", "GD (Gloss Duration)", "block_index", df, n=10)
    print_top_occurrences("tgt", "TGT (Total Gloss Time)", "tgt_index", df, n=10)

    # Save threshold-based outliers
    find_and_save_metrics_threshold(df, "gd", 2000, above=True)
    find_and_save_metrics_threshold(df, "gd", 1000, above=True)

if __name__ == "__main__":
    main()
