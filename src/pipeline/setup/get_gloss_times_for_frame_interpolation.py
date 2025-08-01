import pandas as pd

# Paths
gloss_list_file = "../resources/unique-glosses.csv"
filtered_metrics_file = "../resources/evaluated_gloss_time_metrics_filtered.csv"
output_file = "../resources/gloss_times_for_frame_interpolation.csv"

# Load list of glosses
gloss_df = pd.read_csv(gloss_list_file)
gloss_df["gloss"] = gloss_df["gloss"].astype(str).str.strip()

# Load filtered median metrics
metrics_df = pd.read_csv(filtered_metrics_file)
metrics_df["gloss"] = metrics_df["gloss"].astype(str).str.strip()

# Merge on "gloss"
merged_df = gloss_df.merge(metrics_df, on="gloss", how="left")

# Fill missing metric values with 0
merged_df[["median_igt", "median_ogt", "median_gd"]] = merged_df[["median_igt", "median_ogt", "median_gd"]].fillna(0)

# Save final output
merged_df.to_csv(output_file, index=False)
print(f"✅ gloss_times_for_frame_interpolation.csv written with {len(merged_df)} rows.")
