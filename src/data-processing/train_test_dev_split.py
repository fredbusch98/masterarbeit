import pandas as pd
import os
from sklearn.model_selection import train_test_split

# Path to your original dataset
dataset_path = "original_dataset.csv"

# Load the dataset
df = pd.read_csv(dataset_path)

# Output directory (same as input)
output_dir = os.path.dirname(dataset_path)

# Seeds for reproducibility
seeds = [101, 202, 303]

# Generate 3 independent splits
for i, seed in enumerate(seeds, start=1):
    # First split: 80% train, 20% temp
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=seed)
    
    # Second split: 50% dev, 50% test from temp -> 10% each of total
    dev_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=seed + 1)
    
    # Save each split
    train_df.to_csv(os.path.join(output_dir, f"train_split_{i}.csv"), index=False)
    dev_df.to_csv(os.path.join(output_dir, f"dev_split_{i}.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, f"test_split_{i}.csv"), index=False)

print("Created 3 independent train/dev/test splits.")
