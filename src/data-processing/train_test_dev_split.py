import pandas as pd
import os
from sklearn.model_selection import train_test_split

# Path to your original dataset
dataset_path = "/Volumes/IISY/Text2Gloss-Data/BT-2-paraphrase/augmented_dataset_2_paraphrase.csv"

# Load the dataset
df = pd.read_csv(dataset_path)

# First split: 80% train, 20% temp (dev + test)
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)

# Second split: 50% dev, 50% test of the remaining 20% -> 10% each
dev_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# Output directory (same as input)
output_dir = os.path.dirname(dataset_path)

# Save splits
train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
dev_df.to_csv(os.path.join(output_dir, "dev.csv"), index=False)
test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)

print("Dataset successfully split into train/dev/test.")
