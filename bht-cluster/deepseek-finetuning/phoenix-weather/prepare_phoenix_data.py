"""
PHOENIX Text2Gloss data preprocessor.

Reads `phoenix_train.csv` (pipe-delimited) and produces two outputs:
  1) `phoenix_filtered.csv` — contains only `orth` and `translation`.
  2) `phoenix_train_modified.csv` — contains `full_sentence` and a normalized
     `glosses` column where whitespace-delimited tokens are joined into a
     single comma-separated string to fit the Text2Gloss format of the training pipeline.
"""
import pandas as pd

# Read the original CSV with '|' as the delimiter
df = pd.read_csv("phoenix_train.csv", delimiter="|") # this csv file needs to be downloaded from the original dataset release!

# --- First task: Keep only 'orth' and 'translation' columns ---
df_filtered = df[['orth', 'translation']]
df_filtered.to_csv("phoenix_filtered.csv", index=False)
print("Filtered data saved to 'phoenix_filtered.csv'.")

# --- Second task: Process 'full_sentence' and 'glosses' columns ---
# Ensure 'glosses' exists before transforming
if 'glosses' in df.columns and 'full_sentence' in df.columns:
    df['glosses'] = df['glosses'].apply(lambda x: ','.join(str(x).strip().split()))
    df_modified = df[['full_sentence', 'glosses']]
    df_modified.to_csv("phoenix_train_modified.csv", index=False)
    print("CSV file has been modified and saved as 'phoenix_train_modified.csv'.")
else:
    print("Warning: 'full_sentence' or 'glosses' columns not found in the dataset.")
