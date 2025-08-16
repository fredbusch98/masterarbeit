"""
This script filters out overlapping sentence–gloss pairs from the training data
to prevent data leakage into evaluation. It:
  - Loads test and dev CSV files and collects all (full_sentence, glosses) pairs.
  - Reads the training CSV(s) and removes any rows that exactly match a test/dev pair.
  - Saves the cleaned training data to new CSV files prefixed with 'filtered_'.
Use this to ensure that evaluation sets remain unseen during training.
"""
import pandas as pd

# Load all files
train_files = ['bt2.csv']
test_dev_files = ['./new-splits-kcross-3/test3.csv', './new-splits-kcross-3/dev3.csv']

# Load test and dev (sentence, glosses) pairs into a list to preserve duplicates
test_dev_pairs = []
for file in test_dev_files:
    df = pd.read_csv(file)
    df = df.dropna(subset=['full_sentence', 'glosses'])
    test_dev_pairs.extend(zip(df['full_sentence'].str.strip(), df['glosses'].str.strip()))

# Function to filter training data
def filter_train_data(file, exclude_pairs):
    df = pd.read_csv(file)
    original_count = len(df)
    df['full_sentence'] = df['full_sentence'].str.strip()
    df['glosses'] = df['glosses'].str.strip()
    df_filtered = df[~df.apply(lambda row: (row['full_sentence'], row['glosses']) in exclude_pairs, axis=1)]
    df_filtered.to_csv(f'filtered_{file}', index=False)
    print(f"Filtered {file}: {original_count - len(df_filtered)} entries removed.")

# Apply filter to both training files
for file in train_files:
    filter_train_data(file, test_dev_pairs)
