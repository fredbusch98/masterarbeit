import pandas as pd

# Load all files
train_files = ['bt1-train.csv', 'bt2-train.csv']
test_dev_files = ['test.csv', 'dev.csv']

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
