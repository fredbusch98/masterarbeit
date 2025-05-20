import pandas as pd

# Load the CSV file
df = pd.read_csv("phoenix_test.csv")

# Convert the glosses column to a comma-separated string format
df['glosses'] = df['glosses'].apply(lambda x: ','.join(x.strip().split()))

# Reorder the columns
df = df[['full_sentence', 'glosses']]

# Save the modified DataFrame to a new CSV file
df.to_csv("phoenix_train_modified.csv", index=False)

print("CSV file has been modified and saved as 'phoenix_train_modified.csv'.")
