# import pandas as pd

# # Read the CSV file with '|' as the delimiter
# df = pd.read_csv('phoenix_train.csv', delimiter='|')

# # Keep only the 'orth' and 'translation' columns
# df_filtered = df[['orth', 'translation']]

# # Save the filtered DataFrame to a new CSV file
# df_filtered.to_csv('phoenix_filtered.csv', index=False)

# print("Filtered data saved to 'phoenix_filtered.csv'")

import pandas as pd

# Load the CSV file
df = pd.read_csv("phoenix_train.csv")

# Convert the glosses column to a comma-separated string format
df['glosses'] = df['glosses'].apply(lambda x: ','.join(x.strip().split()))

# Reorder the columns
df = df[['full_sentence', 'glosses']]

# Save the modified DataFrame to a new CSV file
df.to_csv("phoenix_train_modified.csv", index=False)

print("CSV file has been modified and saved as 'phoenix_train_modified.csv'.")
