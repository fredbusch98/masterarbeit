import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
import json

# Step 1: Load and preprocess the dataset
df = pd.read_csv('dataset.csv', encoding='utf-8')
train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

def format_conversations(df):
    data = []
    for row in df.itertuples():
        messages = [
            {"role": "system", "content": "You are a sign language gloss translator. Given a sentence, output the corresponding gloss sequence."},
            {"role": "user", "content": row.full_sentence},
            {"role": "assistant", "content": row.glosses},
        ]
        data.append({"messages": messages})
    return data

train_data = format_conversations(train_df)
val_data = format_conversations(val_df)

train_dataset = Dataset.from_list(train_data)
val_dataset = Dataset.from_list(val_data)

print(json.dumps(train_data[:5], indent=2, ensure_ascii=False))