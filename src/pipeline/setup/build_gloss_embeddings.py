"""
Build embeddings for unique DGS glosses and save mappings.
"""
import os
import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# Configuration
CSV_PATH = '../resources/unique-glosses.csv'
EMBEDDINGS_OUT = '../resources/gloss_embeddings.npz'
MAPPING_OUT = '../resources/gloss_to_idx.json'
MODEL_NAME = 'distiluse-base-multilingual-cased-v1'  # multilingual SBERT

print("📂 Step 1: Loading glosses from CSV...")
df = pd.read_csv(CSV_PATH)
glosses = df['gloss'].astype(str).tolist()
print(f"✅ Loaded {len(glosses)} glosses.")

print("🧠 Step 2: Encoding glosses with SBERT...")
model = SentenceTransformer(MODEL_NAME)
embeddings = model.encode(
    glosses,
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True
)
embeddings = np.array(embeddings, dtype='float32')
print("✅ Embedding completed.")

print("💾 Step 3: Saving embeddings and gloss-to-index mapping...")
# Save embeddings
np.savez_compressed(EMBEDDINGS_OUT, embeddings=embeddings)
# Save mapping
gloss_to_idx = {gloss: idx for idx, gloss in enumerate(glosses)}
with open(MAPPING_OUT, 'w', encoding='utf-8') as f:
    json.dump(gloss_to_idx, f, ensure_ascii=False, indent=2)
print(f"✅ Embeddings saved to {EMBEDDINGS_OUT}")
print(f"✅ Mapping saved to {MAPPING_OUT}")

print(f"🎉 All done! Prepared embeddings for {len(glosses)} glosses.")