'''
gloss_similarity_mapper.py

Implementation Plan:

1. Preprocessing Script (preprocess_pairs):
   - Load unique-glosses.csv and dataset.csv using pandas.
   - For each gloss in unique-glosses.csv, find all sentences in dataset.csv whose glosses column (comma-separated) contains that gloss.
   - Output pairs.csv with columns [gloss, full_sentence].

2. Embedding Script (embed_pairs):
   - Load pairs.csv.
   - Initialize a SentenceTransformer model (e.g., 'paraphrase-multilingual-mpnet-base-v2').
   - For each pair, create a consistent text input for embedding: "{gloss} | {full_sentence}".
   - Compute embeddings in batch for efficiency.
   - Normalize embeddings and fit a NearestNeighbors index (using cosine metric) over them.
   - Save the NearestNeighbors model and metadata (embeddings not needed separately) to disk.

3. Similarity Check Script (similarity_check):
   - Load the same SentenceTransformer model, NearestNeighbors index, and metadata from disk.
   - For a given input sentence and each gloss from the translation module output:
       - Construct the query text "{gloss} | {input_sentence}".
       - Embed and normalize the query.
       - Use NearestNeighbors.kneighbors to find the top-k similar indexed pairs.
       - Convert cosine distances to similarity scores and return matched glosses.

Requirements:
- pandas
- sentence_transformers
- scikit-learn
- numpy
- tqdm

Usage:
    python gloss_similarity_mapper.py preprocess_pairs \
        --unique_glosses ./resources/unique-glosses.csv \
        --dataset ./resources/dataset.csv \
        --output ./resources/pairs.csv

    python gloss_similarity_mapper.py embed_pairs \
        --pairs ./resources/pairs.csv \
        --model_path ./resources/index_dir/nn_model.pkl

    python gloss_similarity_mapper.py similarity_check \
        --input_sentence "Ich habe schöne Erinnerungen an früher." \
        --predicted_glosses ICH,ERINNERUNG,FRÜHER,SCHÖN,HABEN \
        --model_path ./resources/index_dir/nn_model.pkl \
        --top_k 1

'''
import os
import argparse
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import pickle

MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'


def preprocess_pairs(unique_glosses_path, dataset_path, output_path):
    print("🔄 Starting preprocessing of gloss-sentence pairs...")
    df_gloss = pd.read_csv(unique_glosses_path)
    df_data = pd.read_csv(dataset_path)
    records = []
    for gloss in tqdm(df_gloss['gloss'].dropna().unique(), desc="🔍 Matching glosses"):
        mask = df_data['glosses'].str.split(',').apply(lambda lst: gloss in lst)
        for sent in df_data.loc[mask, 'full_sentence']:
            records.append({'gloss': gloss, 'full_sentence': sent})
    df_pairs = pd.DataFrame(records)
    df_pairs.to_csv(output_path, index=False)
    print(f"✅ Saved {len(df_pairs)} gloss-sentence pairs to {output_path} 📄")


def embed_pairs(pairs_path, model_path, batch_size=64):
    print("⚙️ Embedding gloss-sentence pairs...")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    df = pd.read_csv(pairs_path)
    model = SentenceTransformer(MODEL_NAME)
    texts = (df['gloss'] + ' | ' + df['full_sentence']).tolist()

    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="🧠 Encoding batches"):
        batch = texts[i:i+batch_size]
        emb = model.encode(batch, convert_to_numpy=True)
        embeddings.append(emb)
    embeddings = np.vstack(embeddings)
    print("📐 Normalizing embeddings...")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms

    print("🔗 Building NearestNeighbors index...")
    nn = NearestNeighbors(metric='cosine', algorithm='auto')
    nn.fit(embeddings)

    with open(model_path, 'wb') as f:
        pickle.dump({'nn_model': nn,
                     'embeddings': embeddings,
                     'glosses': df['gloss'].tolist(),
                     'sentences': df['full_sentence'].tolist()}, f)
    print(f"✅ NearestNeighbors model and metadata saved to {model_path} 💾")


def similarity_check(input_sentence, predicted_glosses, model_path, top_k=1):
    print("🔍 Performing similarity check...")
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
    nn = data['nn_model']
    embeddings = data['embeddings']
    glosses = data['glosses']
    sentences = data['sentences']
    model = SentenceTransformer(MODEL_NAME)

    results = {}
    for gloss in tqdm(predicted_glosses, desc="🧪 Checking glosses"):
        query_text = f"{gloss} | {input_sentence}"
        q_emb = model.encode([query_text], convert_to_numpy=True)[0]
        q_emb = q_emb / np.linalg.norm(q_emb)
        distances, indices = nn.kneighbors([q_emb], n_neighbors=top_k)
        hits = []
        for dist, idx in zip(distances[0], indices[0]):
            sim_score = 1 - dist
            hits.append({'match_gloss': glosses[idx],
                         'full_sentence': sentences[idx],
                         'score': float(sim_score)})
        results[gloss] = hits
    print("✅ Similarity check completed 🎯")
    return results


def main():
    parser = argparse.ArgumentParser(description="🔧 Gloss Similarity Mapper Tools")
    subparsers = parser.add_subparsers(dest='command', required=True)

    # preprocess_pairs
    p1 = subparsers.add_parser('preprocess_pairs')
    p1.add_argument('--unique_glosses', required=True)
    p1.add_argument('--dataset', required=True)
    p1.add_argument('--output', required=True)

    # embed_pairs
    p2 = subparsers.add_parser('embed_pairs')
    p2.add_argument('--pairs', required=True)
    p2.add_argument('--model_path', required=True)
    p2.add_argument('--batch_size', type=int, default=64)

    # similarity_check
    p3 = subparsers.add_parser('similarity_check')
    p3.add_argument('--input_sentence', required=True)
    p3.add_argument('--predicted_glosses', required=True,
                    help="Comma-separated list of glosses from translation module")
    p3.add_argument('--model_path', required=True)
    p3.add_argument('--top_k', type=int, default=1)

    args = parser.parse_args()
    if args.command == 'preprocess_pairs':
        preprocess_pairs(args.unique_glosses, args.dataset, args.output)
    elif args.command == 'embed_pairs':
        embed_pairs(args.pairs, args.model_path, args.batch_size)
    elif args.command == 'similarity_check':
        predicted = args.predicted_glosses.split(',')
        res = similarity_check(args.input_sentence, predicted, args.model_path, args.top_k)
        for g, hits in res.items():
            print(f"\n🔤 Input Gloss: {g}")
            for h in hits:
                print(f"  👉 Match: {h['match_gloss']} (score: {h['score']:.4f})")
                print(f"     📚 Sentence: {h['full_sentence']}")


if __name__ == '__main__':
    main()
