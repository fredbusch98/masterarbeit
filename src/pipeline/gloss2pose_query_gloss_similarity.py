"""
Given new glosses as a comma-separated list, perform robust 1:1 matching:
1. Exact match
2. Prefix-based variant selection (e.g. ICH→ICH1)
3. Fuzzy fallback via scikit-learn & embeddings
"""
import os
import json
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import argparse
from rapidfuzz import process, fuzz
from gloss2pose_dict import run_from_list

# Config paths
MAPPING_OUT = './resources/gloss_to_idx.json'
EMBEDDINGS_IN = './resources/gloss_embeddings.npz'
MODEL_NAME = 'distiluse-base-multilingual-cased-v1'

# Load mapping
gloss_to_idx = json.load(open(MAPPING_OUT, 'r', encoding='utf-8'))
idx_to_gloss = {v: k for k, v in gloss_to_idx.items()}
all_glosses = list(gloss_to_idx.keys())

# Precompute prefix map: base → [(num:int, sub:str, gloss:str)]
pattern = re.compile(r"^([A-ZÄÖÜ]+)(\d+)?([A-Z]*)$")
prefix_map = {}
for g in all_glosses:
    m = pattern.match(g)
    if not m:
        continue
    base, num_s, sub = m.groups()
    num = int(num_s) if num_s else 0
    sub = sub or ""
    prefix_map.setdefault(base, []).append((num, sub, g))
# Sort each variant‐list by (numeric suffix, then letter)
for base in prefix_map:
    prefix_map[base].sort(key=lambda x: (x[0], x[1]))

# Load embeddings for fuzzy fallback
embeddings = np.load(EMBEDDINGS_IN)['embeddings']
model = SentenceTransformer(MODEL_NAME)
nn = NearestNeighbors(n_neighbors=1, metric='cosine').fit(embeddings)

# Argument parsing
p = argparse.ArgumentParser(description='🔍 Robust 1:1 gloss matcher')
p.add_argument('--glosses', required=True,
               help='Comma-separated gloss list')
args = p.parse_args()
queries = [g.strip().upper() for g in args.glosses.split(',') if g.strip()]

# Matching pipeline
def match_gloss(q):
    q = q.upper()
    # 1) Exact match
    if q in gloss_to_idx:
        return q

    # 2) Prefix‐based variant
    m = pattern.match(q)
    if m:
        base, num_s, sub_s = m.groups()
        variants = prefix_map.get(base)
        if variants:
            # if the query has no digit, only keep true variants (num>0)
            if not num_s:
                filtered = [v for v in variants if v[0] > 0]
                if filtered:
                    variants = filtered
            # take the smallest (num,sub)
            return variants[0][2]

    # 3) rapidfuzz fuzzy match
    result = process.extractOne(q, all_glosses, scorer=fuzz.ratio)
    if result:
        best, score, _ = result
        if score >= 80:
            return best

    # 4) embedding fallback
    emb = model.encode([q], normalize_embeddings=True)
    dist, idx = nn.kneighbors(np.array(emb, dtype='float32'))
    return idx_to_gloss[idx[0][0]]

gloss_sequence = ""
# Run and print
for q in queries:
    match = match_gloss(q)
    gloss_sequence = gloss_sequence + match + ","
    print(f"🔹 {q} → {match}")

gloss_sequence = gloss_sequence.rstrip(",")
gloss_list = gloss_sequence.split(",")
print(f"\nFinal mapped gloss sequence: \n{gloss_list}")

config_path, video_path, cfg_name, vid_name = run_from_list(gloss_list, default_frames=False, fill_pose_sequence=True)
abs_config_path = os.path.abspath(config_path)
abs_video_path = os.path.abspath(video_path)
print("\nCopy video and config to the mimicmotion pod:")
print(f"kubectl cp {abs_config_path} s85468/mimicmotion:/storage/MimicMotion/configs/{cfg_name}")
print(f"kubectl cp {abs_video_path} s85468/mimicmotion:/storage/MimicMotion/configs/{vid_name}\n")
print("Start inference with the config on the mimicmotion pod:")
print(f"python inference.py --inference_config configs/{cfg_name}")

# HIER IST DER PROMPT FÜR ChatGPT API: 
# Du bist ein professioneller Übersetzer für Deutsche Gebärdensprache. Du kannst ganze deutsche Sätze in ihre Gebärden-Gloss-Form übersetzen. Übersetze den folgenden Satz in eine solche Gebärden-Gloss-Sequenz: "In der Schule esse ich meistens einen Apfel in der Mittagspause."
# Gib ausschließlich die Gloss-Sequenz als komma-separierte Liste aus und keinerlei zusätzliche Erklärung oder Gedanken!

# How to run:
# 1. Prompt ChatGPT with the prompt above and change the sentence to your desired sentence!
# The output will looks something like this: GLOSS_1, GLOSS_2, ..., GLOSS_N
# 2. Copy the output and call the script: python query_gloss_similarity.py --glosses "GLOSS_1, GLOSS_2, ..., GLOSS_N"
# This will create a pose sequence video that represents your input sentence.