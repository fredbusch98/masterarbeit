"""
Robustly maps a user-provided sequence of glosses to known variants using exact, prefix, fuzzy, and embedding-based matching, 
then generates a corresponding pose sequence video and MimicMotion config for inference. 
Outputs commands to transfer and run the generated files on a MimicMotion pod.
Basically the initial script for the Gloss2Pose Translator: Gloss Matcher.
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

# Argument parsing
p = argparse.ArgumentParser(description='🔍 Robust 1:1 gloss matcher that will generate a pose sequence video for the given gloss sequence!')
p.add_argument('-g', '--glosses', required=True,
               help='Comma-separated gloss list')
p.add_argument('-o', '--output-filename', required=True,
               help='Output filename of the final pose sequence video.')
p.add_argument('-c', '--config-filename', required=True,
               help='Output filename of the configuration .yaml file for MimicMotion inference.')
args = p.parse_args()
queries = [g.strip().upper() for g in args.glosses.split(',') if g.strip()]
output_filename = args.output_filename
config_filename = args.config_filename

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

# Gloss matching pipeline
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
for q in queries:
    match = match_gloss(q)
    gloss_sequence = gloss_sequence + match + ","
    print(f"🔹 {q} → {match}")

gloss_sequence = gloss_sequence.rstrip(",")
gloss_list = gloss_sequence.split(",")
print(f"\nFinal mapped gloss sequence: \n{gloss_list}")

config_path, video_path, cfg_name, vid_name = run_from_list(gloss_list, output_filename, config_filename, default_frames=False, fill_pose_sequence=True)
abs_config_path = os.path.abspath(config_path)
abs_video_path = os.path.abspath(video_path)
print("\nCopy video and config to the mimicmotion pod:")
print(f"kubectl cp {abs_config_path} s85468/mimicmotion:/storage/MimicMotion/configs/{cfg_name}")
print(f"kubectl cp {abs_video_path} s85468/mimicmotion:/storage/MimicMotion/assets/example_data/videos/{vid_name}\n")
print("kubectl -n s85468 exec -it mimicmotion -- bash")
print("conda activate mimicmotion && cd /storage/MimicMotion\n")
print("Start inference with the config on the mimicmotion pod:")
print(f"python inference.py --inference_config configs/{cfg_name}")