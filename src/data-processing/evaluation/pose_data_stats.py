import os
import json

# Base directory
base_dir = '/Volumes/IISY/DGSKorpus'

# Initialize size counters
openpose_size = 0
gloss2pose_size = 0
dictionary_size = 0

# Frame counters
openpose_frames = 0
gloss2pose_frames = 0
dictionary_frames = 0

# Gloss counters
gloss2pose_total_glosses = 0
gloss2pose_unique_glosses = set()

dictionary_total_glosses = 0
dictionary_unique_glosses = set()

# Gather entry directories
entries = [e for e in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, e)) and e.startswith('entry_')]
total_entries = len(entries)

print(f"🔍 Found {total_entries} entries in {base_dir}.\n")

# Iterate through each entry
for idx, entry in enumerate(entries):
    entry_path = os.path.join(base_dir, entry)
    op_path = os.path.join(entry_path, 'openpose.json')
    gp_path = os.path.join(entry_path, 'gloss2pose.json')

    progress = ((idx + 1) / total_entries) * 100
    print(f"📂 Processing {entry} ({progress:.1f}% done)")

    # --- process openpose.json ---
    if os.path.isfile(op_path):
        openpose_size += os.path.getsize(op_path)
        try:
            with open(op_path, 'r') as f:
                data = json.load(f)
                for obj in data:
                    if obj.get("camera") in ("a1", "b1"):
                        frames_obj = obj.get("frames", {})
                        openpose_frames += len(frames_obj)
        except Exception as e:
            print(f"⚠️ Error reading {op_path}: {e}")

    # --- process gloss2pose.json ---
    if os.path.isfile(gp_path):
        gloss2pose_size += os.path.getsize(gp_path)
        try:
            with open(gp_path, 'r') as f:
                data = json.load(f).get("data", [])
                for item in data:
                    # frame counting
                    seq = item.get("pose_sequence", [])
                    gloss2pose_frames += len(seq)

                    # gloss counting
                    gloss = item.get("gloss")
                    if gloss:
                        gloss2pose_total_glosses += 1
                        gloss2pose_unique_glosses.add(gloss)
        except Exception as e:
            print(f"⚠️ Error reading {gp_path}: {e}")

# --- process gloss2pose_dictionary.json ---
dict_path = os.path.join(base_dir, 'gloss2pose_dictionary.json')
print("\n📘 Processing gloss2pose_dictionary.json ...")
if os.path.isfile(dict_path):
    dictionary_size = os.path.getsize(dict_path)
    try:
        with open(dict_path, 'r') as f:
            d = json.load(f)
            for gloss, arr in d.items():
                dictionary_total_glosses += 1
                dictionary_unique_glosses.add(gloss)
                if isinstance(arr, list):
                    dictionary_frames += len(arr)
    except Exception as e:
        print(f"⚠️ Error reading dictionary file: {e}")
else:
    print("⚠️ gloss2pose_dictionary.json not found!")

# --- convert sizes to GB ---
openpose_gb = openpose_size / (1024 ** 3)
gloss2pose_gb = gloss2pose_size / (1024 ** 3)
dictionary_gb = dictionary_size / (1024 ** 3)

# --- final report ---
print("\n✅ Done! Here's the summary:\n")
print(f"💾 Total size of openpose.json files:         {openpose_gb:.2f} GB")
print(f"💾 Total size of gloss2pose.json files:       {gloss2pose_gb:.2f} GB")
print(f"📘 Size of gloss2pose_dictionary.json:        {dictionary_gb:.2f} GB\n")

print(f"🎞️  Total frames in openpose.json (a1/b1):      {openpose_frames}")
print(f"🎞️  Total frames in gloss2pose.json:           {gloss2pose_frames}")
print(f"🧠 Total frames in gloss2pose_dictionary.json: {dictionary_frames}\n")

print(f"📌 Total glosses in gloss2pose.json:           {gloss2pose_total_glosses}")
print(f"🔁 Unique glosses in gloss2pose.json:          {len(gloss2pose_unique_glosses)}\n")

print(f"📌 Total glosses in gloss2pose_dictionary.json: {dictionary_total_glosses}")
print(f"🔁 Unique glosses in gloss2pose_dictionary.json: {len(dictionary_unique_glosses)}")
