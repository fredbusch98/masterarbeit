import os
import json
from pathlib import Path

# Expected lengths
body_expected = 75
face_expected = 210
left_hand_expected = 63
right_hand_expected = 63

# Directory containing entry subfolders
base_dir = Path('/Volumes/IISY/DGSKorpus')

# Collect mismatches
mismatches = []
# Counters for mismatches per key
mismatch_counts = {
    'pose_keypoints_2d': 0,
    'face_keypoints_2d': 0,
    'hand_left_keypoints_2d': 0,
    'hand_right_keypoints_2d': 0,
}
total_checked = 0
# Set to track unique frames with mismatches
frames_with_mismatches = set()

# Enumerate entry folders
entries = sorted([d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith('entry_')])
total_entries = len(entries)

print(f"🚀 Starting openpose.json validation for {total_entries} entries... 🚀")

# Iterate over entry folders with progress bar
for idx, entry in enumerate(entries, start=1):
    # Compute percentage progress
    percent = idx / total_entries
    bar_length = 20
    filled = int(percent * bar_length)
    bar = '🟩' * filled + '⬜' * (bar_length - filled)
    print(f"{bar} {percent*100:6.2f}% 🔄 Processing {entry.name} ({idx}/{total_entries})")

    openpose_file = entry / 'openpose.json'
    if not openpose_file.exists():
        print(f"⚠️  Warning: {openpose_file} not found, skipping.")
        continue

    # Load JSON
    try:
        records = json.loads(openpose_file.read_text())
    except json.JSONDecodeError as e:
        print(f"❌ Error decoding JSON in {openpose_file}: {e}")
        continue

    # Each record is one parent object
    for rec_idx, rec in enumerate(records):
        # Skip if camera is "c"
        if rec.get('camera') == 'c':
            continue

        frames = rec.get('frames', {})
        # Iterate through each frame
        for frame_num, frame_data in frames.items():
            people = frame_data.get('people', [])
            frame_has_mismatch = False  # Flag to detect mismatch in this frame

            for person_idx, person in enumerate(people):
                total_checked += 1
                # Define checks
                for key, expected in [
                    ('pose_keypoints_2d', body_expected),
                    ('face_keypoints_2d', face_expected),
                    ('hand_left_keypoints_2d', left_hand_expected),
                    ('hand_right_keypoints_2d', right_hand_expected)
                ]:
                    actual_len = len(person.get(key, []))
                    if actual_len != expected:
                        mismatches.append({
                            'entry': entry.name,
                            'record_index': rec_idx,
                            'frame': frame_num,
                            'person_index': person_idx,
                            'key': key,
                            'expected': expected,
                            'actual': actual_len,
                        })
                        mismatch_counts[key] += 1
                        frame_has_mismatch = True

            # If any mismatch was found in this frame, track it
            if frame_has_mismatch:
                frame_id = f"{entry.name}/{rec_idx}/{frame_num}"
                frames_with_mismatches.add(frame_id)

# Final reporting
print(f"✅ Checked {total_checked} people entries across all frames.")
print(f"📝 Total mismatches: {len(mismatches)}")
print(f"   🔹 pose_keypoints_2d mismatches: {mismatch_counts['pose_keypoints_2d']}")
print(f"   🔹 face_keypoints_2d mismatches: {mismatch_counts['face_keypoints_2d']}")
print(f"   🔹 hand_left_keypoints_2d mismatches: {mismatch_counts['hand_left_keypoints_2d']}")
print(f"   🔹 hand_right_keypoints_2d mismatches: {mismatch_counts['hand_right_keypoints_2d']}")
print(f"🎞️  Unique frames with at least one mismatch: {len(frames_with_mismatches)}")

# Write mismatches to file if any
if mismatches:
    out_file = base_dir / 'openpose_length_mismatches.json'
    with open(out_file, 'w') as f:
        json.dump(mismatches, f, indent=2)
    print(f"📁 Detailed mismatches written to {out_file}")
else:
    print("🎉 No mismatches found. All keypoint arrays match the expected lengths.")
