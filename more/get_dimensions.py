import os
import json
from collections import defaultdict

# Path to the main folder
base_path = "/Volumes/IISY/DGSKorpus"

# Dictionary to count unique (width, height) pairs
size_counts = defaultdict(int)

# List to track inconsistent files
warnings = []

# Get all entries in the base path
entries = [entry for entry in os.listdir(base_path) if entry.startswith("entry_") and os.path.isdir(os.path.join(base_path, entry))]
total_entries = len(entries)

print(f"🔍 Found {total_entries} entry folders. Starting processing...\n")

# Iterate through all subfolders
for index, entry in enumerate(entries, start=1):
    entry_path = os.path.join(base_path, entry)
    json_file = os.path.join(entry_path, "openpose.json")

    # Calculate progress percentage
    progress = (index / total_entries) * 100

    # Log progress with percentage
    print(f"📂 [{index}/{total_entries}] ({progress:.2f}%) Processing {entry}...")

    # Check if the file exists
    if os.path.exists(json_file):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Collect all unique (width, height) pairs in this file
            file_sizes = set()
            for obj in data:
                if "width" in obj and "height" in obj:
                    file_sizes.add((obj["width"], obj["height"]))

            # If multiple different sizes exist in one file, log a warning
            if len(file_sizes) > 1:
                warnings.append(f"⚠️ WARNING: Multiple sizes in {json_file}: {file_sizes}")

            # Update the global count
            for size in file_sizes:
                size_counts[size] += 1

        except Exception as e:
            print(f"❌ Error processing {json_file}: {e}")

# Log results
print("\n✅ Processing complete! Unique width/height pairs and their occurrences:")
for (width, height), count in sorted(size_counts.items()):
    print(f"📏 Size ({width}x{height}): {count} times")

# Log warnings if any
if warnings:
    print("\n⚠️ Warnings detected:")
    for warning in warnings:
        print(warning)
else:
    print("\n🎉 No inconsistencies found!")

print("\n🚀 Script execution finished!")
