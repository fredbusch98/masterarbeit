import os
import re
import csv
from all_types import load_gloss_types

def parse_timestamp(timestamp_str):
    """Convert a timestamp 'HH:MM:SS,mmm' into milliseconds (as integer)."""
    hours, minutes, seconds, millis = re.split('[: ,]', timestamp_str)
    return int(hours) * 3600000 + int(minutes) * 60000 + int(seconds) * 1000 + int(millis)

# Load gloss types from CSV
gloss_types = load_gloss_types("/Volumes/IISY/DGSKorpus/all-types-dgs.csv")
gloss_types = {gloss.rstrip('^') for gloss in gloss_types}

# Dictionary to track total duration in ms and occurrence count for each gloss.
gloss_stats = {gloss: {"total_duration": 0, "count": 0} for gloss in gloss_types}

base_path = "/Volumes/IISY/DGSKorpus/"

# Process directories that start with "entry_"
entries = [
    entry for entry in os.listdir(base_path)
    if os.path.isdir(os.path.join(base_path, entry)) and entry.startswith("entry_")
]

print(f"🚀 Processing {len(entries)} entries for average gloss durations...\n")

for entry in entries:
    folder_path = os.path.join(base_path, entry)
    filtered_path = os.path.join(folder_path, "filtered-transcript.srt")
    if not os.path.exists(filtered_path):
        print(f"⚠️  {entry} skipped: filtered-transcript-only-gloss.srt not found.\n")
        continue

    with open(filtered_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Split file into SRT blocks (blocks separated by empty lines)
    blocks = content.strip().split("\n\n")
    
    for block in blocks:
        lines = block.splitlines()
        if len(lines) < 3:
            continue  # Skip incomplete blocks

        # Parse timestamp line (assumed to be the second line)
        timestamp_line = lines[1]
        match = re.match(r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})", timestamp_line)
        if not match:
            continue
        start_str, end_str = match.groups()
        start_ms = parse_timestamp(start_str)
        end_ms = parse_timestamp(end_str)
        duration_ms = end_ms - start_ms

        # Combine the subtitle text lines (all lines after the timestamp)
        text_content = " ".join(lines[2:])
        # Remove leading "A: ", "B: ", or "C: " from each text line.
        text_content = re.sub(r"^(A|B|C): ", "", text_content, flags=re.MULTILINE)

        # For each gloss, if it appears in the text, add the duration and count it.
        for gloss in gloss_types:
            if gloss in text_content:
                gloss_stats[gloss]["total_duration"] += duration_ms
                gloss_stats[gloss]["count"] += 1

    print(f"✅ {entry} processed.")

# Calculate average duration (in ms) for each gloss and sort the results alphabetically.
results = []
for gloss, stats in gloss_stats.items():
    if stats["count"] > 0:  # Only include glosses that have been found at least once
        average_duration_ms = stats["total_duration"] / stats["count"]
        results.append((gloss, average_duration_ms))

results.sort(key=lambda x: x[0])  # Sort alphabetically by gloss

# Write the results to a CSV file.
output_csv = os.path.join(base_path, "average_gloss_times.csv")
with open(output_csv, "w", newline='', encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["gloss", "average_time_ms"])
    for gloss, avg_ms in results:
        writer.writerow([gloss, avg_ms])

print(f"\n🎉 Average gloss times (in ms) written to {output_csv}")

# Calculate total average gloss time across all glosses
total_duration = sum(stats["total_duration"] for stats in gloss_stats.values())
total_count = sum(stats["count"] for stats in gloss_stats.values())

if total_count > 0:
    total_avg_time = total_duration / total_count
    print(f"\n📊 Total Average Gloss Time: {total_avg_time:.2f} ms")
else:
    print("\n⚠️ No gloss occurrences found, cannot calculate total average gloss time.")
