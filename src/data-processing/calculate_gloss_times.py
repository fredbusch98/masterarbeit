import os
import re
import csv
from all_types import load_gloss_types

def parse_timestamp(timestamp_str):
    """Convert a timestamp 'HH:MM:SS,mmm' into milliseconds (as integer)."""
    hours, minutes, seconds, millis = re.split('[: ,]', timestamp_str)
    return int(hours) * 3600000 + int(minutes) * 60000 + int(seconds) * 1000 + int(millis)

def process_sentence_blocks(blocks, gloss_stats, gloss_types):
    """Process a sequence of blocks belonging to the same sentence to calculate IGT and OGT."""
    if not blocks or len(blocks) < 2:
        return  # Need at least 2 glosses to calculate IGT/OGT
    
    # Group blocks by timestamp to handle identical timestamps
    timestamp_groups = {}
    for block in blocks:
        key = (block["start_ms"], block["end_ms"])
        if key not in timestamp_groups:
            timestamp_groups[key] = []
        timestamp_groups[key].append(block)
    
    # Convert groups to a sorted list
    sorted_groups = sorted(timestamp_groups.items(), key=lambda x: x[0][0])  # Sort by start_ms
    
    # Skip IGT for first group and OGT for last group
    for i, (timestamp, group_blocks) in enumerate(sorted_groups):
        # Calculate IGT (skip for first group)
        if i > 0:
            prev_timestamp, prev_group = sorted_groups[i-1]
            # Time from previous group's end to current group's start
            igt = timestamp[0] - prev_timestamp[1]  # current_start - prev_end
            
            # Apply IGT to all glosses in the current group
            for block in group_blocks:
                for gloss in gloss_types:
                    if gloss in block["text"]:
                        gloss_stats[gloss]["igt_total"] += igt
                        gloss_stats[gloss]["igt_count"] += 1
        
        # Calculate OGT (skip for last group)
        if i < len(sorted_groups) - 1:
            next_timestamp, next_group = sorted_groups[i+1]
            # Time from current group's end to next group's start
            ogt = next_timestamp[0] - timestamp[1]  # next_start - current_end
            
            # Apply OGT to all glosses in the current group
            for block in group_blocks:
                for gloss in gloss_types:
                    if gloss in block["text"]:
                        gloss_stats[gloss]["ogt_total"] += ogt
                        gloss_stats[gloss]["ogt_count"] += 1

# Load gloss types from CSV
gloss_types = load_gloss_types("/Volumes/IISY/DGSKorpus/all-types-dgs.csv")
gloss_types = {gloss.rstrip('^') for gloss in gloss_types}

# Dictionary to track total duration in ms and occurrence count for each gloss.
gloss_stats = {gloss: {"total_duration": 0, "count": 0, "igt_total": 0, "igt_count": 0, "ogt_total": 0, "ogt_count": 0} for gloss in gloss_types}

base_path = "/Volumes/IISY/DGSKorpus/"

# Process directories that start with "entry_"
entries = [
    entry for entry in os.listdir(base_path)
    if os.path.isdir(os.path.join(base_path, entry)) and entry.startswith("entry_")
]

print(f"🚀 Processing {len(entries)} entries for average gloss durations, IGT, and OGT...\n")

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
    
    # We need to process blocks by sentence to properly handle IGT and OGT
    current_sentence_blocks = []
    
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
        
        # Strip the speaker prefix for checking uppercase
        clean_text = re.sub(r"^(A|B|C): ", "", lines[2], flags=re.MULTILINE).strip()
        
        # Check if this is a gloss or a sentence
        is_gloss = clean_text.isupper()
        
        # Process based on block type
        if not is_gloss:
            # This is a new sentence - process the previous sentence if we have glosses
            if current_sentence_blocks:
                process_sentence_blocks(current_sentence_blocks, gloss_stats, gloss_types)
            
            # Start a new sentence collection
            current_sentence_blocks = []
        else:
            # This is a gloss - add it to the current sentence collection
            block_data = {
                "start_ms": start_ms,
                "end_ms": end_ms,
                "text": text_content,
                "duration_ms": duration_ms
            }
            current_sentence_blocks.append(block_data)
            
            # Update regular gloss stats
            for gloss in gloss_types:
                if gloss in text_content:
                    gloss_stats[gloss]["total_duration"] += duration_ms
                    gloss_stats[gloss]["count"] += 1
    
    # Process any remaining blocks from the last sentence
    if current_sentence_blocks:
        process_sentence_blocks(current_sentence_blocks, gloss_stats, gloss_types)
            
    print(f"✅ {entry} processed.")

# Calculate average duration (in ms) for each gloss and sort the results alphabetically.
results = []
total_igt = 0
total_igt_count = 0
total_ogt = 0
total_ogt_count = 0

for gloss, stats in gloss_stats.items():
    if stats["count"] > 0:  # Only include glosses that have been found at least once
        # Average In-Gloss Time (AGT)
        average_duration_ms = stats["total_duration"] / stats["count"]
        
        # Calculate average IGT if we have data
        average_igt = 0
        if stats["igt_count"] > 0:
            average_igt = stats["igt_total"] / stats["igt_count"]
            total_igt += stats["igt_total"]
            total_igt_count += stats["igt_count"]
            
        # Calculate average OGT if we have data
        average_ogt = 0
        if stats["ogt_count"] > 0:
            average_ogt = stats["ogt_total"] / stats["ogt_count"]
            total_ogt += stats["ogt_total"]
            total_ogt_count += stats["ogt_count"]

        average_tgt = average_igt + average_duration_ms + average_ogt
            
        results.append((gloss, average_duration_ms, average_igt, average_ogt, average_tgt))

results.sort(key=lambda x: x[0])  # Sort alphabetically by gloss

# Write the results to a CSV file.
output_csv = os.path.join(base_path, "gloss_timing_stats.csv")
with open(output_csv, "w", newline='', encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["gloss", "average_agt_ms", "average_igt_ms", "average_ogt_ms", "average_tgt_ms"])
    for gloss, avg_ms, avg_igt, avg_ogt, avg_tgt in results:
        writer.writerow([gloss, avg_ms, avg_igt, avg_ogt, avg_tgt])

print(f"\n🎉 Gloss timing statistics written to {output_csv}")

# Calculate total average gloss time across all glosses
total_duration = sum(stats["total_duration"] for stats in gloss_stats.values())
total_count = sum(stats["count"] for stats in gloss_stats.values())

if total_count > 0:
    total_avg_time = total_duration / total_count
    print(f"\n📊 Total Average Gloss Time: {total_avg_time:.2f} ms")
else:
    print("\n⚠️ No gloss occurrences found, cannot calculate total average gloss time.")

# Calculate total average IGT across all glosses
if total_igt_count > 0:
    total_avg_igt = total_igt / total_igt_count
    print(f"📊 Total Average Into-Gloss Time (IGT): {total_avg_igt:.2f} ms")
else:
    print("⚠️ No IGT occurrences found, cannot calculate total average IGT.")

# Calculate total average OGT across all glosses
if total_ogt_count > 0:
    total_avg_ogt = total_ogt / total_ogt_count
    print(f"📊 Total Average Out-of-Gloss Time (OGT): {total_avg_ogt:.2f} ms")
else:
    print("⚠️ No OGT occurrences found, cannot calculate total average OGT.")

# Calculate total average TGT across all glosses
total_avg_tgt = total_avg_igt + total_avg_time + total_avg_ogt
print(f"📊 Total Average Total Gloss Time (TGT): {total_avg_tgt:.2f} ms")