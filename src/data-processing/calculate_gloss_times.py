import os
import re
import csv
from all_types import load_gloss_types

def parse_timestamp(timestamp_str):
    """Convert a timestamp 'HH:MM:SS,mmm' into milliseconds (as integer)."""
    hours, minutes, seconds, millis = re.split('[: ,]', timestamp_str)
    return int(hours) * 3600000 + int(minutes) * 60000 + int(seconds) * 1000 + int(millis)

def process_sentence_blocks(blocks, gloss_stats, gloss_types):
    """
    Process gloss blocks from the same sentence & speaker to calculate IGT, OGT, and TGT.
    Updates both maximum and minimum (single-occurrence) values for each gloss occurrence,
    and records the SRT block indices used in the calculation.
    """
    if not blocks or len(blocks) < 2:
        return  # Need at least 2 glosses to calculate IGT/OGT

    # Group blocks by timestamp (to handle identical timestamps)
    timestamp_groups = {}
    for block in blocks:
        key = (block["start_ms"], block["end_ms"])
        timestamp_groups.setdefault(key, []).append(block)
    
    # Sort groups by the starting timestamp
    sorted_groups = sorted(timestamp_groups.items(), key=lambda x: x[0][0])
    
    # For each group, compute IGT and/or OGT when available.
    for i, (timestamp, group_blocks) in enumerate(sorted_groups):
        current_igt = None
        current_ogt = None
        
        # Get representative index for the current group.
        current_group_index = group_blocks[0].get("index", "")
        
        # Compute IGT if not the first group.
        if i > 0:
            prev_group = sorted_groups[i-1][1]
            prev_group_index = prev_group[0].get("index", "")
            current_igt = timestamp[0] - sorted_groups[i-1][0][1]
        # Compute OGT if not the last group.
        if i < len(sorted_groups) - 1:
            next_group = sorted_groups[i+1][1]
            next_group_index = next_group[0].get("index", "")
            current_ogt = next_group[0]["start_ms"] - timestamp[1]
        
        # Now update each gloss occurrence in this group.
        for block in group_blocks:
            for gloss in gloss_types:
                if gloss in block["text"]:
                    # IGT updates
                    if current_igt is not None:
                        gloss_stats[gloss]["igt_total"] += current_igt
                        gloss_stats[gloss]["igt_count"] += 1
                        index_str = f"index:{prev_group_index}-{current_group_index}"
                        if current_igt > gloss_stats[gloss]["max_igt"]:
                            gloss_stats[gloss]["max_igt"] = current_igt
                            gloss_stats[gloss]["max_igt_entry"] = block["entry"]
                            gloss_stats[gloss]["max_igt_index"] = index_str
                        if current_igt < gloss_stats[gloss]["min_igt"]:
                            gloss_stats[gloss]["min_igt"] = current_igt
                            gloss_stats[gloss]["min_igt_entry"] = block["entry"]
                            gloss_stats[gloss]["min_igt_index"] = index_str
                    # OGT updates
                    if current_ogt is not None:
                        gloss_stats[gloss]["ogt_total"] += current_ogt
                        gloss_stats[gloss]["ogt_count"] += 1
                        index_str = f"index:{current_group_index}-{next_group_index}"
                        if current_ogt > gloss_stats[gloss]["max_ogt"]:
                            gloss_stats[gloss]["max_ogt"] = current_ogt
                            gloss_stats[gloss]["max_ogt_entry"] = block["entry"]
                            gloss_stats[gloss]["max_ogt_index"] = index_str
                        if current_ogt < gloss_stats[gloss]["min_ogt"]:
                            gloss_stats[gloss]["min_ogt"] = current_ogt
                            gloss_stats[gloss]["min_ogt_entry"] = block["entry"]
                            gloss_stats[gloss]["min_ogt_index"] = index_str
                    # TGT updates (only if both intervals are available)
                    if current_igt is not None and current_ogt is not None:
                        current_tgt = current_igt + block["duration_ms"] + current_ogt
                        # For TGT, combine the index info from IGT and OGT.
                        tgt_index_str = f"IGT:{prev_group_index}-{current_group_index} | OGT:{current_group_index}-{next_group_index}"
                        if current_tgt > gloss_stats[gloss]["max_tgt"]:
                            gloss_stats[gloss]["max_tgt"] = current_tgt
                            gloss_stats[gloss]["max_tgt_entry"] = block["entry"]
                            gloss_stats[gloss]["max_tgt_index"] = tgt_index_str
                        if current_tgt < gloss_stats[gloss]["min_tgt"]:
                            gloss_stats[gloss]["min_tgt"] = current_tgt
                            gloss_stats[gloss]["min_tgt_entry"] = block["entry"]
                            gloss_stats[gloss]["min_tgt_index"] = tgt_index_str

# Load gloss types from CSV and strip trailing caret characters.
gloss_types = load_gloss_types("/Volumes/IISY/DGSKorpus/all-types-dgs.csv")
gloss_types = {gloss.rstrip('^') for gloss in gloss_types}

# Initialize gloss statistics dictionary.
# For each gloss we track totals and counts for AGT, IGT, OGT,
# as well as min/max values and where (entry and SRT index) they occurred.
gloss_stats = {
    gloss: {"total_duration": 0, "count": 0,
            "igt_total": 0, "igt_count": 0,
            "ogt_total": 0, "ogt_count": 0,
            "max_duration": 0, "min_duration": float('inf'),
            "max_duration_entry": None, "min_duration_entry": None,
            "max_duration_index": None, "min_duration_index": None,
            "max_igt": 0, "min_igt": float('inf'),
            "max_igt_entry": None, "min_igt_entry": None,
            "max_igt_index": None, "min_igt_index": None,
            "max_ogt": 0, "min_ogt": float('inf'),
            "max_ogt_entry": None, "min_ogt_entry": None,
            "max_ogt_index": None, "min_ogt_index": None,
            "max_tgt": 0, "min_tgt": float('inf'),
            "max_tgt_entry": None, "min_tgt_entry": None,
            "max_tgt_index": None, "min_tgt_index": None}
    for gloss in gloss_types
}

base_path = "/Volumes/IISY/DGSKorpus/"

# Process directories that start with "entry_"
entries = [
    entry for entry in os.listdir(base_path)
    if os.path.isdir(os.path.join(base_path, entry)) and entry.startswith("entry_")
]

print(f"🚀 Processing {len(entries)} entries for gloss timing statistics...\n")

# Dictionary to track current gloss blocks per speaker.
current_sentence_blocks = {}  # e.g., { "A": [gloss blocks for A], "B": [gloss blocks for B], ... }

for entry in entries:
    folder_path = os.path.join(base_path, entry)
    filtered_path = os.path.join(folder_path, "filtered-transcript.srt")
    if not os.path.exists(filtered_path):
        print(f"⚠️  {entry} skipped: filtered-transcript.srt not found.\n")
        continue

    with open(filtered_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Split the file into SRT blocks (separated by empty lines)
    blocks = content.strip().split("\n\n")
    
    # Enumerate blocks so we can record the SRT block index.
    for block_index, block in enumerate(blocks, start=1):
        lines = block.splitlines()
        if len(lines) < 3:
            continue  # Skip incomplete blocks

        # Parse timestamp from the second line.
        timestamp_line = lines[1]
        match_ts = re.match(r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})", timestamp_line)
        if not match_ts:
            continue
        start_str, end_str = match_ts.groups()
        start_ms = parse_timestamp(start_str)
        end_ms = parse_timestamp(end_str)
        duration_ms = end_ms - start_ms

        # Extract speaker tag and text (assuming speaker is A, B, or C).
        match_speaker = re.match(r"^(A|B|C):\s*(.*)$", lines[2])
        if not match_speaker:
            continue
        speaker = match_speaker.group(1)
        
        # Combine all subtitle text lines (removing repeated speaker tags).
        text_content = " ".join(lines[2:])
        text_content = re.sub(r"^(A|B|C):\s*", "", text_content, flags=re.MULTILINE).strip()

        # Determine if this block is a gloss or a sentence.
        is_gloss = not text_content.endswith("_FULL_SENTENCE")

        if not is_gloss:
            # This is a sentence block.
            # If there is an existing gloss group for this speaker, process it first.
            if speaker in current_sentence_blocks and current_sentence_blocks[speaker]:
                process_sentence_blocks(current_sentence_blocks[speaker], gloss_stats, gloss_types)
                current_sentence_blocks[speaker] = []  # Reset for that speaker.
            # Sentence blocks are ignored for IGT/OGT calculations.
            continue
        else:
            # This is a gloss block.
            block_data = {
                "start_ms": start_ms,
                "end_ms": end_ms,
                "text": text_content,
                "duration_ms": duration_ms,
                "entry": entry,
                "index": str(block_index)
            }
            current_sentence_blocks.setdefault(speaker, []).append(block_data)
            
            # Update gloss stats for each gloss type found in the text (AGT).
            for gloss in gloss_types:
                if gloss in text_content:
                    gloss_stats[gloss]["total_duration"] += duration_ms
                    gloss_stats[gloss]["count"] += 1
                    # Update max/min duration (AGT) and record the entry and index.
                    if duration_ms > gloss_stats[gloss]["max_duration"]:
                        gloss_stats[gloss]["max_duration"] = duration_ms
                        gloss_stats[gloss]["max_duration_entry"] = entry
                        gloss_stats[gloss]["max_duration_index"] = block_data["index"]
                    if duration_ms < gloss_stats[gloss]["min_duration"]:
                        gloss_stats[gloss]["min_duration"] = duration_ms
                        gloss_stats[gloss]["min_duration_entry"] = entry
                        gloss_stats[gloss]["min_duration_index"] = block_data["index"]

    # After processing all blocks for an entry, flush any remaining gloss groups.
    for spkr, blocks_list in current_sentence_blocks.items():
        if blocks_list:
            process_sentence_blocks(blocks_list, gloss_stats, gloss_types)
            current_sentence_blocks[spkr] = []  # Reset after processing

    print(f"✅ {entry} processed.")

# Calculate average duration (AGT) for each gloss and prepare a results list.
# Tuple structure: (gloss, average_duration_ms, average_igt, average_ogt, average_tgt)
results = []
for gloss, stats in gloss_stats.items():
    if stats["count"] > 0:
        average_duration_ms = stats["total_duration"] / stats["count"]
        average_igt = stats["igt_total"] / stats["igt_count"] if stats["igt_count"] > 0 else 0
        average_ogt = stats["ogt_total"] / stats["ogt_count"] if stats["ogt_count"] > 0 else 0
        average_tgt = average_igt + average_duration_ms + average_ogt
        results.append((gloss, average_duration_ms, average_igt, average_ogt, average_tgt))

results.sort(key=lambda x: x[0])  # Sort results alphabetically by gloss

# Write the average results to a CSV file.
output_csv = os.path.join(base_path, "dgs_gloss_timing_stats.csv")
with open(output_csv, "w", newline='', encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["gloss", "average_agt_ms", "average_igt_ms", "average_ogt_ms", "average_tgt_ms"])
    for gloss, avg_ms, avg_igt, avg_ogt, avg_tgt in results:
        writer.writerow([gloss, avg_ms, avg_igt, avg_ogt, avg_tgt])

print(f"\n🎉 Gloss timing statistics written to {output_csv}")

# Print overall average values.
total_duration = sum(stats["total_duration"] for stats in gloss_stats.values())
total_count = sum(stats["count"] for stats in gloss_stats.values())
if total_count > 0:
    total_avg_time = total_duration / total_count
    print(f"\n📊 Total Average Gloss Time (AGT): {total_avg_time:.2f} ms")
else:
    print("\n⚠️ No gloss occurrences found, cannot calculate total average gloss time.")

total_igt = sum(stats["igt_total"] for stats in gloss_stats.values())
total_igt_count = sum(stats["igt_count"] for stats in gloss_stats.values())
if total_igt_count > 0:
    total_avg_igt = total_igt / total_igt_count
    print(f"📊 Total Average Into-Gloss Time (IGT): {total_avg_igt:.2f} ms")
else:
    print("⚠️ No IGT occurrences found, cannot calculate total average IGT.")

total_ogt = sum(stats["ogt_total"] for stats in gloss_stats.values())
total_ogt_count = sum(stats["ogt_count"] for stats in gloss_stats.values())
if total_ogt_count > 0:
    total_avg_ogt = total_ogt / total_ogt_count
    print(f"📊 Total Average Out-of-Gloss Time (OGT): {total_avg_ogt:.2f} ms")
else:
    print("⚠️ No OGT occurrences found, cannot calculate total average OGT.")

total_avg_tgt = (total_avg_igt if total_igt_count > 0 else 0) + (total_avg_time if total_count > 0 else 0) + (total_avg_ogt if total_ogt_count > 0 else 0)
print(f"📊 Total Average Total Gloss Time (TGT): {total_avg_tgt:.2f} ms")


# --- Additional Statistics: Top-5 Lowest/Highest Averages ---
# For each gloss in results, the tuple is:
# (gloss, average_duration_ms, average_igt, average_ogt, average_tgt)
results_by_agt = sorted(results, key=lambda x: x[1])
results_by_igt = sorted(results, key=lambda x: x[2])
results_by_ogt = sorted(results, key=lambda x: x[3])
results_by_tgt = sorted(results, key=lambda x: x[4])

def print_top_5(metric_name, sorted_list, metric_index):
    """Print top 5 lowest and highest average values for a given metric.
       metric_index: 1 for AGT, 2 for IGT, 3 for OGT, 4 for TGT in the results tuple."""
    lowest = [(item[0], item[metric_index]) for item in sorted_list[:5]]
    highest = [(item[0], item[metric_index]) for item in sorted_list[-5:]][::-1]
    print(f"\nTop 5 lowest {metric_name} averages:")
    for gloss, value in lowest:
        print(f"  {gloss}: {value:.2f} ms")
    print(f"\nTop 5 highest {metric_name} averages:")
    for gloss, value in highest:
        print(f"  {gloss}: {value:.2f} ms")

print_top_5("AGT (Average Gloss Time)", results_by_agt, 1)
print_top_5("IGT (Into-Gloss Time)", results_by_igt, 2)
print_top_5("OGT (Out-of-Gloss Time)", results_by_ogt, 3)
print_top_5("TGT (Total Gloss Time)", results_by_tgt, 4)


# --- Additional Statistics: Top-5 Lowest/Highest Minimum and Maximum (Single Occurrence) Values with Entry and Index ---
# For each metric, build a tuple:
# (gloss, min_value, min_entry, min_index, max_value, max_entry, max_index)
agt_stats = []
igt_stats = []
ogt_stats = []
tgt_stats = []
for gloss, stats in gloss_stats.items():
    if stats["count"] > 0:
        agt_stats.append((
            gloss,
            stats["min_duration"] if stats["min_duration"] != float('inf') else 0,
            stats["min_duration_entry"],
            stats["min_duration_index"],
            stats["max_duration"],
            stats["max_duration_entry"],
            stats["max_duration_index"]
        ))
        igt_stats.append((
            gloss,
            stats["min_igt"] if stats["min_igt"] != float('inf') else 0,
            stats["min_igt_entry"],
            stats["min_igt_index"],
            stats["max_igt"],
            stats["max_igt_entry"],
            stats["max_igt_index"]
        ))
        ogt_stats.append((
            gloss,
            stats["min_ogt"] if stats["min_ogt"] != float('inf') else 0,
            stats["min_ogt_entry"],
            stats["min_ogt_index"],
            stats["max_ogt"],
            stats["max_ogt_entry"],
            stats["max_ogt_index"]
        ))
        tgt_stats.append((
            gloss,
            stats["min_tgt"] if stats["min_tgt"] != float('inf') else 0,
            stats["min_tgt_entry"],
            stats["min_tgt_index"],
            stats["max_tgt"],
            stats["max_tgt_entry"],
            stats["max_tgt_index"]
        ))

def print_top_n_metric(n, metric_name, metric_list):
    """
    Print top 5 lowest and highest values for a given metric from a list of tuples.
    Each tuple: (gloss, min_val, min_entry, min_index, max_val, max_entry, max_index).
    """
    sorted_by_min = sorted(metric_list, key=lambda x: x[1])
    sorted_by_max = sorted(metric_list, key=lambda x: x[4])
    
    print(f"\nTop {n} lowest {metric_name}:")
    for gloss, min_val, min_entry, min_index, _, _, _ in sorted_by_min[:n]:
        print(f"  {gloss}: {min_val} ms (entry: {min_entry}, {min_index})")
    print(f"\nTop {n} highest {metric_name}:")
    for gloss, _, _, _, max_val, max_entry, max_index in sorted_by_max[-n:][::-1]:
        print(f"  {gloss}: {max_val} ms (entry: {max_entry}, {max_index})")

print_top_n_metric(100, "AGT (Gloss Duration)", agt_stats)
print_top_n_metric(5, "IGT (Into-Gloss Time)", igt_stats)
print_top_n_metric(5, "OGT (Out-of-Gloss Time)", ogt_stats)
print_top_n_metric(5, "TGT (Total Gloss Time)", tgt_stats)
