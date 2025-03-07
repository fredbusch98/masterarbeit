import os
import re
import csv
import pysrt  # To handle SRT parsing
from all_types import load_gloss_types

def parse_timestamp(timestamp_str):
    """Convert a timestamp 'HH:MM:SS,mmm' into milliseconds (as integer)."""
    hours, minutes, seconds, millis = re.split('[: ,]', timestamp_str)
    return int(hours) * 3600000 + int(minutes) * 60000 + int(seconds) * 1000 + int(millis)

def process_sentence_blocks(blocks, gloss_stats, gloss_types):
    """
    Process gloss blocks from the same sentence & speaker to calculate IGT, OGT, and TGT.
    Updates the maximum and minimum (single-occurrence) values for each gloss occurrence,
    and records the SRT block indices used in the calculation.
    """
    if not blocks:
        return

    # --- New Handling for Single-Gloss Sentences ---
    if len(blocks) == 1:
        # For a single gloss, there is no preceding or following gloss.
        # Define IGT and OGT as 0 and TGT equals the gloss duration.
        block = blocks[0]
        for gloss in gloss_types:
            if gloss in block["text"]:
                current_tgt = block["duration_ms"]  # TGT equals AGT here
                # Update TGT statistics using the block's own index/speaker info.
                if current_tgt < gloss_stats[gloss]["min_tgt"]:
                    gloss_stats[gloss]["min_tgt"] = current_tgt
                    gloss_stats[gloss]["min_tgt_entry"] = block["entry"]
                    gloss_stats[gloss]["min_tgt_index"] = block["index"]
                    gloss_stats[gloss]["min_tgt_speaker"] = block["speaker"]
                if current_tgt > gloss_stats[gloss]["max_tgt"]:
                    gloss_stats[gloss]["max_tgt"] = current_tgt
                    gloss_stats[gloss]["max_tgt_entry"] = block["entry"]
                    gloss_stats[gloss]["max_tgt_index"] = block["index"]
                    gloss_stats[gloss]["max_tgt_speaker"] = block["speaker"]
        return
    # --------------------------------------------------

    # Group blocks by timestamp (to handle identical timestamps)
    timestamp_groups = {}
    for block in blocks:
        key = (block["start_ms"], block["end_ms"])
        timestamp_groups.setdefault(key, []).append(block)
    
    # Sort groups by the starting timestamp
    sorted_groups = sorted(timestamp_groups.items(), key=lambda x: x[0][0])
    
    # Process each group to compute IGT, OGT and TGT
    for i, (timestamp, group_blocks) in enumerate(sorted_groups):
        # Default to 0 for missing intervals
        current_igt = 0  
        current_ogt = 0  
        
        # Get a representative index for the current group
        current_group_index = group_blocks[0].get("index", "")
        
        # Compute IGT if not the first group
        if i > 0:
            prev_group = sorted_groups[i-1][1]
            prev_group_index = prev_group[0].get("index", "0")
            current_igt = timestamp[0] - sorted_groups[i-1][0][1]
        # Compute OGT if not the last group
        if i < len(sorted_groups) - 1:
            next_group = sorted_groups[i+1][1]
            next_group_index = next_group[0].get("index", "0")
            current_ogt = next_group[0]["start_ms"] - timestamp[1]
        
        # Update each gloss occurrence in this group
        for block in group_blocks:
            for gloss in gloss_types:
                if gloss in block["text"]:
                    # IGT updates (only update if there is a preceding group)
                    if i > 0:
                        gloss_stats[gloss]["igt_total"] += current_igt
                        gloss_stats[gloss]["igt_count"] += 1
                        index_str = f"index:{prev_group_index}-{current_group_index}"
                        if current_igt > gloss_stats[gloss]["max_igt"]:
                            gloss_stats[gloss]["max_igt"] = current_igt
                            gloss_stats[gloss]["max_igt_entry"] = block["entry"]
                            gloss_stats[gloss]["max_igt_index"] = index_str
                            gloss_stats[gloss]["max_igt_speaker"] = block["speaker"]
                        if current_igt < gloss_stats[gloss]["min_igt"]:
                            gloss_stats[gloss]["min_igt"] = current_igt
                            gloss_stats[gloss]["min_igt_entry"] = block["entry"]
                            gloss_stats[gloss]["min_igt_index"] = index_str
                            gloss_stats[gloss]["min_igt_speaker"] = block["speaker"]
                    # OGT updates (only update if there is a following group)
                    if i < len(sorted_groups) - 1:
                        gloss_stats[gloss]["ogt_total"] += current_ogt
                        gloss_stats[gloss]["ogt_count"] += 1
                        index_str = f"index:{current_group_index}-{next_group_index}"
                        if current_ogt > gloss_stats[gloss]["max_ogt"]:
                            gloss_stats[gloss]["max_ogt"] = current_ogt
                            gloss_stats[gloss]["max_ogt_entry"] = block["entry"]
                            gloss_stats[gloss]["max_ogt_index"] = index_str
                            gloss_stats[gloss]["max_ogt_speaker"] = block["speaker"]
                        if current_ogt < gloss_stats[gloss]["min_ogt"]:
                            gloss_stats[gloss]["min_ogt"] = current_ogt
                            gloss_stats[gloss]["min_ogt_entry"] = block["entry"]
                            gloss_stats[gloss]["min_ogt_index"] = index_str
                            gloss_stats[gloss]["min_ogt_speaker"] = block["speaker"]
                    
                    # Calculate TGT: (IGT + AGT + OGT)
                    current_tgt = current_igt + block["duration_ms"] + current_ogt
                    # Build an index string that reflects available intervals
                    if i > 0 and i < len(sorted_groups) - 1:
                        tgt_index_str = f"IGT:{prev_group_index}-{current_group_index} | OGT:{current_group_index}-{next_group_index}"
                    elif i > 0:
                        tgt_index_str = f"IGT:{prev_group_index}-{current_group_index} | OGT:None"
                    elif i < len(sorted_groups) - 1:
                        tgt_index_str = f"IGT:None | OGT:{current_group_index}-{next_group_index}"
                    else:
                        tgt_index_str = block["index"]
    
                    # Update min and max TGT
                    if current_tgt < gloss_stats[gloss]["min_tgt"]:
                        gloss_stats[gloss]["min_tgt"] = current_tgt
                        gloss_stats[gloss]["min_tgt_entry"] = block["entry"]
                        gloss_stats[gloss]["min_tgt_index"] = tgt_index_str
                        gloss_stats[gloss]["min_tgt_speaker"] = block["speaker"]
                    if current_tgt > gloss_stats[gloss]["max_tgt"]:
                        gloss_stats[gloss]["max_tgt"] = current_tgt
                        gloss_stats[gloss]["max_tgt_entry"] = block["entry"]
                        gloss_stats[gloss]["max_tgt_index"] = tgt_index_str
                        gloss_stats[gloss]["max_tgt_speaker"] = block["speaker"]

def process_srt_file(srt_path, speakers_to_process, gloss_stats, gloss_types, entry):
    """Process an SRT file for specified speakers, updating gloss statistics using pysrt."""
    if not os.path.exists(srt_path):
        print(f"⚠️  {srt_path} not found.\n")
        return

    # Open the SRT file using pysrt
    subtitles = pysrt.open(srt_path, encoding="utf-8")
    current_sentence_blocks = {}  # Per speaker

    for subtitle in subtitles:
        block_index = subtitle.index
        start_ms = subtitle.start.ordinal  # pysrt gives time in milliseconds
        end_ms = subtitle.end.ordinal
        duration_ms = end_ms - start_ms

        # Split the subtitle text into lines and try to extract speaker information
        lines = subtitle.text.splitlines()
        if not lines:
            continue

        # Expect the first line to have the speaker info (e.g., "A: some text")
        first_line = lines[0].strip()
        match_speaker = re.match(r"^(A|B):\s*(.*)$", first_line)
        if not match_speaker:
            continue
        speaker = match_speaker.group(1)
        if speaker not in speakers_to_process:
            continue

        # Remove the speaker label from the text content
        text_content = re.sub(r"^(A|B):\s*", "", subtitle.text, flags=re.MULTILINE).strip()
        is_gloss = not text_content.endswith("_FULL_SENTENCE")

        if not is_gloss:
            # Sentence block: process any accumulated gloss blocks for this speaker
            if speaker in current_sentence_blocks and current_sentence_blocks[speaker]:
                process_sentence_blocks(current_sentence_blocks[speaker], gloss_stats, gloss_types)
                current_sentence_blocks[speaker] = []
            continue
        else:
            # Gloss block: build a dictionary with necessary data
            block_data = {
                "start_ms": start_ms,
                "end_ms": end_ms,
                "text": text_content,
                "duration_ms": duration_ms,
                "entry": entry,
                "index": str(block_index),
                "speaker": speaker
            }
            current_sentence_blocks.setdefault(speaker, []).append(block_data)

            # Update AGT stats for each gloss occurrence found in the text
            for gloss in gloss_types:
                if gloss in text_content:
                    gloss_stats[gloss]["total_duration"] += duration_ms
                    gloss_stats[gloss]["count"] += 1
                    if duration_ms > gloss_stats[gloss]["max_duration"]:
                        gloss_stats[gloss]["max_duration"] = duration_ms
                        gloss_stats[gloss]["max_duration_entry"] = entry
                        gloss_stats[gloss]["max_duration_index"] = block_data["index"]
                        gloss_stats[gloss]["max_duration_speaker"] = speaker
                    if duration_ms < gloss_stats[gloss]["min_duration"]:
                        gloss_stats[gloss]["min_duration"] = duration_ms
                        gloss_stats[gloss]["min_duration_entry"] = entry
                        gloss_stats[gloss]["min_duration_index"] = block_data["index"]
                        gloss_stats[gloss]["min_duration_speaker"] = speaker

    # Process any remaining gloss blocks for each speaker
    for spkr, blocks_list in current_sentence_blocks.items():
        if blocks_list:
            process_sentence_blocks(blocks_list, gloss_stats, gloss_types)

# Load gloss types
gloss_types = load_gloss_types("/Volumes/IISY/DGSKorpus/all-types-dgs.csv")
gloss_types = {gloss.rstrip('^') for gloss in gloss_types}

# Initialize gloss statistics dictionary
gloss_stats = {
    gloss: {
        "total_duration": 0, "count": 0,
        "igt_total": 0, "igt_count": 0,
        "ogt_total": 0, "ogt_count": 0,
        "max_duration": 0, "min_duration": float('inf'),
        "max_duration_entry": None, "min_duration_entry": None,
        "max_duration_index": None, "min_duration_index": None,
        "max_duration_speaker": None, "min_duration_speaker": None,
        "max_igt": 0, "min_igt": float('inf'),
        "max_igt_entry": None, "min_igt_entry": None,
        "max_igt_index": None, "min_igt_index": None,
        "max_igt_speaker": None, "min_igt_speaker": None,
        "max_ogt": 0, "min_ogt": float('inf'),
        "max_ogt_entry": None, "min_ogt_entry": None,
        "max_ogt_index": None, "min_ogt_index": None,
        "max_ogt_speaker": None, "min_ogt_speaker": None,
        "max_tgt": 0, "min_tgt": float('inf'),
        "max_tgt_entry": None, "min_tgt_entry": None,
        "max_tgt_index": None, "min_tgt_index": None,
        "max_tgt_speaker": None, "min_tgt_speaker": None
    } for gloss in gloss_types
}

base_path = "/Volumes/IISY/DGSKorpus/"

# Get entry directories
entries = [
    entry for entry in os.listdir(base_path)
    if os.path.isdir(os.path.join(base_path, entry)) and entry.startswith("entry_")
]

print(f"🚀 Processing {len(entries)} entries for gloss timing statistics...\n")

# Process each entry
for entry in entries:
    folder_path = os.path.join(base_path, entry)
    # Process speaker-a.srt for A
    process_srt_file(os.path.join(folder_path, "speaker-a.srt"), ["A"], gloss_stats, gloss_types, entry)
    # Process speaker-b.srt for B
    process_srt_file(os.path.join(folder_path, "speaker-b.srt"), ["B"], gloss_stats, gloss_types, entry)
    print(f"✅ {entry} processed.")

# Calculate average metrics
results = []
for gloss, stats in gloss_stats.items():
    if stats["count"] > 0:
        average_duration_ms = stats["total_duration"] / stats["count"]
        average_igt = stats["igt_total"] / stats["igt_count"] if stats["igt_count"] > 0 else 0
        average_ogt = stats["ogt_total"] / stats["ogt_count"] if stats["ogt_count"] > 0 else 0
        average_tgt = average_igt + average_duration_ms + average_ogt
        results.append((gloss, average_duration_ms, average_igt, average_ogt, average_tgt))

results.sort(key=lambda x: x[0])

# Write results to CSV
output_csv = os.path.join(base_path, "dgs-gloss-times/dgs-gloss-timing-stats-split-speaker.csv")
with open(output_csv, "w", newline='', encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["gloss", "average_agt_ms", "average_igt_ms", "average_ogt_ms", "average_tgt_ms"])
    for gloss, avg_ms, avg_igt, avg_ogt, avg_tgt in results:
        writer.writerow([gloss, avg_ms, avg_igt, avg_ogt, avg_tgt])

print(f"\n🎉 Gloss timing statistics written to {output_csv}")

# Create lists for each metric from the results
agt_list = [(gloss, avg_ms) for gloss, avg_ms, _, _, _ in results]
igt_list = [(gloss, avg_igt) for gloss, _, avg_igt, _, _ in results]
ogt_list = [(gloss, avg_ogt) for gloss, _, _, avg_ogt, _ in results]
tgt_list = [(gloss, avg_tgt) for gloss, _, _, _, avg_tgt in results]

# Sort each list by the average time in descending order
agt_list.sort(key=lambda x: x[1], reverse=True)
igt_list.sort(key=lambda x: x[1], reverse=True)
ogt_list.sort(key=lambda x: x[1], reverse=True)
tgt_list.sort(key=lambda x: x[1], reverse=True)

# Define file paths for the new CSV files
agt_csv = os.path.join(base_path, "dgs-gloss-times/dgs-gloss-agt-stats.csv")
igt_csv = os.path.join(base_path, "dgs-gloss-times/dgs-gloss-igt-stats.csv")
ogt_csv = os.path.join(base_path, "dgs-gloss-times/dgs-gloss-ogt-stats.csv")
tgt_csv = os.path.join(base_path, "dgs-gloss-times/dgs-gloss-tgt-stats.csv")

# Write AGT CSV
with open(agt_csv, "w", newline='', encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["gloss", "average_agt_ms"])
    for gloss, avg_agt in agt_list:
        writer.writerow([gloss, avg_agt])

# Write IGT CSV
with open(igt_csv, "w", newline='', encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["gloss", "average_igt_ms"])
    for gloss, avg_igt in igt_list:
        writer.writerow([gloss, avg_igt])

# Write OGT CSV
with open(ogt_csv, "w", newline='', encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["gloss", "average_ogt_ms"])
    for gloss, avg_ogt in ogt_list:
        writer.writerow([gloss, avg_ogt])

# Write TGT CSV
with open(tgt_csv, "w", newline='', encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["gloss", "average_tgt_ms"])
    for gloss, avg_tgt in tgt_list:
        writer.writerow([gloss, avg_tgt])

# Print confirmation for the new files
print(f"\n🎉 Separate gloss timing statistics written to:")
print(f"  - {agt_csv}")
print(f"  - {igt_csv}")
print(f"  - {ogt_csv}")
print(f"  - {tgt_csv}")

# Print overall averages
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

total_avg_tgt = (total_avg_igt if total_igt_count > 0 else 0) + \
                (total_avg_time if total_count > 0 else 0) + \
                (total_avg_ogt if total_ogt_count > 0 else 0)
print(f"📊 Total Average Total Gloss Time (TGT): {total_avg_tgt:.2f} ms")

# Top-5 Lowest/Highest Averages
results_by_agt = sorted(results, key=lambda x: x[1])
results_by_igt = sorted(results, key=lambda x: x[2])
results_by_ogt = sorted(results, key=lambda x: x[3])
results_by_tgt = sorted(results, key=lambda x: x[4])

def print_top_5(metric_name, sorted_list, metric_index):
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

# Top-5 Lowest/Highest Single Occurrences
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
            stats["min_duration_speaker"],
            stats["max_duration"],
            stats["max_duration_entry"],
            stats["max_duration_index"],
            stats["max_duration_speaker"]
        ))
        igt_stats.append((
            gloss,
            stats["min_igt"] if stats["min_igt"] != float('inf') else 0,
            stats["min_igt_entry"],
            stats["min_igt_index"],
            stats["min_igt_speaker"],
            stats["max_igt"],
            stats["max_igt_entry"],
            stats["max_igt_index"],
            stats["max_igt_speaker"]
        ))
        ogt_stats.append((
            gloss,
            stats["min_ogt"] if stats["min_ogt"] != float('inf') else 0,
            stats["min_ogt_entry"],
            stats["min_ogt_index"],
            stats["min_ogt_speaker"],
            stats["max_ogt"],
            stats["max_ogt_entry"],
            stats["max_ogt_index"],
            stats["max_ogt_speaker"]
        ))
        tgt_stats.append((
            gloss,
            stats["min_tgt"] if stats["min_tgt"] != float('inf') else 0,
            stats["min_tgt_entry"],
            stats["min_tgt_index"],
            stats["min_tgt_speaker"],
            stats["max_tgt"],
            stats["max_tgt_entry"],
            stats["max_tgt_index"],
            stats["max_tgt_speaker"]
        ))

def print_top_n_metric(n, metric_name, metric_list):
    sorted_by_min = sorted(metric_list, key=lambda x: x[1])
    sorted_by_max = sorted(metric_list, key=lambda x: x[5])
    print(f"\nTop {n} lowest {metric_name}:")
    for gloss, min_val, min_entry, min_index, min_speaker, _, _, _, _ in sorted_by_min[:n]:
        print(f"  {gloss}: {min_val} ms (entry: {min_entry}, index: {min_index}, {min_speaker})")
    print(f"\nTop {n} highest {metric_name}:")
    for gloss, _, _, _, _, max_val, max_entry, max_index, max_speaker in sorted_by_max[-n:][::-1]:
        print(f"  {gloss}: {max_val} ms (entry: {max_entry}, index: {max_index}, {max_speaker})")

print_top_n_metric(250, "AGT (Gloss Duration)", agt_stats)
print_top_n_metric(5, "IGT (Into-Gloss Time)", igt_stats)
print_top_n_metric(5, "OGT (Out-of-Gloss Time)", ogt_stats)
print_top_n_metric(5, "TGT (Total Gloss Time)", tgt_stats)
