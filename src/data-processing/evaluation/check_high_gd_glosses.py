import csv
import os

threshold = 2000

# Function to parse an SRT file, returning blocks with timestamp and content
def parse_srt(srt_path):
    blocks = {}
    with open(srt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.isdigit():  # Block index
            index = line
            i += 1
            if i < len(lines):
                timestamp_line = lines[i].strip()
                i += 1
                content_lines = []
                while i < len(lines) and lines[i].strip() != '':
                    content_lines.append(lines[i].strip())
                    i += 1
                content = ' '.join(content_lines)
                blocks[index] = (timestamp_line, content)
            else:
                break
        else:
            i += 1
    return blocks

# Function to parse a timestamp line into start and end times
def parse_timestamp_line(timestamp_line):
    parts = timestamp_line.split(" --> ")
    if len(parts) != 2:
        raise ValueError(f"Invalid timestamp line: {timestamp_line}")
    start = parts[0].strip()
    end = parts[1].strip()
    return start, end

# Define paths (adjust these as needed for your environment)
csv_path = f'/Volumes/IISY/DGSKorpus/dgs-gloss-times/gd_above_{threshold}ms.csv'
base_path = '/Volumes/IISY/DGSKorpus/'
output_non_matching_csv_path = f'/Volumes/IISY/DGSKorpus/dgs-gloss-times/NO_MATCH_high_gd_glosses_with_no_matching_timestamp_{threshold}.csv'
output_matching_csv_path = f'/Volumes/IISY/DGSKorpus/dgs-gloss-times/MATCH_high_gd_glosses_with_matching_timestamp_{threshold}.csv'

count = 0
parsed_srt_cache = {}
non_matching_rows = []
matching_rows = []  # New list for rows where the condition is True

# Read all rows from the CSV file for processing
with open(csv_path, 'r', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    rows = list(reader)
total_rows = len(rows)

current_percentage = 0

# Process each row
for i, row in enumerate(rows, start=1):
    entry = row['entry']            # e.g., 'entry_272'
    block_index = row['block_index']  # e.g., '767'
    speaker = row['speaker']        # e.g., 'A' or 'B'
    
    # Construct path to the entry folder and corresponding speaker SRT file
    entry_path = os.path.join(base_path, entry)
    srt_file = 'speaker-a.srt' if speaker == 'A' else 'speaker-b.srt'
    srt_path = os.path.join(entry_path, srt_file)
    
    # Cache or parse the speaker SRT file
    if srt_path not in parsed_srt_cache:
        parsed_srt_cache[srt_path] = parse_srt(srt_path)
    blocks = parsed_srt_cache[srt_path]
    
    # Get the block data (timestamp_line, content) for the specified block
    block_data = blocks.get(block_index, (None, ''))
    timestamp_line, content = block_data

    condition_met = False

    if timestamp_line is not None:
        # Parse the end timestamp of the speaker block
        start_timestamp, end_timestamp = parse_timestamp_line(timestamp_line)
        # Determine the path to transcript.srt
        og_transcript_srt_path = os.path.join(entry_path, "transcript.srt")
        
        # Cache or parse transcript.srt
        if og_transcript_srt_path not in parsed_srt_cache:
            parsed_srt_cache[og_transcript_srt_path] = parse_srt(og_transcript_srt_path)
        og_transcript_blocks = parsed_srt_cache[og_transcript_srt_path]
        
        # Check transcript.srt for a block with matching start or end timestamp and different content
        for ts_line, og_transcript_content in og_transcript_blocks.values():
            start, end = parse_timestamp_line(ts_line)
            # start OR end are equal BUT NOT BOTH start AND end are equal! (Also the text content is different for further differentiation)
            if (start == end_timestamp or end == end_timestamp) and not (start == start_timestamp and end == end_timestamp) and og_transcript_content != content:
                count += 1
                condition_met = True
                break

    if condition_met:
        matching_rows.append(row)
    else:
        non_matching_rows.append(row)

    # Progress logging with emoji
    new_percentage = int((i / total_rows) * 100)
    if new_percentage > current_percentage:
        print(f"🚀 Progress: {new_percentage}% ({i}/{total_rows} rows processed)")
        current_percentage = new_percentage

# Write non-matching rows to a new CSV file
with open(output_non_matching_csv_path, 'w', encoding='utf-8', newline='') as outfile:
    writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
    writer.writeheader()
    writer.writerows(non_matching_rows)

# Write matching rows to a new CSV file
with open(output_matching_csv_path, 'w', encoding='utf-8', newline='') as outfile:
    writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
    writer.writeheader()
    writer.writerows(matching_rows)

# Explanatory print statement
print("\nExplanation of the count:")
print(f"The following count represents the number of glosses with a high GD (above {threshold}ms) where the end timestamp of the specified block in the speakers SRT file matches either the start or end timestamp in a different block of the original 'transcript.srt' from the DGS Korpus. The text content MUST be different from the original gloss (to exclude matching with the exact same srt block).")
print()
print(f"Total count of rows meeting the condition: {count}/{total_rows} ({(count / total_rows) * 100:.2f}%)")
print(f"Non-matching rows (no conditions met): {len(non_matching_rows)}/{total_rows} ({(len(non_matching_rows) / total_rows) * 100:.2f}%)")
print()
print(f"Rows with matching transcript timestamps have been written to: {output_matching_csv_path}")
print(f"Rows with no matching transcript timestamps have been written to: {output_non_matching_csv_path}")
