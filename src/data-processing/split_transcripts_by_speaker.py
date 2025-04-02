import os
import re

# Define the main folder path where entry_* folders are located
main_folder = '/Volumes/IISY/DGSKorpus'

# Define a list of glosses that define full sentences and all their corresponding glosses to be exluded
exclude_sentence_glosses = ["$PROD", "$ORAL", "$ALPHA", "$ORAL^"]

# These glosses need to be excluded but the rest of the sentence can stay in the transcripts
excluded_glosses = ["$GEST", "$GEST-OFF", "$$EXTRA-LING-MAN", "$PMS"]

# Timestamp parsing functions
def parse_timestamp(ts):
    """Convert SRT timestamp (e.g., '00:00:02,160') to milliseconds."""
    time_part, ms_part = ts.split(',')
    hours, minutes, seconds = time_part.split(':')
    total_ms = (int(hours) * 3600000 + 
                int(minutes) * 60000 + 
                int(seconds) * 1000 + 
                int(ms_part))
    return total_ms

def parse_srt_timestamp(timestamp_line):
    """Parse SRT timestamp line (e.g., '00:00:00,240 --> 00:00:02,160') into start and end times in milliseconds."""
    start_str, end_str = timestamp_line.strip().split(' --> ')
    start_ms = parse_timestamp(start_str)
    end_ms = parse_timestamp(end_str)
    return start_ms, end_ms

# Improved SRT parsing function
def parse_srt(lines):
    """Parse SRT lines into a list of (timestamp, text_lines) tuples, excluding trailing blank lines."""
    entries = []
    current_entry = []
    for line in lines:
        if line.strip() == "":
            if current_entry and len(current_entry) >= 2:
                timestamp = current_entry[1]
                text_lines = current_entry[2:]
                # Remove trailing blank lines
                while text_lines and text_lines[-1].strip() == "":
                    text_lines.pop()
                entries.append((timestamp, text_lines))
            current_entry = []
        else:
            current_entry.append(line)
    # Handle the last entry
    if current_entry and len(current_entry) >= 2:
        timestamp = current_entry[1]
        text_lines = current_entry[2:]
        while text_lines and text_lines[-1].strip() == "":
            text_lines.pop()
        entries.append((timestamp, text_lines))
    return entries

# Entry processing function
def process_entries(entries, speaker):
    speaker_tag = f"{speaker}: "
    processed_entries = []
    i = 0
    
    while i < len(entries):
        entry = entries[i]
        timestamp_line = entry[0]
        _, end_ms = parse_srt_timestamp(timestamp_line)
        text_lines = entry[1]
        full_text = ' '.join(line.strip() for line in text_lines)
        clean_text = full_text[len(speaker_tag):].strip()
        
        if any(gloss in clean_text for gloss in excluded_glosses):
            i += 1
            continue

        if clean_text.endswith("_FULL_SENTENCE"):
            # Begin a sentence block
            sentence_entries = []
            # Add the current entry (it passed the extra-gloss check)
            sentence_entries.append(entry)
            j = i + 1
            
            # Gather subsequent entries belonging to the same sentence period
            while j < len(entries):
                next_entry = entries[j]
                next_timestamp_line = next_entry[0]
                next_start_ms, _ = parse_srt_timestamp(next_timestamp_line)
                next_text = ' '.join(line.strip() for line in next_entry[1])
                next_clean = next_text[len(speaker_tag):].strip()
                
                # Stop if we encounter a new sentence or the time exceeds the sentence block
                if next_clean.endswith("_FULL_SENTENCE") or next_start_ms > end_ms:
                    break

                # Skip entries with the defined excluded glosses
                if any(gloss in next_clean for gloss in excluded_glosses):
                    j += 1
                    continue
                sentence_entries.append(next_entry)
                j += 1

            # Before adding the sentence block, check for any excluded gloss (from the list)
            glosses = []
            for se in sentence_entries:
                se_text = ' '.join(line.strip() for line in se[1])
                se_clean = se_text[len(speaker_tag):].strip()
                glosses.extend(se_clean.split())
            if any(
                gloss in exclude_sentence_glosses or 
                (gloss.startswith('$ALPHA'))
                for gloss in glosses
            ):
                # Skip the entire sentence if any excluded gloss is found
                i = j
                continue
            else:
                processed_entries.extend(sentence_entries)
                # Append _END_SENTENCE to the last gloss of the sentence
                if sentence_entries:
                    last_entry = sentence_entries[-1]
                    if last_entry[1]:
                        last_text_line = last_entry[1][-1]
                        last_entry[1][-1] = last_text_line.rstrip() + "_END_SENTENCE\n"
                i = j
        else:
            # Non-sentence entry / lost gloss
            if any(gloss in clean_text for gloss in exclude_sentence_glosses):
                i += 1
                continue
            processed_entries.append(entry)
            i += 1
    
    return processed_entries

# Main execution
# Get a list of all subfolders named entry_0, entry_1, etc.
entry_folders = [
    f for f in os.listdir(main_folder)
    if f.startswith('entry_') and os.path.isdir(os.path.join(main_folder, f))
]

total_folders = len(entry_folders)
print(f"🚀 Starting to process {total_folders} folders...")

for idx, entry_folder in enumerate(entry_folders):
    transcript_path = os.path.join(main_folder, entry_folder, 'filtered-transcript.srt')
    
    if os.path.exists(transcript_path):
        # Read the SRT file
        with open(transcript_path, 'r') as file:
            lines = file.readlines()
        
        # Parse SRT entries using the improved parser
        entries = parse_srt(lines)
        
        # Split entries by speaker
        entries_a = []
        entries_b = []
        for entry in entries:
            timestamp, text_lines = entry
            if text_lines:
                first_line = text_lines[0].strip()
                match = re.match(r'([A-B]):\s', first_line)
                if match:
                    speaker = match.group(1)
                    if speaker == 'A':
                        entries_a.append((timestamp, text_lines))
                    elif speaker == 'B':
                        entries_b.append((timestamp, text_lines))
        
        # Process entries to append _END_SENTENCE to last gloss and apply exclusions
        entries_a = process_entries(entries_a, "A")
        entries_b = process_entries(entries_b, "B")
        
        # Write to speaker-a.srt
        speaker_a_path = os.path.join(main_folder, entry_folder, 'speaker-a.srt')
        with open(speaker_a_path, 'w') as f:
            for i, (timestamp, text_lines) in enumerate(entries_a, start=1):
                f.write(f"{i}\n")
                f.write(timestamp)
                for line in text_lines:
                    f.write(line)
                f.write("\n")
        
        # Write to speaker-b.srt
        speaker_b_path = os.path.join(main_folder, entry_folder, 'speaker-b.srt')
        with open(speaker_b_path, 'w') as f:
            for i, (timestamp, text_lines) in enumerate(entries_b, start=1):
                f.write(f"{i}\n")
                f.write(timestamp)
                for line in text_lines:
                    f.write(line)
                f.write("\n")
        
        # Progress update
        percentage = (idx + 1) / total_folders * 100
        print(f"[{round(percentage)}%] ✅ Processed folder {entry_folder}.")
    else:
        # Skip if transcript file is missing
        percentage = (idx + 1) / total_folders * 100
        print(f"⚠️ Skipped folder {entry_folder}: filtered-transcript.srt not found. Progress: {round(percentage)}%")

print("🎉 All folders checked successfully! Check the entry_* folders for speaker-a.srt and speaker-b.srt files.")
