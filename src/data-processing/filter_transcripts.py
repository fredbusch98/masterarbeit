import os
import re
from all_types import load_gloss_types

# This script filters and preprocesses the DGS Korpus 3 SRT transcripts for further preprocessing of the DGS Dataset!

def clean_line(line):
    """Clean a single text line by removing specific characters and adjusting formatting."""
    # Remove all asterisks
    line = line.replace("*", "")
    # Remove trailing carets or pipes
    line = line.strip("|^")
    # Remove leading pipes after A: or B: labels
    line = re.sub(r"^(A:|B:)\s*\|+(.*)", r"\1 \2", line)
    return line

def adjust_timestamp(timestamp_line):
    """
    Adjusts the end timestamp of an SRT timestamp line if the duration is greater than 2000ms.
    
    The timestamp_line should have the format:
      HH:MM:SS,mmm --> HH:MM:SS,mmm
      
    If the difference between end and start is more than 2000ms, the end timestamp
    will be set to exactly start timestamp + 2000ms.
    """
    pattern = r"(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})"
    match = re.match(pattern, timestamp_line)
    if not match:
        # If pattern not matched, return original line unmodified
        return timestamp_line
    
    start_h, start_m, start_s, start_ms, end_h, end_m, end_s, end_ms = map(int, match.groups())
    
    # Convert start and end times into milliseconds
    start_total = ((start_h * 3600 + start_m * 60 + start_s) * 1000) + start_ms
    end_total = ((end_h * 3600 + end_m * 60 + end_s) * 1000) + end_ms
    
    duration = end_total - start_total
    
    if duration > 2000:
        # Set duration to exactly 2000ms
        new_end_total = start_total + 2000
        new_hours = new_end_total // (3600 * 1000)
        remainder = new_end_total % (3600 * 1000)
        new_minutes = remainder // (60 * 1000)
        remainder %= (60 * 1000)
        new_seconds = remainder // 1000
        new_milliseconds = remainder % 1000
        # Build new end timestamp string
        new_end_str = f"{new_hours:02d}:{new_minutes:02d}:{new_seconds:02d},{new_milliseconds:03d}"
        # Build start timestamp string as originally provided
        start_str = f"{start_h:02d}:{start_m:02d}:{start_s:02d},{start_ms:03d}"
        return f"{start_str} --> {new_end_str}"
    else:
        return timestamp_line

# Load gloss types and set base path
gloss_types = load_gloss_types("/Volumes/IISY/DGSKorpus/all-types-dgs.csv")
base_path = "/Volumes/IISY/DGSKorpus/"

# Get all directories starting with "entry_"
entries = [
    entry for entry in os.listdir(base_path)
    if os.path.isdir(os.path.join(base_path, entry)) and entry.startswith("entry_")
]
total_entries = len(entries)

print(f"🚀 Starting processing of {total_entries} entries...\n")

for i, entry in enumerate(entries, start=1):
    folder_path = os.path.join(base_path, entry)
    progress_percent = (i / total_entries) * 100
    print(f"🔄 Processing {entry} ({i}/{total_entries}) - {progress_percent:.1f}% complete")
    
    transcript_path = os.path.join(folder_path, "transcript.srt")
    if os.path.exists(transcript_path):
        with open(transcript_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Split content into SRT blocks (separated by empty lines)
        blocks = content.strip().split("\n\n")
        filtered_blocks = []
        new_index = 1
        
        for block in blocks:
            lines = block.splitlines()
            # Ensure the block has exactly 3 lines: index, timestamp, text
            if len(lines) != 3:
                continue
            
            original_text_line = lines[2]  # Single text line (e.g., "A: GLOSS")
            cleaned_text_line = clean_line(original_text_line)
            
            # Check if the cleaned line contains '||' for splitting
            if "||" in cleaned_text_line:
                # Extract speaker and glosses
                match = re.match(r"^(A:|B:)\s*(.*)", cleaned_text_line)
                if match:
                    speaker = match.group(1)
                    glosses_str = match.group(2)
                    glosses = glosses_str.split("||")
                    # Create a new block for each gloss
                    for gloss in glosses:
                        new_text_line = f"{speaker} {gloss.strip()}"
                        new_block = [str(new_index), lines[1], new_text_line]
                        # For glosses, adjust the timestamp if necessary
                        new_block[1] = adjust_timestamp(new_block[1])
                        filtered_blocks.append("\n".join(new_block))
                        new_index += 1
                else:
                    print(f"Warning: Line does not match expected format in {entry}: {cleaned_text_line}")
            else:
                # Check if the cleaned line contains a gloss
                contains_gloss = any(
                    gloss.rstrip("^") in cleaned_text_line.rstrip("^")
                    for gloss in gloss_types
                )
                
                r"""
                Explanation of the full sentence regex pattern:

                ^(?:A:|B:|C:)      # Ensure the line starts with one of the speaker labels (A:, B:, or C:)
                \s*                # Allow optional whitespace after the label
                (?:                # Begin non-capturing group for alternation:
                    [A-Z].*        #  - If the first character after the label is an uppercase letter, match the rest of the line.
                    |              #  OR
                    (?:            #  - If the first character is a digit or '#':
                        [0-9#]\S*  #    Match the contiguous token starting with a digit or '#' (without spaces)
                        (?:\s+\S+)+#    Require at least one space followed by one or more non-space characters, ensuring there is more than one word.
                    )
                )
                $                  # End of line.
                """
                full_sentence_pattern = r"^(?:A:|B:|C:)\s*(?:[A-Z].*|(?:[0-9#]\S*(?:\s+\S+)+))$"

                contains_full_sentence = False
                if not contains_gloss:
                    contains_full_sentence = re.match(full_sentence_pattern, original_text_line)
                
                # Include the block if it meets the criteria
                if contains_gloss or contains_full_sentence:
                    lines[0] = str(new_index)  # Update index
                    # If it's a full sentence, add the prefix
                    if contains_full_sentence:
                        lines[2] = cleaned_text_line + "_FULL_SENTENCE"
                    else:
                        lines[2] = cleaned_text_line
                        # For glosses, adjust the timestamp if the duration is > 2000ms
                        lines[1] = adjust_timestamp(lines[1])
                    filtered_blocks.append("\n".join(lines))
                    new_index += 1
        
        filtered_filename = "filtered-transcript.srt"
        filtered_path = os.path.join(folder_path, filtered_filename)
        
        # Write the filtered blocks to the output file
        with open(filtered_path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(filtered_blocks))
        
        print(f"✅ {entry} processed successfully!\n")
    else:
        print(f"⚠️  {entry} skipped: transcript.srt not found.\n")

print("🎉 All entries processed!")
