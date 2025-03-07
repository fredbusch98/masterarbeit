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
                Explanation of the regex pattern:

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
                pattern = r"^(?:A:|B:|C:)\s*(?:[A-Z].*|(?:[0-9#]\S*(?:\s+\S+)+))$"

                contains_full_sentence = False
                if not contains_gloss:
                    contains_full_sentence = re.match(pattern, original_text_line)
                
                # Include the block if it meets the criteria
                if contains_gloss or contains_full_sentence:
                    lines[0] = str(new_index)  # Update index
                    # If it's a full sentence, add the prefix
                    if contains_full_sentence:
                        lines[2] = cleaned_text_line + "_FULL_SENTENCE"
                    else:
                        lines[2] = cleaned_text_line
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