import os
import re

# Define the main folder path where entry_* folders are located
main_folder = '/Volumes/IISY/DGSKorpus'

# Function to count and collect lost glosses in a single SRT file
def count_lost_glosses(srt_path):
    """
    Counts lost glosses in an SRT file. A lost gloss is one that appears after a gloss
    with '_END_SENTENCE' and before the next '_FULL_SENTENCE'.
    
    Args:
        srt_path (str): Path to the SRT file
        
    Returns:
        list: List of tuples (srt_path, entry_index, gloss_text) for each lost gloss
    """
    # Read the SRT file
    with open(srt_path, 'r') as f:
        lines = f.readlines()

    # Parse the SRT into entries
    entries = []
    current_entry = []
    for line in lines:
        if line.strip().isdigit():
            if current_entry:
                entries.append(current_entry)
            current_entry = [line.strip()]  # Start with the entry index
        else:
            current_entry.append(line)
    if current_entry:
        entries.append(current_entry)

    # Process entries to find lost glosses
    lost_glosses = []
    after_end_sentence = False

    for entry in entries:
        # Check if entry has at least index, timestamp, and one text line
        if len(entry) >= 3:
            text_line = entry[2].strip()
            # Extract text after speaker tag (A: or B:)
            match = re.match(r'[A-B]:\s', text_line)
            if match:
                clean_text = text_line[match.end():].strip()
                # Check for sentence markers
                if clean_text.endswith('_FULL_SENTENCE'):
                    after_end_sentence = False
                elif clean_text.endswith('_END_SENTENCE'):
                    after_end_sentence = True
                elif after_end_sentence:
                    # This is a lost gloss
                    lost_glosses.append((srt_path, entry[0], clean_text))
            else:
                print(f"⚠️ Warning: No speaker tag in {srt_path}, entry {entry[0]}")
        else:
            print(f"⚠️ Warning: Incomplete entry in {srt_path}, entry {entry[0]}")

    return lost_glosses

# Get all entry_* folders
entry_folders = [
    f for f in os.listdir(main_folder)
    if f.startswith('entry_') and os.path.isdir(os.path.join(main_folder, f))
]

# Collect all lost glosses across all files
all_lost_glosses = []
total_files = len(entry_folders) * 2  # Each entry has two speaker files (a & b)
processed_files = 0

print("🔍 Scanning for lost glosses...\n")

for entry_folder in entry_folders:
    for speaker in ['a', 'b']:
        srt_path = os.path.join(main_folder, entry_folder, f'speaker-{speaker}.srt')
        if os.path.exists(srt_path):
            lost_glosses = count_lost_glosses(srt_path)
            all_lost_glosses.extend(lost_glosses)
        else:
            print(f"❌ File not found: {srt_path}")

        # Update progress
        processed_files += 1
        percentage = (processed_files / total_files) * 100
        print(f"📂 Progress: {processed_files}/{total_files} files processed ({percentage:.1f}%)")

# Write lost glosses to a file
output_file = os.path.join(main_folder, 'lost-glosses.txt')
with open(output_file, 'w') as f:
    for path, idx, text in all_lost_glosses:
        f.write(f"In {path}, entry {idx}: {text}\n")

# Final summary
print("\n✅ Processing complete!")
print(f"📜 Total lost glosses found: {len(all_lost_glosses)}")
print(f"📁 Results saved to: {output_file}")
