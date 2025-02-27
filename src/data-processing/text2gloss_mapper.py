import os
import csv
from pysrt import SubRipFile

# Define the base path to the DGSKorpus folder
base_path = "/Volumes/IISY/DGSKorpus/"
# Define a variable to process all folders
process_all_folders = True  # Set to True to process all subfolders, False to process just a single test folder

# Function to process a single folder
def process_folder(folder_path):
    transcript_path = os.path.join(folder_path, "filtered-transcript.srt")
    output_csv_path = os.path.join(folder_path, "text2gloss-filtered-by-all-types.csv")

    # Read the .srt file
    try:
        srt_file = SubRipFile.open(transcript_path)
    except Exception as e:
        print(f"Error reading transcript file in {folder_path}: {e}")
        return []

    rows = []
    
    # Use dictionaries to track the last full sentence and its glosses for each speaker
    last_sentence = {}   # e.g., {'A': "Full sentence", 'B': "Full sentence"}
    gloss_entries = {}   # e.g., {'A': [list of glosses], 'B': [list of glosses]}

    # Iterate through the subtitles in the transcript
    for subtitle in srt_file:
        text = subtitle.text.strip()
        
        if text.startswith("A:") or text.startswith("B:"):
            person_label, content = text.split(":", 1)
            person_label = person_label.strip()
            content = content.strip()

            # Check if it's a full sentence (One or more words, starts with a upper-case letter followed by lower-case letter)
            words = content.split()
            if words and len(words[0]) >= 2 and words[0][0].isupper() and words[0][1].islower():
                # Before overwriting, if there's an existing sentence with glosses for this person, save it.
                if person_label in last_sentence and gloss_entries.get(person_label):
                    rows.append([last_sentence[person_label]] + gloss_entries[person_label])
                
                # Update this person's sentence and reset their gloss list.
                last_sentence[person_label] = content
                gloss_entries[person_label] = []
            else:
                # This is a gloss; add it to the speaker's gloss list if a sentence exists.
                if person_label in last_sentence:
                    gloss_entries.setdefault(person_label, []).append(content)
    
    # After processing all subtitles, add any remaining entries for each speaker.
    for person, sentence in last_sentence.items():
        if gloss_entries.get(person):
            rows.append([sentence] + gloss_entries[person])
    
    # Write the rows to a CSV file
    try:
        with open(output_csv_path, mode='w', newline='', encoding='utf-8') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["Full Sentence", "Words..."])  # Header row
            csv_writer.writerows(rows)
        print(f"CSV file created at {output_csv_path}")
    except Exception as e:
        print(f"Error writing to CSV file in {folder_path}: {e}")

    return rows

# Process all folders or just one
combined_rows = []
if process_all_folders:
    # Iterate through all subfolders in the base path
    for entry in os.listdir(base_path):
        folder_path = os.path.join(base_path, entry)
        if os.path.isdir(folder_path):
            print(f"Processing folder: {folder_path}")
            combined_rows.extend(process_folder(folder_path))

    # Write the combined CSV file
    combined_csv_path = os.path.join(base_path, "dgs-text2gloss-combined-filtered-by-all-types.csv")
    try:
        with open(combined_csv_path, mode='w', newline='', encoding='utf-8') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["Full Sentence", "Words..."])  # Header row
            csv_writer.writerows(combined_rows)
        print(f"Combined CSV file created at {combined_csv_path}")
    except Exception as e:
        print(f"Error writing combined CSV file: {e}")
else:
    # Process only the example folder
    example_folder = os.path.join(base_path, "entry_3")
    print(f"Processing example folder: {example_folder}")
    process_folder(example_folder)
