import os
import csv
from pysrt import SubRipFile

# Define the base path to the DGSKorpus folder
base_path = "/Volumes/IISY/DGSKorpus/"
# Define a variable to process all folders
process_all_folders = True  # Set to True to process all subfolders, False to process just a single test folder

def process_speaker_transcript(transcript_path, speaker):
    """
    Process a single speaker's SRT file to extract full sentences and their glosses.
    
    Args:
        transcript_path (str): Path to the speaker's SRT file.
        speaker (str): Speaker label ("A" or "B").
    
    Returns:
        list: List of rows where each row is [full_sentence, gloss1, gloss2, ...].
    """
    srt_file = SubRipFile.open(transcript_path)
    rows = []
    last_sentence = None
    gloss_entries = []

    # Process each subtitle
    for subtitle in srt_file:
        text = subtitle.text.strip()
        if not text:
            continue
        
        # Remove the speaker tag (e.g., "A: " or "B: ")
        speaker_tag = f"{speaker}: "
        if text.startswith(speaker_tag):
            text = text[len(speaker_tag):].strip()
        else:
            print(f"Warning: Subtitle does not start with expected speaker tag '{speaker_tag}' in {transcript_path}")
            continue  # Skip if the tag doesn't match (optional, adjust as needed)
        
        # Process the cleaned text
        if text.endswith("_FULL_SENTENCE"):
            # If there's a previous sentence with glosses, save it
            if last_sentence is not None and gloss_entries:
                rows.append([last_sentence] + gloss_entries)
            # Set the new sentence, removing "_FULL_SENTENCE"
            last_sentence = text.replace("_FULL_SENTENCE", "").strip()
            gloss_entries = []
        else:
            # This is a gloss; add it to the list
            gloss_entries.append(text)

    # Save any remaining sentence with glosses
    if last_sentence is not None and gloss_entries:
        rows.append([last_sentence] + gloss_entries)

    return rows

def process_folder(folder_path):
    """
    Process a folder containing speaker-a.srt and speaker-b.srt, combining their results into one CSV.
    
    Args:
        folder_path (str): Path to the folder containing the SRT files.
    
    Returns:
        list: Combined list of rows from both speakers.
    """
    # Define paths to the speaker transcript files
    transcript_a_path = os.path.join(folder_path, "speaker-a.srt")
    transcript_b_path = os.path.join(folder_path, "speaker-b.srt")

    # Process each speaker's transcript
    rows_a = process_speaker_transcript(transcript_a_path, speaker="A")
    rows_b = process_speaker_transcript(transcript_b_path, speaker="B")

    # Combine the rows from both speakers
    combined_rows = rows_a + rows_b

    # Define the output CSV path
    output_csv_path = os.path.join(folder_path, "text2gloss-split-speaker.csv")

    # Write the combined rows to a CSV file
    try:
        with open(output_csv_path, mode='w', newline='', encoding='utf-8') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["Full Sentence", "Words..."])  # Header row
            csv_writer.writerows(combined_rows)
        print(f"CSV file created at {output_csv_path}")
    except Exception as e:
        print(f"Error writing to CSV file in {folder_path}: {e}")

    return combined_rows

# Main execution
combined_rows = []

entries = [
    entry for entry in os.listdir(base_path)
    if os.path.isdir(os.path.join(base_path, entry)) and entry.startswith("entry_")
]

if process_all_folders:
    # Process all subfolders in the base path
    for entry in entries:
        folder_path = os.path.join(base_path, entry)
        if os.path.isdir(folder_path):
            print(f"Processing folder: {folder_path}")
            folder_rows = process_folder(folder_path)
            combined_rows.extend(folder_rows)

    # Write the combined CSV file for all folders
    combined_csv_path = os.path.join(base_path, "dgs-text2gloss-split-speaker-combined.csv")
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
    example_folder = os.path.join(base_path, "entry_0")
    print(f"Processing example folder: {example_folder}")
    process_folder(example_folder)