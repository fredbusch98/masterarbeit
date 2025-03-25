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
    current_sentence = None
    gloss_entries = []
    mapping_active = False  # Indicates if we are currently collecting glosses for a sentence

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
        
        # Check if this subtitle marks the start of a new full sentence fragment
        if text.endswith("_FULL_SENTENCE"):
            sentence_fragment = text.replace("_FULL_SENTENCE", "").strip()
            if mapping_active:
                # Instead of ignoring, combine the new sentence fragment with the current sentence
                # This is done because there are occurrences in the transcripts where two full sentences appear directly after each other and are split into two separate subtitles without any glosses in between!
                current_sentence = f"{current_sentence} {sentence_fragment}"
            else:
                current_sentence = sentence_fragment
                mapping_active = True
                gloss_entries = []
        elif mapping_active:
            # If the current gloss marks the end of the sentence mapping, process and finalize
            if text.endswith("_END_SENTENCE"):
                cleaned_gloss = text.replace("_END_SENTENCE", "").strip()
                gloss_entries.append(cleaned_gloss)
                rows.append([current_sentence] + gloss_entries)
                # Reset the mapping until a new _FULL_SENTENCE is encountered
                current_sentence = None
                gloss_entries = []
                mapping_active = False
            else:
                # Otherwise, add this gloss to the current mapping
                gloss_entries.append(text)
        else:
            # If there's no active sentence mapping, glosses are ignored.
            continue

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
    output_csv_path = os.path.join(folder_path, "text2gloss.csv")

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

    # --- New code added here ---
    # Read the only-lost-glosses-output.csv file and append its rows
    lost_glosses_csv = os.path.join(base_path, "only-lost-glosses-output.csv")
    if os.path.exists(lost_glosses_csv):
        try:
            with open(lost_glosses_csv, 'r', encoding='utf-8') as csv_file:
                csv_reader = csv.reader(csv_file)
                # Optionally skip header row if present
                header = next(csv_reader, None)
                for row in csv_reader:
                    combined_rows.append(row)
            print(f"Appended rows from {lost_glosses_csv}")
        except Exception as e:
            print(f"Error reading {lost_glosses_csv}: {e}")
    else:
        print(f"File {lost_glosses_csv} not found.")
        
    # Write the combined CSV file for all folders
    combined_csv_path = os.path.join(base_path, "dgs-text2gloss-combined.csv")
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
