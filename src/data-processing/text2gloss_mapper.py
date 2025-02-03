import os
import csv
from pysrt import SubRipFile

# Define the base path to the DGSKorpus folder
base_path = "/Volumes/IISY/DGSKorpus/"
# Define a variable to process all folders
process_all_folders = True # Set to True to process all subfolders, False to process just a single test folder

def load_gloss_types(csv_path):
    """
    Load gloss types from a CSV file.

    Parameters:
        csv_path (str): Path to the CSV file.

    Returns:
        set: A set of gloss types.
    """
    gloss_types = set()
    try:
        with open(csv_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                gloss_types.add(row[0].strip())  # Assuming the gloss types are in the first column
        print(f"[INFO] Loaded {len(gloss_types)} gloss types from {csv_path}")
    except Exception as e:
        print(f"[ERROR] Could not load gloss types from {csv_path}: {e}")
    return gloss_types

gloss_types = load_gloss_types("/Volumes/IISY/DGSKorpus/all-types-dgs.csv")

# Function to process a single folder
def process_folder(folder_path):
    transcript_path = os.path.join(folder_path, "transcript.srt")
    output_csv_path = os.path.join(folder_path, "text2gloss-filtered-by-all-types.csv")

    # Read the .srt file
    try:
        srt_file = SubRipFile.open(transcript_path)
    except Exception as e:
        print(f"Error reading transcript file in {folder_path}: {e}")
        return []

    # Initialize variables for processing
    full_sentence = None
    word_entries = []
    rows = []

    # Iterate through the subtitles in the transcript
    for subtitle in srt_file:
        text = subtitle.text.strip()

        if text.startswith("A:") or text.startswith("B:"):
            label, content = text.split(":", 1)
            content = content.strip()

            # Check if it's a full sentence (more than one word and starts with a capital letter)
            if len(content.split()) > 1 and content[0].isupper():
                # Save the previous sentence and its word entries to rows if they exist
                if full_sentence and word_entries:
                    rows.append([full_sentence] + word_entries)
                
                # Reset for the new sentence
                full_sentence = content
                word_entries = []
            else:  # This is a single word or an invalid sentence
                if full_sentence:  # Ensure there is a sentence to map it to
                    word = content.lstrip("|").rstrip("*")
                    if any(gloss_type.rstrip('^') == word.rstrip('^') for gloss_type in gloss_types):
                        word_entries.append(word)

    # Handle the last sentence and its word entries
    if full_sentence and word_entries:
        rows.append([full_sentence] + word_entries)

    # Write the rows to a CSV file
    try:
        with open(output_csv_path, mode='w', newline='', encoding='utf-8') as csv_file:
            csv_writer = csv.writer(csv_file)
            # Write rows to the CSV
            csv_writer.writerow(["Full Sentence", "Words..."])  # Add a header row
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
            # Write rows to the CSV
            csv_writer.writerow(["Full Sentence", "Words..."])  # Add a header row
            csv_writer.writerows(combined_rows)
        print(f"Combined CSV file created at {combined_csv_path}")
    except Exception as e:
        print(f"Error writing combined CSV file: {e}")
else:
    # Process only the example folder
    example_folder = os.path.join(base_path, "entry_3")
    print(f"Processing example folder: {example_folder}")
    process_folder(example_folder)
