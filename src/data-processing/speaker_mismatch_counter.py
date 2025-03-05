import os
import csv
from pysrt import SubRipFile

# Define the base path to the DGSKorpus folder
base_path = "/Volumes/IISY/DGSKorpus/"
process_all_folders = True  # Set to True to process all subfolders

# Dictionary to count occurrences of unique sentences *only from stored pairs*
sentence_counts = {}

def process_folder(folder_path):
    transcript_path = os.path.join(folder_path, "filtered-transcript.srt")
    
    try:
        srt_file = SubRipFile.open(transcript_path)
    except Exception as e:
        print(f"❌ Error reading transcript file in {folder_path}: {e}")
        return []
    
    folder_name = os.path.basename(folder_path)
    pairs = []
    
    last_sentence_speaker = None
    last_sentence_text = None

    for subtitle in srt_file:
        text = subtitle.text.strip()
        if text.startswith("A:") or text.startswith("B:"):
            parts = text.split(":", 1)
            if len(parts) < 2:
                continue
            speaker = parts[0].strip()
            content = parts[1].strip()
            if content.endswith("_FULL_SENTENCE"):
                last_sentence_speaker = speaker
                last_sentence_text = content.replace("_FULL_SENTENCE", "")
            else:
                if last_sentence_speaker is not None and last_sentence_speaker != speaker:
                    pairs.append([folder_name, subtitle.index, last_sentence_text, content])
                    
                    # Count unique sentences that actually appear in the final output
                    if last_sentence_text in sentence_counts:
                        sentence_counts[last_sentence_text] += 1
                    else:
                        sentence_counts[last_sentence_text] = 1
    
    return pairs

combined_pairs = []

if process_all_folders:
    # List all subfolders in the base path
    folders = [entry for entry in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, entry))]
    total_folders = len(folders)
    processed_folders = 0

    for entry in folders:
        folder_path = os.path.join(base_path, entry)
        processed_folders += 1
        progress = (processed_folders / total_folders) * 100
        print(f"🚀 Processing folder: {folder_path} ({processed_folders}/{total_folders}, {progress:.1f}% complete)")
        combined_pairs.extend(process_folder(folder_path))
    
    # Write the combined CSV file in the base folder
    combined_csv_path = os.path.join(base_path, "combined_non_sentence_pairs.csv")
    try:
        with open(combined_csv_path, mode='w', newline='', encoding='utf-8') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["Entry Directory", "Subtitle Index", "Sentence", "Non-sentence"])
            csv_writer.writerows(combined_pairs)
        print(f"🎉 Combined CSV file created at {combined_csv_path}")
    except Exception as e:
        print(f"❌ Error writing combined CSV file: {e}")
    
    print(f"🎊 Total pairs found: {len(combined_pairs)}")
    print(f"📌 Total unique sentences (from mismatch pairs): {len(sentence_counts)}")
    print(f"📌 Total occurrences of mismatch sentences: {sum(sentence_counts.values())}")

else:
    # Process only a single example folder (if needed)
    example_folder = os.path.join(base_path, "entry_0")
    print(f"🚀 Processing example folder: {example_folder}")
    pairs = process_folder(example_folder)
    
    combined_csv_path = os.path.join(base_path, "combined_non_sentence_pairs.csv")
    try:
        with open(combined_csv_path, mode='w', newline='', encoding='utf-8') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["Entry Directory", "Subtitle Index", "Sentence", "Non-sentence"])
            csv_writer.writerows(pairs)
        print(f"🎉 Combined CSV file created at {combined_csv_path}")
    except Exception as e:
        print(f"❌ Error writing combined CSV file: {e}")
    
    print(f"🎊 Total pairs found in example folder: {len(pairs)}")
    print(f"📌 Total unique sentences (from mismatch pairs): {len(sentence_counts)}")
