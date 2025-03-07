import os
import csv
from pysrt import SubRipFile

# Base path to the DGSKorpus folder
base_path = "/Volumes/IISY/DGSKorpus/"
threshold = 1000  # Threshold in milliseconds for timed carryover glosses

# Dictionaries to count occurrences of unique sentences for each carryover type
total_carryover_sentence_counts = {}
direct_carryover_sentence_counts = {}
timed_carryover_sentence_counts = {}

def process_folder(folder_path):
    """Process a single folder's subtitle file and collect data for all three tasks."""
    transcript_path = os.path.join(folder_path, "filtered-transcript.srt")
    try:
        srt_file = SubRipFile.open(transcript_path)
    except Exception as e:
        print(f"❌ Error reading transcript file in {folder_path}: {e}")
        return [], [], []
    
    folder_name = os.path.basename(folder_path)
    carryover_glosses = []         # For direct carryover glosses
    timed_carryover_glosses = []   # For timed carryover glosses
    mismatch_pairs = []            # For total carryover glosses
    
    # State variables
    previous_sentence_speaker = None
    last_sentence_text = None
    carryover_collecting = False
    timed_carryover_collecting = False
    P = None  # Previous speaker for carryover and timed carryover
    last_gloss_end = {}  # speaker -> SubRipTime for timing
    last_gloss_before = None
    
    for subtitle in srt_file:
        text = subtitle.text.strip()
        if text.startswith("A:") or text.startswith("B:"):
            parts = text.split(":", 1)
            if len(parts) < 2:
                continue
            speaker = parts[0].strip()
            content = parts[1].strip()
            
            if content.endswith("_FULL_SENTENCE"):
                # Sentence encountered
                if timed_carryover_collecting and speaker == P:
                    timed_carryover_collecting = False  # Stop if sentence from P
                if carryover_collecting and speaker != previous_sentence_speaker:
                    carryover_collecting = False  # Stop if new speaker
                if previous_sentence_speaker is not None and speaker != previous_sentence_speaker:
                    P = previous_sentence_speaker
                    carryover_collecting = True
                    timed_carryover_collecting = True
                    last_gloss_before = last_gloss_end.get(P, None)
                previous_sentence_speaker = speaker
                last_sentence_text = content.replace("_FULL_SENTENCE", "")
            else:
                # Gloss encountered
                last_gloss_end[speaker] = subtitle.end
                # Total carryover gloss (mismatch pair) logic
                if previous_sentence_speaker is not None and previous_sentence_speaker != speaker:
                    mismatch_pairs.append([folder_name, subtitle.index, last_sentence_text, content])
                    # Track unique sentences for total carryover
                    total_carryover_sentence_counts[last_sentence_text] = total_carryover_sentence_counts.get(last_sentence_text, 0) + 1
                # Direct carryover gloss logic
                if carryover_collecting and speaker == P:
                    carryover_glosses.append([folder_name, subtitle.index, content])
                    # Track unique sentences for direct carryover
                    direct_carryover_sentence_counts[last_sentence_text] = direct_carryover_sentence_counts.get(last_sentence_text, 0) + 1
                # Timed carryover gloss logic
                if timed_carryover_collecting and speaker == P and last_gloss_before is not None:
                    gap = (subtitle.start - last_gloss_before).ordinal
                    if gap < threshold:
                        timed_carryover_glosses.append([folder_name, subtitle.index, content])
                        # Track unique sentences for timed carryover
                        timed_carryover_sentence_counts[last_sentence_text] = timed_carryover_sentence_counts.get(last_sentence_text, 0) + 1
                # Stop carryover if gloss from current sentence speaker
                if carryover_collecting and speaker == previous_sentence_speaker:
                    carryover_collecting = False
    
    return carryover_glosses, timed_carryover_glosses, mismatch_pairs

# Main execution
combined_carryover_glosses = []
combined_timed_carryover_glosses = []
combined_mismatch_pairs = []

# List all subfolders
folders = [entry for entry in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, entry))]
total_folders = len(folders)
processed_folders = 0

for entry in folders:
    folder_path = os.path.join(base_path, entry)
    processed_folders += 1
    progress = (processed_folders / total_folders) * 100
    print(f"🚀 Processing folder: {folder_path} ({processed_folders}/{total_folders}, {progress:.1f}% complete)")
    carryover, timed_carryover, mismatch = process_folder(folder_path)
    combined_carryover_glosses.extend(carryover)
    combined_timed_carryover_glosses.extend(timed_carryover)
    combined_mismatch_pairs.extend(mismatch)

# Write results to CSV files
# 1. Direct carryover glosses
csv_path = os.path.join(base_path, "carryover-glosses/direct-carryover-glosses.csv")
try:
    with open(csv_path, mode='w', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["Entry Directory", "Subtitle Index", "Gloss"])
        csv_writer.writerows(combined_carryover_glosses)
    print(f"🎉 CSV file created at {csv_path}")
except Exception as e:
    print(f"❌ Error writing direct carryover glosses CSV: {e}")

# 2. Timed carryover glosses
csv_path = os.path.join(base_path, "carryover-glosses/carryover-timed-glosses.csv")
try:
    with open(csv_path, mode='w', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["Entry Directory", "Subtitle Index", "Gloss"])
        csv_writer.writerows(combined_timed_carryover_glosses)
    print(f"🎉 CSV file created at {csv_path}")
except Exception as e:
    print(f"❌ Error writing timed carryover glosses CSV: {e}")

# 3. Total carryover glosses (mismatch pairs)
csv_path = os.path.join(base_path, "carryover-glosses/total-carryover-glosses.csv")
try:
    with open(csv_path, mode='w', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["Entry Directory", "Subtitle Index", "Sentence", "Gloss"])
        csv_writer.writerows(combined_mismatch_pairs)
    print(f"🎉 CSV file created at {csv_path}")
except Exception as e:
    print(f"❌ Error writing total carryover glosses CSV: {e}")

# Print summary statistics
print()
print("Carryover glosses summary:")
print(f"📌 Total carryover glosses found: {len(combined_mismatch_pairs)}")
print("📌 Total carryover definition: A gloss from the previous speaker (e.g., A) appears somewhere after a sentence of a new speaker (e.g., B) but BEFORE another sentence of the previous speaker.")
print()
print(f"📌 Direct carryover glosses found: {len(combined_carryover_glosses)}")
print("📌 Direct carryover definition: A gloss from the previous speaker (e.g., A) appears directly after a sentence of a new speaker (e.g., B) BEFORE any glosses of the new speaker.")
print()
print(f"📌 Timed carryover glosses found with threshold {threshold}ms: {len(combined_timed_carryover_glosses)}")
print("📌 Timed carryover definition: A gloss from the previous speaker (e.g., A) appears after a sentence of a new speaker (e.g., B), where the time passed between the last gloss of the previous speaker and the carryover gloss is less than the defined threshold.")
print()
print(f"📌 Total unique sentences affected by total carryover glosses: {len(total_carryover_sentence_counts)}")
print(f"📌 Total unique sentences affected by direct carryover glosses: {len(direct_carryover_sentence_counts)}")
print(f"📌 Total unique sentences affected by timed carryover glosses: {len(timed_carryover_sentence_counts)}")