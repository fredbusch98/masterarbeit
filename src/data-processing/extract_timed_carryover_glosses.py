import os
import csv
from pysrt import SubRipFile

# Base path to the DGSKorpus folder
base_path = "/Volumes/IISY/DGSKorpus/"
process_all_folders = True  # Set to True to process all subfolders
threshold = 500 # Threshold in milliseconds - time between last gloss of a speaker and the carryover gloss

def process_folder_timed_carryover(folder_path):
    transcript_path = os.path.join(folder_path, "filtered-transcript.srt")
    try:
        srt_file = SubRipFile.open(transcript_path)
    except Exception as e:
        print(f"❌ Error reading transcript file in {folder_path}: {e}")
        return []
    folder_name = os.path.basename(folder_path)
    collected_glosses = []
    previous_sentence_speaker = None
    collecting_until_P_sentence = False
    P = None
    last_gloss_end = {}  # speaker -> SubRipTime
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
                if collecting_until_P_sentence and speaker == P:
                    collecting_until_P_sentence = False
                if previous_sentence_speaker is not None and speaker != previous_sentence_speaker:
                    P = previous_sentence_speaker
                    collecting_until_P_sentence = True
                    last_gloss_before = last_gloss_end.get(P, None)
                previous_sentence_speaker = speaker
            else:
                # Gloss encountered
                last_gloss_end[speaker] = subtitle.end
                if collecting_until_P_sentence and speaker == P and last_gloss_before is not None:
                    gap = subtitle.start - last_gloss_before
                    if gap.ordinal < threshold:
                        collected_glosses.append([folder_name, subtitle.index])
    return collected_glosses

combined_glosses = []
if process_all_folders:
    folders = [entry for entry in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, entry))]
    total_folders = len(folders)
    processed_folders = 0
    for entry in folders:
        folder_path = os.path.join(base_path, entry)
        processed_folders += 1
        progress = (processed_folders / total_folders) * 100
        print(f"🚀 Processing folder: {folder_path} ({processed_folders}/{total_folders}, {progress:.1f}% complete)")
        combined_glosses.extend(process_folder_timed_carryover(folder_path))

    # Write to CSV
    csv_path = os.path.join(base_path, "carryover_timed_glosses.csv")
    try:
        with open(csv_path, mode='w', newline='', encoding='utf-8') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["Entry Directory", "Subtitle Index"])
            csv_writer.writerows(combined_glosses)
        print(f"🎉 CSV file created at {csv_path}")
    except Exception as e:
        print(f"❌ Error writing CSV file: {e}")
    print(f"🎊 Total glosses found with threshold {threshold}ms: {len(combined_glosses)}")