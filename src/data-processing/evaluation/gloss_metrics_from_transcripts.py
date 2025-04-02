import os
import re
import csv
import math
from collections import Counter
import statistics

# Define the main folder path where entry_* folders are located
main_folder = '/Volumes/IISY/DGSKorpus'

def process_srt_file(srt_path):
    """
    Processes an SRT file to extract lost sentences, lost glosses, and all glosses.
    
    Args:
        srt_path (str): Path to the SRT file.
        
    Returns:
        tuple: (lost_glosses, all_glosses, lost_sentences)
    """
    with open(srt_path, 'r') as f:
        lines = f.readlines()

    entries = []
    current_entry = []
    for line in lines:
        if line.strip().isdigit():
            if current_entry:
                entries.append(current_entry)
            current_entry = [line.strip()]
        else:
            current_entry.append(line)
    if current_entry:
        entries.append(current_entry)

    lost_glosses = []
    all_glosses = []
    lost_sentences = []
    after_end_sentence = False

    for entry in entries:
        if len(entry) >= 3:
            text_line = entry[2].strip()
            match = re.match(r'[A-B]:\s', text_line)
            if match:
                clean_text = text_line[match.end():].strip()
                if clean_text.endswith('_FULL_SENTENCE_END_SENTENCE'):
                    gloss_text = clean_text[:-len('_FULL_SENTENCE_END_SENTENCE')]
                    lost_sentences.append((srt_path, entry[0], gloss_text))
                    after_end_sentence = False  # Reset flag since it includes _FULL_SENTENCE
                elif clean_text.endswith('_END_SENTENCE'):
                    clean_text_base = clean_text[:-len('_END_SENTENCE')]
                    after_end_sentence = True
                    all_glosses.append(clean_text_base)
                elif clean_text.endswith('_FULL_SENTENCE'):
                    after_end_sentence = False
                else:
                    clean_text_base = clean_text
                    all_glosses.append(clean_text_base)
                    if after_end_sentence:
                        lost_glosses.append((srt_path, entry[0], clean_text_base))
            else:
                print(f"⚠️ Warning: No speaker tag in {srt_path}, entry {entry[0]}")
        else:
            print(f"⚠️ Warning: Incomplete entry in {srt_path}, entry {entry[0]}")

    return lost_glosses, all_glosses, lost_sentences

# Get all entry_* folders
entry_folders = [
    f for f in os.listdir(main_folder)
    if f.startswith('entry_') and os.path.isdir(os.path.join(main_folder, f))
]

# Collect all lost glosses, all glosses, and lost sentences across all files
all_lost_glosses = []
all_glosses = []
all_lost_sentences = []
total_files = len(entry_folders) * 2  # Each entry has two speaker files (a & b)
processed_files = 0

print("🔍 Scanning for lost sentences and glosses...\n")

for entry_folder in entry_folders:
    for speaker in ['a', 'b']:
        srt_path = os.path.join(main_folder, entry_folder, f'speaker-{speaker}.srt')
        if os.path.exists(srt_path):
            lost, glosses, sentences = process_srt_file(srt_path)
            all_lost_glosses.extend(lost)
            all_glosses.extend(glosses)
            all_lost_sentences.extend(sentences)
        else:
            print(f"❌ File not found: {srt_path}")

        # Update progress
        processed_files += 1
        percentage = (processed_files / total_files) * 100
        print(f"📂 Progress: {processed_files}/{total_files} files processed ({percentage:.1f}%)")

# Write lost sentences to a text file
output_sentences_txt = os.path.join(main_folder, 'lost-sentences.txt')
with open(output_sentences_txt, 'w') as f:
    for path, idx, text in all_lost_sentences:
        f.write(f"In {path}, entry {idx}: {text}\n")

# Write lost glosses to a text file
output_txt = os.path.join(main_folder, 'lost-glosses.txt')
with open(output_txt, 'w') as f:
    for path, idx, text in all_lost_glosses:
        f.write(f"In {path}, entry {idx}: {text}\n")

print("\n✅ Processing complete!")
print(f"📁 Lost sentences saved to: {output_sentences_txt}")
print(f"📁 Lost glosses saved to: {output_txt}")

# Print the counts
print(f"Total lost sentences: {len(all_lost_sentences)}")

# --- Additional Statistics ---

# Total glosses count (lost + not lost)
total_gloss_count = len(all_glosses)
lost_gloss_count = len(all_lost_glosses)
not_lost_gloss_count = total_gloss_count - lost_gloss_count

# Count unique glosses (all)
unique_glosses = set(all_glosses)
unique_gloss_count = len(unique_glosses)

print("\n📊 Gloss Statistics:")
print(f"Total glosses (all): {total_gloss_count}")
print(f"Unique glosses (all): {unique_gloss_count}")
print(f"Not-lost glosses: {not_lost_gloss_count}")

# Count unique lost glosses and frequency of each
unique_lost = set([text for (_, _, text) in all_lost_glosses])
unique_lost_count = len(unique_lost)

lost_counter = Counter([text for (_, _, text) in all_lost_glosses])
all_counter = Counter(all_glosses)

print(f"Total lost glosses found: {len(all_lost_glosses)}")
print(f"Unique lost glosses: {unique_lost_count}")

# Write lost gloss frequency details to CSV file
csv_output = os.path.join(main_folder, 'lost-gloss-details.csv')
only_lost_count = 0
with open(csv_output, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(["Gloss", "Lost Frequency", "Total Frequency", "Only Lost"])
    
    for gloss in sorted(unique_lost):
        lost_freq = lost_counter[gloss]
        total_freq = all_counter[gloss]
        only_lost = lost_freq == total_freq
        if only_lost:
            only_lost_count += 1
        csv_writer.writerow([gloss, lost_freq, total_freq, only_lost])

print(f"Number of glosses that only appear as lost glosses: {only_lost_count}")
print(f"Lost gloss frequency details saved to: {csv_output}")

# Define output file paths consistent with the second script
output_csv_unique = os.path.join(main_folder, "all-unique-glosses-from-transcripts.csv")
output_csv_counts = os.path.join(main_folder, "all-gloss-counts-from-transcripts.csv")

# Save unique glosses to CSV
print(f"\nSaving unique glosses to CSV file: {output_csv_unique}")
with open(output_csv_unique, 'w', encoding='utf-8', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["gloss"])
    for gloss in sorted(unique_glosses):
        writer.writerow([gloss])
print(f"Unique glosses successfully saved to: {output_csv_unique}")

# Save gloss counts to CSV
print(f"Saving gloss counts to CSV file: {output_csv_counts}")
with open(output_csv_counts, 'w', encoding='utf-8', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["gloss", "count"])
    for gloss, count in all_counter.most_common():  # Sort by count in descending order
        writer.writerow([gloss, count])
print(f"Gloss counts successfully saved to: {output_csv_counts}")

# Additional statistical logs
print("\n📈 Detailed Gloss Statistics:")

# 0.0. Total number of gloss occurrences
total_occurrences = sum(all_counter.values())
print(f"Total number of gloss occurrences (including duplicates): {total_occurrences}")

# 0.1. Total number of unique glosses
print(f"Total number of unique glosses: {len(unique_glosses)}")

# 1. Top 5 glosses by occurrence count
top_5 = all_counter.most_common(5)
print("\nTop 5 glosses by occurrence count:")
for gloss, count in top_5:
    print(f"{gloss}: {count}")

# 2. Average occurrence count of all unique glosses
average_occurrence = total_occurrences / len(unique_glosses) if unique_glosses else 0
print(f"\nAverage occurrence count of all unique glosses: {average_occurrence:.2f} ≈ {math.ceil(average_occurrence)}")

# 2.5. Median occurrence count of all unique glosses
median_occurrence = statistics.median(list(all_counter.values()))
print(f"Median occurrence count of all unique glosses: {median_occurrence}")

# 3. Glosses with occurrence count less than 1000
less_than_1000 = sum(1 for count in all_counter.values() if count < 1000)
print(f"Number of glosses with occurrence count less than 1000: {less_than_1000}")

# 4. Glosses with occurrence count greater than 1000
greater_than_1000 = sum(1 for count in all_counter.values() if count > 1000)
print(f"Number of glosses with occurrence count greater than 1000: {greater_than_1000}")

# 5. Glosses with occurrence count between 100 and 1000
between_100_and_1000 = sum(1 for count in all_counter.values() if 100 <= count <= 1000)
print(f"Number of glosses with occurrence count between 100 and 1000: {between_100_and_1000}")

# 6. Glosses with occurrence count less than 100
less_than_100 = sum(1 for count in all_counter.values() if count < 100)
print(f"Number of glosses with occurrence count less than 100: {less_than_100}")

# 7. Glosses with occurrence count less or equal to the average
less_or_equal_than_average = sum(1 for count in all_counter.values() if count <= math.ceil(average_occurrence))
print(f"Number of glosses with occurrence less or equal than the average ({math.ceil(average_occurrence)}): {less_or_equal_than_average}")

# 7.5. Number of glosses with occurrence count less or equal than the median
less_or_equal_than_median = sum(1 for count in all_counter.values() if count <= median_occurrence)
print(f"Number of glosses with occurrence count less or equal than the median ({median_occurrence}): {less_or_equal_than_median}")

# 8. Glosses with occurrence count less or equal to 10
less_or_equal_than_10 = sum(1 for count in all_counter.values() if count <= 10)
print(f"Number of glosses with occurrence less or equal than 10: {less_or_equal_than_10}")

# 9. Glosses with occurrence count less or equal to 2
less_or_equal_than_2 = sum(1 for count in all_counter.values() if count <= 2)
print(f"Number of glosses with occurrence less or equal than 2: {less_or_equal_than_2}")

# 10. Glosses with occurrence count equal to 1
equal_to_1 = sum(1 for count in all_counter.values() if count == 1)
print(f"Number of glosses with occurrence equal to 1: {equal_to_1}")