import os
import json
import csv
from collections import Counter

json_path = 'gloss2pose-filtered-by-all-types.json'

def collect_glosses_and_counts(root_folder):
    unique_glosses = set()  # Using a set to ensure uniqueness
    gloss_counts = Counter()  # Counter to keep track of gloss occurrences

    # Iterate through all entries in the root folder
    for entry in os.listdir(root_folder):
        entry_path = os.path.join(root_folder, entry)

        # Check if the entry is a directory
        if os.path.isdir(entry_path):
            print(f"Processing directory: {entry_path}")
            gloss2pose_path = os.path.join(entry_path, json_path)

            # Check if gloss2pose.json exists
            if os.path.isfile(gloss2pose_path):
                with open(gloss2pose_path, 'r', encoding='utf-8') as file:
                    try:
                        data = json.load(file)
                        # Collect gloss values if the 'data' field exists
                        if 'data' in data:
                            for item in data['data']:
                                if 'gloss' in item:
                                    gloss = item['gloss']
                                    unique_glosses.add(gloss)
                                    gloss_counts[gloss] += 1
                    except json.JSONDecodeError:
                        print(f"Error decoding JSON in file: {gloss2pose_path}")
            else:
                print(f"File not found: {gloss2pose_path}")
        else:
            print(f"Skipping non-directory entry: {entry_path}")

    print(f"Finished processing directories. Total unique glosses collected: {len(unique_glosses)}")
    return unique_glosses, gloss_counts

def save_glosses_to_csv(glosses, output_path):
    print(f"Saving glosses to CSV file: {output_path}")
    with open(output_path, 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["gloss"])
        for gloss in sorted(glosses):  # Sort the glosses for better readability
            writer.writerow([gloss])
    print(f"Glosses successfully saved to: {output_path}")

def save_gloss_counts_to_csv(gloss_counts, output_path):
    print(f"Saving gloss counts to CSV file: {output_path}")
    with open(output_path, 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["gloss", "count"])
        for gloss, count in gloss_counts.most_common():  # Sort by count in descending order
            writer.writerow([gloss, count])
    print(f"Gloss counts successfully saved to: {output_path}")

if __name__ == "__main__":
    root_folder = "/Volumes/IISY/DGSKorpus"
    output_csv_unique = os.path.join(root_folder, "all-unique-glosses-from-transcripts-filtered.csv")
    output_csv_counts = os.path.join(root_folder, "all-gloss-counts-from-transcripts-filtered.csv")

    print(f"Starting process to collect unique glosses and counts from: {root_folder}")
    
    # Collect unique glosses and their counts
    unique_glosses, gloss_counts = collect_glosses_and_counts(root_folder)

    # Save the unique glosses to a CSV file
    save_glosses_to_csv(unique_glosses, output_csv_unique)

    # Save the gloss counts to a CSV file
    save_gloss_counts_to_csv(gloss_counts, output_csv_counts)

    print(f"Process completed. Unique glosses have been saved to: {output_csv_unique}")
    print(f"Gloss counts have been saved to: {output_csv_counts}")
