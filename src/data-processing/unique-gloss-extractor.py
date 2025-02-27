import os
import json
import csv
import math
from collections import Counter
from all_types import load_gloss_types

json_path = 'gloss2pose-filtered-by-all-types.json'

gloss_types = load_gloss_types("/Volumes/IISY/DGSKorpus/all-types-dgs.csv")

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

def save_missing_glosses_to_csv(missing_glosses, output_path):
    print(f"Saving missing gloss types to CSV file: {output_path}")
    with open(output_path, 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["missing_gloss"])
        for gloss in sorted(missing_glosses):  # Sort for readability
            writer.writerow([gloss])
    print(f"Missing gloss types successfully saved to: {output_path}")

if __name__ == "__main__":
    root_folder = "/Volumes/IISY/DGSKorpus"
    output_csv_unique = os.path.join(root_folder, "all-unique-glosses-from-transcripts-filtered.csv")
    output_csv_counts = os.path.join(root_folder, "all-gloss-counts-from-transcripts-filtered.csv")
    output_csv_missing = os.path.join(root_folder, "all-missing-gloss-types.csv")

    print(f"Starting process to collect unique glosses and counts from: {root_folder}")
    
    # Collect unique glosses and their counts
    unique_glosses, gloss_counts = collect_glosses_and_counts(root_folder)

    # Save the unique glosses to a CSV file
    save_glosses_to_csv(unique_glosses, output_csv_unique)

    # Save the gloss counts to a CSV file
    save_gloss_counts_to_csv(gloss_counts, output_csv_counts)

    # Find missing gloss types
    missing_glosses = gloss_types - unique_glosses  # Compute missing glosses

    # Save the missing gloss types to a CSV file
    save_missing_glosses_to_csv(missing_glosses, output_csv_missing)

    print(f"Process completed. Unique glosses have been saved to: {output_csv_unique}")
    print(f"Gloss counts have been saved to: {output_csv_counts}")
    print(f"Missing gloss types have been saved to: {output_csv_missing}")

    # Additional logs for deeper insights:

    # 0.0. Total number of glosses
    total_occurrences = sum(gloss_counts.values())
    print(f"\nTotal number of gloss occurrences (including duplicates): {total_occurrences}")

    # 0.1. Total number of unique glosses
    print(f"\nTotal number of unique glosses: {len(unique_glosses)}")

    # 1. Print the top 5 glosses ranked by occurrence count and their respective counts
    top_5 = gloss_counts.most_common(5)
    print("\nTop 5 glosses by occurrence count:")
    for gloss, count in top_5:
        print(f"{gloss}: {count}")

    # 2. Print the average occurrence count of all unique glosses
    average_occurrence = total_occurrences / len(unique_glosses) if unique_glosses else 0
    print(f"\nAverage occurrence count of all unique glosses: {average_occurrence:.2f} ≈ {math.ceil(average_occurrence)}")

    # 3. Count all glosses that have an occurrence count less than 1000 and print the number
    less_than_1000 = sum(1 for count in gloss_counts.values() if count < 1000)
    print(f"\nNumber of glosses with occurrence count less than 1000: {less_than_1000}")

    # 4. Count all glosses that have an occurrence count greater than 1000 and print the number
    greater_than_1000 = sum(1 for count in gloss_counts.values() if count > 1000)
    print(f"Number of glosses with occurrence count greater than 1000: {greater_than_1000}")

    # 5. Count all glosses that have an occurrence count between 100 and 1000 (inclusive) and print the number
    between_100_and_1000 = sum(1 for count in gloss_counts.values() if 100 <= count <= 1000)
    print(f"Number of glosses with occurrence count between 100 and 1000: {between_100_and_1000}")

    # 6. Count all glosses that have an occurrence count less than 100 and print the number
    less_than_100 = sum(1 for count in gloss_counts.values() if count < 100)
    print(f"Number of glosses with occurrence count less than 100: {less_than_100}")

    # 7. Count all glosses that have an occurrence count less or equal than the average and print the number
    less_or_equal_than_average = sum(1 for count in gloss_counts.values() if count <= math.ceil(average_occurrence))
    print(f"Number of glosses with occurrence less or equal than {math.ceil(average_occurrence)}: {less_or_equal_than_average}")

    # 8. Count all glosses that have an occurrence count less or equal than 10 and print the number
    less_or_equal_than_10 = sum(1 for count in gloss_counts.values() if count <= 10)
    print(f"Number of glosses with occurrence less or equal than 10: {less_or_equal_than_10}")

    # 9. Count all glosses that have an occurrence count less or equal than 2 and print the number
    less_or_equal_than_2 = sum(1 for count in gloss_counts.values() if count <= 2)
    print(f"Number of glosses with occurrence less or equal than 2: {less_or_equal_than_2}")

    # 10. Count all glosses that have an occurrence count equal to 1 and print the number
    equal_to_1 = sum(1 for count in gloss_counts.values() if count == 1)
    print(f"Number of glosses with occurrence equal to 1: {equal_to_1}")
