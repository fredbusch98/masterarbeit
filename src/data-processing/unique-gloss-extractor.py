import os
import json
import csv

def collect_unique_glosses(root_folder):
    unique_glosses = set()  # Using a set to ensure uniqueness

    # Iterate through all entries in the root folder
    for entry in os.listdir(root_folder):
        entry_path = os.path.join(root_folder, entry)

        # Check if the entry is a directory
        if os.path.isdir(entry_path):
            print(f"Processing directory: {entry_path}")
            gloss2pose_path = os.path.join(entry_path, 'gloss2pose.json')

            # Check if gloss2pose.json exists
            if os.path.isfile(gloss2pose_path):
                print(f"Found file: {gloss2pose_path}")
                entry_glosses = set()  # Temporary set to store glosses for the current entry
                with open(gloss2pose_path, 'r', encoding='utf-8') as file:
                    try:
                        data = json.load(file)
                        # Collect gloss values if the 'data' field exists
                        if 'data' in data:
                            for item in data['data']:
                                if 'gloss' in item:
                                    entry_glosses.add(item['gloss'])
                                    unique_glosses.add(item['gloss'])
                        print(f"Collected {len(entry_glosses)} unique glosses from: {gloss2pose_path}")
                    except json.JSONDecodeError:
                        print(f"Error decoding JSON in file: {gloss2pose_path}")
            else:
                print(f"File not found: {gloss2pose_path}")
        else:
            print(f"Skipping non-directory entry: {entry_path}")

    print(f"Finished processing directories. Total unique glosses collected: {len(unique_glosses)}")
    return unique_glosses

def save_glosses_to_csv(glosses, output_path):
    print(f"Saving glosses to CSV file: {output_path}")
    with open(output_path, 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["gloss"])
        for gloss in sorted(glosses):  # Sort the glosses for better readability
            writer.writerow([gloss])
    print(f"Glosses successfully saved to: {output_path}")

if __name__ == "__main__":
    root_folder = "/Volumes/IISY/DGSKorpus"
    output_csv = os.path.join(root_folder, "all-unique-glosses-from-transcripts.csv")

    print(f"Starting process to collect unique glosses from: {root_folder}")
    
    # Collect unique glosses
    unique_glosses = collect_unique_glosses(root_folder)

    # Save the unique glosses to a CSV file
    save_glosses_to_csv(unique_glosses, output_csv)

    print(f"Process completed. Unique glosses have been saved to: {output_csv}")
