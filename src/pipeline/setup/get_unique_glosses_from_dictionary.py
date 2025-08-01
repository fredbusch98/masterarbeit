import json
import csv
import os

def extract_unique_glosses(json_path, output_csv_path):
    """Extract unique gloss strings from a JSON file and write them to a CSV."""
    if not os.path.isfile(json_path):
        print(f"Error: JSON file not found at {json_path}")
        return
    
    print(f"Loading gloss2pose_dictionary.json from {json_path} and collecting unique glosses ...")
    with open(json_path, 'r', encoding='utf-8') as f:
        gloss_dict = json.load(f)

    unique_glosses = gloss_dict.keys()
    
    with open(output_csv_path, 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["gloss"])  # Optional header
        for gloss in unique_glosses:
            writer.writerow([gloss])

    print(f"Wrote {len(gloss_dict)} unique glosses to {output_csv_path}")

if __name__ == "__main__":
    # Adjust paths as needed
    resources_folder = "../resources/"
    json_path = os.path.join(resources_folder, "gloss2pose_dictionary.json")
    output_csv_path = os.path.join(resources_folder, "unique-glosses.csv")

    extract_unique_glosses(json_path, output_csv_path)
