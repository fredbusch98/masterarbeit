import os
import json
import csv
from tqdm import tqdm

# Expected lengths for keypoint arrays
EXPECTED_LENGTHS = {
    'pose_keypoints_2d': 39, # 39 instead of 75 because we removed the lower-body keypoints in gloss2pose_mapper.py
    'face_keypoints_2d': 210,
    'hand_left_keypoints_2d': 63,
    'hand_right_keypoints_2d': 63
}

def read_unique_glosses(csv_path):
    """Read unique glosses from a CSV file."""
    with open(csv_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        # Use tqdm to show progress if the CSV file is large
        unique_glosses = set(row[0] for row in tqdm(reader, desc="Reading CSV rows"))
    return unique_glosses

def calculate_avg_conf(poses):
    """Calculate the average confidence score for a pose sequence."""
    if not poses:
        return 0
    total_sum = 0
    for pose_object in poses:
        sum_conf = 0
        for key in ['pose_keypoints_2d', 'face_keypoints_2d', 'hand_left_keypoints_2d', 'hand_right_keypoints_2d']:
            if key in pose_object:
                sum_conf += sum(pose_object[key][2::3])  # Sum confidence scores (every 3rd element)
        total_sum += sum_conf
    return total_sum / len(poses)

def is_corrupted(pose_sequence):
    """Check if a pose sequence is corrupted based on keypoint array lengths."""
    for pose_object in pose_sequence:
        for key, expected_length in EXPECTED_LENGTHS.items():
            if key in pose_object and len(pose_object[key]) != expected_length:
                return True
    return False

def update_output_file(best_poses, output_json_path):
    """Save the current best poses to a JSON file."""
    # Convert best_poses to a final dictionary mapping gloss to its best pose sequence.
    final_dict = {gloss: pose_seq for gloss, (avg_conf, pose_seq) in best_poses.items()}
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(final_dict, f)

def collect_best_poses_in_batches(root_folder, unique_glosses, output_json_path, batch_size=10):
    """
    Process the folders in root_folder to collect the best pose sequence for each gloss.
    Writes to disk every batch_size folders.
    """
    best_poses = {}  # Mapping: gloss -> (avg_conf, pose_sequence)
    entries = os.listdir(root_folder)
    processed_folders = 0

    for entry in tqdm(entries, desc="Processing folders"):
        entry_path = os.path.join(root_folder, entry)
        if os.path.isdir(entry_path):
            gloss2pose_path = os.path.join(entry_path, 'gloss2pose.json')
            if os.path.isfile(gloss2pose_path):
                with open(gloss2pose_path, 'r', encoding='utf-8') as file:
                    try:
                        data = json.load(file)
                        if 'data' in data:
                            for item in tqdm(data['data'], desc=f"Processing {entry}", leave=False):
                                if 'gloss' in item and 'pose_sequence' in item:
                                    gloss = item['gloss']
                                    if gloss in unique_glosses:
                                        poses = item['pose_sequence']
                                        if is_corrupted(poses):
                                            continue
                                        avg_conf = calculate_avg_conf(poses)
                                        # Update best pose if higher confidence
                                        if gloss not in best_poses or avg_conf > best_poses[gloss][0]:
                                            best_poses[gloss] = (avg_conf, poses)
                    except json.JSONDecodeError:
                        print(f"Error decoding JSON in file: {gloss2pose_path}")
            processed_folders += 1

            # Write batch to disk
            if processed_folders % batch_size == 0:
                update_output_file(best_poses, output_json_path)
                print(f"Updated output file after processing {processed_folders} folders.")
    
    # Final update if the last batch is smaller than batch_size.
    update_output_file(best_poses, output_json_path)
    return best_poses

def validate_final_dictionary(output_json_path, unique_glosses):
    """
    Validates that the final JSON file contains the same number of glosses as in the unique_glosses.
    Prints missing glosses if validation fails.
    """
    with open(output_json_path, 'r', encoding='utf-8') as f:
        final_dict = json.load(f)
    num_final = len(final_dict)
    num_unique = len(unique_glosses)
    if num_final != num_unique:
        missing_glosses = set(unique_glosses) - set(final_dict.keys())
        print(f"Validation FAILED: Final dictionary contains {num_final} glosses but expected {num_unique}.")
        print("Missing glosses:", missing_glosses)
    else:
        print(f"Validation PASSED: Final dictionary contains the expected {num_final} glosses.")

if __name__ == "__main__":
    # Define paths
    root_folder = "/Volumes/IISY/DGSKorpus"
    unique_glosses_csv = os.path.join(root_folder, "all-unique-glosses-from-transcripts.csv")
    
    # Step 1: Read unique glosses
    print(f"Reading unique glosses from {unique_glosses_csv}")
    unique_glosses = read_unique_glosses(unique_glosses_csv)
    print(f"Loaded {len(unique_glosses)} unique glosses")
    
    # Step 2: Process folders and save results in batches
    output_json_path = os.path.join(root_folder, "gloss2pose_dictionary.json")
    print("Collecting best pose sequences and saving in batches...")
    best_poses = collect_best_poses_in_batches(root_folder, unique_glosses, output_json_path, batch_size=10)
    print(f"Collected best pose sequences for {len(best_poses)} glosses")
    
    # Step 3: Validate the final JSON file against the unique glosses
    validate_final_dictionary(output_json_path, unique_glosses)

    # Step 4: Also save a copy in ../pipeline/resources/ for direct use in the Gloss2Pose module
    script_dir = os.path.dirname(os.path.abspath(__file__))
    target_path = os.path.abspath(os.path.join(script_dir, "../pipeline/resources/gloss2pose_dictionary.json"))
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    with open(target_path, 'w', encoding='utf-8') as f:
        json.dump({gloss: pose_seq for gloss, (avg_conf, pose_seq) in best_poses.items()}, f)
    print(f"Copied final dictionary to {target_path}")

    
    print(f"Final dictionary saved to {output_json_path}")