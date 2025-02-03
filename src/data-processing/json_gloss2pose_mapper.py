import os
import cv2
import json
from pysrt import SubRipFile, open as open_srt
import sys
import csv

process_all_folders = True  # Set to True to process all subfolders, False to process just a single test folder
# An additional command line argument can be passed to start from a specific entry e.g. python gloss2pose_mapper.py entry_100 will process all subfolders starting from entry_100

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

def get_video_fps(video_path):
    """
    Get the frames per second (FPS) of a video file.

    Parameters:
        video_path (str): Path to the video file.

    Returns:
        float: The FPS of the video.
    """
    try:
        print(f"[INFO] Attempting to read FPS for video: {video_path}")
        video = cv2.VideoCapture(video_path)

        if not video.isOpened():
            raise FileNotFoundError(f"Unable to open video file: {video_path}")

        fps = video.get(cv2.CAP_PROP_FPS)
        video.release()
        print(f"[INFO] FPS for video {video_path}: {fps}")
        return fps

    except Exception as e:
        print(f"[ERROR] {e}")
        return None

def map_srt_to_frames(srt_file_path, fps):
    """
    Map subtitles in the SRT file to frame numbers using the FPS.

    Parameters:
        srt_file_path (str): Path to the SRT file.
        fps (float): Frames per second of the video.

    Returns:
        tuple: Two lists containing tuples for Person A and Person B.
    """
    try:
        print(f"[INFO] Parsing SRT file: {srt_file_path}")
        subs = open_srt(srt_file_path)
        srt_entries2frames_personA = []
        srt_entries2frames_personB = []

        for sub in subs:
            start_frame = int(sub.start.ordinal / 1000 * fps)
            end_frame = int(sub.end.ordinal / 1000 * fps)
            frames = list(range(start_frame, end_frame + 1))

            if sub.text[:2] in {"A:", "B:"} and len(sub.text[2:].strip().split()) == 1:
                person = sub.text[0]
                entry = (sub.text[2:].strip().lstrip('|').lstrip('*').rstrip('|').rstrip('*'), frames)
                if person == "A":
                    srt_entries2frames_personA.append(entry)
                else:
                    srt_entries2frames_personB.append(entry)

        print(f"[INFO] Mapped {len(srt_entries2frames_personA)} transcript entries for Person A and {len(srt_entries2frames_personB)} entries for Person B.")
        return srt_entries2frames_personA, srt_entries2frames_personB

    except Exception as e:
        print(f"[ERROR] {e}")
        return [], []

def normalize_keypoints_2d(keypoints_2d, width, height):
    """
    Normalize 2D keypoints to a range between 0 and 1 using width and height.

    Parameters:
        keypoints_2d (list): List of keypoints in the format [x1, y1, c1, x2, y2, c2, ...].
        width (int): Width of the video/frame.
        height (int): Height of the video/frame.

    Returns:
        list: Normalized keypoints.
    """
    normalized = []
    for i in range(0, len(keypoints_2d), 3):  # Step by 3 (x, y, confidence)
        x = keypoints_2d[i] / width if width > 0 else 0
        y = keypoints_2d[i + 1] / height if height > 0 else 0
        c = keypoints_2d[i + 2]  # Confidence remains the same
        normalized.extend([x, y, c])
    return normalized

def filter_upper_body_keypoints(keypoints_2d):
    """
    Filter only the upper body keypoints from a 2D keypoints list by excluding specific triplets.

    Parameters:
        keypoints_2d (list): List of keypoints in the format [x1, y1, c1, x2, y2, c2, ...].

    Returns:
        list: Filtered keypoints with only upper body keypoints.
    """
    excluded_triplet_indices = {9, 10, 11, 12, 13, 14, 19, 20, 21, 22, 23, 24}
    filtered = []
    for i in range(0, len(keypoints_2d), 3):
        triplet_index = i // 3
        if triplet_index not in excluded_triplet_indices:
            filtered.extend(keypoints_2d[i:i+3])  # Keep this triplet
    return filtered

def remove_3d_keypoints_and_normalize_2d_keypoints(pose_data, width, height):
    """
    Remove 3D keypoints, filter upper body keypoints, and normalize 2D keypoints.

    Parameters:
        pose_data (dict): Pose data from OpenPose.
        width (int): Width of the video/frame.
        height (int): Height of the video/frame.

    Returns:
        dict: Pose data with 3D keypoints removed, 2D keypoints filtered and normalized.
    """
    keys_to_remove = [
        "pose_keypoints_3d", "face_keypoints_3d",
        "hand_left_keypoints_3d", "hand_right_keypoints_3d"
    ]

    # Remove 3D keypoints
    for key in keys_to_remove:
        if key in pose_data:
            del pose_data[key]

    # Normalize and filter 2D keypoints
    keys_to_normalize = [
        "pose_keypoints_2d", "face_keypoints_2d",
        "hand_left_keypoints_2d", "hand_right_keypoints_2d"
    ]

    for key in keys_to_normalize:
        if key in pose_data:
            if key == "pose_keypoints_2d":
                pose_data[key] = filter_upper_body_keypoints(pose_data[key])

            pose_data[key] = normalize_keypoints_2d(pose_data[key], width, height)

    return pose_data

def map_text_to_poses(openpose_file_path, srt_entries_personA, srt_entries_personB):
    """
    Map text entries to pose sequences using OpenPose JSON.

    Parameters:
        openpose_file_path (str): Path to the OpenPose JSON file.
        srt_entries_personA (list): List of tuples containing text and frame numbers for Person A.
        srt_entries_personB (list): List of tuples containing text and frame numbers for Person B.

    Returns:
        list: List of dictionaries mapping text to pose sequences.
    """
    try:
        print(f"[INFO] Loading OpenPose data: {openpose_file_path}")
        with open(openpose_file_path, 'r') as f:
            pose_data = json.load(f)

        result = []
        person_a_pose_count = 0  # Counter for Person A's poses
        person_b_pose_count = 0  # Counter for Person B's poses

        width_camera_a = pose_data[0].get("width", 1)  # Default to 1 to avoid division by zero
        height_camera_a = pose_data[0].get("height", 1)
        width_camera_b = pose_data[1].get("width", 1)
        height_camera_b = pose_data[1].get("height", 1)

        # Process Person A's entries
        for text, frames in srt_entries_personA:
            # Check if the text matches any gloss type (with or without ^)
            if any(gloss_type.rstrip('^') == text.rstrip('^') for gloss_type in gloss_types):
                people_data = []
                frame_to_people = {int(frame): data["people"] for frame, data in pose_data[0]["frames"].items()}

                for frame in frames:
                    if frame in frame_to_people:
                        for person in frame_to_people[frame]:
                            # Remove 3D keypoints and normalize 2D keypoints
                            cleaned_pose = remove_3d_keypoints_and_normalize_2d_keypoints(person, width_camera_a, height_camera_a)
                            people_data.append(cleaned_pose)

                result.append({"gloss": text, "pose_sequence": people_data})
                person_a_pose_count += len(people_data)  # Add the number of poses for Person A

        # Process Person B's entries
        for text, frames in srt_entries_personB:
            # Check if the text matches any gloss type (with or without ^)
            if any(gloss_type.rstrip('^') == text.rstrip('^') for gloss_type in gloss_types):
                people_data = []
                frame_to_people = {int(frame): data["people"] for frame, data in pose_data[1]["frames"].items()}

                for frame in frames:
                    if frame in frame_to_people:
                        for person in frame_to_people[frame]:
                            # Remove 3D keypoints and normalize 2D keypoints
                            cleaned_pose = remove_3d_keypoints_and_normalize_2d_keypoints(person, width_camera_b, height_camera_b)
                            people_data.append(cleaned_pose)

                result.append({"gloss": text, "pose_sequence": people_data})
                person_b_pose_count += len(people_data)  # Add the number of poses for Person B

        # Log the total number of poses mapped for each person
        print(f"[INFO] Total poses mapped for Person A: {person_a_pose_count}")
        print(f"[INFO] Total poses mapped for Person B: {person_b_pose_count}")
        
        print(f"[INFO] Total mappings after filtering by all-types: {len(result)}")
        return result

    except Exception as e:
        print(f"[ERROR] {e}")
        return []

def process_folder(folder_path):
    video_a_path = os.path.join(folder_path, "video-a.mp4")
    video_b_path = os.path.join(folder_path, "video-b.mp4")
    srt_path = os.path.join(folder_path, "transcript.srt")
    openpose_path = os.path.join(folder_path, "openpose.json")

    print(f"[INFO] Processing folder: {folder_path}")

    fps = get_video_fps(video_a_path if os.path.exists(video_a_path) else video_b_path)
    if fps is None:
        print(f"[ERROR] Could not determine FPS for folder: {folder_path}. Skipping.")
        return None  # Return None if something went wrong

    # Instead of mapping separately, we'll combine entries for both Person A and Person B
    srt_entries_personA, srt_entries_personB = map_srt_to_frames(srt_path, fps)

    # Map the combined text entries to poses for both Person A and Person B
    mapped_data = map_text_to_poses(openpose_path, srt_entries_personA, srt_entries_personB)

    # Individual output for each folder
    output = {
        "data": mapped_data  # Single list for both persons
    }

    output_path = os.path.join(folder_path, "gloss2pose-filtered-by-all-types.json")
    print(f"[INFO] Writing output to {output_path}")
    try:
        # Open the file with UTF-8 encoding to ensure proper character encoding
        with open(output_path, 'w', encoding='utf-8') as f:
            # Ensure characters like ü, ä, ö are written correctly
            json.dump(output, f, indent=4, ensure_ascii=False)
        print(f"[INFO] Process completed successfully for folder: {folder_path}")
    except Exception as e:
        print(f"[ERROR] Could not write output file for folder {folder_path}: {e}")

    return output  # Return the data to accumulate it for the combined JSON

def main():
    starting_entry = None  # Default value for starting entry
    
    # Check if a folder name is passed as an argument
    if len(sys.argv) > 1:
        starting_entry = sys.argv[1]

    if process_all_folders:
        root_path = "/Volumes/IISY/DGSKorpus/"  # Replace with the root directory containing all subfolders
        
        # Get all entries in the root path
        entries = [entry for entry in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, entry))]

        # If starting_entry is specified, start from that folder
        if starting_entry:
            try:
                start_index = entries.index(starting_entry)
                entries = entries[start_index:]  # Only process from the starting entry onward
            except ValueError:
                print(f"[ERROR] The folder {starting_entry} was not found in {root_path}.")
                return
        
        # Process each folder individually
        for entry in entries:
            entry_path = os.path.join(root_path, entry)
            process_folder(entry_path)

    else:
        folder_path = "/Volumes/IISY/DGSKorpus/entry_3"
        process_folder(folder_path)

if __name__ == "__main__":
    main()