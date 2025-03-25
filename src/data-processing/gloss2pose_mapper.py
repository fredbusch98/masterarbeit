import os
import cv2
import json
from pysrt import SubRipFile
import sys

process_all_folders = True  # Set to True to process all subfolders, False to process just a single test folder
# An additional command line argument can be passed to start from a specific entry e.g. python gloss2pose_mapper.py entry_100 will process all subfolders starting from entry_100
fixed_width = 1280
fixed_height = 720

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

def map_speaker_srt_to_frames(srt_file_path, fps, speaker):
    """
    Map subtitles in the speaker's SRT file to frame numbers using the FPS.
    
    Args:
        srt_file_path (str): Path to the speaker's SRT file.
        fps (float): Frames per second of the video.
        speaker (str): Speaker label ("A" or "B").
    
    Returns:
        list: List of tuples containing (gloss, frames).
    """
    try:
        print(f"[INFO] Parsing SRT file for speaker {speaker}: {srt_file_path}")
        subs = SubRipFile.open(srt_file_path)
        entries = []
        speaker_tag = f"{speaker}: "
        
        for sub in subs:
            text = sub.text.strip()
            if not text:
                continue
            if text.startswith(speaker_tag):
                gloss = text[len(speaker_tag):].strip()
                if gloss.endswith("_END_SENTENCE"): 
                    gloss = gloss.replace("_END_SENTENCE", "")
                if not gloss.endswith("_FULL_SENTENCE"):
                    start_frame = int(sub.start.ordinal / 1000 * fps)
                    end_frame = int(sub.end.ordinal / 1000 * fps)
                    frames = list(range(start_frame, end_frame + 1))
                    entries.append((gloss, frames))
            else:
                print(f"Warning: Subtitle does not start with expected speaker tag '{speaker_tag}' in {srt_file_path}")
        
        print(f"[INFO] Mapped {len(entries)} transcript entries for speaker {speaker}.")
        return entries
    
    except Exception as e:
        print(f"[ERROR] {e}")
        return []

def normalize_keypoints_2d(keypoints_2d, width, height):
    """
    Normalize 2D keypoints to a range between 0 and 1 using width and height.

    Parameters:
        keypoints_2d (list): List of keypoints in the format [x1, y1, c1, x2, y2, c2, ...].
        width (int): Original width of the video/frame.
        height (int): Original height of the video/frame.

    Returns:
        list: Normalized keypoints.
    """
    normalized = []
    scale_x = fixed_width / width
    scale_y = fixed_height / height

    for i in range(0, len(keypoints_2d), 3):  # Step by 3 (x, y, confidence)
        # Scale before normalizing if width and height are different from the fixed dimensions: 1280x720
        x = keypoints_2d[i] * scale_x
        y = keypoints_2d[i + 1] * scale_y
        
        # Normalize
        x /= fixed_width
        y /= fixed_height

        c = keypoints_2d[i + 2]  # Confidence remains unchanged
        normalized.extend([x, y, c])

    return normalized

def remove_3d_keypoints_and_normalize_2d_keypoints(pose_data, width, height):
    """
    Remove 3D keypoints and normalize 2D keypoints.

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

    # Normalize and scale 2D keypoints
    keys_to_normalize = [
        "pose_keypoints_2d", "face_keypoints_2d",
        "hand_left_keypoints_2d", "hand_right_keypoints_2d"
    ]

    for key in keys_to_normalize:
        if key in pose_data:
            pose_data[key] = normalize_keypoints_2d(pose_data[key], width, height)

    return pose_data

def filter_lower_body_keypoints(pose_data):
    """
    Remove lower-body keypoints from pose_keypoints_2d entirely.

    Parameters:
        pose_data (dict): Pose data dictionary containing keypoints.

    Returns:
        dict: Modified pose data with only upper-body keypoints in pose_keypoints_2d.
    """
    included_indexes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 17, 18]  # Upper-body keypoints
    if "pose_keypoints_2d" in pose_data:
        pose_kp = pose_data["pose_keypoints_2d"]
        new_pose_kp = []
        for i in included_indexes:
            start = 3 * i  # Each keypoint has 3 values (x, y, confidence)
            new_pose_kp.extend(pose_kp[start:start + 3])
        pose_data["pose_keypoints_2d"] = new_pose_kp
    return pose_data

def map_text_to_poses(openpose_file_path, srt_entries_personA, srt_entries_personB):
    """
    Map text entries to pose sequences using OpenPose JSON, including only upper-body keypoints.

    Parameters:
        openpose_file_path (str): Path to the OpenPose JSON file.
        srt_entries_personA (list): List of tuples containing text and frame numbers for Person A.
        srt_entries_personB (list): List of tuples containing text and frame numbers for Person B.

    Returns:
        list: List of dictionaries mapping text to pose sequences with upper-body keypoints only.
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
            people_data = []
            frame_to_people = {int(frame): data["people"] for frame, data in pose_data[0]["frames"].items()}

            for frame in frames:
                if frame in frame_to_people:
                    for person in frame_to_people[frame]:
                        # Remove 3D keypoints and normalize 2D keypoints
                        cleaned_pose = remove_3d_keypoints_and_normalize_2d_keypoints(person, width_camera_a, height_camera_a)
                        # Filter out lower-body keypoints
                        cleaned_pose = filter_lower_body_keypoints(cleaned_pose)
                        people_data.append(cleaned_pose)

            result.append({"gloss": text, "pose_sequence": people_data})
            person_a_pose_count += len(people_data)

        # Process Person B's entries
        for text, frames in srt_entries_personB:
            people_data = []
            frame_to_people = {int(frame): data["people"] for frame, data in pose_data[1]["frames"].items()}

            for frame in frames:
                if frame in frame_to_people:
                    for person in frame_to_people[frame]:
                        # Remove 3D keypoints and normalize 2D keypoints
                        cleaned_pose = remove_3d_keypoints_and_normalize_2d_keypoints(person, width_camera_b, height_camera_b)
                        # Filter out lower-body keypoints
                        cleaned_pose = filter_lower_body_keypoints(cleaned_pose)
                        people_data.append(cleaned_pose)

            result.append({"gloss": text, "pose_sequence": people_data})
            person_b_pose_count += len(people_data)

        print(f"[INFO] Total single poses mapped for Person A: {person_a_pose_count}")
        print(f"[INFO] Total single poses mapped for Person B: {person_b_pose_count}")
        print(f"[INFO] Total pose-sequence mappings: {len(result)}")
        return result

    except Exception as e:
        print(f"[ERROR] {e}")
        return []

def process_folder(folder_path):
    video_a_path = os.path.join(folder_path, "video-a.mp4")
    video_b_path = os.path.join(folder_path, "video-b.mp4")
    srt_a_path = os.path.join(folder_path, "speaker-a.srt")
    srt_b_path = os.path.join(folder_path, "speaker-b.srt")
    openpose_path = os.path.join(folder_path, "openpose.json")

    print(f"[INFO] Processing folder: {folder_path}")

    fps = get_video_fps(video_a_path if os.path.exists(video_a_path) else video_b_path)
    if fps is None:
        print(f"[ERROR] Could not determine FPS for folder: {folder_path}. Skipping.")
        return None  # Return None if something went wrong

    # Map entries for each speaker using their respective SRT files
    srt_entries_personA = map_speaker_srt_to_frames(srt_a_path, fps, "A")
    srt_entries_personB = map_speaker_srt_to_frames(srt_b_path, fps, "B")

    # Map the text entries to poses for both Person A and Person B
    mapped_data = map_text_to_poses(openpose_path, srt_entries_personA, srt_entries_personB)

    # Individual output for each folder
    output = {
        "data": mapped_data  # Single list for both persons
    }

    output_path = os.path.join(folder_path, "gloss2pose.json")
    print(f"[INFO] Writing output to {output_path}")
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
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
        entries = [
            entry for entry in os.listdir(root_path)
            if os.path.isdir(os.path.join(root_path, entry)) and entry.startswith("entry_")
        ]
        
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