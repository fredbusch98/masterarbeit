import os
import cv2
import json
from pysrt import SubRipFile, open as open_srt

process_all_folders = False  # Set to True to process all subfolders, False to process just a single test folder

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

            if sub.text.startswith("A:") and len(sub.text[2:].strip().split()) == 1:
                srt_entries2frames_personA.append((sub.text[2:].strip(), frames))
            elif sub.text.startswith("B:") and len(sub.text[2:].strip().split()) == 1:
                srt_entries2frames_personB.append((sub.text[2:].strip(), frames))

        print(f"[INFO] Mapped {len(srt_entries2frames_personA)} transcript entries for Person A and {len(srt_entries2frames_personB)} entries for Person B.")
        return srt_entries2frames_personA, srt_entries2frames_personB

    except Exception as e:
        print(f"[ERROR] {e}")
        return [], []

def remove_3d_keypoints(pose_data):
    """
    Remove 3D keypoints from the pose data while keeping 2D keypoints.

    Parameters:
        pose_data (dict): Pose data from OpenPose.

    Returns:
        dict: Pose data with 3D keypoints removed.
    """
    # Remove any 3D keypoints
    keys_to_remove = [
        "pose_keypoints_3d", "face_keypoints_3d",
        "hand_left_keypoints_3d", "hand_right_keypoints_3d"
    ]
    
    for key in keys_to_remove:
        if key in pose_data:
            del pose_data[key]
    
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

        # Process Person A's entries
        for text, frames in srt_entries_personA:
            people_data = []
            frame_to_people = {int(frame): data["people"] for frame, data in pose_data[0]["frames"].items()}

            for frame in frames:
                if frame in frame_to_people:
                    for person in frame_to_people[frame]:
                        # Remove 3D keypoints and keep 2D ones
                        cleaned_pose = remove_3d_keypoints(person)
                        people_data.append(cleaned_pose)

            result.append({"text": text, "poses": people_data})
            person_a_pose_count += len(people_data)  # Add the number of poses for Person A

        # Process Person B's entries
        for text, frames in srt_entries_personB:
            people_data = []
            frame_to_people = {int(frame): data["people"] for frame, data in pose_data[1]["frames"].items()}

            for frame in frames:
                if frame in frame_to_people:
                    for person in frame_to_people[frame]:
                        # Remove 3D keypoints and keep 2D ones
                        cleaned_pose = remove_3d_keypoints(person)
                        people_data.append(cleaned_pose)

            result.append({"text": text, "poses": people_data})
            person_b_pose_count += len(people_data)  # Add the number of poses for Person B

        # Log the total number of poses mapped for each person
        print(f"[INFO] Total poses mapped for Person A: {person_a_pose_count}")
        print(f"[INFO] Total poses mapped for Person B: {person_b_pose_count}")
        
        print(f"[INFO] Mapped text entries to poses. Total mappings: {len(result)}")
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

    output_path = os.path.join(folder_path, "gloss2pose_mapped.json")
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
    all_data = {}  # Dictionary to store data from all folders

    if process_all_folders:
        root_path = "/Volumes/IISY/DGSKorpus/"  # Replace with the root directory containing all subfolders
        for entry in os.listdir(root_path):
            entry_path = os.path.join(root_path, entry)
            if os.path.isdir(entry_path):
                folder_data = process_folder(entry_path)
                if folder_data:  # Only add data if it's valid
                    all_data[entry] = folder_data

        # After processing all folders, write combined data to a final JSON file
        combined_output_path = os.path.join(root_path, "dgs-gloss2pose-combined.json")
        print(f"[INFO] Writing combined data to {combined_output_path}")
        try:
            with open(combined_output_path, 'w') as f:
                json.dump(all_data, f, indent=4)
            print(f"[INFO] Combined data written successfully.")
        except Exception as e:
            print(f"[ERROR] Could not write combined JSON file: {e}")

    else:
        folder_path = "/Volumes/IISY/DGSKorpus/entry_3"  # Replace with your test folder
        process_folder(folder_path)

if __name__ == "__main__":
    main()