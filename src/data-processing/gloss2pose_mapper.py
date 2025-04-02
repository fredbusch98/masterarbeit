import os
import cv2
import json
from pysrt import SubRipFile
import sys

process_all_folders = True  # Process all subfolders or a test folder
fixed_width = 1280
fixed_height = 720

def get_video_fps(video_path):
    """
    Get the frames per second (FPS) of a video file.
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

def map_speaker_srt_to_frame_range(srt_file_path, fps, speaker):
    """
    Map subtitles to frame ranges (start and end) using the FPS.
    Instead of building a huge list of frames, store only (start, end) tuples.
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
                # Only process entries that are not full sentences
                if not gloss.endswith("_FULL_SENTENCE"):
                    start_frame = int(sub.start.ordinal / 1000 * fps)
                    end_frame = int(sub.end.ordinal / 1000 * fps)
                    entries.append((gloss, (start_frame, end_frame)))
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
    """
    normalized = []
    scale_x = fixed_width / width
    scale_y = fixed_height / height

    for i in range(0, len(keypoints_2d), 3):  # Process each (x, y, confidence)
        x = keypoints_2d[i] * scale_x
        y = keypoints_2d[i + 1] * scale_y
        x /= fixed_width
        y /= fixed_height
        c = keypoints_2d[i + 2]  # Confidence remains unchanged
        normalized.extend([x, y, c])
    return normalized

def remove_3d_keypoints_and_normalize_2d_keypoints(pose_data, width, height):
    """
    Remove 3D keypoints and normalize 2D keypoints.
    """
    keys_to_remove = [
        "pose_keypoints_3d", "face_keypoints_3d",
        "hand_left_keypoints_3d", "hand_right_keypoints_3d"
    ]
    for key in keys_to_remove:
        if key in pose_data:
            del pose_data[key]

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
    Remove lower-body keypoints from pose_keypoints_2d.
    """
    # Only keep indexes corresponding to upper-body keypoints
    included_indexes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 17, 18]
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
    Map text entries to pose sequences using OpenPose JSON.
    Instead of iterating over a huge frame list, we iterate over the available frame keys.
    """
    try:
        print(f"[INFO] Loading OpenPose data: {openpose_file_path}")
        with open(openpose_file_path, 'r') as f:
            pose_data = json.load(f)

        result = []
        person_a_pose_count = 0
        person_b_pose_count = 0

        # Get camera/frame dimensions
        width_camera_a = pose_data[0].get("width", 1)
        height_camera_a = pose_data[0].get("height", 1)
        width_camera_b = pose_data[1].get("width", 1)
        height_camera_b = pose_data[1].get("height", 1)

        # Process Person A's entries
        frames_data_a = pose_data[0].get("frames", {})
        # Convert frame keys to integers once (assuming keys are numeric strings)
        available_frames_a = {int(frame): data["people"] for frame, data in frames_data_a.items()}

        for text, (start_frame, end_frame) in srt_entries_personA:
            people_data = []
            # Only iterate over the frames that actually have data
            for frame, people in available_frames_a.items():
                if start_frame <= frame <= end_frame:
                    for person in people:
                        cleaned_pose = remove_3d_keypoints_and_normalize_2d_keypoints(person, width_camera_a, height_camera_a)
                        cleaned_pose = filter_lower_body_keypoints(cleaned_pose)
                        people_data.append(cleaned_pose)
            result.append({"gloss": text, "pose_sequence": people_data})
            person_a_pose_count += len(people_data)

        # Process Person B's entries
        frames_data_b = pose_data[1].get("frames", {})
        available_frames_b = {int(frame): data["people"] for frame, data in frames_data_b.items()}

        for text, (start_frame, end_frame) in srt_entries_personB:
            people_data = []
            for frame, people in available_frames_b.items():
                if start_frame <= frame <= end_frame:
                    for person in people:
                        cleaned_pose = remove_3d_keypoints_and_normalize_2d_keypoints(person, width_camera_b, height_camera_b)
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
        return None

    # Use optimized mapping that stores frame ranges instead of full lists
    srt_entries_personA = map_speaker_srt_to_frame_range(srt_a_path, fps, "A")
    srt_entries_personB = map_speaker_srt_to_frame_range(srt_b_path, fps, "B")

    mapped_data = map_text_to_poses(openpose_path, srt_entries_personA, srt_entries_personB)
    output = {
        "data": mapped_data
    }

    output_path = os.path.join(folder_path, "gloss2pose.json")
    print(f"[INFO] Writing output to {output_path}")
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=4, ensure_ascii=False)
        print(f"[INFO] Process completed successfully for folder: {folder_path}")
    except Exception as e:
        print(f"[ERROR] Could not write output file for folder {folder_path}: {e}")

    return output

def main():
    starting_entry = None
    if len(sys.argv) > 1:
        starting_entry = sys.argv[1]

    if process_all_folders:
        root_path = "/Volumes/IISY/DGSKorpus/"  # Replace with your root directory
        entries = [
            entry for entry in os.listdir(root_path)
            if os.path.isdir(os.path.join(root_path, entry)) and entry.startswith("entry_")
        ]
        if starting_entry:
            try:
                start_index = entries.index(starting_entry)
                entries = entries[start_index:]
            except ValueError:
                print(f"[ERROR] The folder {starting_entry} was not found in {root_path}.")
                return

        for entry in entries:
            entry_path = os.path.join(root_path, entry)
            process_folder(entry_path)
    else:
        folder_path = "/Volumes/IISY/DGSKorpus/entry_3"
        process_folder(folder_path)

if __name__ == "__main__":
    main()
