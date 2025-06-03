import os
import cv2
import json
from pysrt import SubRipFile
import sys

# If you want to process all subfolders under a root, set this to True.
process_all_folders = True  

# We will normalize all keypoints into a 1280×720 canvas, then scale to [0,1].
fixed_width = 1280
fixed_height = 720


def get_video_fps(video_path):
    """
    Open the video file at video_path and return its FPS.
    If the file cannot be opened, returns None.
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


def parse_srt_for_sentence_entries(srt_file_path, fps, speaker):
    """
    Parse the SRT so that:
      - Any subtitle line beginning with "{speaker}: " whose gloss ends with "_FULL_SENTENCE"
        becomes the anchor for a new “sentence entry.”
      - We extract that entry's timestamp (start and end in milliseconds → frames via FPS).
      - Then we look at each subsequent subtitle entry for the same speaker:
          • If a gloss ends with "_END_SENTENCE", strip that suffix, add it to the gloss list, and stop.
          • Otherwise, assume it is one of the component glosses—strip no suffix and add to the list.
      - If we never find an "_END_SENTENCE" after a "_FULL_SENTENCE", we skip that sentence.
      - We return a list of dicts, each containing:
          { "full_sentence": <string without suffix>, 
            "gloss_sequence": <comma-joined string of glosses (no suffixes)>,
            "start_frame": <int>, "end_frame": <int> }.
    """
    try:
        print(f"[INFO] Parsing SRT file for speaker {speaker}: {srt_file_path}")
        subs = SubRipFile.open(srt_file_path)
        sentence_entries = []
        speaker_tag = f"{speaker}: "

        # Convert SubRipFile to a simple list to allow peeking ahead
        all_subs = [sub for sub in subs if sub.text.strip()]
        i = 0
        while i < len(all_subs):
            entry = all_subs[i]
            text = entry.text.strip()

            # Only process if it starts with "speaker: "
            if text.startswith(speaker_tag):
                gloss_raw = text[len(speaker_tag):].strip()
                # Look for a "_FULL_SENTENCE" suffix
                if gloss_raw.endswith("_FULL_SENTENCE"):
                    # Extract the “full sentence” text itself
                    full_sentence = gloss_raw.replace("_FULL_SENTENCE", "").strip()

                    # Compute start_frame and end_frame from this entry’s timestamps
                    start_frame = int(entry.start.ordinal / 1000 * fps)
                    end_frame = int(entry.end.ordinal / 1000 * fps)

                    # Now collect subsequent glosses until we hit "_END_SENTENCE"
                    gloss_list = []
                    j = i + 1
                    found_end = False
                    while j < len(all_subs):
                        next_entry = all_subs[j]
                        next_text = next_entry.text.strip()

                        if not next_text.startswith(speaker_tag):
                            # If a different speaker appears, stop scanning
                            break

                        next_gloss_raw = next_text[len(speaker_tag):].strip()
                        if next_gloss_raw.endswith("_END_SENTENCE"):
                            cleaned = next_gloss_raw.replace("_END_SENTENCE", "").strip()
                            gloss_list.append(cleaned)
                            found_end = True
                            break
                        else:
                            # A “middle” gloss of the same sentence
                            gloss_list.append(next_gloss_raw)
                        j += 1

                    if found_end and len(gloss_list) > 0:
                        # Build a comma-separated string of the component glosses
                        gloss_sequence = ",".join(gloss_list)
                        sentence_entries.append({
                            "full_sentence": full_sentence,
                            "gloss_sequence": gloss_sequence,
                            "start_frame": start_frame,
                            "end_frame": end_frame
                        })
                        # Jump to the gloss that ended this sentence
                        i = j  
                    else:
                        # If we never found an _END_SENTENCE or no following glosses, skip
                        print(f"[WARNING] No END_SENTENCE found for FULL_SENTENCE '{full_sentence}' in {srt_file_path}. Skipping.")
                # else: gloss is not a FULL_SENTENCE → ignore for this step
            # Move to next subtitle
            i += 1

        print(f"[INFO] Found {len(sentence_entries)} full-sentence entries for speaker {speaker}.")
        return sentence_entries

    except Exception as e:
        print(f"[ERROR] {e}")
        return []


def normalize_keypoints_2d(keypoints_2d, width, height):
    """
    Normalize 2D keypoints into [0..1] by:
      1) Scaling each (x,y) from the camera resolution to a fixed canvas (1280×720),
      2) Dividing by (fixed_width, fixed_height).
    Each keypoint triple is (x, y, confidence). We leave confidence unchanged.
    """
    normalized = []
    scale_x = fixed_width / width
    scale_y = fixed_height / height

    for i in range(0, len(keypoints_2d), 3):
        x = keypoints_2d[i] * scale_x
        y = keypoints_2d[i + 1] * scale_y
        c = keypoints_2d[i + 2]  # confidence unchanged

        x /= fixed_width
        y /= fixed_height

        normalized.extend([x, y, c])

    return normalized


def remove_3d_keypoints_and_normalize_2d_keypoints(pose_data, width, height):
    """
    Given a single “person” dictionary from one frame, remove any 3D data
    and normalize all 2D keypoints. Works in place on pose_data.
    """
    # Keys that contain 3D points
    keys_to_remove = [
        "pose_keypoints_3d", "face_keypoints_3d",
        "hand_left_keypoints_3d", "hand_right_keypoints_3d"
    ]
    for key in keys_to_remove:
        if key in pose_data:
            del pose_data[key]

    # Now normalize each 2D keypoint array
    keys_to_norm = [
        "pose_keypoints_2d", "face_keypoints_2d",
        "hand_left_keypoints_2d", "hand_right_keypoints_2d"
    ]
    for key in keys_to_norm:
        if key in pose_data:
            pose_data[key] = normalize_keypoints_2d(pose_data[key], width, height)
    return pose_data


def filter_lower_body_keypoints(pose_data):
    """
    For each frame’s “pose_keypoints_2d”, only keep indices corresponding
    to upper-body joints. The “included_indexes” list is taken from OpenPose’s 18-keypoint format.
    """
    # In an 18-keypoint layout, indexes 0–17. We only keep:
    # head/neck (0–8) and upper torso (15–18). Indices based on OpenPose’s standard output.
    included_indexes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 17, 18]
    if "pose_keypoints_2d" in pose_data:
        pose_kp = pose_data["pose_keypoints_2d"]
        new_pose_kp = []
        for idx in included_indexes:
            start = 3 * idx
            new_pose_kp.extend(pose_kp[start:start + 3])
        pose_data["pose_keypoints_2d"] = new_pose_kp
    return pose_data


def map_sentences_to_poses(openpose_file_path, sentences_A, sentences_B):
    """
    Given the path to openpose.json and two lists of “sentence entries” for A and B,
    produce a single list of dicts, each with:
      {
        "full_sentence": <string>,
        "gloss_sequence": <string>,
        "pose_sequence": [ <cleaned pose frames> … ]
      }
    We do this by loading openpose.json into memory, then for each sentence
    entry in A (resp. B), iterate only over the frames that fall between start_frame..end_frame
    and collect each person’s cleaned keypoints.
    """
    try:
        print(f"[INFO] Loading OpenPose data: {openpose_file_path}")
        with open(openpose_file_path, "r", encoding="utf-8") as f:
            pose_data = json.load(f)

        merged_results = []
        total_A = 0
        total_B = 0

        # The JSON format is assumed to have two top-level entries, one per camera/person:
        # pose_data[0] is Person A’s camera (with width/height, then a "frames" dict).
        # pose_data[1] is Person B’s camera.
        width_A = pose_data[0].get("width", 1)
        height_A = pose_data[0].get("height", 1)
        width_B = pose_data[1].get("width", 1)
        height_B = pose_data[1].get("height", 1)

        # Build a mapping: frame_number (int) → list_of_people_dicts
        frames_A = pose_data[0].get("frames", {})
        available_A = {int(frm): data["people"] for frm, data in frames_A.items()}

        frames_B = pose_data[1].get("frames", {})
        available_B = {int(frm): data["people"] for frm, data in frames_B.items()}

        # Process each full-sentence entry for Person A
        for sent in sentences_A:
            people_data = []
            sf = sent["start_frame"]
            ef = sent["end_frame"]
            # Only iterate through frames if they exist in available_A
            for frame_num, people in available_A.items():
                if sf <= frame_num <= ef:
                    for person in people:
                        # Clean 3D, normalize 2D, filter lower body
                        cleaned = remove_3d_keypoints_and_normalize_2d_keypoints(
                            person.copy(), width_A, height_A
                        )
                        cleaned = filter_lower_body_keypoints(cleaned)
                        people_data.append(cleaned)
            total_A += len(people_data)
            merged_results.append({
                "full_sentence": sent["full_sentence"],
                "gloss_sequence": sent["gloss_sequence"],
                "pose_sequence": people_data
            })

        # Process each full-sentence entry for Person B
        for sent in sentences_B:
            people_data = []
            sf = sent["start_frame"]
            ef = sent["end_frame"]
            for frame_num, people in available_B.items():
                if sf <= frame_num <= ef:
                    for person in people:
                        cleaned = remove_3d_keypoints_and_normalize_2d_keypoints(
                            person.copy(), width_B, height_B
                        )
                        cleaned = filter_lower_body_keypoints(cleaned)
                        people_data.append(cleaned)
            total_B += len(people_data)
            merged_results.append({
                "full_sentence": sent["full_sentence"],
                "gloss_sequence": sent["gloss_sequence"],
                "pose_sequence": people_data
            })

        print(f"[INFO] Total pose frames mapped for Person A: {total_A}")
        print(f"[INFO] Total pose frames mapped for Person B: {total_B}")
        print(f"[INFO] Total sentence mappings: {len(merged_results)}")
        return merged_results

    except Exception as e:
        print(f"[ERROR] {e}")
        return []


def process_folder(folder_path):
    """
    For a given folder containing:
      - video-a.mp4
      - video-b.mp4
      - speaker-a.srt
      - speaker-b.srt
      - openpose.json
    this will:
      1) Compute FPS from whichever video exists first (A preferred).
      2) Parse each .srt to find full-sentence entries for A and B.
      3) Map those full sentences + gloss sequences → pose sequences.
      4) Write out a JSON {"data": [ ... ]} to sentence2pose.json in that folder.
    """
    video_a = os.path.join(folder_path, "video-a.mp4")
    video_b = os.path.join(folder_path, "video-b.mp4")
    srt_a = os.path.join(folder_path, "speaker-a.srt")
    srt_b = os.path.join(folder_path, "speaker-b.srt")
    openpose = os.path.join(folder_path, "openpose.json")

    print(f"[INFO] Processing folder: {folder_path}")
    fps = get_video_fps(video_a if os.path.exists(video_a) else video_b)
    if fps is None:
        print(f"[ERROR] Could not determine FPS for folder: {folder_path}. Skipping.")
        return None

    # 1) Parse SRTs for full-sentence entries
    sentences_A = parse_srt_for_sentence_entries(srt_a, fps, "A")
    sentences_B = parse_srt_for_sentence_entries(srt_b, fps, "B")

    # 2) Map those sentences to pose sequences
    mapped = map_sentences_to_poses(openpose, sentences_A, sentences_B)
    output = { "data": mapped }

    output_path = os.path.join(folder_path, "sentence2pose.json")
    print(f"[INFO] Writing output to {output_path}")
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=4, ensure_ascii=False)
        print(f"[INFO] Successfully wrote sentence2pose.json for folder: {folder_path}")
    except Exception as e:
        print(f"[ERROR] Could not write output file for folder {folder_path}: {e}")

    return mapped


def main():
    starting_entry = None
    if len(sys.argv) > 1:
        starting_entry = sys.argv[1]

    if process_all_folders:
        root_path = "/Volumes/IISY/DGSKorpus/"  # ← change this to your actual root
        entries = [
            entry for entry in os.listdir(root_path)
            if os.path.isdir(os.path.join(root_path, entry)) and entry.startswith("entry_")
        ]
        if starting_entry:
            try:
                idx = entries.index(starting_entry)
                entries = entries[idx:]
            except ValueError:
                print(f"[ERROR] The folder {starting_entry} was not found in {root_path}.")
                return

        for entry in entries:
            entry_path = os.path.join(root_path, entry)
            process_folder(entry_path)
    else:
        # For testing one folder:
        folder_path = "/Volumes/IISY/DGSKorpus/entry_3"
        process_folder(folder_path)


if __name__ == "__main__":
    main()
