import os
import cv2
import json
from pysrt import SubRipFile, open as open_srt

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

        print(f"[INFO] Mapped {len(srt_entries2frames_personA)} entries for Person A and {len(srt_entries2frames_personB)} entries for Person B.")
        return srt_entries2frames_personA, srt_entries2frames_personB

    except Exception as e:
        print(f"[ERROR] {e}")
        return [], []

def map_text_to_poses(openpose_file_path, srt_entries):
    """
    Map text entries to pose sequences using OpenPose JSON.

    Parameters:
        openpose_file_path (str): Path to the OpenPose JSON file.
        srt_entries (list): List of tuples containing text and frame numbers.

    Returns:
        list: List of dictionaries mapping text to pose sequences.
    """
    try:
        print(f"[INFO] Loading OpenPose data: {openpose_file_path}")
        with open(openpose_file_path, 'r') as f:
            pose_data = json.load(f)

        frame_to_people = {int(frame): data["people"] for frame, data in pose_data[0]["frames"].items()}

        result = []
        for text, frames in srt_entries:
            people_data = []
            for frame in frames:
                if frame in frame_to_people:
                    people_data.append(frame_to_people[frame])

            result.append({"text": text, "poses": people_data})

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
        return

    srt_entries_personA, srt_entries_personB = map_srt_to_frames(srt_path, fps)

    mapped_data_personA = map_text_to_poses(openpose_path, srt_entries_personA)
    mapped_data_personB = map_text_to_poses(openpose_path, srt_entries_personB)

    output = {
        "personA": mapped_data_personA,
        "personB": mapped_data_personB
    }

    output_path = os.path.join(folder_path, "gloss2pose_mapped.json")
    print(f"[INFO] Writing output to {output_path}")
    try:
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=4)
        print(f"[INFO] Process completed successfully for folder: {folder_path}")
    except Exception as e:
        print(f"[ERROR] Could not write output file for folder {folder_path}: {e}")

def main():
    process_all_folders = False  # Set to 1 to process all subfolders, 0 to process a single test folder

    if process_all_folders:
        root_path = "/Volumes/IISY/DGSKorpus/"  # Replace with the root directory containing all subfolders
        for entry in os.listdir(root_path):
            entry_path = os.path.join(root_path, entry)
            if os.path.isdir(entry_path):
                process_folder(entry_path)
    else:
        folder_path = "/Volumes/IISY/DGSKorpus/entry_0"  # Replace with your test folder
        process_folder(folder_path)

if __name__ == "__main__":
    main()
