import os
import json
import sys
import cv2
from datetime import datetime
from tqdm import tqdm
import shutil
import csv
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from visualization.pose_utils import create_upper_body_pose_image
from scale_keypoint_proportions import scale_all_files

skip_interpolation = False

# ---------------------- Configuration ----------------------
# Path to gloss dictionary JSON
root_folder = "/Volumes/IISY/DGSKorpus"
dict_path = "./resources/gloss2pose_dictionary.json"
# Output directory for the intermediary gloss JSON files
gloss_output_dir = "outputs/gloss2pose_dictionary_output"
# Final output directory for pose sequence video and MimicMotion config
final_output_dir = "./outputs/pose-sequence-videos"
# Input directory for testing/debugging with the main method instead of gloss2pose.py (using the intermediary gloss output directory)
input_sentence_path = gloss_output_dir

use_default_num_intermediate_frames = False  # Set to False to use CSV values for frame interpolation
DEFAULT_NUM_INTERMEDIATE_FRAMES = 7 # Seven Because the Median IGT/OGT over all glosses in the entire DGS Korpus Release 3 is 7 Frames!
gloss_times_csv = "./resources/gloss_times_for_frame_interpolation.csv"
# -----------------------------------------------------------

# ---------------------- Helper Functions ----------------------
def load_gloss_dictionary(dict_path):
    try:
        file_size = os.path.getsize(dict_path)
        with open(dict_path, 'r', encoding='utf-8') as f:
            with tqdm(total=file_size, desc="Loading gloss2pose dictionary", unit="B", unit_scale=True) as pbar:
                loaded_dict = json.load(f)
                pbar.update(file_size)
        return loaded_dict
    except FileNotFoundError:
        print(f"Error: Could not find the dictionary file at {dict_path}")
        sys.exit(1)

def clear_directory(directory):
    """
    Removes all files and subdirectories from the specified directory.
    """
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
    else:
        os.makedirs(directory)

def create_gloss_json_files(gloss_list, loaded_dict, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    counter = 1
    for gloss in tqdm(gloss_list, desc="Processing glosses", unit="gloss"):
        if gloss in loaded_dict:
            # Get the pose sequence for this gloss
            pose_sequence = loaded_dict[gloss]
            output_data = {
                "gloss": gloss,
                "pose_sequence": pose_sequence
            }
            output_file = os.path.join(output_dir, f"{counter}.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f)
            counter += 1
        else:
            print(f"Gloss '{gloss}' not found in the dictionary. Skipping.")

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['pose_sequence']

def load_data_frames_from_path(sentence_path):
    """
    Loads all JSON files in the specified directory in numeric order,
    using load_json to extract the pose sequence from each.
    Returns a list of pose sequences.
    """
    data_frames = []
    json_files = [f for f in os.listdir(sentence_path) if f.lower().endswith(".json")]
    json_files.sort(key=lambda x: int(os.path.splitext(x)[0]))
    for file_name in json_files:
        file_path = os.path.join(sentence_path, file_name)
        data = load_json(file_path)
        data_frames.append(data)
    return data_frames

def interpolate_keypoints(current_data, next_data, num_intermediate_frames=7):
    """
    Interpolates between the last frame of current_data and the first frame of next_data.
    For each key in pose_keypoints_2d, face_keypoints_2d, hand_left_keypoints_2d, and hand_right_keypoints_2d,
    the x and y coordinates are linearly interpolated over num_intermediate_frames frames.
    The confidence value is taken from the end frame.
    
    Returns:
        The combined list: current_data followed by the intermediate frames.
    """
    if skip_interpolation: 
        return current_data
    start_frame = current_data[-1]
    end_frame = next_data[0]
    keys = ["pose_keypoints_2d", "face_keypoints_2d", "hand_left_keypoints_2d", "hand_right_keypoints_2d"]
    intermediate_frames = []
    for i in range(1, num_intermediate_frames + 1):
        alpha = i / (num_intermediate_frames + 1)
        new_frame = {}
        for key in keys:
            start_list = start_frame.get(key, [])
            end_list = end_frame.get(key, [])
            interpolated_list = []
            for j in range(0, len(start_list), 3):
                if j + 2 < len(start_list) and j + 2 < len(end_list):
                    start_x = start_list[j]
                    start_y = start_list[j + 1]
                    end_x = end_list[j]
                    end_y = end_list[j + 1]
                    end_conf = end_list[j + 2]
                    new_x = (1 - alpha) * start_x + alpha * end_x
                    new_y = (1 - alpha) * start_y + alpha * end_y
                    interpolated_list.extend([new_x, new_y, end_conf])
            new_frame[key] = interpolated_list
        intermediate_frames.append(new_frame)
    return current_data + intermediate_frames

def stretch_pose_sequence_evenly(seq, target_frames, fps=50):
    """
    If seq has length L and we need T frames total, compute E = T - L extra.
    Split E across the L-1 gaps:
      - base = E // (L-1)
      - remainder = E % (L-1), spread 1 extra into the first `remainder` gaps.
    For each gap i, interpolate that many frames between seq[i] and seq[i+1].
    Returns a new, length‐T list of frames.
    """
    L = len(seq)
    # Nothing to do if already long enough or too few frames to interpolate
    if L < 2 or L >= target_frames:
        return seq

    extra = target_frames - L
    num_gaps = L - 1
    base_per_gap = extra // num_gaps
    remainder   = extra % num_gaps

    new_seq = []
    for i in range(num_gaps):
        # Always keep the original frame i
        new_seq.append(seq[i])

        # Determine how many to insert in this gap
        n_int = base_per_gap + (1 if i < remainder else 0)
        if n_int > 0:
            # interpolate_keypoints([start], [end], n_int) returns [start] + n_int intermediates
            intermediates = interpolate_keypoints(
                current_data=[seq[i]],
                next_data   =[seq[i+1]],
                num_intermediate_frames=n_int
            )[1:]  # drop the duplicated “start” frame
            new_seq.extend(intermediates)

    # Finally add the last original frame
    new_seq.append(seq[-1])
    return new_seq

def fill_pose_sequence_duration(gloss_list, data_frames, gloss_times, fps=50):
    """
    For each gloss in gloss_list, look up its median_gd (ms), compute the
    target frame count at `fps`, then stretch its sequence evenly.
    """
    filled = []
    for gloss, seq in zip(gloss_list, data_frames):
        info = gloss_times.get(gloss, {})
        median_gd = info.get('median_gd')
        # If no valid median_gd, leave untouched
        if median_gd is None:
            filled.append(seq)
            continue

        # compute desired length
        target = int(round((median_gd / 1000.0) * fps))
        # stretch (or leave alone) so that len == target
        new_seq = stretch_pose_sequence_evenly(seq, target, fps=fps)
        filled.append(new_seq)

    return filled

def load_gloss_times(csv_filepath):
    """
    Reads the CSV file with gloss times and returns a dictionary
    mapping each gloss to its median_igt and median_ogt values.
    """
    gloss_times = {}
    try:
        with open(csv_filepath, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                gloss = row['gloss'].strip()
                try:
                    median_igt = float(row['median_igt'])
                    median_ogt = float(row['median_ogt'])
                    median_gd  = float(row['median_gd'])
                except ValueError:
                    median_igt = median_ogt = median_gd = 0.0
                gloss_times[gloss] = {
                    'median_igt': median_igt,
                    'median_ogt': median_ogt,
                    'median_gd':  median_gd
                }
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_filepath}")
        sys.exit(1)
    return gloss_times

def create_config_yml(video_filename, output_dir, num_frames, config_filename):
    config_filepath = os.path.join(output_dir, config_filename)
    config_content = f"""# base svd model path
base_model_path: stabilityai/stable-video-diffusion-img2vid-xt-1-1

# checkpoint path
ckpt_path: models/MimicMotion_1-1.pth

test_case:
  - ref_video_path: assets/example_data/videos/pose-sequence-videos/{video_filename}
    ref_image_path: assets/example_data/images/reference-images/ref.jpg
    num_frames: {num_frames}
    resolution: 576
    frames_overlap: 6
    num_inference_steps: 25
    noise_aug_strength: 0
    guidance_scale: 2.0
    sample_stride: 2
    fps: 50
    seed: 42
    use_preprocessed_video_pose: true 
"""
    with open(config_filepath, "w", encoding='utf-8') as f:
        f.write(config_content)
    print(f"Config file created: {config_filepath}")
    return config_filepath, config_filename

def generate_videos_from_poses_and_create_config_yml(data, output_filename, config_filename, output_dir='output', fps=50):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_filename = f"{output_filename}_pose.mp4"
    video_path = os.path.join(output_dir, video_filename)
    print("Started generating pose sequence video...")
    fixed_width, fixed_height = 1280, 720
    first_frame = True
    video_writer = None

    for frame_data in tqdm(data, desc="Generating video frames", unit="frame"):
        pose_keypoints = frame_data.get("pose_keypoints_2d", [])
        face_keypoints = frame_data.get("face_keypoints_2d", [])
        hand_left_keypoints = frame_data.get("hand_left_keypoints_2d", [])
        hand_right_keypoints = frame_data.get("hand_right_keypoints_2d", [])

        pose_image = create_upper_body_pose_image(pose_keypoints, face_keypoints, hand_left_keypoints, hand_right_keypoints)
        if pose_image is not None:
            resized_frame = cv2.resize(pose_image, (fixed_width, fixed_height))
            if first_frame:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(video_path, fourcc, fps, (fixed_width, fixed_height))
                first_frame = False
            video_writer.write(resized_frame)
    if video_writer:
        video_writer.release()
        print(f"Video created: {video_path}")

    num_frames = len(data)
    config_filepath, config_filename = create_config_yml(video_filename, output_dir, num_frames, config_filename)
    return config_filepath, video_path, config_filename, video_filename

def save_pose_sequence_json(pose_sequence, output_dir, timestamp):
    """
    Saves `pose_sequence` (a list of frame‐dicts) as a JSON file in `output_dir`.
    The filename will be: "final_pose_sequence_<timestamp>.json".
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = f"final_pose_sequence_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(pose_sequence, f)
    print(f"Pose sequence saved to: {filepath}")
    return filepath

def run_from_list(gloss_list, output_filename, config_filename, default_frames=False, fill_pose_sequence=False):
    """
    Given a list of glosses and optionally a reference image, 
    runs the entire pipeline and creates a pose sequence JSON file, 
    a video of the pose sequence and a config file for the inference of the MimicMotion Pose2Sign pipeline.
    """
    # override the module‐level flag if you need to
    global use_default_num_intermediate_frames
    use_default_num_intermediate_frames = default_frames

    # 1) clear old outputs
    clear_directory(gloss_output_dir)

    # 2) load dictionary & dump per‐gloss JSON
    loaded_dict = load_gloss_dictionary(dict_path)
    create_gloss_json_files(gloss_list, loaded_dict, gloss_output_dir)

    # 2.5) Scale all keypoints to match proportions of first glosses signer (to avoid morphing of head and body size)
    scale_all_files(gloss_output_dir)

    # 3) load all pose‐sequences
    data_frames = load_data_frames_from_path(gloss_output_dir)
    if not data_frames:
        raise RuntimeError("No valid pose JSON files generated")

    # 3a) load gloss timing (including median_gd) and evenly stretch each sequence
    gloss_times = load_gloss_times(gloss_times_csv)
    if fill_pose_sequence:
        data_frames = fill_pose_sequence_duration(
            gloss_list,
            data_frames,
            gloss_times,
            fps=50
        )

    # 4) stitch + interpolate
    if len(data_frames) == 1:
        final_pose_sequence = data_frames[0]
    else:
        final_pose_sequence = []
        for i, current_data in enumerate(data_frames):
            if i > 0:
                previous_data = data_frames[i - 1]
                if use_default_num_intermediate_frames:
                    # Use default fixed value
                    num_int_frames = DEFAULT_NUM_INTERMEDIATE_FRAMES
                else:
                    # Calculate num_intermediate_frames using CSV values for the gloss pair.
                    # For gloss at position i-1 (first gloss) use its median_ogt. (out-of-gloss time)
                    # For gloss at position i (next gloss) use its median_igt. (into-gloss time)
                    gloss_prev = gloss_list[i - 1]
                    gloss_curr = gloss_list[i]
                    if gloss_prev in gloss_times and gloss_curr in gloss_times:
                        median_ogt = gloss_times[gloss_prev]['median_ogt']
                        median_igt = gloss_times[gloss_curr]['median_igt']
                        avg_ms = int(round((median_ogt + median_igt) / 2))
                        num_int_frames = int(round((avg_ms / 1000.0) * 50))
                        if num_int_frames < 1:
                            num_int_frames = DEFAULT_NUM_INTERMEDIATE_FRAMES
                    else:
                        print(f"Gloss times not found for pair: {gloss_prev} and/or {gloss_curr}. Using default value.")
                        num_int_frames = DEFAULT_NUM_INTERMEDIATE_FRAMES
                # Use calculated number of intermediate frames
                interpolated = interpolate_keypoints(
                    current_data=previous_data,
                    next_data=current_data,
                    num_intermediate_frames=num_int_frames
                )[len(previous_data):]  # Exclude the overlapping frame from interpolation (Why does "interpolate_keypoints" concat the starting gloss with the interpolated frames in the first place?)

                final_pose_sequence.extend(interpolated)

            final_pose_sequence.extend(current_data)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_pose_sequence_json(final_pose_sequence, final_output_dir, timestamp)
    
    # 5) render video + write config
    cfg_path, vid_path, cfg_name, vid_name = generate_videos_from_poses_and_create_config_yml(
        final_pose_sequence, output_filename, config_filename, output_dir=final_output_dir
    )
    return cfg_path, vid_path, cfg_name, vid_name