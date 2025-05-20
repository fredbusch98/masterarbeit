import os
import json
import sys
import cv2
from datetime import datetime
from scipy.signal import savgol_filter
from tqdm import tqdm
import shutil  # Added for directory removal
import csv  # NEW: Import csv module for reading the interpolation times
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from visualization.pose_utils import create_upper_body_pose_image
from scale_keypoint_proportions import scale_all_files

skip_interpolation = False

# ---------------------- Configuration ----------------------
# Path to gloss dictionary JSON
root_folder = "/Volumes/IISY/DGSKorpus"
dict_path = "./resources/gloss_dictionary.json"
# Output directory for the individual gloss JSON files
gloss_output_dir = "outputs/gloss2pose_dictionary_output"
# Final output directory for video and config
final_output_dir = "./outputs/pose-sequence-videos"
# Temporary input directory (using the gloss output directory)
input_sentence_path = gloss_output_dir

# NEW: Add new option for using the default number of intermediate frames.
use_default_num_intermediate_frames = False  # Set to False to use CSV values for frame interpolation

# NEW: Path for the CSV file with gloss times (assumed to be in the same directory as the script)
gloss_times_csv = "./resources/gloss_times_for_frame_interpolation.csv"
# -----------------------------------------------------------

# ---------------------- Helper Functions ----------------------
def load_gloss_dictionary(dict_path):
    try:
        file_size = os.path.getsize(dict_path)
        with open(dict_path, 'r', encoding='utf-8') as f:
            with tqdm(total=file_size, desc="Loading gloss dictionary", unit="B", unit_scale=True) as pbar:
                loaded_dict = json.load(f)
                pbar.update(file_size)  # Update progress bar fully
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
                "gloss": gloss,  # NEW: Include gloss in JSON for reference
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
    # Return the pose_sequence (expected key)
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

def smooth_pose_sequence_savgol(pose_sequence,
                                window_length=11,
                                polyorder=3):
    """
    Applies a Savitzky–Golay filter independently to each keypoint coordinate
    (x and y) over time. Confidence values are carried through unchanged.

    Args:
        pose_sequence: list of frames, where each frame is a dict
                       containing keys:
                       "pose_keypoints_2d", "face_keypoints_2d",
                       "hand_left_keypoints_2d", "hand_right_keypoints_2d"
                       Each of those is a flat list [x0, y0, c0, x1, y1, c1, …].
        window_length: must be odd; controls smoothing span in frames.
        polyorder: polynomial degree for fitting; < window_length.

    Returns:
        A new list of frames of same length with x/y smoothed.
    """
    # extract the four streams of keypoint lists
    streams = {k: [] for k in ["pose_keypoints_2d",
                               "face_keypoints_2d",
                               "hand_left_keypoints_2d",
                               "hand_right_keypoints_2d"]}
    # build per-keypoint time series
    for frame in pose_sequence:
        for key, buf in streams.items():
            buf.append(frame.get(key, []))

    # helper to smooth one stream
    def _smooth_stream(frames_list):
        if not frames_list:
            return []
        num_kpts = len(frames_list[0]) // 3
        # separate into arrays shape (T, num_kpts)
        xs = [[f[3*i]   for i in range(num_kpts)] for f in frames_list]
        ys = [[f[3*i+1] for i in range(num_kpts)] for f in frames_list]
        # apply filter across axis=0 (time) for each keypoint
        xs_s = savgol_filter(xs, window_length, polyorder, axis=0).tolist()
        ys_s = savgol_filter(ys, window_length, polyorder, axis=0).tolist()

        out = []
        T = len(frames_list)
        for t in range(T):
            new_flat = []
            for i in range(num_kpts):
                new_flat.extend([
                    xs_s[t][i],
                    ys_s[t][i],
                    # just carry confidence from original
                    frames_list[t][3*i+2]
                ])
            out.append(new_flat)
        return out

    # smooth each
    smoothed_streams = {k: _smooth_stream(v) for k, v in streams.items()}

    # rebuild list of frames
    smoothed_seq = []
    T = len(pose_sequence)
    for t in range(T):
        smoothed_frame = {}
        for k in streams:
            smoothed_frame[k] = smoothed_streams[k][t]
        smoothed_seq.append(smoothed_frame)

    return smoothed_seq

# NEW: Helper function to load gloss times for frame interpolation from CSV.
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

def create_config_yml(timestamp, video_filename, output_dir, num_frames):
    config_filename = f"config-{timestamp}.yml"
    config_filepath = os.path.join(output_dir, config_filename)
    config_content = f"""# base svd model path
base_model_path: stabilityai/stable-video-diffusion-img2vid-xt-1-1

# checkpoint path
ckpt_path: models/MimicMotion_1-1.pth

test_case:
  - ref_video_path: assets/example_data/videos/{video_filename}
    ref_image_path: assets/example_data/images/ref.jpg
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

def generate_videos_from_poses_and_create_config_yml(data, output_dir='output', fps=50):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_filename = f"pose-sequence_{timestamp}.mp4"
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
    # Pass correct number of frames
    num_frames = len(data)
    config_filepath, config_filename = create_config_yml(timestamp, video_filename, output_dir, num_frames)
    return config_filepath, video_path, config_filename, video_filename

def run_from_list(gloss_list, default_frames=False, fill_pose_sequence=False):
    """
    Exactly the same pipeline as in main(), but takes
    a Python list of glosses instead of sys.argv.
    Returns (config_path, video_path, config_filename, video_filename).
    """
    # override the module‐level flag if you need to
    global use_default_num_intermediate_frames
    use_default_num_intermediate_frames = default_frames

    # 1) clear old outputs
    clear_directory(gloss_output_dir)

    # 2) load dictionary & dump per‐gloss JSON
    loaded_dict = load_gloss_dictionary(dict_path)
    create_gloss_json_files(gloss_list, loaded_dict, gloss_output_dir)

    # 2.5) Scale all keypoints to match proportions of first glosses signer (to avoid morphing heade and body size!)
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
                    num_int_frames = 7
                else:
                    # NEW: Calculate num_intermediate_frames using CSV values for the gloss pair.
                    # For gloss at position i-1 (first gloss) use its median_ogt.
                    # For gloss at position i (next gloss) use its median_igt.
                    gloss_prev = gloss_list[i - 1]
                    gloss_curr = gloss_list[i]
                    if gloss_prev in gloss_times and gloss_curr in gloss_times:
                        median_ogt = gloss_times[gloss_prev]['median_ogt']
                        median_igt = gloss_times[gloss_curr]['median_igt']
                        avg_ms = int(round((median_ogt + median_igt) / 2))
                        num_int_frames = int(round((avg_ms / 1000.0) * 50))
                    else:
                        print(f"Gloss times not found for pair: {gloss_prev} and/or {gloss_curr}. Using default value.")
                        num_int_frames = 7
                # Use calculated number of intermediate frames
                interpolated = interpolate_keypoints(previous_data, current_data, num_intermediate_frames=num_int_frames)
                # Exclude the overlapping frame from interpolation
                final_pose_sequence.extend(interpolated[len(previous_data):])
            final_pose_sequence.extend(current_data)

    # 5) render video + write config
    cfg_path, vid_path, cfg_name, vid_name = generate_videos_from_poses_and_create_config_yml(
        final_pose_sequence, output_dir=final_output_dir
    )
    return cfg_path, vid_path, cfg_name, vid_name
# -----------------------------------------------------------

def main():
    # Ensure that a gloss sequence is provided via command-line
    if len(sys.argv) < 2:
        print("Please provide a single gloss or a comma-separated sequence of glosses.")
        print("Example (single gloss): python script.py GLOSS1")
        print("Example (multiple glosses): python script.py GLOSS1,GLOSS2,GLOSS3")
        sys.exit(1)
    input_sequence = sys.argv[1]
    gloss_list = [gloss.strip() for gloss in input_sequence.split(',')]

    # Clear the output directory before creating new files
    clear_directory(gloss_output_dir)

    # Step 1: Load gloss dictionary and create JSON files
    loaded_dict = load_gloss_dictionary(dict_path)
    create_gloss_json_files(gloss_list, loaded_dict, gloss_output_dir)
    print(f"Gloss JSON files created in '{gloss_output_dir}'.")

    # NEW: If not using default interpolation frames, load gloss times from CSV.
    if not use_default_num_intermediate_frames:
        gloss_times = load_gloss_times(gloss_times_csv)

    # Step 2: Load pose sequences from created JSON files
    scale_all_files(gloss_output_dir)
    data_frames = load_data_frames_from_path(input_sentence_path)
    if not data_frames:
        print("No valid pose sequence JSON files found. Exiting.")
        sys.exit(1)

    # Step 3: Combine pose sequences with interpolation
    if len(data_frames) == 1:
        final_pose_sequence = data_frames[0]
    else:
        final_pose_sequence = []
        for i, current_data in enumerate(data_frames):
            if i > 0:
                previous_data = data_frames[i - 1]
                if use_default_num_intermediate_frames:
                    # Use default fixed value
                    num_int_frames = 7
                else:
                    # NEW: Calculate num_intermediate_frames using CSV values for the gloss pair.
                    # For gloss at position i-1 (first gloss) use its median_ogt.
                    # For gloss at position i (next gloss) use its median_igt.
                    gloss_prev = gloss_list[i - 1]
                    gloss_curr = gloss_list[i]
                    if gloss_prev in gloss_times and gloss_curr in gloss_times:
                        median_ogt = gloss_times[gloss_prev]['median_ogt']
                        median_igt = gloss_times[gloss_curr]['median_igt']
                        avg_ms = int(round((median_ogt + median_igt) / 2))
                        num_int_frames = int(round((avg_ms / 1000.0) * 50))
                    else:
                        print(f"Gloss times not found for pair: {gloss_prev} and/or {gloss_curr}. Using default value.")
                        num_int_frames = 7
                # Use calculated number of intermediate frames
                interpolated = interpolate_keypoints(previous_data, current_data, num_intermediate_frames=num_int_frames)
                # Exclude the overlapping frame from interpolation
                final_pose_sequence.extend(interpolated[len(previous_data):])
            final_pose_sequence.extend(current_data)

    # Step 3.5 apply smoothing
    # choose window_length odd, e.g. 11 frames (~0.22s at 50fps), polyorder=3
    final_pose_sequence = smooth_pose_sequence_savgol(
        final_pose_sequence,
        window_length=11,
        polyorder=3
    )

    # Step 4: Generate the video and config YAML
    config_path, video_path, config_filename, video_filename = generate_videos_from_poses_and_create_config_yml(
        final_pose_sequence, output_dir=final_output_dir
    )
    
    abs_config_path = os.path.abspath(config_path)
    abs_video_path = os.path.abspath(video_path)
    print("")
    print("Copy video and config to the mimicmotion pod:")
    print(f"kubectl cp {abs_config_path} s85468/mimicmotion:/storage/MimicMotion/configs/{config_filename}")
    print(f"kubectl cp {abs_video_path} s85468/mimicmotion:/storage/MimicMotion/configs/{video_filename}")
    print("")
    print("Start inference with the config on the mimicmotion pod:")
    print(f"python inference.py --inference_config configs/{config_filename}")

if __name__ == "__main__":
    main()

# Example sentence: BEREICH1A,INTERESSE1A,MERKWÜRDIG1,GEBÄRDEN1A,FASZINIEREND2,GEBÄRDEN1A,SPIELEN2,BEREICH1A,INTERESSE1A,SPIELEN2