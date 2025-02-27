import json
import os
import cv2
from datetime import datetime
from pose_utils import create_upper_body_pose_image

process_single_file = False
sentence_path = "../../resources/input/beispielsatz/"

# Load JSON data
def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data['pose_sequence']

def load_data_frames_from_path(sentence_path):
    """
    Loads all JSON files in the specified directory in numeric order (e.g. 1.json, 2.json, ..., 10.json),
    using the load_json function, and returns a list containing the data from each file.
    """
    data_frames = []
    
    # List all JSON files in the directory
    json_files = [file_name for file_name in os.listdir(sentence_path) if file_name.lower().endswith(".json")]
    
    # Sort files by their numeric value extracted from the filename (e.g., "1.json" becomes 1)
    json_files.sort(key=lambda x: int(os.path.splitext(x)[0]))
    
    # Iterate over the sorted file list
    for file_name in json_files:
        file_path = os.path.join(sentence_path, file_name)
        data = load_json(file_path)
        data_frames.append(data)
    
    return data_frames

def create_config_yml(timestamp, video_filename, output_dir):
    """
    Creates a YAML configuration file in the specified output directory.
    The filename will be config-{timestamp}.yml and the YAML content will include the video filename.
    """
    config_filename = f"config-{timestamp}.yml"
    config_filepath = os.path.join(output_dir, config_filename)
    
    config_content = f"""# base svd model path
base_model_path: stabilityai/stable-video-diffusion-img2vid-xt-1-1

# checkpoint path
ckpt_path: models/MimicMotion_1-1.pth

test_case:
  - ref_video_path: assets/example_data/videos/{video_filename}
    ref_image_path: assets/example_data/images/ref.jpg
    num_frames: 72
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
    with open(config_filepath, "w") as f:
        f.write(config_content)
    print(f"Config file created: {config_filepath}")

    return config_filepath, config_filename

# Generate pose images and create video with fixed dimensions (1280x720)
def generate_videos_from_poses_and_create_config_yml(data, output_dir='output', fps=50):
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate a timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_filename = f"pose-sequence_{timestamp}.mp4"
    video_path = os.path.join(output_dir, video_filename)
    print(f"Started generating pose sequence video...")
    
    # Define fixed output dimensions
    fixed_width = 1280
    fixed_height = 720

    first_frame = True
    video_writer = None

    for frame_id, frame_data in enumerate(data):
        pose_keypoints = frame_data.get("pose_keypoints_2d", [])
        face_keypoints = frame_data.get("face_keypoints_2d", [])
        hand_left_keypoints = frame_data.get("hand_left_keypoints_2d", [])
        hand_right_keypoints = frame_data.get("hand_right_keypoints_2d", [])

        # Create the pose image (the function can still add its own padding if needed)
        pose_image = create_upper_body_pose_image(
            pose_keypoints, face_keypoints, hand_left_keypoints, hand_right_keypoints
        )

        if pose_image is not None:
            # Resize the pose image to the fixed dimensions (1280x720)
            resized_frame = cv2.resize(pose_image, (fixed_width, fixed_height))

            # Initialize video writer on the first frame
            if first_frame:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(video_path, fourcc, fps, (fixed_width, fixed_height))
                first_frame = False

            # Write the resized frame to the video
            video_writer.write(resized_frame)

    if video_writer:
        video_writer.release()
        print(f"Video created: {video_path}")
    
    # Create the YAML configuration file with the same timestamp and video filename
    config_filepath, config_filename = create_config_yml(timestamp, video_filename, output_dir)

    return config_filepath, video_path, config_filename, video_filename


def interpolate_keypoints(current_data, next_data, num_intermediate_frames=7):
    """
    Interpolates between the last frame of current_data and the first frame of next_data.
    For each key in pose_keypoints_2d, face_keypoints_2d, hand_left_keypoints_2d, and hand_right_keypoints_2d,
    the x and y coordinates are linearly interpolated over num_intermediate_frames frames.
    The confidence value is taken from the end frame.
    
    Returns:
        A new list containing all frames from current_data with interpolated frames appended.
    """
    # Retrieve the two edge frames to interpolate between
    start_frame = current_data[-1]
    end_frame = next_data[0]
    
    # Define the keys we want to interpolate
    keys = ["pose_keypoints_2d", "face_keypoints_2d", "hand_left_keypoints_2d", "hand_right_keypoints_2d"]
    
    intermediate_frames = []
    
    # For each intermediate frame, calculate an interpolation factor alpha
    for i in range(1, num_intermediate_frames + 1):
        alpha = i / (num_intermediate_frames + 1)  # values from 1/(n+1) to n/(n+1)
        new_frame = {}
        for key in keys:
            start_list = start_frame.get(key, [])
            end_list = end_frame.get(key, [])
            interpolated_list = []
            # Process each keypoint triplet (x, y, confidence)
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

# Main script
if __name__ == "__main__":
    output_dir = "../../resources/output/pose-sequences"
    if process_single_file:
        json_file = "../../resources/input/satz-openpose.json"
        data = load_json(json_file)
        config_path, video_path, config_filename, video_filename = generate_videos_from_poses_and_create_config_yml(data, output_dir=output_dir)
    else:
        data_frames = load_data_frames_from_path(sentence_path)
        interpolated_pose_sequence = []
        for current_data, next_data in zip(data_frames, data_frames[1:]):
            interpolated = interpolate_keypoints(current_data, next_data, num_intermediate_frames=7)
            interpolated_pose_sequence.extend(interpolated)
        
        config_path, video_path, config_filename, video_filename = generate_videos_from_poses_and_create_config_yml(interpolated_pose_sequence, output_dir=output_dir)
    
    abs_config_path = os.path.abspath(config_path)
    abs_video_path = os.path.abspath(video_path)

    print("")
    print("Copy video and config to the mimicmotion pod:")
    print("")
    print(f"kubectl cp {abs_config_path} s85468/mimicmotion:/storage/MimicMotion/configs/{config_filename}")
    print("")
    print(f"kubectl cp {abs_video_path} s85468/mimicmotion:/storage/MimicMotion/configs/{video_filename}")
    print("")
    print("")
    print(f"Start inference with the config on the mimicmotion pod: python inference.py --inference_config configs/{config_filename}")

