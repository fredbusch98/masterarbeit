import json
import os
import cv2
import numpy as np
from pose_utils import create_upper_body_pose_image


process_single_file = False
sentence_path = "../../resources/input/beispielsatz/"

# Load JSON data
def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data['pose_sequence']

def load_data_frames_from_sentence_path(sentence_path):
    """
    Loads all JSON files in the specified directory in numeric order (e.g. 1.json, 2.json, ..., 10.json),
    using the load_json function, and returns a list containing the data from each file.

    Parameters:
        sentence_path (str): Path to the directory containing JSON files.
    
    Returns:
        List: A list of data frames (i.e. the loaded JSON data for each file).
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

# Generate pose images and create video with fixed dimensions (1280x720)
def generate_videos_from_poses(data, output_dir='output', fps=50):
    os.makedirs(output_dir, exist_ok=True)
    
    video_path = os.path.join(output_dir, "pose-sequence-interpolated.mp4")
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

def interpolate_keypoints(current_data, next_data):
    """
    Interpolates between the last frame of current_data and the first frame of next_data.
    For each key in pose_keypoints_2d, face_keypoints_2d, hand_left_keypoints_2d, and hand_right_keypoints_2d,
    the x and y coordinates are linearly interpolated over 20 frames.
    The confidence value is taken from the end frame.
    
    Returns:
        A new list containing all frames from current_data with 20 interpolated frames appended.
    """
    num_intermediate_frames = 7

    # Retrieve the two edge frames to interpolate between
    start_frame = current_data[-1]
    end_frame = next_data[0]
    
    # Define the keys we want to interpolate
    keys = ["pose_keypoints_2d", "face_keypoints_2d", "hand_left_keypoints_2d", "hand_right_keypoints_2d"]
    
    intermediate_frames = []
    
    # For each intermediate frame, calculate an interpolation factor alpha
    for i in range(1, num_intermediate_frames + 1):
        alpha = i / (num_intermediate_frames + 1)  # values from 1/21 to 20/21
        new_frame = {}
        for key in keys:
            start_list = start_frame.get(key, [])
            end_list = end_frame.get(key, [])
            interpolated_list = []
            # Process each keypoint triplet (x, y, confidence)
            for j in range(0, len(start_list), 3):
                # Ensure we have corresponding values in both frames
                if j + 2 < len(start_list) and j + 2 < len(end_list):
                    start_x = start_list[j]
                    start_y = start_list[j + 1]
                    # We now ignore the start frame's confidence and use the end frame's confidence
                    end_x = end_list[j]
                    end_y = end_list[j + 1]
                    end_conf = end_list[j + 2]
                    
                    new_x = (1 - alpha) * start_x + alpha * end_x
                    new_y = (1 - alpha) * start_y + alpha * end_y
                    
                    interpolated_list.extend([new_x, new_y, end_conf])
            new_frame[key] = interpolated_list
        intermediate_frames.append(new_frame)
    
    # Append the interpolated frames to the original current_data and return the new sequence.
    return current_data + intermediate_frames

# Main script
if __name__ == "__main__":
    output_dir = "../../resources/output/pose-sequences"
    if process_single_file:
        json_file = "../../resources/input/satz-openpose.json"
        data = load_json(json_file)
        generate_videos_from_poses(data, output_dir=output_dir)

    else:
        data_frames = load_data_frames_from_sentence_path(sentence_path)
        interpolated_pose_sequence = []
        # Iterate over pairs: current and next element
        for current_data, next_data in zip(data_frames, data_frames[1:]):
            interpolated = interpolate_keypoints(current_data, next_data)
            interpolated_pose_sequence.extend(interpolated)
        
        generate_videos_from_poses(interpolated_pose_sequence, output_dir=output_dir)

     
