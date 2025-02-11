import json
import os
import cv2
import numpy as np
from pose_utils import create_upper_body_pose_image

# Load JSON data
def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data['pose_sequence']

# Generate pose images and create video with fixed dimensions (1280x720)
def generate_videos_from_poses(data, output_dir='output', fps=50):
    os.makedirs(output_dir, exist_ok=True)
    
    video_path = os.path.join(output_dir, "pose-sequence-test.mp4")
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

# Main script
if __name__ == "__main__":
    json_file = "../../resources/input/openpose_processed.json"
    output_dir = "../../resources/output/pose-sequences"

    data = load_json(json_file)
    generate_videos_from_poses(data, output_dir=output_dir)
