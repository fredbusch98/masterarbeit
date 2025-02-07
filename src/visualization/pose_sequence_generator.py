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

# Generate pose images and create videos
def generate_videos_from_poses(data, output_dir='output', fps=24):
    os.makedirs(output_dir, exist_ok=True)
    
    video_path = os.path.join(output_dir, "pose_sequence.mp4")
    print(f"Started generating pose sequence video...")

    bounding_boxes = []

    for frame_id, frame_data in enumerate(data):
        pose_keypoints = frame_data.get("pose_keypoints_2d", [])
        face_keypoints = frame_data.get("face_keypoints_2d", [])
        hand_left_keypoints = frame_data.get("hand_left_keypoints_2d", [])
        hand_right_keypoints = frame_data.get("hand_right_keypoints_2d", [])
        
        _, bbox = create_upper_body_pose_image(
            pose_keypoints, face_keypoints, hand_left_keypoints, hand_right_keypoints, frame_id, hands_and_face=True
        )
        
        if bbox is not None:
            bounding_boxes.append(bbox)

    if bounding_boxes:
        min_x = min(bbox[0] for bbox in bounding_boxes)
        min_y = min(bbox[1] for bbox in bounding_boxes)
        max_x = max(bbox[2] for bbox in bounding_boxes)
        max_y = max(bbox[3] for bbox in bounding_boxes)
        largest_bbox = (min_x, min_y, max_x, max_y)
    else:
        print("No bounding boxes found. Skipping video creation.")
        return

    first_frame = True
    video_writer = None

    for frame_id, frame_data in enumerate(data):
        pose_keypoints = frame_data.get("pose_keypoints_2d", [])
        face_keypoints = frame_data.get("face_keypoints_2d", [])
        hand_left_keypoints = frame_data.get("hand_left_keypoints_2d", [])
        hand_right_keypoints = frame_data.get("hand_right_keypoints_2d", [])

        pose_image, _ = create_upper_body_pose_image(
            pose_keypoints, face_keypoints, hand_left_keypoints, hand_right_keypoints, frame_id, hands_and_face=True, padding=20
        )

        if pose_image is not None:
            min_x, min_y, max_x, max_y = largest_bbox
            cropped_image = pose_image[min_y:max_y, min_x:max_x]

            cropped_h, cropped_w, _ = cropped_image.shape

            # Create black background (padded frame) of 1920x1080 (portrait mode)
            padded_frame = np.zeros((1920, 1080, 3), dtype=np.uint8)

            # Calculate centering offsets
            y_offset = (1920 - cropped_h) // 2
            x_offset = (1080 - cropped_w) // 2

            # Place cropped image at the center of the padded frame
            padded_frame[y_offset:y_offset + cropped_h, x_offset:x_offset + cropped_w] = cropped_image

            # Initialize video writer
            if first_frame:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(video_path, fourcc, fps, (1080, 1920))
                first_frame = False

            # Write frame to video
            video_writer.write(padded_frame)

    if video_writer:
        video_writer.release()
        print(f"Video created: {video_path}")

# Main script
if __name__ == "__main__":
    json_file = "../../resources/input/gloss-SEHEN1.json"
    output_dir = "../../resources/output/pose-sequences"

    data = load_json(json_file)

    generate_videos_from_poses(data, output_dir=output_dir)
