import json
import os
import cv2
from pose_utils import create_upper_body_pose_image

# Load JSON data
def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data['pose_sequence']  # Return only the pose_sequence array

# Generate pose images and create videos
def generate_videos_from_poses(data, output_dir='output', fps=50):
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each frame in the pose_sequence array
    video_path = os.path.join(output_dir, "pose_sequence.mp4")
    print(f"Started generating pose sequence video...")

    # Collect all bounding boxes to find the largest one
    bounding_boxes = []

    for frame_id, frame_data in enumerate(data):  # Use frame index instead of frame IDs
        pose_keypoints = frame_data.get("pose_keypoints_2d", [])
        face_keypoints = frame_data.get("face_keypoints_2d", [])
        hand_left_keypoints = frame_data.get("hand_left_keypoints_2d", [])
        hand_right_keypoints = frame_data.get("hand_right_keypoints_2d", [])
        
        # If you want to use these keypoints to generate pose images, 
        # call the appropriate function to process them.
        # Example: create_pose_image function should be adapted to handle this data
        _, bbox = create_upper_body_pose_image(
            pose_keypoints, face_keypoints, hand_left_keypoints, hand_right_keypoints, frame_id, hands_and_face=True
        )
        
        if bbox is not None:
            bounding_boxes.append(bbox)

    # Determine the largest bounding box (if there are any)
    if bounding_boxes:
        min_x = min(bbox[0] for bbox in bounding_boxes)
        min_y = min(bbox[1] for bbox in bounding_boxes)
        max_x = max(bbox[2] for bbox in bounding_boxes)
        max_y = max(bbox[3] for bbox in bounding_boxes)
        largest_bbox = (min_x, min_y, max_x, max_y)
    else:
        print("No bounding boxes found. Skipping video creation.")
        return

    # Write frames to video with consistent cropping
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
            # Crop image using the largest bounding box
            min_x, min_y, max_x, max_y = largest_bbox
            cropped_image = pose_image[min_y:max_y, min_x:max_x]

            # Initialize video writer on the first frame
            if first_frame:
                height, width, _ = cropped_image.shape
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
                first_frame = False

            # Write the cropped frame to the video
            video_writer.write(cropped_image)

    if video_writer:
        video_writer.release()
        print(f"Video created: {video_path}")

# Main script
if __name__ == "__main__":
    # Directories and file paths
    json_file = "../../resources/input/test-normalized.json"  # Replace with your JSON file path
    output_dir = "../../resources/output/pose-sequences"  # Output directory for videos

    # Load JSON data
    data = load_json(json_file)

    # Generate pose sequence videos
    generate_videos_from_poses(data, output_dir=output_dir)