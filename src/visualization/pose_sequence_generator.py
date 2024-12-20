import json
import os
import cv2
from pose_utils import create_upper_body_pose_image

# Load JSON data
def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# Generate pose images and create videos
def generate_videos_from_poses(data, output_dir='output', fps=50):
    os.makedirs(output_dir, exist_ok=True)
    camera_frames = {}

    # Organize frames by camera
    for entry in data:
        camera = entry["camera"]
        if camera not in camera_frames:
            camera_frames[camera] = {}
        camera_frames[camera].update(entry["frames"])

    # Process each camera
    for camera, frames in camera_frames.items():
        if camera != "c":
            video_path = os.path.join(output_dir, f"{camera}_pose_sequence.mp4")
            print(f"Started generating pose-frames for camera {camera}...")

            # Collect all bounding boxes to find the largest one
            bounding_boxes = []
            for frame_id, frame_data in sorted(frames.items(), key=lambda x: int(x[0])):
                for person in frame_data["people"]:
                    _, bbox = create_upper_body_pose_image(person, frame_id, hands_and_face=True)
                    if bbox is not None:
                        bounding_boxes.append(bbox)

            # Determine the largest bounding box
            if bounding_boxes:
                min_x = min(bbox[0] for bbox in bounding_boxes)
                min_y = min(bbox[1] for bbox in bounding_boxes)
                max_x = max(bbox[2] for bbox in bounding_boxes)
                max_y = max(bbox[3] for bbox in bounding_boxes)
                largest_bbox = (min_x, min_y, max_x, max_y)
            else:
                print(f"No bounding boxes found for camera {camera}. Skipping.")
                continue

            # Write frames to video with consistent cropping
            first_frame = True
            video_writer = None

            for frame_id, frame_data in sorted(frames.items(), key=lambda x: int(x[0])):
                for person in frame_data["people"]:
                    pose_image, _ = create_upper_body_pose_image(person, frame_id, hands_and_face=True, padding=20)

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
                print(f"Video created for camera {camera}: {video_path}")

# Main script
if __name__ == "__main__":
    # Directories and file paths
    json_file = "../resources/input/openpose.json"  # Replace with your JSON file path
    output_dir = "../resources/output/pose-sequences"  # Output directory for videos

    # Load JSON data
    data = load_json(json_file)

    # Generate pose sequence videos
    generate_videos_from_poses(data, output_dir=output_dir)
