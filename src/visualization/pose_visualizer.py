import json
from pose_utils import create_pose_image, create_upper_body_pose_image

# Load JSON data
def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# Visualize OpenPose skeleton
def visualize_openpose_cv(data, save_dir='output'):
    for frame_id, frame_data in data.items():
        for person in frame_data["people"]:
            pose_image = create_upper_body_pose_image(person, frame_id, xinsir=False, save_dir=save_dir)
            print(f"Processed frame {frame_id}")

# Main
if __name__ == "__main__":
    import os

    # Ensure output directory exists
    save_dir = "../resources/output/openpose"
    os.makedirs(save_dir, exist_ok=True)

    # Load JSON data
    json_file = "../resources/input/test-input-3.json"
    data = load_json(json_file)

    # Visualize and save skeletons
    visualize_openpose_cv(data, save_dir=save_dir)