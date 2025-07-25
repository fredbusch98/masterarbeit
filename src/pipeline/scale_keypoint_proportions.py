import os
import json
import glob
import math

def compute_distance(x1, y1, x2, y2):
    """Compute the Euclidean distance between two points."""
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def get_bbox_dims(keypoints):
    """Get the width and height of the bounding box around keypoints."""
    if not keypoints:
        return 0.0, 0.0
    xs = keypoints[0::3]
    ys = keypoints[1::3]
    return max(xs) - min(xs), max(ys) - min(ys)

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def compute_reference_neck_length(first_json_path):
    """Compute the reference neck length from the first frame of 1.json."""
    data = load_json(first_json_path)
    ref_pose = data['pose_sequence'][0]['pose_keypoints_2d']
    neck_x, neck_y = ref_pose[1*3], ref_pose[1*3+1]  # Keypoint 1: Neck
    nose_x, nose_y = ref_pose[0*3], ref_pose[0*3+1]  # Keypoint 0: Nose
    ref_neck_length = compute_distance(neck_x, neck_y, nose_x, nose_y)
    return ref_neck_length

def compute_reference_dims(first_json_path):
    """Compute reference shoulder width and upper-body length."""
    data = load_json(first_json_path)
    ref_pose = data['pose_sequence'][0]['pose_keypoints_2d']
    # Keypoints: 2 (RShoulder), 5 (LShoulder), 0 (Nose), 8 (MidHip)
    r_shoulder_x, r_shoulder_y = ref_pose[2*3], ref_pose[2*3+1]
    l_shoulder_x, l_shoulder_y = ref_pose[5*3], ref_pose[5*3+1]
    nose_x, nose_y = ref_pose[0*3], ref_pose[0*3+1]
    midhip_x, midhip_y = ref_pose[8*3], ref_pose[8*3+1]
    shoulder_width = compute_distance(r_shoulder_x, r_shoulder_y, l_shoulder_x, l_shoulder_y)
    upper_body_length = compute_distance(nose_x, nose_y, midhip_x, midhip_y)
    return shoulder_width, upper_body_length

def compute_default_face_dims(first_json_path):
    """Compute default face dimensions from the first frame."""
    data = load_json(first_json_path)
    kpts = data['pose_sequence'][0].get('face_keypoints_2d', [])
    return get_bbox_dims(kpts)

def scale_frame_array(kpts, sx, sy):
    """Scale an array of keypoints by sx and sy."""
    for i in range(0, len(kpts), 3):
        kpts[i]   *= sx  # x-coordinate
        kpts[i+1] *= sy  # y-coordinate

def fix_neck_length(pose_kpts, ref_neck_length):
    """Adjust the nose position to enforce the reference neck length."""
    neck_x, neck_y = pose_kpts[1*3], pose_kpts[1*3+1]  # Keypoint 1: Neck
    nose_x, nose_y = pose_kpts[0*3], pose_kpts[0*3+1]  # Keypoint 0: Nose
    V_x, V_y = nose_x - neck_x, nose_y - neck_y
    current_length = compute_distance(neck_x, neck_y, nose_x, nose_y)
    if current_length == 0:
        print("Warning: Current neck length is zero, skipping adjustment.")
        return neck_x, neck_y, nose_x, nose_y
    # Scale the vector to the reference length
    scale = ref_neck_length / current_length
    new_nose_x = neck_x + V_x * scale
    new_nose_y = neck_y + V_y * scale
    # Update nose position
    pose_kpts[0*3] = new_nose_x
    pose_kpts[0*3+1] = new_nose_y
    return neck_x, neck_y, new_nose_x, new_nose_y

def scale_and_position_face(fk, nose_x, nose_y, sx_face, sy_face):
    """Scale face keypoints and anchor them to the adjusted nose position."""
    if not fk or len(fk) < 31*3:  # Ensure enough keypoints (nose tip at 30)
        return
    # Nose tip is keypoint 30 in OpenPose 70-keypoint face model
    nose_tip_idx = 30
    orig_nose_x, orig_nose_y = fk[nose_tip_idx*3], fk[nose_tip_idx*3+1]
    # Compute offsets from original nose tip and scale them
    for i in range(0, len(fk), 3):
        offset_x = fk[i] - orig_nose_x
        offset_y = fk[i+1] - orig_nose_y
        fk[i] = nose_x + offset_x * sx_face
        fk[i+1] = nose_y + offset_y * sy_face
        # Confidence (i+2) remains unchanged

def scale_all_files(folder_path, uniform=True):
    """Scale all JSON files, fixing neck length to the reference from 1.json."""
    files = sorted(
        glob.glob(os.path.join(folder_path, '*.json')),
        key=lambda f: int(os.path.splitext(os.path.basename(f))[0])
    )
    if not files:
        raise RuntimeError(f"No JSON files found in {folder_path}")

    # Compute reference dimensions from 1.json
    ref_neck_length = compute_reference_neck_length(files[0])
    ref_shoulder_width, ref_upper_body_length = compute_reference_dims(files[0])
    default_face_w, default_face_h = compute_default_face_dims(files[0])
    print(f"[INFO] Reference from '{os.path.basename(files[0])}' first frame:")
    print(f"- Neck length: {ref_neck_length:.3f}")
    print(f"- Shoulder width: {ref_shoulder_width:.3f}, Upper-body length: {ref_upper_body_length:.3f}")
    print(f"- Face dims: {default_face_w:.3f}×{default_face_h:.3f}")

    # Process target files
    for path in files[1:]:
        data = load_json(path)
        first = data['pose_sequence'][0]
        target_pose = first['pose_keypoints_2d']

        # Compute target body dimensions
        r_shoulder_x, r_shoulder_y = target_pose[2*3], target_pose[2*3+1]
        l_shoulder_x, l_shoulder_y = target_pose[5*3], target_pose[5*3+1]
        nose_x, nose_y = target_pose[0*3], target_pose[0*3+1]
        midhip_x, midhip_y = target_pose[8*3], target_pose[8*3+1]
        target_shoulder_width = compute_distance(r_shoulder_x, r_shoulder_y, l_shoulder_x, l_shoulder_y)
        target_upper_body_length = compute_distance(nose_x, nose_y, midhip_x, midhip_y)

        if target_shoulder_width == 0 or target_upper_body_length == 0:
            print(f"Skipping {os.path.basename(path)} (zero shoulder width or upper-body length)")
            continue

        # Calculate body scaling factors
        sx_body = ref_shoulder_width / target_shoulder_width
        sy_body = ref_upper_body_length / target_upper_body_length
        if uniform:
            s = (sx_body + sy_body) / 2.0
            sx_body = sy_body = s

        # Compute face scaling factors
        face0 = first.get('face_keypoints_2d', [])
        fw, fh = get_bbox_dims(face0)
        if fw > 0 and fh > 0:
            sx_face = default_face_w / fw
            sy_face = default_face_h / fh
            if uniform:
                s = (sx_face + sy_face) / 2.0
                sx_face = sy_face = s
        else:
            sx_face = sy_face = None

        print(f"Scaling {os.path.basename(path)}: body→({sx_body:.3f},{sy_body:.3f}), face→({sx_face},{sy_face})")

        # Apply scaling and neck length fix to all frames
        for frame in data['pose_sequence']:
            # Scale body and hand keypoints
            for key in ('pose_keypoints_2d', 'hand_left_keypoints_2d', 'hand_right_keypoints_2d'):
                arr = frame.get(key, [])
                scale_frame_array(arr, sx_body, sy_body)
                frame[key] = arr

            # Fix neck length by adjusting nose position
            pose_kpts = frame['pose_keypoints_2d']
            neck_x, neck_y, nose_x, nose_y = fix_neck_length(pose_kpts, ref_neck_length)

            # Scale and position face keypoints
            if sx_face is not None and sy_face is not None:
                fk = frame.get('face_keypoints_2d', [])
                scale_and_position_face(fk, nose_x, nose_y, sx_face, sy_face)
                frame['face_keypoints_2d'] = fk

        # Save the modified data
        save_json(data, path)

    print("Scaling complete.")

if __name__ == "__main__":
    scale_all_files("gloss2pose_data", uniform=True)