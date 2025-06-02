import os
import json
import glob
import cv2
import mediapipe as mp

def get_bbox_dims(keypoints):
    if not keypoints:
        return 0.0, 0.0
    xs = keypoints[0::3]
    ys = keypoints[1::3]
    return max(xs) - min(xs), max(ys) - min(ys)

def get_bbox_center(keypoints):
    if not keypoints:
        return 0.0, 0.0
    xs = keypoints[0::3]
    ys = keypoints[1::3]
    return (max(xs) + min(xs)) / 2.0, (max(ys) + min(ys)) / 2.0

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def compute_default_body_dims(first_json_path):
    data = load_json(first_json_path)
    return get_bbox_dims(data['pose_sequence'][0]['pose_keypoints_2d'])

def compute_default_face_dims_and_center(first_json_path):
    data = load_json(first_json_path)
    kpts = data['pose_sequence'][0].get('face_keypoints_2d', [])
    return (*get_bbox_dims(kpts), *get_bbox_center(kpts))

def scale_frame_array(kpts, sx, sy):
    for i in range(0, len(kpts), 3):
        kpts[i]   *= sx
        kpts[i+1] *= sy

def extract_ref_keypoints(image_path):
    """
    1. Load image.
    2. Run MediaPipe Pose & FaceDetection on the original image.
    3. Return normalized keypoints as flat [x_norm, y_norm, c] lists, but only for upper‐body landmarks.
       - x_norm = lm.x (already normalized by MediaPipe)
       - y_norm = lm.y (already normalized by MediaPipe)
       - c = 1.0
    Returns (pose_kpts_norm, face_kpts_norm) or (None, None) if no person detected.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Could not load reference image: {image_path}")
        return None, None

    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Initialize MediaPipe once
    mp_pose      = mp.solutions.pose
    mp_face_det  = mp.solutions.face_detection

    _pose = mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5
    )
    _face_detection = mp_face_det.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.5
    )

    # Run MediaPipe Pose on original image
    pose_results = _pose.process(rgb)
    if not pose_results.pose_landmarks:
        print(f"[WARNING] No person detected in reference image (Pose).")
        return None, None

    # Define exactly the same upper‐body indices as in script #1
    upper_body_indices = [
        mp_pose.PoseLandmark.NOSE,
        mp_pose.PoseLandmark.LEFT_EYE,
        mp_pose.PoseLandmark.RIGHT_EYE,
        mp_pose.PoseLandmark.LEFT_EAR,
        mp_pose.PoseLandmark.RIGHT_EAR,
        mp_pose.PoseLandmark.MOUTH_LEFT,
        mp_pose.PoseLandmark.MOUTH_RIGHT,
        mp_pose.PoseLandmark.LEFT_SHOULDER,
        mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_ELBOW,
        mp_pose.PoseLandmark.RIGHT_ELBOW,
        mp_pose.PoseLandmark.LEFT_WRIST,
        mp_pose.PoseLandmark.RIGHT_WRIST,
        mp_pose.PoseLandmark.LEFT_HIP,
        mp_pose.PoseLandmark.RIGHT_HIP
    ]

    landmarks = pose_results.pose_landmarks.landmark

    upper_body_landmarks = [
        landmarks[idx]
        for idx in upper_body_indices
    ]

    if not upper_body_landmarks:
        print(f"[WARNING] No sufficiently visible upper‐body landmarks detected in reference image.")
        return None, None

    # Build normalized pose keypoints list, but only for upper‐body landmarks
    pose_kpts_norm = []
    for lm in upper_body_landmarks:
        x_norm = lm.x
        y_norm = lm.y
        c      = 1.0
        pose_kpts_norm.extend([x_norm, y_norm, c])

    # Now run FaceDetection to get face bbox (normalized)
    face_results = _face_detection.process(rgb)
    if not face_results.detections:
        print(f"[WARNING] No face detected in reference image (FaceDetection).")
        return pose_kpts_norm, []

    detection = face_results.detections[0]
    bbox = detection.location_data.relative_bounding_box
    x_min = bbox.xmin
    y_min = bbox.ymin
    width = bbox.width
    height = bbox.height

    # Four corners of the normalized face bbox
    face_kpts_norm = [
        x_min,               y_min,               1.0,  # top-left
        x_min + width,       y_min,               1.0,  # top-right
        x_min + width,       y_min + height,      1.0,  # bottom-right
        x_min,               y_min + height,      1.0,  # bottom-left
    ]

    return pose_kpts_norm, face_kpts_norm

def scale_all_files(folder_path, reference_image_path=None, uniform=True):
    files = sorted(
        glob.glob(os.path.join(folder_path, '*.json')),
        key=lambda f: int(os.path.splitext(os.path.basename(f))[0])
    )
    if not files:
        raise RuntimeError(f"No JSON files found in {folder_path}")

    use_ref = False
    if reference_image_path:
        print(f"[INFO] Reference image provided: {reference_image_path}")
        pose_kpts, face_kpts = extract_ref_keypoints(reference_image_path)
        # Only use reference if both upper‐body and face were detected
        if pose_kpts and face_kpts:
            default_body_w, default_body_h = get_bbox_dims(pose_kpts)
            fw, fh = get_bbox_dims(face_kpts)
            cx, cy = get_bbox_center(face_kpts)
            default_face_w, default_face_h = fw, fh
            default_face_cx, default_face_cy = cx, cy

            print(f"[SUCCESS] Upper‐body and face keypoints extracted from reference image (normalized).")
            use_ref = True
        else:
            if not pose_kpts:
                print(f"[WARNING] Falling back to first frame of 1.json as golden reference (no upper‐body detected).")
            elif not face_kpts:
                print(f"[WARNING] Falling back to first frame of 1.json as golden reference (no face detected).")

    if not use_ref:
        # Fallback: use 1.json (already normalized)
        default_body_w, default_body_h = compute_default_body_dims(files[0])
        default_face_w, default_face_h, default_face_cx, default_face_cy = \
            compute_default_face_dims_and_center(files[0])
        print(f"[INFO] Using first frame of '{os.path.basename(files[0])}' as golden reference.")

    print(f"Default body dims (upper‐body bbox): {default_body_w:.3f}×{default_body_h:.3f}")
    print(f"Default face dims: {default_face_w:.3f}×{default_face_h:.3f}, center @ ({default_face_cx:.3f},{default_face_cy:.3f})")

    # Determine which files to scale: if using reference, include all; otherwise skip first
    start_index = 0 if use_ref else 1

    for path in files[start_index:]:
        data = load_json(path)
        first = data['pose_sequence'][0]

        # --- 1) compute body scale from first frame (normalized dims, full‐body from JSON)
        bw, bh = get_bbox_dims(first['pose_keypoints_2d'])
        if bw == 0 or bh == 0:
            print(f"Skipping {os.path.basename(path)} (zero‐body bbox)")
            continue
        sx_body = default_body_w / bw
        sy_body = default_body_h / bh
        if uniform:
            s = (sx_body + sy_body) / 2.0
            sx_body = sy_body = s

        # --- 2) compute face scale from first frame (normalized dims)
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

        # --- 3) apply to every frame
        for frame in data['pose_sequence']:
            # scale body & hands
            for key in ('pose_keypoints_2d', 'hand_left_keypoints_2d', 'hand_right_keypoints_2d'):
                arr = frame.get(key, [])
                scale_frame_array(arr, sx_body, sy_body)
                frame[key] = arr

            # scale & re-attach face
            if sx_face is not None:
                fk = frame['face_keypoints_2d']
                # a) find raw center
                cx, cy = get_bbox_center(fk)
                # b) shift to origin
                for i in range(0, len(fk), 3):
                    fk[i]   -= cx
                    fk[i+1] -= cy
                # c) resize around origin
                for i in range(0, len(fk), 3):
                    fk[i]   *= sx_face
                    fk[i+1] *= sy_face
                # d) translate to where the head should be after body‐scaling
                tx, ty = cx * sx_body, cy * sy_body
                for i in range(0, len(fk), 3):
                    fk[i]   += tx
                    fk[i+1] += ty

                frame['face_keypoints_2d'] = fk

        save_json(data, path)

    print("Scaling & re-centering complete.")

if __name__ == "__main__":
    # Example usage with reference image for scaling:
    # scale_all_files("gloss2pose_data", reference_image_path="ref.jpg")
    scale_all_files("gloss2pose_data", reference_image_path=None)
