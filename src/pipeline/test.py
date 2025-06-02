import os
import cv2
import mediapipe as mp

# Initialize MediaPipe solutions
mp_pose      = mp.solutions.pose
mp_face_det  = mp.solutions.face_detection
mp_hands     = mp.solutions.hands
mp_drawing   = mp.solutions.drawing_utils

# Initialize detectors
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
_hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5
)

def visualize_ref_detections(image_path, output_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # === Pose Detection ===
    pose_results = _pose.process(rgb)
    hand_coords = []  # store hand landmark pixel coordinates for optional bounding

    if pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(
            img,
            pose_results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,200,0), thickness=2)
        )

        landmarks = pose_results.pose_landmarks.landmark

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

        upper_body_landmarks = [
            landmarks[idx] for idx in upper_body_indices
            if landmarks[idx].visibility > 0.5
        ]

        if upper_body_landmarks:
            x_vals = [lm.x * w for lm in upper_body_landmarks]
            y_vals = [lm.y * h for lm in upper_body_landmarks]

            min_x, max_x = int(min(x_vals)), int(max(x_vals))
            min_y, max_y = int(min(y_vals)), int(max(y_vals))

            cv2.rectangle(img, (min_x, min_y), (max_x, max_y), color=(255, 0, 0), thickness=2)
    else:
        print(f"[WARNING] No pose landmarks detected.")

    # === Face Detection ===
    face_results = _face_detection.process(rgb)
    if face_results.detections:
        detection = face_results.detections[0]
        bbox = detection.location_data.relative_bounding_box
        x_min = int(bbox.xmin * w)
        y_min = int(bbox.ymin * h)
        width = int(bbox.width * w)
        height = int(bbox.height * h)

        top_left     = (x_min, y_min)
        bottom_right = (x_min + width, y_min + height)
        cv2.rectangle(img, top_left, bottom_right, color=(0,0,255), thickness=2)

        corners = [
            (x_min, y_min),
            (x_min + width, y_min),
            (x_min + width, y_min + height),
            (x_min, y_min + height)
        ]
        for (cx, cy) in corners:
            cv2.circle(img, (cx, cy), radius=3, color=(0,0,255), thickness=-1)
    else:
        print(f"[WARNING] No face detected.")

    # === Hand Detection ===
    hands_results = _hands.process(rgb)
    if hands_results.multi_hand_landmarks:
        for hand_landmarks in hands_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                img,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(100, 200, 200), thickness=2)
            )

            # Collect coordinates for current hand only
            hand_coords = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]

            # Draw bounding box for current hand
            x_vals = [pt[0] for pt in hand_coords]
            y_vals = [pt[1] for pt in hand_coords]
            cv2.rectangle(
                img,
                (min(x_vals), min(y_vals)),
                (max(x_vals), max(y_vals)),
                color=(0, 255, 255),
                thickness=2
            )
    else:
        print(f"[WARNING] No hands detected.")


    # === Save result ===
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if cv2.imwrite(output_path, img):
        print(f"[SAVED] Visualization written to {output_path}")
    else:
        raise IOError(f"Unable to write output image to {output_path}")

if __name__ == "__main__":
    ref_image = "./ref.jpg"
    out_jpeg  = "./overlay1.jpg"
    visualize_ref_detections(ref_image, out_jpeg)
