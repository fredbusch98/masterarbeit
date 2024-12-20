import numpy as np
import cv2
import math
from typing import List, Union, NamedTuple, Tuple

#### IMPORTANT NOTE: Script is highly inspired and partly copied from the following sources:
# https://github.com/huggingface/controlnet_aux 
# https://huggingface.co/xinsir/controlnet-openpose-sdxl-1.0

eps = 0.01

class Keypoint(NamedTuple):
    x: float
    y: float
    score: float = 1.0
    id: int = -1

CONFIDENCE_THRESHOLD = 0.01

def filter_keypoints(keypoints):
    """Filter keypoints with low confidence or invalid coordinates."""
    filtered_keypoints = []
    for x, y, c in keypoints:
        if c > CONFIDENCE_THRESHOLD and x > 0 and y > 0:
            filtered_keypoints.append((x, y, c))
        else:
            filtered_keypoints.append((0.0, 0.0, 0.0))  # Replace with dummy values for missing keypoints
    return np.array(filtered_keypoints)

def draw_pose(body_keypoints, left_hand_keypoints, right_hand_keypoints, face_keypoints, hands_and_face = False, xinsir=False, H=720, W=1280):
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)
    canvas = draw_bodypose_xinsir(canvas, body_keypoints) if xinsir else draw_bodypose(canvas, body_keypoints)

    if hands_and_face:
        canvas = draw_handpose(canvas, left_hand_keypoints)
        canvas = draw_handpose(canvas, right_hand_keypoints)
        canvas = draw_facepose(canvas, face_keypoints)

    return canvas

def draw_bodypose(canvas: np.ndarray, keypoints: List[Keypoint]) -> np.ndarray:
    """
    Draw keypoints and limbs representing body pose on a given canvas.

    Args:
        canvas (np.ndarray): A 3D numpy array representing the canvas (image) on which to draw the body pose.
        keypoints (List[Keypoint]): A list of Keypoint objects representing the body keypoints to be drawn.

    Returns:
        np.ndarray: A 3D numpy array representing the modified canvas with the drawn body pose.

    Note:
        The function expects the x and y coordinates of the keypoints to be normalized between 0 and 1.
    """
    H, W, C = canvas.shape
    stickwidth = 4

    limbSeq = [
        [2, 3], [2, 6], [3, 4], [4, 5], 
        [6, 7], [7, 8], [2, 9], [9, 10], 
        [10, 11], [2, 12], [12, 13], [13, 14], 
        [2, 1], [1, 15], [15, 17], [1, 16], 
        [16, 18],
    ]

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

    for (k1_index, k2_index), color in zip(limbSeq, colors):
        keypoint1 = keypoints[k1_index - 1]
        keypoint2 = keypoints[k2_index - 1]

        if keypoint1 is None or keypoint2 is None:
            continue
        if keypoint1.score < CONFIDENCE_THRESHOLD or keypoint2.score < CONFIDENCE_THRESHOLD:
            continue  # Skip low-confidence keypoints

        Y = np.array([keypoint1.x, keypoint2.x]) * float(W)
        X = np.array([keypoint1.y, keypoint2.y]) * float(H)
        mX = np.mean(X)
        mY = np.mean(Y)
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(canvas, polygon, [int(float(c) * 0.6) for c in color])

    for keypoint, color in zip(keypoints, colors):
        if keypoint is None:
            continue
        if keypoint.score < CONFIDENCE_THRESHOLD or (keypoint.x == 0.0 and keypoint.y == 0.0):
            continue  # Skip invalid keypoints

        x, y = keypoint.x, keypoint.y
        x = int(x * W)
        y = int(y * H)
        cv2.circle(canvas, (int(x), int(y)), 4, color, thickness=-1)

    return canvas

def draw_bodypose_xinsir(canvas: np.ndarray, keypoints: List[Keypoint]) -> np.ndarray:
    """
    Draw keypoints and limbs representing body pose on a given canvas.

    Args:
        canvas (np.ndarray): A 3D numpy array representing the canvas (image) on which to draw the body pose.
        keypoints (List[Keypoint]): A list of Keypoint objects representing the body keypoints to be drawn.

    Returns:
        np.ndarray: A 3D numpy array representing the modified canvas with the drawn body pose.

    Note:
        The function expects the x and y coordinates of the keypoints to be normalized between 0 and 1.
    """
    H, W, C = canvas.shape

    
    if max(W, H) < 500:
        ratio = 1.0
    elif max(W, H) >= 500 and max(W, H) < 1000:
        ratio = 2.0
    elif max(W, H) >= 1000 and max(W, H) < 2000:
        ratio = 3.0
    elif max(W, H) >= 2000 and max(W, H) < 3000:
        ratio = 4.0
    elif max(W, H) >= 3000 and max(W, H) < 4000:
        ratio = 5.0
    elif max(W, H) >= 4000 and max(W, H) < 5000:
        ratio = 6.0
    else:
        ratio = 7.0

    stickwidth = 4

    limbSeq = [
        [2, 3], [2, 6], [3, 4], [4, 5], 
        [6, 7], [7, 8], [2, 9], [9, 10], 
        [10, 11], [2, 12], [12, 13], [13, 14], 
        [2, 1], [1, 15], [15, 17], [1, 16], 
        [16, 18],
    ]

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

    for (k1_index, k2_index), color in zip(limbSeq, colors):
        keypoint1 = keypoints[k1_index - 1]
        keypoint2 = keypoints[k2_index - 1]

        if keypoint1 is None or keypoint2 is None:
            continue
        if keypoint1.score < CONFIDENCE_THRESHOLD or keypoint2.score < CONFIDENCE_THRESHOLD:
            continue  # Skip low-confidence keypoints

        Y = np.array([keypoint1.x, keypoint2.x]) * float(W)
        X = np.array([keypoint1.y, keypoint2.y]) * float(H)
        mX = np.mean(X)
        mY = np.mean(Y)
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), int(stickwidth * ratio)), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(canvas, polygon, [int(float(c) * 0.6) for c in color])

    for keypoint, color in zip(keypoints, colors):
        if keypoint is None:
            continue
        if keypoint.score < CONFIDENCE_THRESHOLD or (keypoint.x == 0.0 and keypoint.y == 0.0):
            continue  # Skip invalid keypoints

        x, y = keypoint.x, keypoint.y
        x = int(x * W)
        y = int(y * H)
        cv2.circle(canvas, (int(x), int(y)), int(4 * ratio), color, thickness=-1)

    return canvas

def draw_handpose(canvas: np.ndarray, keypoints: Union[List[Keypoint], None]) -> np.ndarray:
    import matplotlib
    """
    Draw keypoints and connections representing hand pose on a given canvas.

    Args:
        canvas (np.ndarray): A 3D numpy array representing the canvas (image) on which to draw the hand pose.
        keypoints (List[Keypoint]| None): A list of Keypoint objects representing the hand keypoints to be drawn
                                          or None if no keypoints are present.

    Returns:
        np.ndarray: A 3D numpy array representing the modified canvas with the drawn hand pose.

    Note:
        The function expects the x and y coordinates of the keypoints to be normalized between 0 and 1.
    """
    if not keypoints:
        return canvas
    
    H, W, C = canvas.shape

    edges = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], \
             [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]

    for ie, (e1, e2) in enumerate(edges):
        k1 = keypoints[e1]
        k2 = keypoints[e2]
        if k1 is None or k2 is None:
            continue
        
        x1 = int(k1.x * W)
        y1 = int(k1.y * H)
        x2 = int(k2.x * W)
        y2 = int(k2.y * H)
        if x1 > eps and y1 > eps and x2 > eps and y2 > eps:
            cv2.line(canvas, (x1, y1), (x2, y2), matplotlib.colors.hsv_to_rgb([ie / float(len(edges)), 1.0, 1.0]) * 255, thickness=2)

    for keypoint in keypoints:
        if keypoint.score < CONFIDENCE_THRESHOLD or (keypoint.x == 0.0 and keypoint.y == 0.0):
            continue  # Skip invalid keypoints
        x, y = keypoint.x, keypoint.y
        x = int(x * W)
        y = int(y * H)
        if x > eps and y > eps:
            cv2.circle(canvas, (x, y), 4, (0, 0, 255), thickness=-1)
    return canvas


def draw_facepose(canvas: np.ndarray, keypoints: Union[List[Keypoint], None]) -> np.ndarray:
    """
    Draw keypoints representing face pose on a given canvas.

    Args:
        canvas (np.ndarray): A 3D numpy array representing the canvas (image) on which to draw the face pose.
        keypoints (List[Keypoint]| None): A list of Keypoint objects representing the face keypoints to be drawn
                                          or None if no keypoints are present.

    Returns:
        np.ndarray: A 3D numpy array representing the modified canvas with the drawn face pose.

    Note:
        The function expects the x and y coordinates of the keypoints to be normalized between 0 and 1.
    """    
    if not keypoints:
        return canvas
    
    H, W, C = canvas.shape
    for keypoint in keypoints:
        if keypoint.score < CONFIDENCE_THRESHOLD or (keypoint.x == 0.0 and keypoint.y == 0.0):
            continue  # Skip invalid keypoints
        x, y = keypoint.x, keypoint.y
        x = int(x * W)
        y = int(y * H)
        if x > eps and y > eps:
            cv2.circle(canvas, (x, y), 3, (255, 255, 255), thickness=-1)
    return canvas

# Parse 2D keypoints with normalization
def parse_keypoints(keypoints, keypoints_length=75, frame_width=1280, frame_height=720):
    """
    Parse and normalize keypoints from JSON data.

    Args:
        keypoints: A flat list of [x, y, score, ...] from the JSON.
        keypoints_length: The expected number of keypoints.
        frame_width: The width of the original frame.
        frame_height: The height of the original frame.

    Returns:
        numpy.ndarray: A 2D array of normalized keypoints [x, y, score].
    """
    keypoints = np.array(keypoints, dtype=np.float64).reshape(-1, 3)[:keypoints_length]
    keypoints[:, 0] /= frame_width  # Normalize x by width
    keypoints[:, 1] /= frame_height  # Normalize y by height
    return keypoints

# Create and return a pose image using OpenCV
def create_pose_image(person, frame_id, hands_and_face=False, xinsir=False, save_dir=None):
    # Parse and draw body keypoints
    pose_keypoints = filter_keypoints(parse_keypoints(
        person['pose_keypoints_2d'], keypoints_length=25, frame_width=1280, frame_height=720
    ))
    body_keypoint_objects = [
        Keypoint(x=x, y=y, score=c, id=idx) for idx, (x, y, c) in enumerate(pose_keypoints)
    ]

    # Optionally parse and draw hand and face keypoints
    left_hand_keypoint_objects, right_hand_keypoint_objects, face_keypoint_objects = [], [], []
    if hands_and_face:
        # Parse and draw left hand keypoints
        left_hand_keypoints = filter_keypoints(parse_keypoints(
            person['hand_left_keypoints_2d'], keypoints_length=21, frame_width=1280, frame_height=720
        ))
        left_hand_keypoint_objects = [
            Keypoint(x=x, y=y, score=c, id=idx) for idx, (x, y, c) in enumerate(left_hand_keypoints)
        ]

        # Parse and draw right hand keypoints
        right_hand_keypoints = filter_keypoints(parse_keypoints(
            person['hand_right_keypoints_2d'], keypoints_length=21, frame_width=1280, frame_height=720
        ))
        right_hand_keypoint_objects = [
            Keypoint(x=x, y=y, score=c, id=idx) for idx, (x, y, c) in enumerate(right_hand_keypoints)
        ]

        # Parse and draw face keypoints
        face_keypoints = filter_keypoints(parse_keypoints(
            person['face_keypoints_2d'], keypoints_length=70, frame_width=1280, frame_height=720
        ))
        face_keypoint_objects = [
            Keypoint(x=x, y=y, score=c, id=idx) for idx, (x, y, c) in enumerate(face_keypoints)
        ]

    # Draw the pose
    pose_img = draw_pose(
        body_keypoint_objects,
        left_hand_keypoint_objects, 
        right_hand_keypoint_objects, 
        face_keypoint_objects, 
        hands_and_face, 
        xinsir
    )

    # Save the resulting image if a save directory is provided
    if save_dir:
        image_path = f"{save_dir}/openpose-frame-{frame_id}.png"
        cv2.imwrite(image_path, pose_img)
        print(f"Saved pose image to {image_path}")

    return pose_img

def create_upper_body_pose_image(person, frame_id, hands_and_face=False, xinsir=False, padding=100, save_dir=None):
    """
    Create an image of the upper-body pose, excluding specified keypoints.

    Args:
        person (dict): The person data containing keypoints.
        frame_id (int): The frame ID, useful for saving the file.
        hands_and_face (bool): Whether to include hands and face in the image.
        save_dir (str | None): Directory to save the resulting image. If None, the image is not saved.

    Returns:
        np.ndarray: The resulting upper-body pose image.
    """
    # Keypoint indexes to exclude for upper body
    excluded_indexes = {9, 10, 11, 12, 13, 14, 19, 20, 21, 22, 23, 24}
    
    # Parse and filter body keypoints
    pose_keypoints = filter_keypoints(parse_keypoints(
        person['pose_keypoints_2d'], keypoints_length=25, frame_width=1280, frame_height=720
    ))
    body_keypoint_objects = [
        Keypoint(x=x, y=y, score=c, id=idx) if idx not in excluded_indexes else None
        for idx, (x, y, c) in enumerate(pose_keypoints)
    ]

    # Optionally parse and draw hand and face keypoints
    left_hand_keypoint_objects, right_hand_keypoint_objects, face_keypoint_objects = [], [], []
    if hands_and_face:
        left_hand_keypoints = filter_keypoints(parse_keypoints(
            person['hand_left_keypoints_2d'], keypoints_length=21, frame_width=1280, frame_height=720
        ))
        left_hand_keypoint_objects = [
            Keypoint(x=x, y=y, score=c, id=idx) for idx, (x, y, c) in enumerate(left_hand_keypoints)
        ]

        right_hand_keypoints = filter_keypoints(parse_keypoints(
            person['hand_right_keypoints_2d'], keypoints_length=21, frame_width=1280, frame_height=720
        ))
        right_hand_keypoint_objects = [
            Keypoint(x=x, y=y, score=c, id=idx) for idx, (x, y, c) in enumerate(right_hand_keypoints)
        ]

        face_keypoints = filter_keypoints(parse_keypoints(
            person['face_keypoints_2d'], keypoints_length=70, frame_width=1280, frame_height=720
        ))
        face_keypoint_objects = [
            Keypoint(x=x, y=y, score=c, id=idx) for idx, (x, y, c) in enumerate(face_keypoints)
        ]

    # Draw the upper-body pose
    pose_img = draw_pose(
        body_keypoint_objects, 
        left_hand_keypoint_objects, 
        right_hand_keypoint_objects, 
        face_keypoint_objects, 
        hands_and_face,
        xinsir
    )

    keypoint_objects_for_cropping = body_keypoint_objects + left_hand_keypoint_objects + right_hand_keypoint_objects + face_keypoint_objects if hands_and_face else body_keypoint_objects
    
    min_x, min_y, max_x, max_y = calculate_cropping_bbox(pose_img, keypoint_objects_for_cropping, padding)

    # Save the resulting image if a save directory is provided
    if save_dir:
        if xinsir:
            image_path = f"{save_dir}/upperbody-pose-xinsir-frame-{frame_id}.png"
        else:
            image_path = f"{save_dir}/upperbody-pose-lllyasviel-frame-{frame_id}.png"
        cv2.imwrite(image_path, pose_img)
        print(f"Saved upper-body pose image to {image_path}")

    return pose_img, (min_x, min_y, max_x, max_y)

def calculate_cropping_bbox(canvas: np.ndarray, keypoints: List[Keypoint], padding: int = 100) -> Tuple[int, int, int, int]:
    """
    Calculate the bounding box for the pose skeleton.

    Args:
        canvas (np.ndarray): The original pose image.
        keypoints (List[Keypoint]): The list of keypoints used to draw the pose.
        padding (int): Padding around the bounding box of the pose skeleton.

    Returns:
        Tuple[int, int, int, int]: The bounding box (min_x, min_y, max_x, max_y).
    """
    H, W, C = canvas.shape

    # Initialize bounding box limits
    min_x, min_y = W, H
    max_x, max_y = 0, 0

    # Loop through keypoints to find the bounding box
    for keypoint in keypoints:
        if keypoint is None or keypoint.score < CONFIDENCE_THRESHOLD:
            continue
        x, y = int(keypoint.x * W), int(keypoint.y * H)
        if x > 0 and y > 0:  # Ignore dummy or invalid keypoints
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)

    # Add padding to the bounding box
    min_x = max(0, min_x - padding)
    min_y = max(0, min_y - padding)
    max_x = min(W, max_x + padding)
    max_y = min(H, max_y + 20)

    return min_x, min_y, max_x, max_y