import numpy as np
import cv2
import math
from typing import List, Union, NamedTuple, Tuple

#### IMPORTANT NOTE: Script is highly inspired and partly copied from the following sources:
# https://github.com/huggingface/controlnet_aux

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

def draw_pose(body_keypoints, left_hand_keypoints, right_hand_keypoints, face_keypoints, H=720, W=1280):
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)
    canvas = draw_bodypose(canvas, body_keypoints)
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

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
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

    edges = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10],
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
            cv2.line(canvas, (x1, y1), (x2, y2), (matplotlib.colors.hsv_to_rgb([ie / float(len(edges)), 1.0, 1.0]) * 255).astype(int).tolist(), thickness=2)

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
def parse_keypoints(keypoints, keypoints_length=75):
    """
    Parse and return keypoints from JSON data.

    Args:
        keypoints: A flat list of [x, y, score, ...] from the JSON.
        keypoints_length: The expected number of keypoints.

    Returns:
        numpy.ndarray: A 2D array of keypoints [x, y, score].
    """
    keypoints = np.array(keypoints, dtype=np.float64).reshape(-1, 3)[:keypoints_length]
    return keypoints

def normalize_all_keypoints(body: List[Keypoint],
                            left: List[Keypoint],
                            right: List[Keypoint],
                            face: List[Keypoint]) -> Tuple[List[Keypoint], List[Keypoint], List[Keypoint], List[Keypoint]]:
    """
    Normalizes all keypoints so that the body keypoint with id==1 is centered at (0.5, 0.4) (i.e., center of a 1280x720 image).
    The function computes an offset based on keypoint id 1 and applies it to all keypoints.

    Returns:
        Tuple of updated keypoint lists (body, left, right, face).
    """
    # Find the reference keypoint from the body keypoints with id 1.
    ref_kp = next((kp for kp in body if kp.id == 1 and kp.score >= CONFIDENCE_THRESHOLD), None)
    if ref_kp is None:
        # If not found, return keypoints unchanged
        return body, left, right, face

    # Compute offset in normalized coordinates
    offset_x = 0.5 - ref_kp.x
    offset_y = 0.4 - ref_kp.y

    def apply_offset(keypoints: List[Keypoint]) -> List[Keypoint]:
        return [Keypoint(x=kp.x + offset_x, y=kp.y + offset_y, score=kp.score, id=kp.id) for kp in keypoints]

    return apply_offset(body), apply_offset(left), apply_offset(right), apply_offset(face)

# Create and return a pose image using OpenCV
def create_pose_image(person):
    # Parse and draw body keypoints
    pose_keypoints = filter_keypoints(parse_keypoints(
        person['pose_keypoints_2d'], keypoints_length=25
    ))
    body_keypoint_objects = [
        Keypoint(x=x, y=y, score=c, id=idx) for idx, (x, y, c) in enumerate(pose_keypoints)
    ]

    # Optionally parse and draw hand and face keypoints
    left_hand_keypoints = filter_keypoints(parse_keypoints(
        person['hand_left_keypoints_2d'], keypoints_length=21
    ))
    left_hand_keypoint_objects = [
        Keypoint(x=x, y=y, score=c, id=idx) for idx, (x, y, c) in enumerate(left_hand_keypoints)
    ]

    right_hand_keypoints = filter_keypoints(parse_keypoints(
        person['hand_right_keypoints_2d'], keypoints_length=21
    ))
    right_hand_keypoint_objects = [
        Keypoint(x=x, y=y, score=c, id=idx) for idx, (x, y, c) in enumerate(right_hand_keypoints)
    ]

    face_keypoints = filter_keypoints(parse_keypoints(
        person['face_keypoints_2d'], keypoints_length=70
    ))
    face_keypoint_objects = [
        Keypoint(x=x, y=y, score=c, id=idx) for idx, (x, y, c) in enumerate(face_keypoints)
    ]

    # Normalize all keypoints so that the body keypoint with id==1 is centered.
    body_keypoint_objects, left_hand_keypoint_objects, right_hand_keypoint_objects, face_keypoint_objects = \
        normalize_all_keypoints(body_keypoint_objects,
                                left_hand_keypoint_objects,
                                right_hand_keypoint_objects,
                                face_keypoint_objects)

    # Draw the pose
    pose_img = draw_pose(
        body_keypoint_objects,
        left_hand_keypoint_objects, 
        right_hand_keypoint_objects, 
        face_keypoint_objects
    )

    return pose_img

def create_upper_body_pose_image(pose2d, face2d, left2d, right2d):
    """
    Create an image of the upper-body pose using the provided keypoints.
    Expects pose2d to have no lower-body keypoints. (Should be done in preprocseeing - gloss2pose_mapper)
    Reconstructs pose2d from reduced upper-body keypoints to full 25-keypoint set.

    Args:
        pose2d: List of 2D body keypoints (39 values for upper-body only).
        face2d: List of 2D face keypoints (70 keypoints).
        left2d: List of 2D left-hand keypoints (21 keypoints).
        right2d: List of 2D right-hand keypoints (21 keypoints).

    Returns:
        np.ndarray: The resulting upper-body pose image.
    """
    # Reconstruct full pose2d with 75 values
    included = [0, 1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 17, 18]  # Upper-body keypoint indexes
    full_pose2d = [0.0] * 75  # Initialize with zeros for all 25 keypoints
    for idx, i in enumerate(included):
        start = 3 * i
        full_pose2d[start:start + 3] = pose2d[3 * idx:3 * idx + 3]

    pose_keypoints = filter_keypoints(parse_keypoints(full_pose2d, keypoints_length=25))
    body_keypoint_objects = [
        Keypoint(x=x, y=y, score=c, id=idx) for idx, (x, y, c) in enumerate(pose_keypoints)
    ]

    left_hand_keypoints = filter_keypoints(parse_keypoints(left2d, keypoints_length=21))
    left_hand_keypoint_objects = [
        Keypoint(x=x, y=y, score=c, id=idx) for idx, (x, y, c) in enumerate(left_hand_keypoints)
    ]

    right_hand_keypoints = filter_keypoints(parse_keypoints(right2d, keypoints_length=21))
    right_hand_keypoint_objects = [
        Keypoint(x=x, y=y, score=c, id=idx) for idx, (x, y, c) in enumerate(right_hand_keypoints)
    ]

    face_keypoints = filter_keypoints(parse_keypoints(face2d, keypoints_length=70))
    face_keypoint_objects = [
        Keypoint(x=x, y=y, score=c, id=idx) for idx, (x, y, c) in enumerate(face_keypoints)
    ]

    # Normalize all keypoints so that the body keypoint with id==1 is centered.
    body_keypoint_objects, left_hand_keypoint_objects, right_hand_keypoint_objects, face_keypoint_objects = \
        normalize_all_keypoints(body_keypoint_objects,
                                left_hand_keypoint_objects,
                                right_hand_keypoint_objects,
                                face_keypoint_objects)

    pose_img = draw_pose(
        body_keypoint_objects,
        left_hand_keypoint_objects,
        right_hand_keypoint_objects,
        face_keypoint_objects
    )

    return pose_img
