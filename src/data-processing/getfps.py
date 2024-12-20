import cv2

def get_video_fps(video_path):
    """
    Get the frames per second (FPS) of a video file.

    Parameters:
        video_path (str): Path to the video file.

    Returns:
        float: The FPS of the video.
    """
    try:
        # Open the video file
        video = cv2.VideoCapture(video_path)

        # Check if the video file is opened successfully
        if not video.isOpened():
            raise FileNotFoundError(f"Unable to open video file: {video_path}")

        # Retrieve the FPS of the video
        fps = video.get(cv2.CAP_PROP_FPS)

        # Release the video capture object
        video.release()

        return fps

    except Exception as e:
        print(f"Error: {e}")
        return None

# Example usage
if __name__ == "__main__":
    video_path = "../resources/input/video-a1.mp4"
    fps = get_video_fps(video_path)

    if fps:
        print(f"The video has {fps:.2f} frames per second.")
    else:
        print("Could not determine the FPS of the video.")
