import os
import argparse
import logging
import math
import time 
from omegaconf import OmegaConf
from datetime import datetime
from pathlib import Path
from datetime import datetime

import numpy as np
import cv2
import torch.jit
from torchvision.datasets.folder import pil_loader
from torchvision.transforms.functional import pil_to_tensor, resize, center_crop
from torchvision.transforms.functional import to_pil_image

from mimicmotion.utils.geglu_patch import patch_geglu_inplace
patch_geglu_inplace()

from constants import ASPECT_RATIO

from mimicmotion.pipelines.pipeline_mimicmotion import MimicMotionPipeline
from mimicmotion.utils.loader import create_pipeline
from mimicmotion.utils.utils import save_to_mp4
from mimicmotion.dwpose.preprocess import get_video_pose, get_image_pose

import decord  # used for loading preprocessed pose video

logging.basicConfig(level=logging.INFO, format="%(asctime)s: [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_preprocessed_video_pose(video_path: str, target_size: tuple):
    vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
    frames = vr.get_batch(range(len(vr))).asnumpy()  # Get all frames

    processed_frames = []
    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized_frame = cv2.resize(frame, (target_size[1], target_size[0]))

        h, w, _ = resized_frame.shape
        if (h, w) != target_size:
            start_y = max((h - target_size[0]) // 2, 0)
            start_x = max((w - target_size[1]) // 2, 0)
            frame_cropped = resized_frame[start_y:start_y+target_size[0], start_x:start_x+target_size[1]]
        else:
            frame_cropped = resized_frame

        processed_frames.append(frame_cropped)

    video_pose_array = np.stack(processed_frames)
    video_pose_array = np.transpose(video_pose_array, (0, 3, 1, 2))  # Convert to (N, C, H, W)

    return video_pose_array.astype(np.uint8)

def save_video_pose(video_pose, output_path, fps=30):
    """
    Save a numpy array of video frames as an MP4 file in RGB format.

    Args:
        video_pose (np.ndarray): Array of video frames. It can have shape either:
            - (N, C, H, W)  OR
            - (N, H, W, C)
          The pixel values are assumed to be normalized in the range [-1, 1].
        output_path (str): Output file path (e.g. '/outputs/dw-pose.mp4').
        fps (int, optional): Frames per second of the output video.
    """
    # If frames are in (N, C, H, W) format, transpose them to (N, H, W, C)
    if video_pose.ndim == 4 and video_pose.shape[1] == 3:
        video_pose = video_pose.transpose(0, 2, 3, 1)

    num_frames, H, W, C = video_pose.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    for i in range(num_frames):
        frame = video_pose[i]
        # Rescale pixel values from [-1, 1] to [0, 255]
        frame = ((frame + 1) * 127.5).clip(0, 255).astype(np.uint8)
        # Write the frame as is (without BGR conversion)
        out.write(frame)

    out.release()
    print(f"Saved video to {output_path}")

def preprocess(video_path, image_path, resolution=576, sample_stride=2, use_preprocessed_video_pose=False):
    """
    Preprocess the reference image and video pose.
    
    If `use_preprocessed_video_pose` is True, the video is assumed to be an MP4 of pose frames.
    Otherwise, the original pose extraction is used.
    """
    print("Preprocessing video and reference image...")
    # --- Process the reference image ---
    # Load the reference image and convert to tensor
    image_pixels = pil_loader(image_path)
    image_pixels = pil_to_tensor(image_pixels)  # (C, H, W)
    h, w = image_pixels.shape[-2:]
    # Compute target height/width based on the aspect ratio
    if h > w:
        w_target, h_target = resolution, int(resolution / ASPECT_RATIO // 64) * 64
    else:
        w_target, h_target = int(resolution / ASPECT_RATIO // 64) * 64, resolution
    h_w_ratio = float(h) / float(w)
    if h_w_ratio < h_target / w_target:
        h_resize, w_resize = h_target, math.ceil(h_target / h_w_ratio)
    else:
        h_resize, w_resize = math.ceil(w_target * h_w_ratio), w_target
    image_pixels = resize(image_pixels, [h_resize, w_resize], antialias=None)
    image_pixels = center_crop(image_pixels, [h_target, w_target])
    # Convert to numpy array for further processing
    image_pixels = image_pixels.permute((1, 2, 0)).numpy()

    # --- Get the image pose ---
    # This always uses get_image_pose to extract the pose from the reference image.
    image_pose = get_image_pose(image_pixels)

    # --- Process the video pose ---
    if use_preprocessed_video_pose:
        logger.info("Using preprocessed video pose branch")
        logger.info(f"Video path: {video_path}, target_size: {(h_target, w_target)}")
        video_pose = load_preprocessed_video_pose(video_path, target_size=(h_target, w_target))
        logger.info(f"Loaded preprocessed video pose shape: {video_pose.shape}, dtype: {video_pose.dtype}, value range: min={video_pose.min()}, max={video_pose.max()}")
    else:
        logger.info("Using computed video pose branch (not preprocessed)")
        logger.info(f"Video path: {video_path}, sample_stride: {sample_stride}")
        video_pose = get_video_pose(video_path, image_pixels, sample_stride=sample_stride)
        logger.info(f"Computed video pose shape: {video_pose.shape}, dtype: {video_pose.dtype}, value range: min={video_pose.min()}, max={video_pose.max()}")
        current_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(current_dir, f"outputs/extracted-pose_{datetime.now().strftime('%Y%m%d%H%M%S')}.mp4")
        save_video_pose(video_pose, output_path, fps=15)

    # Concatenate the image pose (as the first frame) with the video pose
    pose_pixels = np.concatenate([np.expand_dims(image_pose, 0), video_pose])
    # Convert the image pixels to the proper shape: (1, C, H, W)
    image_pixels_tensor = np.transpose(np.expand_dims(image_pixels, 0), (0, 3, 1, 2))
    
    pose_tensor = torch.from_numpy(pose_pixels.copy()) / 127.5 - 1
    image_tensor = torch.from_numpy(image_pixels_tensor) / 127.5 - 1
    if not args.no_use_float16:
        pose_tensor = pose_tensor.half()
        image_tensor = image_tensor.half()
    logger.info(f"Final pose tensor shape: {pose_tensor.shape}, image tensor shape: {image_tensor.shape}")
    return pose_tensor, image_tensor

def run_pipeline(pipeline: MimicMotionPipeline, image_pixels, pose_pixels, device, task_config):
    # Convert image tensor to a list of PIL images (for the pipeline)
    image_pixels_list = [to_pil_image(img.to(torch.uint8)) for img in (image_pixels + 1.0) * 127.5]
    generator = torch.Generator(device=device)
    generator.manual_seed(task_config.seed)
    frames = pipeline(
        image_pixels_list, image_pose=pose_pixels, num_frames=pose_pixels.size(0),
        tile_size=task_config.num_frames, tile_overlap=task_config.frames_overlap,
        height=pose_pixels.shape[-2], width=pose_pixels.shape[-1], fps=7,
        noise_aug_strength=task_config.noise_aug_strength, num_inference_steps=task_config.num_inference_steps,
        generator=generator, min_guidance_scale=task_config.guidance_scale, 
        max_guidance_scale=task_config.guidance_scale, decode_chunk_size=8, output_type="pt", device=device
    ).frames.cpu()
    video_frames = (frames * 255.0).to(torch.uint8)

    # The original code discarded the first frame (used as reference), so do the same.
    for vid_idx in range(video_frames.shape[0]):
        _video_frames = video_frames[vid_idx, 1:]

    return _video_frames


@torch.no_grad()
def main(args):
    start_time = time.time()  # Start timer

    if not args.no_use_float16:
        torch.set_default_dtype(torch.float16)

    infer_config = OmegaConf.load(args.inference_config)
    pipeline = create_pipeline(infer_config, device)

    output_files = []  # Store filenames of processed videos

    for task in infer_config.test_case:
        # Check if the new flag is set; default to False if not provided.
        use_preprocessed = task.get("use_preprocessed_video_pose", False)
        
        # Pre-process the data, including both the reference image and the video pose sequence.
        pose_pixels, image_pixels = preprocess(
            task.ref_video_path, task.ref_image_path, 
            resolution=task.resolution, sample_stride=task.sample_stride,
            use_preprocessed_video_pose=use_preprocessed
        )
        # Run the MimicMotion pipeline
        print("Running the MimicMotion pipeline ...")
        _video_frames = run_pipeline(
            pipeline, 
            image_pixels, pose_pixels, 
            device, task
        )
        print("Finished running pipeline")

        # Generate the output filename
        basename = os.path.basename(task.ref_video_path)
        base_ref_image_name = os.path.basename(task.ref_image_path)
        base_ref_image_name = base_ref_image_name.replace(".png", "")
        new_name = basename.replace("_pose.mp4", f"_gen_{base_ref_image_name}.mp4")
        output_filename = os.path.join(args.output_dir, new_name)
        
        # Save results to the output folder.
        print("Saving result video as mp4 ...")
        save_to_mp4(
            _video_frames, 
            output_filename,
            fps=task.fps,
        )

        output_files.append(output_filename)  # Store the filename

    outputfile = output_files[0].split("/")[-1]
    logger.info(f"--- Finished processing, output files: {', '.join(output_files)} ---")
    logger.info(f"kubectl cp s85468/mimicmotion:/storage/MimicMotion/outputs/{outputfile} /Users/frederikbusch/Developer/master-arbeit/mimicmotion/outputs/{outputfile}")

    # Calculate and display the elapsed time
    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    print(f"Inference took {minutes} minutes and {seconds} seconds!")


def set_logger(log_file=None, log_level=logging.INFO):
    log_handler = logging.FileHandler(log_file, "w")
    log_handler.setFormatter(
        logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s]: %(message)s")
    )
    log_handler.setLevel(log_level)
    logger.addHandler(log_handler)


if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file", type=str, default=None)
    parser.add_argument("--inference_config", type=str, default="configs/test.yaml")
    parser.add_argument("--output_dir", type=str, default="outputs/", help="path to output")
    parser.add_argument("--no_use_float16",
                        action="store_true",
                        help="Whether to use float16 to speed up inference",
    )
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    config_filename = os.path.basename(args.inference_config)
    config_filename = config_filename.replace(".yml", "")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logfile_name = f"{args.output_dir}/{config_filename}_{timestamp}.log"
    set_logger(args.log_file if args.log_file is not None else
               logfile_name)
    main(args)
    logger.info("--- Finished ---")
