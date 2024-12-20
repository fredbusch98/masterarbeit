import json
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, EulerAncestralDiscreteScheduler, StableDiffusionXLControlNetPipeline, AutoencoderKL, UniPCMultistepScheduler
from PIL import Image
import numpy as np
import cv2
from pose_utils import create_upper_body_pose_image

model = "xinsir"
# model = "lllyasviel"

# Hyperparameters
controlnet_conditioning_scale = 1.0
prompt = "A realistic, high-quality, clear and professional upper-body image of a person in the same pose"
negative_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, blurry, blur'
eulera_scheduler = EulerAncestralDiscreteScheduler.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="scheduler")

# File paths
json_file_path = "../resources/input/test-input-3.json"
reference_image_path = "../resources/input/input.jpg"
output_frames_dir = "../resources/output/frames/"
output_video_path = "../resources/output/videos/output_video.mp4"

# XINSIR ControlNet: https://huggingface.co/xinsir/controlnet-openpose-sdxl-1.0
# Load Stable Diffusion and ControlNet models
controlnet1 = ControlNetModel.from_pretrained(
    "xinsir/controlnet-openpose-sdxl-1.0",
    torch_dtype=torch.float32
)
# when test with other base model, you need to change the vae also.
vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix", 
    torch_dtype=torch.float32  # Load directly in float32
)

pipe1 = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet1,
    vae=vae,
    torch_dtype=torch.float32,  # Ensure components load in float32
    scheduler=eulera_scheduler,
)

# Ensure all components are moved to CPU and converted to float32
pipe1.to("cpu")

# Convert specific components manually to float32
for name, module in pipe1.components.items():
    if hasattr(module, "to"):
        module.to(dtype=torch.float32)


# lllyasviel ControlNet: https://huggingface.co/lllyasviel/sd-controlnet-openpose
controlnet2 = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-openpose"
)

pipe2 = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet2, safety_checker=None
)

pipe2.scheduler = UniPCMultistepScheduler.from_config(pipe2.scheduler.config)
pipe2.to("cpu")

def load_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def generate_frame(reference_image, pose_image):
    """Generate a frame based on the reference image and full body pose keypoints."""

    # need to resize the image resolution to 1024 * 1024 or same bucket resolution to get the best performance
    height, width, _  = pose_image.shape
    ratio = np.sqrt(1024. * 1024. / (width * height))
    new_width, new_height = int(width * ratio), int(height * ratio)
    pose_image = cv2.resize(pose_image, (new_width, new_height))
    pose_image = Image.fromarray(pose_image)

    # Generate frame using ControlNet
    print(f"Started generating frame using {model} ControlNet...")
    if model == "xinsir":
        result = pipe1(
            prompt,
            negative_prompt=negative_prompt,
            image=pose_image,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            width=new_width,
            height=new_height,
            num_inference_steps=30,
        ).images[0]
    else:
        result = pipe2(
            prompt,
            pose_image,
            num_inference_steps=20
        ).images[0]
    print(f"Finished generating frame using {model} ControlNet...")
    return result

# Load your reference image
reference_image = Image.open(reference_image_path)

data = load_json(json_file_path)

# Initialize a list to store bounding boxes
bounding_boxes = []

# Initialize a list to store pose images
pose_images = []

# Step 1: Collect pose images and bounding boxes
for frame_id, frame_data in data.items():
    for person in frame_data["people"]:
        print("Generating frame {}...".format(frame_id))
        if model == "xinsir":
            pose_image, bbox = create_upper_body_pose_image(person, frame_id, xinsir=True)
        else:
            pose_image, bbox = create_upper_body_pose_image(person, frame_id, xinsir=False)

        bounding_boxes.append(bbox)
        pose_images.append(pose_image)

# Step 2: Calculate the largest bounding box
min_x = min(bbox[0] for bbox in bounding_boxes)
min_y = min(bbox[1] for bbox in bounding_boxes)
max_x = max(bbox[2] for bbox in bounding_boxes)
max_y = max(bbox[3] for bbox in bounding_boxes)
largest_bbox = (min_x, min_y, max_x, max_y)

# Step 3: Crop all pose images using the largest bounding box and generate frames
frames = []
for frame_id, pose_img in enumerate(pose_images):
    # Crop the pose_img using the largest bounding box
    cropped_pose_img = pose_img[largest_bbox[1]:largest_bbox[3], largest_bbox[0]:largest_bbox[2]]

    # Generate the frame
    if model == "xinsir":
        frame = generate_frame(reference_image, cropped_pose_img)
    else:
        frame = generate_frame(reference_image, pose_img)
        frame_np = np.array(frame)
        cropped_frame_np = frame_np[largest_bbox[1]:largest_bbox[3], largest_bbox[0]:largest_bbox[2]]
        cropped_frame = Image.fromarray(cropped_frame_np)
        cropped_frame.save(f"{output_frames_dir}cropped_frame_{model}_{frame_id}.png")

    # Save the frame and cropped pose images
    frames.append(frame)
    frame.save(f"{output_frames_dir}frame_{model}_{frame_id}.png")
    cv2.imwrite(f"{output_frames_dir}cropped_pose_{model}_{frame_id}.png", cropped_pose_img)

# Step 4: Combine cropped frames into a video
if frames:
    frame_height, frame_width = frames[0].size
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

    for frame_id, cropped_frame in enumerate(frames):
        print("Combining frame {}...".format(frame_id))
        frame_cv = cv2.cvtColor(np.array(cropped_frame), cv2.COLOR_RGB2BGR)
        out.write(frame_cv)

    out.release()

print("Video generation complete.")
