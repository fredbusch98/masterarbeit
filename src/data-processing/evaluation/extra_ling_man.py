import os
import unicodedata
import pysrt
from moviepy.editor import VideoFileClip
from tqdm import tqdm

# Define base paths
base_path = "/Volumes/IISY/DGSKorpus/"
output_dir = os.path.join(base_path, "extra-ling-man")

# Ensure the output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created directory: {output_dir}")

# The string to search for in SRT blocks
search_string = "$$EXTRA-LING-MAN"

# Iterate through all subdirectories in base_path
entry_folders = [os.path.join(base_path, d) for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
print(f"Found {len(entry_folders)} entry folders to process.")

for entry_folder in tqdm(entry_folders, desc="Processing entry folders"):
    for speaker in ['a', 'b']:
        # Define paths for the SRT and video files
        srt_file = os.path.join(entry_folder, f"speaker-{speaker}.srt")
        video_file = os.path.join(entry_folder, f"video-{speaker}.mp4")
        
        # Check if both files exist
        if not os.path.exists(srt_file):
            # Skip if transcript not found
            continue
        if not os.path.exists(video_file):
            # Skip if video file not found
            continue

        # Open the SRT file and iterate over each subtitle block
        try:
            subs = pysrt.open(srt_file)
        except Exception as e:
            print(f"Error reading {srt_file}: {e}")
            continue

        # Load the video clip once for all snippets in this file
        try:
            clip = VideoFileClip(video_file)
        except Exception as e:
            print(f"Error loading video {video_file}: {e}")
            continue

        for sub in subs:
            if search_string in sub.text:
                # Convert subtitle timestamps to seconds
                start_time = sub.start.hours * 3600 + sub.start.minutes * 60 + sub.start.seconds + sub.start.milliseconds / 1000
                end_time = sub.end.hours * 3600 + sub.end.minutes * 60 + sub.end.seconds + sub.end.milliseconds / 1000
                
                # Create a video snippet from the clip
                try:
                    snippet = clip.subclip(start_time, end_time)
                except Exception as e:
                    print(f"Error creating subclip from {video_file} for subtitle index {sub.index}: {e}")
                    continue

                # Prepare a safe filename using the entry folder name, speaker, and subtitle index
                entry_name = os.path.basename(entry_folder)
                safe_text = unicodedata.normalize('NFC', sub.text)
                safe_text = safe_text.replace("$$", "").replace(" ", "_")
                output_filename = f"{entry_name}-speaker-{speaker}-sub{sub.index}-{safe_text}.mp4"
                output_path = os.path.join(output_dir, output_filename)

                # Write the video snippet
                try:
                    snippet.write_videofile(output_path, codec='libx264', audio_codec='aac', verbose=False, logger=None)
                    print(f"Saved snippet: {output_path}")
                except Exception as e:
                    print(f"Error writing snippet to {output_path}: {e}")
                finally:
                    snippet.close()

        clip.close()

print("All processing completed!")
