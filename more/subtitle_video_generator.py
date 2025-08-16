"""
Burns subtitles from a .srt file directly into a video using FFmpeg. 
Takes an input video and subtitle file, producing an output video with embedded subtitles.
"""
import subprocess
import os
import sys
import argparse

def burn_subtitles(input_video, srt_file, output_video):
    # Check if the input files exist
    if not os.path.isfile(input_video):
        print(f"Error: The video file '{input_video}' does not exist.")
        sys.exit(1)
    if not os.path.isfile(srt_file):
        print(f"Error: The subtitle file '{srt_file}' does not exist.")
        sys.exit(1)
    
    # Build the FFmpeg command
    command = [
        "ffmpeg",
        "-i", input_video,
        "-vf", f"subtitles={srt_file}",
        "-c:a", "copy",
        output_video
    ]

    try:
        # Run the command
        subprocess.run(command, check=True)
        print(f"Success: The output video '{output_video}' has been created with integrated subtitles.")
    except subprocess.CalledProcessError as e:
        print("An error occurred while processing the video.")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Burn subtitles into a video using FFmpeg.")
    parser.add_argument("input_video", help="Path to the input video file (e.g., video-a.mp4)")
    parser.add_argument("srt_file", help="Path to the subtitle file (e.g., transcript.srt)")
    parser.add_argument("output_video", help="Path for the output video file (e.g., output.mp4)")
    
    args = parser.parse_args()
    burn_subtitles(args.input_video, args.srt_file, args.output_video)

if __name__ == "__main__":
    main()
