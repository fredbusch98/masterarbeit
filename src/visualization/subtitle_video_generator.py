from moviepy.video.tools.subtitles import SubtitlesClip
import moviepy.editor as mp
from pysrt import SubRipFile
import os
from textwrap import wrap

def srt_to_moviepy_format(srt_file):
    """Convert SRT file to MoviePy format with proper handling of overlapping subtitles."""
    subs = SubRipFile.open(srt_file)
    subtitle_list = []
    
    for sub in subs:
        # Clean up any unnecessary symbols or unwanted characters in subtitles
        clean_text = sub.text.replace('\n', ' ').strip()

        # Here we want to break down the subtitles into segments if they have long durations
        # Split long duration captions into smaller segments
        subtitle_list.append(((sub.start.ordinal / 1000, sub.end.ordinal / 1000), clean_text))
    
    # Handle overlapping captions: Merge any that have the same or very close timestamps
    merged_subtitles = []
    last_subtitle = None
    for current_subtitle in subtitle_list:
        if last_subtitle and current_subtitle[0][0] <= last_subtitle[0][1]:  # Overlap detected
            # Merge the current subtitle with the last one by extending the text
            last_subtitle = ((last_subtitle[0][0], current_subtitle[0][1]), last_subtitle[1] + " " + current_subtitle[1])
        else:
            if last_subtitle:
                merged_subtitles.append(last_subtitle)
            last_subtitle = current_subtitle

    if last_subtitle:
        merged_subtitles.append(last_subtitle)

    return merged_subtitles

def wrap_text_to_fit(text, max_width, font_path, font_size):
    """Wrap text to fit within the screen width."""
    from PIL import ImageFont

    font = ImageFont.truetype(font_path, font_size)
    lines = []
    for line in text.splitlines():
        wrapped_lines = wrap(line, width=max_width)
        for wrapped_line in wrapped_lines:
            # Replace getsize with getbbox (to calculate width correctly)
            if font.getbbox(wrapped_line)[2] > max_width:
                words = wrapped_line.split()
                temp_line = ""
                for word in words:
                    # Use getbbox here to measure width
                    if font.getbbox(temp_line + word + " ")[2] <= max_width:
                        temp_line += word + " "
                    else:
                        lines.append(temp_line.strip())
                        temp_line = word + " "
                lines.append(temp_line.strip())
            else:
                lines.append(wrapped_line)
    return "\n".join(lines)

def create_video_with_subtitles(video_path, srt_path, output_path):
    """Create a new video with subtitles burned in."""
    # Load the video
    video = mp.VideoFileClip(video_path)

    # Convert SRT to MoviePy format
    subtitles = srt_to_moviepy_format(srt_path)

    # Absolute path to the font
    font_path = os.path.abspath("../resources/fonts/arial/ARIAL.TTF")
    if not os.path.exists(font_path):
        raise FileNotFoundError(f"Font file not found at: {font_path}")

    # Define max width for text wrapping (in pixels)
    max_width = int(video.w * 0.8)  # 80% of the video width

    # Define font size
    fontsize = 18

    # Generate subtitles clip
    def generator(txt):
        wrapped_txt = wrap_text_to_fit(txt, max_width, font_path, fontsize)
        return mp.TextClip(
            wrapped_txt, font=font_path, fontsize=fontsize, color='white', align='center'
        )

    subtitles_clip = SubtitlesClip(subtitles, generator)

    # Overlay subtitles on the video
    result = mp.CompositeVideoClip([video, subtitles_clip.set_position(('center', 'bottom'))])

    # Write the output video file
    result.write_videofile(output_path, codec="libx264", audio_codec="aac")

# Specify file paths
video_file = "../resources/input/video-a1.mp4"
subtitle_file = "../resources/input/srt1.srt"
output_file = "../resources/output/videos/output_video_with_subtitles.mp4"

# Run the function
create_video_with_subtitles(video_file, subtitle_file, output_file)
