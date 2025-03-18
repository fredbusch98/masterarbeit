import csv
import os
import unicodedata
import pysrt
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip, concatenate_videoclips
from tqdm import tqdm

# Define base paths and threshold
threshold = 2000
base_path = "/Volumes/IISY/DGSKorpus/"
csv_file = os.path.join(base_path, f"dgs-gloss-times/gd_above_{threshold}ms.csv")
output_dir = os.path.join(base_path, "dgs-gloss-times/video-snippets")
temp_dir = os.path.join(output_dir, "temp_combined")

# Ensure the output and temporary directories exist
for d in [output_dir, temp_dir]:
    if not os.path.exists(d):
        os.makedirs(d)
        print(f"📁 Created directory: {d}")

# Read the CSV file
print("📖 Reading CSV file...")
with open(csv_file, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    rows = list(reader)
print(f"✅ Found {len(rows)} rows to process")

# List to store paths of temporary combined clips
combined_temp_files = []

# Process each row with a progress bar
for row in tqdm(rows, total=len(rows), desc="🔄 Processing video snippets"):
    try:
        # Extract data from the row
        gloss = row['gloss']
        entry = row['entry']
        block_index = int(row['block_index'])
        speaker = row['speaker'].strip().upper()
        entry_folder = os.path.join(base_path, entry)

        # Normalize gloss for filename
        safe_gloss = unicodedata.normalize('NFC', gloss)
        safe_gloss = safe_gloss.replace("$", "")
        safe_gloss = safe_gloss.replace("Ü", "UE")
        safe_gloss = safe_gloss.replace("Ö", "OE")
        safe_gloss = safe_gloss.replace("Ä", "AE")
        safe_gloss = safe_gloss.replace("ß", "SS")

        # Determine SRT and video files based on speaker
        if speaker == 'A':
            srt_file = os.path.join(entry_folder, 'speaker-a.srt')
            video_file = os.path.join(entry_folder, 'video-a.mp4')
        elif speaker == 'B':
            srt_file = os.path.join(entry_folder, 'speaker-b.srt')
            video_file = os.path.join(entry_folder, 'video-b.mp4')
        else:
            print(f"⚠️ Invalid speaker '{speaker}' in row {row}")
            continue

        # Define output file path for individual snippet
        output_file = os.path.join(output_dir, f"{row['gd']}ms-{entry}-{block_index}-{safe_gloss}.mp4")

        # Open the SRT file and find the matching subtitle block
        subs = pysrt.open(srt_file)
        subtitle = next((sub for sub in subs if sub.index == block_index), None)
        if subtitle is None:
            print(f"❌ Block index {block_index} not found in {srt_file}")
            continue

        # Convert timestamps to seconds
        start_time = (subtitle.start.hours * 3600 + subtitle.start.minutes * 60 +
                      subtitle.start.seconds + subtitle.start.milliseconds / 1000)
        end_time = (subtitle.end.hours * 3600 + subtitle.end.minutes * 60 +
                    subtitle.end.seconds + subtitle.end.milliseconds / 1000)

        # Open the video file
        clip = VideoFileClip(video_file)

        # Create subclips for individual and combined outputs
        snippet_individual = clip.subclip(start_time, end_time)
        snippet_combined   = clip.subclip(start_time, end_time)

        # ----- Create and save individual snippet (gloss as subtitle) -----
        text_individual = TextClip(gloss, fontsize=50, color='white').set_duration(snippet_individual.duration)
        text_individual = text_individual.set_position(('center', 'bottom'))
        composite_individual = CompositeVideoClip([snippet_individual, text_individual])
        if not os.path.exists(output_file):
            composite_individual.write_videofile(output_file, codec='libx264', audio_codec='aac')
            print(f"🎥 Saved {output_file}")
        else:
            print(f"ℹ️ {output_file} already exists, skipping individual snippet save")
        # Clean up individual snippet resources
        snippet_individual.close()
        text_individual.close()
        composite_individual.close()

        # ----- Create composite clip for the combined video (with new subtitle format) -----
        subtitle_text = f"{entry}, {block_index}: {gloss}"
        text_combined = TextClip(subtitle_text, fontsize=20, color='white').set_duration(snippet_combined.duration)
        text_combined = text_combined.set_position(('center', 'bottom'))
        composite_combined = CompositeVideoClip([snippet_combined, text_combined])

        # Write the composite combined clip to a temporary file to “bake in” the frames
        temp_file = os.path.join(temp_dir, f"temp_{row['gd']}ms-{entry}-{block_index}-{safe_gloss}.mp4")
        composite_combined.write_videofile(temp_file, codec='libx264', audio_codec='aac')
        combined_temp_files.append(temp_file)

        # Clean up combined snippet resources
        snippet_combined.close()
        text_combined.close()
        composite_combined.close()
        clip.close()

    except Exception as e:
        print(f"❌ Error processing row {row}: {e}")
        continue

print("🏁 All processing completed for individual snippets!")

# ----- Combine all temporary combined snippets into one video -----
if combined_temp_files:
    print("📽 Combining all video snippets into one video...")
    # Load the temporary files as independent VideoFileClip objects
    final_clips = []
    for file in combined_temp_files:
        try:
            final_clips.append(VideoFileClip(file))
        except Exception as e:
            print(f"❌ Error loading temporary file {file}: {e}")
    if final_clips:
        final_clip = concatenate_videoclips(final_clips, method="compose")
        combined_output_file = os.path.join(output_dir, "combined_video.mp4")
        final_clip.write_videofile(combined_output_file, codec='libx264', audio_codec='aac')
        final_clip.close()
        print(f"✅ Combined video saved as {combined_output_file}")
        # Close all final clip objects
        for clip in final_clips:
            clip.close()
    else:
        print("⚠️ No valid clips to combine.")
    
    # Optionally, remove the temporary files after successful concatenation
    for temp_file in combined_temp_files:
        try:
            os.remove(temp_file)
        except Exception as e:
            print(f"⚠️ Could not delete temporary file {temp_file}: {e}")
else:
    print("⚠️ No clips available for combining.")
