"""
Extracts every 10th dual-gloss subtitle from transcript-aligned videos, 
creates short video snippets with burned-in subtitles, 
and saves them into a structured output directory. 
For manual analysis/inspection of the dual glosses.
"""
import os
import unicodedata
import pysrt
import re
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
from tqdm import tqdm

# Configuration
base_dir = "/Volumes/IISY/DGSKorpus/"
output_dir = os.path.join(base_dir, "video-snippets/dual_gloss_snippets")
transcript_name = "transcript.srt"
video_template = "video-{}.mp4"  # speaker 'a' or 'b'
excluded_glosses = ["$PROD", "$ORAL", "$ALPHA", "$$EXTRA-LING-MAN", "$GEST-OFF", "$PMS"]

# Counters
dual_gloss_counter = 0
snippet_counter = 0

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

def clean_filename(text, max_length=30):
    nfkd = unicodedata.normalize('NFKD', text)
    ascii_only = nfkd.encode('ASCII', 'ignore').decode('ASCII')
    return re.sub(r'[^\w\d_\-]', '_', ascii_only[:max_length])

# Iterate over entry folders
for entry in tqdm(sorted(os.listdir(base_dir)), desc="Entries"):
    entry_path = os.path.join(base_dir, entry)
    if not os.path.isdir(entry_path) or not entry.startswith("entry_"):
        continue

    # Transcript path
    srt_path = os.path.join(entry_path, transcript_name)
    if not os.path.exists(srt_path):
        print(f"Missing transcript for {entry}, skipping.")
        continue
    try:
        subs = pysrt.open(srt_path)
    except Exception as e:
        print(f"Error reading {srt_path}: {e}")
        continue

    # Load videos for both speakers
    clips = {}
    for sp in ['a', 'b']:
        vid_path = os.path.join(entry_path, video_template.format(sp))
        if os.path.exists(vid_path):
            try:
                clips[sp] = VideoFileClip(vid_path)
            except Exception as e:
                print(f"Error loading video {vid_path}: {e}")
                clips[sp] = None
        else:
            clips[sp] = None

    # Process each subtitle block
    for sub in subs:
        raw = unicodedata.normalize('NFC', sub.text).strip()
        m = re.match(r'^([AB]):\s*(.*)', raw)
        if not m:
            continue
        speaker = m.group(1).lower()
        text = m.group(2).strip()

        # Detect dual gloss (uppercase with '||')
        text = text.lstrip('||')
        if '||' in text and text.replace('|', '').isupper():
            
            # Check for excluded glosses
            if any(gloss in text for gloss in excluded_glosses):
                continue

            dual_gloss_counter += 1

            # Every 10th dual gloss -> extract snippet
            if dual_gloss_counter % 10 == 0:
                clip = clips.get(speaker)
                if clip is None:
                    print(f"No video for speaker-{speaker} in {entry}.")
                    continue

                # Compute timestamps
                start = (sub.start.hours * 3600 + sub.start.minutes * 60 + 
                         sub.start.seconds + sub.start.milliseconds / 1000)
                end = (sub.end.hours * 3600 + sub.end.minutes * 60 + 
                       sub.end.seconds + sub.end.milliseconds / 1000)

                try:
                    segment = clip.subclip(start, end)
                except Exception as e:
                    print(f"Error creating subclip for {entry} speaker-{speaker} gloss#{dual_gloss_counter}: {e}")
                    continue

                # Burn subtitle onto video
                txt = TextClip(raw, fontsize=24, font='Arial', method='caption',
                               size=(segment.w * 0.8, None), align='center',
                               stroke_width=1, stroke_color='black')
                txt = txt.set_position(('center', 'bottom')).set_duration(segment.duration)
                comp = CompositeVideoClip([segment, txt])

                # Prepare filename
                safe = clean_filename(text)
                snippet_counter += 1
                fname = f"{entry}-dual{dual_gloss_counter:06d}-snip{snippet_counter:03d}-{safe}.mp4"
                out_path = os.path.join(output_dir, fname)

                # Write file
                try:
                    comp.write_videofile(out_path, codec='libx264', audio_codec='aac')
                    print(f"Saved snippet: {out_path}")
                except Exception as e:
                    print(f"Error writing {out_path}: {e}")
                finally:
                    segment.close()
                    comp.close()

    # Close video clips
    for c in clips.values():
        if c:
            c.close()

print(f"Processing completed: {dual_gloss_counter} dual glosses, {snippet_counter} snippets created.")
