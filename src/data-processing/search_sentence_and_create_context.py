import re
import argparse
import subprocess
from pathlib import Path
import unicodedata
from collections import defaultdict
import json

def parse_srt_blocks(srt_path):
    with open(srt_path, encoding='utf-8-sig') as f:
        content = f.read()

    blocks = re.split(r'\n{2,}', content.strip())
    for block in blocks:
        lines = block.splitlines()
        if len(lines) >= 3:
            idx = lines[0].strip()
            times = lines[1].strip()
            m = re.match(r'(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})', times)
            if not m:
                continue
            start_ts, end_ts = m.groups()
            text = ' '.join(lines[2:]).strip()
            is_full_sentence = text.endswith("_FULL_SENTENCE")
            if is_full_sentence:
                text = text.replace("_FULL_SENTENCE", "").strip()
            yield idx, start_ts, end_ts, text, is_full_sentence

def srt_ts_to_seconds(ts):
    hh, mm, rest = ts.split(':')
    ss, ms = rest.split(',')
    return int(hh) * 3600 + int(mm) * 60 + int(ss) + int(ms) / 1000.0

def format_ff_time(seconds):
    hh = int(seconds // 3600)
    mm = int((seconds % 3600) // 60)
    ss = seconds % 60
    return f"{hh:02d}:{mm:02d}:{ss:06.3f}"

def get_video_duration(video_path):
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json",
        str(video_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    data = json.loads(result.stdout)
    return float(data["format"]["duration"])

def extract_snippet(video_path, start_sec, duration, out_path):
    video_duration = get_video_duration(video_path)
    
    # Adapt start time if it exceeds video duration
    if start_sec >= video_duration:
        print(f"Start time {start_sec}s exceeds video duration {video_duration}s for {out_path}. Setting start time to 0.")
        start_sec = 0
        duration = min(duration, video_duration)
    # Adjust duration if it extends beyond video duration
    elif start_sec + duration > video_duration:
        duration = video_duration - start_sec
        print(f"Adjusted duration for {out_path} to {duration}s to fit video duration {video_duration}s")
    
    if duration <= 0:
        print(f"Skipping extraction for {out_path}: duration <= 0 after adjustments")
        return False
    
    start_ff = format_ff_time(start_sec)
    dur_ff = format_ff_time(duration)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss", start_ff,
        "-i", str(video_path),
        "-t", dur_ff,
        "-c:v", "libx264",
        "-c:a", "aac",
        str(out_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Extraction failed for {out_path}: {result.stderr}")
        return False
    if not out_path.exists() or out_path.stat().st_size == 0:
        print(f"Extraction produced empty file for {out_path}")
        return False
    return True

def combine_snippets(snippet_a, snippet_b, out_path):
    if not snippet_a.exists() or not snippet_b.exists():
        print(f"Skipping combination for {out_path}: one or both snippets are missing")
        return
    if snippet_a.stat().st_size == 0 or snippet_b.stat().st_size == 0:
        print(f"Skipping combination for {out_path}: one or both snippets are empty")
        return
    cmd = [
        "ffmpeg",
        "-i", str(snippet_a),
        "-i", str(snippet_b),
        "-filter_complex", "[0:v][1:v]hstack=inputs=2[v]",
        "-map", "[v]",
        "-map", "0:a?",
        "-map", "1:a?",
        "-c:v", "libx264",
        "-c:a", "aac",
        str(out_path)
    ]
    subprocess.run(cmd, check=True)

def sanitize_filename(text):
    text = unicodedata.normalize('NFKD', text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = re.sub(r'[^\w\-.]', '_', text)
    text = re.sub(r'_+', '_', text).strip('_')
    return text or "snippet"

def load_sentences(txt_file):
    with open(txt_file, encoding='utf-8') as f:
        sentences = [line.strip() for line in f if line.strip()]
    return sentences

def main():
    p = argparse.ArgumentParser(description="Extract video snippets before sentences found in SRTs and collect relevant _FULL_SENTENCE texts")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("-s", "--sentence", dest="sentence",
                       help="Single sentence to search for")
    group.add_argument("-l", "--list", dest="list_file",
                       help="Path to a TXT file containing sentences, one per line")
    p.add_argument("--root", default="/Volumes/IISY/DGSKorpus", help="Root folder with entry_* subfolders")
    args = p.parse_args()

    # Load sentences
    if args.list_file:
        sentences = load_sentences(args.list_file)
    else:
        sentences = [args.sentence]

    root = Path(args.root)
    output_dir = root / "video-snippets" / "search-results" / "context"
    output_dir.mkdir(parents=True, exist_ok=True)
    full_sentences_dir = output_dir / "full_sentences"
    full_sentences_dir.mkdir(exist_ok=True)

    entries = [e for e in sorted(root.iterdir()) if e.is_dir() and e.name.startswith("entry_")]
    if not entries:
        print("❌ No entry directories found.")
        return

    total = len(entries)
    print(f"\n🚀 Starting processing for {len(sentences)} sentence(s) across {total} entries...\n")

    full_sentences_per_sentence = defaultdict(set)
    snippet_count = {}

    for i, entry in enumerate(entries, 1):
        percent = (i / total) * 100
        print(f"🔄 [{percent:5.1f}%] Processing {entry.name} ({i}/{total})")

        srt_a = entry / "speaker-a.srt"
        srt_b = entry / "speaker-b.srt"
        video_a = entry / "video-a.mp4"
        video_b = entry / "video-b.mp4"

        if not all([srt_a.exists(), srt_b.exists(), video_a.exists(), video_b.exists()]):
            continue

        # Parse SRT blocks for both speakers
        blocks_a = list(parse_srt_blocks(srt_a))
        blocks_b = list(parse_srt_blocks(srt_b))

        # Find matching sentences in both SRTs
        matches = []
        for speaker, blocks in [("a", blocks_a), ("b", blocks_b)]:
            for idx, start_ts, end_ts, text, _ in blocks:
                for sentence in sentences:
                    if sentence in text:
                        matches.append((speaker, idx, start_ts, end_ts, sentence))

        for match in matches:
            speaker, idx, start_ts, end_ts, sentence = match
            T = srt_ts_to_seconds(start_ts)
            initial_start = max(0, T - 15)

            # Find all _FULL_SENTENCE blocks from both speakers
            full_sentences = [b for b in blocks_a + blocks_b if b[4]]

            # Adjust initial_start if it falls within a _FULL_SENTENCE
            for _, fs_start_ts, fs_end_ts, _, _ in full_sentences:
                fs_start = srt_ts_to_seconds(fs_start_ts)
                fs_end = srt_ts_to_seconds(fs_end_ts)
                if fs_start <= initial_start < fs_end:
                    initial_start = fs_start
                    break

            end_time = T
            duration = end_time - initial_start

            if duration <= 0:
                print(f"Skipping snippet for {entry.name} index {idx}: duration <= 0")
                continue

            # Collect _FULL_SENTENCE entries that overlap with [initial_start, T)
            for _, fs_start_ts, fs_end_ts, fs_text, _ in full_sentences:
                fs_start = srt_ts_to_seconds(fs_start_ts)
                fs_end = srt_ts_to_seconds(fs_end_ts)
                if fs_start < T and fs_end > initial_start:
                    full_sentences_per_sentence[sentence].add(fs_text)

            # Extract snippets for both speakers with adjustments
            snippet_a_path = output_dir / f"temp_snippet_a_{entry.name}_{idx}.mp4"
            snippet_b_path = output_dir / f"temp_snippet_b_{entry.name}_{idx}.mp4"
            success_a = extract_snippet(video_a, initial_start, duration, snippet_a_path)
            success_b = extract_snippet(video_b, initial_start, duration, snippet_b_path)

            if not (success_a and success_b):
                print(f"Skipping combination for {entry.name} index {idx}: extraction failed")
                continue

            # Combine snippets
            safe_name = sanitize_filename(sentence)
            count = snippet_count.get(sentence, 0) + 1
            snippet_count[sentence] = count
            if count > 1:
                out_name = f"{safe_name}_{count:03d}_combined.mp4"
            else:
                out_name = f"{safe_name}_combined.mp4"
            out_path = output_dir / out_name
            combine_snippets(snippet_a_path, snippet_b_path, out_path)

            # Clean up temporary files
            if snippet_a_path.exists():
                snippet_a_path.unlink()
            if snippet_b_path.exists():
                snippet_b_path.unlink()

    # Write _FULL_SENTENCE texts to separate files for each sentence
    for sentence in sentences:
        fs_set = full_sentences_per_sentence[sentence]
        fs_list = sorted(fs_set)
        safe_name = sanitize_filename(sentence)
        txt_path = full_sentences_dir / f"{safe_name}_full_sentences.txt"
        with open(txt_path, "w", encoding="utf-8") as f:
            for fs in fs_list:
                f.write(fs + "\n")
    
    # Combine all full_sentences into a final grouped file
    final_combined_path = output_dir / "all_full_sentences_combined.txt"
    with open(final_combined_path, "w", encoding="utf-8") as final_file:
        for sentence in sentences:
            safe_name = sanitize_filename(sentence)
            txt_path = full_sentences_dir / f"{safe_name}_full_sentences.txt"
            if txt_path.exists():
                final_file.write(f"===== Sentence: {sentence} =====\n\n")
                with open(txt_path, "r", encoding="utf-8") as sf:
                    final_file.write(sf.read().strip() + "\n\n")

    print("\n🎉 Done! All matching snippets have been processed and relevant _FULL_SENTENCE texts collected.\n")

if __name__ == "__main__":
    main()