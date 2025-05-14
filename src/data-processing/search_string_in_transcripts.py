import re
import argparse
import subprocess
from pathlib import Path
import unicodedata

ADD_SUBTITLES = False

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
            text = ' '.join(lines[2:])
            yield idx, start_ts, end_ts, text

def srt_ts_to_seconds(ts):
    hh, mm, rest = ts.split(':')
    ss, ms = rest.split(',')
    return int(hh) * 3600 + int(mm) * 60 + int(ss) + int(ms) / 1000.0

def format_ff_time(seconds):
    hh = int(seconds // 3600)
    mm = int((seconds % 3600) // 60)
    ss = seconds % 60
    return f"{hh:02d}:{mm:02d}:{ss:06.3f}"

def extract_snippet(video_path, start_ts, end_ts, burn_text, out_path):
    start_sec = srt_ts_to_seconds(start_ts)
    end_sec = srt_ts_to_seconds(end_ts)
    duration = end_sec - start_sec

    start_ff = format_ff_time(start_sec)
    dur_ff = format_ff_time(duration)

    cmd = [
        "ffmpeg",
        "-y",
        "-ss", start_ff,
        "-i", str(video_path),
        "-t", dur_ff,
    ]

    if ADD_SUBTITLES:
        cmd += [
            "-vf", (
                f"drawtext="
                f"text='{burn_text}':"
                f"fontcolor=white:fontsize=12:"
                f"box=1:boxcolor=black@0.5:boxborderw=5:"
                f"x=(w-text_w)/2:y=h-(text_h)-10"
            )
        ]

    cmd += ["-c:a", "copy", str(out_path)]
    subprocess.run(cmd, check=True)

def sanitize_filename(text):
    # Normalize to closest ASCII equivalent
    text = unicodedata.normalize('NFKD', text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    # Replace remaining non-filename-safe characters with underscores
    text = re.sub(r'[^\w\-.]', '_', text)
    return text

def main():
    p = argparse.ArgumentParser(description="🔍 Search SRTs and extract video snippets 🎞️")
    p.add_argument("search_string", help="🔎 Substring to look for in the SRT text")
    p.add_argument("--root", default="/Volumes/IISY/DGSKorpus", help="📁 Root folder with entry_* subfolders")
    args = p.parse_args()

    search = args.search_string
    root = Path(args.root)
    output_dir = root / "video-snippets" / "search-results"
    output_dir.mkdir(parents=True, exist_ok=True)

    entries = [e for e in sorted(root.iterdir()) if e.is_dir() and e.name.startswith("entry_")]
    total = len(entries)
    if total == 0:
        print("❌ No entry directories found.")
        return

    print(f"\n🚀 Starting search for '{search}' across {total} entries...\n")

    matches = []  # Store matches for final output

    for i, entry in enumerate(entries, 1):
        percent = (i / total) * 100
        print(f"🔄 [{percent:5.1f}%] Processing {entry.name} ({i}/{total})")

        for speaker in ("a", "b"):
            srt_file = entry / f"speaker-{speaker}.srt"
            video_file = entry / f"video-{speaker}.mp4"
            if not srt_file.exists() or not video_file.exists():
                continue

            for idx, start, end, text in parse_srt_blocks(srt_file):
                if search in text:
                    safe_search = sanitize_filename(search)
                    out_name = f"{safe_search.replace(' ', '')}.mp4"
                    out_path = output_dir / out_name

                    extract_snippet(video_file, start, end, search, out_path)

                    matches.append({
                        "entry": entry.name,
                        "file": srt_file.name,
                        "index": idx,
                        "output": str(out_path)
                    })

    print("\n🎉 Done! All matching snippets have been processed.\n")

    if matches:
        print("📄 Summary of found matches:\n")
        for m in matches:
            print(f"🔹 {m['entry']} | {m['file']} | Index #{m['index']} → 📁 {m['output']}")
    else:
        print("❗ No matches found for your search string.")

if __name__ == "__main__":
    main()
