"""
🔍 Search through SRT subtitle files for specific terms and automatically extract matching video snippets.
🎞️ Supports single-term or multi-term searches and optionally burns matched text onto the clips.
📁 Organizes extracted snippets in a structured output folder for easy review.
"""
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
    # Collapse multiple underscores
    text = re.sub(r'_+', '_', text).strip('_')
    return text or "snippet"


def load_search_terms(txt_file):
    with open(txt_file, encoding='utf-8') as f:
        terms = [line.strip() for line in f if line.strip()]
    return terms


def main():
    p = argparse.ArgumentParser(description="🔍 Search SRTs and extract video snippets 🎞️")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("-s", "--search", dest="search_string",
                       help="🔎 Substring to look for in the SRT text")
    group.add_argument("-l", "--list", dest="list_file",
                       help="📄 Path to a TXT file containing search strings, one per line")
    p.add_argument("--root", default="/Volumes/IISY/DGSKorpus", help="📁 Root folder with entry_* subfolders")
    args = p.parse_args()

    # Load search terms
    if args.list_file:
        search_terms = load_search_terms(args.list_file)
    else:
        search_terms = [args.search_string]

    root = Path(args.root)
    output_dir = root / "video-snippets" / "search-results"
    output_dir.mkdir(parents=True, exist_ok=True)

    entries = [e for e in sorted(root.iterdir()) if e.is_dir() and e.name.startswith("entry_")]
    if not entries:
        print("❌ No entry directories found.")
        return

    total = len(entries)
    print(f"\n🚀 Starting search for {len(search_terms)} term(s) across {total} entries...\n")

    matches = []  # Store matches for final output
    snippet_count = {}

    for i, entry in enumerate(entries, 1):
        percent = (i / total) * 100
        print(f"🔄 [{percent:5.1f}%] Processing {entry.name} ({i}/{total})")

        for speaker in ("a", "b"):
            srt_file = entry / f"speaker-{speaker}.srt"
            video_file = entry / f"video-{speaker}.mp4"
            if not srt_file.exists() or not video_file.exists():
                continue

            for idx, start, end, text in parse_srt_blocks(srt_file):
                for term in search_terms:
                    if term in text:
                        # Count occurrences per term to avoid name clashes
                        count = snippet_count.get(term, 0) + 1
                        snippet_count[term] = count
                        safe_name = sanitize_filename(term)
                        if count > 1:
                            out_name = f"{safe_name}_{count:03d}_og_dgs.mp4"
                        else:
                            out_name = f"{safe_name}_og_dgs.mp4"
                        out_path = output_dir / out_name

                        # Extract the snippet
                        extract_snippet(video_file, start, end, term, out_path)

                        matches.append({
                            "term": term,
                            "entry": entry.name,
                            "file": srt_file.name,
                            "index": idx,
                            "output": str(out_path)
                        })

    print("\n🎉 Done! All matching snippets have been processed.\n")

    if matches:
        print("📄 Summary of found matches:\n")
        for m in matches:
            print(f"🔹 '{m['term']}' in {m['entry']} | {m['file']} | Index #{m['index']} → 📁 {m['output']}")
    else:
        print("❗ No matches found for the given search terms.")

if __name__ == "__main__":
    main()
