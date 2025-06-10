#!/usr/bin/env python3
"""
sentence2video_mapper.py

This script processes folders containing:
  - video-a.mp4
  - video-b.mp4
  - speaker-a.srt
  - speaker-b.srt

For each “full sentence” entry in the SRTs (marked with `_FULL_SENTENCE` and ending with `_END_SENTENCE`),
it outputs:
  1) A per-folder CSV file (`sentence2video.csv`) in each entry_X folder, with one row per full sentence,
     containing:
       - full_sentence: the text without the `_FULL_SENTENCE` suffix
       - gloss_sequence: comma-joined component glosses (no `_END_SENTENCE` suffix)
       - video_path: relative path to the speaker’s video (e.g., entry_X/video-a.mp4 or entry_X/video-b.mp4)
       - start_time: timestamp in `HH:MM:SS.mmm` format (milliseconds precision)
       - end_time: timestamp in `HH:MM:SS.mmm` format (milliseconds precision)

  2) A combined CSV (`sentence2video-mapping-combined.csv`) saved under the DGSKorpus root, merging all per-folder rows.
"""

import os
import csv
from pysrt import SubRipFile
import sys

# If you want to process all subfolders under a root, set this to True.
process_all_folders = True


def format_srt_timestamp(srt_time):
    """
    Given a SubRipTime object, return a string in "HH:MM:SS.mmm" format.
    """
    hours = srt_time.hours
    minutes = srt_time.minutes
    seconds = srt_time.seconds
    milliseconds = srt_time.milliseconds
    return f"{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}"


def parse_srt_for_sentence_entries(srt_file_path, speaker):
    """
    Parse the SRT so that:
      - Any subtitle line beginning with "{speaker}: " whose gloss ends with "_FULL_SENTENCE"
        becomes the anchor for a new “sentence entry.”
      - We extract that entry's start and end timestamps (strings with milliseconds).
      - Then we look at each subsequent subtitle entry for the same speaker:
          • If a gloss ends with "_END_SENTENCE", strip that suffix, add it to the gloss list, and stop.
          • Otherwise, assume it is one of the component glosses—strip no suffix and add to the list.
      - If we never find an "_END_SENTENCE" after a "_FULL_SENTENCE", we skip that sentence.
      - We return a list of dicts, each containing:
          {
            "full_sentence": <string without suffix>,
            "gloss_sequence": <comma-joined string of glosses (no suffixes)>,
            "start_time": <"HH:MM:SS.mmm">,
            "end_time": <"HH:MM:SS.mmm">
          }.
    """
    entries = []
    try:
        print(f"[INFO] Parsing SRT file for speaker {speaker}: {srt_file_path}")
        subs = SubRipFile.open(srt_file_path)
        speaker_tag = f"{speaker}: "

        # Convert SubRipFile to a simple list to allow lookahead
        all_subs = [sub for sub in subs if sub.text.strip()]
        i = 0
        while i < len(all_subs):
            entry = all_subs[i]
            text = entry.text.strip()

            if text.startswith(speaker_tag):
                gloss_raw = text[len(speaker_tag):].strip()
                if gloss_raw.endswith("_FULL_SENTENCE"):
                    full_sentence = gloss_raw.replace("_FULL_SENTENCE", "").strip()
                    start_time_str = format_srt_timestamp(entry.start)
                    end_time_str = format_srt_timestamp(entry.end)

                    gloss_list = []
                    j = i + 1
                    found_end = False
                    while j < len(all_subs):
                        next_entry = all_subs[j]
                        next_text = next_entry.text.strip()

                        if not next_text.startswith(speaker_tag):
                            break

                        next_gloss_raw = next_text[len(speaker_tag):].strip()
                        if next_gloss_raw.endswith("_END_SENTENCE"):
                            cleaned = next_gloss_raw.replace("_END_SENTENCE", "").strip()
                            gloss_list.append(cleaned)
                            found_end = True
                            break
                        else:
                            gloss_list.append(next_gloss_raw)
                        j += 1

                    if found_end and gloss_list:
                        gloss_sequence = ",".join(gloss_list)
                        full_sentence = full_sentence.rstrip("/")
                        entries.append({
                            "full_sentence": full_sentence,
                            "gloss_sequence": gloss_sequence,
                            "start_time": start_time_str,
                            "end_time": end_time_str
                        })
                        i = j
                    else:
                        print(
                            f"[WARNING] No END_SENTENCE found for FULL_SENTENCE '{full_sentence}' in {srt_file_path}. Skipping."
                        )
            i += 1

        print(f"[INFO] Found {len(entries)} full-sentence entries for speaker {speaker}.")
    except Exception as e:
        print(f"[ERROR] {e}")
    return entries


def process_folder(folder_path):
    """
    For a given folder containing:
      - video-a.mp4
      - video-b.mp4
      - speaker-a.srt
      - speaker-b.srt

    This will:
      1) Parse each .srt to find full-sentence entries for A and B.
      2) Write out a per-folder CSV (sentence2video.csv) with columns:
         full_sentence, gloss_sequence, video_path, start_time, end_time
    Returns a list of rows (each a list of 5 strings) for this folder.
    """
    srt_a = os.path.join(folder_path, "speaker-a.srt")
    srt_b = os.path.join(folder_path, "speaker-b.srt")

    print(f"[INFO] Processing folder: {folder_path}")

    # Parse SRTs for full-sentence entries
    sentences_A = parse_srt_for_sentence_entries(srt_a, "A")
    sentences_B = parse_srt_for_sentence_entries(srt_b, "B")

    folder_name = os.path.basename(folder_path)
    rows = []

    # Prepare per-folder CSV
    csv_path = os.path.join(folder_path, "sentence2video.csv")
    print(f"[INFO] Writing output CSV to {csv_path}")
    try:
        with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            # Header
            writer.writerow([
                "full_sentence",
                "gloss_sequence",
                "video_path",
                "start_time",
                "end_time",
            ])

            # Rows for speaker A
            for sent in sentences_A:
                video_rel = os.path.join(folder_name, "video-a.mp4")
                row = [
                    sent["full_sentence"],
                    sent["gloss_sequence"],
                    video_rel,
                    sent["start_time"],
                    sent["end_time"],
                ]
                writer.writerow(row)
                rows.append(row)

            # Rows for speaker B
            for sent in sentences_B:
                video_rel = os.path.join(folder_name, "video-b.mp4")
                row = [
                    sent["full_sentence"],
                    sent["gloss_sequence"],
                    video_rel,
                    sent["start_time"],
                    sent["end_time"],
                ]
                writer.writerow(row)
                rows.append(row)

        print(f"[INFO] Successfully wrote sentence2video.csv for folder: {folder_path}")
    except Exception as e:
        print(f"[ERROR] Could not write CSV file for folder {folder_path}: {e}")

    return rows


def combine_all_csv(root_path, entries, combined_filename="sentence2video-mapping-combined.csv"):
    """
    Combines per-folder CSV rows into one master CSV under root_path.
    Expects:
      - entries: list of folder names (e.g., ["entry_1", "entry_2", ...])
      - Each folder contains sentence2video.csv with matching columns.
    Writes combined CSV at root_path/combined_filename.
    """
    combined_path = os.path.join(root_path, combined_filename)
    header = ["full_sentence", "gloss_sequence", "video_path", "start_time", "end_time"]

    print(f"[INFO] Writing combined CSV to {combined_path}")
    try:
        with open(combined_path, "w", newline="", encoding="utf-8") as combined_file:
            writer = csv.writer(combined_file)
            writer.writerow(header)

            for entry in entries:
                folder_csv = os.path.join(root_path, entry, "sentence2video.csv")
                if not os.path.exists(folder_csv):
                    print(f"[WARNING] Missing CSV in {entry}, skipping.")
                    continue
                with open(folder_csv, "r", newline="", encoding="utf-8") as csvfile:
                    reader = csv.reader(csvfile)
                    next(reader, None)  # skip header
                    for row in reader:
                        writer.writerow(row)

        print(f"[INFO] Successfully wrote combined CSV: {combined_path}")
    except Exception as e:
        print(f"[ERROR] Could not write combined CSV at {combined_path}: {e}")


def main():
    starting_entry = None
    if len(sys.argv) > 1:
        starting_entry = sys.argv[1]

    if process_all_folders:
        root_path = "/Volumes/IISY/DGSKorpus/"  # ← change this to your actual root
        entries = [
            entry for entry in os.listdir(root_path)
            if os.path.isdir(os.path.join(root_path, entry)) and entry.startswith("entry_")
        ]
        if starting_entry:
            try:
                idx = entries.index(starting_entry)
                entries = entries[idx:]
            except ValueError:
                print(f"[ERROR] The folder {starting_entry} was not found in {root_path}.")
                return

        # Process each entry folder, collecting rows
        for entry in entries:
            entry_path = os.path.join(root_path, entry)
            process_folder(entry_path)

        # After all per-folder CSVs are created, combine them
        combine_all_csv(root_path, entries)

    else:
        # For testing one folder:
        folder_path = "/Volumes/IISY/DGSKorpus/entry_3"
        process_folder(folder_path)
        # Optionally combine if you want to test combining single-folder
        combine_all_csv(os.path.dirname(folder_path), [os.path.basename(folder_path)])


if __name__ == "__main__":
    main()
