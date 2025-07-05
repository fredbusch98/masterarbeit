import os
import re
import pandas as pd
import pysrt  # For SRT parsing
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from all_types import load_gloss_types

# Global list to hold raw metric rows.
raw_rows = []

def process_sentence_blocks(blocks, gloss_types, raw_rows):
    """
    Process gloss blocks from the same sentence & speaker.
    For main (non-lost) glosses, compute IGT, OGT, and then TGT = IGT + GD + OGT.
    For lost glosses, record only GD (and set IGT/OGT to 0; hence TGT = GD).
    Each gloss occurrence gets a row with extra details such as the computed index strings.
    """
    if not blocks:
        return
    
    # Separate main vs. lost glosses.
    main_blocks = [b for b in blocks if not b.get("lost", False)]
    lost_blocks = [b for b in blocks if b.get("lost", False)]
    
    # --- Process main gloss blocks ---
    if main_blocks:
        if len(main_blocks) == 1:
            block = main_blocks[0]
            block_text = block["text"].replace("_END_SENTENCE", "")
            block_text = block_text.split(":")[0]
            for gloss in gloss_types:
                if gloss == block_text:
                    gd = block["duration_ms"]
                    igt = 0
                    ogt = 0
                    tgt = gd
                    raw_rows.append({
                        "entry": block["entry"],
                        "block_index": block["index"],
                        "speaker": block["speaker"],
                        "gloss": gloss,
                        "gd": gd,
                        "igt": igt,
                        "ogt": ogt,
                        "tgt": tgt,
                        "igt_index": block["index"],
                        "ogt_index": block["index"],
                        "tgt_index": block["index"],
                        "lost": False
                    })
        else:
            # Group blocks by identical (start_ms, end_ms)
            timestamp_groups = {}
            for block in main_blocks:
                key = (block["start_ms"], block["end_ms"])
                timestamp_groups.setdefault(key, []).append(block)
            sorted_groups = sorted(timestamp_groups.items(), key=lambda x: x[0][0])
            
            for i, (timestamp, group_blocks) in enumerate(sorted_groups):
                current_group_index = group_blocks[0].get("index", "")
                # If *any* block in the current timestamp‑group ends with "_END_SENTENCE"
                # we must NOT compute OGT for it, even when a later group exists.
                ends_with_end_sentence = any(
                    blk["text"].strip().endswith("_END_SENTENCE") for blk in group_blocks
                )
                # Compute IGT for groups after the first.
                if i > 0:
                    prev_group = sorted_groups[i-1][1]
                    prev_group_index = prev_group[0].get("index", "0")
                    current_igt = max(0, timestamp[0] - sorted_groups[i-1][0][1])
                    igt_index = f"index:{prev_group_index}-{current_group_index}"
                else:
                    current_igt = 0
                    igt_index = "None"
                # Compute OGT for groups before the last.
                if i < len(sorted_groups) - 1 and not ends_with_end_sentence:
                    next_group = sorted_groups[i + 1][1]
                    next_group_index = next_group[0].get("index", "0")
                    current_ogt = max(0, next_group[0]["start_ms"] - timestamp[1])
                    ogt_index = f"index:{current_group_index}-{next_group_index}"
                else:  # last group in list *or* it ends with _END_SENTENCE
                    current_ogt = 0
                    ogt_index = "None"
                # Build TGT index string combining IGT and OGT info.
                if i > 0 and i < len(sorted_groups) - 1 and not ends_with_end_sentence:
                    tgt_index = (
                        f"IGT:{prev_group_index}-{current_group_index} | "
                        f"OGT:{current_group_index}-{next_group_index}"
                    )
                elif i > 0:
                    tgt_index = f"IGT:{prev_group_index}-{current_group_index} | OGT:None"
                elif i < len(sorted_groups) - 1 and not ends_with_end_sentence:
                    tgt_index = f"IGT:None | OGT:{current_group_index}-{next_group_index}"
                else:
                    tgt_index = current_group_index
                
                for block in group_blocks:
                    block_text = block["text"].replace("_END_SENTENCE", "")
                    block_text = block_text.split(":")[0]
                    for gloss in gloss_types:
                        if gloss == block_text:
                            gd = block["duration_ms"]
                            tgt = current_igt + gd + current_ogt
                            raw_rows.append({
                                "entry": block["entry"],
                                "block_index": block["index"],
                                "speaker": block["speaker"],
                                "gloss": gloss,
                                "gd": gd,
                                "igt": current_igt,
                                "ogt": current_ogt,
                                "tgt": tgt,
                                "igt_index": igt_index,
                                "ogt_index": ogt_index,
                                "tgt_index": tgt_index,
                                "lost": False
                            })
    # --- Process lost gloss blocks: only GD is used.
    for block in lost_blocks:
        block_text = block["text"].replace("_END_SENTENCE", "")
        block_text = block_text.split(":")[0]
        for gloss in gloss_types:
            if gloss == block_text:
                gd = block["duration_ms"]
                raw_rows.append({
                    "entry": block["entry"],
                    "block_index": block["index"],
                    "speaker": block["speaker"],
                    "gloss": gloss,
                    "gd": gd,
                    "igt": 0,
                    "ogt": 0,
                    "tgt": gd,
                    "igt_index": block["index"],
                    "ogt_index": block["index"],
                    "tgt_index": block["index"],
                    "lost": True
                })

def process_srt_file(srt_path, speakers_to_process, gloss_types, entry, raw_rows):
    """
    Process an SRT file for specified speakers. It uses sentence markers (_FULL_SENTENCE and _END_SENTENCE)
    to segment gloss blocks and mark those that are “lost.”
    """
    if not os.path.exists(srt_path):
        print(f"⚠️ {srt_path} not found.")
        return

    subtitles = pysrt.open(srt_path, encoding="utf-8")
    current_sentence_blocks = {}  # Per speaker.
    lost_mode = {}  # Tracks per speaker whether subsequent glosses are lost.

    for subtitle in subtitles:
        block_index = subtitle.index
        start_ms = subtitle.start.ordinal
        end_ms = subtitle.end.ordinal
        duration_ms = end_ms - start_ms

        lines = subtitle.text.splitlines()
        if not lines:
            continue

        first_line = lines[0].strip()
        match_speaker = re.match(r"^(A|B):\s*(.*)$", first_line)
        if not match_speaker:
            continue
        speaker = match_speaker.group(1)
        if speaker not in speakers_to_process:
            continue

        text_content = re.sub(r"^(A|B):\s*", "", subtitle.text, flags=re.MULTILINE).strip()
        is_full_sentence = text_content.endswith("_FULL_SENTENCE")
        is_end_sentence = text_content.endswith("_END_SENTENCE")

        if speaker not in lost_mode:
            lost_mode[speaker] = False

        if is_full_sentence:
            # Process the sentence and reset lost_mode.
            if speaker in current_sentence_blocks and current_sentence_blocks[speaker]:
                process_sentence_blocks(current_sentence_blocks[speaker], gloss_types, raw_rows)
                current_sentence_blocks[speaker] = []
            lost_mode[speaker] = False
            continue

        block_data = {
            "start_ms": start_ms,
            "end_ms": end_ms,
            "text": text_content,
            "duration_ms": duration_ms,
            "entry": entry,
            "index": str(block_index),
            "speaker": speaker
        }

        if is_end_sentence:
            block_data["lost"] = False
            current_sentence_blocks.setdefault(speaker, []).append(block_data)
            lost_mode[speaker] = True
        else:
            block_data["lost"] = lost_mode[speaker]
            current_sentence_blocks.setdefault(speaker, []).append(block_data)
    
    # Process any remaining blocks.
    for spkr, blocks_list in current_sentence_blocks.items():
        if blocks_list:
            process_sentence_blocks(blocks_list, gloss_types, raw_rows)

def main():
    base_path = "/Volumes/IISY/DGSKorpus/"
    gloss_types = load_gloss_types(os.path.join(base_path, "all-types-dgs.csv"))
    gloss_types = {g.rstrip('^') for g in gloss_types}
    
    entries = [entry for entry in os.listdir(base_path)
               if os.path.isdir(os.path.join(base_path, entry)) and entry.startswith("entry_")]
    
    print(f"Processing {len(entries)} entries for raw gloss metrics...")
    
    for entry in entries:
        folder_path = os.path.join(base_path, entry)
        process_srt_file(os.path.join(folder_path, "speaker-a.srt"), ["A"], gloss_types, entry, raw_rows)
        process_srt_file(os.path.join(folder_path, "speaker-b.srt"), ["B"], gloss_types, entry, raw_rows)
        print(f"{entry} processed.")
    
    # Convert raw_rows to a pandas DataFrame and write to CSV.
    output_csv = os.path.join(base_path, "dgs-gloss-times", "raw_gloss_metrics.csv")
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df = pd.DataFrame(raw_rows)
    df.to_csv(output_csv, index=False)
    print(f"Raw gloss metrics written to {output_csv}")

if __name__ == "__main__":
    main()
