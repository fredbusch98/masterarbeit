import os
import re

# Main folder path
main_folder = '/Volumes/IISY/DGSKorpus'

# Helpers
def is_full_sentence(tokens):
    return any(tok.endswith('_FULL_SENTENCE') for tok in tokens)

def line_ends_full(lines):
    return any('_FULL_SENTENCE' in l for l in lines)

# Timestamp parsing/formatting
def parse_timestamp(ts):
    time_part, ms = ts.split(',')
    h, m, s = time_part.split(':')
    return int(h)*3600000 + int(m)*60000 + int(s)*1000 + int(ms)

def format_timestamp(ms):
    h = ms // 3600000
    ms -= h * 3600000
    m = ms // 60000
    ms -= m * 60000
    s = ms // 1000
    ms -= s * 1000
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def parse_srt_timestamp(line):
    start, end = line.strip().split(' --> ')
    return parse_timestamp(start), parse_timestamp(end)

# Parse raw SRT into list of (timestamp_line, text_lines)
def parse_srt(lines):
    entries, block = [], []
    for ln in lines:
        if not ln.strip():
            if len(block) >= 2:
                entries.append((block[1], block[2:]))
            block = []
        else:
            block.append(ln)
    if len(block) >= 2:
        entries.append((block[1], block[2:]))
    return entries

# Swap duplicates on raw entries before marking (adjust_swaps_before_mark)
# Ensure first doesn't end with FULL_SENTENCE meaning it is a gloss and second ends with FULL_SENTENCE
# Subtract 1ms from FULL_SENETCE start and swap order
# Such cases occur occasionally (a total of 327 times across all transripts of the DGS Korpus Release 3)
# The swaps must be made so that the gloss can be sequentially mapped exactly to its sentence later on during the Text2Gloss Mapper
# Example of such a problematic case in a transcript:

# Original (Problematic for sequential mapping of the Text2Gloss Mapper):
# 24
# 00:00:07,160 --> 00:00:09,600
# B: $GEST-NM-KOPFNICKEN1

# 25
# 00:00:07,160 --> 00:00:09,600
# B: Ja!_FULL_SENTENCE

# Becomes (Fixed for sequential mapping of the Text2Gloss Mapper):
# 24
# 00:00:07,159 --> 00:00:09,600
# B: Ja!_FULL_SENTENCE

# 25
# 00:00:07,160 --> 00:00:09,600
# B: $GEST-NM-KOPFNICKEN1
def adjust_swaps_before_mark(entries, speaker):
    res = []
    i = 0
    label = f"{speaker}:"
    while i < len(entries):
        if i+1 < len(entries):
            ts1, lines1 = entries[i]
            ts2, lines2 = entries[i+1]
            if ts1.strip() == ts2.strip():
                if (lines1[0].strip().startswith(label) and lines2[0].strip().startswith(label)):
                    if not line_ends_full(lines1) and line_ends_full(lines2):
                        # adjust ts2
                        start2, end2 = parse_srt_timestamp(ts2)
                        new_start = max(0, start2 - 1)
                        parts = ts2.split(' --> ')
                        new_ts2 = format_timestamp(new_start) + ' --> ' + parts[1]
                        cleaned_lines2 = lines2.copy()
                        while cleaned_lines2 and not cleaned_lines2[0].strip():
                            cleaned_lines2.pop(0)
                        res.append((new_ts2 + '', cleaned_lines2))
                        res.append((ts1, lines1))
                        i += 2
                        continue
        res.append(entries[i])
        i += 1
    return res

# Process and mark sentences (_END_SENTENCE)
def process_entries(entries, speaker):
    out = []
    tag = f"{speaker}: "
    i = 0
    while i < len(entries):
        ts_line, text_lines = entries[i]
        start_ms, end_ms = parse_srt_timestamp(ts_line)
        tokens = ' '.join(l.strip() for l in text_lines)[len(tag):].split()
        if is_full_sentence(tokens):
            block = [(ts_line, text_lines)]
            j = i + 1
            while j < len(entries):
                next_ts, next_lines = entries[j]
                nxt_tokens = ' '.join(l.strip() for l in next_lines)[len(tag):].split()
                next_start, _ = parse_srt_timestamp(next_ts)
                if is_full_sentence(nxt_tokens) or next_start > end_ms:
                    break
                block.append((next_ts, next_lines))
                j += 1
            # mark last
            for idx, (b_ts, b_lines) in enumerate(block):
                if idx == len(block) - 1:
                    b_lines[-1] = b_lines[-1].rstrip() + "_END_SENTENCE\n"
                out.append((b_ts, b_lines))
            i = j
        else:
            out.append(entries[i])
            i += 1
    return out

# Main
if __name__ == '__main__':
    folders = [d for d in os.listdir(main_folder) if d.startswith('entry_')]
    for entry in folders:
        src = os.path.join(main_folder, entry, 'filtered-transcript.srt')
        if not os.path.exists(src):
            print(f"⚠️ Missing {src}, skipping")
            continue
        print(f"🔍 Processing {entry}...")

        with open(src) as f:
            lines = f.readlines()
        raw_entries = parse_srt(lines)

        # split by speaker
        per_sp = {'A': [], 'B': []}
        for ts, txt in raw_entries:
            m = re.match(r'([AB]): ', txt[0].strip())
            if m:
                per_sp[m.group(1)].append((ts, txt))

        # for each speaker: swap first, then mark
        for sp, arr in per_sp.items():
            swapped = adjust_swaps_before_mark(arr, sp)
            processed = process_entries(swapped, sp)

            out_file = os.path.join(main_folder, entry, f'speaker-{sp.lower()}.srt')
            with open(out_file, 'w') as f:
                for idx, (ts, lines_) in enumerate(processed, 1):
                    f.write(f"{idx}\n{ts}")
                    f.writelines(lines_)
                    f.write("\n")

        print(f"✅ Done splitting, swapping & marking {entry}")
