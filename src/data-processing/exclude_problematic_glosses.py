"""
Filters SRT subtitle files of the DGS-Korpus by removing specified glosses and sentences, 
preserving sentence boundaries, and generating cleaned speaker-specific output files. 
Handles both full sentences and individual entries.
"""
import os

# Main folder path
main_folder = '/Volumes/IISY/DGSKorpus'

# Define the lists
exclude_sentence_glosses = ["$PROD", "$ORAL", "$ALPHA", "$ORAL^"]
excluded_glosses = ["$GEST", "$GEST-NM", "$GEST-OFF", "$$EXTRA-LING-MAN", "$PMS", "$UNKLAR"]

# Function to parse SRT file
def parse_srt(lines):
    entries = []
    current_entry = []
    for line in lines:
        if line.strip() == "":
            if current_entry and len(current_entry) >= 2:
                number = int(current_entry[0])
                timestamp = current_entry[1]
                text_lines = current_entry[2:]
                entries.append((number, timestamp, text_lines))
            current_entry = []
        else:
            current_entry.append(line)
    if current_entry and len(current_entry) >= 2:
        number = int(current_entry[0])
        timestamp = current_entry[1]
        text_lines = current_entry[2:]
        entries.append((number, timestamp, text_lines))
    return entries

# Function to filter an entry by removing excluded glosses
def filter_entry(entry, excluded_glosses, speaker):
    number, timestamp, text_lines = entry
    tag = f"{speaker}: "
    full_text = ' '.join(line.strip() for line in text_lines)
    if full_text.startswith(tag):
        gloss_text = full_text[len(tag):]
        glosses = gloss_text.split()
        filtered_glosses = [g for g in glosses if get_base_gloss(g) not in excluded_glosses]
        if filtered_glosses:
            new_full_text = tag + ' '.join(filtered_glosses) + '\n'
            new_text_lines = [new_full_text]
            return (number, timestamp, new_text_lines)
        else:
            return None
    return None

# Function to get the base gloss by removing specified suffixes
def get_base_gloss(gloss):
    for suffix in ['_FULL_SENTENCE', '_END_SENTENCE']:
        if gloss.endswith(suffix):
            return gloss[:-len(suffix)]
    return gloss

# Main execution
entry_folders = [f for f in os.listdir(main_folder) if f.startswith('entry_') and os.path.isdir(os.path.join(main_folder, f))]

for entry_folder in entry_folders:
    for speaker in ['a', 'b']:
        srt_path = os.path.join(main_folder, entry_folder, f'speaker-{speaker}.srt')
        if os.path.exists(srt_path):
            with open(srt_path, 'r') as f:
                lines = f.readlines()
            
            # Parse the SRT file into entries
            entries = parse_srt(lines)
            
            # Group entries into sentences and individual entries
            groups = []
            current_group = []
            for entry in entries:
                current_group.append(entry)
                if '_END_SENTENCE' in entry[2][-1]:
                    groups.append(('sentence', current_group))
                    current_group = []
            if current_group:
                for entry in current_group:
                    groups.append(('individual', [entry]))
            
            # Process each group with exclusion logic
            output_entries = []
            speaker_upper = speaker.upper()
            tag = f"{speaker_upper}: "
            for group_type, entry_list in groups:
                if group_type == 'sentence':
                    output_group_entries = []
                    for i, entry in enumerate(entry_list):
                        filtered_entry = filter_entry(entry, excluded_glosses, speaker_upper)
                        if filtered_entry is not None:
                            output_group_entries.append(filtered_entry)
                        if i == len(entry_list) - 1 and filtered_entry is None:
                            # Check if the removed last entry had '_END_SENTENCE'
                            last_text = ' '.join(line.strip() for line in entry[2])
                            if last_text.startswith(tag):
                                gloss_text = last_text[len(tag):]
                                glosses = gloss_text.split()
                                if glosses and glosses[-1].endswith('_END_SENTENCE') and output_group_entries:
                                    # Append '_END_SENTENCE' to the last gloss of the previous entry
                                    last_entry = output_group_entries[-1]
                                    number, timestamp, text_lines = last_entry
                                    full_text = ' '.join(line.strip() for line in text_lines)
                                    if full_text.startswith(tag):
                                        gloss_text = full_text[len(tag):]
                                        glosses = gloss_text.split()
                                        if glosses and not glosses[-1].endswith('_END_SENTENCE'):
                                            glosses[-1] += '_END_SENTENCE'
                                            new_full_text = tag + ' '.join(glosses) + '\n'
                                            output_group_entries[-1] = (number, timestamp, [new_full_text])
                    # Apply sentence group exclusion logic
                    if output_group_entries:
                        all_glosses = []
                        for _, _, text_lines in output_group_entries:
                            gloss_text = ' '.join(line.strip() for line in text_lines)[len(tag):]
                            all_glosses.extend(gloss_text.split())
                        if not any(get_base_gloss(g) in exclude_sentence_glosses or get_base_gloss(g).startswith('$ALPHA') for g in all_glosses):
                            output_entries.extend(output_group_entries)
                else:  # individual
                    filtered_entry = filter_entry(entry_list[0], excluded_glosses, speaker_upper)
                    if filtered_entry is not None:
                        all_glosses = []
                        gloss_text = ' '.join(line.strip() for line in filtered_entry[2])[len(tag):]
                        all_glosses.extend(gloss_text.split())
                        if not any(g in exclude_sentence_glosses for g in all_glosses):
                            output_entries.append(filtered_entry)
            
            # Write the filtered entries to a new SRT file
            output_path = os.path.join(main_folder, entry_folder, f'speaker-{speaker}-final.srt')
            with open(output_path, 'w') as f:
                for i, (number, timestamp, text_lines) in enumerate(output_entries, start=1):
                    f.write(f"{i}\n")
                    f.write(timestamp)
                    for line in text_lines:
                        f.write(line)
                    f.write("\n")
            print(f"✅ Processed {srt_path} -> {output_path}")
        else:
            print(f"⚠️ Missing {srt_path}")
print("🎉 Filtering complete for all folders!")