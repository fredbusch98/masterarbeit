import os
import re

def is_full_sentence(text):
    """
    Returns True if the text (after removing the speaker tag) 
    starts with a capital letter and contains multiple words.
    """
    # Remove the speaker tag (A:, B:, or C:) at the beginning
    match = re.match(r'^[ABC]:\s*(.*)', text)
    if match:
        content = match.group(1).strip()
    else:
        content = text.strip()
    
    # Check if content is empty or doesn't start with a capital letter
    if not content or not content[0].isupper():
        return False

    # Check if the content has more than one word
    words = content.split()
    if len(words) < 2:
        return False

    return True

def main():
    base_dir = '/Volumes/IISY/DGSKorpus/'
    non_pipe_count = 0
    pipe_count = 0

    # Loop over subfolders named like entry_*
    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        if os.path.isdir(folder_path) and folder.startswith("entry_"):
            file_path = os.path.join(folder_path, "transcript.srt")
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Split entries by blank lines (SRT entries are separated by an empty line)
                entries = content.strip().split('\n\n')
                for entry in entries:
                    lines = entry.strip().splitlines()
                    # We expect at least 3 lines per entry:
                    # 1: entry number, 2: timestamp, 3: dialogue line
                    if len(lines) >= 3:
                        dialogue = lines[2].strip()
                        dialogue = dialogue[2:].strip()
                        dialogue.lstrip("||")
                        if not is_full_sentence(dialogue):
                            if "||" in dialogue and dialogue.isupper():
                                pipe_count += 1
                            else:
                                non_pipe_count += 1

    total = non_pipe_count + pipe_count
    percentage = (pipe_count / total * 100) if total > 0 else 0

    print("Single gloss entries: {}".format(non_pipe_count))
    print("Dual gloss entries: {}".format(pipe_count))
    print("Total gloss entries: {}".format(total))
    print("Percentage of dual gloss entries: {:.2f}%".format(percentage))

if __name__ == "__main__":
    main()
