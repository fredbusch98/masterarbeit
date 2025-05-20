import os
import json
import itertools

# Configuration
BASE_DIR = '/Volumes/IISY/DGSKorpus'
GLOSSES_FILE = 'glosses.txt'
OUTPUT_DIR = 'output'
BATCH_SIZE = 1000  # total pose sequences per intermediate file
PER_GLOSS_LIMIT = 100  # max per gloss


def load_gloss_list(glosses_file):
    """
    Load glosses from a text file, stripping whitespace and ignoring empty lines.
    """
    with open(glosses_file, 'r', encoding='utf-8') as f:
        glosses = [line.strip() for line in f if line.strip()]
    return glosses


def find_entry_dirs(base_dir):
    """
    Find all subdirectories in base_dir matching 'entry_*'.
    """
    return sorted([
        os.path.join(base_dir, name)
        for name in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, name)) and name.startswith('entry_')
    ])


def process_entries(gloss_list):
    """
    Iterate through all entry directories, collect up to PER_GLOSS_LIMIT pose sequences per gloss,
    dumping intermediate JSON files each BATCH_SIZE total collected sequences.
    """
    counts = {gloss: 0 for gloss in gloss_list}
    collected = []
    batch_index = 0

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    entry_dirs = find_entry_dirs(BASE_DIR)
    total_entries = len(entry_dirs)
    print(f'📁 Found {total_entries} entries to process.')

    for idx, entry_dir in enumerate(entry_dirs, 1):
        percent = (idx / total_entries) * 100
        print(f'🔄 [{idx}/{total_entries}] Processing {os.path.basename(entry_dir)} ({percent:.1f}%)')

        json_path = os.path.join(entry_dir, 'gloss2pose.json')
        if not os.path.isfile(json_path):
            print(f'⚠️  Missing gloss2pose.json in {entry_dir}')
            continue

        with open(json_path, 'r', encoding='utf-8') as jf:
            try:
                data = json.load(jf)
            except json.JSONDecodeError:
                print(f'❌ JSON decoding failed in {json_path}')
                continue

        for obj in data.get('data', []):
            gloss = obj.get('gloss')
            if gloss in counts and counts[gloss] < PER_GLOSS_LIMIT:
                collected.append(obj)
                counts[gloss] += 1

                if len(collected) >= BATCH_SIZE:
                    save_batch(collected, batch_index)
                    batch_index += 1
                    collected = []

    if collected:
        save_batch(collected, batch_index)

    print(f'✅ Done processing all entries. Total batches saved: {batch_index + (1 if collected else 0)}')


def save_batch(batch_data, index):
    """
    Save a batch of collected data to an output JSON file.
    """
    filename = f'batch_{index:03d}.json'
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, 'w', encoding='utf-8') as out_f:
        json.dump({'data': batch_data}, out_f, ensure_ascii=False, indent=2)
    print(f'💾 Saved {len(batch_data)} items to {path}')


if __name__ == '__main__':
    print('🚀 Starting gloss2pose collector...\n')

    glosses = load_gloss_list(GLOSSES_FILE)
    print(f'📜 Loaded {len(glosses)} glosses to collect.\n')

    process_entries(glosses)

    print('\n🎉 Processing complete.')
