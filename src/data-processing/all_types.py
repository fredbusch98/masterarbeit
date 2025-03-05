import csv

def load_gloss_types(csv_path):
    """
    Load gloss types from a CSV file.

    Parameters:
        csv_path (str): Path to the CSV file.

    Returns:
        set: A set of gloss types.
    """
    gloss_types = set()
    try:
        with open(csv_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            next(reader, None)  # Skip the header row
            for row in reader:
                gloss_types.add(row[0].strip())
        print(f"[INFO] Loaded {len(gloss_types)} gloss types from {csv_path}")
    except Exception as e:
        print(f"[ERROR] Could not load gloss types from {csv_path}: {e}")
    return gloss_types