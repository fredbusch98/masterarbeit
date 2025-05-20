import csv
import argparse
from collections import Counter


def load_unique_glosses(unique_file_path):
    """
    Load unique glosses from a CSV file into a set.
    """
    unique = set()
    with open(unique_file_path, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            gloss = row.get('gloss') or row.get('Generated Gloss')
            if gloss:
                unique.add(gloss.strip())
    return unique


def load_gloss_sequences(sequences_file_path):
    """
    Load gloss sequences from a CSV file into a list of lists.
    Each sequence is a comma-separated string of glosses.
    """
    sequences = []
    with open(sequences_file_path, encoding='utf-8') as f:
        reader = csv.reader(f)
        headers = next(reader, None)
        for row in reader:
            if not row:
                continue
            # Assume first column contains the sequence
            sequence_str = row[0]
            # Split on commas and strip whitespace
            glosses = [g.strip() for g in sequence_str.split(',') if g.strip()]
            sequences.append(glosses)
    return sequences


def count_missing_glosses(sequences, unique_set):
    """
    Count gloss occurrences that are not in the unique_set.
    """
    counter = Counter()
    for glosses in sequences:
        for gloss in glosses:
            if gloss not in unique_set:
                counter[gloss] += 1
    return counter


def write_counts(counter, output_file_path):
    """
    Write the missing gloss counts to a CSV file.
    """
    with open(output_file_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Gloss', 'Count'])
        for gloss, count in counter.most_common():
            writer.writerow([gloss, count])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Count gloss occurrences not in a list of unique glosses.'
    )
    parser.add_argument(
        '--sequences', '-s', required=True,
        help='CSV file with gloss sequences (one sequence per row).'
    )
    parser.add_argument(
        '--unique', '-u', required=True,
        help='CSV file with unique glosses (column "gloss").'
    )
    parser.add_argument(
        '--output', '-o', default='missing_gloss_counts.csv',
        help='Output CSV file for missing gloss counts.'
    )
    args = parser.parse_args()

    unique_set = load_unique_glosses(args.unique)
    sequences = load_gloss_sequences(args.sequences)
    counter = count_missing_glosses(sequences, unique_set)
    write_counts(counter, args.output)

    print(f"Found {sum(counter.values())} missing gloss occurrences across {len(counter)} unique glosses.")
