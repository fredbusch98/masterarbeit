import re
import csv
import argparse

def parse_log(file_path):
    """
    Parse the gloss generation log to extract generated glosses.
    """
    glosses = []
    with open(file_path, encoding='utf-8') as f:
        content = f.read()

    # Split entries by the separator line
    entries = content.split('----------------------------------------')
    for entry in entries:
        # Search for Generated Gloss
        gloss_match = re.search(r'Generated Gloss:\s*(.*)', entry)
        if gloss_match:
            gloss = gloss_match.group(1).strip()
            glosses.append([gloss])
    return glosses

def write_csv(glosses, out_path):
    """
    Write the extracted glosses to a CSV file.
    """
    with open(out_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Generated Gloss'])
        writer.writerows(glosses)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract generated glosses from a log and write to CSV.')
    parser.add_argument('log_file', help='Path to the gloss generation log file')
    parser.add_argument('--output', '-o', default='glosses.csv', help='Output CSV file path')
    args = parser.parse_args()

    glosses = parse_log(args.log_file)
    write_csv(glosses, args.output)
    print(f"Extracted {len(glosses)} glosses and wrote to {args.output}")