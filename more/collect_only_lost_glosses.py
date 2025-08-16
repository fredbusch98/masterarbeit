"""
Extracts all entries marked as 'Only Lost' from a CSV file and writes their 'Gloss' values 
to a plain text file, one per line.
"""
import csv

# Define input and output file paths
input_csv = '/Volumes/IISY/DGSKorpus/lost-gloss-details.csv'
output_txt = '/Volumes/IISY/DGSKorpus/only-lost-glosses.txt'

with open(input_csv, mode='r', encoding='utf-8') as infile:
    reader = csv.DictReader(infile)
    with open(output_txt, mode='w', encoding='utf-8') as outfile:
        for row in reader:
            # Check if the "Only Lost" column is True (ignoring case and extra spaces)
            if row.get('Only Lost', '').strip().lower() == 'true':
                # Write only the Gloss column value to the output file
                outfile.write(row.get('Gloss', '').strip() + '\n')
