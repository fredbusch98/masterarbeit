import csv

# This script assumes, that the only-lost-glosses.txt file has been properly preprocessed! Also manually because some entries can NOT be preprocessed using the preprocess_only_lost_glosses script and instead need to be preprocessed by a human!

# Define the input and output file paths.
input_file = '/Volumes/IISY/DGSKorpus/only-lost-glosses-processed.txt'
output_file = '/Volumes/IISY/DGSKorpus/only-lost-glosses-output.csv'

with open(input_file, 'r', encoding='utf-8') as infile, \
     open(output_file, 'w', newline='', encoding='utf-8') as outfile:
    
    csv_writer = csv.writer(outfile)
    
    # Process each line in the input file.
    for line in infile:
        # Split the line using the pipe character and remove extra spaces.
        parts = [part.strip() for part in line.strip().split('|')]
        
        # Check that we have at least two parts.
        if len(parts) >= 2:
            # The first part is the gloss.
            gloss = parts[0]
            # The second part is the full_sentence.
            full_sentence = parts[1]
            
            # Write the result to the CSV file as: full_sentence, gloss.
            csv_writer.writerow([full_sentence, gloss])
