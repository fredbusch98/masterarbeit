import re

# Paths to your files
input_file = "/Volumes/IISY/DGSKorpus/only-lost-glosses.txt"
output_file = "/Volumes/IISY/DGSKorpus/only-lost-glosses-processed.txt"

# Lists to hold lines with '-' and the processed lines.
dash_lines = []
processed_lines = []

with open(input_file, "r", encoding="utf-8") as infile:
    for line in infile:
        line = line.strip()
        # Skip empty lines
        if not line:
            continue
        
        # 1. Skip lines that start with '$'
        if line.startswith('$'):
            continue
        
        # 2. If the line contains '-', add it to dash_lines (it will be unmodified)
        if '-' in line:
            dash_lines.append(line)
            continue
        
        # 3. For all other lines, find the first number and split the line accordingly.
        match = re.search(r'\d', line)
        if match:
            index = match.start()
            part_before = line[:index]
            new_line = f"{line} - {part_before.lower()}"
        else:
            new_line = line
        
        processed_lines.append(new_line)

# Write the dash_lines at the top, followed by the processed lines.
with open(output_file, "w", encoding="utf-8") as outfile:
    for dline in dash_lines:
        outfile.write(dline + "\n")
    for pline in processed_lines:
        outfile.write(pline + "\n")

print("Processing complete. Check the output file:", output_file)
