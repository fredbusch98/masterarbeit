import csv

max_len = 0
with open('dataset.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        gloss_str = row['glosses']
        length = len(gloss_str)
        if length > max_len:
            max_len = length

print("Max character length of glosses string:", max_len)
