import csv
from statistics import mean

# Path to your CSV file
file_path = "/Volumes/IISY/DGSKorpus/dgs-text2gloss-split-speaker-combined.csv"

sentences = []

# Read the CSV file assuming the sentence is in the first column, ignoring the header row
with open(file_path, newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)  # Skip the header row
    for row in reader:
        if row:  # ensure the row is not empty
            sentence = row[0].strip()
            sentences.append(sentence)

# Count unique sentences
unique_sentences = set(sentences)
unique_count = len(unique_sentences)

# Calculate sentence lengths (counting words)
lengths = [len(sentence.split()) for sentence in sentences]

# Compute average sentence length
avg_length = mean(lengths) if lengths else 0

# Identify the shortest sentence(s)
min_length = min(lengths) if lengths else 0
shortest_sentences = [s for s in sentences if len(s.split()) == min_length]
shortest_example = shortest_sentences[0] if shortest_sentences else ""
# Count additional sentences that share the shortest length (excluding the printed example)
shortest_others = len(shortest_sentences) - 1

# Identify the longest sentence(s)
max_length = max(lengths) if lengths else 0
longest_sentences = [s for s in sentences if len(s.split()) == max_length]
longest_example = longest_sentences[0] if longest_sentences else ""
# Count additional sentences that share the longest length (excluding the printed example)
longest_others = len(longest_sentences) - 1

# Log the results
print(f"Total sentences: {len(sentences)}")
print(f"Unique sentences: {unique_count}")
print(f"Average sentence length (in words): {avg_length:.2f}")
print(f"Shortest sentence ({min_length} words): \"{shortest_example}\" (and {shortest_others} more with this length)")
print(f"Longest sentence ({max_length} words): \"{longest_example}\" (and {longest_others} more with this length)")
