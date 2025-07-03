import os
import csv
import math
from statistics import mean, median, stdev
from collections import Counter
from itertools import islice

# Path to your CSV file and results directory
txt_file_path = "/Users/frederikbusch/Developer/master-arbeit/text2gloss2pose/bht-cluster/deepseek-finetuning/text2gloss_data/bt-2/train.csv"
results_dir = "./results/"

# Ensure results directory exists
os.makedirs(results_dir, exist_ok=True)

# Parameters
ngram_sizes = [1, 2, 3]            # Unigrams, bigrams, trigrams
vocab_growth_interval = 100        # Record vocab growth every N tokens

# Containers
sentences = []
gloss_cells = []       # Raw gloss strings from each row

# --- Read CSV ---
with open(txt_file_path, newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    header = next(reader)
    gloss_idx = header.index('gloss') if 'gloss' in header else 1
    for row in reader:
        if not row:
            continue
        sentences.append(row[0].strip())
        # split gloss cell by commas
        cell = row[gloss_idx].strip().strip('"')
        gloss_cells.append(cell)

# --- Flatten lists ---
all_words = [w for s in sentences for w in s.split()]
all_glosses = [g for cell in gloss_cells for g in cell.split(',') if g]

# --- Core sentence stats ---
total_sentences = len(sentences)
unique_sentences = len(set(sentences))
lengths = [len(s.split()) for s in sentences]
avg_len = mean(lengths)
med_len = median(lengths)
std_len = stdev(lengths) if total_sentences > 1 else 0
min_len, max_len = min(lengths), max(lengths)

# --- Word-level stats ---
word_counter = Counter(all_words)
unique_words = len(word_counter)
total_words = len(all_words)

# --- Gloss-level stats ---
gloss_counter = Counter(all_glosses)
unique_glosses = len(gloss_counter)
total_glosses = len(all_glosses)

# --- N-gram extraction ---
ngram_counters = {}
for n in ngram_sizes:
    if n == 1:
        ngram_counters[n] = Counter(all_words)
    else:
        ngram_counters[n] = Counter(zip(*(islice(all_words, i, None) for i in range(n))))

# --- Lexical diversity & hapaxes ---
thapax = sum(1 for w, c in word_counter.items() if c == 1)
ttr_per_sent = [len(set(s.split()))/len(s.split()) for s in sentences]
avg_sent_ttr = mean(ttr_per_sent)

# --- Detailed gloss metrics ---
total_occurrences = total_glosses
avg_occurrence = total_occurrences / unique_glosses if unique_glosses else 0
median_occurrence = median(list(gloss_counter.values())) if unique_glosses else 0
# Frequency buckets
gloss_buckets = {
    '>1000': sum(1 for c in gloss_counter.values() if c > 1000),
    '100-1000': sum(1 for c in gloss_counter.values() if 100 <= c <= 1000),
    '<100': sum(1 for c in gloss_counter.values() if c < 100),
    '<=average_occurrence': sum(1 for c in gloss_counter.values() if c <= math.ceil(avg_occurrence)),
    '<=median_occurrence': sum(1 for c in gloss_counter.values() if c <= median_occurrence),
    '<=10': sum(1 for c in gloss_counter.values() if c <= 10),
    '<=2': sum(1 for c in gloss_counter.values() if c <= 2),
    '==1': sum(1 for c in gloss_counter.values() if c == 1)
}

# --- Export CSV helpers ---
def write_counter(counter, filename, header):
    path = os.path.join(results_dir, filename)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(header)
        for item, cnt in counter.most_common():
            key = ' '.join(item) if isinstance(item, tuple) else item
            w.writerow([key, cnt])
    print(f"Saved {filename}")

# Export frequency files
write_counter(word_counter, 'word_frequencies.csv', ['word', 'count'])
write_counter(gloss_counter, 'gloss_frequencies.csv', ['gloss', 'count'])
for n in ngram_sizes:
    write_counter(ngram_counters[n], f'{n}gram_frequencies.csv', [f'{n}-gram', 'count'])
# Vocab growth
vg_path = os.path.join(results_dir, 'vocab_growth.csv')
with open(vg_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['tokens_processed', 'unique_types'])
print(f"Saved vocab_growth.csv")
# Gloss detailed metrics
gdm_path = os.path.join(results_dir, 'gloss_detailed_metrics.csv')
with open(gdm_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['metric', 'value'])
    writer.writerow(['total_occurrences', total_occurrences])
    writer.writerow(['unique_glosses', unique_glosses])
    writer.writerow(['avg_occurrence', round(avg_occurrence,2)])
    writer.writerow(['median_occurrence', median_occurrence])
    for key, val in gloss_buckets.items():
        writer.writerow([key, val])
print(f"Saved gloss_detailed_metrics.csv")

# --- Print summary ---
print(f"Total sentences: {total_sentences}")
print(f"Unique sentences: {unique_sentences}")
print(f"Average sentence length: {avg_len:.2f}, median: {med_len}, std: {std_len:.2f}")
print(f"Total words: {total_words} ({unique_words} unique)")
print(f"Total glosses: {total_glosses} ({unique_glosses} unique)")
print(f"Hapax words: {thapax}")
print(f"Average sentence TTR: {avg_sent_ttr:.3f}")

print("\nTop 10 words:")
for w,c in word_counter.most_common(10): print(f"  {w}: {c}")

print("\nTop 10 glosses:")
for g,c in gloss_counter.most_common(10): print(f"  {g}: {c}")

for n in ngram_sizes:
    print(f"\nTop 10 {n}-grams:")
    for ng,c in ngram_counters[n].most_common(10):
        label = ' '.join(ng) if isinstance(ng, tuple) else ng
        print(f"  {label}: {c}")

print("\nDetailed gloss buckets:")
print(f"  average_occurrence: {avg_occurrence:.2f}")
print(f"  median_occurrence:  {median_occurrence}")
for k, v in gloss_buckets.items():
    print(f"  {k}: {v}")
