import os
import csv
import math
from statistics import mean, median, stdev
from collections import Counter
from itertools import islice
import string
import re

# Path to your CSV file and results directory
txt_file_path = "/Users/frederikbusch/Developer/master-arbeit/text2gloss2pose/bht-cluster/deepseek-finetuning/text2gloss_data/statistics/bt2.csv"
# txt_file_path = "combined.csv"
dataset = "DGS-BT2"
results_dir = f"./dataset-stats/{dataset}"

# Ensure results directory exists
os.makedirs(results_dir, exist_ok=True)

# Parameters
ngram_sizes = [1, 2, 3, 4]         # Unigrams, bigrams, trigrams, quadgrams

# Containers
sentences = []
gloss_cells = []       # Raw gloss strings from each row

translator_table = str.maketrans("", "", string.punctuation)
def normalize(token: str) -> str:
    """Remove *all* punctuation (even inside the token) and lowercase."""
    if not token:
        return ""
    token_clean = token.translate(translator_table)
    token_clean = re.sub(rf"[{re.escape(string.punctuation)}]", "", token_clean)
    return token_clean.lower().strip()

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
all_words_raw = [w for s in sentences for w in s.split()]
all_words_norm = [t for t in (normalize(w) for w in all_words_raw) if t]
all_glosses = [g for cell in gloss_cells for g in cell.split(',') if g]

# --- Core sentence stats ---
total_sentences = len(sentences)
unique_sentences = len(set(sentences))
lengths = [len(s.split()) for s in sentences]
length_counter = Counter(lengths)
def write_counter_sorted_by_key(counter, filename, header):
    path = os.path.join(results_dir, filename)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(header)
        for length in sorted(counter):         # iterate sorted keys
            w.writerow([length, counter[length]])
    print(f"Saved {filename}")

write_counter_sorted_by_key(length_counter, 'sentence_lengths.csv', ['length','count'])
avg_len = mean(lengths)
med_len = median(lengths)
std_len = stdev(lengths) if total_sentences > 1 else 0
min_len, max_len = min(lengths), max(lengths)

# Find example sentences and counts for min/max lengths
short_sents = [s for s in sentences if len(s.split()) == min_len]
long_sents  = [s for s in sentences if len(s.split()) == max_len]
n_short = len(short_sents)
n_long  = len(long_sents)
example_short = short_sents[0] if short_sents else ''
example_long  = long_sents[0]  if long_sents else ''

# --- Word-level stats ---
word_counter = Counter(all_words_raw)
unique_words = len(word_counter)
total_words = len(all_words_raw)
word_counter_norm = Counter(all_words_norm)
unique_words_norm = len(word_counter_norm)
total_words_norm = len(all_words_norm)

# --- Gloss-level stats ---
gloss_counter = Counter(all_glosses)
unique_glosses = len(gloss_counter)
total_glosses = len(all_glosses)

# --- N-gram extraction ---
ngram_counters = {}
for n in ngram_sizes:
    if n == 1:
        ngram_counters[n] = Counter(all_words_raw)
    else:
        ngram_counters[n] = Counter(zip(*(islice(all_words_raw, i, None) for i in range(n))))

ngram_counters_norm = {}
for n in ngram_sizes:
    if n == 1:
        ngram_counters_norm[n] = Counter(all_words_norm)
    else:
        ngram_counters_norm[n] = Counter(zip(*(islice(all_words_norm, i, None) for i in range(n))))

# --- Lexical diversity & hapaxes ---nt
hapax_raw = sum(1 for w, c in word_counter.items() if c == 1)
hapax_norm = sum(1 for w, c in word_counter_norm.items() if c == 1)
# TTR on normalized tokens per sentence
norm_ttr_per_sent = [len(set(normalize(w) for w in s.split() if normalize(w))) / \
                    len([normalize(w) for w in s.split() if normalize(w)])
                    for s in sentences]
avg_norm_sent_ttr = mean(norm_ttr_per_sent)
# Guiraud's R on normalized tokens
guiraud_r = unique_words_norm / math.sqrt(total_words_norm) if total_words_norm > 0 else 0

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
def write_counter(counter, filename, header, relative):
    path = os.path.join(results_dir, filename)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(header)
        for item, cnt in counter.most_common():
            key = ' '.join(item) if isinstance(item, tuple) else item
            percent = (cnt / relative) * 100 if relative else 0
            w.writerow([key, cnt, percent])
    print(f"Saved {filename}")

# Export frequency files
write_counter(word_counter, 'word_frequencies_raw.csv', ['word', 'count', 'percentage'], total_words)
write_counter(word_counter_norm, 'word_frequencies_norm.csv', ['word', 'count', 'percentage'], total_words_norm)
write_counter(gloss_counter, 'gloss_frequencies.csv', ['gloss', 'count', 'percentage'], total_glosses)
for n in ngram_sizes:
    write_counter(ngram_counters[n], f'{n}gram_frequencies_raw.csv', [f'{n}-gram', 'count', 'percentage'], total_words)
    write_counter(ngram_counters_norm[n], f'{n}gram_frequencies_norm.csv', [f'{n}-gram', 'count', 'percentage'], total_words_norm)
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
print(f"Shortest sentence ({min_len} words): \"{example_short}\" (and {n_short-1} more with this length)")
print(f"Longest sentence ({max_len} words): \"{example_long}\" (and {n_long-1} more with this length)")
print(f"Total words (raw): {total_words} ({unique_words} unique)")
print(f"Total words (normalized): {total_words_norm} ({unique_words_norm} unique)")
print(f"Total glosses: {total_glosses} ({unique_glosses} unique)")
print(f"Hapax words (raw): {hapax_raw}")
print(f"Hapax words (normalized): {hapax_norm}")
print(f"Average sentence TTR: {avg_norm_sent_ttr:.3f}")
print(f"Guiraud's R: {guiraud_r:.3f}")

print("\nTop 10 glosses:")
for g,c in gloss_counter.most_common(10): 
    percent = (c / total_glosses) * 100 if total_glosses else 0
    print(f"  {g}: {c} ({percent:.2f}%)")

for n in ngram_sizes:
    print(f"\nTop 10 {n}-grams (raw):")
    for ng,c in ngram_counters[n].most_common(10):
        label = ' '.join(ng) if isinstance(ng, tuple) else ng
        percent = (c / total_words) * 100 if total_words else 0
        print(f"  {label} - {c} ({percent:.4f}%)")

for n in ngram_sizes:
    print(f"\nTop 10 {n}-grams (normalized):")
    for ng,c in ngram_counters_norm[n].most_common(10):
        label = ' '.join(ng) if isinstance(ng, tuple) else ng
        percent = (c / total_words_norm) * 100 if total_words_norm else 0
        print(f"  {label} - {c} ({percent:.4f}%)")

print("\nDetailed gloss buckets:")
print(f"  average_occurrence: {avg_occurrence:.2f}")
print(f"  median_occurrence:  {median_occurrence}")
for k, v in gloss_buckets.items():
    percent = (v / unique_glosses) * 100 if unique_glosses else 0
    print(f"  {k}: {v} ({percent:.2f}%)")
