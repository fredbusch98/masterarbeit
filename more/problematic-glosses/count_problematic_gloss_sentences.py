"""
Counts sentences in a the Text2Gloss csv where specified problematic glosses ($PROD, $ORAL, $ALPHA) appear, 
individually or in combination. Supports checking for single glosses, pairs, or all three at once.
"""
import csv

def count_sentences_with_target(filename, target_gloss="$PROD"):
    count = 0
    with open(filename, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        # Skip header if present
        header = next(reader, None)
        for row in reader:
            # Check if any word in the row (ignoring the first column) matches exactly or starts with target_gloss + ":"
            if any(word == target_gloss or word.startswith(f"{target_gloss}") for word in row[1:]):
                count += 1
    return count

def count_sentences_with_both(filename, gloss1="$PROD", gloss2="$ORAL"):
    count = 0
    with open(filename, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        # Skip header if present
        header = next(reader, None)
        for row in reader:
            # Check if both glosses appear in the row (ignoring the first column if it's the full sentence)
            if any(word == gloss1 or word.startswith(f"{gloss1}") for word in row[1:]) and any(word == gloss2 or word.startswith(f"{gloss2}") for word in row[1:]):
                count += 1
    return count

def count_sentences_with_all_three(filename, gloss1="$PROD", gloss2="$ORAL", gloss3="$ALPHA1"):
    count = 0
    with open(filename, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        # Skip header if present
        header = next(reader, None)
        for row in reader:
            # Check if both glosses appear in the row (ignoring the first column if it's the full sentence)
            if any(word == gloss1 or word.startswith(f"{gloss1}") for word in row[1:]) and any(word == gloss2 or word.startswith(f"{gloss2}") for word in row[1:]) and any(word == gloss3 or word.startswith(f"{gloss3}") for word in row[1:]):
                count += 1
    return count

if __name__ == "__main__":
    filename = "/Volumes/IISY/DGSKorpus/dgs-text2gloss-combined.csv"
    count = count_sentences_with_target(filename)
    print(f"Count of full sentences with '$PROD': {count}")
    count2 = count_sentences_with_target(filename, target_gloss="$ORAL")
    print(f"Count of full sentences with '$ORAL': {count2}")
    count3 = count_sentences_with_target(filename, target_gloss="$ALPHA1")
    print(f"Count of full sentences with '$ALPHA1': {count3}")


    both_count1 = count_sentences_with_both(filename, gloss1="$PROD", gloss2="$ORAL")
    print(f"Count of full sentences with both '$PROD' and '$ORAL': {both_count1}")

    both_count2 = count_sentences_with_both(filename, gloss1="$PROD", gloss2="$ALPHA1")
    print(f"Count of full sentences with both '$PROD' and '$ALPHA1': {both_count2}")

    both_count3 = count_sentences_with_both(filename, gloss1="$ORAL", gloss2="$ALPHA1")
    print(f"Count of full sentences with both '$ORAL' and '$ALPHA1': {both_count3}")

    all_three_count = count_sentences_with_all_three(filename)
    print(f"Count of full sentences with all three glosses: {all_three_count}")

    count4 = count_sentences_with_target(filename, "$ALPHA")
    print(f"Count of full sentences with '$ALPHA': {count4}")