import logging
import pandas as pd
from transformers import pipeline
import argparse
import sys
from pathlib import Path
from collections import Counter
import torch

# Global default number of paraphrases per sentence
global_num_paraphrases = 1
max_attempts = 5

# Configure logging
log_file = Path(__file__).with_name("data_augment.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file, mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stderr),  # keep progress on stderr as well
    ],
)
logger = logging.getLogger(__name__)

def is_degenerate(s: str, min_words: int = 5, max_ngram: int = 10, repetition_threshold: float = 0.5) -> bool:
    words = s.split()
    if not words:
        return False
    if len(words) < min_words:
        return False  # Allow short valid sentences

    total_words = len(words)

    for n in range(1, min(max_ngram, total_words) + 1):
        ngrams = [" ".join(words[i:i+n]) for i in range(len(words) - n + 1)]
        if not ngrams:
            continue
        counts = Counter(ngrams)
        most_common, freq = counts.most_common(1)[0]
        if freq > 1:
            proportion = (freq * n) / total_words
            if proportion > repetition_threshold:
                return True

    return False

def back_translate(sentence: str,
                   translator_de_en,
                   translator_en_de,
                   exclude: set,
                   max_attempts: int = max_attempts) -> str:
    """Perform back‑translation on a single sentence, avoiding any in `exclude` set."""
    for attempt in range(max_attempts):
        # Translate to English with sampling for variation
        en = translator_de_en(
            sentence,
            max_length=512,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            no_repeat_ngram_size=3,
            repetition_penalty=2.0,
            early_stopping=True
        )[0]['translation_text'].strip()
        logger.debug("DE→EN: %s → %s", sentence, en)

        # Back‑translate to German (greedy with constraints)
        back = translator_en_de(
            en,
            max_length=512,
            no_repeat_ngram_size=3,
            repetition_penalty=2.0,
            early_stopping=True
        )[0]['translation_text'].strip()
        logger.debug("EN→DE: %s → %s", en, back)

        candidate = back
        if candidate and candidate not in exclude and not is_degenerate(candidate):
            return candidate
    return None

def main(input_csv: str, output_csv: str, n_paraphrases: int):
    logger.info("Loading dataset from %s", input_csv)
    df = pd.read_csv(input_csv)
    total = len(df)
    device = 0 if torch.cuda.is_available() else -1

    # Initialize translation pipelines with error handling
    try:
        logger.info("Loading translation pipelines …")
        translator_de_en = pipeline(
            'translation_de_to_en',
            model='Helsinki-NLP/opus-mt-de-en',
            device=device
        )
        translator_en_de = pipeline(
            'translation_en_to_de',
            model='Helsinki-NLP/opus-mt-en-de',
            device=device
        )
    except ImportError:
        logger.exception(
            "Missing dependency for translation pipelines. Install with `pip install transformers sentencepiece`. Exiting."
        )
        sys.exit(1)

    augmented_rows = []
    augmented_count = 0
    problematic_count = 0

    # Process each sentence with progress logging
    for idx, row in enumerate(df.itertuples(index=False), start=1):
        orig = row.full_sentence.strip()
        gloss = row.glosses
        paraphrases = set()
        exclude = {orig}

        # Generate up to n_paraphrases unique paraphrases
        for _ in range(n_paraphrases):
            pt = back_translate(orig, translator_de_en, translator_en_de, exclude)
            if pt:
                paraphrases.add(pt)
                exclude.add(pt)
                augmented_rows.append({'full_sentence': pt, 'glosses': gloss})
                augmented_count += 1
            else:
                break

        if paraphrases:
            status = f"Augmented {len(paraphrases)}"
        else:
            status = "Skipped"
            problematic_count += 1

        percent = idx / total * 100
        logger.info("%s %d/%d (%.1f%%)", status, idx, total, percent)

    # Combine original and augmented
    df_augmented = pd.concat([df, pd.DataFrame(augmented_rows)], ignore_index=True)
    df_augmented.to_csv(output_csv, index=False)

    # Report results
    logger.info("Total original sentences: %d", total)
    logger.info(
        "Successfully augmented sentences: %d (total paraphrases: %d)",
        total - problematic_count,
        augmented_count
    )
    logger.info(
        "Problematic (no new paraphrase after %d attempts each): %d",
        max_attempts,
        problematic_count
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Back-translation augmentation for German sentence dataset'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='dataset.csv',
        help='Path to the input CSV file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='augmented_dataset.csv',
        help='Path to the output CSV file'
    )
    parser.add_argument(
        '--num-paraphrases',
        type=int,
        default=global_num_paraphrases,
        help='Number of paraphrases to generate per sentence'
    )
    args = parser.parse_args()
    main(args.input, args.output, args.num_paraphrases)
