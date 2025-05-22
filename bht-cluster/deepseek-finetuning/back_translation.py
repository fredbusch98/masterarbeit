import logging
import pandas as pd
from transformers import pipeline
import argparse
import sys
from pathlib import Path

# Global default number of paraphrases per sentence
global_num_paraphrases = 2
max_attempts = 10

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
            top_p=0.95
        )[0]['translation_text']
        logger.debug("DE→EN: %s → %s", sentence, en)

        # Back‑translate to German (greedy)
        back = translator_en_de(
            en,
            max_length=512
        )[0]['translation_text']
        logger.debug("EN→DE: %s → %s", en, back)

        # If new (not in exclude), accept
        candidate = back.strip()
        if candidate and candidate not in exclude:
            return candidate
    # All attempts yielded either identical or excluded sentence
    return None

def main(input_csv: str, output_csv: str, n_paraphrases: int):
    logger.info("Loading dataset from %s", input_csv)
    df = pd.read_csv(input_csv)
    total = len(df)

    # Initialize translation pipelines with error handling
    try:
        logger.info("Loading translation pipelines …")
        translator_de_en = pipeline(
            'translation_de_to_en',
            model='Helsinki-NLP/opus-mt-de-en'
        )
        translator_en_de = pipeline(
            'translation_en_to_de',
            model='Helsinki-NLP/opus-mt-en-de'
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
                # No new paraphrase after retries
                break

        # Logging status
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
