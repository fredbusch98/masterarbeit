"""
Script to perform back-translation data augmentation on a German sentence dataset.
Each sentence is translated German->English->German to produce a paraphrase.
If the back-translated sentence is identical to the original, it retries up to 3 times 
before giving up on that sentence. Glosses are retained unchanged.
Provides progress logs with percentage and emojis.
"""
import pandas as pd
from transformers import pipeline
import argparse
import sys

def back_translate(sentence: str,
                   translator_de_en,
                   translator_en_de,
                   max_attempts: int = 3) -> str:
    """
    Perform back-translation on a single sentence.

    Args:
        sentence: Original German sentence.
        translator_de_en: HuggingFace pipeline for German->English.
        translator_en_de: HuggingFace pipeline for English->German.
        max_attempts: Number of retry attempts if paraphrase equals original.

    Returns:
        A paraphrased German sentence, or None if all attempts produce the original.
    """
    for attempt in range(max_attempts):
        # Translate to English with sampling for variation
        en = translator_de_en(
            sentence,
            max_length=512,
            do_sample=True,
            top_k=50,
            top_p=0.95
        )[0]['translation_text']
        # Back-translate to German (greedy)
        back = translator_en_de(
            en,
            max_length=512
        )[0]['translation_text']
        # If different from original, accept
        if back.strip() != sentence.strip():
            return back
    # All attempts yielded identical sentence
    return None

def main(input_csv: str, output_csv: str):
    # Load dataset
    df = pd.read_csv(input_csv)
    total = len(df)

    # Initialize translation pipelines with error handling
    try:
        translator_de_en = pipeline(
            'translation_de_to_en',
            model='Helsinki-NLP/opus-mt-de-en'
        )
        translator_en_de = pipeline(
            'translation_en_to_de',
            model='Helsinki-NLP/opus-mt-en-de'
        )
    except ImportError as e:
        print("Error: Missing dependency for translation pipelines.", file=sys.stderr)
        print("Please install required packages with `pip install transformers sentencepiece`.", file=sys.stderr)
        sys.exit(1)

    augmented_rows = []
    augmented_count = 0
    problematic_count = 0

    # Process each sentence with progress logging
    for idx, row in enumerate(df.itertuples(index=False), start=1):
        orig = row.full_sentence
        gloss = row.glosses
        paraphrase = back_translate(orig, translator_de_en, translator_en_de)

        if paraphrase:
            augmented_rows.append({'full_sentence': paraphrase, 'glosses': gloss})
            augmented_count += 1
            emoji = "🌟"
            status = "Augmented"
        else:
            problematic_count += 1
            emoji = "❌"
            status = "Skipped"

        percent = idx / total * 100
        # Log to stderr to separate from CSV output
        print(f"{emoji} Progress {idx}/{total} ({percent:.1f}%): {status}", file=sys.stderr)

    # Combine original and augmented
    df_augmented = pd.concat([df, pd.DataFrame(augmented_rows)], ignore_index=True)
    df_augmented.to_csv(output_csv, index=False)

    # Report results
    print(f"🎉 Total original sentences: {total}", file=sys.stderr)
    print(f"🌟 Successfully augmented: {augmented_count}", file=sys.stderr)
    print(f"⚠️ Problematic (no new paraphrase after 3 attempts): {problematic_count}", file=sys.stderr)

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
    args = parser.parse_args()
    main(args.input, args.output)
