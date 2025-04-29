import sys
import logging
from unsloth import FastLanguageModel

# Configure basic logging
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger(__name__)

# Define the system prompt for gloss generation
SYSTEM_PROMPT = (
    "You are an expert translator for German Sign Language glossing."
    " Convert a full German sentence into its precise sign language gloss sequence.\n"
    "Guidelines:\n"
    "- Output only the final gloss sequence as a comma-separated list of uppercase gloss tokens.\n"
    "- Do not include any chain-of-thought, explanations, or intermediary reasoning in your output.\n"
    "- Do not output any tokens not part of the gloss sequence."
)


def load_model(model_dir: str):
    """
    Load the fine-tuned model and tokenizer from disk.

    Args:
        model_dir (str): Path to the directory containing the saved model and tokenizer.
    """
    logger.info("Loading fine-tuned model and tokenizer from '%s'...", model_dir)

    # Load base model in 4-bit if available
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_dir,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
        device_map="auto",
    )
    logger.info("Model and tokenizer loaded successfully.")
    return model, tokenizer


def generate_gloss(model, tokenizer, sentence: str) -> str:
    """
    Generate a sign language gloss sequence for the given sentence.

    Args:
        model: The loaded FastLanguageModel.
        tokenizer: The corresponding tokenizer.
        sentence (str): Input German sentence.

    Returns:
        str: Generated gloss sequence.
    """
    # Build the conversation messages
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": sentence},
    ]

    # Prepare input text with generation prompt
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Tokenize and move to device
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    # Generate up to a reasonable number of new tokens
    outputs = model.generate(**inputs, max_new_tokens=512)

    # Decode and post-process
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract gloss after any potential thinker tokens
    gloss = generated.split("<think>")[-1].strip()
    return gloss


def main():
    global logger
    logger = setup_logging()

    # Ensure at least one sentence is provided
    if len(sys.argv) < 2:
        logger.error(
            "Usage: python text2gloss.py \"<German sentence 1>\" [\"<German sentence 2>\" ...]"
        )
        sys.exit(1)

    # Collect all provided sentences
    sentences = sys.argv[1:]
    model_dir = "/storage/text2gloss-finetune/fine_tuned_deepseek"

    # Load model once
    model, tokenizer = load_model(model_dir)
    print("Generating gloss sequences using finetuned DeepSeek model...\n")

    # Loop over each sentence and produce gloss
    for idx, sentence in enumerate(sentences, 1):
        print(f"[{idx}/{len(sentences)}] Input sentence: {sentence}")
        gloss_sequence = generate_gloss(model, tokenizer, sentence)
        print(f"Generated gloss sequence:\n{gloss_sequence}\n")


if __name__ == "__main__":
    main()
