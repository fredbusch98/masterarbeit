import sys
import logging
import threading
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from sacrebleu import corpus_bleu
import os

# Ensure the log directory exists
log_dir = "/storage/text2gloss-finetune"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "process.log")

# Configure logging to write to both a file and stdout if desired
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Define the output file for generated glosses
gloss_output_file = os.path.join(log_dir, "gloss_generation_log.txt")

def main():
    logger.info("========== STARTING PROCESS ==========")
    # Step 1: Load and preprocess the dataset
    logger.info("Loading dataset from CSV...")
    try:
        df = pd.read_csv('/storage/text2gloss-finetune/dataset.csv', encoding='utf-8')
    except Exception as e:
        logger.error("Error reading CSV: %s", e)
        sys.exit(1)
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
    logger.info("Dataset loaded. Train shape: %d, Validation shape: %d", len(train_df), len(val_df))

    def format_conversations(df):
        data = []
        for row in df.itertuples():
            messages = [
                {"role": "system",
                 "content": (
                     "You are an expert translator for German Sign Language glossing. Your task is to convert a full German sentence "
                     "into its precise sign language gloss sequence.\n"
                     "Guidelines:\n"
                     "- Output only the final gloss sequence as a comma-separated list of uppercase gloss tokens.\n"
                     "- Do not include any chain-of-thought, explanations, or intermediary reasoning in your output.\n"
                     "- Do not output any tokens not part of the gloss sequence."
                )},
                {"role": "user", "content": row.full_sentence},
                {"role": "assistant", "content": row.glosses},
            ]
            data.append({"messages": messages})
        return data

    logger.info("Formatting conversations for training and validation...")
    train_data = format_conversations(train_df)
    val_data = format_conversations(val_df)

    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    logger.info("Dataset formatting complete. Train dataset size: %d, Validation dataset size: %d", len(train_dataset), len(val_dataset))

    # Step 2: Load the model and tokenizer
    logger.info("Loading model and tokenizer: %s", "unsloth/DeepSeek-R1-Distill-Llama-8B")
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/DeepSeek-R1-Distill-Llama-8B",
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
            device_map="auto",
        )
        logger.info("Model and tokenizer loaded successfully.")
    except Exception as e:
        logger.error("Error loading model/tokenizer: %s", e)
        sys.exit(1)

    def compute_max_gloss_tokens(df, tokenizer):
        max_tokens = 0
        for gloss in df['glosses']:
            # Tokenize the gloss sequence
            tokens = tokenizer(gloss, add_special_tokens=False, return_tensors="pt")['input_ids']
            token_count = tokens.shape[1]  # Number of tokens
            max_tokens = max(max_tokens, token_count)
        return max_tokens

    # After loading dataset
    logger.info("Computing maximum gloss sequence token length...")
    max_gloss_tokens = compute_max_gloss_tokens(df, tokenizer)
    logger.info("Maximum gloss sequence token length: %d tokens", max_gloss_tokens)

    # Step 3: Apply LoRA configuration
    logger.info("Applying LoRA configuration...")
    try:
        model = FastLanguageModel.get_peft_model(
            model,
            r=8,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_alpha=16,
            lora_dropout=0.0,
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )
        logger.info("LoRA configuration applied.")
    except Exception as e:
        logger.error("Error applying LoRA configuration: %s", e)
        sys.exit(1)

    # Step 4: Tokenize the dataset
    logger.info("Tokenizing the dataset...")

    def formatting_prompts_func(examples):
        convos = examples["messages"]
        texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
        return {"text": texts}

    try:
        train_dataset = train_dataset.map(formatting_prompts_func, batched=True)
        val_dataset = val_dataset.map(formatting_prompts_func, batched=True)
        logger.info("Tokenization complete.")
    except Exception as e:
        logger.error("Error during tokenization: %s", e)
        sys.exit(1)

    # Step 5: Set up the trainer
    logger.info("Setting up the trainer...")
    try:
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            dataset_text_field="text",
            max_seq_length=2048,
            dataset_num_proc=2,
            packing=False,
            args=TrainingArguments(
                per_device_train_batch_size=4,
                gradient_accumulation_steps=4,
                warmup_steps=1000,
                num_train_epochs=5, ## should be 5-10 but doesnt really work with interactive batch 
                learning_rate=5e-5, # try out 5e-5
                fp16=True,
                bf16=False,
                logging_steps=10,
                output_dir="/storage/text2gloss-finetune/outputs",
                optim="adamw_8bit",
                seed=42,
            ),
        )
        logger.info("Trainer setup complete.")
    except Exception as e:
        logger.error("Error setting up trainer: %s", e)
        sys.exit(1)

    # Step 6: Train the model
    logger.info("Starting training...")
    try:
        trainer.train()
        logger.info("Training finished successfully.")
    except Exception as e:
        logger.error("Error during training: %s", e)
        sys.exit(1)

    # Step 7: Evaluate the model
    logger.info("Evaluating the model...")

    def generate_gloss(sentence):
        if not hasattr(generate_gloss, "call_count"):
            generate_gloss.call_count = 0
        generate_gloss.call_count += 1
        call = generate_gloss.call_count
        log_this_call = call <= 10 or call % 500 == 0

        if log_this_call:
            logger.info("[Call %d] Generating gloss for sentence: '%s'", call, sentence)

        messages = [
            {
            "role": "system",
            "content": (
                "You are an expert translator for German Sign Language glossing. Your task is to convert a full German sentence "
                "into its precise sign language gloss sequence.\n"
                "Guidelines:\n"
                "- Output only the final gloss sequence as a comma-separated list of uppercase gloss tokens.\n"
                "- Do not include any chain-of-thought, explanations, or intermediary reasoning in your output.\n"
                "- Do not output any tokens not part of the gloss sequence."
            )
            },
            {"role": "user", "content": sentence},
        ]

        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_gloss_tokens + 1,
            num_beams=5,         # Enable beam search with 5 beams
            early_stopping=True, # Stop when all beams reach EOS
            length_penalty=1.0,  # Neutral penalty, adjust if needed
            no_repeat_ngram_size=2  # Prevent repetitive n-grams
        )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        if log_this_call:
            logger.info("[Call %d] Raw generated text: %s", call, generated_text)

        gloss_sequence = generated_text.split("<think>")[-1].strip()

        if log_this_call:
            logger.info("[Call %d] Generated gloss: '%s'", call, gloss_sequence)

        with open(gloss_output_file, "a", encoding="utf-8") as f:
            f.write(f"Call {call}:\nInput Sentence: {sentence}\nGenerated Gloss: {gloss_sequence}\n{'-' * 40}\n")

        return gloss_sequence
    
    try:
        logger.info("Started generating glosses for evaluation on validation set...")
        predictions = [generate_gloss(row.full_sentence) for row in val_df.itertuples()]
        references = [row.glosses for row in val_df.itertuples()]
        logger.info("Finished generating glosses for validation evaluation...")

        logger.info("Calculating BLEU score on validation set...")
        bleu = corpus_bleu(predictions, [references])
        logger.info("BLEU score on validation set: %f", bleu.score)

        logger.info("Calculating exact match accuracy on validation set...")
        exact_matches = sum(1 for pred, ref in zip(predictions, references) if pred == ref)
        accuracy = exact_matches / len(predictions)
        logger.info("Exact match accuracy on validation set: %f", accuracy)

        with open("/storage/text2gloss-finetune/evaluation_results.txt", "w") as f:
            f.write(f"Validation set evaluation:\n")
            f.write(f"BLEU score: {bleu.score}\n")
            f.write(f"Exact match accuracy: {accuracy}\n")
            
        ## Additional evaluation on 200 random samples from the training set - non-zero-shot:
        logger.info("Evaluating on 200 random samples from train dataset...")
        with open(gloss_output_file, "a", encoding="utf-8") as f:
            f.write("\n===== Begin Train Dataset Evaluation (non-zero-shot)=====\n")
        train_sample_df = train_df.sample(n=200, random_state=42)
        train_predictions = [generate_gloss(row.full_sentence) for row in train_sample_df.itertuples()]
        train_references = [row.glosses for row in train_sample_df.itertuples()]
        train_bleu = corpus_bleu(train_predictions, [train_references])
        train_exact_matches = sum(1 for pred, ref in zip(train_predictions, train_references) if pred == ref)
        train_accuracy = train_exact_matches / len(train_predictions)
        logger.info("Train BLEU score: %f", train_bleu.score)
        logger.info("Train exact match accuracy: %f", train_accuracy)
        with open("/storage/text2gloss-finetune/evaluation_results.txt", "a") as f:
            f.write(f"\nTrain samples evaluation (200 random samples):\n")
            f.write(f"Train BLEU score: {train_bleu.score}\n")
            f.write(f"Train exact match accuracy: {train_accuracy}\n")
    except Exception as e:
        logger.error("Error during evaluation: %s", e)
        sys.exit(1)

    # Step 8: Save the model
    logger.info("Saving the fine-tuned model and tokenizer...")
    try:
        model.save_pretrained("/storage/text2gloss-finetune/fine_tuned_deepseek")
        tokenizer.save_pretrained("/storage/text2gloss-finetune/fine_tuned_deepseek")
        logger.info("Model and tokenizer saved successfully.")
    except Exception as e:
        logger.error("Error saving model/tokenizer: %s", e)
        sys.exit(1)

    # Step 9: Example inference using a new sentence
    logger.info("Running inference on example sentence...")
    try:
        example_sentence = "Wie mein Leben aussieht?"
        generated_gloss = generate_gloss(example_sentence)
        print("Input sentence:", example_sentence)
        print("Generated gloss:", generated_gloss)
    except Exception as e:
        logger.error("Error during example inference: %s", e)
        sys.exit(1)

    # List any lingering threads/processes for debugging purposes
    logger.info("Active threads at end: %s", threading.enumerate())
    logger.info("========== PROCESS COMPLETE ==========")
    sys.exit(0)

if __name__ == '__main__':
    main()