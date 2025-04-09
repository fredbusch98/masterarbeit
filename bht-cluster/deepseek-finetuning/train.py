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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
                {"role": "system", "content": "You are a sign language gloss translator. Given a sentence, output the corresponding gloss sequence."},
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

    # Step 3: Apply LoRA configuration
    logger.info("Applying LoRA configuration...")
    try:
        model = FastLanguageModel.get_peft_model(
            model,
            r=8,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
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
                per_device_train_batch_size=2,
                gradient_accumulation_steps=4,
                warmup_steps=5,
                num_train_epochs=1,
                learning_rate=2e-4,
                fp16=False,
                bf16=True,
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
        messages = [
            {"role": "system", "content": "You are a sign language gloss translator. Given a sentence, output the corresponding gloss sequence."},
            {"role": "user", "content": sentence},
        ]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=50)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        assistant_response = generated_text.split("[Assistant]")[-1].strip()
        return assistant_response

    try:
        predictions = [generate_gloss(row.full_sentence) for row in val_df.itertuples()]
        references = [row.glosses for row in val_df.itertuples()]

        bleu = corpus_bleu(predictions, [references])
        logger.info("BLEU score: %f", bleu.score)

        exact_matches = sum(1 for pred, ref in zip(predictions, references) if pred == ref)
        accuracy = exact_matches / len(predictions)
        logger.info("Exact match accuracy: %f", accuracy)

        with open("/storage/text2gloss-finetune/evaluation_results.txt", "w") as f:
            f.write(f"BLEU score: {bleu.score}\n")
            f.write(f"Exact match accuracy: {accuracy}\n")
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
