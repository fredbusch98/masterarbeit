import sys
import logging
import threading
import pandas as pd
from datasets import Dataset
from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from sacrebleu import corpus_chrf, corpus_ter
from sacrebleu.metrics import BLEU
import os
from rouge_score import rouge_scorer

epochs = 6
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
    logger.info("Loading train and test datasets from CSVs...")
    try:
        train_df = pd.read_csv('/storage/text2gloss-finetune/phoenix-weather/phoenix_train.csv', encoding='utf-8')
        val_df   = pd.read_csv('/storage/text2gloss-finetune/phoenix-weather/phoenix_val.csv',  encoding='utf-8')
        test_df  = pd.read_csv('/storage/text2gloss-finetune/phoenix-weather/phoenix_test.csv',  encoding='utf-8')
    except Exception as e:
        logger.error("Error reading CSVs: %s", e)
        sys.exit(1)
    logger.info("Train shape: %d, Validation shape: %d", len(train_df), len(val_df))

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
    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    max_gloss_tokens = compute_max_gloss_tokens(full_df, tokenizer)
    logger.info("Maximum gloss sequence token length: %d tokens", max_gloss_tokens)

    # Step 3: Apply LoRA configuration
    logger.info("Applying LoRA configuration...")
    try:
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj",],
            lora_alpha=16,
            lora_dropout=0,
            bias = "none",
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
                num_train_epochs=epochs, # als nächstes mit 10
                learning_rate=5e-5,
                fp16 = not torch.cuda.is_bf16_supported(),
                bf16 = torch.cuda.is_bf16_supported(),
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
        outputs = model.generate(**inputs, max_new_tokens=max_gloss_tokens + 1)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        if log_this_call:
            logger.info("[Call %d] Raw generated text: %s", call, generated_text)

        gloss_sequence = generated_text.split("<think>")[-1].strip()

        if log_this_call:
            logger.info("[Call %d] Generated gloss: '%s'", call, gloss_sequence)

        with open(gloss_output_file, "a", encoding="utf-8") as f:
            f.write(f"Call {call}:\nInput Sentence: {sentence}\nGenerated Gloss: {gloss_sequence}\n{'-' * 40}\n")

        return gloss_sequence
    
    def evaluate(mode):
        data_frame = val_df if mode == "DEV" else test_df
        logger.info(f"Started generating glosses for evaluation on {mode} set...")
        predictions = [generate_gloss(row.full_sentence) for row in data_frame.itertuples()]
        references = [row.glosses for row in data_frame.itertuples()]
        logger.info(f"Finished generating glosses for {mode} evaluation...")

        # BLEU-1 to BLEU-4
        logger.info(f"Calculating BLEU-1 through BLEU-4 on {mode} set with SacreBLEU...")
        bleu_scores = {}
        for n in range(1, 5):
            bleu_metric = BLEU(max_ngram_order=n)
            result = bleu_metric.corpus_score(predictions, [references])
            bleu_scores[f"BLEU-{n}"] = result.score
            logger.info("BLEU-%d: %.2f", n, result.score)

        # CHRF
        logger.info(f"Calculating CHRF score on {mode} set..." )
        chrf = corpus_chrf(predictions, [references])
        logger.info(f"CHRF score on {mode} set: %f", chrf.score)

        # TER
        logger.info(f"Calculating TER score on {mode} set..." )
        ter = corpus_ter(predictions, [references])
        logger.info(f"TER score on {mode} set: %f", ter.score)

        # ROUGE
        logger.info(f"Calculating ROUGE scores on {mode} set..." )
        rouge_scorer_obj = rouge_scorer.RougeScorer(["rouge1","rouge2","rougeL"], use_stemmer=True)
        rouge1, rouge2, rougel = 0, 0, 0
        for pred, ref in zip(predictions, references):
            scores = rouge_scorer_obj.score(ref, pred)
            rouge1 += scores["rouge1"].fmeasure
            rouge2 += scores["rouge2"].fmeasure
            rougel += scores["rougeL"].fmeasure
        n = len(predictions)
        rouge1 /= n
        rouge2 /= n
        rougel /= n
        logger.info(f"Average ROUGE-1 on {mode} set: %f", rouge1)
        logger.info(f"Average ROUGE-2 on {mode} set: %f", rouge2)
        logger.info(f"Average ROUGE-L on {mode} set: %f", rougel)

        with open(f"/storage/text2gloss-finetune/evaluation_results_{mode}.txt", "w") as f:
            f.write(f"{mode} set evaluation:\n")
            f.write("=== Detailed BLEU scores ===\n")
            for name, score in bleu_scores.items(): 
                f.write(f"{name}: {score:.2f}\n")
            f.write("\n=== CHRF and TER ===\n")
            f.write(f"CHRF score: {chrf.score:.2f}\n")
            f.write(f"TER score: {ter.score:.2f}\n")
            f.write("\n=== ROUGE F1 scores ===\n")
            f.write(f"ROUGE-1: {rouge1:.2f}\n")
            f.write(f"ROUGE-2: {rouge2:.2f}\n")
            f.write(f"ROUGE-L: {rougel:.2f}\n")
    try:
        evaluate("DEV")
        evaluate("TEST")
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