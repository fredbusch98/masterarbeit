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
from jiwer import wer

epochs = 3
results_dir = "/storage/text2gloss-finetune/phoenix-weather/results"
os.makedirs(results_dir, exist_ok=True)
log_file = os.path.join(results_dir, "process.log")
# ➊ Rename output log file
text_output_file = os.path.join(results_dir, "text_generation_log.txt")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def main():
    logger.info("========== STARTING PROCESS ==========")
    # Load CSVs
    logger.info("Loading train and test datasets from CSVs...")
    try:
        train_df = pd.read_csv('/storage/text2gloss-finetune/phoenix-weather/og_train.csv', encoding='utf-8')
        val_df   = pd.read_csv('/storage/text2gloss-finetune/phoenix-weather/dev.csv', encoding='utf-8')
        test_df  = pd.read_csv('/storage/text2gloss-finetune/phoenix-weather/test.csv', encoding='utf-8')
    except Exception as e:
        logger.error("Error reading CSVs: %s", e)
        sys.exit(1)
    logger.info("Train shape: %d, Validation shape: %d, Test shape: %d", len(train_df), len(val_df), len(test_df))

    # ➋ Swap prompt direction: gloss→text
    def format_conversations(df):
        data = []
        for row in df.itertuples():
            messages = [
                {"role": "system",
                 "content": (
                     "You are an expert translator for German Sign Language glossing. Your task is to convert a sequence of "
                     "German Sign Language gloss tokens into a fluent German sentence.\n"
                     "Guidelines:\n"
                     "- Output only the final German sentence.\n"
                     "- The final German sentence should not include any punctuation marks.\n"
                     "- Do not include any chain-of-thought or explanations.\n"
                     "- Use correct German grammar and word order."
                 )},
                # user now provides glosses
                {"role": "user",    "content": row.glosses},
                # assistant target is the full sentence
                {"role": "assistant","content": row.full_sentence},
            ]
            data.append({"messages": messages})
        return data

    logger.info("Formatting conversations for training and validation...")
    train_data = format_conversations(train_df)
    val_data   = format_conversations(val_df)

    train_dataset = Dataset.from_list(train_data)
    val_dataset   = Dataset.from_list(val_data)
    logger.info("Dataset formatting complete. Train size: %d, Val size: %d", len(train_dataset), len(val_dataset))

    # Load model & tokenizer
    logger.info("Loading model and tokenizer: unsloth/DeepSeek-R1-Distill-Llama-8B")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/DeepSeek-R1-Distill-Llama-8B",
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
        device_map="auto",
    )

    # ➌ Apply LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    logger.info("LoRA configuration applied.")

    # ➍ Compute max target (sentence) length
    def compute_max_output_tokens(df, tokenizer):
        max_tokens = 0
        for sent in df['full_sentence']:
            tokens = tokenizer(sent, add_special_tokens=False, return_tensors="pt")['input_ids']
            max_tokens = max(max_tokens, tokens.shape[1])
        return max_tokens

    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    max_output_tokens = compute_max_output_tokens(full_df, tokenizer)
    logger.info("Maximum target sentence token length: %d", max_output_tokens)

    # Tokenize prompts
    def formatting_prompts_func(examples):
        return {
            "text": [
                tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
                for convo in examples["messages"]
            ]
        }

    train_dataset = train_dataset.map(formatting_prompts_func, batched=True)
    val_dataset   = val_dataset.map(formatting_prompts_func, batched=True)
    logger.info("Tokenization complete.")

    # Trainer setup
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=2048,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            warmup_steps=1000,
            num_train_epochs=epochs,
            learning_rate=5e-5,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=10,
            output_dir="/storage/text2gloss-finetune/phoenix-weather/outputs",
            optim="adamw_8bit",
            seed=42,
        ),
    )
    logger.info("Trainer setup complete.")

    # Train
    logger.info("Starting training...")
    trainer.train()
    logger.info("Training finished successfully.")

    # ➎ Rename & rewrite generation to gloss→text
    def generate_text(reference):
        if not hasattr(generate_text, "call_count"):
            generate_text.call_count = 0
        generate_text.call_count += 1
        call = generate_text.call_count
        log_this = call <= 10 or call % 500 == 0
        gloss_sequence = reference.glosses
        ref_sentence = reference.full_sentence.lower()

        if log_this:
            logger.info("[Call %d] Generating text for gloss: '%s'", call, gloss_sequence)

        messages = [
            {"role": "system", 
             "content": (
                "You are an expert translator for German Sign Language glossing. Your task is to convert a sequence of "
                "German Sign Language gloss tokens into a fluent German sentence.\n"
                "Guidelines:\n"
                "- Output only the final German sentence.\n"
                "- The final German sentence should not include any punctuation marks.\n"
                "- Do not include any chain-of-thought or explanations.\n"
                "- Use correct German grammar and word order."
                 )},
            {"role": "user", "content": gloss_sequence},
        ]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=max_output_tokens + 1)
        text_out = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        generated_sentence = text_out.split("<think>")[-1].strip()
        generated_sentence = generated_sentence.lower()

        if log_this:
            logger.info("[Call %d] Generated text: '%s'", call, text_out)
            logger.info("[Call %d] Generated sentence: '%s'", call, generated_sentence)
            logger.info("[Call %d] Reference sentence: '%s'", call, ref_sentence)

        with open(text_output_file, "a", encoding="utf-8") as f:
            f.write(f"Call {call}:\nInput Gloss Sequence: {gloss_sequence}\nGenerated Sentence: {generated_sentence}\n{'-'*40}\n")

        return generated_sentence

    # ➏ Swap preds/refs in evaluation
    def evaluate(mode):
        df = val_df if mode == "DEV" else test_df
        logger.info(f"Started evaluation on {mode} set...")
        predictions = [generate_text(r) for r in df.itertuples()]
        references  = [r.full_sentence.lower() for r in df.itertuples()]
        logger.info(f"Finished generating texts for {mode} set.")

        # BLEU
        for n in range(1, 5):
            score = BLEU(max_ngram_order=n).corpus_score(predictions, [references]).score
            logger.info("BLEU-%d: %.2f", n, score)

        # CHRF & TER
        logger.info("CHRF: %f", corpus_chrf(predictions, [references]).score)
        logger.info("TER : %f", corpus_ter(predictions, [references]).score)

        # ROUGE
        rouge = rouge_scorer.RougeScorer(["rouge1","rouge2","rougeL"], use_stemmer=True)
        r1=r2=rL=0
        for p, r in zip(predictions, references):
            sc = rouge.score(r, p)
            r1 += sc["rouge1"].fmeasure
            r2 += sc["rouge2"].fmeasure
            rL += sc["rougeL"].fmeasure
        n = len(predictions)
        logger.info("ROUGE-1: %.2f", 100*r1/n)
        logger.info("ROUGE-2: %.2f", 100*r2/n)
        logger.info("ROUGE-L: %.2f", 100*rL/n)

        # --- WER ---
        wer_score = wer(references, predictions)
        logger.info("WER : %.2f%%", wer_score * 100)

        # Write out evaluation
        with open(f"{results_dir}/evaluation_results_{mode}.txt", "w") as f:
            f.write(f"{mode} evaluation:\n")
            for n in range(1,5):
                f.write(f"BLEU-{n}: {BLEU(max_ngram_order=n).corpus_score(predictions,[references]).score:.2f}\n")
            f.write(f"CHRF: {corpus_chrf(predictions,[references]).score:.2f}\n")
            f.write(f"TER : {corpus_ter(predictions,[references]).score:.2f}\n")
            f.write(f"ROUGE-1: {100*r1/n:.2f}\nROUGE-2: {100*r2/n:.2f}\nROUGE-L: {100*rL/n:.2f}\n")
            f.write(f"WER : {wer_score * 100:.2f}%\n")

    try:
        evaluate("DEV")
        evaluate("TEST")
    except Exception as e:
        logger.error("Error during evaluation: %s", e)
        sys.exit(1)

    # Save model & tokenizer
    model.save_pretrained(f"{results_dir}/fine_tuned_deepseek_gloss2text")
    tokenizer.save_pretrained(f"{results_dir}/fine_tuned_deepseek_gloss2text")
    logger.info("Model and tokenizer saved successfully.")

    logger.info("Active threads at end: %s", threading.enumerate())
    logger.info("========== PROCESS COMPLETE ==========")
    sys.exit(0)

if __name__ == '__main__':
    main()
