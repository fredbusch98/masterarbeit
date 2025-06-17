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
import argparse
from jiwer import wer

# Argument parsing
p = argparse.ArgumentParser(description='LLM finetuning for Gloss→Text translation')
p.add_argument('--train_dir', type=str, required=True,
               help='Name of the train directory, e.g.: og, bt-1 or bt-2')
p.add_argument('--num_epochs', type=int, default=6, help='Number of epochs for fine-tuning')
args = p.parse_args()

train_dir = args.train_dir
epochs = args.num_epochs

# Setup directories & logs
results_dir = "/storage/text2gloss-finetune/results/"
os.makedirs(results_dir, exist_ok=True)
log_file = os.path.join(results_dir, "process.log")
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
    # 1️⃣ Load data
    logger.info("Loading train/dev/test CSVs...")
    try:
        train_df = pd.read_csv(f'/storage/text2gloss-finetune/text2gloss_data/{train_dir}/train.csv', encoding='utf-8')
        val_df   = pd.read_csv(f'/storage/text2gloss-finetune/text2gloss_data/dev.csv',  encoding='utf-8')
        test_df  = pd.read_csv(f'/storage/text2gloss-finetune/text2gloss_data/test.csv', encoding='utf-8')
    except Exception as e:
        logger.error("Error reading CSVs: %s", e)
        sys.exit(1)
    logger.info("Data sizes: train=%d, dev=%d, test=%d", len(train_df), len(val_df), len(test_df))

    # 2️⃣ Format as chat for fine-tuning
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
                     "- Do not include any chain-of-thought, explanations, or intermediary reasoning.\n"
                     "- Ensure proper German grammar and word order."
                 )},
                {"role": "user", "content": row.glosses},
                {"role": "assistant", "content": row.full_sentence},
            ]
            data.append({"messages": messages})
        return data

    train_data = format_conversations(train_df)
    val_data   = format_conversations(val_df)
    train_dataset = Dataset.from_list(train_data)
    val_dataset   = Dataset.from_list(val_data)
    logger.info("Formatted datasets: train=%d, dev=%d", len(train_dataset), len(val_dataset))

    # 3️⃣ Load model + LoRA
    logger.info("Loading model unsloth/DeepSeek-R1-Distill-Llama-8B...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/DeepSeek-R1-Distill-Llama-8B",
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
        device_map="auto",
    )
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

    # 4️⃣ Compute max output length
    def compute_max_output_tokens(df, tokenizer):
        max_tokens = 0
        for sent in df['full_sentence']:
            tokens = tokenizer(sent, add_special_tokens=False, return_tensors="pt")['input_ids']
            max_tokens = max(max_tokens, tokens.shape[1])
        return max_tokens

    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    max_output_tokens = compute_max_output_tokens(full_df, tokenizer)
    logger.info("Max target sentence tokens: %d", max_output_tokens)

    # 5️⃣ Tokenize inputs
    def formatting_prompts_func(examples):
        texts = [
            tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
            for convo in examples["messages"]
        ]
        return {"text": texts}

    train_dataset = train_dataset.map(formatting_prompts_func, batched=True)
    val_dataset   = val_dataset.map(formatting_prompts_func, batched=True)
    logger.info("Tokenization done")

    # 6️⃣ Trainer
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
            output_dir="/storage/text2gloss-finetune/outputs",
            optim="adamw_8bit",
            seed=42,
        ),
    )

    # 7️⃣ Train
    trainer.train()

    # 8️⃣ Generation & Evaluation
    def generate_text(reference):
        if not hasattr(generate_text, "call_count"):
            generate_text.call_count = 0
        generate_text.call_count += 1
        call = generate_text.call_count
        log_this_call = call <= 10 or call % 500 == 0

        gloss_sequence = reference.glosses
        ref_sentence = reference.full_sentence

        if log_this_call:
            logger.info("[Call %d] Generating text for gloss: '%s'", call, gloss_sequence)

        messages = [
            {"role": "system",
                 "content": (
                     "You are an expert translator for German Sign Language glossing. Your task is to convert a sequence of "
                     "German Sign Language gloss tokens into a fluent German sentence.\n"
                     "Guidelines:\n"
                     "- Output only the final German sentence.\n"
                     "- The final German sentence should not include any punctuation marks.\n"
                     "- Do not include any chain-of-thought, explanations, or intermediary reasoning.\n"
                     "- Ensure proper German grammar and word order."
                 )},
            {"role": "user", "content": gloss_sequence},
        ]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=max_output_tokens + 1)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

        generated_sentence = generated_text.split("<think>")[-1].strip()
        generated_sentence = generated_sentence.lower()

        if log_this_call:
            logger.info("[Call %d] Generated text: '%s'", call, generated_text)
            logger.info("[Call %d] Generated sentence: '%s'", call, generated_sentence)
            logger.info("[Call %d] Reference sentence: '%s'", call, ref_sentence)

        with open(text_output_file, "a", encoding="utf-8") as f:
            f.write(f"Call {call}:\nInput Gloss: {gloss_sequence}\nGenerated Sentence: {generated_sentence}\n{'-'*40}\n")

        return generated_sentence

    def evaluate(mode):
        df = val_df if mode=="DEV" else test_df
        logger.info(f"Evaluating on {mode} set...")
        preds = [generate_text(r) for r in df.itertuples()]
        refs  = [r.full_sentence.lower() for r in df.itertuples()]

        # BLEU-1..4
        bleu_scores = {}
        for n in range(1,5):
            bleu = BLEU(max_ngram_order=n).corpus_score(preds, [refs]).score
            bleu_scores[f"BLEU-{n}"] = bleu
            logger.info(f"BLEU-{n}: {bleu:.2f}")
        chrf = corpus_chrf(preds, [refs]).score; logger.info(f"CHRF: {chrf:.2f}")
        ter  = corpus_ter (preds, [refs]).score; logger.info(f"TER : {ter:.2f}")

        rouge = rouge_scorer.RougeScorer(["rouge1","rouge2","rougeL"], use_stemmer=True)
        r1=r2=rL=0
        for p, r in zip(preds, refs):
            s = rouge.score(r,p)
            r1+=s["rouge1"].fmeasure; r2+=s["rouge2"].fmeasure; rL+=s["rougeL"].fmeasure
        n = len(preds)
        logger.info(f"ROUGE-1: {100*r1/n:.2f}, ROUGE-2: {100*r2/n:.2f}, ROUGE-L: {100*rL/n:.2f}")

        # --- WER ---
        wer_score = wer(refs, preds)
        logger.info("WER : %.2f%%", wer_score * 100)

        with open(f"{results_dir}/evaluation_results_{mode}.txt","w") as f:
            f.write(f"{mode} evaluation\n")
            for k,v in bleu_scores.items(): f.write(f"{k}: {v:.2f}\n")
            f.write(f"CHRF: {chrf:.2f}\nTER: {ter:.2f}\n")
            f.write(f"ROUGE-1: {100*r1/n:.2f}\nROUGE-2: {100*r2/n:.2f}\nROUGE-L: {100*rL/n:.2f}\n")
            f.write(f"WER : {wer_score * 100:.2f}%\n")

    evaluate("DEV")
    evaluate("TEST")

    # 9️⃣ Save
    model.save_pretrained(f"{results_dir}/fine_tuned_deepseek_gloss2text")
    tokenizer.save_pretrained(f"{results_dir}/fine_tuned_deepseek_gloss2text")

    logger.info("Active threads: %s", threading.enumerate())
    logger.info("========== PROCESS COMPLETE ==========")
    sys.exit(0)

if __name__ == '__main__':
    main()
