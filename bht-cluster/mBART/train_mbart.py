import os
import pandas as pd
import sentencepiece as spm
import torch
from torch.utils.data import Dataset
from transformers import (
    MBartForConditionalGeneration,
    MBartTokenizer,
    Trainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)
from sacrebleu.metrics import BLEU
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Step 1: Train SentencePiece Tokenizer ---
def train_sentencepiece(train_csv_path, sp_model_prefix='tokenizer', vocab_size=19923.):
    df = pd.read_csv(train_csv_path)
    corpus_file = 'corpus.txt'
    with open(corpus_file, 'w', encoding='utf-8') as f:
        for txt in df['full_sentence']:
            f.write(str(txt) + '\n')
        for txt in df['glosses']:
            f.write(str(txt) + '\n')
    spm.SentencePieceTrainer.Train(
        input=corpus_file,
        model_prefix=sp_model_prefix,
        vocab_size=vocab_size,  # lowered vocab size to 16000
        character_coverage=1.0,
        model_type='unigram'
    )
    os.remove(corpus_file)

class Text2GlossDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=128):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        src_text = self.df.loc[idx, 'full_sentence']
        tgt_text = self.df.loc[idx, 'glosses']
        model_inputs = self.tokenizer(src_text, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(tgt_text, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')['input_ids']
        labels = labels.squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {
            'input_ids': model_inputs['input_ids'].squeeze(),
            'attention_mask': model_inputs['attention_mask'].squeeze(),
            'labels': labels
        }

def evaluate_bleu(model, tokenizer, data_frame, mode):
    logger.info(f"Started generating glosses for evaluation on {mode} set...")
    predictions = []
    references = []
    for row in data_frame.itertuples():
        input_ids = tokenizer(row.full_sentence, return_tensors="pt", truncation=True, padding=True).input_ids.to(model.device)
        outputs = model.generate(input_ids, max_length=128)
        pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predictions.append(pred)
        references.append(row.glosses)

    logger.info(f"Finished generating glosses for {mode} evaluation...")
    logger.info(f"Calculating BLEU-1 through BLEU-4 on {mode} set with SacreBLEU...")
    bleu_scores = {}
    for n in range(1, 5):
        bleu_metric = BLEU(max_ngram_order=n)
        result = bleu_metric.corpus_score(predictions, [references])
        bleu_scores[f"BLEU-{n}"] = result.score
        logger.info("BLEU-%d: %.2f", n, result.score)

    output_file = f"evaluation_results_{mode}.txt"
    with open(output_file, "w") as f:
        f.write(f"{mode} set evaluation:\n")
        f.write("=== Detailed BLEU scores ===\n")
        for name, score in bleu_scores.items():
            f.write(f"{name}: {score:.2f}\n")

def main():
    train_csv = 'train.csv'
    dev_csv = 'dev.csv'
    test_csv = 'test.csv'

    sp_model_prefix = 'tokenizer'
    sp_model_file = f'{sp_model_prefix}.model'

    if not os.path.isfile(sp_model_file):
        logger.info("Training SentencePiece tokenizer...")
        train_sentencepiece(train_csv, sp_model_prefix)
        logger.info("Tokenizer training complete.")

    tokenizer = MBartTokenizer(sp_model_file, src_lang="de_DE", tgt_lang="de_DE")
    tokenizer.pad_token = tokenizer.eos_token

    model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-cc25")
    model.config.dropout = 0.3
    model.config.attention_dropout = 0.3

    train_df = pd.read_csv(train_csv)
    dev_df = pd.read_csv(dev_csv)
    test_df = pd.read_csv(test_csv)

    train_dataset = Text2GlossDataset(train_df, tokenizer)
    dev_dataset = Text2GlossDataset(dev_df, tokenizer)

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir='./results',
        num_train_epochs=80,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=1e-5,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        label_smoothing_factor=0.2,
        logging_dir='./logs',
        logging_steps=50,
        predict_with_generate=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    logger.info("Evaluating on DEV and TEST sets with BLEU metrics...")
    evaluate_bleu(model, tokenizer, dev_df, mode="DEV")
    evaluate_bleu(model, tokenizer, test_df, mode="TEST")

    # Save final model and tokenizer
    final_model_dir = './final_model'
    logger.info("Saving final model and tokenizer to %s", final_model_dir)
    model.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)

if __name__ == "__main__":
    main()
