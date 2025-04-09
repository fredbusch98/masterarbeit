import pandas as pd
from datasets import Dataset, load_metric
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import torch

# --- Step 1: Data Preprocessing ---
# Load your CSV file
df = pd.read_csv("text_to_sign_gloss_dataset.csv", header=None)
# Assuming first column is the full sentence and the remaining columns are gloss tokens.
df.columns = ["input_text"] + [f"gloss_{i}" for i in range(1, len(df.columns))]
# Create a target string by concatenating gloss tokens with a space delimiter
def concat_gloss(row):
    # Remove NaNs in case of variable number of tokens.
    gloss_tokens = [str(token) for token in row[1:] if pd.notnull(token)]
    return " ".join(gloss_tokens)

df["target_text"] = df.apply(concat_gloss, axis=1)
# Keep only the necessary columns.
df = df[["input_text", "target_text"]]

# Split data into training and validation splits.
train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# --- Step 2: Load Pre-trained DeepSeek Model & Tokenizer ---
# Replace 'deepseek/DeepSeek-R1-Distill-Llama-8B' with the correct model identifier.
model_name = "deepseek/DeepSeek-R1-Distill-Llama-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Load model for causal language modeling; adjust if your DeepSeek variant uses seq2seq architecture.
model = AutoModelForCausalLM.from_pretrained(model_name)

# --- Step 3: Apply LoRA using PEFT ---
# Define LoRA configuration.
lora_config = LoraConfig(
    r=8,                     # low-rank dimension, can try 8, 16, etc.
    lora_alpha=16,           # scaling factor (often same order as r)
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # these module names may vary based on architecture
    lora_dropout=0.1,        # some dropout to regularize training
    bias="none"
)
# Wrap the model with LoRA adapters.
model = get_peft_model(model, lora_config)
print("Trainable parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

# --- Step 4: Tokenization and Dataset Formatting ---
def preprocess_function(examples):
    # Tokenize input and target texts
    inputs = tokenizer(examples["input_text"], truncation=True, max_length=512)
    # For targets, set the label's tokenization (same tokenizer here)
    targets = tokenizer(examples["target_text"], truncation=True, max_length=128)
    inputs["labels"] = targets["input_ids"]
    return inputs

train_dataset = train_dataset.map(preprocess_function, batched=True)
val_dataset = val_dataset.map(preprocess_function, batched=True)

# --- Step 5: Training Setup ---
training_args = TrainingArguments(
    output_dir="./deepseek_lora_finetuned",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,  # if GPU supports, otherwise set fp16=False
    evaluation_strategy="epoch",
    logging_steps=50,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="loss"
)

# Define a simple compute_metrics function (optional)
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # Shift logits and labels for causal LM evaluation
    predictions = logits.argmax(-1)
    # Here you might integrate BLEU or token-level accuracy metrics if desired.
    return {"accuracy": (predictions == labels).mean()}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    # data_collator can be set automatically by the Trainer
    tokenizer=tokenizer,
    compute_metrics=compute_metrics  # optional; for demonstration
)

# --- Step 6: Fine-Tuning ---
trainer.train()

# --- Step 7: Save the Fine-tuned Model ---
model.save_pretrained("./deepseek_lora_finetuned")
tokenizer.save_pretrained("./deepseek_lora_finetuned")

# --- Step 8: Inference Example ---
def infer_sign_gloss(input_sentence):
    input_ids = tokenizer.encode(input_sentence, return_tensors="pt")
    input_ids = input_ids.to(model.device)
    # Generate output gloss sequence (adjust max_new_tokens as appropriate)
    output_ids = model.generate(input_ids, max_new_tokens=50, num_beams=5)
    gloss_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return gloss_output

# Example inference:
input_sentence = "Wie mein Leben aussieht?"
gloss_sequence = infer_sign_gloss(input_sentence)
print("Input Sentence:", input_sentence)
print("Predicted Gloss Sequence:", gloss_sequence)
