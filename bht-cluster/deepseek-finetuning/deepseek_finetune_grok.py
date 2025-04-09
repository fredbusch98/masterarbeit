import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from sacrebleu import corpus_bleu

# Step 1: Load and preprocess the dataset
df = pd.read_csv('/storage/text2gloss-finetune/dataset.csv')
train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

def format_conversations(df):
    data = []
    for row in df.itertuples():
        messages = [
            {"role": "system", "content": "You are a sign language gloss translator. Given a sentence, output the corresponding gloss sequence."},
            {"role": "user", "content": row.Full_Sentence},
            {"role": "assistant", "content": row.Words},
        ]
        data.append({"messages": messages})
    return data

train_data = format_conversations(train_df)
val_data = format_conversations(val_df)

train_dataset = Dataset.from_list(train_data)
val_dataset = Dataset.from_list(val_data)

# Step 2: Load the model and tokenizer
model_name = "unsloth/DeepSeek-R1-Distill-Llama-8B"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
    device_map="auto",
)

# Step 3: Apply LoRA configuration
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

# Step 4: Tokenize the dataset
def formatting_prompts_func(examples):
    convos = examples["messages"]
    texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
    return {"text": texts}

train_dataset = train_dataset.map(formatting_prompts_func, batched=True)
val_dataset = val_dataset.map(formatting_prompts_func, batched=True)

# Step 5: Set up the trainer
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
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        output_dir="/storage/text2gloss-finetune/outputs",
        optim="adamw_8bit",
        seed=42,
        evaluation_strategy="epoch",
        save_strategy="epoch",
    ),
)

# Step 6: Train the model
trainer.train()

# Step 7: Evaluate the model
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

predictions = [generate_gloss(row.Full_Sentence) for row in val_df.itertuples()]
references = [row.Words for row in val_df.itertuples()]

bleu = corpus_bleu(predictions, [references])
print(f"BLEU score: {bleu.score}")

exact_matches = sum(1 for pred, ref in zip(predictions, references) if pred == ref)
accuracy = exact_matches / len(predictions)
print(f"Exact match accuracy: {accuracy}")

with open("/storage/text2gloss-finetune/evaluation_results.txt", "w") as f:
    f.write(f"BLEU score: {bleu.score}\n")
    f.write(f"Exact match accuracy: {accuracy}\n")

# Step 8: Save the model
model.save_pretrained("/storage/text2gloss-finetune/fine_tuned_deepseek")
tokenizer.save_pretrained("/storage/text2gloss-finetune/fine_tuned_deepseek")

# Step 9: Example inference using a new sentence
example_sentence = "Wie mein Leben aussieht?"
generated_gloss = generate_gloss(example_sentence)
print("Input sentence:", example_sentence)
print("Generated gloss:", generated_gloss)