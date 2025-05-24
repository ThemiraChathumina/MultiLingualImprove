from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, default_data_collator
from peft import LoraConfig, get_peft_model
import torch
import sys
from huggingface_hub import login

token = 'token'
login(token=token)

# Load dataset
dataset = load_dataset("Themira/review_analysis", split="train")

# Load tokenizer and model
model_name = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto"
)

# Define LoRA config
lora_config = LoraConfig(
    r=32,
    lora_alpha=128,
    target_modules=["q_proj", "k_proj", "v_proj"],
    lora_dropout=0.1,
    bias="all",
    task_type="CAUSAL_LM",
)

# Apply LoRA
model = get_peft_model(model, lora_config)

# Define the chat-style system prompt
system_prompt = (
    "You are an intelligent assistant that analyzes product reviews and extracts aspect-based sentiment information. "
    "For each review, identify the following aspects: Overall, Quality, Sizing, Packaging, Support, Description, and Value. "
    "For each aspect, provide:\n"
    " - 'name': The name of the aspect\n"
    " - 'rating': An integer from 0 to 5 representing sentiment (0 means not mentioned)\n"
    " - 'justification': A brief explanation based on the review text.\n"
    "If an aspect is not mentioned, set justification to 'Not mentioned in the review.' and rating to 0.\n"
    "Output must be in JSON format with the key 'aspects' containing a list of the 7 aspect dictionaries."
)

# Mapping function with label masking
def format_chat(example):
    user_prompt = f"Analyze the given review and return the aspect-based sentiment response:\n\n{example['review']}"
    assistant_reply = example['response']
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": assistant_reply}
    ]
    # Full conversation including assistant response
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    # Prompt up to assistant (exclude assistant reply)
    prompt_only_messages = messages[:-1]
    prompt_text = tokenizer.apply_chat_template(prompt_only_messages, tokenize=False, add_generation_prompt=True)
    prompt_tokens = tokenizer(prompt_text, add_special_tokens=False)
    full_tokens = tokenizer(full_text, padding="max_length", truncation=True, max_length=512)

    input_ids = full_tokens["input_ids"]
    attention_mask = full_tokens["attention_mask"]
    labels = [-100] * len(input_ids)

    # Mask: only assistant response tokens get labels
    start = len(prompt_tokens["input_ids"])
    end = len(full_tokens["input_ids"])
    for i in range(start, end):
        labels[i] = input_ids[i]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

# Tokenize and map
tokenized_dataset = dataset.map(format_chat)
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# Data collator (for Causal LM, not MLM)
data_collator = default_data_collator

per_device_train_batch_size = 8
gradient_accumulation_steps = 1

steps_per_epoch = len(tokenized_dataset) // (per_device_train_batch_size * gradient_accumulation_steps)

training_args = TrainingArguments(
    output_dir="./lora-llama3-ds",
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    num_train_epochs=3,
    logging_steps=100,
    save_steps=steps_per_epoch,  # Save every epoch
    save_total_limit=3,
    learning_rate=2e-5,
    report_to="none",
    deepspeed="experiments/ds.json",
    bf16=True,  # If supported
    remove_unused_columns=False,
    lr_scheduler_type="cosine",           # <-- ADD THIS LINE
    warmup_steps=100                      # <-- Optional: you can adjust or remove
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

trainer.train()

# Merge LoRA with base model after training
print("🔁 Merging LoRA adapters with the base model...")
merged_model = model.merge_and_unload()  # Converts LoRA model to standalone Hugging Face model

# Save merged model
save_path = "./lora-llama3-ds-merged"
merged_model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print(f"✅ LoRA-merged model saved to: {save_path}")
