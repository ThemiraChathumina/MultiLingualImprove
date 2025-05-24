from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from datasets import load_dataset
import torch
from huggingface_hub import login

token = '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'
login(token=token)

# Path to checkpoint (replace with your path if needed)
checkpoint_path = "/root/MultiLingualImprove/llama_1b_sentiment_analysis"

# Load tokenizer
model_name = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# Load trained model from checkpoint
model = AutoModelForCausalLM.from_pretrained(
    checkpoint_path,
    device_map="auto",
    # torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
)

# Use evaluation mode
model.eval()

# System prompt
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

# Load dataset and get one example
dataset = load_dataset("Themira/review_analysis", split="train")
# example = dataset[0]
example = "I regret buying this product. Overall, I’m really dissatisfied. The quality is not what I expected—the fabric feels cheap and started to tear after just a few uses. The sizing was also way off; the product was much smaller than described. The description on the website was misleading, and I feel like I didn’t get what I was promised. Considering the high price, I don’t think this product offers good value. I won’t be purchasing from this brand again."

# Construct prompt as you did during training
user_prompt = f"Analyze the given review and return the aspect-based sentiment response:\n\n{example}"
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt},
]
# Use chat template to create prompt string
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print("----------------------------------")
print(prompt)
print("----------------------------------")
# Tokenize
inputs = tokenizer(
    prompt,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=1024,
).to(model.device)

# Generation config (you can tweak as needed)
generation_config = GenerationConfig(
    max_new_tokens=1024,
    # temperature=0.2,
    # top_p=0.95,
    do_sample=False,
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id
)

# Generate output
with torch.no_grad():
    output_ids = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        **generation_config.to_dict()
    )

# The output includes the prompt, so we need to extract only the assistant's reply
generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

# # Remove prompt from generated_text
# # If your tokenizer supports, you can use chat template to parse, or do a simple string split:
# if generated_text.startswith(prompt):
#     assistant_response = generated_text[len(prompt):].strip()
# else:
#     # Fallback: try to extract everything after last user message
#     assistant_response = generated_text.split(user_prompt)[-1].strip()

print("\n===== Generated Assistant Response =====\n")
print(generated_text)

