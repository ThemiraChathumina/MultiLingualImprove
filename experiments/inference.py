# inference.py

import torch
import json
from transformers import AutoTokenizer
from model import MPTModel # Assuming model.py is in the same directory
from read_data import llm_input_features # Assuming read_data.py is in the same directory
from datasets import load_dataset
import csv # Import the csv module
import os # Import os module for path joining
import torch.nn.functional as F # Import F for softmax

# --- Configuration ---
# TODO: Update this path to your saved model checkpoint
CHECKPOINT_PATH = "./outputs/sentiment/epoch_2_aug/pytorch_model.bin" # Example path
OUTPUT_DIR = "./inference_outputs" # Directory to save CSV and other outputs

LLM_PATH = "/root/MultiLingualImprove/experiments/llama_1b_sentiment_analysis" # Example path
MT_PATH = "google/mt5-xl"
EXT_PATH = "facebook/nllb-200-distilled-600M"
MAX_SEQ_LEN = 500
MAX_GEN_LEN = 500

SYSTEM_PROMPT = (
    "You are an intelligent assistant that analyzes product reviews and extracts aspect-based sentiment information. "
    "For each review, identify the following aspects: Overall, Quality, Sizing, Packaging, Support, Description, and Value. "
    "For each aspect, provide:\n"
    " - 'name': The name of the aspect\n"
    " - 'rating': An integer from 0 to 5 representing sentiment (0 means not mentioned)\n"
    " - 'justification': A brief explanation based on the review text.\n"
    "If an aspect is not mentioned, set justification to 'Not mentioned in the review.' and rating to 0.\n"
    "Output must be in JSON format with the key 'aspects' containing a list of the 7 aspect dictionaries."
)

def save_layer_weights_to_csv(model, filename="normalized_layer_weights.csv"):
    """
    Calculates and saves the normalized layer weights (norm_weights)
    from the model's encoder_mt to a CSV file.

    Args:
        model (MPTModel): The loaded MPTModel instance.
        filename (str): The name of the CSV file to save the weights to.
    """
    try:
        # Access the necessary parameters from the model's encoder_mt
        encoder_mt = model.encoder_mt
        layer_weights_lb = encoder_mt.layer_weights_lb.data
        base_temp = encoder_mt.base_temp.data
        temp_param = encoder_mt.temp.data # This is nn.Parameter, so .data to get tensor
        factor = encoder_mt.factor.data

        # Calculate temp_applicable
        # Ensure all tensors are on the same device, e.g., CPU for numpy conversion
        device = layer_weights_lb.device
        temp_applicable = base_temp.to(device) + temp_param.to(device) * factor.to(device)

        # Calculate norm_weights
        norm_weights_tensor = F.softmax(temp_applicable * layer_weights_lb, dim=0)
        norm_weights_numpy = norm_weights_tensor.cpu().numpy()

        # Ensure the output directory exists
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        filepath = os.path.join(OUTPUT_DIR, filename)

        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write a header row
            writer.writerow(['Layer_Index', 'Normalized_Weight'])
            # Write the normalized weights
            for i, weight in enumerate(norm_weights_numpy):
                writer.writerow([i, weight])
        print(f"Successfully saved normalized layer weights to {filepath}")

    except AttributeError as e:
        print(f"Error: Could not find necessary parameters (layer_weights_lb, base_temp, temp, factor) in model.encoder_mt. Details: {e}. Ensure the model structure is correct.")
    except Exception as e:
        print(f"An error occurred while saving normalized layer weights: {e}")


def main():
    """
    Main function to run inference.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- 1. Load Tokenizer ---
    tokenizer_llm = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", use_fast=True)
    tokenizer_llm.pad_token = tokenizer_llm.eos_token
    tokenizer_llm.padding_side = "left"

    # --- 2. Load Model ---
    model_config = {
        'mt_path': MT_PATH,
        'ext_path': EXT_PATH,
        'llm_path': LLM_PATH,
        'max_gen_len': MAX_GEN_LEN,
        'llm_bos_token_id': tokenizer_llm.bos_token_id,
        'llm_pad_token_id': tokenizer_llm.pad_token_id,
        'augmentation': True, # Assuming this was intended for training, might not be needed for inference logic
        'max_seq_len': MAX_SEQ_LEN
    }

    model = MPTModel(model_config)

    # Load the trained weights from the checkpoint
    try:
        print(f"Loading model checkpoint from: {CHECKPOINT_PATH}")
        # Ensure the checkpoint is loaded to the correct device
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=torch.device(device))
        
        # Handle potential 'module.' prefix if model was saved with DataParallel/DistributedDataParallel
        # and then directly loading the state_dict.
        # Your training script uses Accelerator, which should handle unwrapping.
        # This is a general safeguard.
        new_state_dict = {}
        is_ddp_or_dp_artifact = False
        if isinstance(checkpoint, dict): # Check if checkpoint is a state_dict
            for k in checkpoint.keys():
                if k.startswith('module.'):
                    is_ddp_or_dp_artifact = True
                    break
        
        if is_ddp_or_dp_artifact:
            for k, v in checkpoint.items():
                new_state_dict[k[7:]] = v  # remove 'module.'
            model.load_state_dict(new_state_dict, strict=False)
        elif isinstance(checkpoint, dict): # If it's a state_dict without 'module.'
             model.load_state_dict(checkpoint, strict=False)
        else: # If checkpoint is not a state_dict (e.g. could be the model itself, though less common for .bin)
             # This case might need adjustment based on how exactly the model was saved.
             # For `accelerator.save_state`, the model is usually under a 'model' key if saving the whole state.
             # If `save_with_accelerate` saves unwrapped model's state_dict, the above dict checks are fine.
             # Assuming the checkpoint is a state_dict for now.
             print("Warning: Checkpoint might not be a state_dict. Attempting direct load.")
             model.load_state_dict(checkpoint, strict=False)


        print("Model checkpoint loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at {CHECKPOINT_PATH}.")
        print("Please update the CHECKPOINT_PATH variable in the script.")
        return
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        return


    model.to(device)
    model.eval()

    # --- Save Layer Weights ---
    # Call the function to save layer weights after the model is loaded and in eval mode
    # save_layer_weights_to_csv(model, filename="encoder_mt_normalized_layer_weights.csv")

    # --- 3. Prepare Data ---
    # Using Sinhala sample reviews as requested
    sample_reviews = [
        "මට අද ටී-ෂර්ට් එක ලැබුණා. රෙද්ද ඉතාම උසස් තත්ත්වයේ තියෙනවා වගේම ඇඟට හොඳටම ගැළපෙනවා. ඩිලිවරි එකත් හරිම වේගවත්. මිලදී ගැනීම ගැන ගොඩක් සතුටුයි.",
        "මේ හෙඩ්ෆෝන් එකෙන් කිසිම වැඩක් නෑ. සද්දෙ කොලිටි එක හරිම අන්තිමයි. බැටරිය පැයක්වත් පාවිච්චි කරන්න බෑ. මේකට සල්ලි වියදම් කරන එක අපරාදයක්.",
        "ෆ්‍රයිඩ් රයිස් එක නම් ගොඩක් රසයි. ඒත් පැකේජ් එක කැඩිලා ඇවිත් තිබුණේ, ටිකක් බෑග් එක ඇතුළෙම හැලිලා. ගෙවන ගාණට දෙන ප්‍රමාණයත් ටිකක් මදි වගේ.",
        "භාණ්ඩය හොඳයි, නමුත් බෙදාහැරීමේ සේවාව ඉතාමත් කණගාටුදායකයි. එන්න සති දෙකක් ගියා, පාරිභෝගික සේවා කණ්ඩායම මගේ ඊමේල් වලට කිසිම ප්‍රතිචාරයක් දැක්වූයේ නැහැ.",
        "ෆෝන් කවර් එක හරියටම ෆิต් වෙනවා, පෙනුමත් ලස්සනයි. වෙබ්සයිට් එකේ තිබුණු විස්තරය නිවැරදියි. ගෙවන මුදලට හොඳ වටිනාකමක් තියෙනවා."
    ]
    # You can also load from dataset as before:
    # try:
    #     dataset = load_dataset("Themira/review_analysis", split="test")
    #     sample_reviews = [item['review'] for item in dataset.select(range(5))]
    # except Exception as e:
    #     print(f"Could not load dataset. Using custom Sinhala samples. Error: {e}")


    # --- 4. Run Inference ---
    all_generated_outputs = []
    with torch.no_grad():
        for i, review_text in enumerate(sample_reviews):
            print("-" * 50)
            print(f"Sample {i+1} Review (Sinhala):")
            print(review_text)
            print("-" * 50)

            instructional_prompts_list = [{
                "role": "system", "content": SYSTEM_PROMPT
            }, {
                "role": "user", "content": "Analyze the given review and return the aspect-based sentiment response."
            }]
            
            templated_instruction_prompt = tokenizer_llm.apply_chat_template(instructional_prompts_list, tokenize=False, add_generation_prompt=True)

            input_ids_prompt, mask_prompt = llm_input_features(
                [templated_instruction_prompt], 
                tokenizer_llm,
                max_seq_len=MAX_GEN_LEN, 
                add_bos_token=False, 
                add_eos_token=False 
            )
            
            current_batch_encoded_inputs = [review_text] 

            generated_ids = model(
                current_batch_encoded_inputs, 
                input_ids_prompt=input_ids_prompt.to(device),
                mask_prompt=mask_prompt.to(device)
            )
            
            # Assuming generated_ids from model.forward() are *only the new tokens*.
            # If the output includes the input prompt, this decoding needs adjustment.
            decoded_response = tokenizer_llm.decode(generated_ids[0], skip_special_tokens=True)
            
            all_generated_outputs.append({
                "review": review_text,
                "generated_response": decoded_response
            })

            print("\nGenerated Output:")
            try:
                parsed_json = json.loads(decoded_response)
                print(json.dumps(parsed_json, indent=2, ensure_ascii=False)) 
            except json.JSONDecodeError:
                print(decoded_response) 
            print("-" * 50 + "\n")

    output_json_path = os.path.join(OUTPUT_DIR, "inference_results.json")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(all_generated_outputs, f, indent=2, ensure_ascii=False)
    print(f"All inference results saved to {output_json_path}")


if __name__ == "__main__":
    main()