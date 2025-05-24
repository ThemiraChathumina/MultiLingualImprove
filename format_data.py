import json

required_aspects = ['Overall', 'Quality', 'Sizing', 'Packaging', 'Support', 'Description', 'Value']
default_justification = "Not mentioned in the review."
default_rating = 0

input_path = 'C:\\Users\\chath\\Documents\\MultiLingualImprove\\aspect_based_sentiment_results.json'
with open(input_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

reviews = []
responses = []

for i in range(len(data)):
    if data[i]['response'][-1] != '}':
        data[i]['response'] += '}'
    try:
        # Clean response string
        cleaned_response = data[i]['response'].replace('\n', '').replace('\t', '').strip()
        response = json.loads(cleaned_response)

        # Index aspects by name
        aspect_dict = {aspect['name']: aspect for aspect in response.get('aspects', [])}

        for name in required_aspects:
            if name not in aspect_dict:
                # Add missing aspect with defaults
                aspect_dict[name] = {
                    'name': name,
                    'rating': default_rating,
                    'justification': default_justification
                }
            else:
                aspect = aspect_dict[name]

                # Fix missing or invalid justification
                if 'justification' not in aspect or not aspect['justification'].strip():
                    aspect['justification'] = default_justification
                    aspect['rating'] = default_rating

                # Fix non-integer rating
                if 'rating' not in aspect or not isinstance(aspect['rating'], int):
                    aspect['rating'] = default_rating

        # Sort and rebuild aspects
        sorted_aspects = [aspect_dict[name] for name in required_aspects]
        response['aspects'] = sorted_aspects

        responses.append(response)
        reviews.append(data[i]['review'])

    except Exception as e:
        print(f"Error processing index {i}: {e}")
        continue

print(f"Processed {len(responses)} valid responses")

# Save dataset with response as string
dataset_str = []
for i in range(len(responses)):
    dataset_str.append({
        'review': reviews[i],
        'response': str(responses[i])
    })

output_path_str = 'C:\\Users\\chath\\Documents\\MultiLingualImprove\\cleaned_aspect_dataset_str_response.json'
with open(output_path_str, 'w', encoding='utf-8') as f:
    json.dump(dataset_str, f, ensure_ascii=False, indent=4)
print(f"Dataset with string responses written to {output_path_str}")

# Save dataset with proper JSON object responses
dataset = []
for i in range(len(responses)):
    dataset.append({
        'review': reviews[i],
        'response': responses[i]
    })

output_path_json = 'C:\\Users\\chath\\Documents\\MultiLingualImprove\\cleaned_aspect_dataset.json'
with open(output_path_json, 'w', encoding='utf-8') as f:
    json.dump(dataset, f, ensure_ascii=False, indent=4)
print(f"Dataset with structured responses written to {output_path_json}")
