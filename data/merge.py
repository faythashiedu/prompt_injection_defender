import json
#a script that merges two JSON files containing prompt patterns into one file.
with open("adversarial_prompts.json") as f1, open("custom_prompt_patterns.json") as f2:
    kaggle_data = json.load(f1)
    custom_data = json.load(f2)

combined = list(set(kaggle_data + custom_data))

with open("prompt_patterns.json", "w", encoding="utf-8") as out:
    json.dump(combined, out, indent=2)

print(f"✅ Combined dataset has {len(combined)} prompt patterns.")