import csv
import json
import os

# This script extracts prompts from CSV files in a specified directory and saves them to a JSON file.
# Folder where your CSV files are stored
DATA_DIR = "Generated Prompts"

# Output file
output_file = "adversarial_prompts.json"
all_prompts = []

# Loop through all .csv files in the folder
for file in os.listdir(DATA_DIR):
    if file.endswith(".csv"):
        path = os.path.join(DATA_DIR, file)
        with open(path, encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                prompt = row[0].strip()
                if prompt and prompt not in all_prompts:
                    all_prompts.append(prompt.lower())  # normalize

# Save to JSON
with open(output_file, "w", encoding="utf-8") as out:
    json.dump(all_prompts, out, indent=2)

print(f"✅ Extracted {len(all_prompts)} prompts into {output_file}")
