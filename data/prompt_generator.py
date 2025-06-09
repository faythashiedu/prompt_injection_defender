import json

#a script that generates prompt injection patterns based on a set of verbs, adjectives, objects, and prepositions.

def generate_injection_keywords():
    verbs = ["Ignore", "Disregard", "Skip", "Forget", "Bypass", "Delete", "Remove", "Override", "Simulate"]
    adjectives = ["", "previous", "prior", "earlier"]
    objects = ["instructions", "context", "input", "messages", "command", "logs", "history", "jailbreak"]
    prepositions = ["", "and start over", "and reset", "and continue"]

    phrases = []
    for verb in verbs:
        for adj in adjectives:
            for obj in objects:
                for prep in prepositions:
                    phrase = f"{verb} {adj} {obj} {prep}".strip()
                    phrases.append(phrase.lower())

    return list(set(phrases))

# Save to JSON
custom_phrases = generate_injection_keywords()
with open("custom_prompt_patterns.json", "w", encoding="utf-8") as f:
    json.dump(custom_phrases, f, indent=2)

print(f"✅ Generated {len(custom_phrases)} custom patterns.")
