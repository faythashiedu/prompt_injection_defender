import re
import json
from importlib.resources import files


def load_patterns():
    file_path = files("promptinjection_defender").joinpath("prompt_patterns.json")
    with open(file_path, encoding="utf-8") as f:
        return json.load(f)

DANGEROUS_PATTERNS = load_patterns()

def sanitize_input(user_input: str) -> tuple[bool, str]:
    for pattern in DANGEROUS_PATTERNS:
        try:
            if re.search(pattern, user_input, re.IGNORECASE):
                return False, "⚠️This Prompt blocked for potential injection. Please rephrase your question."
        except re.error:
            continue  # skip invalid regex patterns
    return True, user_input