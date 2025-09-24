# import re
# import json
# from importlib.resources import files
# from typing import Literal

# Domain = Literal["financial", "ecommerce", "general", "healthcare"]

# # Load general patterns (shared across all domains)
# GENERAL_PATTERNS = json.loads(
#     files("promptinjection_defender").joinpath("prompt_patterns.json").read_text()
# )

# def load_domain_patterns(domain: Domain) -> list[str]:
#     """Load domain-specific patterns from /domains/ subfolder"""
#     domain_file = f"{domain}_patterns.json"
#     try:
#         return json.loads(
#             files("promptinjection_defender.domains").joinpath(domain_file).read_text()
#         )
#     except FileNotFoundError:
#         return []

# def task_defense(input_text: str, domain: Domain = "general") -> tuple[bool, str]:
#     """
#     Check input against both general and domain-specific rules.
#     Returns: (is_malicious, reason)
#     """
#     # Check general patterns
#     for pattern in GENERAL_PATTERNS:
#         if _pattern_matches(pattern, input_text):
#             return True, "Blocked by general security rules, I am not allowed to answer this."
    
#     # Check domain patterns
#     for pattern in load_domain_patterns(domain):
#         if _pattern_matches(pattern, input_text):
#             return True, f"This prompt is blocked by {domain} security rules, I am not allowed to answer this."
    
#     return False, "OK"

# def _pattern_matches(pattern: str, text: str) -> bool:
#     """Safe pattern matching with error handling"""
#     try:
#         return bool(re.search(pattern, text, re.IGNORECASE))
#     except re.error:
#         return False

# MAIN SEMANTIC CODE THAT WORKS
import re
import json
import numpy as np
from typing import Literal, Tuple
from importlib.resources import files
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

Domain = Literal["financial", "ecommerce", "healthcare", "general"]
class DomainDefender:
    def __init__(self, domain: Domain = "general"):
        self.domain = domain
        self.semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
        self._load_rules()

    def _load_rules(self):
        """Load regex + semantic rules from JSON files."""
        # Load regex patterns
        domain_file = f"{self.domain}_patterns.json"
        try:
            self.regex_rules = json.loads(
                files("promptinjection_defender.domains")
                .joinpath(domain_file)
                .read_text()
            )
        except FileNotFoundError:
            self.regex_rules = []

        # Load semantic phrases
        try:
            self.semantic_phrases = json.loads(
                files("promptinjection_defender.domains")
                .joinpath("semantic_patterns.json")
                .read_text()
            ).get(self.domain, [])
        except FileNotFoundError:
            self.semantic_phrases = []

    def analyze(self, text: str) -> Tuple[bool, str]:
        """Run regex → semantic checks. Always returns (malicious, reason)."""

        # ✅ Ensure text is string
        if not isinstance(text, str):
            return True, "Invalid input: not a string"

        # 1. Regex check
        for pattern in self.regex_rules:
            if re.search(pattern, text, re.IGNORECASE):
                return True, f"⚠️ Blocked by {self.domain} rule: {pattern}"

        # 2. Semantic check
        if self.semantic_phrases:
            try:
                text_embed = self.semantic_model.encode([text])
                phrase_embeds = self.semantic_model.encode(self.semantic_phrases)
                similarity = np.max(cosine_similarity(text_embed, phrase_embeds)[0])
                if similarity > 0.85:
                    return True, f"⚠️ Blocked: semantic similarity {similarity:.2f}"
            except Exception as e:
                return True, f"Semantic check error: {str(e)}"

        return False, "SAFE"
#END OF SEMANTICS CODE THAT WORKS 
