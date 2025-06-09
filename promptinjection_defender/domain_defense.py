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
        """Load both regex and semantic rules from JSON files"""
        # Load existing domain regex
        domain_file = f"{self.domain}_patterns.json"
        self.regex_rules = json.loads(
            files("promptinjection_defender.domains").joinpath(domain_file).read_text()
        )        
        # Load new semantic phrases
        try:
            self.semantic_phrases = json.loads(
                files("promptinjection_defender.domains").joinpath("semantic_patterns.json").read_text()
            ).get(self.domain, [])
        except FileNotFoundError:
            self.semantic_phrases = []
    def analyze(self, text: str) -> Tuple[bool, str]:
        """Defense pipeline: regex -> semantics"""
        # 1. Domain regex check (existing)
        for pattern in self.regex_rules:
            if re.search(pattern, text, re.IGNORECASE):
                return True, f"Blocked by {self.domain} rule: {pattern}"
        # 3. Semantic check (new)
        if self.semantic_phrases:
            text_embed = self.semantic_model.encode([text])
            phrase_embeds = self.semantic_model.encode(self.semantic_phrases)
            similarity = np.max(cosine_similarity(text_embed, phrase_embeds)[0])
            if similarity > 0.85:
                return True, f"Blocked: similar to known attacks ({similarity:.2f})"

        return False, "OK"
    

# import re
# import json
# import numpy as np
# from importlib.resources import files
# from typing import Literal, Optional
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
# from functools import lru_cache


# Domain = Literal["financial", "ecommerce", "general", "healthcare"]

# class DomainDefender:
#     @lru_cache(maxsize=1000)
#     def __init__(self, domain: Domain = "general"):
#         self.domain = domain
#         self.encoder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
#         self.encoder.encode("warmup")
#         self._load_patterns()
#         self._init_attack_embeddings()
        
#     def _load_patterns(self):
#         """Load regex patterns and semantic examples"""
#         # General patterns
#         self.general_patterns = json.loads(
#             files("promptinjection_defender").joinpath("prompt_patterns.json").read_text()
#         )
        
#         # Domain patterns
#         domain_file = f"{self.domain}_patterns.json"
#         try:
#             self.domain_patterns = json.loads(
#                 files("promptinjection_defender.domains").joinpath(domain_file).read_text()
#             )
#         except FileNotFoundError:
#             self.domain_patterns = []
            
#         # Semantic attack examples
#         self.semantic_examples = self._load_semantic_examples()
    
#     def _load_semantic_examples(self) -> list[str]:
#         """Load domain-specific attack examples for semantic matching"""
#         example_file = f"{self.domain}_patterns.json"
#         try:
#             return json.loads(
#                 files("promptinjection_defender.domains").joinpath(example_file).read_text()
#             )
#         except FileNotFoundError:
#             return []
    
#     def _init_attack_embeddings(self):
#         """Precompute embeddings for known attacks"""
#         self.attack_embeddings = self.encoder.encode(
#             self.semantic_examples + self.domain_patterns
#         ) if self.semantic_examples else None
    
#     def analyze(self, text: str) -> tuple[bool, str, Optional[dict]]:
#         """
#         Advanced analysis with risk breakdown
#         Returns: (is_malicious, reason, risk_details)
#         """
#         # Check classic regex patterns first (fast)
#         regex_result = self._check_regex_patterns(text)
#         if regex_result[0]:
#             return (*regex_result, None)
        
#         # Semantic and contextual analysis
#         risk_details = self._calculate_risk(text)
        
#         if risk_details["total_risk"] > 0.7:  # Threshold configurable per domain
#             return True, "Blocked by semantic risk analysis", risk_details
        
#         return False, "OK", risk_details
    
#     def _check_regex_patterns(self, text: str) -> tuple[bool, str]:
#         """Fast pattern matching"""
#         for pattern in self.general_patterns + self.domain_patterns:
#             if self._safe_regex_match(pattern, text):
#                 return True, f"Blocked by {self.domain} security rules"
#         return False, "OK"
    
#     def _safe_regex_match(self, pattern: str, text: str) -> bool:
#         try:
#             return bool(re.search(pattern, text, re.IGNORECASE))
#         except re.error:
#             return False
    
#     def _calculate_risk(self, text: str) -> dict:
#         """Compute multi-factor risk score"""
#         risk = {
#             "semantic_similarity": 0.0,
#             "keyword_risk": 0.0,
#             "entropy_risk": 0.0,
#             "total_risk": 0.0
#         }
        
#         # Semantic similarity to known attacks
#         if self.attack_embeddings is not None:
#             text_embed = self.encoder.encode([text])
#             similarities = cosine_similarity(text_embed, self.attack_embeddings)
#             risk["semantic_similarity"] = float(np.max(similarities))
        
#         # Domain keyword weighting
#         keywords = {
#             "financial": ["transfer", "override", "balance", "credentials"],
#             "ecommerce": ["discount", "refund", "order", "payment"],
#             "healthcare": ["prescribe", "diagnosis", "patient", "record"]
#         }.get(self.domain, [])
        
#         risk["keyword_risk"] = min(
#             1.0, 
#             sum(text.lower().count(kw) * 0.2 for kw in keywords)  # 0.2 per keyword hit
#         )
        
#         # Entropy for obfuscation detection
#         risk["entropy_risk"] = self._normalized_entropy(text)
        
#         # Weighted total (adjust weights per domain)
#         weights = {
#             "financial": [0.6, 0.3, 0.1],  # Emphasize semantic matches
#             "ecommerce": [0.4, 0.4, 0.2],
#             "default": [0.5, 0.3, 0.2]
#         }
        
#         w = weights.get(self.domain, weights["default"])
#         risk["total_risk"] = (
#             w[0] * risk["semantic_similarity"] +
#             w[1] * risk["keyword_risk"] +
#             w[2] * risk["entropy_risk"]
#         )
        
#         return risk
    
#     def _normalized_entropy(self, text: str) -> float:
#         """Enhanced entropy detection for financial attacks"""
#         if not text:
#             return 0.0
        
#         # Count alphanumeric vs special chars
#         alpha = sum(c.isalnum() for c in text)
#         special = len(text) - alpha
#         special_ratio = special / len(text)
        
#         # Shannon entropy
#         from collections import Counter
#         counts = Counter(text.lower())  # Case insensitive
#         probs = [c/len(text) for c in counts.values()]
#         entropy = -sum(p * np.log2(p) for p in probs if p > 0)
        
#         # Normalize with special char boost
#         base_entropy = min(1.0, max(0, (entropy - 3.5) / 3.5))  # 3.5-7 → 0-1
#         return min(1.0, base_entropy + (special_ratio * 2))  # Penalize special chars


# new trial

# import re
# import json
# import numpy as np
# import string
# from collections import Counter
# from functools import lru_cache
# from typing import Literal, Tuple, Optional
# from importlib.resources import files
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity

# Domain = Literal["financial", "ecommerce", "general", "healthcare"]

# class DomainDefender:
#     """
#     Efficient, reusable prompt-injection defense:
#      - Regex + domain patterns
#      - Structural obfuscation detection
#      - Semantic similarity (with cached lightweight model)
#      - Entropy-based obfuscation
#      - Contextual keyword risk
#     """

#     def __init__(self, domain: Domain = "general"):
#         self.domain = domain
#         # Use a smaller, faster model
#         self.encoder = SentenceTransformer("paraphrase-MiniLM-L3-v2")
#         # Pre-warm and cache encode function
#         self._encode("__warmup__")
#         # Load patterns & examples
#         self.general_patterns, self.domain_patterns = self._load_patterns()
#         self.attack_texts = (self.domain_patterns + self._load_semantic_examples())[:20]
#         # Precompute embeddings of top 20 attacks
#         self.attack_embeddings = [self._encode(t) for t in self.attack_texts]

#     @lru_cache(maxsize=1000)
#     def _encode(self, text: str):
#         return self.encoder.encode([text])[0]

#     def _load_patterns(self):
#         gen = json.loads(
#             files("promptinjection_defender")
#             .joinpath("prompt_patterns.json")
#             .read_text()
#         )
#         try:
#             dom = json.loads(
#                 files("promptinjection_defender.domains")
#                 .joinpath(f"{self.domain}_patterns.json")
#                 .read_text()
#             )
#         except FileNotFoundError:
#             dom = []
#         return gen, dom

#     def _load_semantic_examples(self) -> list[str]:
#         try:
#             return json.loads(
#                 files("promptinjection_defender.domains")
#                 .joinpath(f"{self.domain}_patterns.json")
#                 .read_text()
#             )
#         except FileNotFoundError:
#             return []

#     def _has_structural_obfuscation(self, text: str) -> bool:
#         # e.g. a/c/c/o/u/n/t or b.y.p.a.s.s
#         if re.search(r"\b(?:\w[/\.]){2,}\w\b", text):
#             return True
#         # unicode heavy
#         return sum(c not in string.printable for c in text) > 3

#     def _normalized_entropy(self, text: str) -> float:
#         if not text:
#             return 0.0
#         counts = Counter(text.lower())
#         probs = [c/len(text) for c in counts.values()]
#         entropy = -sum(p * np.log2(p) for p in probs if p > 0)
#         special = sum(not c.isalnum() for c in text) / len(text)
#         base = min(1.0, max(0, (entropy - 3.5) / 3.5))
#         return min(1.0, base + special * 2)

#     def check(self, text: str) -> Tuple[bool, str, Optional[dict]]:
#         # 1) Regex
#         for p in self.general_patterns + self.domain_patterns:
#             try:
#                 if re.search(p, text, re.IGNORECASE):
#                     return True, f"Blocked by pattern: {p[:50]}...", None
#             except re.error:
#                 continue

#         # 2) Structural obfuscation
#         if self._has_structural_obfuscation(text):
#             return True, "Blocked structural obfuscation.", None

#         # 3) Semantic similarity
#         txt_emb = self._encode(text)
#         sims = cosine_similarity(txt_emb, self.attack_embeddings)[0]
#         max_sim = float(np.max(sims))
#         if max_sim > 0.75:
#             idx = int(np.argmax(sims))
#             return True, f"Semantic match to attack example: '{self.attack_texts[idx][:50]}...'", None

#         # 4) Entropy
#         ent = self._normalized_entropy(text)
#         if ent > 0.7:
#             return True, "Blocked high-entropy (obfuscated) input.", None

#         # 5) Contextual keyword risk
#         kw_map = {
#             "financial": ["transfer","override","balance","credentials"],
#             "ecommerce": ["discount","refund","order","payment"],
#             "healthcare": ["prescribe","diagnosis","patient","record"]
#         }
#         kws = kw_map.get(self.domain, [])
#         kw_risk = min(1.0, sum(text.lower().count(k)*0.2 for k in kws))
#         if kw_risk > 0.6:
#             return True, "Blocked due to domain-specific keyword risk.", None

#         # Passed all checks
#         return False, "OK", {
#             "semantic_similarity": max_sim,
#             "entropy": ent,
#             "keyword_risk": kw_risk
#         }


# take 4

# Domain = Literal["financial", "ecommerce", "general", "healthcare"]

# # --- Configuration --- #
# SIMILARITY_THRESHOLD = 0.85  # Adjust based on testing
# ENTROPY_THRESHOLD = 4.5      # For obfuscation detection
# MODEL_NAME = "all-MiniLM-L6-v2"  # Lightweight but effective

# # --- Load Embedding Model (Singleton) --- #
# _embedding_model = None
# def get_embedding_model():
#     global _embedding_model
#     if _embedding_model is None:
#         _embedding_model = SentenceTransformer(MODEL_NAME)
#     return _embedding_model

# # --- Semantic Attack Patterns --- #
# class SemanticDefender:
#     def __init__(self, domain: Domain = "general"):
#         self.domain = domain
#         self.model = get_embedding_model()
#         self.domain_patterns = self._load_semantic_patterns()
        
#     def _load_semantic_patterns(self) -> dict:
#         """Load pre-embedded attack patterns for the domain"""
#         try:
#             patterns_file = f"{self.domain}_semantic_patterns.json"
#             raw_data = json.loads(
#                 files("promptinjection_defender.domains").joinpath(patterns_file).read_text()
#             )
#             return {
#                 "phrases": raw_data["phrases"],
#                 "embeddings": np.array(raw_data["embeddings"])
#             }
#         except FileNotFoundError:
#             return {"phrases": [], "embeddings": np.array([])}

#     def analyze(self, text: str) -> dict:
#         """
#         Enhanced analysis with:
#         - Rule-based checks
#         - Semantic similarity
#         - Entropy analysis
#         Returns: {
#             "is_malicious": bool,
#             "risk_score": float (0-1),
#             "triggers": list[str],
#             "explanation": str
#         }
#         """
#         result = {
#             "is_malicious": False,
#             "risk_score": 0.0,
#             "triggers": [],
#             "explanation": "OK"
#         }

#         # --- 1. First Pass: Rule-Based Checks --- #
#         is_malicious, reason = task_defense(text, self.domain)
#         if is_malicious:
#             result.update({
#                 "is_malicious": True,
#                 "risk_score": 1.0,
#                 "triggers": ["rule-based"],
#                 "explanation": reason
#             })
#             return result

#         # --- 2. Semantic Analysis --- #
#         if len(self.domain_patterns["phrases"]) > 0:
#             text_embedding = self.model.encode([text])
#             similarities = cosine_similarity(
#                 text_embedding, 
#                 self.domain_patterns["embeddings"]
#             )[0]
            
#             max_sim_idx = np.argmax(similarities)
#             max_sim = similarities[max_sim_idx]
            
#             if max_sim > SIMILARITY_THRESHOLD:
#                 result["triggers"].append(
#                     f"semantic_match ({max_sim:.2f} similarity to: "
#                     f"{self.domain_patterns['phrases'][max_sim_idx][:50]}...)"
#                 )
#                 result["risk_score"] = max(  # Risk is the highest detected
#                     result["risk_score"], 
#                     float(max_sim)
#                 )

#         # --- 3. Entropy Check (Obfuscation) --- #
#         entropy = self._calculate_entropy(text)
#         if entropy > ENTROPY_THRESHOLD:
#             result["triggers"].append(f"high_entropy ({entropy:.2f})")
#             result["risk_score"] = max(result["risk_score"], 0.7)  # High entropy = suspicious

#         # --- Final Decision --- #
#         if result["risk_score"] > 0.7:  # Threshold adjustable per domain
#             result.update({
#                 "is_malicious": True,
#                 "explanation": f"Blocked due to: {', '.join(result['triggers'])}"
#             })

#         return result

#     def _calculate_entropy(self, text: str) -> float:
#         """Calculate Shannon entropy for obfuscation detection"""
#         from collections import Counter
#         import math
#         counts = Counter(text)
#         probs = [c / len(text) for c in counts.values()]
#         return -sum(p * math.log(p) for p in probs)

# # --- Original Rule-Based Defense (Kept for compatibility) --- #
# # ... (Keep your existing task_defense() and helper functions) ...

# take 6

# Domain = Literal["financial", "ecommerce", "general", "healthcare"]

# # --- Configuration --- #
# SEMANTIC_THRESHOLD = 0.85    # Similarity score cutoff
# ENTROPY_THRESHOLD = 4.5      # Obfuscation detection
# MODEL_NAME = "all-MiniLM-L6-v2"  # Lightweight embedding model

# class DomainDefender:
#     def __init__(self, domain: Domain = "general"):
#         self.domain = domain
#         self.embedding_model = SentenceTransformer(MODEL_NAME)
#         self._load_patterns()

#     def _load_patterns(self) -> None:
#         """Load both regex rules and semantic attack patterns"""
#         # Rule-based patterns
#         self.general_rules = json.loads(
#             files("promptinjection_defender").joinpath("prompt_patterns.json").read_text()
#         )
#         try:
#             domain_file = f"{self.domain}_patterns.json"
#             self.domain_rules = json.loads(
#                 files("promptinjection_defender.domains").joinpath(domain_file).read_text()
#             )
#         except FileNotFoundError:
#             self.domain_rules = []

#         # Semantic attack patterns
#         try:
#             semantic_file = f"{self.domain}_patterns.json"
#             data = json.loads(
#                 files("promptinjection_defender.domains").joinpath(semantic_file).read_text()
#             )
#             self.semantic_phrases = data["phrases"]
#             self.semantic_embeddings = np.array(data["embeddings"])
#         except FileNotFoundError:
#             self.semantic_phrases = []
#             self.semantic_embeddings = np.array([])

#     def analyze(self, text: str) -> dict:
#         """Unified defense pipeline with risk scoring"""
#         result = {
#             "is_malicious": False,
#             "risk_score": 0.0,
#             "triggers": [],
#             "explanation": "OK"
#         }

#         # --- 1. Fast Rule-Based Checks --- #
#         if self._check_rules(text):
#             result.update({
#                 "is_malicious": True,
#                 "risk_score": 1.0,
#                 "triggers": ["rule-based"],
#                 "explanation": f"Blocked by {self.domain} security rules"
#             })
#             return result

#         # --- 2. Semantic Analysis --- #
#         if len(self.semantic_phrases) > 0:
#             text_embed = self.embedding_model.encode([text])
#             similarities = cosine_similarity(
#                 text_embed, 
#                 self.semantic_embeddings
#             )[0]
#             max_sim = np.max(similarities)
#             if max_sim > SEMANTIC_THRESHOLD:
#                 result["risk_score"] = max(result["risk_score"], max_sim)
#                 result["triggers"].append(
#                     f"semantic_match ({max_sim:.2f} similarity)"
#                 )

#         # --- 3. Entropy Check --- #
#         entropy = self._calculate_entropy(text)
#         if entropy > ENTROPY_THRESHOLD:
#             result["risk_score"] = max(result["risk_score"], 0.7)
#             result["triggers"].append(f"high_entropy ({entropy:.2f})")

#         # --- Final Decision --- #
#         if result["risk_score"] >= 0.7:  # Configurable threshold
#             result.update({
#                 "is_malicious": True,
#                 "explanation": f"Blocked: {', '.join(result['triggers'])}"
#             })

#         return result

#     def _check_rules(self, text: str) -> bool:
#         """Apply all regex rules"""
#         for pattern in self.general_rules + self.domain_rules:
#             try:
#                 if re.search(pattern, text, re.IGNORECASE):
#                     return True
#             except re.error:
#                 continue
#         return False

#     @staticmethod
#     def _calculate_entropy(text: str) -> float:
#         """Detect obfuscation via Shannon entropy"""
#         from collections import Counter
#         import math
#         counts = Counter(text.lower())
#         probs = [c / len(text) for c in counts.values()]
#         return -sum(p * math.log(p) for p in probs if p > 0)