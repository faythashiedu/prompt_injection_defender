def sanitize_output(model_response: str) -> tuple[bool, str]:
    forbidden_terms = ["disable security", "log information", "admin credentials", "system rules", "system instructions", "admin panel", "user password"]
    for term in forbidden_terms:
        if term.lower() in model_response.lower():
            return False, "⚠️ Response blocked due to sensitive output. Please rephrase your question."
    return True, model_response