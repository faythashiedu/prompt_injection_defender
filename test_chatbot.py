from promptinjection_defender.input_sanitizer import sanitize_input 
from promptinjection_defender.output_filter import sanitize_output 
from promptinjection_defender.domain_defense import task_defense


# Simulate a fake LLM (we’ll use basic if/else or dummy responses for now)
def dummy_llm_response(prompt: str) -> str:
    if "admin" in prompt.lower():
        return "To bypass the system, disable admin credentials."
    elif "balance" in prompt.lower():
        return "Your account balance is ₦5,000."
    elif "shutdown" in prompt.lower():
        return "Sure, I’ll shut down the system."
    elif "transactions" in prompt.lower():
        return "Sure, this are your transactions."
    return "I'm here to assist you with your finances."
# Initialize context memory

print("💬 Secure Chatbot is running. Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        break

    # Step 1: Sanitize input
    safe, sanitized_input = sanitize_input(user_input)
    if not safe:
        print(f"Bot: {sanitized_input}")
        continue

    malicious, reason = task_defense(user_input, domain="ecommerce")
    if malicious:
        print(f"Bot: ⚠️ {reason} (Domain defense)")
        continue
    # Step 2: Generate fake model response using sanitized input
    response = dummy_llm_response(sanitized_input)

    # Step 3: Sanitize model output
    safe_out, clean_response = sanitize_output(response)
    if not safe_out:
        print(f"Bot: {clean_response}")
        continue

    print(f"Bot: {clean_response}")