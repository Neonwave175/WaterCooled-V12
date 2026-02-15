from mlx_lm import load, generate

# Load model
model, tokenizer = load("mlx-community/Mistral-7B-Instruct-v0.3-4bit")

# Store conversation history
history = []

def build_prompt(history):
    prompt = ""
    for user, assistant in history:
        prompt += f"<s>[INST] {user} [/INST] {assistant}</s>\n"
    # Add new turn (assistant will complete this)
    prompt += "<s>[INST] "
    return prompt

print("Chat started. Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        break

    # Build prompt including previous conversation
    history.append((user_input, ""))  # placeholder for assistant
    prompt = build_prompt(history)

    # Generate response
    response = generate(model, tokenizer, prompt=prompt, max_tokens=300)

    print("Assistant:", response)

    # Save assistant response
    history[-1] = (user_input, response)
