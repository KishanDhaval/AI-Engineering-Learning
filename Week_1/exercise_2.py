from models import get_groq_model


# PROMPTS

prompts = [
    "Explain what an LLM is in simple words.",
    "Write a short story about an AI engineer.",
    "Write a Python function to call database.",
]

hallucination_prompts = [
    "Tell me about the first president of the fictional country Velmora.",
    "Explain how the Quantum Neural Compression Protocol works.",
    "What happened at the Global AI Conference held on February 31, 2026?",
    "What is the latest AI incident that happened today?",
    "Explain how a perpetual motion machine can produce unlimited energy.",
]


# TEMPERATURE

temperatures = [0.0, 0.5, 1.0]

for temperature in temperatures:

    print("\n" + "=" * 60)
    print(f"TEMPERATURE = {temperature}")
    print("=" * 60)

    model = get_groq_model(
        temperature=temperature,
        top_p=1.0,
        max_tokens=300,
    )

    # Normal prompts
    for prompt in prompts:

        response = model.invoke(prompt)

        print("\nPrompt:", prompt)
        print("Response:", response.content)

    # Hallucination prompts
    for prompt in hallucination_prompts:

        response = model.invoke(prompt)

        print("\nHallucination Prompt:", prompt)
        print("Response:", response.content)


# TOP-P

top_p_values = [0.1, 0.5, 1.0]

for top_p in top_p_values:

    print("\n" + "=" * 60)
    print(f"TOP-P = {top_p}")
    print("=" * 60)

    model = get_groq_model(
        temperature=0.7,
        top_p=top_p,
        max_tokens=300,
    )

    for prompt in prompts:

        response = model.invoke(prompt)

        print("\nPrompt:", prompt)
        print("Response:", response.content)

    for prompt in hallucination_prompts:

        response = model.invoke(prompt)

        print("\nHallucination Prompt:", prompt)
        print("Response:", response.content)

# MAX TOKENS

max_token_values = [50, 150, 300]

for max_tokens in max_token_values:

    print("\n" + "=" * 60)
    print(f"MAX TOKENS = {max_tokens}")
    print("=" * 60)

    model = get_groq_model(
        temperature=0.7,
        top_p=1.0,
        max_tokens=max_tokens,
    )

    for prompt in prompts:

        response = model.invoke(prompt)

        print("\nPrompt:", prompt)
        print("Response:", response.content)

    for prompt in hallucination_prompts:

        response = model.invoke(prompt)

        print("\nHallucination Prompt:", prompt)
        print("Response:", response.content)