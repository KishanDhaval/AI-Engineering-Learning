import json
from models import get_groq_model, get_mistral_model
from prompts import PROMPTS
from utils import run_model


groq = get_groq_model(
    temperature=0.2,
    top_p=1.0,
    max_tokens=300,
)

mistral = get_mistral_model(
    temperature=0.2,
    top_p=1.0,
    max_tokens=300,
)


results = {}

for prompt_name, prompt in PROMPTS.items():

    print("\n" + "=" * 80)
    print(f"PROMPT: {prompt_name}")
    print("=" * 80)

    print("\nRunning Groq...")

    groq_result = run_model(
        groq,
        prompt
    )

    print("\nRunning Mistral...")

    mistral_result = run_model(
        mistral,
        prompt
    )

    results[prompt_name] = {
        "groq": groq_result,
        "mistral": mistral_result,
    }

    print("\n--- GROQ ---")
    print(groq_result["response"])

    print("\n--- MISTRAL ---")
    print(mistral_result["response"])