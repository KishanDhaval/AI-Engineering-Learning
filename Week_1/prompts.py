PROMPTS = {
    "general_knowledge": """
Explain how the Internet works to a beginner.
Use a simple analogy and then give a technically accurate explanation.
Keep the answer under 200 words.
""",

    "reasoning": """
A farmer has 17 sheep. All but 9 die.
How many sheep are left?

Explain your reasoning clearly and give the final answer.
""",

    "coding": """
Write a Python function that finds the first non-repeating character
in a string.

Requirements:
1. Return the character and its index.
2. Return (-1, -1) if no such character exists.
3. Explain the time and space complexity.
""",

    "debugging": """
Find the bug in this Python code:

numbers = [1, 2, 3, 4, 5]

for i in range(len(numbers)):
    if numbers[i] % 2 == 0:
        numbers.remove(numbers[i])

print(numbers)

Explain why the bug occurs and provide a correct solution.
""",

    "structured_instruction": """
Give me exactly 5 ways an AI engineer can reduce hallucinations
in an LLM application.

Return the answer as a numbered list.
Do not include anything outside the list.
""",
}