from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)

# 1) ZERO-SHOT
zero_shot = ChatPromptTemplate.from_messages([
    ("system", "Analyze the customer review."),
    ("human", "{review}"),
])


# 2) FEW-SHOT
few_shot_examples = [
    {
        "review": "Shipped late and the box was crushed, but the item inside was fine.",
        "output": '{"sentiment": "neutral", "key_issues": ["late shipping", "crushed packaging"], "summary": "Shipping was late and the box was crushed, though the item itself was undamaged."}',
    },
    {
        "review": "Absolutely love this, works perfectly and looks great.",
        "output": '{"sentiment": "positive", "key_issues": [], "summary": "Reviewer loves the product; it works perfectly and looks great."}',
    },
]

example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{review}"),
    ("ai", "{output}"),
])

few_shot_examples_prompt = FewShotChatMessagePromptTemplate(
    examples=few_shot_examples,
    example_prompt=example_prompt,
)

few_shot = ChatPromptTemplate.from_messages([
    (
        "system",
        "Analyze the customer review and extract structured data."
    ),
    few_shot_examples_prompt,
    ("human", "{review}"),
])


# 3) PRECISE RULES
precise_rules = ChatPromptTemplate.from_messages([
    (
        "system",
        "Analyze the customer review and extract structured data.\n\n"
        "Rules:\n"
        "- sentiment: 'positive' only if the reviewer is clearly satisfied overall, "
        "'negative' only if clearly dissatisfied, 'neutral' for anything mixed, "
        "lukewarm, or purely factual.\n"
        "- key_issues: only concrete problems the reviewer names. Never include "
        "praise. Short noun phrases, deduplicated, max 5, empty list if none.\n"
        "- summary: exactly one sentence, <=25 words, plain language, do not "
        "repeat the sentiment word itself."
    ),
    ("human", "{review}"),
])


# 4) REASONING-BASED
cot_then_extract = ChatPromptTemplate.from_messages([
    (
        "system",
        "Before answering, silently think through what the reviewer liked and "
        "disliked, and which points are genuine problems vs. praise. "
        "Do NOT show this reasoning. Respond with ONLY the structured output."
    ),
    ("human", "{review}"),
])


# 5) ROLE-BASED
role_based = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a senior customer insights analyst who reviews product feedback "
        "for a living. Analyze the following review the way a professional "
        "analyst would."
    ),
    ("human", "{review}"),
])


PROMPT_VARIANTS = {
    "zero_shot": zero_shot,
    "few_shot": few_shot,
    "precise_rules": precise_rules,
    "cot_then_extract": cot_then_extract,
    "role_based": role_based,
}