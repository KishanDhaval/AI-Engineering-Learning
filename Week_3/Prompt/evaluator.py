import json
import re
from model import llm

# sentiment is graded with plain exact-match in run_experiment.py — it's an
# enum, so there's nothing fuzzy to judge. key_issues and summary are free
# text with valid paraphrasing, so exact-match or ROUGE would unfairly
# punish a correct answer worded differently. An LLM judge scores on
# *meaning* instead.

JUDGE_PROMPT = """You are grading an AI's extraction against a human-written reference for a customer review. Judge on meaning, not exact wording.

Review: {review}

Reference key_issues: {gold_issues}
Predicted key_issues: {pred_issues}

Reference summary: {gold_summary}
Predicted summary: {pred_summary}

Score each 1-5:
- issues_score: 5 = predicted issues cover the same real problems as the reference, no hallucinated issues, no missed issues. Lower for each hallucination or omission.
- summary_score: 5 = predicted summary is faithful to the review and captures the same core point as the reference. Lower for missing the point, adding unsupported claims, or being unclear.

Respond with ONLY this JSON, no other text: {{"issues_score": <int>, "summary_score": <int>}}"""


def judge(review: str, gold_issues, pred_issues, gold_summary: str, pred_summary: str) -> dict:
    prompt = JUDGE_PROMPT.format(
        review=review,
        gold_issues=gold_issues,
        pred_issues=pred_issues,
        gold_summary=gold_summary,
        pred_summary=pred_summary,
    )
    try:
        raw = llm.invoke(prompt)
        text = raw.content
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            text = match.group(0)
        return json.loads(text)
    except Exception:
        return {"issues_score": None, "summary_score": None}

