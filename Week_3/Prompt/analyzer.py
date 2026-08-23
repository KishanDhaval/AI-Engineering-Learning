import time
from schema import ReviewAnalysis
from model import llm
from prompts import PROMPT_VARIANTS

structured_llm = llm.with_structured_output(ReviewAnalysis)

def analyze(review: str, prompt_name: str = "zero_shot", max_retries: int = 2) -> dict:
    prompt = PROMPT_VARIANTS[prompt_name]
    chain = prompt | structured_llm

    for attempt in range(max_retries + 1):
        try:
            res = chain.invoke({"review": review})
            if isinstance(res, ReviewAnalysis):
                return {"ok": True, **res.model_dump()}
            elif isinstance(res, dict):
                return {"ok": True, **res}
        except Exception as e:
            if attempt < max_retries:
                time.sleep(1.0)
            else:
                return {"ok": False, "error": str(e)}

    return {"ok": False, "error": "Unknown failure"}