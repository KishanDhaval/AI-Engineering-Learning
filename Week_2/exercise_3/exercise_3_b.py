"""
WITH structured output
"""

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from schema import ReviewAnalysis
from reviews import reviews
from model import llm
import json
import time

prompt = ChatPromptTemplate.from_messages([
    ('system', "Analyze the customer review."),
    ('human', "{review}")
])

parser = PydanticOutputParser(pydantic_object=ReviewAnalysis)

# structured llm with ReviewAnalysis
structured_llm = llm.with_structured_output(ReviewAnalysis)

chain = prompt | structured_llm
start = time.perf_counter()

results = chain.batch(
    [review for review in reviews]
)

# list of pydantic object to dict
result = [
    item.model_dump()
    for item in results
]

print(json.dumps(
    result,
    indent=4,
    ensure_ascii=False
))

print("TIME: ", round(time.perf_counter() - start, 3))