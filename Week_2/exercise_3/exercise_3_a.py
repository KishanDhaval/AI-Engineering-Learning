"""
Pydantic parser
pydantic use
"""

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from schema import ReviewAnalysis
from reviews import reviews
from model import llm
import json

parser = PydanticOutputParser(pydantic_object=ReviewAnalysis)

# manualy giving instruction for structure
prompt = ChatPromptTemplate.from_messages([
    ('system', "Analyze the customer review. {format_instructions}"),
    ('human', "{review}")
]).partial(format_instructions = parser.get_format_instructions())


def analyze_with_parser(review: str) -> dict: 
    chain = prompt | llm
    try : 
        raw = chain.invoke({review})
        parsed = parser.parse(raw.content) 
        return parsed.model_dump()
    except Exception as e:
        return {"error": str(e), "raw_output": raw.content}


result =  [analyze_with_parser(r) for r in reviews]

print(json.dumps(
    result,
    indent=4,
    ensure_ascii=False
))