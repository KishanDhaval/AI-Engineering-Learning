
from pydantic import BaseModel, Field
from typing import List


class ReviewAnalysis(BaseModel): 
    sentiment: str = Field(description = "One of: positive, negative, nuatral")
    key_issues: List[str] = Field(description="Specific problem mentioned; empty list if none.")
    summary: str = Field(description="One-sentence summary of the review")