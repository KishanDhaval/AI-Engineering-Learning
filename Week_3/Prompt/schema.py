
from pydantic import BaseModel, Field
from typing import List, Literal


class ReviewAnalysis(BaseModel): 
    sentiment: Literal["positive", "negative", "neutral"]  = Field(
        description = "Overall sentiment of the review. Must be exactly one of: positive, negative, neutral"
    )
    key_issues: List[str] = Field(
        description="List of specific problems mentioned in the review; empty list [] if none."
    )
    summary: str = Field(
        description="One-sentence summary of the review"
    )