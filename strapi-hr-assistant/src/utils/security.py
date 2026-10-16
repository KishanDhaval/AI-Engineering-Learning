
def validate_question(question: str) -> str:
    if not question or not question.strip(): 
        raise ValueError("Question cannot be empty")
    return question.strip()
