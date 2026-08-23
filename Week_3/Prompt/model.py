from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

llm = ChatMistralAI(
    model="mistral-small-latest",
    temperature=0.2,
    max_retries=2,
    max_tokens=1000
)

