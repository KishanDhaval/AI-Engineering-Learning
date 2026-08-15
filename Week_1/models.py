import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI

load_dotenv()

def get_groq_model(
    temperature=0.2,
    top_p=1.0,
    max_tokens=300,
):
    return ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

def get_mistral_model(
    temperature=0.2,
    top_p=1.0,
    max_tokens=300,
):
    return ChatMistralAI(
        model="mistral-small-latest",
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )