import os
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings
from Week_4 import config

load_dotenv()

# Cloud Model (Mistral AI)
cloud_llm = ChatMistralAI(model=config.MISTRAL_MODEL, temperature=0.2)
cloud_embeddings = MistralAIEmbeddings(model=config.MISTRAL_EMBED)

# Local Model (Ollama)
local_llm = ChatOllama(model=config.OLLAMA_MODEL, num_ctx=1024, temperature=0.2)
local_embeddings = OllamaEmbeddings(model=config.OLLAMA_EMBED)
