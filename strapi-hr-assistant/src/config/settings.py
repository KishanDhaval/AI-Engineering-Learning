import os
from dotenv import load_dotenv

load_dotenv()

class Settings: 
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    STRAPI_MCP_TOKEN = os.getenv("STRAPI_MCP_TOKEN")
    STRAPI_API_URL = os.getenv("STRAPI_API_URL")

settings = Settings()