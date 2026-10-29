import os
from dotenv import load_dotenv

load_dotenv()


class Settings: 
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    STRAPI_MCP_TOKEN: str = os.getenv("STRAPI_MCP_TOKEN", "")
    STRAPI_API_URL: str = os.getenv("STRAPI_API_URL", "")
    DATABASE_URL: str = os.getenv("DATABASE_URL", "")

    # LangSmith Observability (Optional: enabled automatically when key is provided)
    LANGCHAIN_TRACING_V2: str = os.getenv("LANGCHAIN_TRACING_V2", "false")
    LANGCHAIN_API_KEY: str = os.getenv("LANGCHAIN_API_KEY", "")
    LANGCHAIN_PROJECT: str = "innvonix-careers-hub"
    LANGCHAIN_ENDPOINT: str = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

    def validate(self) -> None:
        missing = []
        if not self.GEMINI_API_KEY and not self.GROQ_API_KEY:
            missing.append("GEMINI_API_KEY or GROQ_API_KEY")
        if not self.STRAPI_MCP_TOKEN:
            missing.append("STRAPI_MCP_TOKEN")
        if not self.STRAPI_API_URL:
            missing.append("STRAPI_API_URL")
        if not self.DATABASE_URL:
            missing.append("DATABASE_URL")
        
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

    def setup_langsmith(self) -> None:
        """Enable LangSmith tracing automatically if an API key is configured."""
        if self.LANGCHAIN_API_KEY:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_API_KEY"] = self.LANGCHAIN_API_KEY
            os.environ["LANGCHAIN_PROJECT"] = self.LANGCHAIN_PROJECT
            os.environ["LANGCHAIN_ENDPOINT"] = self.LANGCHAIN_ENDPOINT


settings = Settings()
settings.validate()
settings.setup_langsmith()
