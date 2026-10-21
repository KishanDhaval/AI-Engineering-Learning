import os
from dotenv import load_dotenv

load_dotenv()


class Settings: 
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    STRAPI_MCP_TOKEN: str = os.getenv("STRAPI_MCP_TOKEN", "")
    STRAPI_API_URL: str = os.getenv("STRAPI_API_URL", "")
    DATABASE_URL: str = os.getenv("DATABASE_URL", "")

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


settings = Settings()
settings.validate()
