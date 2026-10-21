
from services.preference_services import get_preferences
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

from agent.prompts import HR_SYSTEM_PROMPT
from agent.tools import get_strapi_tools
from services.preference_services import save_preferences
from config.settings import settings


async def build_agent(checkpointer, store, user_id):
    if not settings.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is missing")

    mcp_tools = await get_strapi_tools()

    @tool
    async def remember_hr_preferences(
        response_style: str = "",
        candidate_summary_format: str = "",
    ) -> dict:
        """Save HR preferences explicitly requested by the user.

        response_style: concise or detailed, or empty to leave unchanged.
        candidate_summary_format: comma-separated fields, or empty
        to leave unchanged.
        """
        changes = {}

        if response_style:
            if response_style not in {"concise", "detailed"}:
                return {"error": "Unsupported response style"}

            changes["response_style"] = response_style

        if candidate_summary_format:
            allowed_fields = {
                "skills",
                "experience",
                "notice_period",
                "education",
            }

            fields = [
                value.strip()
                for value in candidate_summary_format.split(",")
                if value.strip()
            ]

            if not fields or not set(fields).issubset(allowed_fields):
                return {"error": "Unsupported summary fields"}

            changes["candidate_summary_format"] = fields

        if not changes:
            return {"error": "No preference changes provided"}

        return await save_preferences(store, user_id, changes)

    model = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        google_api_key=settings.GEMINI_API_KEY,
        temperature=0,
    )
    
    # model = ChatGroq(
    #     model="openai/gpt-oss-120b",
    #     google_api_key=settings.GROQ_API_KEY,
    #     temperature=0,
    # )

    preferences = await get_preferences(store, user_id)

    preference_instructions = (
        f"\n\nCurrent user's saved preferences: {preferences or {}}"
    )
    
    return create_agent(
        model=model,
        tools=[*mcp_tools, remember_hr_preferences],
        system_prompt=HR_SYSTEM_PROMPT + preference_instructions,
        checkpointer=checkpointer,
        store=store,
    )
