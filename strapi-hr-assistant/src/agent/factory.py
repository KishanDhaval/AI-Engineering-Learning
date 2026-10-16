from agent.prompts import HR_SYSTEM_PROMPT
from agent.tools import get_strapi_tools
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from config.settings import settings



async def build_agent():
    if not settings.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set")

    llm = ChatGoogleGenerativeAI(
        model = "gemini-3.5-flash-lite",
        api_key = settings.GEMINI_API_KEY,
        temperature = 0.1,
        max_tokens=1000,
        timeout=30,
    )
        
    tools = await get_strapi_tools()

    agent = create_agent(
        name="hr-agent",
        model = llm,
        system_prompt = HR_SYSTEM_PROMPT,
        tools = tools,
    )

    return agent