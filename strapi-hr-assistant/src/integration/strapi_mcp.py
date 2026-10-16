from langchain_mcp_adapters.client import MultiServerMCPClient
from config.settings import settings
import asyncio


def create_strapi_mcp_client():

    if not settings.STRAPI_MCP_TOKEN:
        raise ValueError("STRAPI_MCP_TOKEN not set")

    if not settings.STRAPI_API_URL:
        raise ValueError("STRAPI_API_URL not set")
    

    client = MultiServerMCPClient(
        {
            "strapi" : {
                "transport" : "http",
                "url" : settings.STRAPI_API_URL,
                "headers": {
                    "Authorization": f"Bearer {settings.STRAPI_MCP_TOKEN}"
                }
            }
        }
    )
    return client


async def get_mcp_tools():

    client = create_strapi_mcp_client()
    tools = await client.get_tools()

    return tools

if __name__ == "__main__":
    asyncio.run(get_mcp_tools())