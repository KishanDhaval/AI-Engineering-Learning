from langchain_mcp_adapters.client import MultiServerMCPClient
from config.settings import settings
import asyncio

client = None
cached_tools = None


def get_strapi_mcp_client():
    global client
    if client is None:
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


async def get_mcp_tools(refresh: bool = False):
    """Retrieve Strapi MCP tools, caching the schema in memory."""

    global cached_tools
    if cached_tools is None or refresh:
        client = get_strapi_mcp_client()
        cached_tools = await client.get_tools()
    return cached_tools

if __name__ == "__main__":
    asyncio.run(get_mcp_tools())