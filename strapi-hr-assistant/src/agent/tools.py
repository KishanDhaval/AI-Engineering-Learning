

from integration.strapi_mcp import get_mcp_tools


async def get_strapi_tools():
    tools = await get_mcp_tools()
    return tools