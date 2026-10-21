

from integration.strapi_mcp import get_mcp_tools


async def get_strapi_tools():
    tools = await get_mcp_tools()

    for tool in tools:
        if (tool.name != "create_job_opening" and
            tool.name != "update_job_opening" and
            tool.name != "update_job-application" 
            ):
            tool.args_schema = None

    print(tools)
    return tools