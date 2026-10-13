import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent

load_dotenv()

# Select LLM based on available environment key
if os.getenv("GROQ_API_KEY"):
    from langchain_groq import ChatGroq
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
elif os.getenv("GEMINI_API_KEY"):
    from langchain_google_genai import ChatGoogleGenerativeAI
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0)
else:
    raise ValueError("Neither GROQ_API_KEY nor GEMINI_API_KEY found in environment.")

SERVER_PATH = str(Path(__file__).resolve().parent.parent / "server" / "server.py")


async def main():
    # 1. Connect to FastMCP server via stdio
    client = MultiServerMCPClient({
        "enterprise_catalog": {
            "transport": "stdio",
            "command": sys.executable,
            "args": [SERVER_PATH],
        }
    })

    # 2. Dynamically fetch tools from MCP server
    tools = await client.get_tools()
    print(f"Discovered {len(tools)} tools from MCP server:")
    for tool in tools:
        print(f" - {tool.name}: {tool.description}")

    # 3. Adapt tools with LangChain agent
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=(
            "You are an enterprise assistant with access to the Product Catalog and "
            "Employee Directory via MCP tools. Always query the tools to answer questions."
        )
    )

    # 4. Call agent with queries
    queries = [
        "Find all electronics under $200 that are currently in stock.",
        "Check the inventory for the ErgoMotion Adjustable Standing Desk. Can I order one today?",
        "Who is leading AI infrastructure in Engineering and what skills do they have?",
    ]

    for query in queries:
        print(f"\n{'='*70}\nUser: {query}\n{'='*70}")
        result = await agent.ainvoke({"messages": query})
        response = result["messages"][-1].content
        if isinstance(response, list):
            response = "".join(
                b.get("text", "") if isinstance(b, dict) else str(b)
                for b in response
            )
        print(f"Agent:\n{response}")


if __name__ == "__main__":
    asyncio.run(main())
