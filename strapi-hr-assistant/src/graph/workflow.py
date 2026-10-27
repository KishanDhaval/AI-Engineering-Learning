# graph/workflow.py
# Wires all nodes and edges into a compiled LangGraph StateGraph.
# build_graph() is the single entry point used by chat_service.py.

import functools

from langchain.tools import tool
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from agent.prompts import HR_SYSTEM_PROMPT
from agent.tools import get_strapi_tools
from config.settings import settings
from graph.nodes import model_node, create_job_node, approval_node
from graph.routing import router, after_hitl_router
from graph.state import AgentState
from services.preference_services import get_preferences, compact_and_save_preferences


def _get_base_model():
    model_name = settings.GROQ_MODEL if hasattr(settings, "GROQ_MODEL") and settings.GROQ_MODEL else "openai/gpt-oss-120b"
    if settings.GROQ_API_KEY:
        return ChatGroq(
            model=model_name,
            groq_api_key=settings.GROQ_API_KEY,
            temperature=0,
        )
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=settings.GEMINI_API_KEY,
        temperature=0,
    )


async def build_graph(checkpointer, store, user_id):
    """Build and compile StateGraph."""

    # Load tools from Strapi MCP (cached after first request)
    mcp_tools = await get_strapi_tools()
    base_model = _get_base_model()

    # In-process tools to remember and retrieve user preferences
    @tool
    async def remember_hr_preferences(preference: str) -> str:
        """Save or update user preferences (e.g. 'Always default to Ahmedabad', 'Use concise bullet points', 'Focus on senior roles')."""
        return await compact_and_save_preferences(store, user_id, base_model, preference)

    @tool
    async def get_user_preferences() -> str:
        """Retrieve the user's currently saved HR preferences and hiring guidelines."""
        prefs = await get_preferences(store, user_id)
        return prefs if prefs else "No user preferences have been saved yet."

    all_tools = [*mcp_tools, remember_hr_preferences, get_user_preferences]
    model = base_model.bind_tools(all_tools)

    # Build system prompt with current user preferences
    preferences = await get_preferences(store, user_id)
    preference_section = (
        f"\n\nCurrent User Preferences:\n{preferences}"
        if preferences
        else "\n\nCurrent User Preferences: None set."
    )
    system_prompt = (
        HR_SYSTEM_PROMPT
        + preference_section
        + (
            "\n\nIMPORTANT RULES FOR TOOL CALLS:"
            "\n- Call only ONE tool per response turn."
            "\n- Never mix read tools and write tools in the same response."
            "\n- If you need multiple pieces of information, call tools one at a time."
        )
    )

    # Assemble the StateGraph
    builder = StateGraph(AgentState)

    # Nodes
    builder.add_node(
        "model",
        functools.partial(model_node, model=model, system_prompt=system_prompt),
    )
    builder.add_node("tools", ToolNode(all_tools))
    builder.add_node(
        "create_job_node",
        functools.partial(create_job_node, model=base_model),
    )
    builder.add_node("approval_node", approval_node)

    # Edges
    builder.add_edge(START, "model")

    # After model speaks -> router classifies the tool calls
    builder.add_conditional_edges(
        "model",
        router,
        {
            "__end__": END,
            "tools": "tools",
            "create_job_node": "create_job_node",
            "approval_node": "approval_node",
        },
    )

    # Read-only tools complete -> return to model to present answer
    builder.add_edge("tools", "model")

    # After HITL nodes:
    # If approved (pending_tool_calls set) -> run tools
    # If rejected (pending_tool_calls is None) -> return to model to speak to user
    builder.add_conditional_edges(
        "create_job_node",
        after_hitl_router,
        {"tools": "tools", "model": "model"},
    )
    builder.add_conditional_edges(
        "approval_node",
        after_hitl_router,
        {"tools": "tools", "model": "model"},
    )

    # Compile with checkpointer and store 
    return builder.compile(
        checkpointer=checkpointer,
        store=store,
    )
