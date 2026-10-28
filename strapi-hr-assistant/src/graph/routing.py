# graph/routing.py
# Pure Python — no LLM calls.
# classify_tool()  → categorises a tool by its name prefix.
# router()         → decides which node runs after the model speaks.

from langchain_core.messages import AIMessage


# ── Tool name prefix classification ──────────────────────────────────
# If Strapi adds new tools, prefix rules auto-classify them.
# Unknown tools always fall to "approval" (safest default).

READ_PREFIXES = ("get_", "list_", "log", "remember_hr_preferences", "get_user_preferences")
CREATE_PREFIXES = ("create_",)
APPROVAL_PREFIXES = ("update_", "discard_", "delete_", "publish_", "draft_")


def classify_tool(name: str) -> str:
    """Return 'read', 'create', or 'approval' based on tool name prefix.

    Priority: read → create → approval → approval (unknown default).
    """
    if any(name.startswith(p) or name == p for p in READ_PREFIXES):
        return "read"
    if any(name.startswith(p) for p in CREATE_PREFIXES):
        return "create"
    if any(name.startswith(p) for p in APPROVAL_PREFIXES):
        return "approval"
    # Unknown prefix → always require human approval (safest path)
    return "approval"


def router(state) -> str:
    """Conditional edge after model_node.

    Returns the name of the next node to execute:
    - "__end__" -> model answered directly, no tool calls
    - "tools" -> all tool calls are read-only, execute immediately
    - "create_job_node" -> at least one create_* tool was requested
    - "approval_node" -> at least one update/delete/unknown tool was requested
    """
    last = state["messages"][-1]

    # Model returned plain text with no tool calls
    if not isinstance(last, AIMessage) or not last.tool_calls:
        return "__end__"

    # Classify every tool call the model requested in this turn
    categories = {classify_tool(call["name"]) for call in last.tool_calls}

    # Priority: most restrictive category wins when mixed calls happen
    if "approval" in categories:
        return "approval_node"
    if "create" in categories:
        return "create_job_node"

    return "tools"  # all read-only 


def after_hitl_router(state) -> str:
    """Conditional edge after create_job_node / approval_node.

    If pending_tool_calls was set → user approved → run ToolNode.
    If pending_tool_calls is None → rejected / missing fields → model responds.
    """
    if state.get("pending_tool_calls"):
        return "tools"
    return "model"
