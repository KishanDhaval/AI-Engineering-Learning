# graph/state.py
# Shared state that flows through every node in the graph.

from typing import Annotated, Any
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    # ── Conversation history ──────────────────────────────────────────
    # add_messages reducer: appends new messages rather than replacing.
    # LangGraph automatically merges updates using this reducer.
    messages: Annotated[list[BaseMessage], add_messages]

    # ── User context ──────────────────────────────────────────────────
    user_id: str

    # ── HITL: tool calls approved by user and ready to execute ────────
    # Set by approval_node / create_job_node when user says yes.
    # Cleared to None after tools node runs or on rejection.
    pending_tool_calls: list[dict] | None
