
from utils.security import validate_question
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command

from graph.workflow import build_graph
from integration.postgres_memory import open_memory


async def stream_answer(question: str, thread_id: str, user_id: str):    
    """Stream tokens, tool activity, and HITL interrupt events."""
    question = validate_question(question)

    async with open_memory() as (checkpointer, store):
        graph = await build_graph(
            checkpointer=checkpointer,
            store=store,
            user_id=user_id,
        )

        config = {
            "configurable": {"thread_id": thread_id},
            "run_name": f"Innvonix HR Turn - {thread_id[:8]}",
            "tags": ["innvonix-careers-hub", "recruitment", "chat"],
            "metadata": {
                "user_id": user_id,
                "thread_id": thread_id,
                "application": "Innvonix Careers Hub",
            },
        }

        async for mode, chunk in graph.astream(
            {
                "messages": [{"role": "user", "content": question}],
                "user_id": user_id,
                "pending_tool_calls": None,
            },
            config=config,
            stream_mode=["messages", "updates"],
        ):
            if mode == "messages":
                message_chunk, metadata = chunk

                # Only yield tokens produced by the model node
                if metadata.get("langgraph_node") != "model":
                    continue

                # Thinking / reasoning tokens (Groq thinking models)
                reasoning = (
                    message_chunk.additional_kwargs.get("reasoning_content")
                    or message_chunk.additional_kwargs.get("reasoning")
                )
                if reasoning:
                    yield {"type": "reasoning", "content": reasoning}

                # Final answer tokens
                content = message_chunk.content
                if isinstance(content, str) and content:
                    yield {"type": "token", "content": content}

            elif mode == "updates":
                for node_name, update in chunk.items():
                    if not isinstance(update, dict):
                        continue
                    for message in update.get("messages", []):
                        if isinstance(message, AIMessage):
                            for call in message.tool_calls:
                                yield {
                                    "type": "tool_call",
                                    "name": call["name"],
                                }
                        elif isinstance(message, ToolMessage):
                            yield {
                                "type": "tool_result",
                                "name": message.name or node_name,
                                "content": message.content,
                            }

        # ── After stream ends: check if graph is paused at an interrupt ─
        # snapshot.next is non-empty when graph is suspended via interrupt()
        snapshot = await graph.aget_state(config)
        if snapshot.next:
            for task in snapshot.tasks:
                for intr in task.interrupts:
                    yield {"type": "interrupt", "payload": intr.value}


async def resume_graph(thread_id: str, user_id: str, user_response: dict):
    """Resume a suspended graph after human-in-the-loop decision.

    Sends Command(resume=user_response) into the graph and streams the
    resulting tokens / tool activity / further interrupts.

    Yields the same event types as stream_answer().
    """
    async with open_memory() as (checkpointer, store):
        graph = await build_graph(
            checkpointer=checkpointer,
            store=store,
            user_id=user_id,
        )

        decision = str(user_response.get("action") or ("approved" if user_response.get("approved") else "rejected"))
        config = {
            "configurable": {"thread_id": thread_id},
            "run_name": f"HITL Decision ({decision}) - {thread_id[:8]}",
            "tags": ["innvonix-careers-hub", "hitl", "human-decision"],
            "metadata": {
                "user_id": user_id,
                "thread_id": thread_id,
                "decision": decision,
                "application": "Innvonix Careers Hub",
            },
        }

        async for mode, chunk in graph.astream(
            Command(resume=user_response),
            config=config,
            stream_mode=["messages", "updates"],
        ):
            if mode == "messages":
                message_chunk, metadata = chunk

                if metadata.get("langgraph_node") != "model":
                    continue

                reasoning = (
                    message_chunk.additional_kwargs.get("reasoning_content")
                    or message_chunk.additional_kwargs.get("reasoning")
                )
                if reasoning:
                    yield {"type": "reasoning", "content": reasoning}

                content = message_chunk.content
                if isinstance(content, str) and content:
                    yield {"type": "token", "content": content}

            elif mode == "updates":
                for node_name, update in chunk.items():
                    if not isinstance(update, dict):
                        continue
                    for message in update.get("messages", []):
                        if isinstance(message, AIMessage):
                            for call in message.tool_calls:
                                yield {
                                    "type": "tool_call",
                                    "name": call["name"],
                                }
                        elif isinstance(message, ToolMessage):
                            yield {
                                "type": "tool_result",
                                "name": message.name or node_name,
                                "content": message.content,
                            }

        # Check if still interrupted (e.g. after an edit loop)
        snapshot = await graph.aget_state(config)
        if snapshot.next:
            for task in snapshot.tasks:
                for intr in task.interrupts:
                    yield {"type": "interrupt", "payload": intr.value}


async def get_saved_threads(limit: int = 20) -> list[dict]:
    """Retrieve list of past conversation threads from Postgres checkpointer."""
    async with open_memory() as (checkpointer, _):
        threads = {}
        async for state in checkpointer.alist(None):
            tid = state.config.get("configurable", {}).get("thread_id")
            if not tid or tid in threads:
                continue
            messages = state.checkpoint.get("channel_values", {}).get("messages", [])
            title = "New conversation"
            for m in messages:
                content = getattr(m, "content", "")
                m_type = getattr(m, "type", "")
                if m_type in ("human", "user") and content:
                    title = content[:32] + ("..." if len(content) > 32 else "")
                    break
            threads[tid] = {"thread_id": tid, "title": title}
            if len(threads) >= limit:
                break
        return list(threads.values())


async def get_thread_history(thread_id: str) -> list[dict]:
    """Load chat history for a specific thread from Postgres."""
    async with open_memory() as (checkpointer, _):
        state = await checkpointer.aget({"configurable": {"thread_id": thread_id}})
        if not state:
            return []
        messages = state.get("channel_values", {}).get("messages", [])
        history = []
        for m in messages:
            m_type = getattr(m, "type", "")
            content = getattr(m, "content", "")
            if isinstance(content, str) and content.strip():
                if m_type in ("human", "user"):
                    history.append({"role": "user", "content": content})
                elif m_type == "ai":
                    history.append({"role": "assistant", "content": content})
        return history


async def load_thread_state(thread_id: str, user_id: str) -> dict:
    """Load message history AND any paused interrupt for a thread from Postgres."""
    async with open_memory() as (checkpointer, store):
        graph = await build_graph(
            checkpointer=checkpointer,
            store=store,
            user_id=user_id,
        )
        config = {"configurable": {"thread_id": thread_id}}
        snapshot = await graph.aget_state(config)

        # 1. Recover conversation history
        messages = snapshot.values.get("messages", [])
        history = []
        for m in messages:
            m_type = getattr(m, "type", "")
            content = getattr(m, "content", "")
            if isinstance(content, str) and content.strip():
                if m_type in ("human", "user"):
                    history.append({"role": "user", "content": content})
                elif m_type == "ai":
                    history.append({"role": "assistant", "content": content})

        # 2. Recover paused interrupt if waiting for approval
        pending_interrupt = None
        if snapshot.next:
            for task in snapshot.tasks:
                for intr in task.interrupts:
                    pending_interrupt = intr.value
                    break

        return {
            "chat_history": history,
            "pending_interrupt": pending_interrupt,
        }


