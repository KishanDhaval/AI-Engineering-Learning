
from langchain_core.messages import AIMessage, ToolMessage

from agent.factory import build_agent
from integration.postgres_memory import open_memory
from services.preference_services import get_preferences


async def stream_answer(question, thread_id, user_id):
    question = question.strip()

    if not question:
        raise ValueError("Question cannot be empty.")

    async with open_memory() as (checkpointer, store):
        preferences = await get_preferences(store, user_id)

        preference_instructions = (
            "\n\nPreferences for this authenticated HR user:\n"
            f"{preferences}\n"
            "Apply these preferences to the response when relevant."
        )

        agent = await build_agent(
            checkpointer=checkpointer,
            store=store,
            user_id=user_id,
        )

        # The preferences should be added to the agent's system
        # prompt in build_agent, rather than sent as user content.
        # See the small adjustment below.

        async for mode, chunk in agent.astream(
            {"messages": [{"role": "user", "content": question}]},
            config={"configurable": {"thread_id": thread_id}},
            stream_mode=["messages", "updates"],
        ):
            if mode == "messages":
                message_chunk, metadata = chunk

                if metadata.get("langgraph_node") != "model":
                    continue

                content = message_chunk.content

                if isinstance(content, str) and content:
                    yield {"type": "token", "content": content}

                elif isinstance(content, list):
                    for block in content:
                        if (
                            isinstance(block, dict)
                            and block.get("type") == "text"
                            and block.get("text")
                        ):
                            yield {
                                "type": "token",
                                "content": block["text"],
                            }

            elif mode == "updates":
                for node_name, update in chunk.items():
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
