
from utils.security import validate_question
from langchain_core.messages import AIMessage, ToolMessage

from agent.factory import build_agent
from integration.postgres_memory import open_memory
from services.preference_services import get_preferences


async def stream_answer(question, thread_id, user_id):
    question = validate_question(question)

    async with open_memory() as (checkpointer, store):

        agent = await build_agent(
            checkpointer=checkpointer,
            store=store,
            user_id=user_id,
        )

        async for mode, chunk in agent.astream(
            {"messages": [{"role": "user", "content": question}]},
            config={"configurable": {"thread_id": thread_id}},
            stream_mode=["messages", "updates"],
        ):
            if mode == "messages":
                message_chunk, metadata = chunk

                if metadata.get("langgraph_node") != "model":
                    continue

                # Capture thinking/reasoning tokens
                reasoning = (
                    message_chunk.additional_kwargs.get("reasoning_content")
                    or message_chunk.additional_kwargs.get("reasoning")
                )
                if reasoning:
                    yield {"type": "reasoning", "content": reasoning}

                # Capture final answer tokens
                content = message_chunk.content
                if isinstance(content, str) and content:
                    yield {"type": "token", "content": content}

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
