
import asyncio

async def ask_agent(agent, question : str, thread_id: str) -> str:
    
    result = await agent.ainvoke(
        {
            "messages": [
                {"role" : "user", "content" : question}
            ]
        },
        config = {
            "configurable": {
                "thread_id": thread_id
            }
        }
    )
    last_message = result["messages"][-1]
    content = last_message.content

    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        texts = []
        for part in content:
            if isinstance(part, dict) and "text" in part:
                texts.append(part["text"])
            elif isinstance(part, str):
                texts.append(part)
            elif hasattr(part, "text"):
                texts.append(getattr(part, "text"))
        if texts:
            return "\n".join(texts)
    return str(content)