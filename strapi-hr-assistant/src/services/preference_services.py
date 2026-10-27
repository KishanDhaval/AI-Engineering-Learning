# services/preference_services.py
# Simple string-based user memory profile with automatic LLM compaction.

from langchain_core.messages import HumanMessage


async def get_preferences(store, user_id: str) -> str:
    """Retrieve the user's stored preferences profile as a clean string."""
    namespace = (user_id, "preferences")
    item = await store.aget(namespace, "profile")
    if not item or not isinstance(item.value, dict):
        return ""
    return item.value.get("text", "")


async def save_preferences(store, user_id: str, text: str) -> str:
    """Save preferences string into Postgres long-term store."""
    namespace = (user_id, "preferences")
    clean_text = text.strip()
    await store.aput(namespace, "profile", {"text": clean_text})
    return clean_text


async def compact_and_save_preferences(
    store, user_id: str, model, new_instruction: str
) -> str:
    """Merge a new user instruction into existing profile and compact redundant rules."""
    current = await get_preferences(store, user_id)

    if not current:
        compacted = f"- {new_instruction.strip().lstrip('-* ')}"
    else:
        prompt = (
            "You are an AI memory manager for an HR Assistant.\n"
            "Merge the new user preference with the existing preferences.\n"
            "Rules:\n"
            "1. Resolve any contradictions (newer preference takes priority).\n"
            "2. Remove duplicate or outdated rules.\n"
            "3. Output ONLY 2 to 5 concise bullet points, starting each with '- '.\n\n"
            f"Existing Preferences:\n{current}\n\n"
            f"New Preference:\n{new_instruction}"
        )
        try:
            response = await model.ainvoke([HumanMessage(content=prompt)])
            compacted = response.content.strip()
        except Exception:
            compacted = f"{current}\n- {new_instruction.strip().lstrip('-* ')}"

    await save_preferences(store, user_id, compacted)
    return compacted
