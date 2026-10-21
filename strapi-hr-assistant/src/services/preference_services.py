
ALLOWED_PREFERENCES = {
    "response_style",
    "candidate_summary_format",
}


async def get_preferences(store, user_id):
    namespace = (user_id, "preferences")

    item = await store.aget(
        namespace,
        "hr_settings",
    )

    return item.value if item else {}


async def save_preferences(store, user_id, changes):
    invalid_keys = set(changes) - ALLOWED_PREFERENCES

    if invalid_keys:
        raise ValueError(
            f"Unsupported preferences: {invalid_keys}"
        )

    current = await get_preferences(store, user_id)
    updated = {**current, **changes}

    await store.aput(
        (user_id, "preferences"),
        "hr_settings",
        updated,
    )

    return updated
