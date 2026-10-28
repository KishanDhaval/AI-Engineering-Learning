import asyncio
import uuid
import streamlit as st

from config.settings import settings
from services.chat_service import (
    stream_answer,
    resume_graph,
    get_saved_threads,
    load_thread_state,
)

# Friendly display names and descriptions for internal tool actions
ACTION_LABELS = {
    "update_job-application": "Update Candidate Application Stage",
    "update_job-opening": "Update Job Vacancy Details",
    "create_job-opening": "Publish New Job Vacancy",
    "publish_job-opening": "Publish Job Vacancy (Make Live)",
    "unpublish_job-opening": "Unpublish Job Vacancy (Take Offline)",
    "discard_job-opening_draft": "Discard Job Draft",
    "get_user_preferences": "Check Saved Preferences",
    "remember_hr_preferences": "Save User Preference",
}

ACTION_DESCRIPTIONS = {
    "publish_job-opening": "Publish this vacancy to the official Innvonix Careers portal so candidates can view and apply.",
    "unpublish_job-opening": "Take this job posting offline from the careers portal without deleting it.",
    "discard_job-opening_draft": "Permanently discard this unpublished job draft from the system.",
    "update_job-opening": "Apply the requested adjustments to this job vacancy in the recruitment system.",
    "update_job-application": "Update the candidate's stage in the hiring pipeline.",
}


st.set_page_config(
    page_title="Innvonix Careers Hub",
    page_icon="💼",
    layout="wide",
)

st.title("Innvonix Careers Hub")
st.caption("AI Talent Acquisition & Recruitment Copilot for Innvonix")

# ── Custom Styling: Right-align User Chat Messages ───────────────────────
st.markdown(
    """
    <style>
    /* Reverse flex direction so user avatar and message align to the right */
    .stChatMessage:has([data-testid="stChatMessageAvatarUser"]) {
        flex-direction: row-reverse;
        text-align: right;
    }

    /* Align text content to the right */
    .stChatMessage:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] {
        text-align: right;
    }

    .stChatMessage:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] p {
        text-align: right;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Demo user ID (can be mapped to real authentication)
user_id = st.session_state.get("authenticated_user_id", "Test-User-1")

# ── Session state & URL thread synchronization ──────────────────────────
url_thread_id = st.query_params.get("thread")

if "thread_id" not in st.session_state or (url_thread_id and url_thread_id != st.session_state.get("thread_id")):
    if url_thread_id:
        st.session_state.thread_id = url_thread_id
        # Recover full thread state (history + any paused interrupt) from Postgres
        state = asyncio.run(load_thread_state(url_thread_id, user_id))
        st.session_state.chat_history = state["chat_history"]
        st.session_state.pending_interrupt = state["pending_interrupt"]
    else:
        new_id = str(uuid.uuid4())
        st.session_state.thread_id = new_id
        st.session_state.chat_history = []
        st.session_state.pending_interrupt = None
        st.query_params["thread"] = new_id


# ── Sidebar: Thread Management & Past Conversations ─────────────────────
with st.sidebar:
    st.subheader("Innvonix Careers Hub")
    st.write("Manage talent pipelines, job openings, and candidate applications.")

    if st.button("➕ New Conversation", use_container_width=True, type="primary"):
        new_id = str(uuid.uuid4())
        st.session_state.thread_id = new_id
        st.session_state.chat_history = []
        st.session_state.pending_interrupt = None
        st.query_params["thread"] = new_id
        st.rerun()

    st.divider()
    st.subheader("📜 Chat History")

    # Fetch past threads from Postgres checkpointer
    try:
        saved_threads = asyncio.run(get_saved_threads(limit=15))
    except Exception:
        saved_threads = []

    if saved_threads:
        for t in saved_threads:
            tid = t["thread_id"]
            title = t["title"]
            is_active = tid == st.session_state.thread_id
            btn_label = f"{title}" if not is_active else f"👉 {title}"

            if st.button(
                btn_label,
                key=f"thread_btn_{tid}",
                use_container_width=True,
                disabled=is_active,
            ):
                st.session_state.thread_id = tid
                st.query_params["thread"] = tid
                # Recover full thread state + paused review card from Postgres
                state = asyncio.run(load_thread_state(tid, user_id))
                st.session_state.chat_history = state["chat_history"]
                st.session_state.pending_interrupt = state["pending_interrupt"]
                st.rerun()
    else:
        st.caption("No past conversations yet.")


# ── Chat history display ─────────────────────────────────────────────────
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ── Helper: Resume suspended graph ──────────────────────────────────────
async def _do_resume(user_response: dict):
    """Send human decision back to the graph and stream remaining tokens."""
    answer_parts = []

    with st.chat_message("assistant"):
        answer_box = st.empty()

        async for event in resume_graph(
            thread_id=st.session_state.thread_id,
            user_id=user_id,
            user_response=user_response,
        ):
            if event["type"] == "token":
                answer_parts.append(event["content"])
                answer_box.markdown("".join(answer_parts))

            elif event["type"] == "tool_call":
                friendly_name = ACTION_LABELS.get(event["name"], event["name"])
                st.write(f"🔍 Accessing: `{friendly_name}`")

            elif event["type"] == "tool_result":
                friendly_name = ACTION_LABELS.get(event["name"], event["name"])
                st.write(f"✅ Completed: `{friendly_name}`")

            elif event["type"] == "interrupt":
                st.session_state.pending_interrupt = event["payload"]
                st.query_params["thread"] = st.session_state.thread_id
                st.rerun()

    # Interrupt resolved
    st.session_state.pending_interrupt = None
    st.query_params["thread"] = st.session_state.thread_id

    if answer_parts:
        st.session_state.chat_history.append(
            {"role": "assistant", "content": "".join(answer_parts)}
        )
    st.rerun()


# ── HITL Interrupt UI (Key:Value layout, no raw JSON) ───────────────────
if st.session_state.pending_interrupt:
    payload = st.session_state.pending_interrupt

    # ── Job Draft Review UI ──────────────────────────────────────────────
    if payload.get("type") == "job_draft_review":
        st.warning("📋 **Job Vacancy Posting — Team Review Required**")
        draft = payload.get("draft", {})
        responsibilities = payload.get("responsibilities", [])

        # Display details in simple, clean Key:Value format
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Job Title:** {draft.get('title')}")
            st.markdown(f"**Department:** {draft.get('department')}")
            st.markdown(f"**Location:** {draft.get('location')}")
        with col2:
            st.markdown(f"**Employment Type:** {draft.get('employmentType')}")
            st.markdown(f"**Experience Required:** {draft.get('experience')}")
            st.markdown(f"**Openings:** {draft.get('openings')}")

        if responsibilities:
            st.markdown("---")
            st.markdown("🎯 **Proposed Responsibilities:**")
            for i, r in enumerate(responsibilities, 1):
                st.markdown(f"{i}. {r}")

        st.markdown("---")

        # Natural language edit input (no JSON required)
        with st.expander("✏️ Request Changes in Plain English", expanded=False):
            feedback = st.text_area(
                "Describe what you'd like adjusted:",
                placeholder="e.g. Change openings to 3, make location Remote, and add GraphQL expertise to responsibilities",
                key="nl_feedback",
            )
            if st.button("🔄 Update Draft", key="btn_apply_nl_edit"):
                if feedback.strip():
                    asyncio.run(_do_resume({"action": "edit", "feedback": feedback.strip()}))

        # Primary action buttons
        btn_col1, btn_col2, btn_col3 = st.columns([1.5, 2, 1.2])

        with btn_col1:
            if st.button("✅ Approve & Publish Posting", key="approve_job", type="primary"):
                asyncio.run(_do_resume({"action": "approve"}))

        with btn_col2:
            if st.button("⚡ Publish Without Responsibilities", key="approve_no_resp"):
                asyncio.run(_do_resume({"action": "approve_without_responsibilities"}))

        with btn_col3:
            reject_reason = st.text_input(
                "Cancel reason",
                placeholder="e.g. On hold",
                key="job_reject_reason",
                label_visibility="collapsed",
            )
            if st.button("❌ Cancel Posting", key="reject_job"):
                asyncio.run(_do_resume({"action": "reject", "reason": reject_reason or "Cancelled by you"}))

    # ── Generic Action Approval UI (Key:Value layout) ────────────────────
    elif payload.get("type") == "action_approval":
        st.warning("⚠️ **Hiring Pipeline Action — Confirmation Needed**")

        for action in payload.get("actions", []):
            tool_name = action.get("tool", "")
            friendly_name = ACTION_LABELS.get(tool_name, tool_name.replace("_", " ").title())
            description = ACTION_DESCRIPTIONS.get(tool_name, "")
            args = action.get("args", {})
            doc_id = args.get("documentId", "N/A")
            data = args.get("data", {})
            target_info = action.get("target_info", {})

            with st.container(border=True):
                st.markdown(f"#### ⚙️ {friendly_name}")
                if description:
                    st.caption(description)

                # Show friendly target entity if resolved
                target_title = target_info.get("title") or target_info.get("fullName")
                if target_title:
                    dept = target_info.get("department") or target_info.get("position")
                    loc = target_info.get("location")
                    details_str = f"**Target Role / Candidate:** `{target_title}`"
                    if dept:
                        details_str += f" ({dept})"
                    if loc:
                        details_str += f" • {loc}"
                    st.markdown(details_str)
                else:
                    st.markdown(f"**Reference ID:** `{doc_id}`")

                if data:
                    st.markdown("**Proposed Changes:**")
                    for k, v in data.items():
                        st.markdown(f"- **{k}:** `{v}`")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Confirm & Apply", key="approve_action", type="primary"):
                asyncio.run(_do_resume({"approved": True}))

        with col2:
            reject_reason = st.text_input(
                "Rejection reason",
                placeholder="e.g. Status incorrect",
                key="action_reject_reason",
                label_visibility="collapsed",
            )
            if st.button("❌ Cancel Action", key="reject_action"):
                asyncio.run(_do_resume({"approved": False, "reason": reject_reason or "Cancelled by you"}))

    st.divider()


# ── Main Chat Input ──────────────────────────────────────────────────────
question = st.chat_input(
    "Ask about vacancies, candidates, or request a new job posting...",
    disabled=st.session_state.pending_interrupt is not None,
)

if question:
    st.session_state.chat_history.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        answer_box = st.empty()
        answer_parts = []

        with st.expander("💭 Recruitment Reasoning & Activity", expanded=True):
            reasoning_box = st.empty()
            reasoning_parts = []
            activity_box = st.container()

        async def run_stream():
            with activity_box:
                async for event in stream_answer(
                    question=question,
                    thread_id=st.session_state.thread_id,
                    user_id=user_id,
                ):
                    if event["type"] == "reasoning":
                        reasoning_parts.append(event["content"])
                        reasoning_box.markdown(
                            "**Thinking:**\n\n> "
                            + "".join(reasoning_parts).replace("\n", "\n> ")
                        )

                    elif event["type"] == "token":
                        answer_parts.append(event["content"])
                        answer_box.markdown("".join(answer_parts))

                    elif event["type"] == "tool_call":
                        friendly_name = ACTION_LABELS.get(event["name"], event["name"])
                        st.write(f"🔍 Accessing: `{friendly_name}`")

                    elif event["type"] == "tool_result":
                        friendly_name = ACTION_LABELS.get(event["name"], event["name"])
                        st.write(f"✅ Completed: `{friendly_name}`")

                    elif event["type"] == "interrupt":
                        st.session_state.pending_interrupt = event["payload"]

        try:
            asyncio.run(run_stream())
        except Exception as e:
            err_msg = str(e)
            if hasattr(e, "exceptions") and e.exceptions:
                err_msg = " | ".join(
                    f"{type(sub).__name__}: {sub}" for sub in e.exceptions
                )
            if "ConnectError" in err_msg or "connection attempts failed" in err_msg:
                st.error(
                    "⚠️ Connection issue: Could not connect to the Careers backend service. "
                    "Please verify the backend server is running."
                )
            else:
                st.error(f"⚠️ An error occurred: {err_msg}")

    answer_text = "".join(answer_parts)
    if answer_text:
        st.session_state.chat_history.append(
            {"role": "assistant", "content": answer_text}
        )

    if st.session_state.pending_interrupt:
        st.rerun()