
import asyncio
import uuid

import streamlit as st

from services.chat_service import stream_answer


st.set_page_config(
    page_title="Strapi HR Assistant",
    page_icon="💼",
    layout="wide",
)

st.title("💼 Strapi HR Assistant")
st.caption("Recruitment assistant powered by Strapi MCP and Gemini")

# Demo only. Replace with the ID from your authentication system.
user_id = st.session_state.get("authenticated_user_id", "Test-User-1")

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with st.sidebar:
    st.subheader("HR Assistant")
    st.write("Ask about jobs, applications, and recruitment data.")

    if st.button("New conversation"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.chat_history = []
        st.rerun()

# Display previous messages.
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask about jobs or applications...")

if question:
    st.session_state.chat_history.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        answer_box = st.empty()
        answer_parts = []

        with st.expander("Agent activity", expanded=True):
            activity_box = st.container()

        async def run_stream():
            with activity_box:
                async for event in stream_answer(
                    question=question,
                    thread_id=st.session_state.thread_id,
                    user_id=user_id,
                ):
                    if event["type"] == "token":
                        answer_parts.append(event["content"])
                        answer_box.markdown("".join(answer_parts))

                    elif event["type"] == "tool_call":
                        st.write(
                            f"🔧 Calling tool: `{event['name']}`"
                        )

                    elif event["type"] == "tool_result":
                        st.write(
                            f"✅ Tool completed: `{event['name']}`"
                        )

        asyncio.run(run_stream())

    answer_text = "".join(answer_parts)

    if answer_text:
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer_text,
        })