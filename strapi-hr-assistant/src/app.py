import uuid

import streamlit as st

from agent.factory import build_agent
from services.chat_service import ask_agent
from utils.async_runner import run_async
from utils.security import validate_question


st.set_page_config(
    page_title="HR Assistant",
    page_icon="💼",
    layout="wide",
)

st.title("💼 Strapi HR Assistant")
st.caption(
    "Ask questions about jobs and applications in Strapi."
)


# Initialize the agent once for this Streamlit session.
if "agent" not in st.session_state:
    with st.spinner("Connecting to Strapi MCP..."):
        st.session_state.agent = run_async(build_agent())

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# Display previous messages.
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Get the next question.
question = st.chat_input("Ask HR Assistant...")

if question:
    try:
        question = validate_question(question)

        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing Strapi data..."):
                answer = run_async(
                    ask_agent(
                        agent=st.session_state.agent,
                        question=question,
                        thread_id=st.session_state.thread_id,
                    )
                )

            st.markdown(answer)

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )

    except Exception as error:
        st.error(
            f"Unable to process the request: {error}"
        )