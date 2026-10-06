from datetime import date

from dotenv import load_dotenv

# Load environment variables FIRST (same as main.py)
load_dotenv()

import streamlit as st

DAILY_LIMIT = 20
LIMIT_MESSAGE = (
    "🚫 Daily limit reached. You can ask up to 20 questions per day. "
    "Please come back tomorrow."
)

GREETING = {
    "role": "assistant",
    "content": "Hi! 👋 Ask me anything about on PCFitment.",
    "sources": [],
}


st.set_page_config(
    page_title="PCFitment Signing-Up Assistant",
    page_icon="🛠️",
    layout="centered",
)


@st.cache_resource(show_spinner="Checking vector database and loading the chatbot...")
def load_graph():
    """Runs once per server session (not on every message)."""
    from rag import sync_vector_database

    # Must run BEFORE importing graph, because graph.py opens the
    # vector store at import time.
    sync_vector_database()

    from graph import build_graph

    return build_graph()


def render_sources(sources):
    """Show the article links used for an answer."""
    if not sources:
        return
    st.markdown("---")
    st.markdown("**📚 Related articles:**")
    for src in sources:
        title = src.get("title", src["url"])
        st.markdown(f"- [{title}]({src['url']})")


# ---------------------------------------------------------
# Daily counter (resets on a new day)
# ---------------------------------------------------------

today = date.today().isoformat()

if st.session_state.get("count_date") != today:
    st.session_state.count_date = today
    st.session_state.question_count = 0


def counter_text():
    left = max(DAILY_LIMIT - st.session_state.question_count, 0)
    return f"Questions left today: {left}/{DAILY_LIMIT}"


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.title("🛠️ PCFitment Signing-Up Chatbot")
st.caption("Ask questions about the PCFitment Signing-Up guide.")

with st.sidebar:
    st.header("Options")
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = [GREETING]
        st.rerun()
    st.markdown("---")
    st.markdown(
        "This assistant answers **only** from the PCFitment "
        "Signing-Up documentation."
    )
    st.markdown("---")
    counter_slot = st.empty()          # updated after each question
    counter_slot.caption(counter_text())

# ---------------------------------------------------------
# Load graph (with friendly error handling)
# ---------------------------------------------------------

try:
    graph = load_graph()
except Exception as error:
    st.error(f"Failed to start the chatbot:\n\n{error}")
    st.stop()

# ---------------------------------------------------------
# Chat history
# ---------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = [GREETING]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        render_sources(message.get("sources", []))

# ---------------------------------------------------------
# Chat input
# ---------------------------------------------------------

question = st.chat_input("Type your question here...")

if question:
    question = question.strip()

    # Limit reached -> show message, don't call the graph
    if st.session_state.question_count >= DAILY_LIMIT:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            st.warning(LIMIT_MESSAGE)
        st.stop()

    st.session_state.question_count += 1
    counter_slot.caption(counter_text())

    # Build history BEFORE adding the new question
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
    ]

    st.session_state.messages.append(
        {"role": "user", "content": question, "sources": []}
    )
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        sources = []
        with st.spinner("Searching PCFitment documentation..."):
            try:
                result = graph.invoke(
                    {
                        "question": question,
                        "history": history,
                        "standalone": "",
                        "context": "",
                        "answer": "",
                        "relevant": False,
                        "sources": [],
                    }
                )
                answer = result["answer"]
                sources = result.get("sources", [])
            except Exception as error:
                answer = f"Error while processing question:\n\n{error}"

        st.markdown(answer)
        render_sources(sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )