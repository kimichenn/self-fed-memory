from __future__ import annotations

import os

import requests
import streamlit as st

API_BASE = os.environ.get("SELF_MEMORY_API", "http://localhost:8000")


st.set_page_config(page_title="Self-Fed Memory", page_icon="🧠", layout="centered")

st.title("🧠 Self-Fed Memory")
st.caption("Personal AI with long-term memory")

with st.sidebar:
    st.header("Settings")
    name = st.text_input("Your name", value="User")
    intelligent = st.toggle(
        "Intelligent mode", value=True, help="Use preferences and multi-query retrieval"
    )
    k = st.slider("Memories to retrieve (k)", min_value=1, max_value=15, value=5)
    st.divider()
    st.subheader("Backend")
    api_base = st.text_input("API base URL", value=API_BASE)
    if api_base != API_BASE:
        API_BASE = api_base
    use_test_index = st.toggle(
        "Use TEST Pinecone index",
        value=False,
        help="Toggle between production and test Pinecone indices",
    )


if "history" not in st.session_state:
    st.session_state.history = []  # list of {role, content}


def call_chat_api(question: str, conversation_history: str | None) -> str:
    url = f"{API_BASE}/chat"
    payload = {
        "question": question,
        "name": name,
        "k": k,
        "intelligent": intelligent,
        "conversation_history": conversation_history or "",
        "use_test_index": use_test_index,
    }
    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data.get("answer") or data.get("result") or "(no answer)"


for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


prompt = st.chat_input("Ask anything…")
if prompt:
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Build conversation text for preference extraction
    conversation_text = "\n".join(
        f"{m['role'].capitalize()}: {m['content']}"
        for m in st.session_state.history[-6:]
    )

    with st.chat_message("assistant"), st.spinner("Thinking..."):
        try:
            answer = call_chat_api(prompt, conversation_text)
        except Exception as e:
            answer = f"Error: {e}"
        st.markdown(answer)
    st.session_state.history.append({"role": "assistant", "content": answer})
