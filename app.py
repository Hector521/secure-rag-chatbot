"""app.py - Streamlit web interface for the Secure RAG Chatbot.

Run:  streamlit run app.py
Reuses the exact same guarded pipeline as chat.py (all logic lives in rag.py),
so the web app gets every security layer automatically.
"""
import os
from pathlib import Path

import streamlit as st
from langchain_openai import ChatOpenAI

from ingest import CHROMA_DIR
from rag import CHAT_MODEL, MIN_RELEVANCE, answer_question, get_vector_store

st.set_page_config(page_title="Secure RAG Chatbot", page_icon="🔒")


@st.cache_resource
def load_pipeline():
    """Open the vector store and chat model once, instead of on every click."""
    return get_vector_store(), ChatOpenAI(model=CHAT_MODEL)


def render_answer(result: dict, debug: bool) -> None:
    """Show one assistant answer, its sources, and optional retrieval details."""
    if result["blocked"]:
        st.warning(result["answer"], icon="🛡️")
    else:
        # st.markdown escapes HTML by default, so model output can't inject scripts
        st.markdown(result["answer"])

    if result["sources"]:
        st.markdown("**Sources**")
        for n, label in result["sources"]:
            st.markdown(f"- [{n}] {label}")

    if debug:
        with st.expander("Retrieval details"):
            if result["best_score"] is not None:
                st.write(f"Best relevance score: {result['best_score']} (threshold {MIN_RELEVANCE})")
            st.write(f"Chat model called: {result['model_called']}")
            for n, label in result["retrieved"]:
                st.write(f"[{n}] {label}")


st.title("🔒 Secure RAG Chatbot")
st.caption("Answers come only from official CISA and NIST security guides, with page citations.")

# Startup checks: fail with a clear message instead of a stack trace
if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY is not set. Copy .env.example to .env and add your key.")
    st.stop()
if not Path(CHROMA_DIR).exists():
    st.error("No vector database found. Run `python ingest.py` first.")
    st.stop()

store, llm = load_pipeline()

with st.sidebar:
    st.header("Settings")
    debug = st.toggle("Show retrieval details", value=False)
    st.markdown(
        "**Indexed documents**\n"
        "- CISA #StopRansomware Guide\n"
        "- CISA Phishing Guidance\n"
        "- CISA Phishing-Resistant MFA fact sheet\n"
        "- NIST SP 800-63B-4"
    )
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# Chat history lives only in this browser session (nothing is saved to disk)
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["content"])
        else:
            render_answer(message["result"], debug)

if question := st.chat_input("Ask about ransomware, phishing, MFA, or passwords..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the guides..."):
            try:
                result = answer_question(question, store, llm)
            except Exception as error:
                # Show only the error type, not details that could expose internals
                result = {
                    "answer": f"Sorry, something went wrong ({type(error).__name__}). Please try again.",
                    "sources": [], "retrieved": [], "best_score": None,
                    "blocked": False, "model_called": False,
                }
        render_answer(result, debug)

    st.session_state.messages.append({"role": "assistant", "result": result})
    