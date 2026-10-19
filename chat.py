"""chat.py - Interactive command-line chat over the CISA/NIST security guides.

Run:  python chat.py
Type a question and press Enter.
Commands:  /debug  toggle retrieval details    /quit  exit (or press Ctrl+C)

All RAG and security logic lives in rag.py; this file is only the interface.
"""
import os
from pathlib import Path

from langchain_openai import ChatOpenAI

from ingest import CHROMA_DIR
from rag import CHAT_MODEL, MIN_RELEVANCE, answer_question, get_vector_store

BANNER = """
==============================================================
 Secure RAG Chatbot - CISA & NIST security guides
 Answers come only from the indexed PDFs, with page citations.
 Commands: /debug (toggle details)   /quit (exit)
=============================================================="""


def print_result(result: dict, debug: bool) -> None:
    """Show the answer, its sources, and (in debug mode) how it was produced."""
    print(f"\nAssistant: {result['answer']}")

    if result["sources"]:
        print("\nSources:")
        for n, label in result["sources"]:
            print(f"  [{n}] {label}")

    if debug:
        print("\n--- debug ---")
        if result["blocked"]:
            print("  Blocked by a security layer")
        if result["best_score"] is not None:
            print(f"  Best relevance score: {result['best_score']} (threshold {MIN_RELEVANCE})")
        print(f"  Chat model called: {result['model_called']}")
        if result["retrieved"]:
            print("  Retrieved chunks:")
            for n, label in result["retrieved"]:
                print(f"    [{n}] {label}")


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set. Copy .env.example to .env and add your key.")
    if not Path(CHROMA_DIR).exists():
        raise SystemExit("No vector database found. Run this first:  python ingest.py")

    store = get_vector_store()
    llm = ChatOpenAI(model=CHAT_MODEL)
    debug = False
    print(BANNER)

    while True:
        try:
            question = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):  # Ctrl+C or end of input
            print("\nGoodbye!")
            break

        if not question:
            continue
        command = question.lower()
        if command in ("/quit", "/exit", "quit", "exit"):
            print("Goodbye!")
            break
        if command == "/debug":
            debug = not debug
            print(f"Debug mode {'ON' if debug else 'OFF'}")
            continue

        try:
            result = answer_question(question, store, llm)
        except Exception as error:
            # Show only the error type, not its details, which could expose internals
            print(f"\nAssistant: Sorry, something went wrong ({type(error).__name__}). Please try again.")
            continue

        print_result(result, debug)


if __name__ == "__main__":
    main()