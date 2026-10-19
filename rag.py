"""rag.py - Answer questions using ONLY the indexed PDFs, with citations.

Retrieve -> Augment -> Generate:
  1. Retrieve the chunks most relevant to the question from Chroma (MMR for diversity).
  2. Augment the prompt with those chunks, numbered [1], [2], ...
  3. Generate an answer with the chat model, citing chunk numbers.
Citations are mapped back to file + page from chunk metadata, so the model
never invents a source name or page number.
"""
import os
import re

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from ingest import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL

load_dotenv()  # reads OPENAI_API_KEY (and optional CHAT_MODEL) from .env

CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-6-luna")  # low-cost model; override in .env
TOP_K = 6       # how many chunks go into the prompt
FETCH_K = 20    # how many candidates MMR considers before picking diverse ones
NOT_FOUND = "I don't know based on these documents."

SYSTEM_PROMPT = f"""You are a cybersecurity assistant. You answer questions using ONLY the
numbered context excerpts provided, which come from official CISA and NIST security guides.

Rules:
1. Use only facts stated in the context excerpts. Do not use outside knowledge.
2. Cite the excerpt number(s) in square brackets after each fact, like [1] or [2][3].
3. If the context does not contain the answer, reply with exactly: {NOT_FOUND}
4. Keep answers concise: 2-5 sentences, or a short bulleted list."""


def get_vector_store() -> Chroma:
    """Open the Chroma database that ingest.py saved (no re-embedding needed)."""
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=OpenAIEmbeddings(model=EMBEDDING_MODEL),
        persist_directory=CHROMA_DIR,
    )


def source_label(doc: Document) -> str:
    """Human-readable citation for a chunk, built from its metadata."""
    return f"{doc.metadata['source']}, page {doc.metadata['page']}"


def format_context(docs: list[Document]) -> str:
    """Number each retrieved chunk and label it with its source, for the prompt."""
    return "\n\n".join(
        f"[{i}] (source: {source_label(doc)})\n{doc.page_content}"
        for i, doc in enumerate(docs, start=1)
    )


def answer_question(question: str, store: Chroma, llm: ChatOpenAI) -> dict:
    """Run the full RAG pipeline for one question.

    Returns {
        "answer": str,
        "sources": [(n, "file.pdf, page N"), ...]    # chunks the answer cited
        "retrieved": [(n, "file.pdf, page N"), ...]  # every chunk the model was given
    }
    """
    # 1. Retrieve: relevant AND diverse chunks, so near-duplicates don't crowd out the answer
    docs = store.max_marginal_relevance_search(question, k=TOP_K, fetch_k=FETCH_K)
    retrieved = [(i, source_label(doc)) for i, doc in enumerate(docs, start=1)]

    # 2. Augment
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Context:\n{format_context(docs)}\n\nQuestion: {question}"),
    ]

    # 3. Generate
    answer = llm.invoke(messages).text.strip()
    answer = answer.replace("\u2019", "'")  # normalize curly apostrophes for the check below

    if answer.startswith("I don't know"):
        return {"answer": NOT_FOUND, "sources": [], "retrieved": retrieved}

    # Map the [n] citations in the answer back to real file + page metadata,
    # ignoring any number that doesn't match a retrieved chunk
    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer)})
    sources = [(n, retrieved[n - 1][1]) for n in cited if 1 <= n <= len(docs)]
    return {"answer": answer, "sources": sources, "retrieved": retrieved}


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set. Copy .env.example to .env and add your key.")

    store = get_vector_store()
    llm = ChatOpenAI(model=CHAT_MODEL)
    print(f"Chat model: {CHAT_MODEL}")

    test_questions = [
        "How long should passwords be according to NIST?",             # in the documents
        "What is the minimum password length for single-factor authentication?",  # same fact, more specific wording
        "What is the capital of France?",                               # NOT in the documents
    ]
    for question in test_questions:
        result = answer_question(question, store, llm)
        print(f"\nQ: {question}")
        print(f"A: {result['answer']}")
        if result["sources"]:
            print("Cited:")
            for n, label in result["sources"]:
                print(f"  [{n}] {label}")
        print("Retrieved (all chunks the model was given):")
        for n, label in result["retrieved"]:
            print(f"  [{n}] {label}")