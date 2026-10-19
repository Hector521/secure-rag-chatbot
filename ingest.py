"""ingest.py - Prepare the PDFs for the chatbot (run this once, or after changing PDFs).

Pipeline: load PDFs -> split into chunks -> embed each chunk -> store in Chroma.
"""
import os
import shutil
from pathlib import Path

# Turn off Chroma's anonymous usage telemetry (privacy: no usage data leaves the machine)
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

DATA_DIR = Path("data")
CHROMA_DIR = "chroma_db"              # local vector database folder (git-ignored)
COLLECTION_NAME = "security_guides"
EMBEDDING_MODEL = "text-embedding-3-small"

# Chunking settings (measured in characters, not words)
CHUNK_SIZE = 1000     # roughly one long paragraph
CHUNK_OVERLAP = 150   # repeated text between neighbors so nothing is cut in half


def load_pdfs(data_dir: Path = DATA_DIR) -> list[Document]:
    """Read every PDF in data_dir and return one Document per page.

    Each Document keeps the page text plus metadata (file name and page
    number), so the chatbot can cite exactly where an answer came from.
    """
    documents = []
    for pdf_path in sorted(data_dir.glob("*.pdf")):
        reader = PdfReader(pdf_path)
        for page_index, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if not text.strip():
                continue  # skip blank or image-only pages
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": pdf_path.name,
                        "page": page_index + 1,  # people count pages from 1, Python from 0
                    },
                )
            )
    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    """Split page-level Documents into smaller overlapping chunks.

    Each chunk inherits its page's metadata (source + page), so citations
    still point to the right file and page after splitting.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # Try the most natural break points first: paragraphs, lines, sentences, words
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def build_vector_store(chunks: list[Document]) -> Chroma:
    """Embed every chunk with OpenAI and save the vectors to a local Chroma database.

    The old database is deleted first, so re-running ingest never stores
    duplicate chunks.
    """
    if Path(CHROMA_DIR).exists():
        shutil.rmtree(CHROMA_DIR)

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
    )


if __name__ == "__main__":
    # Read OPENAI_API_KEY from .env into the environment (the key never appears in code)
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set. Copy .env.example to .env and add your key.")

    pages = load_pdfs()
    print(f"Loaded {len(pages)} pages from {len({p.metadata['source'] for p in pages})} PDFs")

    chunks = split_documents(pages)
    print(f"Split into {len(chunks)} chunks")

    print(f"Embedding with {EMBEDDING_MODEL} and saving to {CHROMA_DIR}/ ...")
    store = build_vector_store(chunks)
    print(f"Stored {store._collection.count()} chunks in Chroma")

    # Quick retrieval test: embeds one short question (costs a tiny fraction of a cent)
    question = "How long should passwords be?"
    print(f"\nTest search: {question!r}")
    for i, doc in enumerate(store.similarity_search(question, k=3), start=1):
        preview = " ".join(doc.page_content.split())[:150]
        print(f"  {i}. [{doc.metadata['source']}, page {doc.metadata['page']}] {preview}...")