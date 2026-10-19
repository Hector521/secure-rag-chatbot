"""ingest.py - Prepare the PDFs for the chatbot.

Step 2.2: load each PDF page as a LangChain Document with its source metadata.
Step 2.3: split pages into smaller overlapping chunks for precise retrieval.
(Embedding and storing get added in Step 3.)
"""
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

DATA_DIR = Path("data")

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


if __name__ == "__main__":
    pages = load_pdfs()
    print(f"Loaded {len(pages)} pages with text")

    pages_per_file = {}
    for doc in pages:
        name = doc.metadata["source"]
        pages_per_file[name] = pages_per_file.get(name, 0) + 1
    for name, count in pages_per_file.items():
        print(f"  {name}: {count} pages")

    chunks = split_documents(pages)
    sizes = [len(c.page_content) for c in chunks]
    print(f"\nSplit into {len(chunks)} chunks")
    print(f"  Average chunk size: {sum(sizes) // len(sizes)} characters")
    print(f"  Largest chunk: {max(sizes)} characters")

    sample = chunks[len(chunks) // 2]  # a chunk from the middle, not the cover page
    print("\nSample chunk - metadata and text:")
    print(sample.metadata)
    print(sample.page_content)