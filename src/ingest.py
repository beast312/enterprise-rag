"""Load PDFs/text from data/, split into chunks, embed, and save a FAISS index."""
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader, DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.embeddings import get_embeddings
from langchain_community.vectorstores import FAISS

load_dotenv()
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
INDEX_DIR = Path(__file__).resolve().parent.parent / "vectorstore"
SUPPORTED = (".pdf", ".txt")


def list_documents():
    """Names of the documents currently in the knowledge base."""
    return sorted(p.name for p in DATA_DIR.iterdir() if p.suffix.lower() in SUPPORTED)


def build_index():
    """Rebuild the FAISS index from every document in data/. Returns (files, chunks)."""
    docs = PyPDFDirectoryLoader(str(DATA_DIR)).load()
    docs += DirectoryLoader(
        str(DATA_DIR), glob="**/*.txt", loader_cls=TextLoader,
        loader_kwargs={"autodetect_encoding": True},
    ).load()
    if not docs:
        raise ValueError(f"No documents found in {DATA_DIR}. Add PDFs or .txt files first.")
    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
    store = FAISS.from_documents(chunks, get_embeddings())
    store.save_local(str(INDEX_DIR))
    return len(list_documents()), len(chunks)


if __name__ == "__main__":
    files, chunks = build_index()
    print(f"Indexed {chunks} chunks from {files} documents -> {INDEX_DIR}")
