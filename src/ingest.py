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


def build_index():
    docs = PyPDFDirectoryLoader(str(DATA_DIR)).load()
    docs += DirectoryLoader(str(DATA_DIR), glob="**/*.txt", loader_cls=TextLoader).load()
    if not docs:
        raise SystemExit(f"No documents found in {DATA_DIR}. Add PDFs or .txt files first.")
    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
    store = FAISS.from_documents(chunks, get_embeddings())
    store.save_local(str(INDEX_DIR))
    print(f"Indexed {len(chunks)} chunks from {len(docs)} documents -> {INDEX_DIR}")


if __name__ == "__main__":
    build_index()
