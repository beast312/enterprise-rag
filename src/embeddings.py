"""Free, local embeddings (runs on CPU via ONNX, no API key needed)."""
from pathlib import Path
from langchain_community.embeddings import FastEmbedEmbeddings

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
# Keep the downloaded model inside the project (not the temp folder) so it survives cleanups.
CACHE_DIR = Path(__file__).resolve().parent.parent / ".models"


def get_embeddings():
    return FastEmbedEmbeddings(model_name=EMBED_MODEL, cache_dir=str(CACHE_DIR))
