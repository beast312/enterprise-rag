from pathlib import Path
import streamlit as st
from src.ingest import DATA_DIR, INDEX_DIR, build_index, list_documents
from src.rag import ask, load_retriever

st.set_page_config(page_title="Enterprise RAG Q&A", page_icon="📄", initial_sidebar_state="expanded")
st.title("Enterprise RAG Q&A")
st.caption("Upload your documents, then ask questions. Answers come only from those documents.")


@st.cache_resource
def get_retriever():
    return load_retriever()


def rebuild():
    files, chunks = build_index()
    get_retriever.clear()
    return files, chunks


# ---------- Sidebar: knowledge base ----------
with st.sidebar:
    st.header("📁 Knowledge base")
    uploads = st.file_uploader(
        "Upload PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True
    )
    if uploads and st.button("Add to knowledge base", type="primary"):
        for f in uploads:
            (DATA_DIR / Path(f.name).name).write_bytes(f.getbuffer())
        with st.spinner("Reading, chunking and embedding documents..."):
            files, chunks = rebuild()
        st.success(f"Indexed {chunks} chunks from {files} documents.")

    st.subheader("Documents")
    docs = list_documents()
    for name in docs:
        col1, col2 = st.columns([5, 1])
        col1.write(f"📄 {name}")
        if col2.button("✕", key=f"del-{name}", help=f"Remove {name}"):
            (DATA_DIR / name).unlink()
            if list_documents():
                with st.spinner("Rebuilding index..."):
                    rebuild()
            st.rerun()
    if not docs:
        st.info("No documents yet. Upload one above.")

# ---------- Main: Q&A ----------
if not docs:
    st.stop()
if not (INDEX_DIR / "index.faiss").exists():
    with st.spinner("Building index for the first time..."):
        rebuild()

question = st.text_input("Ask a question about your documents")
if question:
    with st.spinner("Searching documents and generating answer..."):
        answer, sources = ask(get_retriever(), question)
    st.markdown(answer)
    with st.expander("📚 Sources used for this answer"):
        for i, d in enumerate(sources, 1):
            name = Path(d.metadata.get("source", "unknown")).name
            page = d.metadata.get("page")
            label = f"{name}, page {page + 1}" if isinstance(page, int) else name
            st.markdown(f"**{i}. {label}**")
            st.text(d.page_content[:500] + ("..." if len(d.page_content) > 500 else ""))
