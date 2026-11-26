# Enterprise RAG

Practice project following Krish Naik's "Build Enterprise-Grade RAG Applications | LIVE 8-Hour Marathon"
(https://www.youtube.com/watch?v=bjkjaqUZl4E).

A Retrieval-Augmented Generation (RAG) app: it answers questions **only from your own documents**.

**Pipeline:** load documents -> split into chunks -> embed locally (FastEmbed, `BAAI/bge-small-en-v1.5`) -> store in FAISS -> retrieve top 4 chunks -> LLM (Groq `openai/gpt-oss-20b`) answers from that context.

Runs fully on free tiers: embeddings are computed on your CPU, answers use a free Groq API key.

## Setup (Windows PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env   # put your free Groq key (console.groq.com/keys) in .env
```

## Run
```powershell
.\run_demo.ps1
```
This builds the index (first run only) and opens the web app at http://localhost:8501.

In the web app:
- **Upload documents** (PDF/TXT) from the sidebar and click *Add to knowledge base*. They are saved to `data/` and the index is rebuilt.
- **Remove documents** with the ✕ next to each file.
- **Ask questions**. Each answer shows the source file and page it came from.

A sample PDF to try uploading is in `demo_uploads/`.

Or step by step:
1. Put PDFs or .txt files in `data/` (a sample handbook is included)
2. `python -m src.ingest` - builds the vector index (re-run after adding documents)
3. `streamlit run app.py` - web UI, or `python -m src.rag` - terminal Q&A

## Project structure
| File | What it does |
|---|---|
| `src/ingest.py` | Loads documents, chunks them (1000 chars, 200 overlap), embeds and saves a FAISS index |
| `src/embeddings.py` | Local embedding model (no API key needed) |
| `src/rag.py` | Retriever + prompt + LLM chain (LangChain LCEL) |
| `app.py` | Streamlit web interface: upload/remove documents, Q&A with sources |
| `run_demo.ps1` | One-click demo launcher |
