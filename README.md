# Enterprise RAG

Practice project following Krish Naik's "Build Enterprise-Grade RAG Applications | LIVE 8-Hour Marathon"
(https://www.youtube.com/watch?v=bjkjaqUZl4E). Starter RAG pipeline: load documents -> chunk -> embed -> FAISS -> LLM answer.

## Setup (Windows PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env   # then put your API keys in .env
```

## Run
1. Put PDFs or .txt files in `data/`
2. `python -m src.ingest` - builds the vector index
3. `python -m src.rag` - ask questions in the terminal, or `streamlit run app.py` for a web UI

Add your own code from the video in `src/` and `notebooks/` as you follow along.
