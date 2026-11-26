# One-click demo: activates the venv, builds the index if missing, and opens the web app.
Set-Location $PSScriptRoot
& .\.venv\Scripts\Activate.ps1
if (-not (Test-Path .\vectorstore\index.faiss)) {
    Write-Host "Building vector index from data/ ..."
    python -m src.ingest
}
streamlit run app.py
