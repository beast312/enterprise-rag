# One-click demo: activates the venv, builds the index if missing, and opens the web app.
Set-Location $PSScriptRoot
$port = 8501

# Stop any old copy of the app so it always runs on the same port
Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue |
    ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }

& .\.venv\Scripts\Activate.ps1
if (-not (Test-Path .\vectorstore\index.faiss)) {
    Write-Host "Building vector index from data/ ..."
    python -m src.ingest
}

Write-Host ""
Write-Host "Starting the app... it will open at http://localhost:$port"
Write-Host "Keep this window open during the demo. Press Ctrl+C here to stop."
Write-Host ""

# Open the browser a few seconds after the server starts
Start-Job { Start-Sleep 6; Start-Process "http://localhost:$using:port" } | Out-Null
streamlit run app.py --server.port $port --server.headless true
