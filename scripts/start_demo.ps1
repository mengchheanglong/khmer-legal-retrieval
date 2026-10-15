# Start the local defense demo using already downloaded model files.
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
if (-not (Test-Path -LiteralPath 'data/indices/khmer_benchmark_embeddings.pt')) {
    throw 'Demo index missing. Run: python scripts/build_demo_index.py'
}
$env:HF_HUB_OFFLINE = '1'
$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
$env:PYTHONHASHSEED = '42'
python -m streamlit run streamlit_app.py --server.address 127.0.0.1
