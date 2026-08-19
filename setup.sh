#!/usr/bin/env bash
# Bootstrap the ml_pipeline venv on Ubuntu.
# CUDA 13 driver / RTX 3090 — install torch from PyTorch's CUDA 12.4 wheels (forward-compatible).
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -d .venv ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate
pip install --upgrade pip wheel

echo "=== Installing CPU+CUDA deps ==="
pip install -r requirements.txt

echo "=== Installing PyTorch (CUDA 12.4 wheels, works on CUDA 13 driver) ==="
pip install torch --index-url https://download.pytorch.org/whl/cu124

echo "=== Smoke test: CUDA visible? ==="
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); \
print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu only')"

echo
echo "Done. Activate with: source ~/ml_pipeline/.venv/bin/activate"
