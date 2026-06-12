#!/usr/bin/env bash
# ============================================================================
# One-shot environment setup for Ubuntu 24.04 (WSL2).
# Creates a venv and installs PyTorch (CUDA if available, else CPU) + deps.
# Usage:  bash setup_wsl.sh
# ============================================================================
set -e

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

echo "==> Installing system packages (python venv, OpenCV runtime libs)"
sudo apt-get update -y
sudo apt-get install -y python3-venv python3-pip libgl1 libglib2.0-0

echo "==> Creating virtual environment (.venv)"
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip wheel

# --- Pick a Torch build: CUDA (WSL2 + NVIDIA driver) or CPU -----------------
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  echo "==> NVIDIA GPU detected in WSL -> installing CUDA build of PyTorch"
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
else
  echo "==> No GPU detected -> installing CPU build of PyTorch"
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

echo "==> Installing project requirements"
pip install -r requirements.txt

echo ""
echo "==> Verifying install"
python - <<'PY'
import torch, cv2, ultralytics
print("torch      :", torch.__version__, "| CUDA available:", torch.cuda.is_available())
print("opencv     :", cv2.__version__)
print("ultralytics:", ultralytics.__version__)
PY

echo ""
echo "Done. Activate with:  source .venv/bin/activate"
