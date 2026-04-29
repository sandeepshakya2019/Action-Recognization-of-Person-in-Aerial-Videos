#!/bin/bash
# -----------------------------
# Setup Python environment for GTX 1080 Ti (sm_61)
# Latest stable PyTorch + dependencies
# -----------------------------

ENV_DIR="$HOME/mtp/python-env-1080ti"
PYTHON_BIN=$(which python3)

echo "[INFO] Creating virtual environment at $ENV_DIR"
$PYTHON_BIN -m venv $ENV_DIR

echo "[INFO] Activating environment"
source $ENV_DIR/bin/activate

echo "[INFO] Upgrading pip, setuptools, and wheel"
pip install --upgrade pip setuptools wheel

echo "[INFO] Installing latest stable PyTorch with CUDA support"
# Automatically installs the latest version with CUDA if supported
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo "[INFO] Installing other latest dependencies"
pip install pandas matplotlib scikit-learn pillow tqdm opencv-python jupyter

echo "[INFO] Verifying installation and GPU"
python - <<EOF
import torch
import numpy as np
print("NumPy version:", np.__version__)
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device:", torch.cuda.get_device_name(0))
EOF

echo "[INFO] Setup complete!"
echo "To activate this environment in the future:"
echo "source $ENV_DIR/bin/activate"
