#!/bin/bash
# scripts/01_upgrade_to_gpu.sh
# Upgrade PyTorch to GPU version when GPU becomes available

set -e

echo "=========================================="
echo "Upgrading to GPU PyTorch + DeepSpeed"
echo "=========================================="

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "❌ Error: Virtual environment not activated!"
    echo "Run: source venv/bin/activate"
    exit 1
fi

echo ""
echo "[1/4] Uninstalling CPU PyTorch..."
pip uninstall torch torchvision torchaudio -y
echo "✓ CPU PyTorch removed"

echo ""
echo "[2/4] Installing GPU PyTorch (CUDA 12.1)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
echo "✓ GPU PyTorch installed"

echo ""
echo "[3/4] Installing DeepSpeed..."
pip install deepspeed>=0.12.0
echo "✓ DeepSpeed installed"

echo ""
echo "[4/4] Verifying GPU setup..."

python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"
python -c "import torch; print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
python -c "import deepspeed; print(f'DeepSpeed version: {deepspeed.__version__}')"

echo ""
echo "=========================================="
echo "✓ GPU Setup Complete!"
echo "=========================================="