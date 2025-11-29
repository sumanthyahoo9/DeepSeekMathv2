#!/bin/bash
# scripts/00_setup_environment.sh
# DeepSeekMath-V2 Environment Setup Script
# Purpose: Automated setup of Python virtual environment and dependencies

set -e  # Exit immediately if any command fails

echo "=========================================="
echo "DeepSeekMath-V2 Environment Setup"
echo "=========================================="

# ============================================================================
# STEP 1: Check Python Version
# ============================================================================
echo ""
echo "[1/6] Checking Python version..."

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
REQUIRED_VERSION="3.11"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "❌ Error: Python $REQUIRED_VERSION or higher required. Found: $PYTHON_VERSION"
    exit 1
fi

echo "✓ Python version: $PYTHON_VERSION"

# ============================================================================
# STEP 2: Create Virtual Environment
# ============================================================================
echo ""
echo "[2/6] Creating virtual environment..."

if [ -d "venv" ]; then
    echo "⚠️  Virtual environment already exists. Removing old one..."
    rm -rf venv
fi

python3 -m venv venv
echo "✓ Virtual environment created"

# ============================================================================
# STEP 3: Activate Virtual Environment
# ============================================================================
echo ""
echo "[3/6] Activating virtual environment..."

source venv/bin/activate
echo "✓ Virtual environment activated"

# ============================================================================
# STEP 4: Upgrade Core Tools
# ============================================================================
echo ""
echo "[4/6] Upgrading pip, setuptools, and wheel..."

pip install --upgrade pip setuptools wheel --quiet
echo "✓ Core tools upgraded"

# ============================================================================
# STEP 5: Install PyTorch (CPU version - will upgrade when GPU available)
# ============================================================================
echo ""
echo "[5/6] Installing PyTorch (CPU version)..."
echo "Note: This installs CPU-only PyTorch. When GPU is available, run upgrade script."

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --quiet
echo "✓ PyTorch installed"

# Verify PyTorch installation
TORCH_VERSION=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "failed")
if [ "$TORCH_VERSION" = "failed" ]; then
    echo "❌ PyTorch installation verification failed"
    exit 1
fi
echo "   PyTorch version: $TORCH_VERSION"

# ============================================================================
# STEP 6: Install Project Requirements
# ============================================================================
echo ""
echo "[6/6] Installing project requirements..."

if [ ! -f "requirements.txt" ]; then
    echo "❌ requirements.txt not found!"
    exit 1
fi

pip install -r requirements.txt --quiet
echo "✓ Requirements installed"

# ============================================================================
# Create Project Directory Structure
# ============================================================================
echo ""
echo "Creating project directories..."

# Data directories
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/annotations
mkdir -p data/auto_labeled

# Experiment directories
mkdir -p experiments/logs
mkdir -p experiments/checkpoints
mkdir -p experiments/results
mkdir -p experiments/runs

# Cache directories
mkdir -p .cache/models
mkdir -p .cache/datasets

echo "✓ Directory structure created"

# ============================================================================
# Verify Installation
# ============================================================================
echo ""
echo "Verifying installation..."

python -c "import torch; print('✓ PyTorch OK')" || echo "❌ PyTorch import failed"
python -c "import transformers; print('✓ Transformers OK')" || echo "❌ Transformers import failed"
python -c "import yaml; print('✓ PyYAML OK')" || echo "❌ PyYAML import failed"
python -c "import datasets; print('✓ Datasets OK')" || echo "❌ Datasets import failed"

# ============================================================================
# Success Message
# ============================================================================
echo ""
echo "=========================================="
echo "✓ Setup Complete!"
echo "=========================================="
echo ""
echo "Virtual environment created and activated."
echo ""
echo "📝 Next steps:"
echo ""
echo "1. Keep the environment activated:"
echo "   source venv/bin/activate"
echo ""
echo "2. When GPU becomes available, upgrade PyTorch:"
echo "   bash scripts/01_upgrade_to_gpu.sh"
echo ""
echo "3. Start coding:"
echo "   - Build prompts: src/utils/prompts.py"
echo "   - Build configs: src/utils/config_loader.py"
echo "   - Build rewards: src/training/reward_functions.py"
echo ""
echo "4. Run tests:"
echo "   pytest tests/"
echo ""
echo "Happy coding! 🚀"
echo ""