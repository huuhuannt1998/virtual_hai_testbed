#!/bin/bash
# ==============================================================================
# HAI-ML Environment Setup Script
# ==============================================================================
# Sets up the Python virtual environment with all required dependencies.
#
# Usage:
#   chmod +x scripts/setup_env.sh
#   ./scripts/setup_env.sh
# ==============================================================================

set -e  # Exit on error

echo "=============================================="
echo "HAI-ML Environment Setup"
echo "=============================================="

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

echo "Found Python version: $PYTHON_VERSION"

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 9 ]); then
    echo "ERROR: Python 3.9+ is required. Found: $PYTHON_VERSION"
    exit 1
fi

# Create virtual environment
VENV_DIR=".venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in $VENV_DIR..."
    python3 -m venv $VENV_DIR
else
    echo "Virtual environment already exists in $VENV_DIR"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source $VENV_DIR/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip wheel setuptools

# Install PyTorch (CUDA 11.8 by default, adjust as needed)
echo "Installing PyTorch..."
if command -v nvidia-smi &> /dev/null; then
    echo "CUDA detected, installing PyTorch with CUDA support..."
    pip install torch>=2.2.0 --index-url https://download.pytorch.org/whl/cu118
else
    echo "No CUDA detected, installing CPU-only PyTorch..."
    pip install torch>=2.2.0 --index-url https://download.pytorch.org/whl/cpu
fi

# Install main requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Install d3rlpy (offline RL)
echo "Installing d3rlpy for offline RL..."
pip install d3rlpy>=2.6.0

# Install development dependencies
echo "Installing development dependencies..."
pip install pytest pytest-cov black isort mypy

# Create necessary directories
echo "Creating directory structure..."
mkdir -p data/hai
mkdir -p data/processed
mkdir -p models
mkdir -p logs
mkdir -p results
mkdir -p paper/figs

# Set up pre-commit hooks (optional)
if command -v pre-commit &> /dev/null; then
    echo "Setting up pre-commit hooks..."
    pre-commit install
fi

# Verify installation
echo ""
echo "=============================================="
echo "Verifying installation..."
echo "=============================================="

python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import d3rlpy; print(f'd3rlpy: {d3rlpy.__version__}')"
python3 -c "import gymnasium; print(f'Gymnasium: {gymnasium.__version__}')"
python3 -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python3 -c "import matplotlib; print(f'Matplotlib: {matplotlib.__version__}')"

# Check CUDA availability
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    python3 -c "import torch; print(f'CUDA device: {torch.cuda.get_device_name(0)}')"
fi

echo ""
echo "=============================================="
echo "Setup complete!"
echo "=============================================="
echo ""
echo "To activate the environment:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "To run Track A (Behavior Cloning):"
echo "  ./scripts/run_trackA.sh p3"
echo ""
echo "To run Track B (Offline RL):"
echo "  ./scripts/run_trackB.sh p3"
echo ""
echo "To run Track C (MPC):"
echo "  ./scripts/run_trackC.sh p3"
echo ""
