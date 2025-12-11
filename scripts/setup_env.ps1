# ==============================================================================
# HAI-ML Environment Setup Script (Windows PowerShell)
# ==============================================================================

Write-Host "=============================================="
Write-Host "HAI-ML Environment Setup"
Write-Host "=============================================="

# Check Python version
$pythonVersion = python --version 2>&1
Write-Host "Found Python version: $pythonVersion"

# Create virtual environment
$venvDir = ".venv"
if (-not (Test-Path $venvDir)) {
    Write-Host "Creating virtual environment in $venvDir..."
    python -m venv $venvDir
} else {
    Write-Host "Virtual environment already exists in $venvDir"
}

# Activate virtual environment
Write-Host "Activating virtual environment..."
& "$venvDir\Scripts\Activate.ps1"

# Upgrade pip
Write-Host "Upgrading pip..."
python -m pip install --upgrade pip wheel setuptools

# Install PyTorch
Write-Host "Installing PyTorch..."
pip install torch>=2.0.0

# Install main requirements
Write-Host "Installing requirements..."
pip install -r requirements.txt

# Install additional ML dependencies
Write-Host "Installing ML dependencies..."
pip install d3rlpy gymnasium seaborn pytest

# Create necessary directories
Write-Host "Creating directory structure..."
New-Item -ItemType Directory -Force -Path data/hai | Out-Null
New-Item -ItemType Directory -Force -Path data/processed | Out-Null
New-Item -ItemType Directory -Force -Path models | Out-Null
New-Item -ItemType Directory -Force -Path logs | Out-Null
New-Item -ItemType Directory -Force -Path results | Out-Null
New-Item -ItemType Directory -Force -Path paper/figs | Out-Null
New-Item -ItemType Directory -Force -Path paper/tables | Out-Null

Write-Host ""
Write-Host "=============================================="
Write-Host "Verifying installation..."
Write-Host "=============================================="

python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python -c "import matplotlib; print(f'Matplotlib: {matplotlib.__version__}')"
python -c "import hai_ml; print(f'hai_ml: {hai_ml.__version__}')"

# Check for optional packages
try { python -c "import d3rlpy; print(f'd3rlpy: {d3rlpy.__version__}')" } catch { Write-Host "d3rlpy: not installed" }
try { python -c "import gymnasium; print(f'Gymnasium: {gymnasium.__version__}')" } catch { Write-Host "Gymnasium: not installed" }

Write-Host ""
Write-Host "=============================================="
Write-Host "Setup complete!"
Write-Host "=============================================="
