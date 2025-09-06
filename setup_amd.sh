#!/bin/bash

# AI Auto Transcripts Setup for AMD Radeon 6900 XT
# This script sets up the environment for AMD GPU acceleration

set -e

echo "Setting up AI Auto Transcripts with AMD GPU support..."

# Create and activate virtual environment
echo "Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install ROCm PyTorch first (AMD GPU support)
echo "Installing PyTorch with ROCm support for AMD GPU..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6

# Install other requirements
echo "Installing other Python dependencies..."
pip install -r requirements_amd.txt

# Check AMD GPU availability
echo "Checking AMD GPU availability..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU device: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('No GPU detected - will use CPU')
"

echo ""
echo "Setup complete! To use the environment:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Copy .env.example to .env and add your API keys"
echo "3. Run test: python test_pipeline.py"
echo "4. Run main pipeline: python main.py"