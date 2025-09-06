#!/bin/bash

# Setup virtual environment for AI Auto Transcripts (AMD GPU compatible)
# Run this script to set up the project in a Python virtual environment

set -e

echo "Setting up AI Auto Transcripts in Python virtual environment..."

# Create virtual environment
echo "Creating Python virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with ROCm support for AMD GPU (since ROCm is already on system)
echo "Installing PyTorch with ROCm support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6

# Install other requirements (without the problematic whisper-cpp-python)
echo "Installing other dependencies..."
pip install openai anthropic feedparser requests pydub pyyaml python-dotenv openai-whisper pyannote-audio transformers

# Test GPU availability
echo "Testing GPU availability..."
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'ROCm/CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU device: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

echo ""
echo "Virtual environment setup complete!"
echo "To activate: source venv/bin/activate"
echo "To deactivate: deactivate"