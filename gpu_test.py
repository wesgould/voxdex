#!/usr/bin/env python3
"""Test script to verify AMD GPU support"""

import torch
import sys

def test_gpu_support():
    """Test if AMD GPU is detected and usable"""
    print("=== GPU Support Test ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA/ROCm available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"Device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            device_name = torch.cuda.get_device_name(i)
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / 1e9
            print(f"Device {i}: {device_name}")
            print(f"  Memory: {memory_gb:.1f} GB")
        
        # Test tensor operations on GPU
        try:
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()
            z = torch.matmul(x, y)
            print("✓ GPU tensor operations working")
        except Exception as e:
            print(f"✗ GPU tensor operations failed: {e}")
            
    else:
        print("No GPU detected - check ROCm installation")
        print("Make sure to install PyTorch with: pip install torch --index-url https://download.pytorch.org/whl/rocm5.6")
    
    print("\n=== Test Complete ===")


if __name__ == "__main__":
    test_gpu_support()