#!/bin/bash
set -e

# Detect operating system
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Warning: macOS detected. Distributed training with torchrun is not fully supported on macOS."
    echo "This script is primarily designed for Linux systems with multiple GPUs."
    echo "For best performance, please run this on a Linux system with CUDA-capable GPUs."
    exit 1
elif [[ "$OSTYPE" == "msys"* ]] || [[ "$OSTYPE" == "win32"* ]]; then
    echo "Warning: Windows detected. Distributed training with torchrun is not fully supported on Windows."
    echo "This script is primarily designed for Linux systems with multiple GPUs."
    echo "For best performance, please run this on a Linux system with CUDA-capable GPUs."
    exit 1
fi

echo "Installing requirements..."
pip install -U -r requirements.txt

echo "Running train_ddp_single_run.py..."
torchrun --nproc_per_node=2 --nnodes=1 train_ddp_single_run.py
