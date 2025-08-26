#!/bin/bash

# Exit on any error
set -e

echo "=== Quant Benchmark Runner ==="
echo "Starting gpt-oss-mxfp4 benchmark..."

# Set environment variables for GPU optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NVIDIA_VISIBLE_DEVICES=0
export NVIDIA_DRIVER_CAPABILITIES=compute,utility
export CUDA_VISIBLE_DEVICES=0

# Clone the repository if it doesn't exist
if [ ! -d "quant" ]; then
    echo "Cloning repository from github.com/jalalirs/quant.git..."
    git clone https://github.com/jalalirs/quant.git
    cd quant
else
    echo "Repository already exists, using existing code..."
    cd quant
fi

# Upgrade pip first
echo "=== Upgrading pip ==="
pip install --upgrade pip

# Check pre-installed versions from TensorRT-LLM container
echo "=== Pre-installed Software Versions ==="
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python3 -c "import tensorrt; print(f'TensorRT version: {tensorrt.__version__}')"
python3 -c "import triton; print(f'Triton version (pre-installed): {triton.__version__}')" || echo "Standard Triton not found"
python3 -c "import sys; print(f'Python version: {sys.version}')"
nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -1 | xargs -I {} echo "CUDA Driver version: {}"

# Install Triton 3.4+ for MXFP4 support (critical upgrade)
echo "=== Installing Triton 3.4+ for MXFP4 support ==="
pip install "triton>=3.4" --upgrade --break-system-packages

# Install kernels library for MXFP4 and Flash Attention 3
echo "=== Installing kernels library ==="
pip install --upgrade kernels --break-system-packages

# Install requirements
echo "=== Installing requirements ==="
pip install -r docker/requirements.llm-gpu.txt

# Verify MXFP4 and Flash Attention 3 readiness
echo "=== MXFP4 & Flash Attention 3 Verification ==="
python3 -c "import triton; print(f'Triton version (upgraded): {triton.__version__}')"
python3 -c "import kernels; print('Kernels library ready for MXFP4/Flash Attention 3')"

# Create necessary directories for Azure ML
echo "=== Creating Azure ML directories ==="
mkdir -p /tmp/benchmark_results
mkdir -p /tmp/models
mkdir -p /tmp/model_cache
mkdir -p results models model_cache

# Set Python path
export PYTHONPATH=$(pwd)

# Verify GPU availability
echo "=== GPU Verification ==="
python3 -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
else:
    print('No CUDA GPU available')
"

# Run the gpt-oss-mxfp4 benchmark
echo "=== Running gpt-oss-mxfp4 Benchmark ==="
python3 quant.py --config configs/gpt_oss_20b_mxfp4.yml --verbose
python3 quant.py --config configs/gpt_oss_20b_standard.yml --verbose
python3 quant.py --config configs/gpt_oss_20b_moe_only.yml --verbose



echo "=== Benchmark Complete ==="
echo "Results saved to: /tmp/benchmark_results/gpt_oss_20b_mxfp4_benchmark.json"

# Exit successfully
exit 0
