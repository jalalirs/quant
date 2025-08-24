# Docker Deployment Guide

This directory contains specialized Docker configurations for different use cases of the quantization pipeline.

## Available Images

### 1. Dashboard-Only (`Dockerfile.dashboard`)
**Lightweight container for generating comparison dashboards**
- ✅ **Minimal dependencies** - No CUDA, ONNX, or TensorRT
- ✅ **Fast startup** - Quick dashboard generation
- ✅ **Small size** - ~200MB base image
- ❌ **No model inference** - Dashboard generation only

### 2. CPU-Only LLM (`Dockerfile.llm-cpu`)  
**Full pipeline without GPU acceleration**
- ✅ **No CUDA required** - Works on any machine
- ✅ **Complete pipeline** - Model conversion, inference, benchmarking
- ✅ **ONNX optimization** - CPU-optimized inference
- ❌ **Slower inference** - Limited by CPU performance

### 3. GPU-Enabled LLM (`Dockerfile.llm-gpu`)
**Full-featured pipeline with GPU acceleration**
- ✅ **Maximum performance** - CUDA + TensorRT acceleration  
- ✅ **Complete quantization** - All precision modes (FP16, INT8, MXFP4)
- ✅ **Production ready** - Optimized for high throughput
- ❌ **Requires NVIDIA GPU** - CUDA-compatible hardware needed

## Quick Start

### Using Docker Compose (Recommended)

```bash
# Start all services
cd docker
docker-compose up

# Start specific services
docker-compose up dashboard        # Dashboard only
docker-compose up llm-cpu         # CPU LLM pipeline  
docker-compose up llm-gpu         # GPU LLM pipeline
```

### Manual Docker Commands

#### Dashboard Generation
```bash
# Build and run dashboard
docker build -f docker/Dockerfile.dashboard -t quant-dashboard .
docker run -v $(pwd)/results:/app/results -p 6001:6001 quant-dashboard
```

#### CPU LLM Pipeline
```bash
# Build and run CPU pipeline
docker build -f docker/Dockerfile.llm-cpu -t quant-llm-cpu .
docker run -v $(pwd)/models:/app/models -p 6001:6001 quant-llm-cpu
```

#### GPU LLM Pipeline
```bash
# Build and run GPU pipeline (requires NVIDIA Docker)
docker build -f docker/Dockerfile.llm-gpu -t quant-llm-gpu .
docker run --gpus all -v $(pwd)/models:/app/models -p 6002:6001 quant-llm-gpu
```

## Build Script

Use the automated build script for convenience:

```bash
cd docker

# Build all images
./build.sh --all

# Build specific images
./build.sh --dashboard-only
./build.sh --cpu-only  
./build.sh --gpu-only

# Build and push to registry
./build.sh --all --push --registry docker.io/myuser --tag v1.0
```

## Service Endpoints

| Service | Port | Description |
|---------|------|-------------|
| Dashboard | 6001 | HTML dashboard generation |
| LLM CPU | 6001 | OpenAI-compatible API (CPU) |
| LLM GPU | 6002 | OpenAI-compatible API (GPU) |
| File Server | 8082 | Static file serving for results |

## Configuration

### Environment Variables

#### Dashboard Container
```bash
DASHBOARD_ONLY=1          # Enable dashboard-only mode
```

#### CPU Container  
```bash
CUDA_VISIBLE_DEVICES=""   # Force CPU mode
OMP_NUM_THREADS=4         # CPU thread count
MKL_NUM_THREADS=4         # Intel MKL threads
```

#### GPU Container
```bash
NVIDIA_VISIBLE_DEVICES=all                    # Use all GPUs
NVIDIA_DRIVER_CAPABILITIES=compute,utility     # Required capabilities
```

### Volume Mounts

#### Required Volumes
```bash
# Model storage and cache
-v $(pwd)/models:/app/models
-v $(pwd)/model_cache:/app/model_cache

# Results and configuration  
-v $(pwd)/results:/app/results
-v $(pwd)/configs:/app/configs
```

#### Optional Volumes
```bash
# Custom datasets
-v $(pwd)/dataset:/app/dataset

# Logs
-v $(pwd)/logs:/app/logs
```

## Docker Compose Configuration

### Full Stack Deployment

```yaml
version: '3.8'
services:
  # Generate dashboards from benchmark results
  dashboard:
    build:
      dockerfile: docker/Dockerfile.dashboard
    ports:
      - "6001:6001"
    volumes:
      - ./results:/app/results
      - ./configs:/app/configs
    
  # CPU inference for development/testing
  llm-cpu:
    build:
      dockerfile: docker/Dockerfile.llm-cpu
    ports:
      - "6001:6001"  
    volumes:
      - ./models:/app/models
      - ./results:/app/results
    
  # GPU inference for production
  llm-gpu:
    build:
      dockerfile: docker/Dockerfile.llm-gpu
    ports:
      - "6002:6001"
    volumes:
      - ./models:/app/models
      - ./results:/app/results
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## Production Deployment

### Kubernetes Deployment

```yaml
# kubernetes/dashboard-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: quant-dashboard
spec:
  replicas: 2
  selector:
    matchLabels:
      app: quant-dashboard
  template:
    metadata:
      labels:
        app: quant-dashboard
    spec:
      containers:
      - name: dashboard
        image: quant-dashboard:latest
        ports:
        - containerPort: 6001
        volumeMounts:
        - name: results
          mountPath: /app/results
        - name: configs  
          mountPath: /app/configs
      volumes:
      - name: results
        persistentVolumeClaim:
          claimName: quant-results-pvc
      - name: configs
        configMap:
          name: quant-configs
```

### Docker Swarm Deployment

```yaml
# docker-stack.yml
version: '3.8'
services:
  dashboard:
    image: quant-dashboard:latest
    ports:
      - "6001:6001"
    volumes:
      - results:/app/results
    deploy:
      replicas: 2
      resources:
        limits:
          memory: 1G
        reservations:
          memory: 512M
          
  llm-gpu:
    image: quant-llm-gpu:latest
    ports:
      - "6001:6001"
    volumes:
      - models:/app/models
    deploy:
      replicas: 1
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

volumes:
  results:
  models:
```

Deploy with:
```bash
docker stack deploy -c docker-stack.yml quant
```

## Image Optimization

### Multi-stage Build Example

```dockerfile
# Optimized dashboard Dockerfile
FROM python:3.10-slim as builder

# Install build dependencies
RUN pip install --no-cache-dir build

# Copy source and build wheel
COPY . /src
WORKDIR /src
RUN python -m build

# Runtime stage
FROM python:3.10-slim

# Install only runtime dependencies
COPY --from=builder /src/dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && rm /tmp/*.whl

# Runtime setup
WORKDIR /app
COPY configs/ configs/
CMD ["python", "-m", "quant.dashboard"]
```

### Size Optimization Tips

```dockerfile
# Use alpine for smaller images
FROM python:3.10-alpine

# Multi-stage builds to reduce size
FROM node:16 AS frontend-build
# ... build frontend assets ...

FROM python:3.10-slim AS runtime
COPY --from=frontend-build /app/dist /app/static

# Combine RUN commands to reduce layers
RUN apt-get update && apt-get install -y \
    gcc g++ && \
    pip install --no-cache-dir -r requirements.txt && \
    apt-get purge -y gcc g++ && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*
```

## Monitoring and Logging

### Health Checks

```dockerfile
# Custom health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "
import requests
import sys
try:
    r = requests.get('http://localhost:6001/health', timeout=5)
    sys.exit(0 if r.status_code == 200 else 1)
except:
    sys.exit(1)
"
```

### Logging Configuration

```yaml
# docker-compose.yml logging
services:
  llm-gpu:
    logging:
      driver: "json-file"
      options:
        max-size: "100m"
        max-file: "3"
        
  # Or use external logging
  llm-gpu:
    logging:
      driver: "fluentd"
      options:
        fluentd-address: "localhost:24224"
        tag: "quant.llm-gpu"
```

## Security Best Practices

### Non-root User

```dockerfile
# Create non-root user
RUN groupadd -r quant && useradd -r -g quant quant

# Change ownership
RUN chown -R quant:quant /app
USER quant

# Run as non-root
CMD ["python", "quant.py"]
```

### Resource Limits

```yaml
# docker-compose.yml
services:
  llm-cpu:
    mem_limit: 8g
    mem_reservation: 4g
    cpus: '4.0'
    
  llm-gpu:
    mem_limit: 16g
    mem_reservation: 8g
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## Troubleshooting

### Common Issues

#### 1. GPU Not Detected
```bash
# Check NVIDIA Docker runtime
docker info | grep nvidia

# Test GPU access
docker run --gpus all nvidia/cuda:12.1-base nvidia-smi

# Fix: Install nvidia-container-toolkit
sudo apt install nvidia-container-toolkit
sudo systemctl restart docker
```

#### 2. Out of Memory
```bash
# Monitor memory usage
docker stats

# Reduce model size in config
model:
  max_length: 1024        # Reduce from 2048
  max_batch_size: 1       # Reduce batch size
```

#### 3. Permission Issues
```bash
# Fix volume permissions
sudo chown -R $(id -u):$(id -g) ./models ./results

# Or use user mapping in compose
services:
  llm-cpu:
    user: "${UID}:${GID}"
```

#### 4. Model Download Issues
```bash
# Pre-download models
docker run -v $(pwd)/model_cache:/app/model_cache quant-llm-cpu \
  python -c "
from transformers import AutoTokenizer, AutoModel
model = AutoModel.from_pretrained('microsoft/DialoGPT-medium', 
                                 cache_dir='/app/model_cache')
"
```

### Debug Mode

```bash
# Run with debug logging
docker run -e LOG_LEVEL=DEBUG quant-llm-cpu

# Interactive debugging
docker run -it --entrypoint /bin/bash quant-llm-cpu

# Check container logs
docker logs quant-llm-cpu -f
```

### Performance Tuning

#### CPU Optimization
```dockerfile
# CPU-specific optimizations
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4
ENV NUMEXPR_NUM_THREADS=4

# Use optimized BLAS
RUN pip install intel-mkl
```

#### GPU Optimization  
```dockerfile
# GPU memory optimization
ENV CUDA_LAUNCH_BLOCKING=1
ENV CUDA_CACHE_DISABLE=1

# Multi-GPU support
ENV NCCL_DEBUG=INFO
```

## Registry and Distribution

### Push to Registry

```bash
# Build and tag
./build.sh --all --tag v1.2.0

# Push to Docker Hub
docker login
docker push myuser/quant-dashboard:v1.2.0
docker push myuser/quant-llm-cpu:v1.2.0  
docker push myuser/quant-llm-gpu:v1.2.0

# Push to private registry
./build.sh --registry registry.company.com/ai --push
```

### Image Scanning

```bash
# Scan for vulnerabilities
docker scan quant-llm-gpu:latest

# Use trivy for detailed scanning
trivy image quant-llm-gpu:latest
```

This comprehensive Docker setup provides flexible deployment options for different environments and use cases, from development to production!