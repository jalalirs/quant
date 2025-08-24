#!/bin/bash
# Build script for all Docker images

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================${NC}"
}

# Default values
BUILD_DASHBOARD=true
BUILD_CPU=true
BUILD_GPU=false
PUSH_IMAGES=false
IMAGE_TAG="latest"
REGISTRY=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dashboard-only)
            BUILD_DASHBOARD=true
            BUILD_CPU=false
            BUILD_GPU=false
            shift
            ;;
        --cpu-only)
            BUILD_DASHBOARD=false
            BUILD_CPU=true
            BUILD_GPU=false
            shift
            ;;
        --gpu-only)
            BUILD_DASHBOARD=false
            BUILD_CPU=false
            BUILD_GPU=true
            shift
            ;;
        --all)
            BUILD_DASHBOARD=true
            BUILD_CPU=true
            BUILD_GPU=true
            shift
            ;;
        --gpu)
            BUILD_GPU=true
            shift
            ;;
        --push)
            PUSH_IMAGES=true
            shift
            ;;
        --tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        --registry)
            REGISTRY="$2/"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dashboard-only    Build only dashboard image"
            echo "  --cpu-only         Build only CPU LLM image"
            echo "  --gpu-only         Build only GPU LLM image"
            echo "  --all              Build all images (default: dashboard + cpu)"
            echo "  --gpu              Include GPU image in build"
            echo "  --push             Push images to registry"
            echo "  --tag TAG          Image tag (default: latest)"
            echo "  --registry REG     Registry prefix (e.g., docker.io/myuser)"
            echo "  --help             Show this help message"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check Docker
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed or not in PATH"
    exit 1
fi

# Check if we're in the right directory
if [[ ! -f "../quant.py" ]]; then
    print_error "Please run this script from the docker/ directory"
    exit 1
fi

print_header "QUANT DOCKER BUILD SCRIPT"

print_status "Build configuration:"
echo "  Dashboard: $BUILD_DASHBOARD"
echo "  CPU LLM:   $BUILD_CPU" 
echo "  GPU LLM:   $BUILD_GPU"
echo "  Tag:       $IMAGE_TAG"
echo "  Registry:  ${REGISTRY:-'(local)'}"
echo "  Push:      $PUSH_IMAGES"

# Build dashboard image
if [[ "$BUILD_DASHBOARD" == "true" ]]; then
    print_header "Building Dashboard Image"
    
    IMAGE_NAME="${REGISTRY}quant-dashboard:${IMAGE_TAG}"
    print_status "Building $IMAGE_NAME..."
    
    docker build \
        -f Dockerfile.dashboard \
        -t "$IMAGE_NAME" \
        --build-arg BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ') \
        --build-arg VERSION="$IMAGE_TAG" \
        .. || {
        print_error "Failed to build dashboard image"
        exit 1
    }
    
    print_status "Dashboard image built successfully: $IMAGE_NAME"
    
    if [[ "$PUSH_IMAGES" == "true" ]]; then
        print_status "Pushing dashboard image..."
        docker push "$IMAGE_NAME"
    fi
fi

# Build CPU LLM image
if [[ "$BUILD_CPU" == "true" ]]; then
    print_header "Building CPU LLM Image"
    
    IMAGE_NAME="${REGISTRY}quant-llm-cpu:${IMAGE_TAG}"
    print_status "Building $IMAGE_NAME..."
    
    docker build \
        -f Dockerfile.llm-cpu \
        -t "$IMAGE_NAME" \
        --build-arg BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ') \
        --build-arg VERSION="$IMAGE_TAG" \
        .. || {
        print_error "Failed to build CPU LLM image"
        exit 1
    }
    
    print_status "CPU LLM image built successfully: $IMAGE_NAME"
    
    if [[ "$PUSH_IMAGES" == "true" ]]; then
        print_status "Pushing CPU LLM image..."
        docker push "$IMAGE_NAME"
    fi
fi

# Build GPU LLM image
if [[ "$BUILD_GPU" == "true" ]]; then
    print_header "Building GPU LLM Image"
    
    # Check if NVIDIA Docker is available
    if ! docker info | grep -q nvidia; then
        print_warning "NVIDIA Docker runtime not detected. GPU image may not work properly."
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_status "Skipping GPU image build"
        else
            BUILD_GPU_CONFIRMED=true
        fi
    else
        BUILD_GPU_CONFIRMED=true
    fi
    
    if [[ "$BUILD_GPU_CONFIRMED" == "true" ]]; then
        IMAGE_NAME="${REGISTRY}quant-llm-gpu:${IMAGE_TAG}"
        print_status "Building $IMAGE_NAME..."
        
        docker build \
            -f Dockerfile.llm-gpu \
            -t "$IMAGE_NAME" \
            --build-arg BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ') \
            --build-arg VERSION="$IMAGE_TAG" \
            .. || {
            print_error "Failed to build GPU LLM image"
            exit 1
        }
        
        print_status "GPU LLM image built successfully: $IMAGE_NAME"
        
        if [[ "$PUSH_IMAGES" == "true" ]]; then
            print_status "Pushing GPU LLM image..."
            docker push "$IMAGE_NAME"
        fi
    fi
fi

print_header "BUILD COMPLETE"

# Show built images
print_status "Built images:"
docker images | grep quant | grep "$IMAGE_TAG"

print_status "Usage examples:"
if [[ "$BUILD_DASHBOARD" == "true" ]]; then
    echo "  Dashboard:  docker run -v \$(pwd)/results:/app/results -p 6001:6001 ${REGISTRY}quant-dashboard:${IMAGE_TAG}"
fi
if [[ "$BUILD_CPU" == "true" ]]; then
    echo "  CPU LLM:    docker run -v \$(pwd)/models:/app/models -p 6001:6001 ${REGISTRY}quant-llm-cpu:${IMAGE_TAG}"
fi
if [[ "$BUILD_GPU" == "true" ]]; then
    echo "  GPU LLM:    docker run --gpus all -v \$(pwd)/models:/app/models -p 6002:6001 ${REGISTRY}quant-llm-gpu:${IMAGE_TAG}"
fi

print_status "Or use docker-compose:"
echo "  cd docker && docker-compose up"