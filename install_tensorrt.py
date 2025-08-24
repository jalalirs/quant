#!/usr/bin/env python3
"""
Smart TensorRT installation script that detects CUDA version and installs appropriate TensorRT
"""

import subprocess
import sys
import os
import logging
from packaging import version

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_cuda_available():
    """Check if CUDA is available"""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        logger.warning("PyTorch not found. Installing basic requirements first...")
        return False

def get_cuda_version():
    """Get CUDA version from nvidia-smi or torch"""
    try:
        # Try to get CUDA version from nvidia-smi
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'CUDA Version:' in line:
                    cuda_version = line.split('CUDA Version:')[1].strip().split()[0]
                    return cuda_version
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    try:
        # Fallback to torch CUDA version
        import torch
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            return cuda_version
    except ImportError:
        pass
    
    return None

def get_tensorrt_package(cuda_version):
    """Determine the correct TensorRT package based on CUDA version"""
    if not cuda_version:
        return None
    
    try:
        cuda_ver = version.parse(cuda_version)
        
        if cuda_ver >= version.parse("12.0"):
            return "tensorrt-cu12"
        elif cuda_ver >= version.parse("11.0"):
            return "tensorrt-cu11"
        else:
            logger.warning(f"CUDA version {cuda_version} may not be supported by recent TensorRT versions")
            return "tensorrt"  # Generic package
            
    except Exception as e:
        logger.error(f"Error parsing CUDA version {cuda_version}: {e}")
        return "tensorrt"

def install_package(package_name):
    """Install a package using pip"""
    try:
        logger.info(f"Installing {package_name}...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', package_name], 
                      check=True, capture_output=True, text=True)
        logger.info(f"Successfully installed {package_name}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install {package_name}: {e}")
        logger.error(f"stdout: {e.stdout}")
        logger.error(f"stderr: {e.stderr}")
        return False

def test_tensorrt_installation():
    """Test if TensorRT is properly installed"""
    try:
        import tensorrt as trt
        logger.info(f"TensorRT version: {trt.__version__}")
        
        # Test basic TensorRT functionality
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(TRT_LOGGER)
        logger.info("TensorRT installation verified successfully!")
        return True
        
    except ImportError as e:
        logger.error(f"TensorRT import failed: {e}")
        return False
    except Exception as e:
        logger.error(f"TensorRT test failed: {e}")
        return False

def main():
    logger.info("=== TensorRT Installation Script ===")
    
    # Check if CUDA is available
    if not check_cuda_available():
        logger.warning("CUDA is not available. TensorRT requires CUDA for GPU acceleration.")
        logger.info("You can still install TensorRT for CPU inference, but performance will be limited.")
        
        choice = input("Do you want to install TensorRT anyway? (y/N): ").lower().strip()
        if choice not in ['y', 'yes']:
            logger.info("Skipping TensorRT installation.")
            return 0
    
    # Get CUDA version
    cuda_version = get_cuda_version()
    if cuda_version:
        logger.info(f"Detected CUDA version: {cuda_version}")
    else:
        logger.warning("Could not detect CUDA version.")
    
    # Determine TensorRT package
    tensorrt_package = get_tensorrt_package(cuda_version)
    
    if tensorrt_package:
        logger.info(f"Installing TensorRT package: {tensorrt_package}")
        
        # Try installing the specific package
        if install_package(tensorrt_package):
            if test_tensorrt_installation():
                logger.info("✅ TensorRT installation completed successfully!")
                return 0
            else:
                logger.warning("TensorRT installed but verification failed.")
        else:
            logger.warning(f"Failed to install {tensorrt_package}, trying generic tensorrt package...")
            if install_package("tensorrt"):
                if test_tensorrt_installation():
                    logger.info("✅ TensorRT installation completed successfully!")
                    return 0
    else:
        logger.error("Could not determine appropriate TensorRT package.")
        logger.info("Please install TensorRT manually:")
        logger.info("  For CUDA 11.x: pip install tensorrt-cu11")
        logger.info("  For CUDA 12.x: pip install tensorrt-cu12")
        logger.info("  Generic:       pip install tensorrt")
        return 1
    
    logger.error("❌ TensorRT installation failed!")
    logger.info("\nTroubleshooting:")
    logger.info("1. Make sure CUDA is properly installed")
    logger.info("2. Check NVIDIA driver compatibility") 
    logger.info("3. Try manual installation with specific version:")
    logger.info("   pip install tensorrt-cu11==8.6.1 (for CUDA 11.x)")
    logger.info("   pip install tensorrt-cu12==8.6.1 (for CUDA 12.x)")
    
    return 1

if __name__ == "__main__":
    exit(main())