"""
Quantization package for model optimization and serving
"""

import os
import logging
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
logger = logging.getLogger(__name__)

# Import all submodules
try:
    from .convertor import ONNXConverter
except ImportError:
    logger.warning("TensorRT not found, MXFP4 quantization will not be available")
try:
    from .trt import MXFp4Quantizer
except ImportError:
    logger.warning("TensorRT not found, MXFP4 quantization will not be available")
try:
    from .interface import ServerOpenAI
except ImportError:
    logger.warning("ServerOpenAI not found, server interface will not be available")
try:
    from .client import SpeedBenchmarkClient
except ImportError:
    logger.warning("SpeedBenchmarkClient not found, speed benchmark client will not be available")
try:
    from .loaders import OptimizedModelLoader
except ImportError:
    logger.warning("OptimizedModelLoader not found, optimized loading will not be available")
try:
    from .utils import GPUCompatibilityChecker
except ImportError:
    logger.warning("GPUCompatibilityChecker not found, compatibility checking will not be available")
from .dashboard import DashboardRenderer



class BaseQuant(ABC):
    """Base class for quantization pipelines"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_config = config.get('model', {})
        self.interface_config = config.get('interface', {})
        self.client_config = config.get('client', {})
        
        # Create output directories
        os.makedirs("models", exist_ok=True)
        os.makedirs("results", exist_ok=True)
    
    @classmethod
    def load_from_dict(cls, config: Dict[str, Any]) -> 'BaseQuant':
        """Load quantization pipeline from configuration dictionary"""
        # Determine pipeline type from configuration
        pipeline_type = config.get('type', 'standard')
        
        logger.info(f"Detected pipeline type: {pipeline_type}")
        
        if pipeline_type == 'dashboard':
            logger.info("Loading dashboard pipeline...")
            return QuantDashboard.load_from_dict(config)
        elif pipeline_type == 'server_benchmark':
            logger.info("Loading server benchmark pipeline...")
            return QuantServerBenchmark.load_from_dict(config)
        else:
            # Default to standard Quant pipeline
            logger.info("Loading standard quantization pipeline...")
            return Quant.load_from_dict(config)
    
    @abstractmethod
    def run(self):
        """Run the quantization pipeline"""
        pass

class Quant(BaseQuant):
    """Main quantization pipeline orchestrator"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.converter = None
        self.quantizer = None
        self.interface = None
        self.client = None
    
    @classmethod
    def load_from_dict(cls, config: Dict[str, Any]) -> 'Quant':
        """Load quantization pipeline from configuration dictionary"""
        instance = cls(config)
        
        # Initialize components based on configuration
        instance._initialize_converter()
        instance._initialize_quantizer()
        instance._initialize_interface()
        instance._initialize_client()
        
        return instance
    
    def _initialize_converter(self):
        """Initialize model converter"""
        conversion_config = self.model_config.get('conversion', {})
        converter_type = conversion_config.get('type', 'onnx')
        
        if converter_type == 'onnx':
            self.converter = ONNXConverter.load_from_dict({
                **self.model_config,
                **conversion_config
            })
        else:
            raise ValueError(f"Unknown converter type: {converter_type}")
    
    def _initialize_quantizer(self):
        """Initialize quantizer or optimized loader"""
        loader_type = self.model_config.get('loader_type', 'quantizer')
        
        if loader_type == 'optimized_loader':
            # Use the new optimized loader for direct model loading
            self.quantizer = OptimizedModelLoader.load_from_dict(self.model_config)
        else:
            # Use traditional quantizer pipeline
            conversion_config = self.model_config.get('conversion', {})
            quantization_config = conversion_config.get('quantization', {})
            quantizer_type = quantization_config.get('type', 'trt_mxfp4')
            
            if quantizer_type == 'trt_mxfp4':
                self.quantizer = MXFp4Quantizer.load_from_dict({
                    **self.model_config,
                    **conversion_config,
                    **quantization_config
                })
            else:
                raise ValueError(f"Unknown quantizer type: {quantizer_type}")
    
    def _initialize_interface(self):
        """Initialize serving interface"""
        interface_type = self.interface_config.get('type', 'server_openai')
        
        if interface_type == 'server_openai':
            self.interface = ServerOpenAI.load_from_dict({
                **self.interface_config,  # Pass interface config first
                **self.model_config,      # Then model config
                'quantizer': self.quantizer
            })
        else:
            raise ValueError(f"Unknown interface type: {interface_type}")
    
    def _initialize_client(self):
        """Initialize client"""
        client_type = self.client_config.get('type')
        
        if client_type == 'speed_benchmark':
            self.client = SpeedBenchmarkClient.load_from_dict({
                **self.client_config,
                'interface_config': self.interface_config
            })
        elif client_type is None:
            self.client = None  # No client configured
        else:
            raise ValueError(f"Unknown client type: {client_type}")
    
    def run(self):
        """Run the complete quantization pipeline"""
        logger.info("Starting quantization pipeline...")
        
        # Step 1: Convert model
        logger.info("Converting model...")
        self.converter.convert()
        
        # Step 2: Quantize model
        logger.info("Quantizing model...")
        self.quantizer.quantize(self.converter.output_path)
        
        # Step 3: Start interface (if needed)
        if self.client is not None:
            logger.info("Starting interface and running client...")
            # Run interface in background and execute client
            import asyncio
            import threading
            
            # Start server in background thread
            server_thread = threading.Thread(
                target=self.interface.start_async,
                daemon=True
            )
            server_thread.start()
            
            # Wait a bit for server to start
            import time
            time.sleep(5)
            
            # Run client
            self.client.run()
        else:
            logger.info("Starting interface...")
            self.interface.start()

class QuantDashboard(BaseQuant):
    """Quantization pipeline for dashboard generation"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.dashboard_config = config.get('dashboard', {})
        self.renderer = None
    
    @classmethod
    def load_from_dict(cls, config: Dict[str, Any]) -> 'QuantDashboard':
        """Load dashboard quantization pipeline from configuration"""
        instance = cls(config)
        instance._initialize_dashboard()
        return instance
    
    def _initialize_dashboard(self):
        """Initialize dashboard renderer"""
        self.renderer = DashboardRenderer.load_from_dict(self.config)
    
    def run(self):
        """Generate dashboard from benchmark results"""
        logger.info("Starting dashboard generation...")
        
        if self.renderer.render():
            logger.info("Dashboard generation completed successfully!")
            output_file = self.renderer.output_file
            logger.info(f"Dashboard available at: file://{os.path.abspath(output_file)}")
        else:
            logger.error("Dashboard generation failed!")

class QuantServerBenchmark(BaseQuant):
    """Specialized quantization pipeline for server benchmarking"""
    
    @classmethod
    def load_from_dict(cls, config: Dict[str, Any]) -> 'QuantServerBenchmark':
        """Load server benchmark pipeline from configuration"""
        instance = cls(config)
        # Specialized initialization for server benchmarking
        return instance
    
    def run(self):
        """Run server benchmark pipeline"""
        # Implementation for server-specific benchmarking
        pass

# Export main classes
__all__ = ['Quant', 'QuantDashboard', 'QuantServerBenchmark', 'BaseQuant']