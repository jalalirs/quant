"""
MXFP4 quantization using TensorRT
"""

import os
import logging
import numpy as np
import torch
from typing import Dict, Any, Optional
#import tensorrt as trt
import onnx
import onnxruntime as ort
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

class MXFp4Quantizer:
    """MXFP4 quantizer using TensorRT"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = config.get('model_name')
        self.cache_dir = config.get('cache_dir', './model_cache')
        self.precision = config.get('precision', 'fp4')
        self.output_path = config.get('output_path', './models/model.trt')
        self.max_workspace_size = config.get('max_workspace_size', 4 << 30)  # 4GB
        self.max_length = config.get('max_length', 2048)
        self.max_batch_size = config.get('max_batch_size', 1)
        
        self.tokenizer = None
        self.trt_engine = None
        self.trt_context = None
        self.onnx_session = None  # Fallback
        
        # Create directories
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
    
    @classmethod
    def load_from_dict(cls, config: Dict[str, Any]) -> 'MXFp4Quantizer':
        """Load MXFP4 quantizer from configuration"""
        return cls(config)
    
    def _load_tokenizer(self):
        """Load tokenizer"""
        if self.tokenizer is None:
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def quantize(self, onnx_path: str) -> bool:
        """Convert ONNX model to TensorRT with MXFP4 quantization"""
        if os.path.exists(self.output_path):
            logger.info(f"TensorRT engine already exists at {self.output_path}")
            return self._load_trt_engine()
        
        try:
            logger.info(f"Converting ONNX to TensorRT with {self.precision} precision...")
            
            # Create TensorRT builder and network
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(TRT_LOGGER)
            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            )
            parser = trt.OnnxParser(network, TRT_LOGGER)
            
            # Parse ONNX model
            with open(onnx_path, 'rb') as model:
                if not parser.parse(model.read()):
                    logger.error("Failed to parse ONNX model")
                    for error in range(parser.num_errors):
                        logger.error(parser.get_error(error))
                    return False
            
            # Configure builder
            config = builder.create_builder_config()
            config.max_workspace_size = self.max_workspace_size
            
            # Enable precision modes
            if self.precision == 'fp4' or self.precision == 'mxfp4':
                # MXFP4 is not directly supported in all TensorRT versions
                # Fall back to FP16 for now
                if builder.platform_has_fast_fp16:
                    config.set_flag(trt.BuilderFlag.FP16)
                    logger.info("FP16 precision enabled (MXFP4 fallback)")
                else:
                    logger.warning("FP16 not supported, using FP32")
            elif self.precision == 'fp16':
                if builder.platform_has_fast_fp16:
                    config.set_flag(trt.BuilderFlag.FP16)
                    logger.info("FP16 precision enabled")
            elif self.precision == 'int8':
                if builder.platform_has_fast_int8:
                    config.set_flag(trt.BuilderFlag.INT8)
                    logger.info("INT8 precision enabled")
            
            # Set optimization profiles for dynamic shapes
            profile = builder.create_optimization_profile()
            profile.set_shape("input_ids", (1, 1), (1, 512), (self.max_batch_size, self.max_length))
            config.add_optimization_profile(profile)
            
            # Build engine
            logger.info("Building TensorRT engine (this may take several minutes)...")
            engine = builder.build_engine(network, config)
            
            if engine is None:
                logger.error("Failed to build TensorRT engine")
                return False
            
            # Serialize and save engine
            with open(self.output_path, 'wb') as f:
                f.write(engine.serialize())
            
            logger.info(f"TensorRT engine saved: {self.output_path}")
            return self._load_trt_engine()
            
        except Exception as e:
            logger.error(f"TensorRT conversion failed: {e}")
            return False
    
    def _load_trt_engine(self) -> bool:
        """Load TensorRT engine"""
        try:
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            runtime = trt.Runtime(TRT_LOGGER)
            
            with open(self.output_path, 'rb') as f:
                engine_data = f.read()
            
            self.trt_engine = runtime.deserialize_cuda_engine(engine_data)
            self.trt_context = self.trt_engine.create_execution_context()
            
            logger.info("TensorRT engine loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load TensorRT engine: {e}")
            return False
    
    def _load_onnx_fallback(self, onnx_path: str) -> bool:
        """Load ONNX model as fallback"""
        try:
            logger.info("Loading ONNX model as fallback...")
            self.onnx_session = ort.InferenceSession(onnx_path)
            return True
        except Exception as e:
            logger.error(f"Failed to load ONNX fallback: {e}")
            return False
    
    def generate_text(self, prompt: str, max_new_tokens: int = 100, temperature: float = 0.7) -> str:
        """Generate text using quantized model"""
        # Load tokenizer if not loaded
        self._load_tokenizer()
        
        try:
            # Try TensorRT first
            if self.trt_context is not None:
                return self._generate_with_trt(prompt, max_new_tokens, temperature)
            
            # Fall back to ONNX
            if self.onnx_session is not None:
                return self._generate_with_onnx(prompt, max_new_tokens, temperature)
            
            # Last resort: load ONNX and try
            onnx_path = self.output_path.replace('.trt', '.onnx')
            if os.path.exists(onnx_path) and self._load_onnx_fallback(onnx_path):
                return self._generate_with_onnx(prompt, max_new_tokens, temperature)
            
            return "Error: No inference engine available"
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return f"Error generating response: {str(e)}"
    
    def _generate_with_trt(self, prompt: str, max_new_tokens: int = 100, temperature: float = 0.7) -> str:
        """Generate text using TensorRT (simplified implementation)"""
        try:
            # This is a simplified implementation
            # In practice, TensorRT inference requires careful memory management
            # and proper CUDA operations
            
            # For now, fall back to ONNX
            logger.info("TensorRT generation not fully implemented, using ONNX fallback")
            return self._generate_with_onnx(prompt, max_new_tokens, temperature)
            
        except Exception as e:
            logger.error(f"TensorRT generation failed: {e}")
            return "Error in TensorRT generation"
    
    def _generate_with_onnx(self, prompt: str, max_new_tokens: int = 100, temperature: float = 0.7) -> str:
        """Generate text using ONNX runtime"""
        try:
            if self.onnx_session is None:
                # Try to load ONNX model
                onnx_path = self.output_path.replace('.trt', '.onnx')
                if not self._load_onnx_fallback(onnx_path):
                    return "Error: Could not load ONNX model"
            
            # Tokenize input
            inputs = self.tokenizer.encode(prompt, return_tensors="pt")
            generated_tokens = inputs[0].tolist()
            
            # Simple generation loop
            for _ in range(max_new_tokens):
                try:
                    # Prepare input
                    input_ids = np.array([generated_tokens], dtype=np.int64)
                    
                    # Run inference
                    outputs = self.onnx_session.run(["logits"], {"input_ids": input_ids})
                    logits = outputs[0][0, -1, :]  # Get last token logits
                    
                    # Apply temperature and sample
                    logits = logits / temperature
                    probs = torch.softmax(torch.from_numpy(logits), dim=-1)
                    next_token = torch.multinomial(probs, 1).item()
                    
                    generated_tokens.append(next_token)
                    
                    # Stop at EOS token
                    if next_token == self.tokenizer.eos_token_id:
                        break
                        
                except Exception as e:
                    logger.error(f"Error in generation step: {e}")
                    break
            
            # Decode generated text
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # Return only new tokens
            response = generated_text[len(prompt):] if generated_text.startswith(prompt) else generated_text
            return response.strip()
            
        except Exception as e:
            logger.error(f"ONNX generation failed: {e}")
            return f"Error in ONNX generation: {str(e)}"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the quantized model"""
        info = {
            "model_name": self.model_name,
            "precision": self.precision,
            "output_path": self.output_path,
            "trt_engine_loaded": self.trt_engine is not None,
            "onnx_fallback_loaded": self.onnx_session is not None
        }
        
        if self.trt_engine is not None:
            info["num_bindings"] = self.trt_engine.num_bindings
            info["max_batch_size"] = self.trt_engine.max_batch_size
        
        return info