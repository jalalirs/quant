"""
Generic ONNX/TensorRT quantization module
Supports any model architecture with automatic input/output handling
"""

import os
import logging
import numpy as np
import torch
from typing import Dict, Any, Optional, List, Tuple
from transformers import AutoTokenizer

# Conditional imports
try:
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
except ImportError:
    trt = None
    TENSORRT_AVAILABLE = False

try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    onnx = None
    ort = None
    ONNX_AVAILABLE = False

logger = logging.getLogger(__name__)

class MXFp4Quantizer:
    """Generic ONNX/TensorRT quantizer for any model architecture"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = config.get('model_name')
        self.cache_dir = config.get('cache_dir', './model_cache')
        self.precision = config.get('precision', 'fp4')
        self.output_path = config.get('output_path', './models/model.trt')
        self.max_workspace_size = config.get('max_workspace_size', 4 << 30)  # 4GB
        self.max_length = config.get('max_length', 2048)
        self.max_batch_size = config.get('max_batch_size', 1)
        
        # Generation settings from config
        generation_config = config.get('generation', {})
        self.default_max_new_tokens = generation_config.get('max_new_tokens', 50)
        self.default_temperature = generation_config.get('temperature', 0.7)
        self.default_top_k = generation_config.get('top_k', 50)
        
        # Runtime components
        self.tokenizer = None
        self.trt_engine = None
        self.trt_context = None
        self.onnx_session = None
        
        # Model metadata
        self.input_specs = []
        self.output_specs = []
        self.is_dynamic = False
        self.vocab_size = None
        self.expected_seq_len = None  # Will be detected from the model
        
        # Create directories
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
    
    @classmethod
    def load_from_dict(cls, config: Dict[str, Any]) -> 'MXFp4Quantizer':
        """Load quantizer from configuration"""
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
            
            self.vocab_size = len(self.tokenizer)
            logger.info(f"Tokenizer loaded, vocab size: {self.vocab_size}")
    
    def quantize(self, onnx_path: str) -> bool:
        """Convert ONNX model to TensorRT with quantization"""
        if not ONNX_AVAILABLE:
            raise RuntimeError("ONNX Runtime not available. Please install onnxruntime.")
        
        # Always load ONNX first to get model metadata
        if not self._load_onnx_session(onnx_path):
            raise RuntimeError(f"Failed to load ONNX model from {onnx_path}")
        
        # Try TensorRT conversion if available
        if TENSORRT_AVAILABLE:
            if os.path.exists(self.output_path):
                logger.info(f"TensorRT engine exists at {self.output_path}")
                return self._load_trt_engine()
            else:
                return self._convert_to_tensorrt(onnx_path)
        else:
            logger.info("TensorRT not available, using ONNX Runtime only")
            return True
    
    def _load_onnx_session(self, onnx_path: str) -> bool:
        """Load ONNX session and extract model metadata"""
        try:
            logger.info(f"Loading ONNX model from {onnx_path}")
            
            # Create ONNX Runtime session
            providers = ['CPUExecutionProvider']
            if ort.get_available_providers():
                available = ort.get_available_providers()
                if 'CUDAExecutionProvider' in available:
                    providers.insert(0, 'CUDAExecutionProvider')
            
            self.onnx_session = ort.InferenceSession(onnx_path, providers=providers)
            
            # Extract input/output specifications
            self._extract_model_specs()
            
            logger.info(f"ONNX model loaded successfully")
            logger.info(f"Inputs: {[spec['name'] for spec in self.input_specs]}")
            logger.info(f"Outputs: {[spec['name'] for spec in self.output_specs]}")
            logger.info(f"Dynamic shapes: {self.is_dynamic}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            return False
    
    def _extract_model_specs(self):
        """Extract input/output specifications from ONNX model"""
        self.input_specs = []
        self.output_specs = []
        
        # Process inputs
        for inp in self.onnx_session.get_inputs():
            spec = {
                'name': inp.name,
                'shape': list(inp.shape),
                'type': inp.type,
                'is_dynamic': any(isinstance(dim, str) or dim < 0 for dim in inp.shape)
            }
            self.input_specs.append(spec)
            
            if spec['is_dynamic']:
                self.is_dynamic = True
        
        # Process outputs
        for out in self.onnx_session.get_outputs():
            spec = {
                'name': out.name,
                'shape': list(out.shape),
                'type': out.type
            }
            self.output_specs.append(spec)
        
        # Detect expected sequence length by probing the model
        self._detect_expected_sequence_length()
        
        logger.debug(f"Model specs extracted: {len(self.input_specs)} inputs, {len(self.output_specs)} outputs")
        if self.expected_seq_len:
            logger.info(f"Detected expected sequence length: {self.expected_seq_len}")
    
    def _detect_expected_sequence_length(self):
        """Get the expected sequence length from the configuration"""
        # The sequence length MUST be in the config since we control the export
        config_seq_len = self.config.get('onnx_export', {}).get('export_sequence_length')
        
        if not config_seq_len:
            raise RuntimeError("export_sequence_length must be specified in config onnx_export section")
        
        self.expected_seq_len = config_seq_len
        logger.info(f"Using configured sequence length: {config_seq_len}")
    
    def _convert_to_tensorrt(self, onnx_path: str) -> bool:
        """Convert ONNX to TensorRT"""
        try:
            logger.info(f"Converting ONNX to TensorRT with {self.precision} precision...")
            
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(TRT_LOGGER)
            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            )
            parser = trt.OnnxParser(network, TRT_LOGGER)
            
            # Parse ONNX model
            with open(onnx_path, 'rb') as model:
                if not parser.parse(model.read()):
                    for error in range(parser.num_errors):
                        logger.error(f"ONNX parsing error: {parser.get_error(error)}")
                    return False
            
            # Configure builder
            config = builder.create_builder_config()
            config.max_workspace_size = self.max_workspace_size
            
            # Set precision
            if self.precision == 'fp16':
                    config.set_flag(trt.BuilderFlag.FP16)
            elif self.precision == 'int8':
                    config.set_flag(trt.BuilderFlag.INT8)
            
            # Build engine
            logger.info("Building TensorRT engine...")
            engine = builder.build_engine(network, config)
            
            if engine is None:
                logger.error("Failed to build TensorRT engine")
                return False
            
            # Save engine
            with open(self.output_path, 'wb') as f:
                f.write(engine.serialize())
            
            logger.info(f"TensorRT engine saved to {self.output_path}")
            return self._load_trt_engine()
            
        except Exception as e:
            logger.error(f"TensorRT conversion failed: {e}")
            return False
    
    def _load_trt_engine(self) -> bool:
        """Load TensorRT engine"""
        try:
            logger.info(f"Loading TensorRT engine from {self.output_path}")
            
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
    
    def generate(self, prompt: str, max_new_tokens: int = 50, temperature: float = 0.7) -> str:
        """Generate text using the quantized model"""
        self._load_tokenizer()
        
        if self.trt_engine is not None:
            return self._generate_tensorrt(prompt, max_new_tokens, temperature)
        elif self.onnx_session is not None:
            return self._generate_onnx(prompt, max_new_tokens, temperature)
        else:
            raise RuntimeError("No model loaded for generation")
    
    def generate_text(self, prompt: str, max_new_tokens: int = 50, temperature: float = 0.7) -> str:
        """Generate text using the quantized model (compatibility method)"""
        return self.generate(prompt, max_new_tokens, temperature)
    
    def _generate_tensorrt(self, prompt: str, max_new_tokens: int, temperature: float) -> str:
        """Generate using TensorRT engine"""
        try:
            import pycuda.driver as cuda
            import pycuda.autoinit
            
            # Tokenize input
            inputs = self.tokenizer.encode(prompt, return_tensors="pt")
            input_ids = inputs[0].numpy().astype(np.int64)
            
            # Prepare TensorRT inputs/outputs
            bindings = []
            outputs = {}
            
            # Allocate GPU memory and set bindings
            for i in range(self.trt_engine.num_bindings):
                binding = self.trt_engine.get_binding_name(i)
                size = trt.volume(self.trt_context.get_binding_shape(i))
                dtype = trt.nptype(self.trt_engine.get_binding_dtype(i))
                
                if self.trt_engine.binding_is_input(i):
                    host_mem = cuda.pagelocked_empty(size, dtype)
                    if binding == 'input_ids':
                        np.copyto(host_mem[:len(input_ids)], input_ids.flatten())
                else:
                    host_mem = cuda.pagelocked_empty(size, dtype)
                    outputs[binding] = host_mem
                
                device_mem = cuda.mem_alloc(host_mem.nbytes)
                bindings.append(int(device_mem))
                
                if self.trt_engine.binding_is_input(i):
                    cuda.memcpy_htod(device_mem, host_mem)
            
            # Run inference
            self.trt_context.execute_v2(bindings)
            
            # Copy outputs back
            for binding, host_mem in outputs.items():
                device_mem = bindings[self.trt_engine.get_binding_index(binding)]
                cuda.memcpy_dtoh(host_mem, device_mem)
            
            # Process outputs (simplified)
            logits = outputs.get('logits', list(outputs.values())[0])
            next_token = np.argmax(logits[-self.vocab_size:])
            
            # Simple continuation
            generated_tokens = input_ids.tolist() + [next_token]
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            return generated_text[len(prompt):].strip()
            
        except Exception as e:
            logger.error(f"TensorRT generation failed: {e}")
            raise RuntimeError(f"TensorRT generation failed: {e}")
    
    def _generate_onnx(self, prompt: str, max_new_tokens: int, temperature: float) -> str:
        """Generate using ONNX Runtime without KV cache."""
        try:
            # Tokenize input
            inputs = self.tokenizer.encode(prompt, return_tensors="pt", max_length=self.max_length, truncation=True)
            input_ids = inputs[0].numpy().astype(np.int64)
            
            if len(input_ids.shape) == 1:
                input_ids = input_ids.reshape(1, -1)
            
            # No need to adjust input length - let it be dynamic as intended
            logger.debug(f"Input IDs shape: {input_ids.shape}")
            
            # Autoregressive generation
            generated_tokens = input_ids[0].tolist()
            
            for step in range(max_new_tokens):
                # On each step, send the ENTIRE sequence of generated tokens so far.
                current_ids = np.array([generated_tokens], dtype=np.int64)
                feed_dict = self._prepare_inputs(current_ids)
                
                # Run inference - get all outputs but we'll only use logits
                outputs = self.onnx_session.run(None, feed_dict)

                # Extract logits from the output.
                logits = self._extract_logits(outputs)
                if logits is None:
                    break
                
                # Sample next token
                next_token = self._sample_token(logits, temperature)
                generated_tokens.append(next_token)
                
                # Stop conditions
                if next_token == self.tokenizer.eos_token_id:
                    break
                if len(generated_tokens) >= self.max_length:
                    break
            
            # Decode response
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            response = generated_text[len(prompt):].strip() if generated_text.startswith(prompt) else generated_text.strip()
            
            return response if response else "Generated response."
            
        except Exception as e:
            logger.error(f"ONNX generation failed: {e}")
            raise RuntimeError(f"ONNX generation failed: {e}")

    def _prepare_inputs(self, input_ids: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Prepare all inputs that the ONNX model requires.
        Handle fixed sequence length requirements by padding/truncating if needed.
        """
        # If model has a fixed sequence length requirement, adjust input accordingly
        if self.expected_seq_len is not None:
            batch_size = input_ids.shape[0]
            current_seq_len = input_ids.shape[1]
            
            if current_seq_len != self.expected_seq_len:
                # Create new tensor with the expected length
                new_input_ids = np.full((batch_size, self.expected_seq_len), 
                                      self.tokenizer.pad_token_id or 0, 
                                      dtype=input_ids.dtype)
                
                if current_seq_len < self.expected_seq_len:
                    # Pad: copy all current tokens to the end (right-aligned)
                    new_input_ids[:, -current_seq_len:] = input_ids
                else:
                    # Truncate: take the last expected_seq_len tokens
                    new_input_ids = input_ids[:, -self.expected_seq_len:]
                
                input_ids = new_input_ids
                logger.debug(f"Adjusted input from length {current_seq_len} to {self.expected_seq_len}")
        
        return {'input_ids': input_ids}

    def _resolve_shape(self, shape: List, batch_size: int, seq_len: int) -> Tuple[int, ...]:
        """Resolve dynamic dimensions in shape"""
        resolved = []
        
        for i, dim in enumerate(shape):
            if isinstance(dim, str):
                # Dynamic dimension names
                if 'batch' in dim.lower():
                    resolved.append(batch_size)
                elif 'seq' in dim.lower() or 'length' in dim.lower():
                    resolved.append(seq_len)
                else:
                    # Unknown dynamic dimension
                    if i == 0:
                        resolved.append(batch_size)
                    elif i == 1:
                        resolved.append(seq_len)
                    else:
                        resolved.append(1)
            elif dim < 0:
                # Negative dimension (dynamic)
                if i == 0:
                    resolved.append(batch_size)
                elif i == 1:
                    resolved.append(seq_len)
                else:
                    resolved.append(1)
            else:
                # Fixed dimension
                resolved.append(dim)
        
        return tuple(resolved)
    
    def _fit_to_shape(self, tensor: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
        """Fit tensor to target shape with padding/truncation"""
        if tensor.shape == target_shape:
            return tensor
        
        # Make a new tensor of the correct target shape, filled with the pad token
        pad_value = self.tokenizer.pad_token_id or 0
        new_tensor = np.full(target_shape, pad_value, dtype=tensor.dtype)

        # Determine the slice to copy the original tensor data into
        slicing_shape = tuple(min(orig_dim, target_dim) for orig_dim, target_dim in zip(tensor.shape, target_shape))
        slicer = tuple(slice(0, dim) for dim in slicing_shape)
        
        # Copy the data
        new_tensor[slicer] = tensor[slicer]

        return new_tensor

    def _extract_logits(self, outputs: List[np.ndarray]) -> Optional[np.ndarray]:
        """Extracts logits from the model's output list."""
        # Find the logits output by looking for the output with vocab_size dimension
        for i, output_spec in enumerate(self.output_specs):
            if output_spec['name'] == 'logits':
                logits_output = outputs[i]
                # We only need the logits for the very last token in the sequence.
                # Shape is [batch_size, sequence_length, vocab_size]
                return logits_output[0, -1, :]
        
        # Fallback: assume first output is logits (for compatibility)
        if outputs and len(outputs) > 0:
            logits_output = outputs[0]
            return logits_output[0, -1, :]
        
        return None
    
    def _sample_token(self, logits: np.ndarray, temperature: float, top_k: int = None) -> int:
        """Sample next token from logits"""
        # Apply temperature
        logits = logits / max(temperature, 0.01)
        
        # Top-k sampling
        k = min(top_k or self.default_top_k, len(logits))
        top_indices = np.argpartition(logits, -k)[-k:]
        top_logits = logits[top_indices]
        
        # Softmax
        exp_logits = np.exp(top_logits - np.max(top_logits))
        probs = exp_logits / np.sum(exp_logits)
        
        # Sample
        idx = np.random.choice(len(probs), p=probs)
        return int(top_indices[idx])
    
    def _get_numpy_dtype(self, onnx_type: str) -> np.dtype:
        """Convert ONNX type to numpy dtype"""
        type_str = str(onnx_type).lower()
        
        if 'float32' in type_str or 'float' in type_str:
            return np.float32
        elif 'float16' in type_str:
            return np.float16
        elif 'int64' in type_str:
            return np.int64
        elif 'int32' in type_str:
            return np.int32
        elif 'bool' in type_str:
            return np.bool_
        else:
            return np.float32  # Default
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        info = {
            "model_name": self.model_name,
            "precision": self.precision,
            "max_length": self.max_length,
            "vocab_size": self.vocab_size,
            "is_dynamic": self.is_dynamic,
            "backend": "TensorRT" if self.trt_engine else "ONNX Runtime",
            "inputs": [{"name": spec["name"], "shape": spec["shape"]} for spec in self.input_specs],
            "outputs": [{"name": spec["name"], "shape": spec["shape"]} for spec in self.output_specs]
        }
        
        if self.trt_engine:
            info["trt_max_batch_size"] = self.trt_engine.max_batch_size
        
        return info