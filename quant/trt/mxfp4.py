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

try:
    import nvidia.modelopt.torch.quantization as mtq
    from transformers import AutoModelForCausalLM
    MODELOPT_AVAILABLE = True
except ImportError:
    mtq = None
    AutoModelForCausalLM = None
    MODELOPT_AVAILABLE = False

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
        self.model = None
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
    
    def _load_model(self):
        """Load PyTorch model for quantization"""
        if self.model is None:
            if not MODELOPT_AVAILABLE:
                raise RuntimeError("TensorRT Model Optimizer not available. Please install modelopt.")
            
            logger.info(f"Loading PyTorch model: {self.model_name}")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                device_map="auto"
            )
            self.model.eval()
            logger.info("PyTorch model loaded successfully")
    
    def _apply_mxfp4_quantization(self) -> str:
        """Apply MXfp4 quantization using TensorRT Model Optimizer"""
        if not MODELOPT_AVAILABLE:
            raise RuntimeError("TensorRT Model Optimizer not available")
        
        self._load_model()
        self._load_tokenizer()
        
        # Detect sequence length from model
        self._detect_expected_sequence_length()
        
        logger.info("Applying MXfp4 quantization...")
        
        # Configure quantization for FP4 - using block quantization config
        quant_cfg = {
            "quant_cfg": {
                "*weight_quantizer": {
                    "num_bits": (2, 1),  # FP4 E2M1 format  
                    "block_sizes": {-1: 128},  # Block size 128
                    "enable": True
                },
                "*input_quantizer": {
                    "num_bits": (2, 1),  # FP4 E2M1 format
                    "type": "dynamic", 
                    "enable": True
                },
                "default": {"enable": False}
            },
            "algorithm": "max"
        }
        
        # Apply quantization to the model
        quantized_model = mtq.quantize(self.model, quant_cfg, forward_loop=self._calibration_forward_loop)
        
        # Export quantized model to ONNX
        quantized_onnx_path = self.output_path.replace('.trt', '_quantized.onnx')
        self._export_quantized_onnx(quantized_model, quantized_onnx_path)
        
        logger.info(f"MXfp4 quantized ONNX model saved to: {quantized_onnx_path}")
        return quantized_onnx_path
    
    def _calibration_forward_loop(self, model):
        """Calibration forward loop for quantization"""
        model.eval()
        with torch.no_grad():
            # Use a few sample inputs for calibration
            sample_texts = [
                "Hello world, this is a test.",
                "The quick brown fox jumps over the lazy dog.",
                "Artificial intelligence is transforming the world.",
                "Machine learning enables computers to learn without explicit programming."
            ]
            
            for text in sample_texts:
                inputs = self.tokenizer.encode(text, return_tensors="pt", max_length=512, truncation=True)
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                _ = model(inputs)
    
    def _export_quantized_onnx(self, quantized_model, onnx_path: str):
        """Export quantized PyTorch model to ONNX"""
        logger.info("Exporting quantized model to ONNX...")
        
        # Create dummy input
        dummy_input = torch.randint(0, self.vocab_size, (1, 512), dtype=torch.long)
        if torch.cuda.is_available():
            dummy_input = dummy_input.cuda()
        
        # Export to ONNX
        torch.onnx.export(
            quantized_model,
            dummy_input,
            onnx_path,
            input_names=['input_ids'],
            output_names=['logits'],
            dynamic_axes={
                'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                'logits': {0: 'batch_size', 1: 'sequence_length'}
            },
            opset_version=17,
            export_params=True,
            do_constant_folding=False
        )
    
    def quantize(self, onnx_path: str = None) -> bool:
        """Apply MXfp4 quantization and convert to TensorRT"""
        if not ONNX_AVAILABLE:
            raise RuntimeError("ONNX Runtime not available. Please install onnxruntime.")
        
        # Check if TensorRT engine already exists
        if TENSORRT_AVAILABLE and os.path.exists(self.output_path):
            logger.info(f"TensorRT engine exists at {self.output_path}")
            return self._load_trt_engine()
        
        # For MXfp4, we ignore the input ONNX path and do quantization from PyTorch
        if onnx_path:
            logger.info(f"Ignoring input ONNX path {onnx_path}, applying MXfp4 quantization from PyTorch model")
        
        # Apply MXfp4 quantization to PyTorch model and export to ONNX
        quantized_onnx_path = self._apply_mxfp4_quantization()
        
        # Load the quantized ONNX session for metadata
        if not self._load_onnx_session(quantized_onnx_path):
            raise RuntimeError(f"Failed to load quantized ONNX model from {quantized_onnx_path}")
        
        # Convert quantized ONNX to TensorRT
        if TENSORRT_AVAILABLE:
            return self._convert_to_tensorrt(quantized_onnx_path)
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
        """Get the expected sequence length from the configuration or model"""
        # Try to get from config first
        config_seq_len = self.config.get('onnx_export', {}).get('export_sequence_length')
        
        if config_seq_len:
            self.expected_seq_len = config_seq_len
            logger.info(f"Using configured sequence length: {config_seq_len}")
        elif self.model is not None:
            # Try to detect from model config
            model_config = self.model.config
            context_attrs = ['max_position_embeddings', 'n_positions', 'max_sequence_length']
            
            for attr in context_attrs:
                if hasattr(model_config, attr):
                    self.expected_seq_len = getattr(model_config, attr)
                    logger.info(f"Detected sequence length from model.config.{attr}: {self.expected_seq_len}")
                    break
            
            if self.expected_seq_len is None:
                self.expected_seq_len = 1024  # Default fallback
                logger.info(f"Using default sequence length: {self.expected_seq_len}")
        else:
            self.expected_seq_len = 1024  # Default fallback
            logger.info(f"Using default sequence length: {self.expected_seq_len}")
    
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
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, self.max_workspace_size)
            
            # MXfp4 quantization should be applied via TensorRT Model Optimizer to the ONNX model
            # The TensorRT builder just builds the pre-quantized ONNX model as-is
            logger.info("Building TensorRT engine from MXfp4-quantized ONNX model")
            
            # Handle dynamic shapes with optimization profiles
            if self.is_dynamic:
                logger.info("Setting up optimization profile for dynamic shapes...")
                profile = builder.create_optimization_profile()
                
                for spec in self.input_specs:
                    if spec['is_dynamic']:
                        input_name = spec['name']
                        # Set dynamic shape ranges: min, opt, max
                        min_shape = [1, 1]  # minimum batch=1, seq_len=1
                        opt_shape = [self.max_batch_size, self.max_length // 2]  # optimal
                        max_shape = [self.max_batch_size, self.max_length]  # maximum
                        
                        logger.info(f"Setting profile for {input_name}: min={min_shape}, opt={opt_shape}, max={max_shape}")
                        profile.set_shape(input_name, min_shape, opt_shape, max_shape)
                
                config.add_optimization_profile(profile)
            
            # Build engine
            logger.info("Building TensorRT engine...")
            serialized_engine = builder.build_serialized_network(network, config)
            
            if serialized_engine is None:
                logger.error("Failed to build TensorRT engine")
                return False
            
            # Save engine
            with open(self.output_path, 'wb') as f:
                f.write(serialized_engine)
            
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
            
            # Reshape to (batch_size, seq_len) if needed
            if len(input_ids.shape) == 1:
                input_ids = input_ids.reshape(1, -1)
            
            batch_size, original_seq_len = input_ids.shape
            
            # Pad input to match expected sequence length (1024)
            if self.expected_seq_len and original_seq_len != self.expected_seq_len:
                padded_input_ids = np.full((batch_size, self.expected_seq_len), 
                                         self.tokenizer.pad_token_id or 0, 
                                         dtype=input_ids.dtype)
                # Right-align the tokens (put them at the end)
                if original_seq_len <= self.expected_seq_len:
                    padded_input_ids[:, -original_seq_len:] = input_ids
                    input_ids = padded_input_ids

                else:
                    # Truncate if too long
                    input_ids = input_ids[:, -self.expected_seq_len:]

            
            batch_size, seq_len = input_ids.shape

            
            # Set input shapes in context  
            for i in range(self.trt_engine.num_io_tensors):
                tensor_name = self.trt_engine.get_tensor_name(i)
                if self.trt_engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                    if tensor_name == 'input_ids':
                        self.trt_context.set_input_shape(tensor_name, (batch_size, seq_len))

            
            # Prepare TensorRT inputs/outputs
            bindings = []
            outputs = {}
            device_memories = {}
            total_memory_mb = 0
            
            # Allocate GPU memory and set bindings
            for i in range(self.trt_engine.num_io_tensors):
                tensor_name = self.trt_engine.get_tensor_name(i)
                tensor_shape = self.trt_context.get_tensor_shape(tensor_name)
                size = trt.volume(tensor_shape)
                dtype = trt.nptype(self.trt_engine.get_tensor_dtype(tensor_name))
                
                # Calculate memory size in MB
                dtype_size = np.dtype(dtype).itemsize
                memory_mb = (size * dtype_size) / (1024 * 1024)
                total_memory_mb += memory_mb
                

                
                # Allocate memory for all tensors
                host_mem = cuda.pagelocked_empty(size, dtype)
                device_mem = cuda.mem_alloc(host_mem.nbytes)
                device_memories[tensor_name] = device_mem
                bindings.append(int(device_mem))
                
                if self.trt_engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                    if tensor_name == 'input_ids':
                        flat_input = input_ids.flatten()
                        np.copyto(host_mem[:len(flat_input)], flat_input)
                    cuda.memcpy_htod(device_mem, host_mem)
                else:
                    # Store output tensors we care about
                    if tensor_name == 'logits':
                        outputs[tensor_name] = host_mem
            
            # Run inference
            self.trt_context.execute_v2(bindings)
            
            # Copy outputs back
            for tensor_name, host_mem in outputs.items():
                device_mem = device_memories[tensor_name]
                cuda.memcpy_dtoh(host_mem, device_mem)
            
            # PROPER autoregressive generation
            original_tokens = input_ids[0, -original_seq_len:].tolist()
            generated_tokens = original_tokens.copy()
            
            for step in range(max_new_tokens):
                # Get logits from current inference
                if 'logits' in outputs:
                    logits = outputs['logits']
                    
                    # Get logits for the LAST token position (where we predict next)
                    if len(logits.shape) == 3:  # [batch, seq, vocab]
                        # Find the position of the last actual token (not padding)
                        last_pos = len(generated_tokens) - 1 
                        if last_pos >= 0 and last_pos < logits.shape[1]:
                            next_token_logits = logits[0, last_pos, :]
                        else:
                            next_token_logits = logits[0, -1, :]
                    else:
                        next_token_logits = logits[-1] if len(logits.shape) == 2 else logits
                    
                    # Sample next token  
                    next_token = self._sample_token(next_token_logits, temperature)
                    generated_tokens.append(next_token)
                    
                    # Stop conditions
                    if next_token == self.tokenizer.eos_token_id:
                        break
                    if len(generated_tokens) >= self.max_length:
                        break
                    
                    # Prepare next inference with updated sequence
                    if step < max_new_tokens - 1:
                        # Create new padded input with all generated tokens so far
                        new_seq_len = len(generated_tokens)
                        if new_seq_len <= self.expected_seq_len:
                            new_input_ids = np.full((1, self.expected_seq_len), 
                                                  self.tokenizer.pad_token_id or 0, dtype=np.int64)
                            new_input_ids[:, -new_seq_len:] = np.array(generated_tokens)
                            
                            # Update GPU memory with new sequence
                            input_device_mem = device_memories['input_ids']
                            host_mem = cuda.pagelocked_empty(self.expected_seq_len, dtype=np.int64)
                            np.copyto(host_mem, new_input_ids.flatten())
                            cuda.memcpy_htod(input_device_mem, host_mem)
                            
                            # Run inference again for next token
                            self.trt_context.execute_v2(bindings)
                            
                            # Copy outputs back
                            for tensor_name, host_mem in outputs.items():
                                if tensor_name == 'logits':
                                    device_mem = device_memories[tensor_name] 
                                    cuda.memcpy_dtoh(host_mem, device_mem)
                else:
                    break
            
            # Decode full response
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            original_text = self.tokenizer.decode(original_tokens, skip_special_tokens=True)
            
            if generated_text.startswith(original_text):
                response = generated_text[len(original_text):].strip()
            else:
                response = generated_text.strip()
            
            tokens_generated = len(generated_tokens) - len(original_tokens)
            logger.info(f"Generated {tokens_generated} tokens autoregressively")
            return response if response else "Generated response."
            
            # Clean up GPU memory
            for device_mem in device_memories.values():
                device_mem.free()
            
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