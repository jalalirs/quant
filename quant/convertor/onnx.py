"""
ONNX model converter
"""

import os
import logging
import torch
from typing import Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2Model, GPT2LMHeadModel

# Conditional ONNX imports
try:
    import onnx
    ONNX_AVAILABLE = True
except ImportError:
    onnx = None
    ONNX_AVAILABLE = False
    logging.getLogger(__name__).warning("ONNX not available. Please install with: pip install onnx")

logger = logging.getLogger(__name__)

class ONNXConverter:
    """Convert PyTorch models to ONNX format"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = config.get('model_name')
        self.cache_dir = config.get('cache_dir', './model_cache')
        self.output_path = config.get('output_path', './models/model.onnx')
        self.max_length = config.get('max_length', 2048)
        
        # ONNX export settings from config
        onnx_export = config.get('onnx_export', {})
        self.opset_version = onnx_export.get('opset_version', 14)
        self.use_dynamic_axes = onnx_export.get('use_dynamic_axes', True)
        self.input_names = onnx_export.get('input_names', ['input_ids'])
        self.output_names = onnx_export.get('output_names', ['logits'])
        self.config_export_seq_len = onnx_export.get('export_sequence_length')
        
        # Model-specific settings (will be detected from model config)
        self.model_max_length = None
        self.export_sequence_length = None  # Will be set from model
        self.export_batch_size = 1
        
        self.tokenizer = None
        self.model = None
        
        # Create directories
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
    
    @classmethod
    def load_from_dict(cls, config: Dict[str, Any]) -> 'ONNXConverter':
        """Load ONNX converter from configuration"""
        return cls(config)
    
    def _load_model(self) -> bool:
        """Load the original PyTorch model"""
        try:
            logger.info(f"Loading model: {self.model_name}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                trust_remote_code=True
            )
            
            # Add pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            

            # Load any AutoModelForCausalLM - should work for any model type
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                torch_dtype=torch.float32,
                trust_remote_code=True,
                device_map=None,
                use_cache=True  # Enable KV cache
            )
            
            # Detect model's maximum context length
            self._detect_model_context_length()
            
            logger.info("Model loaded successfully")
            logger.info(f"Model context length: {self.model_max_length}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def _detect_model_context_length(self):
        """Detect the model's maximum context length from its configuration"""
        try:
            config = self.model.config
            
            # Try different common attribute names for context length
            context_attrs = [
                'max_position_embeddings',
                'n_positions', 
                'max_sequence_length',
                'seq_length',
                'context_length',
                'max_length'
            ]
            
            for attr in context_attrs:
                if hasattr(config, attr):
                    self.model_max_length = getattr(config, attr)
                    logger.info(f"Detected context length from {attr}: {self.model_max_length}")
                    break
            
            # If we couldn't detect it, use tokenizer's model_max_length
            if self.model_max_length is None and hasattr(self.tokenizer, 'model_max_length'):
                if self.tokenizer.model_max_length < 1000000:  # Reasonable upper bound
                    self.model_max_length = self.tokenizer.model_max_length
                    logger.info(f"Using tokenizer max length: {self.model_max_length}")
            
            # Final fallback
            if self.model_max_length is None:
                self.model_max_length = 1024  # Common default
                logger.warning(f"Could not detect context length, using default: {self.model_max_length}")
            
            # Use the model's actual max length for export
            self.export_sequence_length = self.model_max_length
            
        except Exception as e:
            logger.warning(f"Error detecting model context length: {e}")
            self.model_max_length = 1024
            self.export_sequence_length = 1024
    
    def convert(self) -> bool:
        """Convert PyTorch model to ONNX"""
        if not ONNX_AVAILABLE:
            logger.error("ONNX not available. Please install with: pip install onnx")
            return False
            
        if os.path.exists(self.output_path):
            logger.info(f"ONNX model already exists at {self.output_path}")
            return True
        
        # Load model if not already loaded
        if self.model is None:
            if not self._load_model():
                return False
        
        try:
            logger.info("Converting to ONNX...")
            
            # Create a wrapper that handles KV cache properly
            class KVCacheModel(torch.nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model
                
                def forward(self, input_ids, past_key_values=None):
                    # Forward pass with KV cache
                    outputs = self.model(input_ids, past_key_values=past_key_values, use_cache=True)
                    
                    # Return logits and present key values
                    return outputs.logits, outputs.past_key_values
            
            # Wrap the model
            wrapped_model = KVCacheModel(self.model)
            wrapped_model.eval()
            
            # Use sequence length from config or model's max length
            export_seq_len = self.config_export_seq_len or self.model_max_length or 1024
            logger.info(f"Using sequence length {export_seq_len} for ONNX export")
            
            dummy_input = torch.randint(
                0, self.tokenizer.vocab_size, 
                (self.export_batch_size, export_seq_len),
                dtype=torch.long
            )
            
            # Get device of the first model parameter
            device = next(self.model.parameters()).device
            dummy_input = dummy_input.to(device)
            
            # Prepare export arguments from config
            export_args = {
                'model': wrapped_model,
                'args': (dummy_input,),  # Tuple for single input
                'f': self.output_path,
                'export_params': True,
                'opset_version': self.opset_version,
                'do_constant_folding': False,  # Disable constant folding for better compatibility
                'input_names': ['input_ids'],
                'output_names': ['logits', 'present_key_values'],
                'verbose': False
            }
            
            # Add dynamic axes if enabled in config
            if self.use_dynamic_axes:
                dynamic_axes = {
                    'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                    'logits': {0: 'batch_size', 1: 'sequence_length'}
                    # Skip present_key_values for now as it's complex
                }
                export_args['dynamic_axes'] = dynamic_axes
                logger.info(f"Using dynamic axes: {dynamic_axes}")
            else:
                logger.info("Using fixed axes (no dynamic shapes)")
            
            # Export to ONNX using config settings
            torch.onnx.export(**export_args)
            
            # Verify ONNX model
            onnx_model = onnx.load(self.output_path)
            onnx.checker.check_model(onnx_model)
            
            logger.info(f"ONNX conversion completed: {self.output_path}")
            
            # Log the actual input/output names for debugging
            logger.info(f"ONNX model inputs: {[input.name for input in onnx_model.graph.input]}")
            logger.info(f"ONNX model outputs: {[output.name for output in onnx_model.graph.output]}")
            
            return True
            
        except Exception as e:
            logger.error(f"ONNX conversion failed: {e}")
            # Clean up partial file
            if os.path.exists(self.output_path):
                os.remove(self.output_path)
            return False
    
    def get_tokenizer(self):
        """Get the tokenizer (load if necessary)"""
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                trust_remote_code=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer
    
    @property
    def output_path(self) -> str:
        """Get the output path for the converted model"""
        return self._output_path
    
    @output_path.setter
    def output_path(self, value: str):
        """Set the output path for the converted model"""
        self._output_path = value
        os.makedirs(os.path.dirname(value), exist_ok=True)