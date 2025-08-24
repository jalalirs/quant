"""
ONNX model converter
"""

import os
import logging
import torch
import onnx
from typing import Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)

class ONNXConverter:
    """Convert PyTorch models to ONNX format"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = config.get('model_name')
        self.cache_dir = config.get('cache_dir', './model_cache')
        self.output_path = config.get('output_path', './models/model.onnx')
        self.max_length = config.get('max_length', 2048)
        
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
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            logger.info("Model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def convert(self) -> bool:
        """Convert PyTorch model to ONNX"""
        if os.path.exists(self.output_path):
            logger.info(f"ONNX model already exists at {self.output_path}")
            return True
        
        # Load model if not already loaded
        if self.model is None:
            if not self._load_model():
                return False
        
        try:
            logger.info("Converting to ONNX...")
            
            # Put model in eval mode
            self.model.eval()
            
            # Prepare dummy input
            dummy_input = torch.randint(
                0, self.tokenizer.vocab_size, 
                (1, 128), 
                dtype=torch.long,
                device=next(self.model.parameters()).device
            )
            
            # Export to ONNX
            torch.onnx.export(
                self.model,
                dummy_input,
                self.output_path,
                export_params=True,
                opset_version=14,
                do_constant_folding=True,
                input_names=['input_ids'],
                output_names=['logits'],
                dynamic_axes={
                    'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                    'logits': {0: 'batch_size', 1: 'sequence_length'}
                },
                verbose=False
            )
            
            # Verify ONNX model
            onnx_model = onnx.load(self.output_path)
            onnx.checker.check_model(onnx_model)
            
            logger.info(f"ONNX conversion completed: {self.output_path}")
            return True
            
        except Exception as e:
            logger.error(f"ONNX conversion failed: {e}")
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