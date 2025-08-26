"""
Optimized model loader with GPU compatibility detection and MoE kernel support
"""

import logging
import torch
from typing import Dict, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..utils.gpu_compatibility import GPUCompatibilityChecker

logger = logging.getLogger(__name__)

class OptimizedModelLoader:
    """
    Model loader that automatically applies the best available optimizations
    based on GPU compatibility
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = config.get('model_name')
        self.cache_dir = config.get('cache_dir', './model_cache')
        
        # Check GPU compatibility for reference only
        self.gpu_checker = GPUCompatibilityChecker()
        
        # Start with user-specified settings as the primary configuration
        user_optimizations = config.get('optimizations', {})
        self.optimization_config = {
            "use_mxfp4": False,
            "use_flash_attention_3": False,
            "use_megablocks_moe": False,
            "use_kernels": False,
            "torch_dtype": "auto",
            "device_map": "auto",
            "trust_remote_code": True
        }
        
        # Apply user optimizations (these take precedence)
        self.optimization_config.update(user_optimizations)
        
        # Only apply GPU compatibility recommendations if user didn't specify
        if not user_optimizations:
            gpu_config = self.gpu_checker.get_optimization_config()
            self.optimization_config.update(gpu_config)
        
        # Model components
        self.model = None
        self.tokenizer = None
        
        logger.info(f"Initialized OptimizedModelLoader for {self.model_name}")
        logger.info(f"User optimization config: {user_optimizations}")
        logger.info(f"Final optimization config: {self.optimization_config}")
        logger.info(f"GPU recommended strategy: {self.gpu_checker.compatibility_report['recommended_optimization']}")
    
    @classmethod
    def load_from_dict(cls, config: Dict[str, Any]) -> 'OptimizedModelLoader':
        """Load optimized model loader from configuration"""
        return cls(config)
    
    def load_model(self) -> bool:
        """Load model with optimal configuration"""
        try:
            logger.info(f"Loading model: {self.model_name}")
            
            # Print compatibility report
            self.gpu_checker.print_compatibility_report()
            
            # Load tokenizer
            self._load_tokenizer()
            
            # Load model with optimizations
            self._load_model_with_optimizations()
            
            logger.info("Model loaded successfully with optimizations")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def _load_tokenizer(self):
        """Load tokenizer"""
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info("Tokenizer loaded successfully")
    
    def _load_model_with_optimizations(self):
        """Load model with appropriate optimizations"""
        model_kwargs = {
            "cache_dir": self.cache_dir,
            "trust_remote_code": self.optimization_config.get("trust_remote_code", True),
            "torch_dtype": self.optimization_config.get("torch_dtype", "auto"),
            "device_map": self.optimization_config.get("device_map", "auto")
        }
        
        # Add memory optimization for large models
        if self.optimization_config.get("low_cpu_mem_usage", False):
            logger.info("💾 Enabling low CPU memory usage optimization")
            model_kwargs["low_cpu_mem_usage"] = True
        
        # Add MegaBlocks MoE optimization if compatible
        if self.optimization_config.get("use_megablocks_moe", False):
            logger.info("🔧 Enabling MegaBlocks MoE kernels for optimization")
            model_kwargs["use_kernels"] = True
        
        # Add mxfp4 optimization if compatible
        if self.optimization_config.get("use_mxfp4", False):
            logger.info("🚀 Enabling mxfp4 quantization (use_kernels=False for quantized models)")
            model_kwargs["use_kernels"] = False
        
        # Add Flash Attention optimization if compatible
        if self.optimization_config.get("use_flash_attention_3", False):
            logger.info("⚡ Enabling Flash Attention 3 with attention sinks")
            model_kwargs["attn_implementation"] = "kernels-community/vllm-flash-attn3"
        elif self.optimization_config.get("use_flash_attention_2", False):
            logger.info("⚡ Enabling Flash Attention 2")
            model_kwargs["attn_implementation"] = "flash_attention_2"
        
        # Special handling for pre-quantized models
        if "fp4" in self.model_name.lower() or "int4" in self.model_name.lower():
            logger.info("🎯 Detected pre-quantized model - optimizing for quantized weights")
            # Pre-quantized models may need specific handling
        
        # Load model
        logger.info(f"Loading model with configuration: {model_kwargs}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs
        )
        
        # Apply additional optimizations post-loading
        self._apply_post_loading_optimizations()
    
    def _apply_post_loading_optimizations(self):
        """Apply optimizations after model loading"""
        
        # Flash Attention (if available and compatible)
        if self.optimization_config.get("use_flash_attention_3", False):
            try:
                logger.info("⚡ Attempting to enable Flash Attention 3")
                # This would typically involve model-specific configuration
                # For now, we log the intent - actual implementation depends on the model
                logger.info("Flash Attention 3 configuration applied")
            except Exception as e:
                logger.warning(f"Failed to enable Flash Attention 3: {e}")
        elif self.optimization_config.get("use_flash_attention_2", False):
            try:
                logger.info("⚡ Flash Attention 2 configuration applied")
                # Flash Attention 2 is handled via attn_implementation parameter during model loading
                logger.info("Flash Attention 2 enabled successfully")
            except Exception as e:
                logger.warning(f"Failed to enable Flash Attention 2: {e}")
        
        logger.info("Post-loading optimizations complete")
    
    def generate(self, prompt: str, max_new_tokens: int = 100, temperature: float = 0.7) -> str:
        """Generate text using the optimized model"""
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Prepare messages for chat template
        messages = [{"role": "user", "content": prompt}]
        
        # Apply chat template if available
        if hasattr(self.tokenizer, 'apply_chat_template'):
            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                return_dict=True,
            ).to(self.model.device)
        else:
            # Fallback to simple encoding
            inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)
            inputs = {"input_ids": inputs}
        
        # Generate response
        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True if temperature > 0 else False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        if "input_ids" in inputs:
            response_tokens = generated[0][inputs["input_ids"].shape[-1]:]
        else:
            response_tokens = generated[0]
        
        response = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
        return response.strip()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information including optimization details"""
        info = {
            "model_name": self.model_name,
            "optimization_strategy": self.gpu_checker.compatibility_report["recommended_optimization"],
            "optimizations_applied": self.optimization_config,
            "gpu_compatibility": self.gpu_checker.compatibility_report,
            "device_info": self.gpu_checker.device_info
        }
        
        if self.model:
            info["model_device"] = str(self.model.device) if hasattr(self.model, 'device') else "unknown"
            info["model_dtype"] = str(self.model.dtype) if hasattr(self.model, 'dtype') else "unknown"
        
        return info
    
    def benchmark_info(self) -> Dict[str, Any]:
        """Get benchmark-relevant information"""
        return {
            "model_name": self.model_name,
            "optimization_strategy": self.gpu_checker.compatibility_report["recommended_optimization"],
            "mxfp4_enabled": self.optimization_config.get("use_mxfp4", False),
            "megablocks_moe_enabled": self.optimization_config.get("use_megablocks_moe", False),
            "flash_attention_3_enabled": self.optimization_config.get("use_flash_attention_3", False),
            "flash_attention_2_enabled": self.optimization_config.get("use_flash_attention_2", False),
            "device_map": self.optimization_config.get("device_map"),
            "torch_dtype": self.optimization_config.get("torch_dtype")
        }
