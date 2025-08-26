#!/usr/bin/env python3
"""
Main entry point for the quantization pipeline
"""

import yaml
import argparse
import logging
from pathlib import Path
from typing import Dict, Any

from quant import BaseQuant

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Disable noisy debug logs from HTTP libraries
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('openai._base_client').setLevel(logging.INFO)

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Quantization Pipeline")
    parser.add_argument("--config", required=True, help="Path to configuration YAML file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration
    config = load_config(args.config)
    logger.info(f"Loaded configuration from {args.config}")
    
    # Create and run quantization pipeline
    quant_pipeline = BaseQuant.load_from_dict(config)
    quant_pipeline.run()

if __name__ == "__main__":
    main()