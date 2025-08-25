"""
GPU compatibility checker for advanced quantization and optimization features
"""

import logging
import torch
from typing import Dict, List, Optional, Tuple
from packaging import version

logger = logging.getLogger(__name__)

class GPUCompatibilityChecker:
    """Check GPU compatibility for various optimization features"""
    
    def __init__(self):
        self.device_info = self._get_device_info()
        self.compatibility_report = self._generate_compatibility_report()
    
    def _get_device_info(self) -> Dict:
        """Get detailed GPU device information"""
        if not torch.cuda.is_available():
            return {"cuda_available": False}
        
        device_info = {
            "cuda_available": True,
            "cuda_version": torch.version.cuda,
            "pytorch_version": torch.__version__,
            "device_count": torch.cuda.device_count(),
            "devices": []
        }
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            device_info["devices"].append({
                "id": i,
                "name": props.name,
                "compute_capability": f"{props.major}.{props.minor}",
                "total_memory": props.total_memory,
                "multi_processor_count": props.multi_processor_count,
                "major": props.major,
                "minor": props.minor
            })
        
        return device_info
    
    def _generate_compatibility_report(self) -> Dict:
        """Generate comprehensive compatibility report"""
        if not self.device_info.get("cuda_available"):
            return {
                "mxfp4_compatible": False,
                "flash_attention_3_compatible": False,
                "megablocks_moe_compatible": False,
                "recommended_optimization": "cpu_only",
                "warnings": ["CUDA not available - CPU-only inference recommended"]
            }
        
        report = {
            "mxfp4_compatible": False,
            "flash_attention_3_compatible": False, 
            "megablocks_moe_compatible": False,
            "recommended_optimization": "standard",
            "warnings": [],
            "device_recommendations": []
        }
        
        for device in self.device_info["devices"]:
            device_report = self._check_device_compatibility(device)
            report["device_recommendations"].append(device_report)
            
            # Update global compatibility based on best available device
            if device_report["mxfp4_compatible"]:
                report["mxfp4_compatible"] = True
            if device_report["flash_attention_3_compatible"]:
                report["flash_attention_3_compatible"] = True
            if device_report["megablocks_moe_compatible"]:
                report["megablocks_moe_compatible"] = True
        
        # Determine recommended optimization strategy
        report["recommended_optimization"] = self._determine_optimization_strategy(report)
        
        return report
    
    def _check_device_compatibility(self, device: Dict) -> Dict:
        """Check compatibility for a specific device"""
        device_report = {
            "device_id": device["id"],
            "device_name": device["name"],
            "mxfp4_compatible": False,
            "flash_attention_3_compatible": False,
            "megablocks_moe_compatible": False,
            "reasons": []
        }
        
        # Check mxfp4 compatibility (requires compute capability >= 8.0 for Ampere+)
        if device["major"] >= 8:
            device_report["mxfp4_compatible"] = True
        else:
            device_report["reasons"].append(
                f"mxfp4 requires compute capability >= 8.0, got {device['compute_capability']}"
            )
        
        # Check Flash Attention 3 compatibility (requires compute capability >= 8.0)
        if device["major"] >= 8:
            device_report["flash_attention_3_compatible"] = True
        else:
            device_report["reasons"].append(
                f"Flash Attention 3 requires compute capability >= 8.0, got {device['compute_capability']}"
            )
        
        # Check MegaBlocks MoE compatibility (more lenient, requires >= 7.0)
        if device["major"] >= 7:
            device_report["megablocks_moe_compatible"] = True
        else:
            device_report["reasons"].append(
                f"MegaBlocks MoE requires compute capability >= 7.0, got {device['compute_capability']}"
            )
        
        return device_report
    
    def _determine_optimization_strategy(self, report: Dict) -> str:
        """Determine the best optimization strategy"""
        if report["mxfp4_compatible"] and report["flash_attention_3_compatible"]:
            return "mxfp4_with_flash_attention"
        elif report["mxfp4_compatible"]:
            return "mxfp4_only"
        elif report["megablocks_moe_compatible"]:
            return "megablocks_moe"
        else:
            return "standard"
    
    def get_optimization_config(self) -> Dict:
        """Get optimization configuration based on GPU compatibility"""
        strategy = self.compatibility_report["recommended_optimization"]
        
        configs = {
            "mxfp4_with_flash_attention": {
                "use_mxfp4": True,
                "use_flash_attention_3": True,
                "use_megablocks_moe": False,
                "torch_dtype": "auto",
                "device_map": "auto"
            },
            "mxfp4_only": {
                "use_mxfp4": True,
                "use_flash_attention_3": False,
                "use_megablocks_moe": False,
                "torch_dtype": "auto", 
                "device_map": "auto"
            },
            "megablocks_moe": {
                "use_mxfp4": False,
                "use_flash_attention_3": False,
                "use_megablocks_moe": True,
                "use_kernels": True,
                "torch_dtype": "auto",
                "device_map": "auto"
            },
            "standard": {
                "use_mxfp4": False,
                "use_flash_attention_3": False,
                "use_megablocks_moe": False,
                "torch_dtype": "auto",
                "device_map": "auto"
            },
            "cpu_only": {
                "use_mxfp4": False,
                "use_flash_attention_3": False,
                "use_megablocks_moe": False,
                "torch_dtype": "float32",
                "device_map": "cpu"
            }
        }
        
        return configs.get(strategy, configs["standard"])
    
    def print_compatibility_report(self):
        """Print detailed compatibility report"""
        print("=" * 60)
        print("GPU COMPATIBILITY REPORT")
        print("=" * 60)
        
        if not self.device_info.get("cuda_available"):
            print("❌ CUDA not available - CPU-only inference")
            return
        
        print(f"CUDA Version: {self.device_info['cuda_version']}")
        print(f"PyTorch Version: {self.device_info['pytorch_version']}")
        print(f"Available Devices: {self.device_info['device_count']}")
        print()
        
        for i, device_rec in enumerate(self.compatibility_report["device_recommendations"]):
            device = self.device_info["devices"][i]
            print(f"Device {i}: {device['name']}")
            print(f"  Compute Capability: {device['compute_capability']}")
            print(f"  Memory: {device['total_memory'] / 1024**3:.1f} GB")
            
            # Compatibility status
            mxfp4_status = "✅" if device_rec["mxfp4_compatible"] else "❌"
            fa3_status = "✅" if device_rec["flash_attention_3_compatible"] else "❌"
            moe_status = "✅" if device_rec["megablocks_moe_compatible"] else "❌"
            
            print(f"  mxfp4: {mxfp4_status}")
            print(f"  Flash Attention 3: {fa3_status}")
            print(f"  MegaBlocks MoE: {moe_status}")
            
            if device_rec["reasons"]:
                print("  Limitations:")
                for reason in device_rec["reasons"]:
                    print(f"    - {reason}")
            print()
        
        # Overall recommendations
        print("RECOMMENDATIONS:")
        strategy = self.compatibility_report["recommended_optimization"]
        
        if strategy == "mxfp4_with_flash_attention":
            print("🚀 Best performance: Use mxfp4 + Flash Attention 3")
        elif strategy == "mxfp4_only":
            print("⚡ Good performance: Use mxfp4 quantization")
        elif strategy == "megablocks_moe":
            print("🔧 Alternative optimization: Use MegaBlocks MoE kernels")
        else:
            print("📊 Standard inference without advanced optimizations")
        
        if self.compatibility_report["warnings"]:
            print("\nWARNINGS:")
            for warning in self.compatibility_report["warnings"]:
                print(f"  ⚠️  {warning}")
        
        print("=" * 60)

# Convenience function
def check_gpu_compatibility() -> GPUCompatibilityChecker:
    """Quick function to check GPU compatibility and get recommendations"""
    checker = GPUCompatibilityChecker()
    checker.print_compatibility_report()
    return checker

if __name__ == "__main__":
    check_gpu_compatibility()
