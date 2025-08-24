"""
Speed benchmarking client for testing model performance
"""

import json
import time
import asyncio
import logging
import statistics
from typing import Dict, List, Any
from pathlib import Path
import aiohttp
import openai
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class BenchmarkResult:
    """Single benchmark result"""
    prompt_id: int
    prompt: str
    response: str
    latency: float
    tokens_generated: int
    tokens_per_second: float
    success: bool
    error_message: str = None

class SpeedBenchmarkClient:
    """Client for speed benchmarking of language models"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.dataset_path = config.get('dataset')
        self.output_path = config.get('output')
        self.num_requests = config.get('num_requests', 100)
        self.concurrent_requests = config.get('concurrent_requests', 1)
        self.warmup_requests = config.get('warmup_requests', 5)
        
        # Interface configuration
        interface_config = config.get('interface_config', {})
        self.host = interface_config.get('host', 'localhost')
        self.port = interface_config.get('port', 6001)
        self.base_url = f"http://{self.host}:{self.port}/v1"
        
        # Load dataset
        self.prompts = self._load_dataset()
        
        # Initialize OpenAI client
        self.client = openai.OpenAI(
            base_url=self.base_url,
            api_key="dummy"  # Not used but required
        )
    
    @classmethod
    def load_from_dict(cls, config: Dict[str, Any]) -> 'SpeedBenchmarkClient':
        """Load speed benchmark client from configuration"""
        return cls(config)
    
    def _load_dataset(self) -> List[Dict[str, Any]]:
        """Load benchmark dataset from JSON file"""
        try:
            with open(self.dataset_path, 'r') as f:
                data = json.load(f)
                return data.get('prompts', [])
        except Exception as e:
            logger.error(f"Failed to load dataset from {self.dataset_path}: {e}")
            return []
    
    async def _single_request(self, prompt_data: Dict[str, Any]) -> BenchmarkResult:
        """Execute a single benchmark request"""
        prompt_id = prompt_data.get('id')
        prompt = prompt_data.get('prompt')
        expected_tokens = prompt_data.get('expected_tokens', 100)
        
        start_time = time.time()
        
        try:
            response = self.client.chat.completions.create(
                model="llama3-trt",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=expected_tokens
            )
            
            end_time = time.time()
            latency = end_time - start_time
            
            response_text = response.choices[0].message.content
            tokens_generated = len(response_text.split())  # Rough token count
            tokens_per_second = tokens_generated / latency if latency > 0 else 0
            
            return BenchmarkResult(
                prompt_id=prompt_id,
                prompt=prompt,
                response=response_text,
                latency=latency,
                tokens_generated=tokens_generated,
                tokens_per_second=tokens_per_second,
                success=True
            )
            
        except Exception as e:
            end_time = time.time()
            latency = end_time - start_time
            
            logger.error(f"Request failed for prompt {prompt_id}: {e}")
            return BenchmarkResult(
                prompt_id=prompt_id,
                prompt=prompt,
                response="",
                latency=latency,
                tokens_generated=0,
                tokens_per_second=0,
                success=False,
                error_message=str(e)
            )
    
    async def _warmup(self):
        """Perform warmup requests"""
        logger.info(f"Performing {self.warmup_requests} warmup requests...")
        
        warmup_prompts = self.prompts[:self.warmup_requests]
        for prompt_data in warmup_prompts:
            await self._single_request(prompt_data)
        
        logger.info("Warmup completed")
    
    async def _run_benchmark(self) -> List[BenchmarkResult]:
        """Run the main benchmark"""
        logger.info(f"Running benchmark with {self.num_requests} requests, {self.concurrent_requests} concurrent")
        
        # Select prompts for benchmark (cycle through if needed)
        benchmark_prompts = []
        for i in range(self.num_requests):
            prompt_idx = i % len(self.prompts)
            benchmark_prompts.append(self.prompts[prompt_idx])
        
        results = []
        
        # Run requests in batches based on concurrency
        for i in range(0, len(benchmark_prompts), self.concurrent_requests):
            batch = benchmark_prompts[i:i + self.concurrent_requests]
            
            # Execute batch concurrently
            batch_tasks = [self._single_request(prompt) for prompt in batch]
            batch_results = await asyncio.gather(*batch_tasks)
            results.extend(batch_results)
            
            # Progress logging
            if (i + len(batch)) % 10 == 0:
                logger.info(f"Completed {i + len(batch)}/{len(benchmark_prompts)} requests")
        
        return results
    
    def _calculate_statistics(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Calculate benchmark statistics"""
        successful_results = [r for r in results if r.success]
        
        if not successful_results:
            return {"error": "No successful requests"}
        
        latencies = [r.latency for r in successful_results]
        tokens_per_second = [r.tokens_per_second for r in successful_results]
        tokens_generated = [r.tokens_generated for r in successful_results]
        
        stats = {
            "total_requests": len(results),
            "successful_requests": len(successful_results),
            "failed_requests": len(results) - len(successful_results),
            "success_rate": len(successful_results) / len(results) * 100,
            
            "latency_stats": {
                "mean": statistics.mean(latencies),
                "median": statistics.median(latencies),
                "min": min(latencies),
                "max": max(latencies),
                "std": statistics.stdev(latencies) if len(latencies) > 1 else 0
            },
            
            "throughput_stats": {
                "mean_tokens_per_second": statistics.mean(tokens_per_second),
                "median_tokens_per_second": statistics.median(tokens_per_second),
                "total_tokens_generated": sum(tokens_generated),
                "total_time": sum(latencies)
            }
        }
        
        # Add percentiles
        sorted_latencies = sorted(latencies)
        n = len(sorted_latencies)
        stats["latency_stats"]["p50"] = sorted_latencies[int(n * 0.5)]
        stats["latency_stats"]["p90"] = sorted_latencies[int(n * 0.9)]
        stats["latency_stats"]["p95"] = sorted_latencies[int(n * 0.95)]
        stats["latency_stats"]["p99"] = sorted_latencies[int(n * 0.99)]
        
        return stats
    
    def _save_results(self, results: List[BenchmarkResult], stats: Dict[str, Any]):
        """Save benchmark results to file"""
        output_data = {
            "configuration": self.config,
            "statistics": stats,
            "detailed_results": [
                {
                    "prompt_id": r.prompt_id,
                    "prompt": r.prompt,
                    "response": r.response,
                    "latency": r.latency,
                    "tokens_generated": r.tokens_generated,
                    "tokens_per_second": r.tokens_per_second,
                    "success": r.success,
                    "error_message": r.error_message
                }
                for r in results
            ]
        }
        
        # Create output directory
        output_path = Path(self.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save results
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Results saved to {self.output_path}")
    
    def run(self):
        """Run the speed benchmark"""
        async def _async_run():
            # Warmup
            await self._warmup()
            
            # Main benchmark
            results = await self._run_benchmark()
            
            # Calculate statistics
            stats = self._calculate_statistics(results)
            
            # Print summary
            logger.info("Benchmark Summary:")
            logger.info(f"Total requests: {stats.get('total_requests', 0)}")
            logger.info(f"Success rate: {stats.get('success_rate', 0):.2f}%")
            logger.info(f"Mean latency: {stats.get('latency_stats', {}).get('mean', 0):.3f}s")
            logger.info(f"Mean throughput: {stats.get('throughput_stats', {}).get('mean_tokens_per_second', 0):.2f} tokens/s")
            
            # Save results
            self._save_results(results, stats)
        
        # Run async benchmark
        asyncio.run(_async_run())