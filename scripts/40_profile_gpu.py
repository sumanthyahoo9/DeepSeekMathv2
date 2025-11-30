#!/usr/bin/env python3
"""
scripts/40_profile_gpu.py

Standalone GPU profiling and benchmarking script.
Run this before training to check GPU capabilities.
Refer to docs/PROFILING.md for details about this script.
"""

import argparse
import json
import sys
from pathlib import Path

from src.utils.profiling import (
    GPUMemoryProfiler,
    get_gpu_info,
)
from src.model.base_model import estimate_model_memory
# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def profile_gpu_memory(device: int = 0) -> dict:
    """
    Profile GPU memory capacity and availability.
    
    Args:
        device: GPU device index
        
    Returns:
        Dictionary with memory profiling results
    """
    print(f"\n{'='*60}")
    print("GPU Memory Profiling")
    print(f"{'='*60}\n")
    
    profiler = GPUMemoryProfiler(device=device)
    
    # Baseline measurement
    profiler.set_baseline()
    baseline = profiler.measure("baseline")
    
    print("Baseline Memory:")
    print(f"  Allocated: {baseline['allocated_gb']:.2f} GB")
    print(f"  Reserved:  {baseline['reserved_gb']:.2f} GB")
    print(f"  Peak:      {baseline['max_allocated_gb']:.2f} GB")
    
    # Get total memory
    gpu_info = get_gpu_info()
    if gpu_info['available'] and device < len(gpu_info['devices']):
        total_memory = gpu_info['devices'][device]['total_memory_gb']
        available = total_memory - baseline['allocated_gb']
        
        print(f"\nTotal GPU Memory: {total_memory:.2f} GB")
        print(f"Available:        {available:.2f} GB")
        print(f"Utilization:      {(baseline['allocated_gb']/total_memory*100):.1f}%")
    
    return profiler.get_summary()


def estimate_model_requirements() -> dict:
    """
    Estimate memory requirements for different model sizes.
    
    Returns:
        Dictionary with memory estimates
    """
    print(f"\n{'='*60}")
    print("Model Memory Requirements")
    print(f"{'='*60}\n")
    
    models = [
        ("DeepSeek-Math-7B", 7_000_000_000),
        ("DeepSeek-Math-70B", 70_000_000_000),
        ("DeepSeek-V3 (active)", 37_000_000_000),
        ("DeepSeek-V3 (total)", 236_000_000_000),
    ]
    
    estimates = {}
    
    print(f"{'Model':<25} {'Inference (fp16)':<20} {'Training (fp16+Adam)':<25}")
    print("-" * 70)
    
    for model_name, num_params in models:
        # Inference: just model weights
        inference_mem = estimate_model_memory(
            num_params,
            dtype="float16",
            include_gradients=False,
            include_optimizer=False
        )
        
        # Training: weights + gradients + optimizer
        training_mem = estimate_model_memory(
            num_params,
            dtype="float16",
            include_gradients=True,
            include_optimizer=True
        )
        
        print(f"{model_name:<25} {inference_mem:>6.1f} GB          {training_mem:>8.1f} GB")
        
        estimates[model_name] = {
            'num_params': num_params,
            'inference_gb': inference_mem,
            'training_gb': training_mem
        }
    
    return estimates


def benchmark_throughput(device: int = 0, batch_sizes = None) -> dict:
    """
    Benchmark throughput for different batch sizes (mock).
    
    Args:
        device: GPU device index
        batch_sizes: List of batch sizes to test
        
    Returns:
        Dictionary with benchmark results
    """
    if not batch_sizes:
        batch_sizes = [1, 2, 4, 8]
    print(f"\n{'='*60}")
    print("Throughput Benchmarking (Mock)")
    print(f"{'='*60}\n")
    
    print("Note: This is a mock benchmark. For real benchmarking,")
    print("      run with actual model loaded.\n")
    
    print(f"{'Batch Size':<15} {'Samples/sec':<15} {'Memory (GB)':<15}")
    print("-" * 45)
    
    results = {}
    
    for batch_size in batch_sizes:
        # Mock values (real would measure actual throughput)
        samples_per_sec = 10.0 / batch_size  # Decreases with batch size
        memory_gb = 5.0 + (batch_size * 2.0)  # Increases with batch size
        
        print(f"{batch_size:<15} {samples_per_sec:<15.2f} {memory_gb:<15.2f}")
        
        results[f'batch_{batch_size}'] = {
            'batch_size': batch_size,
            'samples_per_sec': samples_per_sec,
            'memory_gb': memory_gb
        }
    
    return results


def check_model_compatibility(device: int = 0) -> dict:
    """
    Check which models can fit on the GPU.
    
    Args:
        device: GPU device index
        
    Returns:
        Dictionary with compatibility results
    """
    print(f"\n{'='*60}")
    print("Model Compatibility Check")
    print(f"{'='*60}\n")
    
    gpu_info = get_gpu_info()
    
    if not gpu_info['available'] or device >= len(gpu_info['devices']):
        print("⚠ GPU not available for compatibility check")
        return {}
    
    total_memory = gpu_info['devices'][device]['total_memory_gb']
    
    print(f"Available GPU Memory: {total_memory:.2f} GB\n")
    
    models = [
        ("DeepSeek-Math-7B (inference)", 14),
        ("DeepSeek-Math-7B (training)", 60),
        ("DeepSeek-Math-70B (inference)", 140),
        ("DeepSeek-V3 active (inference)", 74),
        ("DeepSeek-V3 active (training)", 296),
    ]
    
    compatibility = {}
    
    print(f"{'Model':<35} {'Required':<15} {'Status':<15}")
    print("-" * 65)
    
    for model_name, required_gb in models:
        fits = required_gb <= total_memory
        status = "✓ Fits" if fits else "✗ Too large"
        
        print(f"{model_name:<35} {required_gb:>6.1f} GB      {status:<15}")
        
        compatibility[model_name] = {
            'required_gb': required_gb,
            'fits': fits
        }
    
    return compatibility


def main():
    """Main profiling script."""
    parser = argparse.ArgumentParser(
        description="Profile GPU capabilities for DeepSeekMath-V2 training"
    )
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        help='GPU device index (default: 0)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='experiments/profiling_results/gpu_profile.json',
        help='Output JSON file path'
    )
    parser.add_argument(
        '--skip-throughput',
        action='store_true',
        help='Skip throughput benchmarking'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("DeepSeekMath-V2 GPU Profiling")
    print("="*60)
    
    # Get GPU info
    gpu_info = get_gpu_info()
    
    print("\nGPU Information:")
    if gpu_info['available']:
        print(f"  Devices: {gpu_info['device_count']}")
        for device in gpu_info['devices']:
            print(f"  [{device['index']}] {device['name']}")
            print(f"      Memory: {device['total_memory_gb']:.2f} GB")
            print(f"      Compute: {device['compute_capability']}")
    else:
        print(f"  {gpu_info['message']}")
    
    # Run profiling
    results = {
        'timestamp': str(Path(__file__).name),
        'device': args.device,
        'gpu_info': gpu_info
    }
    
    # Memory profiling
    results['memory_profile'] = profile_gpu_memory(args.device)
    
    # Model requirements
    results['model_estimates'] = estimate_model_requirements()
    
    # Throughput benchmarking
    if not args.skip_throughput:
        results['throughput_benchmark'] = benchmark_throughput(args.device)
    
    # Compatibility check
    results['compatibility'] = check_model_compatibility(args.device)
    
    # Export results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("✓ Profiling complete! Results saved to:")
    print(f"  {output_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()