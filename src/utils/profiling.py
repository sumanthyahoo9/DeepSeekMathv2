"""
src/utils/profiling.py

GPU profiling utilities for DeepSeekMath-V2 training.
Tracks memory, timing, and throughput metrics.
"""

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional, Union
from datetime import datetime

try:
    import torch
    import torch.cuda as cuda
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    cuda = None


# ============================================================================
# GPU Memory Profiler
# ============================================================================

class GPUMemoryProfiler:
    """
    Track GPU memory usage during training.
    
    Monitors:
    - Current allocated memory
    - Peak memory usage
    - Reserved memory
    - Memory leaks (increasing baseline)
    
    Args:
        device: GPU device index or 'cpu'
        track_history: Whether to store history of measurements
    """
    
    def __init__(
        self,
        device: Union[int, str] = 0,
        track_history: bool = True
    ):
        self.device = device
        self.track_history = track_history
        self.measurements = []
        self.baseline_memory = 0.0
        
        if TORCH_AVAILABLE and isinstance(device, int):
            self.device_name = f"cuda:{device}"
        else:
            self.device_name = "cpu"
    
    def measure(self, tag: str = "default") -> Dict[str, float]:
        """
        Take a memory measurement.
        
        Args:
            tag: Label for this measurement
            
        Returns:
            Dictionary with memory stats in GB
        """
        if not TORCH_AVAILABLE or self.device_name == "cpu":
            return self._mock_measure(tag)
        
        # Synchronize to get accurate measurement
        torch.cuda.synchronize(self.device)
        
        stats = {
            'timestamp': time.time(),
            'tag': tag,
            'allocated_gb': torch.cuda.memory_allocated(self.device) / (1024**3),
            'reserved_gb': torch.cuda.memory_reserved(self.device) / (1024**3),
            'max_allocated_gb': torch.cuda.max_memory_allocated(self.device) / (1024**3),
            'max_reserved_gb': torch.cuda.max_memory_reserved(self.device) / (1024**3)
        }
        
        if self.track_history:
            self.measurements.append(stats)
        
        return stats
    
    def set_baseline(self) -> None:
        """Set current memory as baseline for leak detection."""
        measurement = self.measure("baseline")
        self.baseline_memory = measurement['allocated_gb']
    
    def detect_leak(self, threshold_gb: float = 0.5) -> bool:
        """
        Detect memory leak.
        
        Args:
            threshold_gb: Threshold for leak detection (GB above baseline)
            
        Returns:
            True if leak detected
        """
        current = self.measure("leak_check")
        leak = current['allocated_gb'] - self.baseline_memory
        return leak > threshold_gb
    
    def reset_peak_stats(self) -> None:
        """Reset peak memory statistics."""
        if TORCH_AVAILABLE and self.device_name != "cpu":
            torch.cuda.reset_peak_memory_stats(self.device)
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of memory usage.
        
        Returns:
            Summary statistics
        """
        if not self.measurements:
            return {'num_measurements': 0}
        
        allocated = [m['allocated_gb'] for m in self.measurements]
        reserved = [m['reserved_gb'] for m in self.measurements]
        
        return {
            'num_measurements': len(self.measurements),
            'mean_allocated_gb': sum(allocated) / len(allocated),
            'peak_allocated_gb': max(allocated),
            'mean_reserved_gb': sum(reserved) / len(reserved),
            'peak_reserved_gb': max(reserved),
            'baseline_gb': self.baseline_memory,
            'device': self.device_name
        }
    
    def _mock_measure(self, tag: str) -> Dict[str, float]:
        """Mock measurement for CPU/testing."""
        stats = {
            'timestamp': time.time(),
            'tag': tag,
            'allocated_gb': 0.0,
            'reserved_gb': 0.0,
            'max_allocated_gb': 0.0,
            'max_reserved_gb': 0.0
        }
        
        if self.track_history:
            self.measurements.append(stats)
        
        return stats


# ============================================================================
# Training Timer
# ============================================================================

class TrainingTimer:
    """
    Track timing for training operations.
    
    Monitors:
    - Forward pass time
    - Backward pass time
    - Optimizer step time
    - Data loading time
    - Total iteration time
    
    Args:
        device: GPU device for synchronization
    """
    
    def __init__(self, device: Union[int, str] = 0):
        self.device = device
        self.timings = defaultdict(list)
        self.start_times = {}
        
        if TORCH_AVAILABLE and isinstance(device, int):
            self.device_name = f"cuda:{device}"
        else:
            self.device_name = "cpu"
    
    def start(self, operation: str) -> None:
        """
        Start timing an operation.
        
        Args:
            operation: Name of operation (e.g., 'forward', 'backward')
        """
        # Synchronize GPU before timing
        if TORCH_AVAILABLE and self.device_name != "cpu":
            torch.cuda.synchronize(self.device)
        
        self.start_times[operation] = time.time()
    
    def stop(self, operation: str) -> float:
        """
        Stop timing an operation.
        
        Args:
            operation: Name of operation
            
        Returns:
            Elapsed time in seconds
        """
        if operation not in self.start_times:
            raise ValueError(f"Timer for '{operation}' was not started")
        
        # Synchronize GPU before stopping
        if TORCH_AVAILABLE and self.device_name != "cpu":
            torch.cuda.synchronize(self.device)
        
        elapsed = time.time() - self.start_times[operation]
        self.timings[operation].append(elapsed)
        
        del self.start_times[operation]
        return elapsed
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Get summary of all timings.
        
        Returns:
            Dictionary mapping operation to timing stats
        """
        summary = {}
        
        for operation, times in self.timings.items():
            if not times:
                continue
            
            summary[operation] = {
                'count': len(times),
                'total_sec': sum(times),
                'mean_sec': sum(times) / len(times),
                'min_sec': min(times),
                'max_sec': max(times)
            }
        
        return summary


# ============================================================================
# Throughput Tracker
# ============================================================================

class ThroughputTracker:
    """
    Track training throughput metrics.
    
    Monitors:
    - Samples per second
    - Tokens per second
    - Iterations per second
    - GPU utilization
    
    Args:
        window_size: Number of iterations for moving average
    """
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.samples_history = []
        self.tokens_history = []
        self.time_history = []
        self.last_update = time.time()
    
    def update(
        self,
        num_samples: int,
        num_tokens: Optional[int] = None
    ) -> None:
        """
        Update throughput metrics.
        
        Args:
            num_samples: Number of samples processed
            num_tokens: Number of tokens processed (optional)
        """
        current_time = time.time()
        elapsed = current_time - self.last_update
        
        self.samples_history.append(num_samples)
        self.time_history.append(elapsed)
        
        if num_tokens is not None:
            self.tokens_history.append(num_tokens)
        
        # Keep only last window_size measurements
        if len(self.samples_history) > self.window_size:
            self.samples_history.pop(0)
            self.time_history.pop(0)
            if self.tokens_history:
                self.tokens_history.pop(0)
        
        self.last_update = current_time
    
    def get_current_throughput(self) -> Dict[str, float]:
        """
        Get current throughput metrics.
        
        Returns:
            Dictionary with throughput stats
        """
        if not self.samples_history:
            return {'samples_per_sec': 0.0}
        
        total_samples = sum(self.samples_history)
        total_time = sum(self.time_history)
        
        metrics = {
            'samples_per_sec': total_samples / total_time if total_time > 0 else 0.0,
            'window_size': len(self.samples_history)
        }
        
        if self.tokens_history:
            total_tokens = sum(self.tokens_history)
            metrics['tokens_per_sec'] = total_tokens / total_time if total_time > 0 else 0.0
        
        return metrics


# ============================================================================
# Comprehensive Profiler
# ============================================================================

class ComprehensiveProfiler:
    """
    Combined profiler tracking memory, timing, and throughput.
    
    Args:
        device: GPU device index
        profile_memory: Whether to profile memory
        profile_timing: Whether to profile timing
        profile_throughput: Whether to profile throughput
    """
    
    def __init__(
        self,
        device: Union[int, str] = 0,
        profile_memory: bool = True,
        profile_timing: bool = True,
        profile_throughput: bool = True
    ):
        self.device = device
        
        self.memory_profiler = GPUMemoryProfiler(device) if profile_memory else None
        self.timer = TrainingTimer(device) if profile_timing else None
        self.throughput_tracker = ThroughputTracker() if profile_throughput else None
        
        self.start_time = time.time()
    
    def start_iteration(self) -> None:
        """Start profiling a training iteration."""
        if self.timer:
            self.timer.start('iteration')
        
        if self.memory_profiler:
            self.memory_profiler.measure('iteration_start')
    
    def end_iteration(
        self,
        num_samples: int,
        num_tokens: Optional[int] = None
    ) -> None:
        """
        End profiling a training iteration.
        
        Args:
            num_samples: Number of samples in iteration
            num_tokens: Number of tokens in iteration
        """
        if self.timer:
            self.timer.stop('iteration')
        
        if self.memory_profiler:
            self.memory_profiler.measure('iteration_end')
        
        if self.throughput_tracker:
            self.throughput_tracker.update(num_samples, num_tokens)
    
    def get_full_summary(self) -> Dict[str, Any]:
        """
        Get complete profiling summary.
        
        Returns:
            Dictionary with all profiling data
        """
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_runtime_sec': time.time() - self.start_time,
            'device': str(self.device)
        }
        
        if self.memory_profiler:
            summary['memory'] = self.memory_profiler.get_summary()
        
        if self.timer:
            summary['timing'] = self.timer.get_summary()
        
        if self.throughput_tracker:
            summary['throughput'] = self.throughput_tracker.get_current_throughput()
        
        return summary
    
    def export_to_json(self, output_path: Union[str, Path]) -> None:
        """
        Export profiling data to JSON file.
        
        Args:
            output_path: Path to output JSON file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        summary = self.get_full_summary()
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✓ Profiling data exported to {output_path}")


# ============================================================================
# Utility Functions
# ============================================================================

def get_gpu_info() -> Dict[str, Any]:
    """
    Get GPU device information.
    
    Returns:
        Dictionary with GPU details
    """
    if not TORCH_AVAILABLE:
        return {'available': False, 'message': 'PyTorch not available'}
    
    if not torch.cuda.is_available():
        return {'available': False, 'message': 'CUDA not available'}
    
    info = {
        'available': True,
        'device_count': torch.cuda.device_count(),
        'devices': []
    }
    
    for i in range(torch.cuda.device_count()):
        device_info = {
            'index': i,
            'name': torch.cuda.get_device_name(i),
            'total_memory_gb': torch.cuda.get_device_properties(i).total_memory / (1024**3),
            'compute_capability': f"{torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}"
        }
        info['devices'].append(device_info)
    
    return info


def check_memory_available(required_gb: float, device: int = 0) -> bool:
    """
    Check if sufficient GPU memory is available.
    
    Args:
        required_gb: Required memory in GB
        device: GPU device index
        
    Returns:
        True if sufficient memory available
    """
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        return False
    
    free_memory = torch.cuda.get_device_properties(device).total_memory / (1024**3)
    allocated = torch.cuda.memory_allocated(device) / (1024**3)
    available = free_memory - allocated
    
    return available >= required_gb


# ============================================================================
# Export
# ============================================================================

__all__ = [
    'GPUMemoryProfiler',
    'TrainingTimer',
    'ThroughputTracker',
    'ComprehensiveProfiler',
    'get_gpu_info',
    'check_memory_available',
]