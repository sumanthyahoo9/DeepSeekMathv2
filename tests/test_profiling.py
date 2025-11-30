"""
tests/test_profiling.py

Unit tests for GPU profiling utilities.
"""
import time
import sys
import tempfile
from pathlib import Path
import pytest

from src.utils.profiling import (
    GPUMemoryProfiler,
    TrainingTimer,
    ThroughputTracker,
    ComprehensiveProfiler,
    get_gpu_info,
    check_memory_available
)
# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================================
# Test GPUMemoryProfiler
# ============================================================================

def test_memory_profiler_init():
    """Test memory profiler initialization"""
    profiler = GPUMemoryProfiler(device=0)
    assert profiler.device == 0
    assert profiler.track_history is True


def test_memory_profiler_measure():
    """Test taking memory measurements"""
    profiler = GPUMemoryProfiler(device="cpu")
    
    measurement = profiler.measure("test")
    
    assert 'timestamp' in measurement
    assert 'tag' in measurement
    assert measurement['tag'] == "test"
    assert 'allocated_gb' in measurement


def test_memory_profiler_baseline():
    """Test setting baseline"""
    profiler = GPUMemoryProfiler()
    
    profiler.set_baseline()
    
    assert profiler.baseline_memory >= 0.0


def test_memory_profiler_leak_detection():
    """Test memory leak detection"""
    profiler = GPUMemoryProfiler()
    
    profiler.set_baseline()
    leak = profiler.detect_leak(threshold_gb=0.5)
    
    # In mock mode, should not detect leak
    assert leak is False


def test_memory_profiler_summary():
    """Test getting summary statistics"""
    profiler = GPUMemoryProfiler()
    
    profiler.measure("test1")
    profiler.measure("test2")
    
    summary = profiler.get_summary()
    
    assert 'num_measurements' in summary
    assert summary['num_measurements'] == 2


# ============================================================================
# Test TrainingTimer
# ============================================================================

def test_timer_init():
    """Test timer initialization"""
    timer = TrainingTimer(device=0)
    assert timer.device == 0


def test_timer_start_stop():
    """Test timing an operation"""
    timer = TrainingTimer()
    
    timer.start("test_op")
    time.sleep(0.01)
    elapsed = timer.stop("test_op")
    
    assert elapsed >= 0.01
    assert "test_op" in timer.timings


def test_timer_invalid_stop():
    """Test stopping timer that wasn't started"""
    timer = TrainingTimer()
    
    with pytest.raises(ValueError):
        timer.stop("nonexistent")


def test_timer_summary():
    """Test getting timing summary"""
    timer = TrainingTimer()
    
    timer.start("op1")
    timer.stop("op1")
    timer.start("op1")
    timer.stop("op1")
    
    summary = timer.get_summary()
    
    assert 'op1' in summary
    assert summary['op1']['count'] == 2
    assert 'mean_sec' in summary['op1']


# ============================================================================
# Test ThroughputTracker
# ============================================================================

def test_throughput_tracker_init():
    """Test throughput tracker initialization"""
    tracker = ThroughputTracker(window_size=10)
    assert tracker.window_size == 10


def test_throughput_tracker_update():
    """Test updating throughput"""
    tracker = ThroughputTracker()
    
    time.sleep(0.01)
    tracker.update(num_samples=8, num_tokens=1024)
    
    throughput = tracker.get_current_throughput()
    
    assert 'samples_per_sec' in throughput
    assert 'tokens_per_sec' in throughput
    assert throughput['samples_per_sec'] > 0


def test_throughput_tracker_window():
    """Test windowing of throughput measurements"""
    tracker = ThroughputTracker(window_size=2)
    
    for i in range(5):
        tracker.update(num_samples=1)
    
    # Should only keep last 2
    assert len(tracker.samples_history) == 2


# ============================================================================
# Test ComprehensiveProfiler
# ============================================================================

def test_comprehensive_profiler_init():
    """Test comprehensive profiler initialization"""
    profiler = ComprehensiveProfiler(device=0)
    
    assert profiler.memory_profiler is not None
    assert profiler.timer is not None
    assert profiler.throughput_tracker is not None


def test_comprehensive_profiler_iteration():
    """Test profiling a training iteration"""
    profiler = ComprehensiveProfiler()
    
    profiler.start_iteration()
    time.sleep(0.01)
    profiler.end_iteration(num_samples=4)
    
    summary = profiler.get_full_summary()
    
    assert 'timestamp' in summary
    assert 'memory' in summary
    assert 'timing' in summary
    assert 'throughput' in summary


def test_comprehensive_profiler_export():
    """Test exporting profiling data"""
    profiler = ComprehensiveProfiler()
    
    profiler.start_iteration()
    profiler.end_iteration(num_samples=4)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "profile.json"
        profiler.export_to_json(output_path)
        
        assert output_path.exists()
        
        # Verify JSON is valid
        import json
        with open(output_path) as f:
            data = json.load(f)
        
        assert 'timestamp' in data


# ============================================================================
# Test Utility Functions
# ============================================================================

def test_get_gpu_info():
    """Test getting GPU information"""
    info = get_gpu_info()
    
    assert 'available' in info
    assert isinstance(info['available'], bool)


def test_check_memory_available():
    """Test checking memory availability"""
    # Should return False on CPU/mock
    available = check_memory_available(required_gb=10.0, device=0)
    
    assert isinstance(available, bool)


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])