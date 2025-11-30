"""
tests/test_base_model.py

Unit tests for base model utilities.
"""

import pytest
import sys
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.base_model import (
    BaseProofModel,
    count_parameters,
    estimate_model_memory,
    get_available_memory
)


# ============================================================================
# Test BaseProofModel Initialization
# ============================================================================

def test_base_model_init_default():
    """Test base model initialization with defaults"""
    model = BaseProofModel()
    
    assert model.model_name == "deepseek-ai/DeepSeek-V3.2-Exp-SFT"
    assert model.device == "auto"
    assert model.torch_dtype == "bfloat16"
    assert model.load_in_8bit is False


def test_base_model_init_custom():
    """Test base model initialization with custom params"""
    model = BaseProofModel(
        model_name="test-model",
        device="cpu",
        torch_dtype="float16",
        load_in_8bit=True
    )
    
    assert model.model_name == "test-model"
    assert model.device == "cpu"
    assert model.torch_dtype == "float16"
    assert model.load_in_8bit is True


# ============================================================================
# Test Model Loading (Mock Mode)
# ============================================================================

def test_base_model_load_mock():
    """Test model loading in mock mode (no torch)"""
    model = BaseProofModel(model_name="test-model")
    
    # Should work in mock mode
    model.load_model()
    
    # Model and tokenizer will be None in mock mode
    assert model.model is None
    assert model.tokenizer is None


# ============================================================================
# Test Checkpoint Operations
# ============================================================================

def test_save_checkpoint_mock():
    """Test saving checkpoint in mock mode"""
    model = BaseProofModel()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "checkpoint"
        
        # Should work in mock mode
        model.save_checkpoint(save_path)
        
        # Directory should be created
        assert save_path.exists()


def test_load_checkpoint_not_found():
    """Test loading non-existent checkpoint"""
    model = BaseProofModel()
    
    with pytest.raises(FileNotFoundError):
        model.load_checkpoint("nonexistent_checkpoint")


def test_load_checkpoint_mock():
    """Test loading checkpoint in mock mode"""
    model = BaseProofModel()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "checkpoint"
        checkpoint_path.mkdir()
        
        # Should update model_name
        model.load_checkpoint(checkpoint_path)
        assert str(checkpoint_path) in model.model_name


# ============================================================================
# Test Device Management
# ============================================================================

def test_get_device_mock():
    """Test get_device in mock mode"""
    model = BaseProofModel()
    device = model.get_device()
    
    assert "cpu" in device.lower() or "mock" in device.lower()


def test_to_device_mock():
    """Test moving model to device in mock mode"""
    model = BaseProofModel()
    
    # Should work without error
    model.to("cpu")
    model.to("cuda")


# ============================================================================
# Test Model Size
# ============================================================================

def test_get_model_size_mock():
    """Test getting model size in mock mode"""
    model = BaseProofModel()
    size_info = model.get_model_size()
    
    assert 'total_params' in size_info
    assert 'trainable_params' in size_info
    assert 'size_mb' in size_info
    assert isinstance(size_info['total_params'], int)


# ============================================================================
# Test Training/Eval Modes
# ============================================================================

def test_eval_mode():
    """Test setting model to eval mode"""
    model = BaseProofModel()
    # Should work without error
    model.eval()


def test_train_mode():
    """Test setting model to train mode"""
    model = BaseProofModel()
    # Should work without error
    model.train()


# ============================================================================
# Test String Representation
# ============================================================================

def test_repr():
    """Test string representation"""
    model = BaseProofModel(model_name="test-model")
    repr_str = repr(model)
    
    assert "BaseProofModel" in repr_str
    assert "test-model" in repr_str
    assert "total_params" in repr_str


# ============================================================================
# Test Utility Functions
# ============================================================================

def test_count_parameters_with_base_model():
    """Test counting parameters with BaseProofModel"""
    model = BaseProofModel()
    params = count_parameters(model)
    
    assert 'total_params' in params
    assert 'trainable_params' in params


def test_count_parameters_with_none():
    """Test counting parameters with None"""
    params = count_parameters(None)
    
    assert params['total_params'] == 0
    assert params['trainable_params'] == 0


def test_estimate_model_memory_small():
    """Test memory estimation for small model"""
    # 1B parameters
    memory_gb = estimate_model_memory(
        num_parameters=1_000_000_000,
        dtype="float16",
        include_gradients=False,
        include_optimizer=False
    )
    
    # 1B params * 2 bytes = 2GB
    assert 1.5 < memory_gb < 2.5


def test_estimate_model_memory_with_gradients():
    """Test memory estimation with gradients"""
    memory_no_grad = estimate_model_memory(
        num_parameters=1_000_000_000,
        dtype="float16",
        include_gradients=False,
        include_optimizer=False
    )
    
    memory_with_grad = estimate_model_memory(
        num_parameters=1_000_000_000,
        dtype="float16",
        include_gradients=True,
        include_optimizer=False
    )
    
    # With gradients should be ~2x (weights + gradients)
    assert memory_with_grad > memory_no_grad * 1.8


def test_estimate_model_memory_with_optimizer():
    """Test memory estimation with optimizer"""
    memory_no_opt = estimate_model_memory(
        num_parameters=1_000_000_000,
        dtype="float16",
        include_gradients=True,
        include_optimizer=False
    )
    
    memory_with_opt = estimate_model_memory(
        num_parameters=1_000_000_000,
        dtype="float16",
        include_gradients=True,
        include_optimizer=True
    )
    
    # With optimizer (Adam) should be significantly more
    assert memory_with_opt > memory_no_opt * 2


def test_estimate_model_memory_dtype_comparison():
    """Test that different dtypes give different memory"""
    memory_fp32 = estimate_model_memory(
        num_parameters=1_000_000_000,
        dtype="float32",
        include_gradients=False,
        include_optimizer=False
    )
    
    memory_fp16 = estimate_model_memory(
        num_parameters=1_000_000_000,
        dtype="float16",
        include_gradients=False,
        include_optimizer=False
    )
    
    memory_int8 = estimate_model_memory(
        num_parameters=1_000_000_000,
        dtype="int8",
        include_gradients=False,
        include_optimizer=False
    )
    
    # fp32 > fp16 > int8
    assert memory_fp32 > memory_fp16
    assert memory_fp16 > memory_int8
    
    # Roughly correct ratios
    assert 1.8 < memory_fp32 / memory_fp16 < 2.2  # ~2x
    assert 1.8 < memory_fp16 / memory_int8 < 2.2  # ~2x


def test_get_available_memory():
    """Test getting available memory"""
    memory_info = get_available_memory()
    
    assert isinstance(memory_info, dict)
    assert 'cpu_available_gb' in memory_info
    assert 'gpu_available_gb' in memory_info


# ============================================================================
# Integration Tests
# ============================================================================

def test_full_workflow():
    """Test complete workflow: init → load → save → load checkpoint"""
    model = BaseProofModel(model_name="test-model", device="cpu")
    
    # Load model (mock)
    model.load_model()
    
    # Get size
    size_info = model.get_model_size()
    assert size_info is not None
    
    # Save checkpoint
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "test_checkpoint"
        model.save_checkpoint(checkpoint_path)
        
        # Load checkpoint
        model2 = BaseProofModel()
        model2.load_checkpoint(checkpoint_path)
        
        assert str(checkpoint_path) in model2.model_name


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])