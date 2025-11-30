"""
tests/test_distributed_setup.py

Unit tests for distributed training setup.
"""

import pytest
import sys
import tempfile
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.distributed_setup import (
    setup_distributed_environment,
    is_main_process,
    DeepSpeedConfig,
    get_effective_batch_size,
    reduce_value,
    synchronize_processes
)


# ============================================================================
# Test Distributed Environment Setup
# ============================================================================

def test_setup_distributed_cpu():
    """Test distributed setup on CPU"""
    config = setup_distributed_environment()
    
    assert 'distributed' in config
    assert 'device' in config
    assert 'local_rank' in config
    assert 'world_size' in config


def test_is_main_process():
    """Test main process check"""
    # On CPU/single GPU, should always be main
    assert is_main_process() is True


# ============================================================================
# Test DeepSpeedConfig
# ============================================================================

def test_deepspeed_config_init_default():
    """Test DeepSpeed config initialization with defaults"""
    config = DeepSpeedConfig()
    
    assert config.zero_stage == 3
    assert config.gradient_accumulation_steps == 4
    assert config.train_micro_batch_size_per_gpu == 1


def test_deepspeed_config_init_custom():
    """Test DeepSpeed config initialization with custom values"""
    config = DeepSpeedConfig(
        zero_stage=2,
        gradient_accumulation_steps=8,
        train_micro_batch_size_per_gpu=2
    )
    
    assert config.zero_stage == 2
    assert config.gradient_accumulation_steps == 8
    assert config.train_micro_batch_size_per_gpu == 2


def test_deepspeed_config_create_default():
    """Test creating default config"""
    config = DeepSpeedConfig(zero_stage=3)
    config_dict = config.to_dict()
    
    assert 'train_micro_batch_size_per_gpu' in config_dict
    assert 'gradient_accumulation_steps' in config_dict
    assert 'optimizer' in config_dict
    assert 'zero_optimization' in config_dict


def test_deepspeed_config_zero_stage_0():
    """Test config with ZeRO stage 0 (disabled)"""
    config = DeepSpeedConfig(zero_stage=0)
    config_dict = config.to_dict()
    
    # ZeRO stage 0 means no zero_optimization
    assert 'zero_optimization' not in config_dict


def test_deepspeed_config_zero_stage_3():
    """Test config with ZeRO stage 3"""
    config = DeepSpeedConfig(zero_stage=3)
    config_dict = config.to_dict()
    
    zero_config = config_dict['zero_optimization']
    
    assert zero_config['stage'] == 3
    assert 'offload_optimizer' in zero_config
    assert 'offload_param' in zero_config
    assert zero_config['offload_optimizer']['device'] == 'cpu'


def test_deepspeed_config_save():
    """Test saving config to file"""
    config = DeepSpeedConfig()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_config.json"
        config.save_config(output_path)
        
        assert output_path.exists()
        
        # Verify JSON is valid
        with open(output_path) as f:
            loaded = json.load(f)
        
        assert 'zero_optimization' in loaded


def test_deepspeed_config_load():
    """Test loading config from file"""
    # Create and save config
    config1 = DeepSpeedConfig(zero_stage=2)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.json"
        config1.save_config(config_path)
        
        # Load config
        config2 = DeepSpeedConfig(config_path=config_path)
        
        assert config2.config == config1.config


# ============================================================================
# Test Utility Functions
# ============================================================================

def test_get_effective_batch_size_single_gpu():
    """Test effective batch size calculation for single GPU"""
    batch_size = get_effective_batch_size(
        train_micro_batch_size_per_gpu=2,
        gradient_accumulation_steps=4,
        world_size=1
    )
    
    # 2 * 4 * 1 = 8
    assert batch_size == 8


def test_get_effective_batch_size_multi_gpu():
    """Test effective batch size calculation for multi-GPU"""
    batch_size = get_effective_batch_size(
        train_micro_batch_size_per_gpu=2,
        gradient_accumulation_steps=4,
        world_size=8
    )
    
    # 2 * 4 * 8 = 64
    assert batch_size == 64


def test_reduce_value_no_distributed():
    """Test value reduction without distributed setup"""
    # Should return same value
    result = reduce_value(5.0, average=True)
    assert result == 5.0


def test_synchronize_processes():
    """Test process synchronization"""
    # Should not crash without distributed setup
    synchronize_processes()


# ============================================================================
# Test Configuration Values
# ============================================================================

def test_deepspeed_config_optimizer():
    """Test optimizer configuration"""
    config = DeepSpeedConfig()
    config_dict = config.to_dict()
    
    optimizer = config_dict['optimizer']
    
    assert optimizer['type'] == 'AdamW'
    assert 'lr' in optimizer['params']
    assert 'betas' in optimizer['params']
    assert 'weight_decay' in optimizer['params']


def test_deepspeed_config_scheduler():
    """Test scheduler configuration"""
    config = DeepSpeedConfig()
    config_dict = config.to_dict()
    
    scheduler = config_dict['scheduler']
    
    assert scheduler['type'] == 'WarmupDecayLR'
    assert 'warmup_num_steps' in scheduler['params']
    assert 'total_num_steps' in scheduler['params']


def test_deepspeed_config_mixed_precision():
    """Test mixed precision configuration"""
    config = DeepSpeedConfig()
    config_dict = config.to_dict()
    
    assert 'bf16' in config_dict
    assert config_dict['bf16']['enabled'] is True


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])