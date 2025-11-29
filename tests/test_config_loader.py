"""
tests/test_config_loader.py

Unit tests for configuration loading utilities.
"""

import pytest
import sys
import tempfile
from pathlib import Path
from pydantic import ValidationError

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config_loader import (
    ModelConfig,
    TrainingConfig,
    DataConfig,
    ExperimentConfig,
    ConfigLoader,
    load_config,
    save_config,
    merge_configs,
    create_default_config
)


# ============================================================================
# Test Configuration Classes
# ============================================================================

def test_model_config_valid():
    """Test creating valid model config"""
    config = ModelConfig(model_name="test-model")
    assert config.model_name == "test-model"
    assert config.max_length == 128000  # Default value


def test_training_config_defaults():
    """Test training config default values"""
    config = TrainingConfig()
    assert config.learning_rate == 1e-6
    assert config.batch_size == 1
    assert config.alpha == 0.76
    assert config.beta == 0.24


def test_experiment_config_creation():
    """Test creating full experiment config"""
    config = ExperimentConfig(
        experiment_name="test",
        output_dir="./test_output",
        model=ModelConfig(model_name="test-model"),
        training=TrainingConfig(),
        data=DataConfig(data_dir="./data")
    )
    
    assert config.experiment_name == "test"
    assert config.model.model_name == "test-model"
    assert config.training.learning_rate == 1e-6


# ============================================================================
# Test Config File Operations
# ============================================================================

def test_save_and_load_config():
    """Test saving and loading config from YAML"""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "test_config.yaml"
        
        # Create and save config
        original_config = create_default_config("test_exp")
        save_config(original_config, config_path)
        
        # Load config back
        loaded_config = load_config(config_path)
        
        # Verify
        assert loaded_config.experiment_name == original_config.experiment_name
        assert loaded_config.model.model_name == original_config.model.model_name
        assert loaded_config.training.learning_rate == original_config.training.learning_rate


def test_config_loader_file_not_found():
    """Test ConfigLoader with non-existent file"""
    with pytest.raises(FileNotFoundError):
        ConfigLoader("nonexistent_config.yaml")


# ============================================================================
# Test Config Utilities
# ============================================================================

def test_merge_configs():
    """Test merging two config dictionaries"""
    base = {
        "model": {"name": "base-model", "size": "large"},
        "training": {"lr": 1e-5}
    }
    
    override = {
        "model": {"name": "override-model"},  # Override name, keep size
        "training": {"lr": 1e-6, "batch_size": 32}  # Override lr, add batch_size
    }
    
    merged = merge_configs(base, override)
    
    assert merged["model"]["name"] == "override-model"
    assert merged["model"]["size"] == "large"  # Kept from base
    assert merged["training"]["lr"] == 1e-6
    assert merged["training"]["batch_size"] == 32


def test_create_default_config():
    """Test creating default configuration"""
    config = create_default_config("my_experiment", "./experiments")
    
    assert config.experiment_name == "my_experiment"
    assert "./experiments/my_experiment" in config.output_dir
    assert config.model.model_name is not None
    assert config.training.learning_rate > 0


# ============================================================================
# Test Validation
# ============================================================================

def test_invalid_config_validation():
    """Test that invalid configs raise ValidationError"""
    
    # Missing required fields
    with pytest.raises(ValidationError):
        ExperimentConfig(
            experiment_name="test"
            # Missing output_dir, model, training, data
        )


def test_extra_fields_allowed():
    """Test that extra fields are allowed in configs"""
    config = ModelConfig(
        model_name="test",
        custom_field="custom_value"  # Extra field
    )
    
    assert config.model_name == "test"
    # Extra field should be accepted due to Config.extra = "allow"


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])