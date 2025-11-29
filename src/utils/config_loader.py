"""
src/utils/config_loader.py

Configuration loading utilities for DeepSeekMath-V2.
Loads and validates YAML configs with type checking.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml
from pydantic import BaseModel, Field, ValidationError


# ============================================================================
# Base Configuration Classes (using Pydantic for validation)
# ============================================================================

class ModelConfig(BaseModel):
    """Model architecture configuration"""
    model_name: str = Field(description="HuggingFace model name or path")
    max_length: int = Field(default=128000, description="Maximum sequence length")
    use_flash_attention: bool = Field(default=True, description="Use flash attention")
    torch_dtype: str = Field(default="bfloat16", description="Model dtype")
    device_map: str = Field(default="auto", description="Device placement strategy")
    
    class Config:
        """
        Additional Config
        """
        extra = "allow"  # Allow additional fields


class TrainingConfig(BaseModel):
    """Training hyperparameters"""
    learning_rate: float = Field(default=1e-6, description="Learning rate")
    batch_size: int = Field(default=1, description="Batch size per device")
    gradient_accumulation_steps: int = Field(default=16, description="Gradient accumulation")
    num_epochs: int = Field(default=1, description="Number of training epochs")
    warmup_steps: int = Field(default=100, description="Warmup steps")
    max_grad_norm: float = Field(default=1.0, description="Gradient clipping")
    save_steps: int = Field(default=500, description="Save checkpoint every N steps")
    logging_steps: int = Field(default=10, description="Log every N steps")
    
    # GRPO specific
    grpo_group_size: int = Field(default=4, description="Group size for GRPO")
    grpo_kl_coef: float = Field(default=0.1, description="KL divergence coefficient")
    
    # Reward weights
    alpha: float = Field(default=0.76, description="Proof reward weight")
    beta: float = Field(default=0.24, description="Self-verification reward weight")
    
    class Config:
        """
        Extra config
        """
        extra = "allow"


class DataConfig(BaseModel):
    """Data pipeline configuration"""
    data_dir: str = Field(description="Directory containing data")
    train_file: Optional[str] = Field(default=None, description="Training data file")
    val_file: Optional[str] = Field(default=None, description="Validation data file")
    num_workers: int = Field(default=4, description="DataLoader workers")
    max_samples: Optional[int] = Field(default=None, description="Limit number of samples")
    
    class Config:
        """
        Extra Config
        """
        extra = "allow"


class DeepSpeedConfig(BaseModel):
    """DeepSpeed configuration"""
    enabled: bool = Field(default=False, description="Enable DeepSpeed")
    config_file: Optional[str] = Field(default=None, description="DeepSpeed config JSON path")
    zero_stage: int = Field(default=3, description="ZeRO optimization stage")
    
    class Config:
        """
        Extra Config
        """
        extra = "allow"


class ExperimentConfig(BaseModel):
    """Full experiment configuration"""
    experiment_name: str = Field(description="Experiment name")
    output_dir: str = Field(description="Output directory for checkpoints and logs")
    
    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    deepspeed: Optional[DeepSpeedConfig] = None
    
    class Config:
        """
        Extra Config
        """
        extra = "allow"


# ============================================================================
# Configuration Loader
# ============================================================================

class ConfigLoader:
    """
    Loads and validates YAML configuration files.
    
    Example:
        loader = ConfigLoader("configs/training/verifier_train.yaml")
        config = loader.load()
        print(config.model.model_name)
    """
    
    def __init__(self, config_path: Union[str, Path]):
        """
        Initialize config loader.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = Path(config_path)
        
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
    
    def load(self) -> ExperimentConfig:
        """
        Load and validate configuration.
        
        Returns:
            Validated ExperimentConfig object
            
        Raises:
            ValidationError: If config is invalid
            yaml.YAMLError: If YAML is malformed
        """
        with open(self.config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        try:
            config = ExperimentConfig(**config_dict)
            return config
        except ValidationError as e:
            print(f"Configuration validation failed for {self.config_path}")
            print(e)
            raise
    
    def load_raw(self) -> Dict[str, Any]:
        """
        Load raw YAML without validation.
        
        Returns:
            Dictionary of configuration
        """
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)


# ============================================================================
# Configuration Utilities
# ============================================================================

def load_config(config_path: Union[str, Path]) -> ExperimentConfig:
    """
    Convenience function to load config.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Validated ExperimentConfig object
    """
    loader = ConfigLoader(config_path)
    return loader.load()


def save_config(config: Union[ExperimentConfig, Dict], output_path: Union[str, Path]) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration object or dictionary
        output_path: Where to save the config
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert Pydantic model to dict if needed
    if isinstance(config, BaseModel):
        config_dict = config.dict()
    else:
        config_dict = config
    
    with open(output_path, 'w') as f:
        yaml.safe_dump(config_dict, f, default_flow_style=False, sort_keys=False)


def merge_configs(base_config: Dict, override_config: Dict) -> Dict:
    """
    Merge two config dictionaries (override takes precedence).
    
    Args:
        base_config: Base configuration
        override_config: Override configuration
        
    Returns:
        Merged configuration dictionary
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            # Recursively merge nested dicts
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged


def create_default_config(experiment_name: str, output_dir: str = "./experiments") -> ExperimentConfig:
    """
    Create a default configuration.
    
    Args:
        experiment_name: Name of the experiment
        output_dir: Base output directory
        
    Returns:
        Default ExperimentConfig
    """
    config = ExperimentConfig(
        experiment_name=experiment_name,
        output_dir=os.path.join(output_dir, experiment_name),
        model=ModelConfig(
            model_name="deepseek-ai/DeepSeek-V3.2-Exp-SFT",
        ),
        training=TrainingConfig(),
        data=DataConfig(
            data_dir="./data/processed"
        ),
        deepspeed=DeepSpeedConfig()
    )
    
    return config


# ============================================================================
# Example Usage & Testing
# ============================================================================

def example_usage():
    """Example of how to use the config loader"""
    
    # Create a default config
    config = create_default_config("test_experiment")
    
    # Save it
    save_config(config, "configs/example_config.yaml")
    
    # Load it back
    loaded_config = load_config("configs/example_config.yaml")
    
    # Access config values
    print(f"Model: {loaded_config.model.model_name}")
    print(f"Learning rate: {loaded_config.training.learning_rate}")
    print(f"Batch size: {loaded_config.training.batch_size}")
    
    return loaded_config


if __name__ == "__main__":
    # Test the config loader
    print("Testing ConfigLoader...")
    config = example_usage()
    print("✓ ConfigLoader test passed!")


# ============================================================================
# Export
# ============================================================================

__all__ = [
    'ModelConfig',
    'TrainingConfig',
    'DataConfig',
    'DeepSpeedConfig',
    'ExperimentConfig',
    'ConfigLoader',
    'load_config',
    'save_config',
    'merge_configs',
    'create_default_config',
]