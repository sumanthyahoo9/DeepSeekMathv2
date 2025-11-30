"""
src/model/base_model.py

Base model utilities for DeepSeekMath-V2.
Handles loading, saving, and managing pre-trained models.
Refer to the document docs/BASE_MODEL.md for more details about this module.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

try:
    import torch
    import torch.nn as nn
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Mock classes for testing without torch
    class nn:
        class Module:
            pass
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None


# ============================================================================
# Base Model Class
# ============================================================================

class BaseProofModel:
    """
    Base class for proof-related models (verifier, generator).
    
    Handles:
    - Loading pre-trained models from HuggingFace
    - Saving/loading checkpoints
    - Device management (CPU/GPU)
    - Memory-efficient loading
    
    Args:
        model_name: HuggingFace model name or path to local checkpoint
        device: Device to load model on ('cpu', 'cuda', 'auto')
        torch_dtype: Data type for model weights ('float32', 'float16', 'bfloat16')
        load_in_8bit: Whether to load model in 8-bit quantization
    """
    
    def __init__(
        self,
        model_name: str = "deepseek-ai/DeepSeek-V3.2-Exp-SFT",
        device: str = "auto",
        torch_dtype: str = "bfloat16",
        load_in_8bit: bool = False
    ):
        self.model_name = model_name
        self.device = device
        self.torch_dtype = torch_dtype
        self.load_in_8bit = load_in_8bit
        
        self.model = None
        self.tokenizer = None
        
        # Check if torch is available
        if not TORCH_AVAILABLE:
            print("Warning: PyTorch not available. Running in mock mode.")
    
    def load_model(self, **kwargs) -> None:
        """
        Load model and tokenizer from HuggingFace or local path.
        
        Args:
            **kwargs: Additional arguments for AutoModelForCausalLM.from_pretrained
        """
        if not TORCH_AVAILABLE:
            print(f"Mock: Would load model {self.model_name}")
            return
        
        # Convert torch_dtype string to torch dtype
        dtype_map = {
            'float32': torch.float32,
            'float16': torch.float16,
            'bfloat16': torch.bfloat16
        }
        torch_dtype = dtype_map.get(self.torch_dtype, torch.bfloat16)
        
        # Load tokenizer
        print(f"Loading tokenizer from {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Load model
        print(f"Loading model from {self.model_name}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=self.device,
            torch_dtype=torch_dtype,
            load_in_8bit=self.load_in_8bit,
            **kwargs
        )
        
        print(f"✓ Model loaded on {self.get_device()}")
    
    def save_checkpoint(
        self,
        save_path: Union[str, Path],
        save_tokenizer: bool = True
    ) -> None:
        """
        Save model checkpoint.
        
        Args:
            save_path: Directory to save checkpoint
            save_tokenizer: Whether to save tokenizer along with model
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        if not TORCH_AVAILABLE or self.model is None:
            print(f"Mock: Would save checkpoint to {save_path}")
            return
        
        # Save model
        print(f"Saving model to {save_path}...")
        self.model.save_pretrained(save_path)
        
        # Save tokenizer
        if save_tokenizer and self.tokenizer is not None:
            self.tokenizer.save_pretrained(save_path)
        
        print(f"✓ Checkpoint saved to {save_path}")
    
    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> None:
        """
        Load model from a saved checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint directory
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Update model_name to checkpoint path and reload
        self.model_name = str(checkpoint_path)
        self.load_model()
    
    def get_device(self) -> str:
        """
        Get the device model is loaded on.
        
        Returns:
            Device string ('cpu', 'cuda:0', etc.)
        """
        if not TORCH_AVAILABLE or self.model is None:
            return "cpu (mock)"
        
        # Get device of first parameter
        return str(next(self.model.parameters()).device)
    
    def get_model_size(self) -> Dict[str, Any]:
        """
        Get model size information.
        
        Returns:
            Dictionary with parameter counts and memory usage
        """
        if not TORCH_AVAILABLE or self.model is None:
            return {
                'total_params': 0,
                'trainable_params': 0,
                'size_mb': 0,
                'size_gb': 0
            }
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Estimate size in MB (assuming float32)
        size_mb = total_params * 4 / (1024 ** 2)
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'size_mb': size_mb,
            'size_gb': size_mb / 1024
        }
    
    def to(self, device: str) -> None:
        """
        Move model to specified device.
        
        Args:
            device: Target device ('cpu', 'cuda', 'cuda:0', etc.)
        """
        if not TORCH_AVAILABLE or self.model is None:
            print(f"Mock: Would move model to {device}")
            return
        
        self.model = self.model.to(device)
        self.device = device
        print(f"✓ Model moved to {device}")
    
    def eval(self) -> None:
        """Set model to evaluation mode."""
        if TORCH_AVAILABLE and self.model is not None:
            self.model.eval()
    
    def train(self) -> None:
        """Set model to training mode."""
        if TORCH_AVAILABLE and self.model is not None:
            self.model.train()
    
    def __repr__(self) -> str:
        """String representation of model."""
        size_info = self.get_model_size()
        return (
            f"BaseProofModel(\n"
            f"  model_name={self.model_name},\n"
            f"  device={self.get_device()},\n"
            f"  total_params={size_info['total_params']:,},\n"
            f"  size_gb={size_info['size_gb']:.2f}\n"
            f")"
        )


# ============================================================================
# Utility Functions
# ============================================================================

def load_model_and_tokenizer(
    model_name: str,
    device: str = "auto",
    **kwargs
) -> tuple:
    """
    Convenience function to load model and tokenizer.
    
    Args:
        model_name: HuggingFace model name or local path
        device: Device to load on
        **kwargs: Additional arguments for model loading
        
    Returns:
        Tuple of (model, tokenizer)
    """
    base_model = BaseProofModel(model_name=model_name, device=device, **kwargs)
    base_model.load_model()
    return base_model.model, base_model.tokenizer


def count_parameters(model) -> Dict[str, int]:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model or BaseProofModel
        
    Returns:
        Dictionary with parameter counts
    """
    if isinstance(model, BaseProofModel):
        return model.get_model_size()
    
    if not TORCH_AVAILABLE or model is None:
        return {'total_params': 0, 'trainable_params': 0}
    
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_params': total,
        'trainable_params': trainable,
        'frozen_params': total - trainable
    }


def estimate_model_memory(
    num_parameters: int,
    dtype: str = "float16",
    include_gradients: bool = True,
    include_optimizer: bool = True
) -> float:
    """
    Estimate memory required for model.
    
    Args:
        num_parameters: Number of model parameters
        dtype: Data type ('float32', 'float16', 'bfloat16', 'int8')
        include_gradients: Include memory for gradients
        include_optimizer: Include memory for optimizer states (Adam)
        
    Returns:
        Estimated memory in GB
    """
    # Bytes per parameter
    bytes_per_param = {
        'float32': 4,
        'float16': 2,
        'bfloat16': 2,
        'int8': 1
    }
    
    bytes_pp = bytes_per_param.get(dtype, 2)
    
    # Model weights
    memory_gb = (num_parameters * bytes_pp) / (1024 ** 3)
    
    # Gradients (same size as weights)
    if include_gradients:
        memory_gb += (num_parameters * bytes_pp) / (1024 ** 3)
    
    # Optimizer states (Adam: momentum + variance, both fp32)
    if include_optimizer:
        # Adam stores 2 states per parameter in fp32
        memory_gb += (num_parameters * 4 * 2) / (1024 ** 3)
    
    return memory_gb


def get_available_memory() -> Dict[str, float]:
    """
    Get available GPU/CPU memory.
    
    Returns:
        Dictionary with memory information in GB
    """
    info = {'cpu_available_gb': 0, 'gpu_available_gb': 0}
    
    if not TORCH_AVAILABLE:
        return info
    
    # Get CPU memory (approximate)
    try:
        import psutil
        mem = psutil.virtual_memory()
        info['cpu_available_gb'] = mem.available / (1024 ** 3)
        info['cpu_total_gb'] = mem.total / (1024 ** 3)
    except ImportError:
        pass
    
    # Get GPU memory
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(i)
            info[f'gpu_{i}_available_gb'] = free / (1024 ** 3)
            info[f'gpu_{i}_total_gb'] = total / (1024 ** 3)
    
    return info


# ============================================================================
# Export
# ============================================================================

__all__ = [
    'BaseProofModel',
    'load_model_and_tokenizer',
    'count_parameters',
    'estimate_model_memory',
    'get_available_memory',
]