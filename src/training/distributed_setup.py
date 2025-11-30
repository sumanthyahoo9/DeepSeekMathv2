"""
src/training/distributed_setup.py

Distributed training setup with DeepSpeed support.
Handles single-GPU and multi-GPU configurations.
"""

import os
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union
# Pytorch imports
import torch
import torch.distributed as dist
TORCH_AVAILABLE = True

try:
    import deepspeed
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False
    deepspeed = None


# ============================================================================
# Distributed Environment Setup
# ============================================================================

def setup_distributed_environment() -> Dict[str, Any]:
    """
    Setup distributed training environment.
    
    Detects and configures:
    - Single GPU
    - Multi-GPU (single node)
    - Multi-node (if RANK/WORLD_SIZE set)
    
    Returns:
        Dictionary with distributed config
    """
    if not TORCH_AVAILABLE:
        return {
            'distributed': False,
            'local_rank': 0,
            'world_size': 1,
            'device': 'cpu'
        }
    
    # Check if distributed environment is set up
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # Multi-GPU/Multi-node via torch.distributed.launch
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        
        # Initialize process group
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl')
        
        torch.cuda.set_device(local_rank)
        
        return {
            'distributed': True,
            'local_rank': local_rank,
            'rank': rank,
            'world_size': world_size,
            'device': f'cuda:{local_rank}'
        }
    
    elif torch.cuda.is_available():
        # Single GPU
        return {
            'distributed': False,
            'local_rank': 0,
            'rank': 0,
            'world_size': 1,
            'device': 'cuda:0'
        }
    
    else:
        # CPU only
        return {
            'distributed': False,
            'local_rank': 0,
            'rank': 0,
            'world_size': 1,
            'device': 'cpu'
        }


def is_main_process() -> bool:
    """
    Check if current process is main process (rank 0).
    
    Returns:
        True if main process
    """
    if not TORCH_AVAILABLE:
        return True
    
    if not dist.is_initialized():
        return True
    
    return dist.get_rank() == 0


def cleanup_distributed() -> None:
    """Cleanup distributed training resources."""
    if TORCH_AVAILABLE and dist.is_initialized():
        dist.destroy_process_group()


# ============================================================================
# DeepSpeed Initialization
# ============================================================================

class DeepSpeedConfig:
    """
    DeepSpeed configuration manager.
    
    Args:
        config_path: Path to DeepSpeed config JSON
        zero_stage: ZeRO optimization stage (0, 1, 2, or 3)
        gradient_accumulation_steps: Steps to accumulate gradients
        train_micro_batch_size_per_gpu: Batch size per GPU
    """
    
    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        zero_stage: int = 3,
        gradient_accumulation_steps: int = 4,
        train_micro_batch_size_per_gpu: int = 1
    ):
        self.config_path = Path(config_path) if config_path else None
        self.zero_stage = zero_stage
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.train_micro_batch_size_per_gpu = train_micro_batch_size_per_gpu
        
        # Load config if provided, otherwise create default
        if self.config_path and self.config_path.exists():
            self.config = self._load_config()
        else:
            self.config = self._create_default_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load DeepSpeed config from file."""
        with open(self.config_path) as f:
            return json.load(f)
    
    def _create_default_config(self) -> Dict[str, Any]:
        """
        Create default DeepSpeed configuration.
        
        Returns:
            DeepSpeed config dictionary
        """
        config = {
            "train_micro_batch_size_per_gpu": self.train_micro_batch_size_per_gpu,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "steps_per_print": 100,
            "gradient_clipping": 1.0,
            
            # Mixed precision (bf16 for modern GPUs)
            "bf16": {
                "enabled": True
            },
            
            # Optimizer (offloaded to CPU for ZeRO-3)
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": 1e-5,
                    "betas": [0.9, 0.999],
                    "eps": 1e-8,
                    "weight_decay": 0.01
                }
            },
            
            # Scheduler
            "scheduler": {
                "type": "WarmupDecayLR",
                "params": {
                    "warmup_min_lr": 0,
                    "warmup_max_lr": 1e-5,
                    "warmup_num_steps": 100,
                    "total_num_steps": 10000
                }
            }
        }
        
        # Add ZeRO configuration
        if self.zero_stage > 0:
            config["zero_optimization"] = self._get_zero_config()
        
        return config
    
    def _get_zero_config(self) -> Dict[str, Any]:
        """
        Get ZeRO optimization config.
        
        Returns:
            ZeRO config dictionary
        """
        zero_config = {
            "stage": self.zero_stage,
            
            # Stage 3: Partition everything
            "contiguous_gradients": True,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "allgather_bucket_size": 5e8,
        }
        
        if self.zero_stage == 3:
            # Offload optimizer and parameters to CPU
            zero_config.update({
                "stage3_max_live_parameters": 1e9,
                "stage3_max_reuse_distance": 1e9,
                "stage3_prefetch_bucket_size": 5e8,
                "stage3_param_persistence_threshold": 1e6,
                
                # CPU offloading (essential for large models)
                "offload_optimizer": {
                    "device": "cpu",
                    "pin_memory": True
                },
                "offload_param": {
                    "device": "cpu",
                    "pin_memory": True
                }
            })
        
        return zero_config
    
    def save_config(self, output_path: Union[str, Path]) -> None:
        """
        Save config to file.
        
        Args:
            output_path: Path to save config
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        print(f"✓ DeepSpeed config saved to {output_path}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Get config as dictionary."""
        return self.config


# ============================================================================
# DeepSpeed Model Initialization
# ============================================================================

def initialize_deepspeed(
    model: Any,
    config_path: Optional[Union[str, Path]] = None,
    zero_stage: int = 3,
    gradient_accumulation_steps: int = 4,
    train_micro_batch_size_per_gpu: int = 1
) -> tuple:
    """
    Initialize model with DeepSpeed.
    
    Args:
        model: PyTorch model to wrap
        config_path: Path to DeepSpeed config JSON
        zero_stage: ZeRO optimization stage
        gradient_accumulation_steps: Gradient accumulation steps
        train_micro_batch_size_per_gpu: Batch size per GPU
        
    Returns:
        Tuple of (model_engine, optimizer, lr_scheduler, config)
    """
    if not DEEPSPEED_AVAILABLE:
        print("⚠ DeepSpeed not available, returning unwrapped model")
        return model, None, None, None
    
    # Create or load config
    ds_config = DeepSpeedConfig(
        config_path=config_path,
        zero_stage=zero_stage,
        gradient_accumulation_steps=gradient_accumulation_steps,
        train_micro_batch_size_per_gpu=train_micro_batch_size_per_gpu
    )
    
    # Initialize DeepSpeed
    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        config=ds_config.to_dict()
    )
    
    return model_engine, optimizer, lr_scheduler, ds_config


# ============================================================================
# Utility Functions
# ============================================================================

def get_effective_batch_size(
    train_micro_batch_size_per_gpu: int,
    gradient_accumulation_steps: int,
    world_size: int = 1
) -> int:
    """
    Calculate effective batch size.
    
    Formula: micro_batch * grad_accum * world_size
    
    Args:
        train_micro_batch_size_per_gpu: Batch size per GPU
        gradient_accumulation_steps: Gradient accumulation steps
        world_size: Number of GPUs
        
    Returns:
        Effective batch size
    """
    return train_micro_batch_size_per_gpu * gradient_accumulation_steps * world_size


def print_distributed_config(config: Dict[str, Any]) -> None:
    """
    Print distributed configuration.
    
    Args:
        config: Distributed config dictionary
    """
    print("\n" + "="*60)
    print("Distributed Training Configuration")
    print("="*60)
    print(f"Distributed:  {config['distributed']}")
    print(f"Device:       {config['device']}")
    print(f"Local Rank:   {config['local_rank']}")
    print(f"World Size:   {config['world_size']}")
    if config['distributed']:
        print(f"Global Rank:  {config['rank']}")
    print("="*60 + "\n")


def synchronize_processes() -> None:
    """Synchronize all distributed processes."""
    if TORCH_AVAILABLE and dist.is_initialized():
        dist.barrier()


def reduce_value(value: float, average: bool = True) -> float:
    """
    Reduce value across all processes.
    
    Args:
        value: Value to reduce
        average: Whether to average (True) or sum (False)
        
    Returns:
        Reduced value
    """
    if not TORCH_AVAILABLE or not dist.is_initialized():
        return value
    
    tensor = torch.tensor(value, device='cuda')
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    
    if average:
        tensor /= dist.get_world_size()
    
    return tensor.item()


# ============================================================================
# Export
# ============================================================================

__all__ = [
    'setup_distributed_environment',
    'is_main_process',
    'cleanup_distributed',
    'DeepSpeedConfig',
    'initialize_deepspeed',
    'get_effective_batch_size',
    'print_distributed_config',
    'synchronize_processes',
    'reduce_value',
]