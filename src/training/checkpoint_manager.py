"""
src/training/checkpoint_manager.py

Checkpoint management for distributed training.
Handles saving, loading, and rotation of checkpoints.
"""

import shutil
from pathlib import Path
import json
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


# ============================================================================
# Checkpoint Manager
# ============================================================================

class CheckpointManager:
    """
    Manage training checkpoints with rotation.
    
    Features:
    - Save/load model, optimizer, scheduler
    - Automatic rotation (keep last N checkpoints)
    - Best checkpoint tracking
    - Resume from interruption
    
    Args:
        checkpoint_dir: Directory to store checkpoints
        max_checkpoints: Maximum checkpoints to keep (None = keep all)
        save_optimizer: Whether to save optimizer state
        save_scheduler: Whether to save scheduler state
    """
    
    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        max_checkpoints: Optional[int] = 3,
        save_optimizer: bool = True,
        save_scheduler: bool = True
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_checkpoints = max_checkpoints
        self.save_optimizer = save_optimizer
        self.save_scheduler = save_scheduler
        
        self.checkpoints = []  # List of (step, path) tuples
        self.best_checkpoint = None
        self.best_metric = None
    
    def save_checkpoint(
        self,
        model: Any,
        step: int,
        optimizer: Optional[Any] = None,
        scheduler: Optional[Any] = None,
        metrics: Optional[Dict[str, float]] = None,
        is_best: bool = False
    ) -> Path:
        """
        Save a checkpoint.
        
        Args:
            model: Model to save
            step: Training step
            optimizer: Optimizer (optional)
            scheduler: Scheduler (optional)
            metrics: Training metrics (optional)
            is_best: Whether this is the best checkpoint
            
        Returns:
            Path to saved checkpoint
        """
        checkpoint_name = f"checkpoint_step_{step}"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        checkpoint_path.mkdir(exist_ok=True)
        
        # Prepare checkpoint data
        checkpoint_data = {
            'step': step,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics or {}
        }
        
        # Save model
        model_path = checkpoint_path / "model.pt"
        if hasattr(model, 'module'):
            # Unwrap DDP/DeepSpeed wrapper
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()
        
        if TORCH_AVAILABLE:
            torch.save(state_dict, model_path)
        
        # Save optimizer
        if self.save_optimizer and optimizer is not None:
            optimizer_path = checkpoint_path / "optimizer.pt"
            if TORCH_AVAILABLE:
                torch.save(optimizer.state_dict(), optimizer_path)
        
        # Save scheduler
        if self.save_scheduler and scheduler is not None:
            scheduler_path = checkpoint_path / "scheduler.pt"
            if TORCH_AVAILABLE:
                torch.save(scheduler.state_dict(), scheduler_path)
        metadata_path = checkpoint_path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        # Track checkpoint
        self.checkpoints.append((step, checkpoint_path))
        
        # Handle best checkpoint
        if is_best:
            self._save_best_checkpoint(checkpoint_path, metrics)
        
        # Rotate old checkpoints
        self._rotate_checkpoints()
        
        print(f"✓ Checkpoint saved: {checkpoint_path}")
        return checkpoint_path
    
    def load_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        model: Any,
        optimizer: Optional[Any] = None,
        scheduler: Optional[Any] = None,
        strict: bool = True
    ) -> Dict[str, Any]:
        """
        Load a checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint directory
            model: Model to load into
            optimizer: Optimizer to load into (optional)
            scheduler: Scheduler to load into (optional)
            strict: Whether to strictly enforce state dict keys match
            
        Returns:
            Checkpoint metadata
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load model
        model_path = checkpoint_path / "model.pt"
        if model_path.exists() and TORCH_AVAILABLE:
            state_dict = torch.load(model_path, map_location='cpu')
            
            if hasattr(model, 'module'):
                model.module.load_state_dict(state_dict, strict=strict)
            else:
                model.load_state_dict(state_dict, strict=strict)
            
            print(f"✓ Model loaded from {model_path}")
        
        # Load optimizer
        if optimizer is not None:
            optimizer_path = checkpoint_path / "optimizer.pt"
            if optimizer_path.exists() and TORCH_AVAILABLE:
                optimizer.load_state_dict(torch.load(optimizer_path))
                print(f"✓ Optimizer loaded from {optimizer_path}")
        
        # Load scheduler
        if scheduler is not None:
            scheduler_path = checkpoint_path / "scheduler.pt"
            if scheduler_path.exists() and TORCH_AVAILABLE:
                scheduler.load_state_dict(torch.load(scheduler_path))
                print(f"✓ Scheduler loaded from {scheduler_path}")
        
        metadata_path = checkpoint_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
        else:
            metadata = {}
        
        return metadata
    
    def load_latest_checkpoint(
        self,
        model: Any,
        optimizer: Optional[Any] = None,
        scheduler: Optional[Any] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Load the latest checkpoint.
        
        Args:
            model: Model to load into
            optimizer: Optimizer (optional)
            scheduler: Scheduler (optional)
            
        Returns:
            Checkpoint metadata or None if no checkpoints
        """
        checkpoints = self._find_checkpoints()
        
        if not checkpoints:
            print("⚠ No checkpoints found")
            return None
        
        # Get latest
        latest = max(checkpoints, key=lambda x: x[0])
        latest_path = latest[1]
        
        print(f"Loading latest checkpoint from step {latest[0]}")
        return self.load_checkpoint(latest_path, model, optimizer, scheduler)
    
    def load_best_checkpoint(
        self,
        model: Any,
        optimizer: Optional[Any] = None,
        scheduler: Optional[Any] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Load the best checkpoint.
        
        Args:
            model: Model to load into
            optimizer: Optimizer (optional)
            scheduler: Scheduler (optional)
            
        Returns:
            Checkpoint metadata or None if no best checkpoint
        """
        best_path = self.checkpoint_dir / "best_checkpoint"
        
        if not best_path.exists():
            print("⚠ No best checkpoint found")
            return None
        
        print("Loading best checkpoint")
        return self.load_checkpoint(best_path, model, optimizer, scheduler)
    
    def _save_best_checkpoint(
        self,
        checkpoint_path: Path,
        metrics: Optional[Dict[str, float]]
    ) -> None:
        """
        Save checkpoint as best checkpoint.
        
        Args:
            checkpoint_path: Source checkpoint path
            metrics: Metrics for this checkpoint
        """
        best_path = self.checkpoint_dir / "best_checkpoint"
        
        # Remove old best if exists
        if best_path.exists():
            shutil.rmtree(best_path)
        
        # Copy current as best
        shutil.copytree(checkpoint_path, best_path)
        
        self.best_checkpoint = checkpoint_path
        self.best_metric = metrics
        
        print(f"✓ Saved as best checkpoint with metrics: {metrics}")
    
    def _rotate_checkpoints(self) -> None:
        """Remove old checkpoints beyond max_checkpoints."""
        if self.max_checkpoints is None:
            return
        
        # Sort by step
        self.checkpoints.sort(key=lambda x: x[0])
        
        # Remove oldest if too many
        while len(self.checkpoints) > self.max_checkpoints:
            old_step, old_path = self.checkpoints.pop(0)
            
            # Don't remove if it's the best checkpoint
            if old_path == self.best_checkpoint:
                continue
            
            if old_path.exists():
                shutil.rmtree(old_path)
                print(f"✓ Removed old checkpoint: step {old_step}")
    
    def _find_checkpoints(self) -> List[tuple]:
        """
        Find all checkpoints in directory.
        
        Returns:
            List of (step, path) tuples
        """
        checkpoints = []
        
        for path in self.checkpoint_dir.iterdir():
            if path.is_dir() and path.name.startswith("checkpoint_step_"):
                try:
                    step = int(path.name.split("_")[-1])
                    checkpoints.append((step, path))
                except ValueError:
                    continue
        
        return checkpoints
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """
        List all available checkpoints.
        
        Returns:
            List of checkpoint info dictionaries
        """
        checkpoints = self._find_checkpoints()
        
        checkpoint_info = []
        for step, path in checkpoints:
            metadata_path = path / "metadata.json"
            
            info = {'step': step, 'path': str(path)}
            
            if metadata_path.exists():
                with open(metadata_path) as f:
                    metadata = json.load(f)
                    info.update(metadata)
            
            checkpoint_info.append(info)
        
        return sorted(checkpoint_info, key=lambda x: x['step'])


# ============================================================================
# Utility Functions
# ============================================================================

def get_checkpoint_info(checkpoint_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Get information about a checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint
        
    Returns:
        Checkpoint information dictionary
    """
    checkpoint_path = Path(checkpoint_path)
    
    metadata_path = checkpoint_path / "metadata.json"
    
    if metadata_path.exists():
        with open(metadata_path) as f:
            return json.load(f)
    
    return {}


# ============================================================================
# Export
# ============================================================================

__all__ = [
    'CheckpointManager',
    'get_checkpoint_info',
]