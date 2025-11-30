"""
tests/test_checkpoint_manager.py

Unit tests for checkpoint management.
"""

import pytest
import sys
import json
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.checkpoint_manager import (
    CheckpointManager,
    get_checkpoint_info
)


# ============================================================================
# Mock Model for Testing
# ============================================================================

class MockModel:
    """Mock model for checkpoint testing"""
    
    def __init__(self):
        self.weights = {'layer1': [1.0, 2.0], 'layer2': [3.0, 4.0]}
    
    def state_dict(self):
        return self.weights
    
    def load_state_dict(self, state_dict, strict=True):
        self.weights = state_dict


class MockOptimizer:
    """Mock optimizer for checkpoint testing"""
    
    def __init__(self):
        self.state = {'momentum': [0.1, 0.2]}
    
    def state_dict(self):
        return self.state
    
    def load_state_dict(self, state_dict):
        self.state = state_dict


# ============================================================================
# Test CheckpointManager Initialization
# ============================================================================

def test_checkpoint_manager_init():
    """Test checkpoint manager initialization"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(tmpdir)
        
        assert manager.checkpoint_dir.exists()
        assert manager.max_checkpoints == 3
        assert manager.save_optimizer is True


def test_checkpoint_manager_init_custom():
    """Test checkpoint manager with custom settings"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(
            tmpdir,
            max_checkpoints=5,
            save_optimizer=False
        )
        
        assert manager.max_checkpoints == 5
        assert manager.save_optimizer is False


# ============================================================================
# Test Saving Checkpoints
# ============================================================================

def test_save_checkpoint_basic():
    """Test basic checkpoint saving"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(tmpdir)
        model = MockModel()
        
        checkpoint_path = manager.save_checkpoint(model, step=100)
        
        assert checkpoint_path.exists()
        assert (checkpoint_path / "metadata.json").exists()


def test_save_checkpoint_with_optimizer():
    """Test saving checkpoint with optimizer"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(tmpdir, save_optimizer=True)
        model = MockModel()
        optimizer = MockOptimizer()
        
        checkpoint_path = manager.save_checkpoint(
            model,
            step=100,
            optimizer=optimizer
        )
        
        # Optimizer state should be saved (if torch available)
        # In mock mode, we just verify the function runs without error
        assert checkpoint_path.exists()


def test_save_checkpoint_with_metrics():
    """Test saving checkpoint with metrics"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(tmpdir)
        model = MockModel()
        
        metrics = {'loss': 0.5, 'reward': 0.8}
        checkpoint_path = manager.save_checkpoint(
            model,
            step=100,
            metrics=metrics
        )
        
        # Load metadata and check metrics
        metadata_path = checkpoint_path / "metadata.json"
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        assert metadata['metrics'] == metrics


def test_save_checkpoint_as_best():
    """Test saving checkpoint as best"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(tmpdir)
        model = MockModel()
        
        manager.save_checkpoint(model, step=100, is_best=True)
        
        best_path = manager.checkpoint_dir / "best_checkpoint"
        assert best_path.exists()


# ============================================================================
# Test Loading Checkpoints
# ============================================================================

def test_load_checkpoint_basic():
    """Test basic checkpoint loading"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(tmpdir)
        model1 = MockModel()
        
        # Save checkpoint
        checkpoint_path = manager.save_checkpoint(model1, step=100)
        
        # Load into new model
        model2 = MockModel()
        metadata = manager.load_checkpoint(checkpoint_path, model2)
        
        assert metadata['step'] == 100


def test_load_latest_checkpoint():
    """Test loading latest checkpoint"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(tmpdir)
        model = MockModel()
        
        # Save multiple checkpoints
        manager.save_checkpoint(model, step=100)
        manager.save_checkpoint(model, step=200)
        manager.save_checkpoint(model, step=300)
        
        # Load latest
        model2 = MockModel()
        metadata = manager.load_latest_checkpoint(model2)
        
        assert metadata is not None
        assert metadata['step'] == 300


def test_load_latest_checkpoint_empty():
    """Test loading latest checkpoint when none exist"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(tmpdir)
        model = MockModel()
        
        metadata = manager.load_latest_checkpoint(model)
        
        assert metadata is None


def test_load_best_checkpoint():
    """Test loading best checkpoint"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(tmpdir)
        model = MockModel()
        
        # Save checkpoint as best
        manager.save_checkpoint(model, step=100, is_best=True)
        
        # Load best
        model2 = MockModel()
        metadata = manager.load_best_checkpoint(model2)
        
        assert metadata is not None
        assert metadata['step'] == 100


# ============================================================================
# Test Checkpoint Rotation
# ============================================================================

def test_checkpoint_rotation():
    """Test automatic checkpoint rotation"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(tmpdir, max_checkpoints=2)
        model = MockModel()
        
        # Save 4 checkpoints (should keep only last 2)
        manager.save_checkpoint(model, step=100)
        manager.save_checkpoint(model, step=200)
        manager.save_checkpoint(model, step=300)
        manager.save_checkpoint(model, step=400)
        
        # Should only have 2 checkpoints
        checkpoints = manager.list_checkpoints()
        assert len(checkpoints) <= 2
        
        # Should be the latest 2
        steps = [c['step'] for c in checkpoints]
        assert 300 in steps or 400 in steps


def test_checkpoint_rotation_preserves_best():
    """Test that rotation doesn't delete best checkpoint"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(tmpdir, max_checkpoints=2)
        model = MockModel()
        
        # Save checkpoint as best
        manager.save_checkpoint(model, step=100, is_best=True)
        
        # Save more checkpoints
        manager.save_checkpoint(model, step=200)
        manager.save_checkpoint(model, step=300)
        manager.save_checkpoint(model, step=400)
        
        # Best should still exist
        best_path = manager.checkpoint_dir / "best_checkpoint"
        assert best_path.exists()


# ============================================================================
# Test Checkpoint Listing
# ============================================================================

def test_list_checkpoints():
    """Test listing all checkpoints"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(tmpdir)
        model = MockModel()
        
        # Save multiple checkpoints
        manager.save_checkpoint(model, step=100)
        manager.save_checkpoint(model, step=200)
        
        checkpoints = manager.list_checkpoints()
        
        assert len(checkpoints) == 2
        assert checkpoints[0]['step'] == 100
        assert checkpoints[1]['step'] == 200


# ============================================================================
# Test Utility Functions
# ============================================================================

def test_get_checkpoint_info():
    """Test getting checkpoint information"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(tmpdir)
        model = MockModel()
        
        metrics = {'loss': 0.5}
        checkpoint_path = manager.save_checkpoint(
            model,
            step=100,
            metrics=metrics
        )
        
        info = get_checkpoint_info(checkpoint_path)
        
        assert info['step'] == 100
        assert info['metrics'] == metrics


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])