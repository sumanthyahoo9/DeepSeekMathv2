"""
tests/test_grpo_trainer.py

Unit tests for GRPO trainer.
"""
import sys
from pathlib import Path
import pytest
import torch
from src.training.grpo_trainer import GRPOTrainer
sys.path.insert(0, str(Path(__file__).parent.parent))
TORCH_AVAILABLE = True


class TestGRPOTrainerInitialization:
    """Test GRPO trainer initialization."""
    
    def test_init_mock_mode(self):
        """Test initialization in mock mode (no PyTorch)."""
        trainer = GRPOTrainer(
            model=None,
            reward_fn=None,
            optimizer=None,
            group_size=4
        )
        
        if not TORCH_AVAILABLE:
            assert trainer.mock_mode is True
            assert trainer.device == 'cpu'
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_init_with_pytorch(self):
        """Test initialization with PyTorch."""
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters())
        
        def mock_reward_fn(inputs, outputs):
            return [0.5] * len(outputs)
        
        trainer = GRPOTrainer(
            model=model,
            reward_fn=mock_reward_fn,
            optimizer=optimizer,
            group_size=4,
            kl_coef=0.1,
            clip_range=0.2
        )
        
        assert trainer.mock_mode is False
        assert trainer.group_size == 4
        assert trainer.kl_coef == 0.1
        assert trainer.clip_range == 0.2


class TestAdvantageComputation:
    """Test advantage computation with group normalization."""
    
    def test_compute_advantages_simple(self):
        """Test basic advantage computation."""
        trainer = GRPOTrainer(
            model=None,
            reward_fn=None,
            optimizer=None,
            group_size=4
        )
        
        # Two groups of 4 samples
        rewards = [1.0, 0.5, 0.5, 0.0,  # Group 1: mean=0.5, std≈0.35
                   0.8, 0.6, 0.4, 0.2]  # Group 2: mean=0.5, std≈0.22
        
        advantages, stats = trainer.compute_advantages(rewards, group_size=4)
        
        # Check we got correct number of advantages
        assert len(advantages) == len(rewards)
        
        # Check statistics are computed
        assert 'mean_reward' in stats
        assert 'std_reward' in stats
        assert 'mean_advantage' in stats
        
        # Mean advantage should be close to 0 (normalized)
        assert abs(stats['mean_advantage']) < 0.1
    
    def test_compute_advantages_perfect_group(self):
        """Test with identical rewards in a group."""
        trainer = GRPOTrainer(
            model=None,
            reward_fn=None,
            optimizer=None,
            group_size=3
        )
        
        # All same rewards in group
        rewards = [0.5, 0.5, 0.5]
        
        advantages, stats = trainer.compute_advantages(rewards, group_size=3)
        
        # All advantages should be 0 (no variance)
        for adv in advantages:
            assert abs(adv) < 1e-6
    
    def test_compute_advantages_invalid_group_size(self):
        """Test error handling for invalid group size."""
        trainer = GRPOTrainer(
            model=None,
            reward_fn=None,
            optimizer=None,
            group_size=4
        )
        
        # 7 samples, not divisible by 4
        rewards = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        
        if not trainer.mock_mode:
            with pytest.raises(ValueError, match="divisible by group_size"):
                trainer.compute_advantages(rewards, group_size=4)
    
    def test_compute_advantages_ranking(self):
        """Test that advantages preserve ranking within groups."""
        trainer = GRPOTrainer(
            model=None,
            reward_fn=None,
            optimizer=None,
            group_size=4
        )
        
        # Group with clear ordering: [worst, bad, good, best]
        rewards = [0.0, 0.3, 0.7, 1.0]
        
        advantages, _ = trainer.compute_advantages(rewards, group_size=4)
        
        if not trainer.mock_mode:
            # Advantages should preserve ordering
            assert advantages[0] < advantages[1] < advantages[2] < advantages[3]
            
            # Best should have positive advantage
            assert advantages[3] > 0
            
            # Worst should have negative advantage
            assert advantages[0] < 0


class TestRewardComputation:
    """Test reward computation."""
    
    def test_compute_group_rewards_mock(self):
        """Test reward computation in mock mode."""
        trainer = GRPOTrainer(
            model=None,
            reward_fn=None,
            optimizer=None,
            group_size=4
        )
        
        inputs = ["Problem 1", "Problem 2"]
        outputs = ["Solution 1a", "Solution 1b", "Solution 2a", "Solution 2b"]
        
        rewards = trainer.compute_group_rewards(inputs, outputs)
        
        assert len(rewards) == len(outputs)
        assert all(isinstance(r, (int, float)) for r in rewards)
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_compute_group_rewards_custom_fn(self):
        """Test reward computation with custom function."""
        # Custom reward function: length-based
        def length_reward(inputs, outputs, **kwargs):
            return [len(output) / 100.0 for output in outputs]
        
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters())
        
        trainer = GRPOTrainer(
            model=model,
            reward_fn=length_reward,
            optimizer=optimizer,
            group_size=2
        )
        
        inputs = ["Problem"]
        outputs = ["Short", "Much longer solution"]
        
        rewards = trainer.compute_group_rewards(inputs, outputs)
        
        # Longer output should have higher reward
        assert rewards[1] > rewards[0]


class TestPolicyLoss:
    """Test policy loss computation."""
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_compute_policy_loss_basic(self):
        """Test basic policy loss computation."""
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters())
        
        trainer = GRPOTrainer(
            model=model,
            reward_fn=lambda i, o: [0.5] * len(o),
            optimizer=optimizer,
            group_size=2,
            clip_range=0.0  # No clipping for this test
        )
        
        # Create dummy log probs
        batch_size = 4
        seq_len = 10
        log_probs = torch.randn(batch_size, seq_len) * 0.1
        old_log_probs = torch.randn(batch_size, seq_len) * 0.1
        advantages = torch.tensor([1.0, -1.0, 0.5, -0.5])
        
        loss, stats = trainer.compute_policy_loss(
            log_probs, old_log_probs, advantages
        )
        
        assert isinstance(loss, torch.Tensor)
        assert 'policy_loss' in stats
        assert 'kl_div' in stats
        assert 'total_loss' in stats
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_compute_policy_loss_with_clipping(self):
        """Test policy loss with PPO-style clipping."""
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters())
        
        trainer = GRPOTrainer(
            model=model,
            reward_fn=lambda i, o: [0.5] * len(o),
            optimizer=optimizer,
            group_size=2,
            clip_range=0.2  # Enable clipping
        )
        
        # Create dummy data
        log_probs = torch.randn(4, 10) * 0.1
        old_log_probs = torch.randn(4, 10) * 0.1
        advantages = torch.tensor([1.0, -1.0, 0.5, -0.5])
        
        loss, stats = trainer.compute_policy_loss(
            log_probs, old_log_probs, advantages
        )
        
        # Check ratio is being tracked
        assert 'mean_ratio' in stats
        assert 'max_ratio' in stats
        assert 'min_ratio' in stats
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_compute_policy_loss_with_mask(self):
        """Test policy loss with attention mask."""
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters())
        
        trainer = GRPOTrainer(
            model=model,
            reward_fn=lambda i, o: [0.5] * len(o),
            optimizer=optimizer,
            group_size=2
        )
        
        # Create dummy data with mask
        batch_size, seq_len = 4, 10
        log_probs = torch.randn(batch_size, seq_len) * 0.1
        old_log_probs = torch.randn(batch_size, seq_len) * 0.1
        advantages = torch.tensor([1.0, -1.0, 0.5, -0.5])
        
        # Mask out last 3 tokens
        mask = torch.ones(batch_size, seq_len)
        mask[:, -3:] = 0
        
        loss, stats = trainer.compute_policy_loss(
            log_probs, old_log_probs, advantages, mask
        )
        
        assert isinstance(loss, torch.Tensor)


class TestTrainingStep:
    """Test training step execution."""
    
    def test_train_step_mock(self):
        """Test training step in mock mode."""
        trainer = GRPOTrainer(
            model=None,
            reward_fn=None,
            optimizer=None,
            group_size=2
        )
        
        batch = {
            'inputs': ['Problem'],
            'outputs': ['Solution A', 'Solution B'],
            'log_probs': None,
            'old_log_probs': None
        }
        
        stats = trainer.train_step(batch, step=0)
        
        assert 'loss' in stats or 'mean_reward' in stats
        assert all(isinstance(v, (int, float)) for v in stats.values())
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_train_step_with_pytorch(self):
        """Test training step with PyTorch."""
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters())
        
        def reward_fn(inputs, outputs):
            # Simple reward: longer is better
            return [len(o) / 10.0 for o in outputs]
        
        trainer = GRPOTrainer(
            model=model,
            reward_fn=reward_fn,
            optimizer=optimizer,
            group_size=2,
            gradient_accumulation_steps=1
        )
        
        # Create realistic batch
        batch = {
            'inputs': ['Problem'],
            'outputs': ['Short', 'Longer solution'],
            'log_probs': torch.randn(2, 10) * 0.1,
            'old_log_probs': torch.randn(2, 10) * 0.1
        }
        
        stats = trainer.train_step(batch, step=0)
        
        # Check all expected stats are present
        assert 'mean_reward' in stats
        assert 'policy_loss' in stats


class TestTrainingEpoch:
    """Test full epoch training."""
    
    def test_train_epoch_mock(self):
        """Test epoch training in mock mode."""
        trainer = GRPOTrainer(
            model=None,
            reward_fn=None,
            optimizer=None,
            group_size=2
        )
        
        # Mock dataloader
        dataloader = [
            {'inputs': ['P1'], 'outputs': ['S1', 'S2']},
            {'inputs': ['P2'], 'outputs': ['S3', 'S4']}
        ]
        
        stats = trainer.train_epoch(dataloader, epoch=0)
        
        assert 'epoch' in stats
        assert stats['epoch'] == 0


class TestCheckpointing:
    """Test checkpoint save/load."""
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_save_load_checkpoint(self, tmp_path):
        """Test saving and loading checkpoints."""
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters())
        
        trainer = GRPOTrainer(
            model=model,
            reward_fn=lambda i, o: [0.5] * len(o),
            optimizer=optimizer,
            group_size=2
        )
        
        # Save checkpoint
        checkpoint_path = tmp_path / "checkpoint.pt"
        trainer.save_checkpoint(checkpoint_path, epoch=5, custom_data="test")
        
        assert checkpoint_path.exists()
        
        # Load checkpoint
        data = trainer.load_checkpoint(checkpoint_path)
        
        assert data['epoch'] == 5
        assert data['custom_data'] == "test"


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_zero_std_group(self):
        """Test handling of zero standard deviation in group."""
        trainer = GRPOTrainer(
            model=None,
            reward_fn=None,
            optimizer=None,
            group_size=3
        )
        
        # All identical rewards (std = 0)
        rewards = [0.5, 0.5, 0.5]
        
        # Should not crash, advantages should be 0
        advantages, stats = trainer.compute_advantages(rewards)
        
        if not trainer.mock_mode:
            for adv in advantages:
                assert abs(adv) < 1e-6
    
    def test_single_group(self):
        """Test with single group."""
        trainer = GRPOTrainer(
            model=None,
            reward_fn=None,
            optimizer=None,
            group_size=4
        )
        
        rewards = [1.0, 0.7, 0.3, 0.0]
        
        advantages, stats = trainer.compute_advantages(rewards)
        
        assert len(advantages) == 4
        assert 'mean_reward' in stats


def test_grpo_algorithm_example():
    """
    End-to-end example demonstrating GRPO algorithm.
    
    This test shows how GRPO works with a concrete example.
    """
    trainer = GRPOTrainer(
        model=None,
        reward_fn=None,
        optimizer=None,
        group_size=4
    )
    
    # Example: 2 problems, 4 solutions each
    # Problem 1: Good, OK, OK, Bad
    # Problem 2: Perfect, Good, Bad, Terrible
    rewards = [
        1.0, 0.6, 0.5, 0.2,  # Problem 1
        1.0, 0.8, 0.3, 0.0   # Problem 2
    ]
    
    advantages, stats = trainer.compute_advantages(rewards, group_size=4)
    
    if not trainer.mock_mode:
        # Within each group:
        # - Best solutions should have positive advantage
        # - Worst solutions should have negative advantage
        
        # Problem 1: index 0 is best
        assert advantages[0] > 0  # Best in group 1
        assert advantages[3] < 0  # Worst in group 1
        
        # Problem 2: index 4 is best
        assert advantages[4] > 0  # Best in group 2
        assert advantages[7] < 0  # Worst in group 2
        
        print("\n=== GRPO Example ===")
        print(f"Rewards: {rewards}")
        print(f"Advantages: {[f'{a:.2f}' for a in advantages]}")
        print(f"Mean reward: {stats['mean_reward']:.2f}")
        print(f"Mean advantage: {stats['mean_advantage']:.2f}")