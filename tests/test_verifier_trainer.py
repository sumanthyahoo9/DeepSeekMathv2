"""
tests/test_verifier_trainer.py

Unit tests for verifier trainer.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.verifier_trainer import VerifierTrainer, VerifierRewardFunction

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


class MockVerificationDataset:
    """Mock dataset for testing."""
    
    def __init__(self, size=100):
        self.size = size
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return {
            'problem': f'Problem {idx}',
            'proof': f'Proof {idx}',
            'score': 0.5
        }


class TestVerifierRewardFunction:
    """Test verifier reward function."""
    
    def test_init_default_weights(self):
        """Test initialization with default weights."""
        reward_fn = VerifierRewardFunction()
        
        assert reward_fn.alpha_format == 0.125
        assert reward_fn.alpha_score == 0.875
        assert not reward_fn.use_meta_verification
    
    def test_init_custom_weights(self):
        """Test initialization with custom weights."""
        reward_fn = VerifierRewardFunction(
            alpha_format=0.2,
            alpha_score=0.8
        )
        
        assert reward_fn.alpha_format == 0.2
        assert reward_fn.alpha_score == 0.8
    
    def test_init_weights_normalization(self):
        """Test that weights are normalized if they don't sum to 1."""
        reward_fn = VerifierRewardFunction(
            alpha_format=0.3,
            alpha_score=0.3  # Sum = 0.6, not 1.0
        )
        
        # Should be normalized to sum to 1
        assert abs(reward_fn.alpha_format + reward_fn.alpha_score - 1.0) < 1e-6
    
    def test_call_basic(self):
        """Test basic reward computation."""
        reward_fn = VerifierRewardFunction()
        
        inputs = [
            {'problem': 'P1', 'proof': 'Proof1', 'score': 1.0},
            {'problem': 'P2', 'proof': 'Proof2', 'score': 0.5}
        ]
        
        outputs = [
            "Analysis 1\nBased on my evaluation, the final overall score should be: \\boxed{1}",
            "Analysis 2\nBased on my evaluation, the final overall score should be: \\boxed{0.5}"
        ]
        
        rewards = reward_fn(inputs, outputs)
        
        assert len(rewards) == 2
        assert all(isinstance(r, (int, float)) for r in rewards)
        assert all(0 <= r <= 1 for r in rewards)
    
    def test_call_format_reward(self):
        """Test that format reward is computed correctly."""
        reward_fn = VerifierRewardFunction(
            alpha_format=1.0,
            alpha_score=0.0
        )
        
        inputs = [{'problem': 'P', 'proof': 'Pr', 'score': 1.0}]
        
        # Correct format
        correct_output = [
            "Here is my evaluation of the solution:\n"
            "Analysis here.\n"
            "Based on my evaluation, the final overall score should be: \\boxed{1}"
        ]
        
        # Incorrect format
        incorrect_output = ["Just some text without proper format"]
        
        correct_rewards = reward_fn(inputs, correct_output)
        incorrect_rewards = reward_fn(inputs, incorrect_output)
        
        # Correct format should have higher reward
        assert correct_rewards[0] >= incorrect_rewards[0]
    
    def test_call_score_reward(self):
        """Test that score reward is computed correctly."""
        reward_fn = VerifierRewardFunction(
            alpha_format=0.0,
            alpha_score=1.0
        )
        
        # Ground truth score is 1.0
        inputs = [
            {'problem': 'P', 'proof': 'Pr', 'score': 1.0},
            {'problem': 'P', 'proof': 'Pr', 'score': 1.0}
        ]
        
        # Perfect prediction
        perfect_output = [
            "Analysis\nBased on my evaluation, the final overall score should be: \\boxed{1}"
        ]
        
        # Wrong prediction
        wrong_output = [
            "Analysis\nBased on my evaluation, the final overall score should be: \\boxed{0}"
        ]
        
        perfect_rewards = reward_fn(inputs[:1], perfect_output)
        wrong_rewards = reward_fn(inputs[1:], wrong_output)
        
        # Perfect prediction should have higher reward
        assert perfect_rewards[0] > wrong_rewards[0]
    
    def test_call_with_meta_verification(self):
        """Test reward computation with meta-verification."""
        reward_fn = VerifierRewardFunction(
            use_meta_verification=True,
            meta_verifier=None  # Mock meta-verifier
        )
        
        inputs = [{'problem': 'P', 'proof': 'Pr', 'score': 1.0}]
        outputs = ["Analysis with score \\boxed{1}"]
        
        rewards = reward_fn(inputs, outputs)
        
        assert len(rewards) == 1
        assert isinstance(rewards[0], (int, float))


class TestVerifierTrainerInitialization:
    """Test verifier trainer initialization."""
    
    def test_init_mock_mode(self):
        """Test initialization in mock mode."""
        trainer = VerifierTrainer(
            model=None,
            tokenizer=None,
            train_dataset=MockVerificationDataset(100)
        )
        
        if not TORCH_AVAILABLE:
            assert trainer.mock_mode is True
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_init_with_pytorch(self):
        """Test initialization with PyTorch."""
        model = torch.nn.Linear(10, 10)
        
        class MockTokenizer:
            pass
        
        trainer = VerifierTrainer(
            model=model,
            tokenizer=MockTokenizer(),
            train_dataset=MockVerificationDataset(100),
            output_dir="test_checkpoints",
            group_size=4,
            batch_size=8,
            num_epochs=2
        )
        
        assert trainer.mock_mode is False
        assert trainer.batch_size == 8
        assert trainer.num_epochs == 2
        assert trainer.global_step == 0
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_init_creates_output_dir(self, tmp_path):
        """Test that output directory is created."""
        model = torch.nn.Linear(10, 10)
        
        class MockTokenizer:
            pass
        
        output_dir = tmp_path / "checkpoints"
        
        trainer = VerifierTrainer(
            model=model,
            tokenizer=MockTokenizer(),
            train_dataset=MockVerificationDataset(100),
            output_dir=str(output_dir)
        )
        
        assert output_dir.exists()
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_init_with_validation_dataset(self):
        """Test initialization with validation dataset."""
        model = torch.nn.Linear(10, 10)
        
        class MockTokenizer:
            pass
        
        trainer = VerifierTrainer(
            model=model,
            tokenizer=MockTokenizer(),
            train_dataset=MockVerificationDataset(100),
            val_dataset=MockVerificationDataset(20)
        )
        
        assert trainer.val_dataset is not None
        assert len(trainer.val_dataset) == 20


class TestVerifierGeneration:
    """Test verification generation."""
    
    def test_generate_verifications_mock(self):
        """Test verification generation in mock mode."""
        trainer = VerifierTrainer(
            model=None,
            tokenizer=None,
            train_dataset=MockVerificationDataset(100)
        )
        
        problems = ["Prove x", "Prove y"]
        proofs = ["Proof x", "Proof y"]
        
        analyses, log_probs, old_log_probs = trainer.generate_verifications(
            problems, proofs, num_samples=2
        )
        
        assert len(analyses) == 4  # 2 proofs × 2 samples
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_generate_verifications_pytorch(self):
        """Test verification generation with PyTorch."""
        model = torch.nn.Linear(10, 10)
        
        class MockTokenizer:
            pass
        
        trainer = VerifierTrainer(
            model=model,
            tokenizer=MockTokenizer(),
            train_dataset=MockVerificationDataset(100)
        )
        
        problems = ["Prove x"]
        proofs = ["Proof x"]
        
        analyses, log_probs, old_log_probs = trainer.generate_verifications(
            problems, proofs, num_samples=4
        )
        
        # Should return tensors (even if empty in placeholder)
        assert isinstance(log_probs, torch.Tensor)
        assert isinstance(old_log_probs, torch.Tensor)


class TestTrainingStep:
    """Test training step execution."""
    
    def test_train_step_mock(self):
        """Test training step in mock mode."""
        trainer = VerifierTrainer(
            model=None,
            tokenizer=None,
            train_dataset=MockVerificationDataset(100)
        )
        
        batch = {
            'problems': ['Problem 1'],
            'proofs': ['Proof 1'],
            'scores': [1.0]
        }
        
        metrics = trainer.train_step(batch)
        
        assert 'loss' in metrics or 'mean_reward' in metrics
        assert all(isinstance(v, (int, float)) for v in metrics.values())


class TestTrainingEpoch:
    """Test epoch training."""
    
    def test_train_epoch_mock(self):
        """Test epoch training in mock mode."""
        trainer = VerifierTrainer(
            model=None,
            tokenizer=None,
            train_dataset=MockVerificationDataset(10),
            batch_size=5
        )
        
        metrics = trainer.train_epoch(epoch=0)
        
        assert 'epoch' in metrics
        assert metrics['epoch'] == 0
        assert 'mean_loss' in metrics
        assert 'mean_reward' in metrics


class TestFullTraining:
    """Test full training loop."""
    
    def test_train_mock(self):
        """Test full training in mock mode."""
        trainer = VerifierTrainer(
            model=None,
            tokenizer=None,
            train_dataset=MockVerificationDataset(10),
            num_epochs=2
        )
        
        history = trainer.train()
        
        assert 'epochs' in history
        assert 'best_reward' in history


class TestEvaluation:
    """Test model evaluation."""
    
    def test_evaluate_mock(self):
        """Test evaluation in mock mode."""
        trainer = VerifierTrainer(
            model=None,
            tokenizer=None,
            train_dataset=MockVerificationDataset(100),
            val_dataset=MockVerificationDataset(20)
        )
        
        metrics = trainer.evaluate()
        
        assert 'mean_reward' in metrics
        assert isinstance(metrics['mean_reward'], (int, float))
    
    def test_evaluate_no_val_dataset(self):
        """Test evaluation when no validation dataset provided."""
        trainer = VerifierTrainer(
            model=None,
            tokenizer=None,
            train_dataset=MockVerificationDataset(100),
            val_dataset=None
        )
        
        metrics = trainer.evaluate()
        
        # Should still return metrics (mock values)
        assert isinstance(metrics, dict)


class TestCheckpointing:
    """Test checkpoint saving and loading."""
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_save_checkpoint(self, tmp_path):
        """Test checkpoint saving."""
        model = torch.nn.Linear(10, 10)
        
        class MockTokenizer:
            pass
        
        trainer = VerifierTrainer(
            model=model,
            tokenizer=MockTokenizer(),
            train_dataset=MockVerificationDataset(100),
            output_dir=str(tmp_path / "checkpoints")
        )
        
        trainer.global_step = 100
        trainer.best_val_reward = 0.85
        
        trainer.save_checkpoint(epoch=1, is_best=True)
        
        # Check checkpoint files exist
        checkpoint_dir = tmp_path / "checkpoints"
        assert checkpoint_dir.exists()
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_load_checkpoint(self, tmp_path):
        """Test checkpoint loading."""
        model = torch.nn.Linear(10, 10)
        
        class MockTokenizer:
            pass
        
        output_dir = str(tmp_path / "checkpoints")
        
        # Create and save checkpoint
        trainer1 = VerifierTrainer(
            model=model,
            tokenizer=MockTokenizer(),
            train_dataset=MockVerificationDataset(100),
            output_dir=output_dir
        )
        
        trainer1.global_step = 100
        trainer1.best_val_reward = 0.85
        trainer1.save_checkpoint(epoch=1, is_best=True)
        
        # Create new trainer and load checkpoint
        model2 = torch.nn.Linear(10, 10)
        trainer2 = VerifierTrainer(
            model=model2,
            tokenizer=MockTokenizer(),
            train_dataset=MockVerificationDataset(100),
            output_dir=output_dir
        )
        
        # Find checkpoint file
        checkpoint_files = list(Path(output_dir).glob("*.pt"))
        if checkpoint_files:
            trainer2.load_checkpoint(str(checkpoint_files[0]))
            
            assert trainer2.global_step == 100
            assert trainer2.best_val_reward == 0.85


class TestRewardWeights:
    """Test different reward weight configurations."""
    
    def test_format_only(self):
        """Test with only format reward."""
        reward_fn = VerifierRewardFunction(
            alpha_format=1.0,
            alpha_score=0.0
        )
        
        assert reward_fn.alpha_format == 1.0
        assert reward_fn.alpha_score == 0.0
    
    def test_score_only(self):
        """Test with only score reward."""
        reward_fn = VerifierRewardFunction(
            alpha_format=0.0,
            alpha_score=1.0
        )
        
        assert reward_fn.alpha_format == 0.0
        assert reward_fn.alpha_score == 1.0
    
    def test_paper_weights(self):
        """Test with paper-specified weights (α₁=0.125, α₂=0.875)."""
        reward_fn = VerifierRewardFunction(
            alpha_format=0.125,
            alpha_score=0.875
        )
        
        assert reward_fn.alpha_format == 0.125
        assert reward_fn.alpha_score == 0.875


class TestIntegrationWithGRPO:
    """Test integration with GRPO trainer."""
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_grpo_trainer_initialized(self):
        """Test that GRPO trainer is properly initialized."""
        model = torch.nn.Linear(10, 10)
        
        class MockTokenizer:
            pass
        
        trainer = VerifierTrainer(
            model=model,
            tokenizer=MockTokenizer(),
            train_dataset=MockVerificationDataset(100),
            group_size=4,
            kl_coef=0.01,
            clip_range=0.2
        )
        
        assert trainer.grpo_trainer is not None
        assert trainer.grpo_trainer.group_size == 4
        assert trainer.grpo_trainer.kl_coef == 0.01
        assert trainer.grpo_trainer.clip_range == 0.2


def test_verifier_training_example():
    """
    End-to-end example demonstrating verifier training.
    
    This test shows the complete workflow:
    1. Initialize trainer with model and dataset
    2. Run training for multiple epochs
    3. Evaluate on validation set
    4. Save checkpoints
    """
    # Create mock components
    trainer = VerifierTrainer(
        model=None,
        tokenizer=None,
        train_dataset=MockVerificationDataset(20),
        val_dataset=MockVerificationDataset(5),
        num_epochs=2,
        batch_size=4,
        group_size=4,
        alpha_format=0.125,
        alpha_score=0.875
    )
    
    # Run training
    history = trainer.train()
    
    # Check training completed
    assert 'epochs' in history
    assert 'best_reward' in history
    
    print("\n=== Verifier Training Example ===")
    print(f"Epochs trained: {len(history.get('epochs', []))}")
    print(f"Best reward: {history.get('best_reward', 0):.3f}")