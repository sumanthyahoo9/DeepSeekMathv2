"""
src/training/verifier_trainer.py

Trainer for proof verification models using GRPO.

Trains a verifier to:
1. Identify issues in mathematical proofs
2. Assign scores (0, 0.5, 1) based on proof quality
3. Follow evaluation rubrics faithfully

Uses GRPO with combined rewards:
- R_format: Output format correctness
- R_score: Score prediction accuracy
- R_meta: Meta-verification quality (optional)
"""
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging

try:
    import torch
    from torch.utils.data import DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    DataLoader = None

from src.training.grpo_trainer import GRPOTrainer
from src.training.reward_functions import compute_format_reward, compute_score_reward
from src.utils.metrics import GRPOMetricsTracker
from src.training.checkpoint_manager import CheckpointManager

logger = logging.getLogger(__name__)


class VerifierRewardFunction:
    """
    Reward function for verifier training.
    
    Combines three reward components:
    - R_format: Ensures proper output format
    - R_score: Rewards accurate score prediction
    - R_meta: Rewards faithful issue identification (optional)
    """
    
    def __init__(
        self,
        alpha_format: float = 0.125,
        alpha_score: float = 0.875,
        use_meta_verification: bool = False,
        meta_verifier: Optional[Any] = None
    ):
        """
        Initialize verifier reward function.
        
        Args:
            alpha_format: Weight for format reward (default: 0.125)
            alpha_score: Weight for score reward (default: 0.875)
            use_meta_verification: Whether to use meta-verification
            meta_verifier: Meta-verifier model for R_meta (if using)
        """
        self.alpha_format = alpha_format
        self.alpha_score = alpha_score
        self.use_meta_verification = use_meta_verification
        self.meta_verifier = meta_verifier
        
        # Validate weights sum to 1
        if not use_meta_verification:
            total = alpha_format + alpha_score
            if abs(total - 1.0) > 1e-6:
                logger.warning(
                    f"Reward weights sum to {total}, not 1.0. "
                    f"Normalizing: α_format={alpha_format/total:.3f}, "
                    f"α_score={alpha_score/total:.3f}"
                )
                self.alpha_format = alpha_format / total
                self.alpha_score = alpha_score / total
    
    def __call__(
        self,
        inputs: List[Dict[str, str]],
        outputs: List[str],
        **kwargs
    ) -> List[float]:
        """
        Compute rewards for verifier outputs.
        
        Args:
            inputs: List of dicts with 'problem' and 'proof' keys
            outputs: List of verifier analyses (proof evaluations)
            
        Returns:
            List of reward values (one per output)
        """
        rewards = []
        
        for inp, output in zip(inputs, outputs):
            problem = inp['problem']
            proof = inp['proof']
            true_score = inp.get('score', None)
            
            # Compute format reward
            r_format = compute_format_reward(output)
            
            # Compute score reward (if ground truth available)
            if true_score is not None:
                r_score = compute_score_reward(output, true_score)
            else:
                r_score = 0.0
            
            # Compute meta-verification reward (if enabled)
            if self.use_meta_verification and self.meta_verifier is not None:
                r_meta = self._compute_meta_reward(problem, proof, output)
            else:
                r_meta = 1.0  # No penalty if not using meta-verification
            
            # Combined reward
            if self.use_meta_verification:
                reward = r_format * r_score * r_meta
            else:
                reward = self.alpha_format * r_format + self.alpha_score * r_score
            
            rewards.append(reward)
        
        return rewards
    
    def _compute_meta_reward(
        self,
        problem: str,
        proof: str,
        analysis: str
    ) -> float:
        """
        Compute meta-verification reward using meta-verifier.
        
        Args:
            problem: The mathematical problem
            proof: The proof to be verified
            analysis: The verifier's analysis of the proof
            
        Returns:
            Meta-verification quality score (0, 0.5, or 1)
        """
        if self.meta_verifier is None:
            return 1.0
        
        # Use meta-verifier to assess analysis quality
        # This would call the actual meta-verifier model
        # For now, return placeholder
        return 1.0


class VerifierTrainer:
    """
    Trainer for proof verification models.
    
    Uses GRPO to train verifiers that:
    1. Identify issues in proofs without reference solutions
    2. Assign accurate quality scores
    3. Provide faithful analysis (via meta-verification)
    """
    
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        train_dataset: Any,
        val_dataset: Optional[Any] = None,
        output_dir: str = "checkpoints/verifier",
        # GRPO parameters
        group_size: int = 4,
        kl_coef: float = 0.01,
        clip_range: float = 0.2,
        # Reward parameters
        alpha_format: float = 0.125,
        alpha_score: float = 0.875,
        use_meta_verification: bool = False,
        meta_verifier: Optional[Any] = None,
        # Training parameters
        learning_rate: float = 1e-6,
        batch_size: int = 8,
        gradient_accumulation_steps: int = 4,
        num_epochs: int = 3,
        max_grad_norm: float = 1.0,
        warmup_steps: int = 100,
        logging_steps: int = 10,
        eval_steps: int = 500,
        save_steps: int = 1000,
        # Generation parameters
        max_new_tokens: int = 2048,
        temperature: float = 0.8,
        top_p: float = 0.95,
        device: Optional[str] = None
    ):
        """
        Initialize verifier trainer.
        
        Args:
            model: The verifier model to train
            tokenizer: Tokenizer for the model
            train_dataset: Training dataset (VerificationDataset)
            val_dataset: Validation dataset (optional)
            output_dir: Directory for checkpoints and logs
            group_size: Number of samples per problem for GRPO
            kl_coef: KL divergence penalty coefficient
            clip_range: PPO-style clipping range
            alpha_format: Weight for format reward
            alpha_score: Weight for score reward
            use_meta_verification: Whether to use meta-verification
            meta_verifier: Meta-verifier model (if using)
            learning_rate: Learning rate for optimizer
            batch_size: Batch size (number of problems)
            gradient_accumulation_steps: Steps to accumulate gradients
            num_epochs: Number of training epochs
            max_grad_norm: Maximum gradient norm for clipping
            warmup_steps: Number of warmup steps for LR scheduler
            logging_steps: Log every N steps
            eval_steps: Evaluate every N steps
            save_steps: Save checkpoint every N steps
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling threshold
            device: Device to use ('cuda' or 'cpu')
        """
        if not TORCH_AVAILABLE:
            self.mock_mode = True
            logger.warning("PyTorch not available, running in mock mode")
            return
        
        self.mock_mode = False
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.1
        )
        
        # Initialize learning rate scheduler
        total_steps = (len(train_dataset) // batch_size) * num_epochs
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=learning_rate * 0.1
        )
        
        # Initialize reward function
        self.reward_fn = VerifierRewardFunction(
            alpha_format=alpha_format,
            alpha_score=alpha_score,
            use_meta_verification=use_meta_verification,
            meta_verifier=meta_verifier
        )
        
        # Initialize GRPO trainer
        self.grpo_trainer = GRPOTrainer(
            model=self.model,
            reward_fn=self.reward_fn,
            optimizer=self.optimizer,
            group_size=group_size,
            kl_coef=kl_coef,
            clip_range=clip_range,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_grad_norm=max_grad_norm,
            device=self.device
        )
        
        # Initialize metrics tracker
        self.metrics_tracker = GRPOMetricsTracker()
        
        # Initialize checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=str(self.output_dir),
            max_checkpoints=3,
            metric_for_best='mean_reward'
        )
        
        # Store training parameters
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.logging_steps = logging_steps
        self.eval_steps = eval_steps
        self.save_steps = save_steps
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        
        # Training state
        self.global_step = 0
        self.best_val_reward = -float('inf')
    
    def generate_verifications(
        self,
        problems: List[str],
        proofs: List[str],
        num_samples: int = 1
    ) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
        """
        Generate verification analyses for given proofs.
        
        Args:
            problems: List of problem statements
            proofs: List of proofs to verify
            num_samples: Number of analyses to generate per proof
            
        Returns:
            Tuple of (analyses, log_probs, old_log_probs)
        """
        if self.mock_mode:
            analyses = [f"Mock analysis {i}" for i in range(len(proofs) * num_samples)]
            log_probs = torch.randn(len(analyses), 100)
            old_log_probs = log_probs.clone()
            return analyses, log_probs, old_log_probs
        
        # TODO: Implement actual generation logic
        # This would format prompts, generate with model, compute log probs
        analyses = []
        log_probs_list = []
        
        # Placeholder implementation
        return analyses, torch.tensor([]), torch.tensor([])
    
    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        Perform one training step.
        
        Args:
            batch: Batch containing problems, proofs, and ground truth scores
            
        Returns:
            Dictionary of training metrics
        """
        if self.mock_mode:
            return {'loss': 0.5, 'mean_reward': 0.6}
        
        problems = batch['problems']
        proofs = batch['proofs']
        scores = batch['scores']
        
        # Generate multiple verification analyses per proof
        analyses, log_probs, old_log_probs = self.generate_verifications(
            problems, proofs, num_samples=self.grpo_trainer.group_size
        )
        
        # Prepare batch for GRPO
        grpo_batch = {
            'inputs': [
                {'problem': p, 'proof': pr, 'score': s}
                for p, pr, s in zip(problems, proofs, scores)
                for _ in range(self.grpo_trainer.group_size)
            ],
            'outputs': analyses,
            'log_probs': log_probs,
            'old_log_probs': old_log_probs
        }
        
        # GRPO training step
        stats = self.grpo_trainer.train_step(grpo_batch, self.global_step)
        
        # Update learning rate
        self.scheduler.step()
        
        return stats
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary of epoch metrics
        """
        if self.mock_mode:
            return {'epoch': epoch, 'mean_loss': 0.5, 'mean_reward': 0.6}
        
        self.model.train()
        
        # Create dataloader
        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=lambda x: x  # Custom collation in train_step
        )
        
        epoch_metrics = []
        
        for batch_idx, batch in enumerate(dataloader):
            # Training step
            step_metrics = self.train_step(batch)
            
            # Track metrics
            self.metrics_tracker.update(
                loss=step_metrics.get('total_loss', 0.0),
                reward=step_metrics.get('mean_reward', 0.0),
                policy_loss=step_metrics.get('policy_loss', 0.0),
                kl_div=step_metrics.get('kl_div', 0.0)
            )
            
            epoch_metrics.append(step_metrics)
            self.global_step += 1
            
            # Logging
            if self.global_step % self.logging_steps == 0:
                avg_metrics = self.metrics_tracker.get_average_metrics()
                logger.info(
                    f"Epoch {epoch} Step {self.global_step}: "
                    f"Loss={avg_metrics['loss']:.4f}, "
                    f"Reward={avg_metrics['reward']:.4f}"
                )
                self.metrics_tracker.reset()
            
            # Evaluation
            if self.global_step % self.eval_steps == 0 and self.val_dataset is not None:
                val_metrics = self.evaluate()
                logger.info(f"Validation: {val_metrics}")
                
                # Save best checkpoint
                if val_metrics['mean_reward'] > self.best_val_reward:
                    self.best_val_reward = val_metrics['mean_reward']
                    self.save_checkpoint(epoch, is_best=True)
            
            # Checkpointing
            if self.global_step % self.save_steps == 0:
                self.save_checkpoint(epoch, is_best=False)
        
        # Compute epoch statistics
        avg_loss = sum(m.get('total_loss', 0) for m in epoch_metrics) / len(epoch_metrics)
        avg_reward = sum(m.get('mean_reward', 0) for m in epoch_metrics) / len(epoch_metrics)
        
        return {
            'epoch': epoch,
            'mean_loss': avg_loss,
            'mean_reward': avg_reward
        }
    
    def train(self) -> Dict[str, Any]:
        """
        Run full training loop.
        
        Returns:
            Dictionary with training history
        """
        if self.mock_mode:
            return {'epochs': [], 'best_reward': 0.6}
        
        logger.info(f"Starting training for {self.num_epochs} epochs")
        logger.info(f"Training samples: {len(self.train_dataset)}")
        logger.info(f"Output directory: {self.output_dir}")
        
        training_history = {
            'epochs': [],
            'best_reward': -float('inf')
        }
        
        for epoch in range(self.num_epochs):
            logger.info(f"\n{'='*50}")
            logger.info(f"Epoch {epoch + 1}/{self.num_epochs}")
            logger.info(f"{'='*50}")
            
            epoch_metrics = self.train_epoch(epoch)
            training_history['epochs'].append(epoch_metrics)
            
            # Update best reward
            if epoch_metrics['mean_reward'] > training_history['best_reward']:
                training_history['best_reward'] = epoch_metrics['mean_reward']
        
        logger.info("\nTraining complete!")
        logger.info(f"Best reward: {training_history['best_reward']:.4f}")
        
        return training_history
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate model on validation set.
        
        Returns:
            Dictionary of evaluation metrics
        """
        if self.mock_mode or self.val_dataset is None:
            return {'mean_reward': 0.6, 'mean_score_accuracy': 0.7}
        
        self.model.eval()
        
        # TODO: Implement actual evaluation logic
        # Generate verifications, compute rewards, aggregate metrics
        
        return {'mean_reward': 0.0, 'mean_score_accuracy': 0.0}
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """
        Save training checkpoint.
        
        Args:
            epoch: Current epoch
            is_best: Whether this is the best checkpoint so far
        """
        if self.mock_mode:
            return
        
        checkpoint_data = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_reward': self.best_val_reward
        }
        
        self.checkpoint_manager.save_checkpoint(
            checkpoint_data,
            step=self.global_step,
            metric_value=self.best_val_reward,
            is_best=is_best
        )
        
        logger.info(f"Checkpoint saved at step {self.global_step}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load checkpoint from file.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        if self.mock_mode:
            return
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.best_val_reward = checkpoint['best_val_reward']
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
        logger.info(f"Resuming from step {self.global_step}")