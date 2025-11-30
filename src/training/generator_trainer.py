"""
src/training/generator_trainer.py

Trainer for proof generation models using GRPO with self-verification.

Trains a generator to:
1. Generate rigorous mathematical proofs
2. Self-verify proof quality (identify own issues)
3. Iteratively refine proofs based on self-analysis

Uses GRPO with combined rewards:
- R_Y: Proof quality (from verifier)
- R_Z: Self-evaluation accuracy (meta-verification of self-analysis)
- R = α·R_Y + β·R_Z (α=0.76, β=0.24 from paper)
"""
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging
import re

try:
    import torch
    from torch.utils.data import DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    DataLoader = None

from src.training.grpo_trainer import GRPOTrainer
from src.utils.metrics import GRPOMetricsTracker
from src.training.checkpoint_manager import CheckpointManager

logger = logging.getLogger(__name__)


class GeneratorRewardFunction:
    """
    Reward function for generator training with self-verification.
    
    Combines two reward components:
    - R_Y: Proof quality (scored by verifier)
    - R_Z: Self-evaluation accuracy (meta-verification of self-analysis)
    
    Formula: R = α·R_Y + β·R_Z
    Paper values: α=0.76, β=0.24
    """
    
    def __init__(
        self,
        verifier: Any,
        alpha_proof: float = 0.76,
        beta_self_eval: float = 0.24
    ):
        """
        Initialize generator reward function.
        
        Args:
            verifier: Verifier model to score proofs
            alpha_proof: Weight for proof quality reward (default: 0.76)
            beta_self_eval: Weight for self-evaluation reward (default: 0.24)
        """
        self.verifier = verifier
        self.alpha_proof = alpha_proof
        self.beta_self_eval = beta_self_eval
        
        # Validate weights sum to 1
        total = alpha_proof + beta_self_eval
        if abs(total - 1.0) > 1e-6:
            logger.warning(
                f"Reward weights sum to {total}, not 1.0. "
                f"Normalizing: α={alpha_proof/total:.3f}, "
                f"β={beta_self_eval/total:.3f}"
            )
            self.alpha_proof = alpha_proof / total
            self.beta_self_eval = beta_self_eval / total
    
    def __call__(
        self,
        inputs: List[Dict[str, str]],
        outputs: List[str],
        **kwargs
    ) -> List[float]:
        """
        Compute rewards for generator outputs.
        
        Args:
            inputs: List of dicts with 'problem' key
            outputs: List of generator outputs (proof + self-analysis)
            
        Returns:
            List of reward values (one per output)
        """
        rewards = []
        
        for inp, output in zip(inputs, outputs):
            problem = inp['problem']
            
            # Parse output into proof and self-analysis
            proof, self_analysis = self._parse_output(output)
            
            # Check format
            r_format = self._check_format(proof, self_analysis)
            
            if r_format == 0.0:
                # Invalid format, no reward
                rewards.append(0.0)
                continue
            
            # Compute R_Y: Proof quality from verifier
            r_y = self._compute_proof_reward(problem, proof)
            
            # Compute R_Z: Self-evaluation accuracy
            r_z = self._compute_self_eval_reward(
                problem, proof, self_analysis, r_y
            )
            
            # Combined reward
            reward = r_format * (self.alpha_proof * r_y + self.beta_self_eval * r_z)
            rewards.append(reward)
        
        return rewards
    
    def _parse_output(self, output: str) -> Tuple[str, str]:
        """
        Parse generator output into proof and self-analysis.
        
        Expected format:
        ## Solution
        [proof content]
        
        ## Self Evaluation
        [self-analysis content]
        
        Args:
            output: Generator output string
            
        Returns:
            Tuple of (proof, self_analysis)
        """
        # Split by section headers
        if "## Solution" in output and "## Self Evaluation" in output:
            parts = output.split("## Self Evaluation")
            proof = parts[0].replace("## Solution", "").strip()
            self_analysis = parts[1].strip()
        else:
            # Fallback: treat entire output as proof
            proof = output
            self_analysis = ""
        
        return proof, self_analysis
    
    def _check_format(self, proof: str, self_analysis: str) -> float:
        """
        Check if output follows required format.
        
        Required:
        - Has "## Solution" section
        - Has "## Self Evaluation" section
        - Self-analysis contains evaluation phrase and score
        
        Args:
            proof: Proof content
            self_analysis: Self-analysis content
            
        Returns:
            1.0 if format correct, 0.0 otherwise
        """
        if not proof or not self_analysis:
            return 0.0
        
        # Check self-analysis format
        has_eval_phrase = "Here is my evaluation of the solution:" in self_analysis
        has_score_phrase = "the final overall score should be:" in self_analysis
        has_boxed = "\\boxed{" in self_analysis
        
        if has_eval_phrase and has_score_phrase and has_boxed:
            return 1.0
        else:
            return 0.0
    
    def _compute_proof_reward(self, problem: str, proof: str) -> float:
        """
        Compute proof quality reward using verifier.
        
        Args:
            problem: Problem statement
            proof: Generated proof
            
        Returns:
            Proof quality score from verifier (0, 0.5, or 1)
        """
        if self.verifier is None:
            # Mock mode
            return 0.7
        
        # Use verifier to score the proof
        # This would call the actual verifier model
        # For now, return placeholder
        return 0.5
    
    def _compute_self_eval_reward(
        self,
        problem: str,
        proof: str,
        self_analysis: str,
        true_score: float
    ) -> float:
        """
        Compute self-evaluation accuracy reward.
        
        This is meta-verification of the generator's self-analysis:
        - Extract predicted score from self-analysis
        - Compare with verifier's score (ground truth)
        - Multiply by meta-verification quality
        
        Args:
            problem: Problem statement
            proof: Generated proof
            self_analysis: Generator's self-analysis
            true_score: Verifier's score (ground truth)
            
        Returns:
            Self-evaluation reward
        """
        # Extract self-predicted score
        self_score = self._extract_score(self_analysis)
        
        if self_score is None:
            return 0.0
        
        # R_score: Accuracy of self-assessment
        r_score = 1.0 - abs(self_score - true_score)
        
        # R_meta: Quality of self-analysis
        # Would use verifier in meta-verification mode
        r_meta = self._compute_meta_quality(problem, proof, self_analysis)
        
        # Combined self-evaluation reward
        return r_score * r_meta
    
    def _extract_score(self, text: str) -> Optional[float]:
        """Extract score from boxed notation."""
        
        match = re.search(r'\\boxed\{([\d.]+)\}', text)
        if match:
            try:
                score = float(match.group(1))
                if score in [0.0, 0.5, 1.0]:
                    return score
            except ValueError:
                pass
        
        return None
    
    def _compute_meta_quality(
        self,
        problem: str,
        proof: str,
        self_analysis: str
    ) -> float:
        """
        Compute meta-verification quality of self-analysis.
        
        Uses verifier in meta-verification mode to check:
        - Are identified issues real?
        - Is the analysis accurate and justified?
        
        Args:
            problem: Problem statement
            proof: Generated proof
            self_analysis: Generator's analysis of its own proof
            
        Returns:
            Meta-verification quality score (0, 0.5, or 1)
        """
        if self.verifier is None:
            return 1.0
        
        # Use verifier as meta-verifier
        # This would call verifier with meta-verification prompt
        return 1.0


class GeneratorTrainer:
    """
    Trainer for proof generation models with self-verification.
    
    Uses GRPO to train generators that:
    1. Generate rigorous proofs
    2. Self-verify their proofs
    3. Iteratively refine based on self-identified issues
    """
    
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        verifier: Any,
        train_dataset: Any,
        val_dataset: Optional[Any] = None,
        output_dir: str = "checkpoints/generator",
        # GRPO parameters
        group_size: int = 4,
        kl_coef: float = 0.01,
        clip_range: float = 0.2,
        # Reward parameters
        alpha_proof: float = 0.76,
        beta_self_eval: float = 0.24,
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
        max_new_tokens: int = 4096,
        temperature: float = 0.8,
        top_p: float = 0.95,
        # Refinement parameters
        enable_refinement: bool = False,
        max_refinement_iterations: int = 3,
        device: Optional[str] = None
    ):
        """
        Initialize generator trainer.
        
        Args:
            model: The generator model to train
            tokenizer: Tokenizer for the model
            verifier: Verifier model for proof scoring
            train_dataset: Training dataset (GenerationDataset)
            val_dataset: Validation dataset (optional)
            output_dir: Directory for checkpoints and logs
            group_size: Number of samples per problem for GRPO
            kl_coef: KL divergence penalty coefficient
            clip_range: PPO-style clipping range
            alpha_proof: Weight for proof quality (R_Y)
            beta_self_eval: Weight for self-evaluation (R_Z)
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
            enable_refinement: Whether to enable iterative refinement
            max_refinement_iterations: Max refinement iterations
            device: Device to use ('cuda' or 'cpu')
        """
        if not TORCH_AVAILABLE:
            self.mock_mode = True
            logger.warning("PyTorch not available, running in mock mode")
            return
        
        self.mock_mode = False
        self.model = model
        self.tokenizer = tokenizer
        self.verifier = verifier
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Move models to device
        self.model = self.model.to(self.device)
        if self.verifier is not None:
            self.verifier = self.verifier.to(self.device)
        
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
        self.reward_fn = GeneratorRewardFunction(
            verifier=self.verifier,
            alpha_proof=alpha_proof,
            beta_self_eval=beta_self_eval
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
        self.enable_refinement = enable_refinement
        self.max_refinement_iterations = max_refinement_iterations
        
        # Training state
        self.global_step = 0
        self.best_val_reward = -float('inf')
    
    def generate_proofs_with_self_verification(
        self,
        problems: List[str],
        num_samples: int = 1
    ) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
        """
        Generate proofs with self-verification for given problems.
        
        Output format:
        ## Solution
        [proof content]
        
        ## Self Evaluation
        [self-analysis]
        
        Args:
            problems: List of problem statements
            num_samples: Number of proofs to generate per problem
            
        Returns:
            Tuple of (outputs, log_probs, old_log_probs)
        """
        if self.mock_mode:
            outputs = [
                f"## Solution\nMock proof {i}\n## Self Evaluation\nMock analysis {i}"
                for i in range(len(problems) * num_samples)
            ]
            log_probs = torch.randn(len(outputs), 100)
            old_log_probs = log_probs.clone()
            return outputs, log_probs, old_log_probs
        
        # TODO: Implement actual generation logic
        # This would format prompts, generate with model, compute log probs
        outputs = []
        log_probs_list = []
        
        # Placeholder implementation
        return outputs, torch.tensor([]), torch.tensor([])
    
    def refine_proof(
        self,
        problem: str,
        proof_with_analysis: str
    ) -> str:
        """
        Refine a proof based on its self-analysis.
        
        If self-analysis identifies issues, prompt the generator
        to fix them.
        
        Args:
            problem: Problem statement
            proof_with_analysis: Previous proof + self-analysis
            
        Returns:
            Refined proof with new self-analysis
        """
        if self.mock_mode:
            return f"Refined: {proof_with_analysis}"
        
        # TODO: Implement refinement logic
        # This would extract issues from self-analysis
        # and prompt model to address them
        return proof_with_analysis
    
    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        Perform one training step.
        
        Args:
            batch: Batch containing problems
            
        Returns:
            Dictionary of training metrics
        """
        if self.mock_mode:
            return {
                'loss': 0.5,
                'mean_reward': 0.65,
                'mean_proof_quality': 0.7,
                'mean_self_eval_accuracy': 0.55
            }
        
        problems = batch['problems']
        
        # Generate proofs with self-verification
        outputs, log_probs, old_log_probs = self.generate_proofs_with_self_verification(
            problems, num_samples=self.grpo_trainer.group_size
        )
        
        # Prepare batch for GRPO
        grpo_batch = {
            'inputs': [
                {'problem': p}
                for p in problems
                for _ in range(self.grpo_trainer.group_size)
            ],
            'outputs': outputs,
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
            return {
                'epoch': epoch,
                'mean_loss': 0.5,
                'mean_reward': 0.65
            }
        
        self.model.train()
        if self.verifier is not None:
            self.verifier.eval()  # Verifier in eval mode
        
        # Create dataloader
        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=lambda x: x
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
            return {'epochs': [], 'best_reward': 0.65}
        
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
            return {
                'mean_reward': 0.65,
                'mean_proof_quality': 0.7,
                'mean_self_eval_accuracy': 0.6
            }
        
        self.model.eval()
        
        # TODO: Implement actual evaluation logic
        # Generate proofs, compute rewards, aggregate metrics
        
        return {
            'mean_reward': 0.0,
            'mean_proof_quality': 0.0,
            'mean_self_eval_accuracy': 0.0
        }
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save training checkpoint."""
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
        """Load checkpoint from file."""
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