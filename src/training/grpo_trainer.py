"""
src/training/grpo_trainer.py

Group Relative Policy Optimization (GRPO) trainer for LLM fine-tuning.

GRPO improves upon standard policy gradient methods by computing advantages
relative to groups of samples rather than using absolute rewards. This reduces
variance and improves training stability.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
import numpy as np
import torch
TORCH_AVAILABLE = True


class GRPOTrainer:
    """
    Group Relative Policy Optimization trainer.
    
    GRPO computes advantages by comparing rewards within groups of samples
    generated for the same input, reducing variance compared to absolute rewards.
    
    Algorithm:
        1. Generate K samples per input
        2. Compute reward for each sample
        3. Group samples by input (K samples per group)
        4. Compute advantage = (reward - group_mean) / group_std
        5. Update policy to increase log_prob of high-advantage samples
    """
    
    def __init__(
        self,
        model: Any,
        reward_fn: Callable,
        optimizer: Any,
        group_size: int = 4,
        kl_coef: float = 0.0,
        clip_range: float = 0.2,
        value_clip_range: Optional[float] = None,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        device: Optional[str] = None
    ):
        """
        Initialize GRPO trainer.
        
        Args:
            model: The policy model to train
            reward_fn: Function that computes rewards given (inputs, outputs)
            optimizer: PyTorch optimizer for model parameters
            group_size: Number of samples to generate per input (K)
            kl_coef: Coefficient for KL divergence penalty
            clip_range: PPO-style clipping range for policy updates
            value_clip_range: Optional clipping range for value function
            gradient_accumulation_steps: Number of steps to accumulate gradients
            max_grad_norm: Maximum gradient norm for clipping
            device: Device to run on ('cuda' or 'cpu')
        """
        if not TORCH_AVAILABLE:
            # Mock mode for CPU testing
            self.mock_mode = True
            self.model = None
            self.device = 'cpu'
            return
            
        self.mock_mode = False
        self.model = model
        self.reward_fn = reward_fn
        self.optimizer = optimizer
        self.group_size = group_size
        self.kl_coef = kl_coef
        self.clip_range = clip_range
        self.value_clip_range = value_clip_range
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        
        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        # Move model to device
        if hasattr(self.model, 'to'):
            self.model = self.model.to(self.device)
    
    def compute_group_rewards(
        self,
        inputs: List[str],
        outputs: List[str],
        **reward_kwargs
    ) -> List[float]:
        """
        Compute rewards for generated outputs.
        
        Args:
            inputs: List of input prompts (length N)
            outputs: List of generated outputs (length N * group_size)
            reward_kwargs: Additional arguments for reward function
            
        Returns:
            List of reward values (length N * group_size)
        """
        if self.mock_mode:
            # Return mock rewards for testing
            return [0.5 + 0.3 * np.random.randn() for _ in outputs]
        
        # Compute rewards using the reward function
        rewards = self.reward_fn(inputs, outputs, **reward_kwargs)
        return rewards
    
    def compute_advantages(
        self,
        rewards: List[float],
        group_size: Optional[int] = None
    ) -> Tuple[List[float], Dict[str, float]]:
        """
        Compute advantages using group normalization.
        
        For each group of K samples from the same input:
            advantage_i = (reward_i - mean(group_rewards)) / std(group_rewards)
        
        Args:
            rewards: List of rewards (length N * K)
            group_size: Size of each group (default: self.group_size)
            
        Returns:
            Tuple of (advantages, stats_dict)
            - advantages: List of advantage values (same length as rewards)
            - stats_dict: Dictionary with mean_reward, mean_advantage, etc.
        """
        if self.mock_mode:
            # Return mock advantages
            advantages = [r - 0.5 for r in rewards]
            stats = {
                'mean_reward': 0.5,
                'std_reward': 0.3,
                'mean_advantage': 0.0,
                'std_advantage': 1.0
            }
            return advantages, stats
        
        if group_size is None:
            group_size = self.group_size
        
        # Convert to numpy for easier computation
        rewards_array = np.array(rewards)
        n_samples = len(rewards)
        
        # Check that we have complete groups
        if n_samples % group_size != 0:
            raise ValueError(
                f"Number of samples ({n_samples}) must be divisible by "
                f"group_size ({group_size})"
            )
        
        n_groups = n_samples // group_size
        
        # Reshape into groups
        rewards_grouped = rewards_array.reshape(n_groups, group_size)
        
        # Compute group statistics
        group_means = rewards_grouped.mean(axis=1, keepdims=True)  # (n_groups, 1)
        group_stds = rewards_grouped.std(axis=1, keepdims=True)    # (n_groups, 1)
        
        # Avoid division by zero
        group_stds = np.maximum(group_stds, 1e-8)
        
        # Compute advantages
        advantages_grouped = (rewards_grouped - group_means) / group_stds
        advantages = advantages_grouped.reshape(-1).tolist()
        
        # Compute statistics
        stats = {
            'mean_reward': float(rewards_array.mean()),
            'std_reward': float(rewards_array.std()),
            'mean_advantage': float(np.mean(advantages)),
            'std_advantage': float(np.std(advantages)),
            'min_reward': float(rewards_array.min()),
            'max_reward': float(rewards_array.max())
        }
        
        return advantages, stats
    
    def compute_policy_loss(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute policy gradient loss with optional PPO-style clipping.
        
        Args:
            log_probs: Current policy log probabilities (batch_size, seq_len)
            old_log_probs: Reference policy log probabilities (batch_size, seq_len)
            advantages: Advantage values (batch_size,)
            mask: Optional mask for valid tokens (batch_size, seq_len)
            
        Returns:
            Tuple of (loss, stats_dict)
        """
        if self.mock_mode:
            return torch.tensor(0.5), {'policy_loss': 0.5, 'kl_div': 0.01}
        
        # Sum log probs over sequence length
        if mask is not None:
            log_probs_sum = (log_probs * mask).sum(dim=-1)
            old_log_probs_sum = (old_log_probs * mask).sum(dim=-1)
        else:
            log_probs_sum = log_probs.sum(dim=-1)
            old_log_probs_sum = old_log_probs.sum(dim=-1)
        
        # Compute ratio
        ratio = torch.exp(log_probs_sum - old_log_probs_sum)
        
        # Expand advantages to match batch size
        if advantages.dim() == 1:
            advantages = advantages.unsqueeze(-1)
        
        # PPO-style clipping
        if self.clip_range > 0:
            ratio_clipped = torch.clamp(
                ratio,
                1.0 - self.clip_range,
                1.0 + self.clip_range
            )
            policy_loss_unclipped = -ratio * advantages.squeeze()
            policy_loss_clipped = -ratio_clipped * advantages.squeeze()
            policy_loss = torch.mean(torch.max(policy_loss_unclipped, policy_loss_clipped))
        else:
            # Standard policy gradient
            policy_loss = -torch.mean(ratio * advantages.squeeze())
        
        # KL divergence penalty
        kl_div = torch.mean(old_log_probs_sum - log_probs_sum)
        
        # Total loss
        loss = policy_loss + self.kl_coef * kl_div
        
        # Statistics
        stats = {
            'policy_loss': policy_loss.item(),
            'kl_div': kl_div.item(),
            'total_loss': loss.item(),
            'mean_ratio': ratio.mean().item(),
            'max_ratio': ratio.max().item(),
            'min_ratio': ratio.min().item()
        }
        
        return loss, stats
    
    def train_step(
        self,
        batch: Dict[str, Any],
        step: int
    ) -> Dict[str, float]:
        """
        Perform one training step.
        
        Args:
            batch: Dictionary containing:
                - 'inputs': List of input prompts
                - 'outputs': List of generated outputs (K per input)
                - 'log_probs': Log probabilities from generation
                - 'old_log_probs': Reference log probabilities
                - Optional: 'mask' for valid tokens
            step: Current training step (for gradient accumulation)
            
        Returns:
            Dictionary of training statistics
        """
        if self.mock_mode:
            return {
                'loss': 0.5,
                'policy_loss': 0.45,
                'kl_div': 0.05,
                'mean_reward': 0.6,
                'mean_advantage': 0.0
            }
        
        # Extract batch components
        inputs = batch['inputs']
        outputs = batch['outputs']
        log_probs = batch['log_probs']
        old_log_probs = batch['old_log_probs']
        mask = batch.get('mask', None)
        
        # Compute rewards
        rewards = self.compute_group_rewards(inputs, outputs)
        
        # Compute advantages
        advantages, reward_stats = self.compute_advantages(rewards)
        advantages_tensor = torch.tensor(advantages, device=self.device)
        
        # Compute policy loss
        loss, loss_stats = self.compute_policy_loss(
            log_probs,
            old_log_probs,
            advantages_tensor,
            mask
        )
        
        # Scale loss by gradient accumulation steps
        loss = loss / self.gradient_accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Update weights if accumulation is complete
        if (step + 1) % self.gradient_accumulation_steps == 0:
            # Gradient clipping
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm
                )
            
            # Optimizer step
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        # Combine statistics
        stats = {**reward_stats, **loss_stats}
        
        return stats
    
    def train_epoch(
        self,
        dataloader: Any,
        epoch: int
    ) -> Dict[str, Any]:
        """
        Train for one epoch.
        
        Args:
            dataloader: DataLoader providing batches
            epoch: Current epoch number
            
        Returns:
            Dictionary of epoch statistics
        """
        if self.mock_mode:
            return {
                'epoch': epoch,
                'mean_loss': 0.5,
                'mean_reward': 0.6,
                'mean_advantage': 0.0
            }
        
        self.model.train()
        
        epoch_stats = {
            'loss': [],
            'policy_loss': [],
            'kl_div': [],
            'mean_reward': [],
            'mean_advantage': []
        }
        
        for step, batch in enumerate(dataloader):
            # Move batch to device if needed
            if isinstance(batch, dict):
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
            
            # Training step
            stats = self.train_step(batch, step)
            
            # Accumulate statistics
            for key in epoch_stats:
                if key in stats:
                    epoch_stats[key].append(stats[key])
        
        # Average statistics
        avg_stats = {
            f'mean_{key}': np.mean(values)
            for key, values in epoch_stats.items()
        }
        avg_stats['epoch'] = epoch
        
        return avg_stats
    
    def save_checkpoint(self, path: Path, epoch: int, **extra_data):
        """Save training checkpoint."""
        if self.mock_mode:
            return
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            **extra_data
        }
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: Path) -> Dict[str, Any]:
        """Load training checkpoint."""
        if self.mock_mode:
            return {'epoch': 0}
        
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return checkpoint