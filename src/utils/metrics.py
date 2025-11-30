"""
src/utils/metrics.py

Training metrics collection and logging for DeepSeekMath-V2.
Tracks loss, rewards, scores, and exports to JSON/TensorBoard.
"""

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

import numpy as np


# ============================================================================
# Metric Tracker
# ============================================================================

class MetricTracker:
    """
    Track training metrics over time.
    
    Stores metrics with timestamps and supports:
    - Running statistics (mean, std)
    - Windowed statistics
    - Export to JSON
    - TensorBoard logging (optional)
    
    Args:
        window_size: Size of sliding window for statistics
    """
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics = defaultdict(list)
        self.timestamps = defaultdict(list)
        self.start_time = time.time()
    
    def update(self, metric_name: str, value: float, step: Optional[int] = None) -> None:
        """
        Update a metric.
        
        Args:
            metric_name: Name of metric (e.g., 'loss', 'reward')
            value: Metric value
            step: Optional step/iteration number
        """
        self.metrics[metric_name].append(value)
        
        timestamp = {
            'time': time.time(),
            'step': step if step is not None else len(self.metrics[metric_name]) - 1
        }
        self.timestamps[metric_name].append(timestamp)
    
    def update_batch(self, metrics_dict: Dict[str, float], step: Optional[int] = None) -> None:
        """
        Update multiple metrics at once.
        
        Args:
            metrics_dict: Dictionary of metric_name -> value
            step: Optional step/iteration number
        """
        for metric_name, value in metrics_dict.items():
            self.update(metric_name, value, step)
    
    def get_latest(self, metric_name: str) -> Optional[float]:
        """
        Get latest value for a metric.
        
        Args:
            metric_name: Name of metric
            
        Returns:
            Latest value or None if no data
        """
        if metric_name not in self.metrics or not self.metrics[metric_name]:
            return None
        return self.metrics[metric_name][-1]
    
    def get_mean(self, metric_name: str, window: Optional[int] = None) -> Optional[float]:
        """
        Get mean value for a metric.
        
        Args:
            metric_name: Name of metric
            window: Window size (None = all values)
            
        Returns:
            Mean value or None if no data
        """
        if metric_name not in self.metrics or not self.metrics[metric_name]:
            return None
        
        values = self.metrics[metric_name]
        if window:
            values = values[-window:]
        
        return float(np.mean(values))
    
    def get_std(self, metric_name: str, window: Optional[int] = None) -> Optional[float]:
        """
        Get standard deviation for a metric.
        
        Args:
            metric_name: Name of metric
            window: Window size (None = all values)
            
        Returns:
            Standard deviation or None if no data
        """
        if metric_name not in self.metrics or not self.metrics[metric_name]:
            return None
        
        values = self.metrics[metric_name]
        if window:
            values = values[-window:]
        
        return float(np.std(values))
    
    def get_summary(self, metric_name: str) -> Dict[str, float]:
        """
        Get summary statistics for a metric.
        
        Args:
            metric_name: Name of metric
            
        Returns:
            Dictionary with summary statistics
        """
        if metric_name not in self.metrics or not self.metrics[metric_name]:
            return {}
        
        values = self.metrics[metric_name]
        
        return {
            'count': len(values),
            'latest': values[-1],
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'median': float(np.median(values)),
            'p25': float(np.percentile(values, 25)),
            'p75': float(np.percentile(values, 75)),
            'p95': float(np.percentile(values, 95))
        }
    
    def get_all_summaries(self) -> Dict[str, Dict[str, float]]:
        """
        Get summaries for all metrics.
        
        Returns:
            Dictionary mapping metric_name to summary
        """
        return {
            metric_name: self.get_summary(metric_name)
            for metric_name in self.metrics.keys()
        }
    
    def export_to_json(self, output_path: Union[str, Path]) -> None:
        """
        Export all metrics to JSON file.
        
        Args:
            output_path: Path to output JSON file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'timestamp': datetime.now().isoformat(),
            'total_runtime_sec': time.time() - self.start_time,
            'summaries': self.get_all_summaries(),
            'raw_metrics': {
                metric_name: {
                    'values': values,
                    'timestamps': self.timestamps[metric_name]
                }
                for metric_name, values in self.metrics.items()
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✓ Metrics exported to {output_path}")


# ============================================================================
# GRPO Metrics Tracker
# ============================================================================

class GRPOMetricsTracker(MetricTracker):
    """
    Specialized tracker for GRPO training metrics.
    
    Tracks:
    - Policy loss
    - Value loss (if applicable)
    - Rewards (total, format, score, meta)
    - KL divergence
    - Advantage statistics
    """
    
    def __init__(self, window_size: int = 100):
        super().__init__(window_size)
        self.group_rewards = []  # Store group-wise rewards
    
    def update_grpo_step(
        self,
        step: int,
        policy_loss: float,
        mean_reward: float,
        reward_components: Optional[Dict[str, float]] = None,
        kl_divergence: Optional[float] = None
    ) -> None:
        """
        Update GRPO training step metrics.
        
        Args:
            step: Training step
            policy_loss: Policy loss value
            mean_reward: Mean reward across batch
            reward_components: Dictionary of reward components
            kl_divergence: KL divergence from reference policy
        """
        self.update('policy_loss', policy_loss, step)
        self.update('mean_reward', mean_reward, step)
        
        if reward_components:
            for component, value in reward_components.items():
                self.update(f'reward_{component}', value, step)
        
        if kl_divergence is not None:
            self.update('kl_divergence', kl_divergence, step)
    
    def update_group_rewards(self, group_rewards: List[float]) -> None:
        """
        Store group rewards for analysis.
        
        Args:
            group_rewards: List of rewards for each sample in group
        """
        self.group_rewards.append(group_rewards)
    
    def get_reward_distribution(self) -> Dict[str, Any]:
        """
        Analyze distribution of rewards across groups.
        
        Returns:
            Dictionary with reward distribution statistics
        """
        if not self.group_rewards:
            return {}
        
        # Flatten all group rewards
        all_rewards = [r for group in self.group_rewards for r in group]
        
        # Get distribution of ranks within groups
        ranks = []
        for group in self.group_rewards:
            sorted_indices = np.argsort(group)
            group_ranks = np.empty_like(sorted_indices)
            group_ranks[sorted_indices] = np.arange(len(group))
            ranks.extend(group_ranks.tolist())
        
        return {
            'total_samples': len(all_rewards),
            'total_groups': len(self.group_rewards),
            'reward_mean': float(np.mean(all_rewards)),
            'reward_std': float(np.std(all_rewards)),
            'reward_min': float(np.min(all_rewards)),
            'reward_max': float(np.max(all_rewards)),
            'rank_distribution': {
                'mean': float(np.mean(ranks)),
                'std': float(np.std(ranks))
            }
        }


# ============================================================================
# Score Distribution Tracker
# ============================================================================

class ScoreDistributionTracker:
    """
    Track distribution of verification scores.
    
    Monitors:
    - Score frequencies (0, 0.5, 1)
    - Score trends over time
    - Invalid score rate
    """
    
    def __init__(self):
        self.scores = []
        self.timestamps = []
        self.invalid_count = 0
    
    def update(self, scores: List[Optional[float]]) -> None:
        """
        Update with batch of scores.
        
        Args:
            scores: List of scores (can include None for invalid)
        """
        timestamp = time.time()
        
        for score in scores:
            if score is None:
                self.invalid_count += 1
            else:
                self.scores.append(score)
                self.timestamps.append(timestamp)
    
    def get_distribution(self) -> Dict[str, Any]:
        """
        Get score distribution.
        
        Returns:
            Dictionary with distribution statistics
        """
        if not self.scores:
            return {
                'total': 0,
                'invalid_count': self.invalid_count,
                'invalid_rate': 0.0
            }
        
        total = len(self.scores) + self.invalid_count
        
        # Count each score value
        score_counts = {
            '0.0': self.scores.count(0.0),
            '0.5': self.scores.count(0.5),
            '1.0': self.scores.count(1.0)
        }
        
        return {
            'total': total,
            'valid_count': len(self.scores),
            'invalid_count': self.invalid_count,
            'invalid_rate': self.invalid_count / total if total > 0 else 0.0,
            'score_counts': score_counts,
            'score_percentages': {
                k: (v / len(self.scores) * 100) if self.scores else 0.0
                for k, v in score_counts.items()
            },
            'mean_score': float(np.mean(self.scores)) if self.scores else 0.0
        }


# ============================================================================
# Comprehensive Training Logger
# ============================================================================

class TrainingLogger:
    """
    Comprehensive training logger combining all metrics.
    
    Args:
        log_dir: Directory for saving logs
        experiment_name: Name of experiment
    """
    
    def __init__(
        self,
        log_dir: Union[str, Path],
        experiment_name: str = "experiment"
    ):
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.metric_tracker = MetricTracker()
        self.score_tracker = ScoreDistributionTracker()
        self.start_time = time.time()
        
        # Create experiment-specific directory
        self.experiment_dir = self.log_dir / experiment_name
        self.experiment_dir.mkdir(exist_ok=True)
    
    def log_step(
        self,
        step: int,
        metrics: Dict[str, float],
        scores: Optional[List[Optional[float]]] = None
    ) -> None:
        """
        Log a training step.
        
        Args:
            step: Training step number
            metrics: Dictionary of metrics to log
            scores: Optional list of scores
        """
        # Update metrics
        self.metric_tracker.update_batch(metrics, step)
        
        # Update scores if provided
        if scores is not None:
            self.score_tracker.update(scores)
    
    def save_checkpoint_metrics(self, checkpoint_name: str) -> None:
        """
        Save metrics at checkpoint.
        
        Args:
            checkpoint_name: Name of checkpoint
        """
        output_path = self.experiment_dir / f"{checkpoint_name}_metrics.json"
        self.export_all(output_path)
    
    def export_all(self, output_path: Optional[Union[str, Path]] = None) -> None:
        """
        Export all tracking data to JSON.
        
        Args:
            output_path: Path to output file (default: auto-generated)
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.experiment_dir / f"metrics_{timestamp}.json"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'experiment_name': self.experiment_name,
            'timestamp': datetime.now().isoformat(),
            'total_runtime_sec': time.time() - self.start_time,
            'metric_summaries': self.metric_tracker.get_all_summaries(),
            'score_distribution': self.score_tracker.get_distribution(),
            'raw_metrics': {
                metric_name: values
                for metric_name, values in self.metric_tracker.metrics.items()
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✓ Training logs exported to {output_path}")


# ============================================================================
# Utility Functions
# ============================================================================

def compute_percentiles(values: List[float], percentiles = None) -> Dict[str, float]:
    """
    Compute percentiles of values.
    
    Args:
        values: List of values
        percentiles: List of percentile points
        
    Returns:
        Dictionary mapping percentile to value
    """
    if not values:
        return {}
    if not percentiles:
        percentiles = [25, 50, 75, 95]
    
    return {
        f'p{p}': float(np.percentile(values, p))
        for p in percentiles
    }


# ============================================================================
# Export
# ============================================================================

__all__ = [
    'MetricTracker',
    'GRPOMetricsTracker',
    'ScoreDistributionTracker',
    'TrainingLogger',
    'compute_percentiles',
]