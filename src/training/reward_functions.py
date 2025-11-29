"""
src/training/reward_functions.py

Reward functions for GRPO training in DeepSeekMath-V2.
Implements R_format, R_score, and R_meta as described in the paper.
"""

from typing import Optional
import numpy as np
from src.utils.prompts import check_format_compliance


# ============================================================================
# Core Reward Functions
# ============================================================================

def compute_format_reward(response: str, response_type: str = "generation") -> float:
    """
    R_format: Binary reward for format compliance.
    
    Checks if response follows the required format with:
    - Required markdown sections
    - Required key phrases
    - Score in \boxed{} format
    
    Args:
        response: Model's generated response
        response_type: Type of response ("generation", "verification", "meta_verification")
        
    Returns:
        1.0 if format is correct, 0.0 otherwise
    """
    is_compliant = check_format_compliance(response, response_type)
    return 1.0 if is_compliant else 0.0


def compute_score_reward(predicted_score: Optional[float], ground_truth_score: float) -> float:
    """
    R_score: Reward based on score proximity.
    
    Formula: R_score = 1 - |predicted_score - ground_truth_score|
    
    Args:
        predicted_score: Score predicted by model (0, 0.5, or 1)
        ground_truth_score: Ground truth score (0, 0.5, or 1)
        
    Returns:
        Score reward in [0, 1]
    """
    if predicted_score is None:
        return 0.0
    
    # Ensure scores are valid
    if predicted_score not in [0, 0.5, 1] or ground_truth_score not in [0, 0.5, 1]:
        return 0.0
    
    return 1.0 - abs(predicted_score - ground_truth_score)


def compute_meta_reward(meta_verifier_score: float) -> float:
    """
    R_meta: Quality score from meta-verifier.
    
    This is simply the quality score assigned by the meta-verifier
    to the verifier's analysis.
    
    Args:
        meta_verifier_score: Meta-verifier's quality assessment (0, 0.5, or 1)
        
    Returns:
        The meta-verification score
    """
    if meta_verifier_score not in [0, 0.5, 1]:
        return 0.0
    
    return float(meta_verifier_score)


# ============================================================================
# Combined Reward Functions
# ============================================================================

def compute_verifier_reward(
    response: str,
    predicted_score: Optional[float],
    ground_truth_score: float,
    meta_score: Optional[float] = None
) -> float:
    """
    Combined reward for verifier training.
    
    Without meta-verification:
        R = R_format * R_score
        
    With meta-verification:
        R = R_format * R_score * R_meta
    
    Args:
        response: Verifier's response
        predicted_score: Score predicted by verifier
        ground_truth_score: Expert annotation
        meta_score: Optional meta-verifier quality score
        
    Returns:
        Combined reward value
    """
    r_format = compute_format_reward(response, response_type="verification")
    r_score = compute_score_reward(predicted_score, ground_truth_score)
    
    if meta_score is not None:
        r_meta = compute_meta_reward(meta_score)
        return r_format * r_score * r_meta
    else:
        return r_format * r_score


def compute_generator_reward(
    proof_response: str,
    proof_score: float,
    self_eval_score: Optional[float] = None,
    meta_score: Optional[float] = None,
    alpha: float = 0.76,
    beta: float = 0.24
) -> float:
    """
    Combined reward for generator training with self-verification.
    
    For basic generation (no self-verification):
        R = R_Y (proof score from verifier)
        
    For self-verification training:
        R = R_format * (α * R_Y + β * R_Z)
        
        where R_Z = R_score(self_score, verifier_score) * R_meta(self_analysis)
    
    Args:
        proof_response: Generator's full response (proof + self-analysis)
        proof_score: Verifier's score of the proof (R_Y)
        self_eval_score: Generator's self-assigned score
        meta_score: Meta-verification score of self-analysis quality
        alpha: Weight for proof quality (default 0.76)
        beta: Weight for self-evaluation quality (default 0.24)
        
    Returns:
        Combined reward value
    """
    # Basic generation (no self-verification)
    if self_eval_score is None:
        return proof_score
    
    # Self-verification training
    r_format = compute_format_reward(proof_response, response_type="generation")
    r_y = proof_score  # Verifier's score of proof
    
    # R_Z: Self-evaluation quality
    r_score_self = compute_score_reward(self_eval_score, proof_score)
    r_meta_self = compute_meta_reward(meta_score) if meta_score is not None else 1.0
    r_z = r_score_self * r_meta_self
    
    # Combined reward
    reward = r_format * (alpha * r_y + beta * r_z)
    
    return reward


# ============================================================================
# Batch Reward Computation
# ============================================================================

def compute_batch_rewards(
    responses: list[str],
    predicted_scores: list[Optional[float]],
    ground_truth_scores: list[float],
    meta_scores: Optional[list[float]] = None,
    reward_type: str = "verifier"
) -> list[float]:
    """
    Compute rewards for a batch of samples.
    
    Args:
        responses: List of model responses
        predicted_scores: List of predicted scores
        ground_truth_scores: List of ground truth scores
        meta_scores: Optional list of meta-verification scores
        reward_type: Type of reward ("verifier" or "generator")
        
    Returns:
        List of reward values
    """
    rewards = []
    
    if meta_scores is None:
        meta_scores = [None] * len(responses)
    
    for _, (response, pred_score, gt_score, meta_score) in enumerate(
        zip(responses, predicted_scores, ground_truth_scores, meta_scores)
    ):
        if reward_type == "verifier":
            reward = compute_verifier_reward(
                response, pred_score, gt_score, meta_score
            )
        elif reward_type == "generator":
            # For generator, pred_score is self_eval_score
            reward = compute_generator_reward(
                response, 
                proof_score=gt_score,  # Actually verifier score here
                self_eval_score=pred_score,
                meta_score=meta_score
            )
        else:
            raise ValueError(f"Unknown reward_type: {reward_type}")
        
        rewards.append(reward)
    
    return rewards


# ============================================================================
# Reward Analysis Utilities
# ============================================================================

def analyze_reward_components(
    response: str,
    predicted_score: Optional[float],
    ground_truth_score: float,
    meta_score: Optional[float] = None
) -> dict:
    """
    Break down reward into components for analysis/debugging.
    
    Args:
        response: Model response
        predicted_score: Predicted score
        ground_truth_score: Ground truth score
        meta_score: Optional meta-verification score
        
    Returns:
        Dictionary with reward components
    """
    r_format = compute_format_reward(response, "verification")
    r_score = compute_score_reward(predicted_score, ground_truth_score)
    r_meta = compute_meta_reward(meta_score) if meta_score is not None else None
    
    total_reward = compute_verifier_reward(
        response, predicted_score, ground_truth_score, meta_score
    )
    
    components = {
        "r_format": r_format,
        "r_score": r_score,
        "r_meta": r_meta,
        "total_reward": total_reward,
        "format_compliant": r_format == 1.0,
        "score_diff": abs(predicted_score - ground_truth_score) if predicted_score is not None else None
    }
    
    return components


def get_reward_statistics(rewards: list[float]) -> dict:
    """
    Compute statistics on a list of rewards.
    
    Args:
        rewards: List of reward values
        
    Returns:
        Dictionary with mean, min, max, std
    """
    rewards_array = np.array(rewards)
    
    stats = {
        "mean": float(np.mean(rewards_array)),
        "median": float(np.median(rewards_array)),
        "min": float(np.min(rewards_array)),
        "max": float(np.max(rewards_array)),
        "std": float(np.std(rewards_array)),
        "num_samples": len(rewards),
        "num_perfect": int(np.sum(rewards_array == 1.0)),
        "num_zero": int(np.sum(rewards_array == 0.0))
    }
    
    return stats


# ============================================================================
# Export
# ============================================================================

__all__ = [
    'compute_format_reward',
    'compute_score_reward',
    'compute_meta_reward',
    'compute_verifier_reward',
    'compute_generator_reward',
    'compute_batch_rewards',
    'analyze_reward_components',
    'get_reward_statistics',
]