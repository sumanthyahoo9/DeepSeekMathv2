"""
tests/test_reward_functions.py

Unit tests for reward functions used in GRPO training.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.reward_functions import (
    compute_format_reward,
    compute_score_reward,
    compute_meta_reward,
    compute_verifier_reward,
    compute_generator_reward,
    compute_batch_rewards,
    analyze_reward_components,
    get_reward_statistics
)


# ============================================================================
# Test Individual Reward Components
# ============================================================================

def test_compute_format_reward_valid():
    """Test format reward with valid response"""
    valid_response = """
Here is my evaluation of the solution:
The proof is correct.
Based on my evaluation, the final overall score should be: \\boxed{1}
"""
    
    reward = compute_format_reward(valid_response, "verification")
    assert reward == 1.0


def test_compute_format_reward_invalid():
    """Test format reward with invalid response"""
    invalid_response = "Just some text without proper format"
    
    reward = compute_format_reward(invalid_response, "verification")
    assert reward == 0.0


def test_compute_score_reward_exact_match():
    """Test score reward with exact match"""
    # Exact match
    assert compute_score_reward(1.0, 1.0) == 1.0
    assert compute_score_reward(0.5, 0.5) == 1.0
    assert compute_score_reward(0.0, 0.0) == 1.0


def test_compute_score_reward_partial_match():
    """Test score reward with partial match"""
    # Off by 0.5
    assert compute_score_reward(1.0, 0.5) == 0.5
    assert compute_score_reward(0.5, 0.0) == 0.5
    
    # Off by 1.0
    assert compute_score_reward(1.0, 0.0) == 0.0
    assert compute_score_reward(0.0, 1.0) == 0.0


def test_compute_score_reward_none():
    """Test score reward with None predicted score"""
    assert compute_score_reward(None, 1.0) == 0.0


def test_compute_score_reward_invalid():
    """Test score reward with invalid scores"""
    assert compute_score_reward(2.0, 1.0) == 0.0  # Invalid predicted score
    assert compute_score_reward(1.0, 1.5) == 0.0  # Invalid ground truth


def test_compute_meta_reward():
    """Test meta-verification reward"""
    assert compute_meta_reward(1.0) == 1.0
    assert compute_meta_reward(0.5) == 0.5
    assert compute_meta_reward(0.0) == 0.0
    assert compute_meta_reward(2.0) == 0.0  # Invalid score


# ============================================================================
# Test Combined Rewards
# ============================================================================

def test_compute_verifier_reward_no_meta():
    """Test verifier reward without meta-verification"""
    valid_response = """
Here is my evaluation of the solution:
Analysis...
Based on my evaluation, the final overall score should be: \\boxed{1}
"""
    
    # Perfect case: format correct, score exact match
    reward = compute_verifier_reward(valid_response, 1.0, 1.0)
    assert reward == 1.0  # 1.0 * 1.0 = 1.0
    
    # Format correct, score off by 0.5
    reward = compute_verifier_reward(valid_response, 0.5, 1.0)
    assert reward == 0.5  # 1.0 * 0.5 = 0.5


def test_compute_verifier_reward_with_meta():
    """Test verifier reward with meta-verification"""
    valid_response = """
Here is my evaluation of the solution:
Analysis...
Based on my evaluation, the final overall score should be: \\boxed{1}
"""
    
    # Perfect case with perfect meta score
    reward = compute_verifier_reward(valid_response, 1.0, 1.0, meta_score=1.0)
    assert reward == 1.0  # 1.0 * 1.0 * 1.0 = 1.0
    
    # Perfect match but low meta score
    reward = compute_verifier_reward(valid_response, 1.0, 1.0, meta_score=0.5)
    assert reward == 0.5  # 1.0 * 1.0 * 0.5 = 0.5


def test_compute_verifier_reward_format_failure():
    """Test verifier reward when format is wrong"""
    invalid_response = "No proper format"
    
    # Even with perfect score match, format failure gives 0
    reward = compute_verifier_reward(invalid_response, 1.0, 1.0)
    assert reward == 0.0


def test_compute_generator_reward_basic():
    """Test generator reward without self-verification"""
    # Basic generation just returns proof score
    reward = compute_generator_reward(
        proof_response="",
        proof_score=1.0,
        self_eval_score=None  # No self-verification
    )
    assert reward == 1.0


def test_compute_generator_reward_self_verification():
    """Test generator reward with self-verification"""
    valid_response = """
## Solution
Proof here...

## Self Evaluation
Here is my evaluation of the solution:
Analysis...
Based on my evaluation, the final overall score should be: \\boxed{1}
"""
    
    # Perfect case: format correct, proof score 1.0, self-eval matches
    reward = compute_generator_reward(
        proof_response=valid_response,
        proof_score=1.0,
        self_eval_score=1.0,
        meta_score=1.0,
        alpha=0.76,
        beta=0.24
    )
    
    # R = R_format * (α * R_Y + β * R_Z)
    # R_format = 1.0
    # R_Y = 1.0
    # R_Z = R_score(1.0, 1.0) * R_meta(1.0) = 1.0 * 1.0 = 1.0
    # R = 1.0 * (0.76 * 1.0 + 0.24 * 1.0) = 1.0
    assert reward == 1.0


def test_compute_generator_reward_poor_self_eval():
    """Test generator reward with poor self-evaluation"""
    valid_response = """
## Solution
Proof here...

## Self Evaluation
Here is my evaluation of the solution:
Analysis...
Based on my evaluation, the final overall score should be: \\boxed{0}
"""
    
    # Proof is perfect (1.0) but self-eval says 0 (wrong)
    reward = compute_generator_reward(
        proof_response=valid_response,
        proof_score=1.0,
        self_eval_score=0.0,  # Wrong self-assessment
        meta_score=1.0,
        alpha=0.76,
        beta=0.24
    )
    
    # R_Y = 1.0
    # R_Z = R_score(0.0, 1.0) * 1.0 = 0.0
    # R = 1.0 * (0.76 * 1.0 + 0.24 * 0.0) = 0.76
    assert reward == 0.76


# ============================================================================
# Test Batch Operations
# ============================================================================

def test_compute_batch_rewards():
    """Test computing rewards for a batch"""
    responses = [
        "Valid format \\boxed{1}",
        "Valid format \\boxed{0.5}",
        "Valid format \\boxed{0}"
    ]
    
    # Mock responses as valid format
    for i, resp in enumerate(responses):
        responses[i] = f"Here is my evaluation of the solution:\n{resp}\nBased on my evaluation, the final overall score should be: {resp}"
    
    predicted_scores = [1.0, 0.5, 0.0]
    ground_truth_scores = [1.0, 1.0, 0.5]  # Different from predicted
    
    rewards = compute_batch_rewards(
        responses,
        predicted_scores,
        ground_truth_scores,
        reward_type="verifier"
    )
    
    assert len(rewards) == 3
    assert rewards[0] == 1.0  # Perfect match
    assert rewards[1] == 0.5  # Off by 0.5
    assert rewards[2] == 0.5  # Off by 0.5


# ============================================================================
# Test Analysis Utilities
# ============================================================================

def test_analyze_reward_components():
    """Test reward component analysis"""
    response = """
Here is my evaluation of the solution:
Analysis...
Based on my evaluation, the final overall score should be: \\boxed{1}
"""
    
    analysis = analyze_reward_components(response, 0.5, 1.0, meta_score=1.0)
    
    assert analysis["r_format"] == 1.0
    assert analysis["r_score"] == 0.5
    assert analysis["r_meta"] == 1.0
    assert analysis["total_reward"] == 0.5  # 1.0 * 0.5 * 1.0
    assert analysis["format_compliant"] is True
    assert analysis["score_diff"] == 0.5


def test_get_reward_statistics():
    """Test reward statistics computation"""
    rewards = [1.0, 1.0, 0.5, 0.5, 0.0, 0.0, 1.0]
    
    stats = get_reward_statistics(rewards)
    
    assert stats["num_samples"] == 7
    assert stats["num_perfect"] == 3  # Three 1.0s
    assert stats["num_zero"] == 2  # Two 0.0s
    assert stats["min"] == 0.0
    assert stats["max"] == 1.0
    assert 0 < stats["mean"] < 1
    assert "std" in stats


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])