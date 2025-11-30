"""
tests/test_verifier.py

Unit tests for proof verifier models.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.verifier import (
    ProofVerifier,
    MetaVerifier,
    load_verifier,
    load_meta_verifier
)


# ============================================================================
# Test ProofVerifier Initialization
# ============================================================================

def test_verifier_init_default():
    """Test verifier initialization with defaults"""
    verifier = ProofVerifier()
    
    assert verifier.model_name == "deepseek-ai/DeepSeek-V3.2-Exp-SFT"
    assert verifier.use_greedy is True
    assert verifier.generation_config is not None


def test_verifier_init_custom():
    """Test verifier initialization with custom params"""
    verifier = ProofVerifier(
        model_name="test-model",
        device="cpu",
        use_greedy=False
    )
    
    assert verifier.model_name == "test-model"
    assert verifier.use_greedy is False


# ============================================================================
# Test Single Verification
# ============================================================================

def test_verify_basic():
    """Test basic proof verification"""
    verifier = ProofVerifier()
    
    problem = "Prove that 2+2=4"
    proof = "By the definition of addition in the natural numbers, 2+2=4. QED."
    
    result = verifier.verify(problem, proof)
    
    assert 'analysis' in result
    assert 'score' in result
    assert 'raw_output' in result
    assert isinstance(result['analysis'], str)


def test_verify_short_proof():
    """Test verification of short proof (should get low score in mock)"""
    verifier = ProofVerifier()
    
    problem = "Test problem"
    proof = "Too short"
    
    result = verifier.verify(problem, proof)
    
    # Mock mode should give low score for short proof
    assert result['score'] == 0.0


def test_verify_medium_proof():
    """Test verification of medium-length proof"""
    verifier = ProofVerifier()
    
    problem = "Test problem"
    proof = "A" * 100  # Medium length
    
    result = verifier.verify(problem, proof)
    
    # Mock mode should give medium score
    assert result['score'] == 0.5


def test_verify_long_proof():
    """Test verification of comprehensive proof"""
    verifier = ProofVerifier()
    
    problem = "Test problem"
    proof = "A" * 250  # Long proof
    
    result = verifier.verify(problem, proof)
    
    # Mock mode should give high score
    assert result['score'] == 1.0


# ============================================================================
# Test Batch Verification
# ============================================================================

def test_verify_batch():
    """Test batch verification"""
    verifier = ProofVerifier()
    
    problems = ["P1", "P2", "P3"]
    proofs = ["Short", "A" * 100, "A" * 250]
    
    results = verifier.verify_batch(problems, proofs)
    
    assert len(results) == 3
    assert all('score' in r for r in results)
    
    # Check scores based on length heuristic
    assert results[0]['score'] == 0.0  # Short
    assert results[1]['score'] == 0.5  # Medium
    assert results[2]['score'] == 1.0  # Long


def test_verify_batch_mismatch():
    """Test batch verification with mismatched lengths"""
    verifier = ProofVerifier()
    
    problems = ["P1", "P2"]
    proofs = ["Proof1"]
    
    with pytest.raises(ValueError):
        verifier.verify_batch(problems, proofs)


# ============================================================================
# Test MetaVerifier Initialization
# ============================================================================

def test_meta_verifier_init_default():
    """Test meta-verifier initialization with defaults"""
    meta_verifier = MetaVerifier()
    
    assert meta_verifier.model_name == "deepseek-ai/DeepSeek-V3.2-Exp-SFT"
    assert meta_verifier.use_greedy is True


def test_meta_verifier_init_custom():
    """Test meta-verifier initialization with custom params"""
    meta_verifier = MetaVerifier(
        model_name="test-model",
        use_greedy=False
    )
    
    assert meta_verifier.model_name == "test-model"
    assert meta_verifier.use_greedy is False


# ============================================================================
# Test Meta-Verification
# ============================================================================

def test_meta_verify_basic():
    """Test basic meta-verification"""
    meta_verifier = MetaVerifier()
    
    problem = "Test problem"
    proof = "Test proof"
    analysis = "Detailed analysis of the proof covering multiple aspects..."
    
    result = meta_verifier.meta_verify(problem, proof, analysis)
    
    assert 'meta_analysis' in result
    assert 'meta_score' in result
    assert 'raw_output' in result


def test_meta_verify_short_analysis():
    """Test meta-verification of brief analysis"""
    meta_verifier = MetaVerifier()
    
    problem = "Test problem"
    proof = "Test proof"
    analysis = "Brief"
    
    result = meta_verifier.meta_verify(problem, proof, analysis)
    
    # Mock mode should give medium score for short analysis
    assert result['meta_score'] == 0.5


def test_meta_verify_detailed_analysis():
    """Test meta-verification of detailed analysis"""
    meta_verifier = MetaVerifier()
    
    problem = "Test problem"
    proof = "Test proof"
    analysis = "A" * 150  # Detailed analysis
    
    result = meta_verifier.meta_verify(problem, proof, analysis)
    
    # Mock mode should give high score
    assert result['meta_score'] == 1.0


# ============================================================================
# Test Meta-Verification Batch
# ============================================================================

def test_meta_verify_batch():
    """Test batch meta-verification"""
    meta_verifier = MetaVerifier()
    
    problems = ["P1", "P2"]
    proofs = ["Proof1", "Proof2"]
    analyses = ["Brief", "A" * 150]
    
    results = meta_verifier.meta_verify_batch(problems, proofs, analyses)
    
    assert len(results) == 2
    assert all('meta_score' in r for r in results)
    assert results[0]['meta_score'] == 0.5  # Brief
    assert results[1]['meta_score'] == 1.0  # Detailed


def test_meta_verify_batch_mismatch():
    """Test batch meta-verification with mismatched lengths"""
    meta_verifier = MetaVerifier()
    
    problems = ["P1", "P2"]
    proofs = ["Proof1"]
    analyses = ["Analysis1", "Analysis2"]
    
    with pytest.raises(ValueError):
        meta_verifier.meta_verify_batch(problems, proofs, analyses)


# ============================================================================
# Test Integration
# ============================================================================

def test_verifier_to_meta_verifier_pipeline():
    """Test complete verification pipeline"""
    verifier = ProofVerifier()
    meta_verifier = MetaVerifier()
    
    problem = "Prove that sqrt(2) is irrational"
    proof = "Assume sqrt(2) = p/q in lowest terms..." + "A" * 200
    
    # Step 1: Verify proof
    verification = verifier.verify(problem, proof)
    assert verification['score'] is not None
    
    # Step 2: Meta-verify the verification
    meta_result = meta_verifier.meta_verify(
        problem,
        proof,
        verification['analysis']
    )
    
    assert meta_result['meta_score'] is not None


# ============================================================================
# Test Model Info
# ============================================================================

def test_verifier_repr():
    """Test verifier string representation"""
    verifier = ProofVerifier(model_name="test-model")
    repr_str = repr(verifier)
    
    assert "BaseProofModel" in repr_str or "test-model" in repr_str


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])