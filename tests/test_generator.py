"""
tests/test_generator.py

Unit tests for proof generator model.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.generator import (
    ProofGenerator,
    load_generator
)


# ============================================================================
# Test ProofGenerator Initialization
# ============================================================================

def test_generator_init_default():
    """Test generator initialization with defaults"""
    generator = ProofGenerator()
    
    assert generator.model_name == "deepseek-ai/DeepSeek-V3.2-Exp-SFT"
    assert generator.use_greedy is False  # Sampling by default
    assert generator.enable_self_verification is True


def test_generator_init_custom():
    """Test generator initialization with custom params"""
    generator = ProofGenerator(
        model_name="test-model",
        device="cpu",
        use_greedy=True,
        enable_self_verification=False
    )
    
    assert generator.model_name == "test-model"
    assert generator.use_greedy is True
    assert generator.enable_self_verification is False


# ============================================================================
# Test Single Generation
# ============================================================================

def test_generate_basic():
    """Test basic proof generation"""
    generator = ProofGenerator()
    
    problem = "Prove that 2+2=4"
    result = generator.generate(problem)
    
    assert 'solution' in result
    assert 'evaluation' in result
    assert 'self_score' in result
    assert 'raw_output' in result
    assert isinstance(result['solution'], str)


def test_generate_with_self_verification():
    """Test generation with self-verification enabled"""
    generator = ProofGenerator(enable_self_verification=True)
    
    problem = "Test problem"
    result = generator.generate(problem)
    
    # Should have self-verification components
    assert result['evaluation'] is not None
    assert result['self_score'] is not None


def test_generate_without_self_verification():
    """Test generation with self-verification disabled"""
    generator = ProofGenerator(enable_self_verification=False)
    
    problem = "Test problem"
    result = generator.generate(problem, use_self_verification=False)
    
    # Should not have self-verification components
    assert result['evaluation'] is None
    assert result['self_score'] is None


def test_generate_override_self_verification():
    """Test overriding default self-verification setting"""
    generator = ProofGenerator(enable_self_verification=False)
    
    # Override default
    result = generator.generate("Test", use_self_verification=True)
    
    # Should have self-verification despite default=False
    assert result['evaluation'] is not None
    assert result['self_score'] is not None


# ============================================================================
# Test Multiple Generation (Parallel Search)
# ============================================================================

def test_generate_multiple():
    """Test generating multiple candidate proofs"""
    generator = ProofGenerator()
    
    problem = "Test problem"
    results = generator.generate_multiple(problem, num_samples=3)
    
    assert len(results) == 3
    assert all('solution' in r for r in results)


def test_generate_multiple_with_temperature():
    """Test multiple generation with custom temperature"""
    generator = ProofGenerator()
    
    problem = "Test problem"
    results = generator.generate_multiple(
        problem,
        num_samples=2,
        temperature=0.9
    )
    
    assert len(results) == 2


# ============================================================================
# Test Refinement (Sequential Refinement)
# ============================================================================

def test_refine_basic():
    """Test basic proof refinement"""
    generator = ProofGenerator()
    
    problem = "Prove that sqrt(2) is irrational"
    previous_proof = "Initial proof attempt..."
    feedback = "The proof has a gap in step 3."
    
    result = generator.refine(problem, previous_proof, feedback)
    
    assert 'solution' in result
    assert isinstance(result['solution'], str)


def test_refine_with_self_verification():
    """Test refinement includes self-verification"""
    generator = ProofGenerator(enable_self_verification=True)
    
    problem = "Test problem"
    previous = "Previous attempt"
    feedback = "Needs improvement"
    
    result = generator.refine(problem, previous, feedback)
    
    # Should include self-verification
    assert result['evaluation'] is not None or result['self_score'] is not None


# ============================================================================
# Test Mock Generation Details
# ============================================================================

def test_mock_generation_structure():
    """Test structure of mock-generated proof"""
    generator = ProofGenerator()
    
    problem = "Test problem with specific content"
    result = generator.generate(problem)
    
    # Mock should include problem reference
    assert 'Mock proof' in result['solution']
    assert 'Step 1' in result['solution']
    assert 'Step 2' in result['solution']
    assert 'Step 3' in result['solution']


def test_mock_self_verification_score():
    """Test mock self-verification always gives perfect score"""
    generator = ProofGenerator(enable_self_verification=True)
    
    result = generator.generate("Any problem")
    
    # Mock always gives score of 1.0
    assert result['self_score'] == 1.0


# ============================================================================
# Test Generation Configs
# ============================================================================

def test_greedy_generation():
    """Test generator using greedy decoding"""
    generator = ProofGenerator(use_greedy=True)
    
    problem = "Test problem"
    result = generator.generate(problem)
    
    # Should still work in greedy mode
    assert 'solution' in result


def test_sampling_generation():
    """Test generator using sampling"""
    generator = ProofGenerator(use_greedy=False)
    
    problem = "Test problem"
    result = generator.generate(problem, temperature=0.8)
    
    # Should still work with sampling
    assert 'solution' in result


# ============================================================================
# Test Integration Scenarios
# ============================================================================

def test_generate_verify_refine_loop():
    """Test complete generate-verify-refine loop"""
    from src.model.verifier import ProofVerifier
    
    generator = ProofGenerator(enable_self_verification=True)
    verifier = ProofVerifier()
    
    problem = "Prove that 2+2=4"
    
    # Step 1: Generate initial proof
    initial = generator.generate(problem)
    assert initial['solution'] is not None
    
    # Step 2: Verify the proof
    verification = verifier.verify(problem, initial['solution'])
    assert verification['score'] is not None
    
    # Step 3: If score < 1, refine
    if verification['score'] < 1.0:
        refined = generator.refine(
            problem,
            initial['solution'],
            verification['analysis']
        )
        assert refined['solution'] is not None


def test_parallel_search_with_verification():
    """Test parallel search followed by verification"""
    from src.model.verifier import ProofVerifier
    
    generator = ProofGenerator()
    verifier = ProofVerifier()
    
    problem = "Test problem"
    
    # Generate multiple candidates
    candidates = generator.generate_multiple(problem, num_samples=3)
    
    # Verify each candidate
    scores = []
    for candidate in candidates:
        verification = verifier.verify(problem, candidate['solution'])
        scores.append(verification['score'])
    
    # Should have verification for all candidates
    assert len(scores) == 3
    assert all(s is not None for s in scores)


# ============================================================================
# Test Error Handling
# ============================================================================

def test_generate_empty_problem():
    """Test generation with empty problem"""
    generator = ProofGenerator()
    
    # Should still work, just generate generic proof
    result = generator.generate("")
    assert 'solution' in result


# ============================================================================
# Test Model Info
# ============================================================================

def test_generator_repr():
    """Test generator string representation"""
    generator = ProofGenerator(model_name="test-model")
    repr_str = repr(generator)
    
    assert "BaseProofModel" in repr_str or "test-model" in repr_str


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])