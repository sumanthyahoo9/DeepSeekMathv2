"""
tests/test_data_collator.py

Unit tests for data collators.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.data_collator import (
    VerificationCollator,
    MetaVerificationCollator,
    GenerationCollator,
    InferenceCollator,
    get_collator,
    estimate_batch_memory
)


# ============================================================================
# Test VerificationCollator
# ============================================================================

def test_verification_collator_basic():
    """Test verification collator without tokenizer"""
    collator = VerificationCollator(tokenizer=None)
    
    batch = [
        {'problem': 'P1', 'proof': 'Proof1', 'score': 1.0},
        {'problem': 'P2', 'proof': 'Proof2', 'score': 0.5},
        {'problem': 'P3', 'proof': 'Proof3', 'score': 0.0}
    ]
    
    result = collator(batch)
    
    assert 'texts' in result
    assert 'scores' in result
    assert result['batch_size'] == 3
    assert len(result['texts']) == 3
    assert result['scores'] == [1.0, 0.5, 0.0]
    
    # Check text formatting
    assert 'Problem: P1' in result['texts'][0]
    assert 'Proof: Proof1' in result['texts'][0]


def test_verification_collator_text_formatting():
    """Test that problem and proof are properly formatted"""
    collator = VerificationCollator(tokenizer=None)
    
    batch = [
        {
            'problem': 'Prove that 2+2=4',
            'proof': 'By definition of addition, 2+2=4. QED.',
            'score': 1.0
        }
    ]
    
    result = collator(batch)
    text = result['texts'][0]
    
    # Check structure
    assert text.startswith('Problem: ')
    assert '\n\nProof: ' in text
    assert 'Prove that 2+2=4' in text
    assert 'By definition' in text


# ============================================================================
# Test MetaVerificationCollator
# ============================================================================

def test_meta_verification_collator_basic():
    """Test meta-verification collator without tokenizer"""
    collator = MetaVerificationCollator(tokenizer=None)
    
    batch = [
        {
            'problem': 'P1',
            'proof': 'Proof1',
            'analysis': 'Analysis1',
            'meta_score': 1.0
        },
        {
            'problem': 'P2',
            'proof': 'Proof2',
            'analysis': 'Analysis2',
            'meta_score': 0.5
        }
    ]
    
    result = collator(batch)
    
    assert 'texts' in result
    assert 'meta_scores' in result
    assert result['batch_size'] == 2
    assert len(result['texts']) == 2
    assert result['meta_scores'] == [1.0, 0.5]


def test_meta_verification_collator_text_formatting():
    """Test that problem, proof, and analysis are properly formatted"""
    collator = MetaVerificationCollator(tokenizer=None)
    
    batch = [
        {
            'problem': 'Test problem',
            'proof': 'Test proof',
            'analysis': 'Test analysis',
            'meta_score': 1.0
        }
    ]
    
    result = collator(batch)
    text = result['texts'][0]
    
    # Check structure
    assert text.startswith('Problem: ')
    assert '\n\nProof: ' in text
    assert '\n\nAnalysis: ' in text
    assert 'Test problem' in text
    assert 'Test proof' in text
    assert 'Test analysis' in text


# ============================================================================
# Test GenerationCollator
# ============================================================================

def test_generation_collator_basic():
    """Test generation collator without tokenizer"""
    collator = GenerationCollator(tokenizer=None)
    
    batch = [
        {'problem': 'P1'},
        {'problem': 'P2'},
        {'problem': 'P3'}
    ]
    
    result = collator(batch)
    
    assert 'texts' in result
    assert result['batch_size'] == 3
    assert len(result['texts']) == 3


def test_generation_collator_text_formatting():
    """Test that problem is formatted as prompt"""
    collator = GenerationCollator(tokenizer=None)
    
    batch = [
        {'problem': 'Prove that sqrt(2) is irrational'}
    ]
    
    result = collator(batch)
    text = result['texts'][0]
    
    # Check structure - should be a prompt for generation
    assert text.startswith('Problem: ')
    assert '\n\nSolution:' in text
    assert 'Prove that sqrt(2) is irrational' in text


# ============================================================================
# Test InferenceCollator
# ============================================================================

def test_inference_collator_generation_mode():
    """Test inference collator in generation mode (problem only)"""
    collator = InferenceCollator(tokenizer=None)
    
    batch = [
        {'problem': 'P1', 'sample_id': 'id1'},
        {'problem': 'P2', 'sample_id': 'id2'}
    ]
    
    result = collator(batch)
    
    assert 'texts' in result
    assert 'metadata' in result
    assert result['batch_size'] == 2
    
    # Check metadata is preserved
    assert result['metadata'] == batch
    
    # Check text format (should be generation prompt)
    assert 'Solution:' in result['texts'][0]


def test_inference_collator_verification_mode():
    """Test inference collator in verification mode (problem + proof)"""
    collator = InferenceCollator(tokenizer=None)
    
    batch = [
        {
            'problem': 'P1',
            'proof': 'Proof1',
            'sample_id': 'id1',
            'category': 'algebra'
        }
    ]
    
    result = collator(batch)
    
    assert 'texts' in result
    assert 'metadata' in result
    
    # Check metadata is preserved (including extra fields)
    assert result['metadata'][0]['category'] == 'algebra'
    
    # Check text format (should be verification format)
    assert 'Problem: P1' in result['texts'][0]
    assert 'Proof: Proof1' in result['texts'][0]


def test_inference_collator_preserves_metadata():
    """Test that inference collator preserves all metadata fields"""
    collator = InferenceCollator(tokenizer=None)
    
    batch = [
        {
            'problem': 'P1',
            'sample_id': 'test_id',
            'category': 'geometry',
            'difficulty': 'hard',
            'custom_field': 'custom_value'
        }
    ]
    
    result = collator(batch)
    metadata = result['metadata'][0]
    
    assert metadata['sample_id'] == 'test_id'
    assert metadata['category'] == 'geometry'
    assert metadata['difficulty'] == 'hard'
    assert metadata['custom_field'] == 'custom_value'


# ============================================================================
# Test Batch Sizes
# ============================================================================

def test_collator_single_sample():
    """Test collators with batch size of 1"""
    collator = VerificationCollator(tokenizer=None)
    
    batch = [{'problem': 'P1', 'proof': 'Proof1', 'score': 1.0}]
    result = collator(batch)
    
    assert result['batch_size'] == 1
    assert len(result['texts']) == 1


def test_collator_large_batch():
    """Test collators with larger batch"""
    collator = GenerationCollator(tokenizer=None)
    
    batch = [{'problem': f'P{i}'} for i in range(32)]
    result = collator(batch)
    
    assert result['batch_size'] == 32
    assert len(result['texts']) == 32


# ============================================================================
# Test Factory Function
# ============================================================================

def test_get_collator_verification():
    """Test factory function for verification collator"""
    collator = get_collator('verification', tokenizer=None)
    assert isinstance(collator, VerificationCollator)


def test_get_collator_meta_verification():
    """Test factory function for meta-verification collator"""
    collator = get_collator('meta_verification', tokenizer=None)
    assert isinstance(collator, MetaVerificationCollator)


def test_get_collator_generation():
    """Test factory function for generation collator"""
    collator = get_collator('generation', tokenizer=None)
    assert isinstance(collator, GenerationCollator)


def test_get_collator_inference():
    """Test factory function for inference collator"""
    collator = get_collator('inference', tokenizer=None)
    assert isinstance(collator, InferenceCollator)


def test_get_collator_invalid_type():
    """Test factory function with invalid type"""
    with pytest.raises(ValueError):
        get_collator('invalid_type', tokenizer=None)


def test_get_collator_with_kwargs():
    """Test factory function with additional kwargs"""
    collator = get_collator(
        'verification',
        tokenizer=None,
        max_length=64000,
        padding='max_length'
    )
    
    assert collator.max_length == 64000
    assert collator.padding == 'max_length'


# ============================================================================
# Test Memory Estimation
# ============================================================================

def test_estimate_batch_memory_small():
    """Test memory estimation for small batch"""
    memory_gb = estimate_batch_memory(
        batch_size=1,
        max_length=2048,
        model_size_gb=50.0
    )
    
    # Should be > model size (50 GB) due to activations
    assert memory_gb > 50.0
    # But not too large for small batch
    assert memory_gb < 100.0


def test_estimate_batch_memory_large():
    """Test memory estimation for large batch"""
    memory_gb = estimate_batch_memory(
        batch_size=8,
        max_length=128000,
        model_size_gb=50.0
    )
    
    # Should be significantly larger for big batch
    assert memory_gb > 100.0


def test_estimate_batch_memory_scales():
    """Test that memory estimate scales with batch size"""
    mem_small = estimate_batch_memory(
        batch_size=1,
        max_length=1024,
        model_size_gb=50.0
    )
    
    mem_large = estimate_batch_memory(
        batch_size=4,
        max_length=1024,
        model_size_gb=50.0
    )
    
    # Larger batch should require more memory
    assert mem_large > mem_small


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])