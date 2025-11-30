"""
tests/test_model_utils.py

Unit tests for model utilities.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.model_utils import (
    create_generation_config,
    get_greedy_config,
    get_sampling_config,
    clean_generated_text,
    extract_solution_and_evaluation,
    truncate_at_stop_sequence,
    extract_score,
    extract_multiple_scores,
    batch_texts,
    flatten_batches,
    format_parameter_count,
    get_model_family,
    validate_response_format,
    check_response_completeness,
    summarize_batch_results
)


# ============================================================================
# Test Generation Config
# ============================================================================

def test_create_generation_config_defaults():
    """Test creating generation config with defaults"""
    config = create_generation_config()
    
    assert config['max_new_tokens'] == 2048
    assert config['temperature'] == 0.7
    assert config['top_p'] == 0.9
    assert config['do_sample'] is True


def test_create_generation_config_custom():
    """Test creating generation config with custom params"""
    config = create_generation_config(
        max_new_tokens=4096,
        temperature=0.9,
        top_p=0.95
    )
    
    assert config['max_new_tokens'] == 4096
    assert config['temperature'] == 0.9
    assert config['top_p'] == 0.95


def test_get_greedy_config():
    """Test greedy decoding config"""
    config = get_greedy_config(max_new_tokens=1024)
    
    assert config['max_new_tokens'] == 1024
    assert config['do_sample'] is False
    assert config['num_return_sequences'] == 1


def test_get_sampling_config():
    """Test sampling config"""
    config = get_sampling_config(
        max_new_tokens=2048,
        temperature=0.8,
        num_samples=4
    )
    
    assert config['max_new_tokens'] == 2048
    assert config['temperature'] == 0.8
    assert config['num_return_sequences'] == 4
    assert config['do_sample'] is True


# ============================================================================
# Test Output Formatting
# ============================================================================

def test_clean_generated_text_basic():
    """Test basic text cleaning"""
    text = "  \n  Some generated text  \n  "
    cleaned = clean_generated_text(text, remove_prompt=False)
    
    assert cleaned == "Some generated text"


def test_clean_generated_text_with_solution_marker():
    """Test removing solution marker"""
    text = "Problem: ...\n\nSolution: This is the proof"
    cleaned = clean_generated_text(text, remove_prompt=True)
    
    assert cleaned == "This is the proof"
    assert "Solution:" not in cleaned


def test_clean_generated_text_with_markdown():
    """Test removing markdown solution header"""
    text = "Some prompt\n\n## Solution\nThis is the proof"
    cleaned = clean_generated_text(text, remove_prompt=True)
    
    assert cleaned == "This is the proof"
    assert "## Solution" not in cleaned


def test_extract_solution_and_evaluation():
    """Test extracting solution and evaluation sections"""
    text = """
## Solution
This is the proof.
Step 1: ...
Step 2: ...

## Self Evaluation
Here is my evaluation of the solution:
The proof is correct.
"""
    
    result = extract_solution_and_evaluation(text)
    
    assert 'solution' in result
    assert 'evaluation' in result
    assert 'This is the proof' in result['solution']
    assert 'evaluation of the solution' in result['evaluation']


def test_truncate_at_stop_sequence():
    """Test truncating at stop sequences"""
    text = "Some generated text<|endoftext|>Extra text"
    truncated = truncate_at_stop_sequence(text)
    
    assert truncated == "Some generated text"
    assert "<|endoftext|>" not in truncated


def test_truncate_at_custom_stop():
    """Test truncating with custom stop sequences"""
    text = "Generated text[STOP]More text"
    truncated = truncate_at_stop_sequence(text, stop_sequences=['[STOP]'])
    
    assert truncated == "Generated text"


# ============================================================================
# Test Score Extraction
# ============================================================================

def test_extract_score_valid():
    """Test extracting valid scores"""
    assert extract_score("Analysis... \\boxed{1}") == 1.0
    assert extract_score("Analysis... \\boxed{0.5}") == 0.5
    assert extract_score("Analysis... \\boxed{0}") == 0.0


def test_extract_score_invalid():
    """Test extracting invalid scores"""
    assert extract_score("No score here") is None
    assert extract_score("\\boxed{2}") is None  # Invalid score
    assert extract_score("\\boxed{abc}") is None  # Non-numeric


def test_extract_score_multiple_boxes():
    """Test extracting when multiple boxed values exist"""
    text = "First: \\boxed{0.5} ... Final: \\boxed{1}"
    # Should return last score
    assert extract_score(text) == 1.0


def test_extract_multiple_scores():
    """Test extracting scores from multiple texts"""
    texts = [
        "Score: \\boxed{1}",
        "Score: \\boxed{0.5}",
        "No score",
        "Score: \\boxed{0}"
    ]
    
    scores = extract_multiple_scores(texts)
    
    assert scores == [1.0, 0.5, None, 0.0]


# ============================================================================
# Test Batch Processing
# ============================================================================

def test_batch_texts_even():
    """Test batching with even division"""
    texts = ['a', 'b', 'c', 'd', 'e', 'f']
    batches = batch_texts(texts, batch_size=2)
    
    assert len(batches) == 3
    assert batches[0] == ['a', 'b']
    assert batches[1] == ['c', 'd']
    assert batches[2] == ['e', 'f']


def test_batch_texts_uneven():
    """Test batching with uneven division"""
    texts = ['a', 'b', 'c', 'd', 'e']
    batches = batch_texts(texts, batch_size=2)
    
    assert len(batches) == 3
    assert batches[2] == ['e']  # Last batch has 1 item


def test_flatten_batches():
    """Test flattening batches"""
    batches = [['a', 'b'], ['c', 'd'], ['e']]
    flattened = flatten_batches(batches)
    
    assert flattened == ['a', 'b', 'c', 'd', 'e']


def test_batch_and_flatten_roundtrip():
    """Test that batch → flatten is identity"""
    original = ['a', 'b', 'c', 'd', 'e']
    batches = batch_texts(original, batch_size=2)
    result = flatten_batches(batches)
    
    assert result == original


# ============================================================================
# Test Model Utilities
# ============================================================================

def test_format_parameter_count_billions():
    """Test formatting billions of parameters"""
    assert format_parameter_count(7_000_000_000) == "7.0B"
    assert format_parameter_count(175_000_000_000) == "175.0B"


def test_format_parameter_count_millions():
    """Test formatting millions of parameters"""
    assert format_parameter_count(350_000_000) == "350.0M"
    assert format_parameter_count(1_500_000) == "1.5M"


def test_format_parameter_count_thousands():
    """Test formatting thousands of parameters"""
    assert format_parameter_count(5_000) == "5.0K"


def test_format_parameter_count_small():
    """Test formatting small parameter counts"""
    assert format_parameter_count(500) == "500"


def test_get_model_family_deepseek():
    """Test identifying DeepSeek models"""
    assert get_model_family("deepseek-ai/DeepSeek-V3.2-Exp-SFT") == "deepseek"
    assert get_model_family("DeepSeek-Math-Base") == "deepseek"


def test_get_model_family_other():
    """Test identifying other model families"""
    assert get_model_family("meta-llama/Llama-2-7b") == "llama"
    assert get_model_family("mistralai/Mistral-7B") == "mistral"
    assert get_model_family("Qwen/Qwen-7B") == "qwen"


def test_get_model_family_unknown():
    """Test unknown model family"""
    assert get_model_family("some-random-model") == "unknown"


# ============================================================================
# Test Validation
# ============================================================================

def test_validate_response_format_complete():
    """Test validating complete response"""
    response = """
## Solution
Proof here

## Self Evaluation
Evaluation here
"""
    
    result = validate_response_format(response)
    
    assert result['Solution'] is True
    assert result['Self Evaluation'] is True


def test_validate_response_format_incomplete():
    """Test validating incomplete response"""
    response = "## Solution\nOnly solution, no evaluation"
    
    result = validate_response_format(response)
    
    assert result['Solution'] is True
    assert result['Self Evaluation'] is False


def test_check_response_completeness_complete():
    """Test checking complete response"""
    response = "A long response... " * 10 + "\\boxed{1}"
    
    assert check_response_completeness(response) is True


def test_check_response_completeness_too_short():
    """Test checking too short response"""
    response = "Too short"
    
    assert check_response_completeness(response, min_length=50) is False


def test_check_response_completeness_no_score():
    """Test checking response without score"""
    long_response = "A long response... " * 10 + "final overall score should be: 1"
    
    assert check_response_completeness(long_response) is True


# ============================================================================
# Test Debug Utilities
# ============================================================================

def test_summarize_batch_results_all_valid():
    """Test summarizing batch with all valid scores"""
    scores = [1.0, 0.5, 0.0, 1.0, 0.5]
    summary = summarize_batch_results(scores)
    
    assert summary['total'] == 5
    assert summary['valid'] == 5
    assert summary['invalid'] == 0
    assert summary['success_rate'] == 1.0
    # Mean of [1.0, 0.5, 0.0, 1.0, 0.5] = 3.0 / 5 = 0.6
    assert summary['mean_score'] == 0.6


def test_summarize_batch_results_with_none():
    """Test summarizing batch with invalid scores"""
    scores = [1.0, None, 0.5, None, 1.0]
    summary = summarize_batch_results(scores)
    
    assert summary['total'] == 5
    assert summary['valid'] == 3
    assert summary['invalid'] == 2
    assert summary['success_rate'] == 0.6


def test_summarize_batch_results_distribution():
    """Test score distribution in summary"""
    scores = [1.0, 1.0, 0.5, 0.0, 1.0]
    summary = summarize_batch_results(scores, show_distribution=True)
    
    assert 'distribution' in summary
    assert summary['distribution']['0.0'] == 1
    assert summary['distribution']['0.5'] == 1
    assert summary['distribution']['1.0'] == 3


def test_summarize_batch_results_empty():
    """Test summarizing empty batch"""
    summary = summarize_batch_results([])
    
    assert summary['total'] == 0
    assert summary['success_rate'] == 0.0


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])