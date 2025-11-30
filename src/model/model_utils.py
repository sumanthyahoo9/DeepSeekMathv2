"""
Utilities/Model helper functions
"""
"""
src/model/model_utils.py

Helper utilities for model operations in DeepSeekMath-V2.
Handles generation configs, output formatting, and common utilities.
"""

from typing import Any, Dict, List, Optional
import re

try:
    from transformers import GenerationConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    GenerationConfig = None


# ============================================================================
# Generation Configuration
# ============================================================================

def create_generation_config(
    max_new_tokens: int = 2048,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    do_sample: bool = True,
    num_return_sequences: int = 1,
    repetition_penalty: float = 1.0,
    **kwargs
) -> Dict[str, Any]:
    """
    Create generation configuration for model inference.
    
    Args:
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_p: Nucleus sampling threshold
        top_k: Top-k sampling
        do_sample: Whether to use sampling (vs greedy)
        num_return_sequences: Number of sequences to generate
        repetition_penalty: Penalty for repeating tokens
        **kwargs: Additional generation parameters
        
    Returns:
        Dictionary with generation config
    """
    config = {
        'max_new_tokens': max_new_tokens,
        'temperature': temperature,
        'top_p': top_p,
        'top_k': top_k,
        'do_sample': do_sample,
        'num_return_sequences': num_return_sequences,
        'repetition_penalty': repetition_penalty
    }
    
    # Add any additional kwargs
    config.update(kwargs)
    
    return config


def get_greedy_config(max_new_tokens: int = 2048) -> Dict[str, Any]:
    """
    Get configuration for greedy decoding (deterministic).
    
    Args:
        max_new_tokens: Maximum tokens to generate
        
    Returns:
        Greedy generation config
    """
    return create_generation_config(
        max_new_tokens=max_new_tokens,
        temperature=1.0,
        do_sample=False,
        num_return_sequences=1
    )


def get_sampling_config(
    max_new_tokens: int = 2048,
    temperature: float = 0.7,
    num_samples: int = 1
) -> Dict[str, Any]:
    """
    Get configuration for sampling-based generation.
    
    Args:
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        num_samples: Number of samples to generate
        
    Returns:
        Sampling generation config
    """
    return create_generation_config(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        num_return_sequences=num_samples
    )


# ============================================================================
# Output Formatting
# ============================================================================

def clean_generated_text(text: str, remove_prompt: bool = True) -> str:
    """
    Clean up generated text.
    
    Args:
        text: Raw generated text
        remove_prompt: Whether to try removing the prompt
        
    Returns:
        Cleaned text
    """
    # Remove leading/trailing whitespace
    text = text.strip()
    
    # Try to remove the prompt if requested
    if remove_prompt:
        # Look for "Solution:" or "## Solution" markers
        if "## Solution" in text:
            text = text.split("## Solution", 1)[1].strip()
        elif "Solution:" in text:
            text = text.split("Solution:", 1)[1].strip()
    
    return text


def extract_solution_and_evaluation(text: str) -> Dict[str, str]:
    """
    Extract solution and self-evaluation sections from generated text.
    
    Args:
        text: Generated text with ## Solution and ## Self Evaluation sections
        
    Returns:
        Dictionary with 'solution' and 'evaluation' keys
    """
    result = {'solution': '', 'evaluation': ''}
    
    # Split by markdown headers
    parts = text.split('## ')
    
    for part in parts:
        part = part.strip()
        if part.lower().startswith('solution'):
            content = part[len('solution'):].strip()
            # Stop at next ## header
            if '## ' in content:
                content = content.split('## ')[0].strip()
            result['solution'] = content
        
        elif part.lower().startswith('self evaluation'):
            content = part[len('self evaluation'):].strip()
            result['evaluation'] = content
    
    return result


def truncate_at_stop_sequence(
    text: str,
    stop_sequences: Optional[List[str]] = None
) -> str:
    """
    Truncate text at first occurrence of stop sequence.
    
    Args:
        text: Generated text
        stop_sequences: List of sequences to stop at
        
    Returns:
        Truncated text
    """
    if stop_sequences is None:
        # Default stop sequences for proof generation
        stop_sequences = ['<|endoftext|>', '</s>', '<|im_end|>']
    
    for stop_seq in stop_sequences:
        if stop_seq in text:
            text = text.split(stop_seq)[0]
    
    return text.strip()


# ============================================================================
# Score Extraction
# ============================================================================

def extract_score(text: str) -> Optional[float]:
    """
    Extract score from text (looks for \\boxed{score} pattern).
    
    Args:
        text: Text containing score
        
    Returns:
        Extracted score (0, 0.5, or 1) or None if not found
    """
    # Look for \boxed{score} pattern
    pattern = r'\\boxed\{([0-9.]+)\}'
    matches = re.findall(pattern, text)
    
    if matches:
        try:
            score = float(matches[-1])  # Take last match
            if score in [0, 0.5, 1, 0.0, 1.0]:
                return float(score) if score != 1.0 else 1.0
        except ValueError:
            pass
    
    return None


def extract_multiple_scores(texts: List[str]) -> List[Optional[float]]:
    """
    Extract scores from multiple texts.
    
    Args:
        texts: List of texts containing scores
        
    Returns:
        List of extracted scores
    """
    return [extract_score(text) for text in texts]


# ============================================================================
# Batch Processing Utilities
# ============================================================================

def batch_texts(
    texts: List[str],
    batch_size: int
) -> List[List[str]]:
    """
    Split list of texts into batches.
    
    Args:
        texts: List of texts
        batch_size: Size of each batch
        
    Returns:
        List of batches
    """
    batches = []
    for i in range(0, len(texts), batch_size):
        batches.append(texts[i:i + batch_size])
    return batches


def flatten_batches(batches: List[List[Any]]) -> List[Any]:
    """
    Flatten list of batches into single list.
    
    Args:
        batches: List of batches
        
    Returns:
        Flattened list
    """
    return [item for batch in batches for item in batch]


# ============================================================================
# Model Parameter Utilities
# ============================================================================

def format_parameter_count(num_params: int) -> str:
    """
    Format parameter count in human-readable form.
    
    Args:
        num_params: Number of parameters
        
    Returns:
        Formatted string (e.g., "7.2B", "175M")
    """
    if num_params >= 1_000_000_000:
        return f"{num_params / 1_000_000_000:.1f}B"
    elif num_params >= 1_000_000:
        return f"{num_params / 1_000_000:.1f}M"
    elif num_params >= 1_000:
        return f"{num_params / 1_000:.1f}K"
    else:
        return str(num_params)


def get_model_family(model_name: str) -> str:
    """
    Get model family from model name.
    
    Args:
        model_name: Model name (e.g., "deepseek-ai/DeepSeek-V3.2-Exp-SFT")
        
    Returns:
        Model family name (e.g., "deepseek")
    """
    model_name_lower = model_name.lower()
    
    if 'deepseek' in model_name_lower:
        return 'deepseek'
    elif 'llama' in model_name_lower:
        return 'llama'
    elif 'mistral' in model_name_lower:
        return 'mistral'
    elif 'qwen' in model_name_lower:
        return 'qwen'
    else:
        return 'unknown'


# ============================================================================
# Response Validation
# ============================================================================

def validate_response_format(
    response: str,
    required_sections: Optional[List[str]] = None
) -> Dict[str, bool]:
    """
    Validate that response contains required sections.
    
    Args:
        response: Generated response
        required_sections: List of required section names
        
    Returns:
        Dictionary mapping section names to presence (bool)
    """
    if required_sections is None:
        required_sections = ['Solution', 'Self Evaluation']
    
    result = {}
    for section in required_sections:
        # Check for markdown header
        pattern = f"## {section}"
        result[section] = pattern in response
    
    return result


def check_response_completeness(response: str, min_length: int = 50) -> bool:
    """
    Check if response is complete (not truncated).
    
    Args:
        response: Generated response
        min_length: Minimum acceptable length
        
    Returns:
        True if response appears complete
    """
    # Check minimum length
    if len(response) < min_length:
        return False
    
    # Check if ends with expected markers
    has_score = '\\boxed{' in response
    has_evaluation_end = 'final overall score should be' in response.lower()
    
    return has_score or has_evaluation_end


# ============================================================================
# Debug Utilities
# ============================================================================

def print_generation_info(
    prompt_length: int,
    generated_length: int,
    config: Dict[str, Any]
) -> None:
    """
    Print information about generation for debugging.
    
    Args:
        prompt_length: Length of input prompt (tokens)
        generated_length: Length of generated text (tokens)
        config: Generation config used
    """
    print("=" * 60)
    print("Generation Info:")
    print(f"  Prompt tokens: {prompt_length}")
    print(f"  Generated tokens: {generated_length}")
    print(f"  Total tokens: {prompt_length + generated_length}")
    print(f"  Temperature: {config.get('temperature', 'N/A')}")
    print(f"  Top-p: {config.get('top_p', 'N/A')}")
    print(f"  Max new tokens: {config.get('max_new_tokens', 'N/A')}")
    print("=" * 60)


def summarize_batch_results(
    scores: List[Optional[float]],
    show_distribution: bool = True
) -> Dict[str, Any]:
    """
    Summarize results from a batch of generations.
    
    Args:
        scores: List of extracted scores
        show_distribution: Whether to show score distribution
        
    Returns:
        Summary statistics
    """
    valid_scores = [s for s in scores if s is not None]
    
    summary = {
        'total': len(scores),
        'valid': len(valid_scores),
        'invalid': len(scores) - len(valid_scores),
        'success_rate': len(valid_scores) / len(scores) if scores else 0.0
    }
    
    if valid_scores:
        summary['mean_score'] = sum(valid_scores) / len(valid_scores)
        summary['min_score'] = min(valid_scores)
        summary['max_score'] = max(valid_scores)
    
    if show_distribution and valid_scores:
        summary['distribution'] = {
            '0.0': valid_scores.count(0.0),
            '0.5': valid_scores.count(0.5),
            '1.0': valid_scores.count(1.0)
        }
    
    return summary


# ============================================================================
# Export
# ============================================================================

__all__ = [
    # Generation config
    'create_generation_config',
    'get_greedy_config',
    'get_sampling_config',
    
    # Output formatting
    'clean_generated_text',
    'extract_solution_and_evaluation',
    'truncate_at_stop_sequence',
    
    # Score extraction
    'extract_score',
    'extract_multiple_scores',
    
    # Batch processing
    'batch_texts',
    'flatten_batches',
    
    # Model utilities
    'format_parameter_count',
    'get_model_family',
    
    # Validation
    'validate_response_format',
    'check_response_completeness',
    
    # Debug
    'print_generation_info',
    'summarize_batch_results',
]