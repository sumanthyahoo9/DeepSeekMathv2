"""
src/model/generator.py

Proof generator model for DeepSeekMath-V2.
Generates proofs with optional self-verification.
Refer to docs/MODEL_WRAPPERS.md for details about how and why this module exists
"""

from typing import Dict, List, Optional, Union
from pathlib import Path

from .base_model import BaseProofModel
from .model_utils import (
    extract_score,
    extract_solution_and_evaluation,
    clean_generated_text,
    get_greedy_config,
    get_sampling_config,
    truncate_at_stop_sequence
)
from ..utils.prompts import (
    get_proof_generation_prompt,
    get_proof_refinement_prompt
)


# ============================================================================
# Proof Generator
# ============================================================================

class ProofGenerator(BaseProofModel):
    """
    Proof generation model with self-verification.
    
    Generates proofs and optionally includes self-verification analysis.
    Supports sequential refinement and parallel search.
    
    Args:
        model_name: HuggingFace model name or path
        device: Device to load model on
        use_greedy: Whether to use greedy decoding by default
        enable_self_verification: Whether to include self-verification
        **kwargs: Additional arguments for BaseProofModel
    """
    
    def __init__(
        self,
        model_name: str = "deepseek-ai/DeepSeek-V3.2-Exp-SFT",
        device: str = "auto",
        use_greedy: bool = False,  # Sampling by default for generation
        enable_self_verification: bool = True,
        **kwargs
    ):
        super().__init__(model_name=model_name, device=device, **kwargs)
        self.use_greedy = use_greedy
        self.enable_self_verification = enable_self_verification
        
        # Default generation configs
        self.greedy_config = get_greedy_config()
        self.sampling_config = get_sampling_config(temperature=0.7, num_samples=1)
    
    def generate(
        self,
        problem: str,
        max_new_tokens: int = 4096,
        temperature: Optional[float] = None,
        use_self_verification: Optional[bool] = None
    ) -> Dict[str, Union[str, Optional[float]]]:
        """
        Generate a proof for the given problem.
        
        Args:
            problem: Mathematical problem to solve
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (None = use default)
            use_self_verification: Whether to include self-verification
            
        Returns:
            Dictionary with:
                - 'solution': Generated proof
                - 'evaluation': Self-evaluation (if enabled)
                - 'self_score': Self-assigned score (if enabled)
                - 'raw_output': Raw model output
        """
        # Determine if using self-verification
        if use_self_verification is None:
            use_self_verification = self.enable_self_verification
        
        # Generate prompt
        prompt = get_proof_generation_prompt(problem)
        
        # Mock mode
        if self.model is None or self.tokenizer is None:
            return self._mock_generate(problem, use_self_verification)
        
        # Prepare generation config
        if self.use_greedy:
            gen_config = self.greedy_config.copy()
        else:
            gen_config = self.sampling_config.copy()
            if temperature is not None:
                gen_config['temperature'] = temperature
        
        gen_config['max_new_tokens'] = max_new_tokens
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = inputs.to(self.get_device())
        
        # Generate proof
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **gen_config
            )
        
        # Decode output
        raw_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        raw_output = truncate_at_stop_sequence(raw_output)
        
        # Parse output
        return self._parse_generation_output(raw_output, use_self_verification)
    
    def generate_multiple(
        self,
        problem: str,
        num_samples: int = 4,
        max_new_tokens: int = 4096,
        temperature: float = 0.7
    ) -> List[Dict[str, Union[str, Optional[float]]]]:
        """
        Generate multiple candidate proofs (parallel search).
        
        Args:
            problem: Mathematical problem to solve
            num_samples: Number of proofs to generate
            max_new_tokens: Maximum tokens per generation
            temperature: Sampling temperature
            
        Returns:
            List of generation results
        """
        # Mock mode
        if self.model is None or self.tokenizer is None:
            return [
                self._mock_generate(problem, self.enable_self_verification)
                for _ in range(num_samples)
            ]
        
        # Generate prompt
        prompt = get_proof_generation_prompt(problem)
        
        # Prepare config for multiple samples
        gen_config = self.sampling_config.copy()
        gen_config['temperature'] = temperature
        gen_config['num_return_sequences'] = num_samples
        gen_config['max_new_tokens'] = max_new_tokens
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = inputs.to(self.get_device())
        
        # Generate multiple proofs
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **gen_config
            )
        
        # Decode and parse each output
        results = []
        for output in outputs:
            raw_output = self.tokenizer.decode(output, skip_special_tokens=True)
            raw_output = truncate_at_stop_sequence(raw_output)
            result = self._parse_generation_output(
                raw_output,
                self.enable_self_verification
            )
            results.append(result)
        
        return results
    
    def refine(
        self,
        problem: str,
        previous_proof: str,
        feedback: str,
        max_new_tokens: int = 4096
    ) -> Dict[str, Union[str, Optional[float]]]:
        """
        Refine a proof based on feedback (sequential refinement).
        
        Args:
            problem: Original problem
            previous_proof: Previous proof attempt
            feedback: Feedback/issues identified
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Refined proof result
        """
        # Generate refinement prompt
        prompt = get_proof_refinement_prompt(problem, previous_proof, feedback)
        
        # Mock mode
        if self.model is None or self.tokenizer is None:
            return self._mock_generate(problem, self.enable_self_verification)
        
        # Use greedy for refinement (more deterministic)
        gen_config = self.greedy_config.copy()
        gen_config['max_new_tokens'] = max_new_tokens
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = inputs.to(self.get_device())
        
        # Generate refined proof
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **gen_config
            )
        
        # Decode output
        raw_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        raw_output = truncate_at_stop_sequence(raw_output)
        
        # Parse output
        return self._parse_generation_output(raw_output, self.enable_self_verification)
    
    def _parse_generation_output(
        self,
        raw_output: str,
        has_self_verification: bool
    ) -> Dict[str, Union[str, Optional[float]]]:
        """
        Parse generated output into structured result.
        
        Args:
            raw_output: Raw model output
            has_self_verification: Whether output includes self-verification
            
        Returns:
            Parsed generation result
        """
        # Clean output
        cleaned = clean_generated_text(raw_output, remove_prompt=True)
        
        if has_self_verification:
            # Extract solution and evaluation sections
            parts = extract_solution_and_evaluation(cleaned)
            
            # Extract self-score from evaluation
            self_score = extract_score(parts.get('evaluation', ''))
            
            return {
                'solution': parts.get('solution', cleaned),
                'evaluation': parts.get('evaluation', ''),
                'self_score': self_score,
                'raw_output': raw_output
            }
        else:
            # No self-verification, just return solution
            return {
                'solution': cleaned,
                'evaluation': None,
                'self_score': None,
                'raw_output': raw_output
            }
    
    def _mock_generate(
        self,
        problem: str,
        use_self_verification: bool
    ) -> Dict[str, Union[str, Optional[float]]]:
        """
        Mock generation for testing without actual model.
        
        Args:
            problem: Problem statement
            use_self_verification: Whether to include self-verification
            
        Returns:
            Mock generation result
        """
        # Generate a simple mock proof
        solution = f"Mock proof for: {problem[:50]}...\n\nStep 1: Analysis\nStep 2: Construction\nStep 3: Verification\n\nTherefore, the proof is complete."
        
        if use_self_verification:
            evaluation = "The proof is well-structured and logically sound. No issues detected. \\boxed{1}"
            self_score = 1.0
        else:
            evaluation = None
            self_score = None
        
        return {
            'solution': solution,
            'evaluation': evaluation,
            'self_score': self_score,
            'raw_output': f"## Solution\n{solution}" + (f"\n\n## Self Evaluation\n{evaluation}" if evaluation else "")
        }


# ============================================================================
# Utility Functions
# ============================================================================

def load_generator(checkpoint_path: Union[str, Path]) -> ProofGenerator:
    """
    Load generator from checkpoint.
    
    Args:
        checkpoint_path: Path to saved checkpoint
        
    Returns:
        Loaded ProofGenerator instance
    """
    generator = ProofGenerator()
    generator.load_checkpoint(checkpoint_path)
    return generator


# Add torch import for actual model usage
try:
    import torch
except ImportError:
    torch = None


# ============================================================================
# Export
# ============================================================================

__all__ = [
    'ProofGenerator',
    'load_generator',
]