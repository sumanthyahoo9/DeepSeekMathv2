"""
src/model/verifier.py

Proof verifier model for DeepSeekMath-V2.
Analyzes proofs and assigns scores (0, 0.5, or 1).
Refer to docs/MODEL_WRAPPERS.md for details about how and why this module exists
"""

from typing import Dict, List, Optional, Union
from pathlib import Path

from .base_model import BaseProofModel
from .model_utils import (
    extract_score,
    clean_generated_text,
    get_greedy_config,
)
from ..utils.prompts import (
    get_proof_verification_prompt,
    get_meta_verification_prompt
)
import torch


# ============================================================================
# Proof Verifier
# ============================================================================

class ProofVerifier(BaseProofModel):
    """
    Proof verification model.
    
    Takes (problem, proof) as input and generates:
    - Detailed analysis of the proof
    - Score (0, 0.5, or 1)
    
    Args:
        model_name: HuggingFace model name or path
        device: Device to load model on
        use_greedy: Whether to use greedy decoding (deterministic)
        **kwargs: Additional arguments for BaseProofModel
    """
    
    def __init__(
        self,
        model_name: str = "deepseek-ai/DeepSeek-V3.2-Exp-SFT",
        device: str = "auto",
        use_greedy: bool = True,
        **kwargs
    ):
        super().__init__(model_name=model_name, device=device, **kwargs)
        self.use_greedy = use_greedy
        self.generation_config = get_greedy_config() if use_greedy else None
    
    def verify(
        self,
        problem: str,
        proof: str,
        max_new_tokens: int = 2048
    ) -> Dict[str, Union[str, Optional[float]]]:
        """
        Verify a single proof.
        
        Args:
            problem: Mathematical problem statement
            proof: Candidate proof to verify
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Dictionary with:
                - 'analysis': Detailed verification analysis
                - 'score': Extracted score (0, 0.5, 1) or None if invalid
                - 'raw_output': Raw model output
        """
        # Generate verification prompt
        prompt = get_proof_verification_prompt(problem, proof)
        
        # Mock mode (no actual model)
        if self.model is None or self.tokenizer is None:
            return self._mock_verify(problem, proof)
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = inputs.to(self.get_device())
        
        # Generate verification
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                **self.generation_config
            )
        
        # Decode output
        raw_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        analysis = clean_generated_text(raw_output, remove_prompt=True)
        
        # Extract score
        score = extract_score(analysis)
        
        return {
            'analysis': analysis,
            'score': score,
            'raw_output': raw_output
        }
    
    def verify_batch(
        self,
        problems: List[str],
        proofs: List[str],
        max_new_tokens: int = 2048
    ) -> List[Dict[str, Union[str, Optional[float]]]]:
        """
        Verify multiple proofs.
        
        Args:
            problems: List of problem statements
            proofs: List of candidate proofs
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            List of verification results
        """
        if len(problems) != len(proofs):
            raise ValueError("Number of problems and proofs must match")
        
        results = []
        for problem, proof in zip(problems, proofs):
            result = self.verify(problem, proof, max_new_tokens)
            results.append(result)
        
        return results
    
    def _mock_verify(
        self,
        problem: str,
        proof: str
    ) -> Dict[str, Union[str, Optional[float]]]:
        """
        Mock verification for testing without actual model.
        
        Args:
            problem: Problem statement
            proof: Candidate proof
            
        Returns:
            Mock verification result
        """
        # Simple heuristic: score based on proof length
        if len(proof) < 50:
            score = 0.0
            analysis = "The proof is too short and lacks detail. Score: \\boxed{0}"
        elif len(proof) < 200:
            score = 0.5
            analysis = "The proof has some merit but is incomplete. Score: \\boxed{0.5}"
        else:
            score = 1.0
            analysis = "The proof is comprehensive and correct. Score: \\boxed{1}"
        
        return {
            'analysis': analysis,
            'score': score,
            'raw_output': f"Problem: {problem}\n\nProof: {proof}\n\nAnalysis: {analysis}"
        }


# ============================================================================
# Meta-Verifier
# ============================================================================

class MetaVerifier(BaseProofModel):
    """
    Meta-verification model.
    
    Takes (problem, proof, verifier_analysis) as input and generates:
    - Assessment of the verifier's analysis quality
    - Meta-score (0, 0.5, or 1)
    
    Used to detect hallucinated issues in verifier output.
    
    Args:
        model_name: HuggingFace model name or path
        device: Device to load model on
        use_greedy: Whether to use greedy decoding
        **kwargs: Additional arguments for BaseProofModel
    """
    
    def __init__(
        self,
        model_name: str = "deepseek-ai/DeepSeek-V3.2-Exp-SFT",
        device: str = "auto",
        use_greedy: bool = True,
        **kwargs
    ):
        super().__init__(model_name=model_name, device=device, **kwargs)
        self.use_greedy = use_greedy
        self.generation_config = get_greedy_config() if use_greedy else None
    
    def meta_verify(
        self,
        problem: str,
        proof: str,
        verifier_analysis: str,
        max_new_tokens: int = 2048
    ) -> Dict[str, Union[str, Optional[float]]]:
        """
        Assess quality of verifier's analysis.
        
        Args:
            problem: Mathematical problem statement
            proof: Candidate proof
            verifier_analysis: Verifier's analysis of the proof
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Dictionary with:
                - 'meta_analysis': Assessment of verifier quality
                - 'meta_score': Quality score (0, 0.5, 1) or None
                - 'raw_output': Raw model output
        """
        # Generate meta-verification prompt
        prompt = get_meta_verification_prompt(problem, proof, verifier_analysis)
        
        # Mock mode
        if self.model is None or self.tokenizer is None:
            return self._mock_meta_verify(verifier_analysis)
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = inputs.to(self.get_device())
        
        # Generate meta-verification
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                **self.generation_config
            )
        
        # Decode output
        raw_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        meta_analysis = clean_generated_text(raw_output, remove_prompt=True)
        
        # Extract meta-score
        meta_score = extract_score(meta_analysis)
        
        return {
            'meta_analysis': meta_analysis,
            'meta_score': meta_score,
            'raw_output': raw_output
        }
    
    def meta_verify_batch(
        self,
        problems: List[str],
        proofs: List[str],
        verifier_analyses: List[str],
        max_new_tokens: int = 2048
    ) -> List[Dict[str, Union[str, Optional[float]]]]:
        """
        Meta-verify multiple verifier analyses.
        
        Args:
            problems: List of problem statements
            proofs: List of candidate proofs
            verifier_analyses: List of verifier analyses
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            List of meta-verification results
        """
        if not (len(problems) == len(proofs) == len(verifier_analyses)):
            raise ValueError("All input lists must have same length")
        
        results = []
        for problem, proof, analysis in zip(problems, proofs, verifier_analyses):
            result = self.meta_verify(problem, proof, analysis, max_new_tokens)
            results.append(result)
        
        return results
    
    def _mock_meta_verify(
        self,
        verifier_analysis: str
    ) -> Dict[str, Union[str, Optional[float]]]:
        """
        Mock meta-verification for testing.
        
        Args:
            verifier_analysis: Verifier's analysis to assess
            
        Returns:
            Mock meta-verification result
        """
        # Simple heuristic: good if analysis is detailed
        if len(verifier_analysis) < 100:
            meta_score = 0.5
            meta_analysis = "The analysis lacks detail. Score: \\boxed{0.5}"
        else:
            meta_score = 1.0
            meta_analysis = "The analysis is thorough and accurate. Score: \\boxed{1}"
        
        return {
            'meta_analysis': meta_analysis,
            'meta_score': meta_score,
            'raw_output': f"Analysis: {verifier_analysis}\n\nMeta-analysis: {meta_analysis}"
        }


# ============================================================================
# Utility Functions
# ============================================================================

def load_verifier(checkpoint_path: Union[str, Path]) -> ProofVerifier:
    """
    Load verifier from checkpoint.
    
    Args:
        checkpoint_path: Path to saved checkpoint
        
    Returns:
        Loaded ProofVerifier instance
    """
    verifier = ProofVerifier()
    verifier.load_checkpoint(checkpoint_path)
    return verifier


def load_meta_verifier(checkpoint_path: Union[str, Path]) -> MetaVerifier:
    """
    Load meta-verifier from checkpoint.
    
    Args:
        checkpoint_path: Path to saved checkpoint
        
    Returns:
        Loaded MetaVerifier instance
    """
    meta_verifier = MetaVerifier()
    meta_verifier.load_checkpoint(checkpoint_path)
    return meta_verifier


# ============================================================================
# Export
# ============================================================================

__all__ = [
    'ProofVerifier',
    'MetaVerifier',
    'load_verifier',
    'load_meta_verifier',
]