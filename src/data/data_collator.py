"""
src/data/data_collator.py

Data collators for batching and preparing inputs for DeepSeekMath-V2.
Handles tokenization, padding, and truncation for different training phases.
"""

from typing import Any, Dict, List, Optional

try:
    from transformers import PreTrainedTokenizer
except ImportError:
    # Fallback for testing without transformers
    class PreTrainedTokenizer:
        """Minimal tokenizer class for testing"""
        pass


# ============================================================================
# Base Data Collator
# ============================================================================

class ProofCollator:
    """
    Base collator for proof-related data.
    
    Handles:
    - Text concatenation (problem + proof)
    - Tokenization
    - Padding to max length in batch
    - Truncation if too long
    
    Args:
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
        padding: Padding strategy ('max_length' or 'longest')
        return_tensors: Return type ('pt' for PyTorch tensors)
    """
    
    def __init__(
        self,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        max_length: int = 128000,
        padding: str = "longest",
        return_tensors: str = "pt"
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding
        self.return_tensors = return_tensors
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate a batch of samples.
        
        Args:
            batch: List of samples from dataset
            
        Returns:
            Dictionary with batched tensors
        """
        raise NotImplementedError("Subclasses must implement __call__")


# ============================================================================
# Verification Collator
# ============================================================================

class VerificationCollator(ProofCollator):
    """
    Collator for verification training.
    
    Combines problem + proof into single text, tokenizes, and adds labels.
    
    Input sample format:
    {
        'problem': str,
        'proof': str,
        'score': float
    }
    
    Output batch format:
    {
        'input_ids': Tensor,
        'attention_mask': Tensor,
        'labels': Tensor (scores)
    }
    """
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Extract fields
        problems = [sample['problem'] for sample in batch]
        proofs = [sample['proof'] for sample in batch]
        scores = [sample['score'] for sample in batch]
        
        # Combine problem and proof
        texts = [
            f"Problem: {problem}\n\nProof: {proof}"
            for problem, proof in zip(problems, proofs)
        ]
        
        # Tokenize if tokenizer available
        if self.tokenizer:
            encoded = self.tokenizer(
                texts,
                max_length=self.max_length,
                padding=self.padding,
                truncation=True,
                return_tensors=self.return_tensors
            )
            
            # Add scores as labels
            import torch
            encoded['labels'] = torch.tensor(scores, dtype=torch.float32)
            
            return encoded
        else:
            # Return raw text for testing
            return {
                'texts': texts,
                'scores': scores,
                'batch_size': len(batch)
            }


# ============================================================================
# Meta-Verification Collator
# ============================================================================

class MetaVerificationCollator(ProofCollator):
    """
    Collator for meta-verification training.
    
    Combines problem + proof + verifier analysis into single text.
    
    Input sample format:
    {
        'problem': str,
        'proof': str,
        'analysis': str,
        'meta_score': float
    }
    
    Output batch format:
    {
        'input_ids': Tensor,
        'attention_mask': Tensor,
        'labels': Tensor (meta_scores)
    }
    """
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Extract fields
        problems = [sample['problem'] for sample in batch]
        proofs = [sample['proof'] for sample in batch]
        analyses = [sample['analysis'] for sample in batch]
        meta_scores = [sample['meta_score'] for sample in batch]
        
        # Combine problem, proof, and analysis
        texts = [
            f"Problem: {problem}\n\nProof: {proof}\n\nAnalysis: {analysis}"
            for problem, proof, analysis in zip(problems, proofs, analyses)
        ]
        
        # Tokenize if tokenizer available
        if self.tokenizer:
            encoded = self.tokenizer(
                texts,
                max_length=self.max_length,
                padding=self.padding,
                truncation=True,
                return_tensors=self.return_tensors
            )
            
            # Add meta_scores as labels
            import torch
            encoded['labels'] = torch.tensor(meta_scores, dtype=torch.float32)
            
            return encoded
        else:
            # Return raw text for testing
            return {
                'texts': texts,
                'meta_scores': meta_scores,
                'batch_size': len(batch)
            }


# ============================================================================
# Generation Collator
# ============================================================================

class GenerationCollator(ProofCollator):
    """
    Collator for generation training.
    
    Only needs to tokenize the problem (no proof provided).
    
    Input sample format:
    {
        'problem': str
    }
    
    Output batch format:
    {
        'input_ids': Tensor,
        'attention_mask': Tensor
    }
    """
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Extract problems
        problems = [sample['problem'] for sample in batch]
        
        # Format as prompts
        texts = [f"Problem: {problem}\n\nSolution:" for problem in problems]
        
        # Tokenize if tokenizer available
        if self.tokenizer:
            encoded = self.tokenizer(
                texts,
                max_length=self.max_length,
                padding=self.padding,
                truncation=True,
                return_tensors=self.return_tensors
            )
            
            return encoded
        else:
            # Return raw text for testing
            return {
                'texts': texts,
                'batch_size': len(batch)
            }


# ============================================================================
# Inference Collator
# ============================================================================

class InferenceCollator(ProofCollator):
    """
    Collator for inference/evaluation.
    
    More flexible - can handle problems only or problem+proof pairs.
    Preserves original sample metadata.
    
    Input sample format:
    {
        'problem': str,
        'proof': str (optional),
        ... other metadata ...
    }
    
    Output batch format:
    {
        'input_ids': Tensor,
        'attention_mask': Tensor,
        'metadata': List[Dict] (original samples)
    }
    """
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Determine if proofs are present
        has_proofs = 'proof' in batch[0]
        
        if has_proofs:
            # Verification mode: problem + proof
            texts = [
                f"Problem: {sample['problem']}\n\nProof: {sample['proof']}"
                for sample in batch
            ]
        else:
            # Generation mode: problem only
            texts = [
                f"Problem: {sample['problem']}\n\nSolution:"
                for sample in batch
            ]
        
        # Tokenize if tokenizer available
        if self.tokenizer:
            encoded = self.tokenizer(
                texts,
                max_length=self.max_length,
                padding=self.padding,
                truncation=True,
                return_tensors=self.return_tensors
            )
            
            # Preserve metadata
            encoded['metadata'] = batch
            
            return encoded
        else:
            # Return raw text for testing
            return {
                'texts': texts,
                'metadata': batch,
                'batch_size': len(batch)
            }


# ============================================================================
# Utility Functions
# ============================================================================

def get_collator(
    collator_type: str,
    tokenizer: Optional[PreTrainedTokenizer] = None,
    **kwargs
) -> ProofCollator:
    """
    Factory function to get appropriate collator.
    
    Args:
        collator_type: Type of collator ('verification', 'meta_verification', 'generation', 'inference')
        tokenizer: HuggingFace tokenizer
        **kwargs: Additional arguments for collator
        
    Returns:
        Appropriate collator instance
    """
    collators = {
        'verification': VerificationCollator,
        'meta_verification': MetaVerificationCollator,
        'generation': GenerationCollator,
        'inference': InferenceCollator
    }
    
    if collator_type not in collators:
        raise ValueError(f"Unknown collator_type: {collator_type}. Choose from {list(collators.keys())}")
    
    return collators[collator_type](tokenizer=tokenizer, **kwargs)


def estimate_batch_memory(
    batch_size: int,
    max_length: int,
    model_size_gb: float = 50.0
) -> float:
    """
    Estimate GPU memory required for a batch.
    
    Args:
        batch_size: Number of samples in batch
        max_length: Maximum sequence length
        model_size_gb: Model size in GB
        
    Returns:
        Estimated memory in GB
    """
    # Rough estimate:
    # - Activations: batch_size * max_length * 2 bytes (fp16) * layers
    # - Assume 100 layers, 4096 hidden size
    activation_gb = (batch_size * max_length * 2 * 100 * 4096) / (1024**3)
    
    # Add model weights
    total_gb = model_size_gb + activation_gb
    
    # Add 20% overhead for gradients, optimizer states
    total_gb *= 1.2
    
    return total_gb


# ============================================================================
# Export
# ============================================================================

__all__ = [
    'ProofCollator',
    'VerificationCollator',
    'MetaVerificationCollator',
    'GenerationCollator',
    'InferenceCollator',
    'get_collator',
    'estimate_batch_memory',
]