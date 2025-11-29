"""
src/data/proof_dataset.py

PyTorch Dataset classes for DeepSeekMath-V2 training.
Handles problem-proof-score tuples for verifier and generator training.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

try:
    from torch.utils.data import Dataset
except ImportError:
    # Fallback for testing without torch
    class Dataset:
        """Minimal Dataset class for testing without torch"""
        pass


# ============================================================================
# Base Proof Dataset
# ============================================================================

class ProofDataset(Dataset):
    """
    Base dataset for proof verification and generation.
    
    Stores (problem, proof, score) tuples.
    
    Args:
        data_path: Path to JSON or JSONL file
        max_samples: Optional limit on number of samples
    """
    
    def __init__(
        self,
        data_path: Union[str, Path],
        max_samples: Optional[int] = None
    ):
        self.data_path = Path(data_path)
        self.samples = self._load_data(max_samples)
    
    def _load_data(self, max_samples: Optional[int]) -> List[Dict]:
        """Load data from JSON or JSONL file"""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        samples = []
        
        # Handle JSONL (one JSON object per line)
        if self.data_path.suffix == '.jsonl':
            with open(self.data_path, 'r') as f:
                for i, line in enumerate(f):
                    if max_samples and i >= max_samples:
                        break
                    samples.append(json.loads(line))
        
        # Handle JSON (single array)
        elif self.data_path.suffix == '.json':
            with open(self.data_path, 'r') as f:
                all_data = json.load(f)
                samples = all_data[:max_samples] if max_samples else all_data
        
        else:
            raise ValueError(f"Unsupported file format: {self.data_path.suffix}")
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single sample.
        
        Returns:
            Dictionary with keys: 'problem', 'proof', 'score'
        """
        return self.samples[idx]


# ============================================================================
# Verification Dataset
# ============================================================================

class VerificationDataset(ProofDataset):
    """
    Dataset for training proof verifier.
    
    Each sample contains:
    - problem: The mathematical problem
    - proof: A candidate solution
    - score: Expert annotation (0, 0.5, or 1)
    
    Expected JSON format:
    {
        "problem": "Prove that...",
        "proof": "Solution: ...",
        "score": 1.0
    }
    """
    
    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        
        # Validate required fields
        required = ['problem', 'proof', 'score']
        for field in required:
            if field not in sample:
                raise KeyError(f"Sample {idx} missing required field: {field}")
        
        # Validate score
        if sample['score'] not in [0, 0.5, 1, 0.0, 1.0]:
            raise ValueError(f"Invalid score: {sample['score']}")
        
        return {
            'problem': sample['problem'],
            'proof': sample['proof'],
            'score': float(sample['score']),
            'sample_id': sample.get('id', f'sample_{idx}')
        }


# ============================================================================
# Meta-Verification Dataset
# ============================================================================

class MetaVerificationDataset(ProofDataset):
    """
    Dataset for training meta-verifier.
    
    Each sample contains:
    - problem: The mathematical problem
    - proof: A candidate solution
    - analysis: Verifier's analysis of the proof
    - meta_score: Quality of the analysis (0, 0.5, or 1)
    
    Expected JSON format:
    {
        "problem": "Prove that...",
        "proof": "Solution: ...",
        "analysis": "Here is my evaluation...",
        "meta_score": 1.0
    }
    """
    
    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        
        # Validate required fields
        required = ['problem', 'proof', 'analysis', 'meta_score']
        for field in required:
            if field not in sample:
                raise KeyError(f"Sample {idx} missing required field: {field}")
        
        # Validate meta_score
        if sample['meta_score'] not in [0, 0.5, 1, 0.0, 1.0]:
            raise ValueError(f"Invalid meta_score: {sample['meta_score']}")
        
        return {
            'problem': sample['problem'],
            'proof': sample['proof'],
            'analysis': sample['analysis'],
            'meta_score': float(sample['meta_score']),
            'sample_id': sample.get('id', f'sample_{idx}')
        }


# ============================================================================
# Generation Dataset
# ============================================================================

class GenerationDataset(ProofDataset):
    """
    Dataset for training proof generator.
    
    Each sample contains only:
    - problem: The mathematical problem to solve
    
    Expected JSON format:
    {
        "problem": "Prove that..."
    }
    
    Optional fields:
    - category: Problem category (algebra, geometry, etc.)
    - difficulty: Problem difficulty level
    """
    
    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        
        # Validate required fields
        if 'problem' not in sample:
            raise KeyError(f"Sample {idx} missing required field: problem")
        
        result = {
            'problem': sample['problem'],
            'sample_id': sample.get('id', f'sample_{idx}')
        }
        
        # Optional metadata
        if 'category' in sample:
            result['category'] = sample['category']
        if 'difficulty' in sample:
            result['difficulty'] = sample['difficulty']
        
        return result


# ============================================================================
# Utility Functions
# ============================================================================

def create_sample_data(
    output_path: Union[str, Path],
    dataset_type: str = "verification",
    num_samples: int = 10
) -> None:
    """
    Create sample data for testing.
    
    Args:
        output_path: Where to save the sample data
        dataset_type: Type of dataset ("verification", "meta_verification", "generation")
        num_samples: Number of samples to generate
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    samples = []
    
    for i in range(num_samples):
        if dataset_type == "verification":
            sample = {
                "id": f"sample_{i}",
                "problem": f"Test problem {i}",
                "proof": f"Test proof {i}",
                "score": [0, 0.5, 1][i % 3]
            }
        
        elif dataset_type == "meta_verification":
            sample = {
                "id": f"sample_{i}",
                "problem": f"Test problem {i}",
                "proof": f"Test proof {i}",
                "analysis": f"Test analysis {i}",
                "meta_score": [0, 0.5, 1][i % 3]
            }
        
        elif dataset_type == "generation":
            sample = {
                "id": f"sample_{i}",
                "problem": f"Test problem {i}",
                "category": ["algebra", "geometry", "number_theory"][i % 3],
                "difficulty": ["easy", "medium", "hard"][i % 3]
            }
        
        else:
            raise ValueError(f"Unknown dataset_type: {dataset_type}")
        
        samples.append(sample)
    
    # Save as JSONL
    with open(output_path, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')


def get_dataset_statistics(dataset: Dataset) -> Dict:
    """
    Compute statistics on a dataset.
    
    Args:
        dataset: PyTorch Dataset
        
    Returns:
        Dictionary with dataset statistics
    """
    stats = {
        'num_samples': len(dataset),
        'dataset_type': type(dataset).__name__
    }
    
    # Compute score distribution for verification datasets
    if isinstance(dataset, (VerificationDataset, MetaVerificationDataset)):
        score_key = 'score' if isinstance(dataset, VerificationDataset) else 'meta_score'
        scores = [dataset[i][score_key] for i in range(len(dataset))]
        
        stats['score_distribution'] = {
            '0.0': scores.count(0.0),
            '0.5': scores.count(0.5),
            '1.0': scores.count(1.0)
        }
        stats['mean_score'] = sum(scores) / len(scores)
    
    # Compute category distribution for generation datasets
    if isinstance(dataset, GenerationDataset):
        categories = {}
        for i in range(len(dataset)):
            sample = dataset[i]
            if 'category' in sample:
                cat = sample['category']
                categories[cat] = categories.get(cat, 0) + 1
        
        if categories:
            stats['category_distribution'] = categories
    
    return stats


# ============================================================================
# Export
# ============================================================================

__all__ = [
    'ProofDataset',
    'VerificationDataset',
    'MetaVerificationDataset',
    'GenerationDataset',
    'create_sample_data',
    'get_dataset_statistics',
]