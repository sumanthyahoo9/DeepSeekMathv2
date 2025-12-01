"""
src/data/dataset_loaders.py

Dataset loaders for reading proof data from JSONL files.

Supports loading:
- Verification datasets: (problem, proof, score) triples
- Generation datasets: problems only
- Meta-verification datasets: (problem, proof, analysis, meta_score) tuples

JSONL format examples:

Verification:
{"problem": "Prove √2 is irrational", "proof": "Assume √2 = p/q...", "score": 1.0}

Generation:
{"problem": "Prove √2 is irrational"}

Meta-verification:
{"problem": "...", "proof": "...", "analysis": "...", "meta_score": 1.0}
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Union
from dataclasses import dataclass

try:
    from torch.utils.data import Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    Dataset = object

logger = logging.getLogger(__name__)


@dataclass
class VerificationExample:
    """Single verification example."""
    problem: str
    proof: str
    score: float
    metadata: Optional[Dict] = None


@dataclass
class GenerationExample:
    """Single generation example."""
    problem: str
    metadata: Optional[Dict] = None


@dataclass
class MetaVerificationExample:
    """Single meta-verification example."""
    problem: str
    proof: str
    analysis: str
    meta_score: float
    metadata: Optional[Dict] = None


def load_jsonl(file_path: Union[str, Path]) -> List[Dict]:
    """
    Load data from JSONL file.
    
    Args:
        file_path: Path to JSONL file
        
    Returns:
        List of dictionaries (one per line)
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                entry = json.loads(line)
                data.append(entry)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse line {i} in {file_path}: {e}")
                continue
    
    logger.info(f"Loaded {len(data)} entries from {file_path}")
    return data


def save_jsonl(data: List[Dict], file_path: Union[str, Path]):
    """
    Save data to JSONL file.
    
    Args:
        data: List of dictionaries to save
        file_path: Path to output JSONL file
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    logger.info(f"Saved {len(data)} entries to {file_path}")


class VerificationDataset(Dataset):
    """
    Dataset for proof verification training.
    
    Expected JSONL format:
    {"problem": "...", "proof": "...", "score": 0/0.5/1}
    
    Optional fields:
    - problem_id: Unique identifier
    - source: Data source (e.g., "AIME 2023")
    - category: Problem category (e.g., "algebra", "geometry")
    """
    
    def __init__(
        self,
        file_path: Union[str, Path],
        max_examples: Optional[int] = None,
        filter_scores: Optional[List[float]] = None,
        shuffle: bool = False,
        seed: int = 42
    ):
        """
        Initialize verification dataset.
        
        Args:
            file_path: Path to JSONL file
            max_examples: Maximum examples to load (None = all)
            filter_scores: Only include these scores (None = all)
            shuffle: Whether to shuffle data
            seed: Random seed for shuffling
        """
        self.file_path = Path(file_path)
        self.max_examples = max_examples
        self.filter_scores = set(filter_scores) if filter_scores else None
        
        # Load data
        raw_data = load_jsonl(file_path)
        
        # Filter by score if specified
        if self.filter_scores:
            raw_data = [
                d for d in raw_data 
                if d.get('score') in self.filter_scores
            ]
            logger.info(f"Filtered to {len(raw_data)} examples with scores {self.filter_scores}")
        
        # Shuffle if requested
        if shuffle:
            import random
            random.seed(seed)
            random.shuffle(raw_data)
        
        # Limit to max_examples
        if max_examples:
            raw_data = raw_data[:max_examples]
        
        # Parse into examples
        self.examples = []
        for i, entry in enumerate(raw_data):
            try:
                example = self._parse_entry(entry, i)
                self.examples.append(example)
            except Exception as e:
                logger.warning(f"Failed to parse entry {i}: {e}")
                continue
        
        logger.info(f"Loaded {len(self.examples)} verification examples")
        
        # Statistics
        self._compute_statistics()
    
    def _parse_entry(self, entry: Dict, index: int) -> VerificationExample:
        """Parse JSONL entry into VerificationExample."""
        # Required fields
        if 'problem' not in entry:
            raise ValueError(f"Entry {index} missing 'problem' field")
        if 'proof' not in entry:
            raise ValueError(f"Entry {index} missing 'proof' field")
        if 'score' not in entry:
            raise ValueError(f"Entry {index} missing 'score' field")
        
        problem = entry['problem']
        proof = entry['proof']
        score = float(entry['score'])
        
        # Validate score
        if score not in [0.0, 0.5, 1.0]:
            logger.warning(f"Entry {index} has invalid score {score}, clamping to {0, 0.5, 1}")
            score = min([0.0, 0.5, 1.0], key=lambda x: abs(x - score))
        
        # Optional metadata
        metadata = {
            k: v for k, v in entry.items() 
            if k not in ['problem', 'proof', 'score']
        }
        
        return VerificationExample(
            problem=problem,
            proof=proof,
            score=score,
            metadata=metadata if metadata else None
        )
    
    def _compute_statistics(self):
        """Compute dataset statistics."""
        scores = [ex.score for ex in self.examples]
        
        self.stats = {
            'total': len(self.examples),
            'score_distribution': {
                0.0: scores.count(0.0),
                0.5: scores.count(0.5),
                1.0: scores.count(1.0)
            },
            'avg_problem_length': sum(len(ex.problem) for ex in self.examples) / len(self.examples),
            'avg_proof_length': sum(len(ex.proof) for ex in self.examples) / len(self.examples)
        }
        
        logger.info(f"Dataset statistics: {self.stats}")
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get example by index."""
        example = self.examples[idx]
        return {
            'problem': example.problem,
            'proof': example.proof,
            'score': example.score,
            'metadata': example.metadata
        }
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        return self.stats


class GenerationDataset(Dataset):
    """
    Dataset for proof generation training.
    
    Expected JSONL format:
    {"problem": "..."}
    
    Optional fields:
    - problem_id: Unique identifier
    - source: Data source
    - category: Problem category
    - difficulty: Problem difficulty level
    """
    
    def __init__(
        self,
        file_path: Union[str, Path],
        max_examples: Optional[int] = None,
        filter_categories: Optional[List[str]] = None,
        shuffle: bool = False,
        seed: int = 42
    ):
        """
        Initialize generation dataset.
        
        Args:
            file_path: Path to JSONL file
            max_examples: Maximum examples to load (None = all)
            filter_categories: Only include these categories (None = all)
            shuffle: Whether to shuffle data
            seed: Random seed for shuffling
        """
        self.file_path = Path(file_path)
        self.max_examples = max_examples
        self.filter_categories = set(filter_categories) if filter_categories else None
        
        # Load data
        raw_data = load_jsonl(file_path)
        
        # Filter by category if specified
        if self.filter_categories:
            raw_data = [
                d for d in raw_data 
                if d.get('category') in self.filter_categories
            ]
            logger.info(f"Filtered to {len(raw_data)} examples in categories {self.filter_categories}")
        
        # Shuffle if requested
        if shuffle:
            import random
            random.seed(seed)
            random.shuffle(raw_data)
        
        # Limit to max_examples
        if max_examples:
            raw_data = raw_data[:max_examples]
        
        # Parse into examples
        self.examples = []
        for i, entry in enumerate(raw_data):
            try:
                example = self._parse_entry(entry, i)
                self.examples.append(example)
            except Exception as e:
                logger.warning(f"Failed to parse entry {i}: {e}")
                continue
        
        logger.info(f"Loaded {len(self.examples)} generation examples")
    
    def _parse_entry(self, entry: Dict, index: int) -> GenerationExample:
        """Parse JSONL entry into GenerationExample."""
        # Required field
        if 'problem' not in entry:
            raise ValueError(f"Entry {index} missing 'problem' field")
        
        problem = entry['problem']
        
        # Optional metadata
        metadata = {
            k: v for k, v in entry.items() 
            if k != 'problem'
        }
        
        return GenerationExample(
            problem=problem,
            metadata=metadata if metadata else None
        )
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get example by index."""
        example = self.examples[idx]
        return {
            'problem': example.problem,
            'metadata': example.metadata
        }


class MetaVerificationDataset(Dataset):
    """
    Dataset for meta-verification training.
    
    Expected JSONL format:
    {"problem": "...", "proof": "...", "analysis": "...", "meta_score": 0/0.5/1}
    
    Meta-score indicates quality of the analysis:
    - 1.0: Analysis is accurate and identifies real issues
    - 0.5: Analysis has minor issues
    - 0.0: Analysis has hallucinated or incorrect issues
    """
    
    def __init__(
        self,
        file_path: Union[str, Path],
        max_examples: Optional[int] = None,
        shuffle: bool = False,
        seed: int = 42
    ):
        """
        Initialize meta-verification dataset.
        
        Args:
            file_path: Path to JSONL file
            max_examples: Maximum examples to load (None = all)
            shuffle: Whether to shuffle data
            seed: Random seed for shuffling
        """
        self.file_path = Path(file_path)
        
        # Load data
        raw_data = load_jsonl(file_path)
        
        # Shuffle if requested
        if shuffle:
            import random
            random.seed(seed)
            random.shuffle(raw_data)
        
        # Limit to max_examples
        if max_examples:
            raw_data = raw_data[:max_examples]
        
        # Parse into examples
        self.examples = []
        for i, entry in enumerate(raw_data):
            try:
                example = self._parse_entry(entry, i)
                self.examples.append(example)
            except Exception as e:
                logger.warning(f"Failed to parse entry {i}: {e}")
                continue
        
        logger.info(f"Loaded {len(self.examples)} meta-verification examples")
    
    def _parse_entry(self, entry: Dict, index: int) -> MetaVerificationExample:
        """Parse JSONL entry into MetaVerificationExample."""
        required_fields = ['problem', 'proof', 'analysis', 'meta_score']
        for field in required_fields:
            if field not in entry:
                raise ValueError(f"Entry {index} missing '{field}' field")
        
        problem = entry['problem']
        proof = entry['proof']
        analysis = entry['analysis']
        meta_score = float(entry['meta_score'])
        
        # Validate meta_score
        if meta_score not in [0.0, 0.5, 1.0]:
            logger.warning(f"Entry {index} has invalid meta_score {meta_score}")
            meta_score = min([0.0, 0.5, 1.0], key=lambda x: abs(x - meta_score))
        
        # Optional metadata
        metadata = {
            k: v for k, v in entry.items() 
            if k not in required_fields
        }
        
        return MetaVerificationExample(
            problem=problem,
            proof=proof,
            analysis=analysis,
            meta_score=meta_score,
            metadata=metadata if metadata else None
        )
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get example by index."""
        example = self.examples[idx]
        return {
            'problem': example.problem,
            'proof': example.proof,
            'analysis': example.analysis,
            'meta_score': example.meta_score,
            'metadata': example.metadata
        }


def create_sample_datasets(output_dir: Union[str, Path]):
    """
    Create sample JSONL datasets for testing.
    
    Args:
        output_dir: Directory to save sample datasets
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample verification dataset
    verification_data = [
        {
            "problem": "Prove that √2 is irrational.",
            "proof": "Assume √2 = p/q where p, q are integers in lowest terms. Then 2 = p²/q², so p² = 2q². This means p² is even, so p is even. Write p = 2k. Then 4k² = 2q², so q² = 2k², meaning q is also even. This contradicts our assumption that p/q is in lowest terms.",
            "score": 1.0,
            "category": "number_theory",
            "difficulty": "medium"
        },
        {
            "problem": "Show that there are infinitely many prime numbers.",
            "proof": "Assume there are finitely many primes p₁, p₂, ..., pₙ. Consider N = p₁ × p₂ × ... × pₙ + 1.",
            "score": 0.5,
            "category": "number_theory",
            "difficulty": "hard"
        },
        {
            "problem": "Prove the Pythagorean theorem.",
            "proof": "Consider a right triangle with sides a, b, and hypotenuse c.",
            "score": 0.0,
            "category": "geometry",
            "difficulty": "easy"
        }
    ]
    
    save_jsonl(verification_data, output_dir / "sample_verification.jsonl")
    
    # Sample generation dataset
    generation_data = [
        {"problem": "Prove that √2 is irrational.", "category": "number_theory"},
        {"problem": "Show that there are infinitely many prime numbers.", "category": "number_theory"},
        {"problem": "Prove the Pythagorean theorem.", "category": "geometry"}
    ]
    
    save_jsonl(generation_data, output_dir / "sample_generation.jsonl")
    
    # Sample meta-verification dataset
    meta_verification_data = [
        {
            "problem": "Prove that √2 is irrational.",
            "proof": "Assume √2 = p/q in lowest terms...",
            "analysis": "The proof is correct and rigorous. Score: 1.0",
            "meta_score": 1.0
        }
    ]
    
    save_jsonl(meta_verification_data, output_dir / "sample_meta_verification.jsonl")
    
    logger.info(f"Created sample datasets in {output_dir}")


if __name__ == "__main__":
    # Create sample datasets for testing
    create_sample_datasets("data/samples")
    
    # Test loading
    print("\n=== Testing Verification Dataset ===")
    verif_ds = VerificationDataset("data/samples/sample_verification.jsonl")
    print(f"Loaded {len(verif_ds)} examples")
    print(f"First example: {verif_ds[0]}")
    print(f"Statistics: {verif_ds.get_statistics()}")
    
    print("\n=== Testing Generation Dataset ===")
    gen_ds = GenerationDataset("data/samples/sample_generation.jsonl")
    print(f"Loaded {len(gen_ds)} examples")
    print(f"First example: {gen_ds[0]}")