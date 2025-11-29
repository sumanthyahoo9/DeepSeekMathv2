"""
tests/test_proof_dataset.py

Unit tests for proof dataset classes.
"""

import json
import pytest
import sys
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.proof_dataset import (
    ProofDataset,
    VerificationDataset,
    MetaVerificationDataset,
    GenerationDataset,
    create_sample_data,
    get_dataset_statistics
)


# ============================================================================
# Test Base ProofDataset
# ============================================================================

def test_proof_dataset_json():
    """Test loading from JSON file"""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_file = Path(tmpdir) / "test_data.json"
        
        # Create test data
        test_data = [
            {"problem": "P1", "proof": "Proof1", "score": 1.0},
            {"problem": "P2", "proof": "Proof2", "score": 0.5},
            {"problem": "P3", "proof": "Proof3", "score": 0.0}
        ]
        
        with open(data_file, 'w') as f:
            json.dump(test_data, f)
        
        # Load dataset
        dataset = ProofDataset(data_file)
        
        assert len(dataset) == 3
        assert dataset[0]['problem'] == "P1"
        assert dataset[1]['score'] == 0.5


def test_proof_dataset_jsonl():
    """Test loading from JSONL file"""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_file = Path(tmpdir) / "test_data.jsonl"
        
        # Create test data
        test_data = [
            {"problem": "P1", "proof": "Proof1", "score": 1.0},
            {"problem": "P2", "proof": "Proof2", "score": 0.5}
        ]
        
        with open(data_file, 'w') as f:
            for item in test_data:
                f.write(json.dumps(item) + '\n')
        
        # Load dataset
        dataset = ProofDataset(data_file)
        
        assert len(dataset) == 2
        assert dataset[0]['problem'] == "P1"


def test_proof_dataset_max_samples():
    """Test limiting number of samples"""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_file = Path(tmpdir) / "test_data.json"
        
        test_data = [{"problem": f"P{i}"} for i in range(100)]
        
        with open(data_file, 'w') as f:
            json.dump(test_data, f)
        
        # Load only 10 samples
        dataset = ProofDataset(data_file, max_samples=10)
        
        assert len(dataset) == 10


def test_proof_dataset_file_not_found():
    """Test error handling for missing file"""
    with pytest.raises(FileNotFoundError):
        ProofDataset("nonexistent_file.json")


def test_proof_dataset_invalid_format():
    """Test error handling for invalid file format"""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_file = Path(tmpdir) / "test_data.txt"
        data_file.write_text("some text")
        
        with pytest.raises(ValueError):
            ProofDataset(data_file)


# ============================================================================
# Test VerificationDataset
# ============================================================================

def test_verification_dataset_valid():
    """Test verification dataset with valid data"""
    with tempfile.TemporaryDirectory() as tmpdir:
        create_sample_data(
            Path(tmpdir) / "verification.jsonl",
            dataset_type="verification",
            num_samples=9
        )
        
        dataset = VerificationDataset(Path(tmpdir) / "verification.jsonl")
        
        assert len(dataset) == 9
        
        sample = dataset[0]
        assert 'problem' in sample
        assert 'proof' in sample
        assert 'score' in sample
        assert 'sample_id' in sample
        assert sample['score'] in [0.0, 0.5, 1.0]


def test_verification_dataset_missing_field():
    """Test error handling for missing required fields"""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_file = Path(tmpdir) / "test.jsonl"
        
        # Missing 'proof' field
        with open(data_file, 'w') as f:
            f.write(json.dumps({"problem": "P1", "score": 1.0}) + '\n')
        
        dataset = VerificationDataset(data_file)
        
        with pytest.raises(KeyError):
            _ = dataset[0]


def test_verification_dataset_invalid_score():
    """Test error handling for invalid scores"""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_file = Path(tmpdir) / "test.jsonl"
        
        # Invalid score (not 0, 0.5, or 1)
        with open(data_file, 'w') as f:
            f.write(json.dumps({
                "problem": "P1",
                "proof": "Proof1",
                "score": 2.0
            }) + '\n')
        
        dataset = VerificationDataset(data_file)
        
        with pytest.raises(ValueError):
            _ = dataset[0]


# ============================================================================
# Test MetaVerificationDataset
# ============================================================================

def test_meta_verification_dataset_valid():
    """Test meta-verification dataset with valid data"""
    with tempfile.TemporaryDirectory() as tmpdir:
        create_sample_data(
            Path(tmpdir) / "meta_verification.jsonl",
            dataset_type="meta_verification",
            num_samples=6
        )
        
        dataset = MetaVerificationDataset(Path(tmpdir) / "meta_verification.jsonl")
        
        assert len(dataset) == 6
        
        sample = dataset[0]
        assert 'problem' in sample
        assert 'proof' in sample
        assert 'analysis' in sample
        assert 'meta_score' in sample
        assert sample['meta_score'] in [0.0, 0.5, 1.0]


def test_meta_verification_dataset_missing_field():
    """Test error handling for missing required fields"""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_file = Path(tmpdir) / "test.jsonl"
        
        # Missing 'analysis' field
        with open(data_file, 'w') as f:
            f.write(json.dumps({
                "problem": "P1",
                "proof": "Proof1",
                "meta_score": 1.0
            }) + '\n')
        
        dataset = MetaVerificationDataset(data_file)
        
        with pytest.raises(KeyError):
            _ = dataset[0]


# ============================================================================
# Test GenerationDataset
# ============================================================================

def test_generation_dataset_valid():
    """Test generation dataset with valid data"""
    with tempfile.TemporaryDirectory() as tmpdir:
        create_sample_data(
            Path(tmpdir) / "generation.jsonl",
            dataset_type="generation",
            num_samples=12
        )
        
        dataset = GenerationDataset(Path(tmpdir) / "generation.jsonl")
        
        assert len(dataset) == 12
        
        sample = dataset[0]
        assert 'problem' in sample
        assert 'sample_id' in sample
        # Optional fields
        assert 'category' in sample
        assert 'difficulty' in sample


def test_generation_dataset_minimal():
    """Test generation dataset with only required fields"""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_file = Path(tmpdir) / "test.jsonl"
        
        # Only problem field
        with open(data_file, 'w') as f:
            f.write(json.dumps({"problem": "Test problem"}) + '\n')
        
        dataset = GenerationDataset(data_file)
        sample = dataset[0]
        
        assert sample['problem'] == "Test problem"
        assert 'sample_id' in sample
        # Optional fields should not be present
        assert 'category' not in sample
        assert 'difficulty' not in sample


def test_generation_dataset_missing_problem():
    """Test error handling when problem field is missing"""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_file = Path(tmpdir) / "test.jsonl"
        
        with open(data_file, 'w') as f:
            f.write(json.dumps({"category": "algebra"}) + '\n')
        
        dataset = GenerationDataset(data_file)
        
        with pytest.raises(KeyError):
            _ = dataset[0]


# ============================================================================
# Test Utility Functions
# ============================================================================

def test_create_sample_data_verification():
    """Test creating sample verification data"""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = Path(tmpdir) / "verification.jsonl"
        
        create_sample_data(output_file, "verification", num_samples=15)
        
        assert output_file.exists()
        
        # Load and verify
        dataset = VerificationDataset(output_file)
        assert len(dataset) == 15


def test_create_sample_data_meta_verification():
    """Test creating sample meta-verification data"""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = Path(tmpdir) / "meta_verification.jsonl"
        
        create_sample_data(output_file, "meta_verification", num_samples=12)
        
        dataset = MetaVerificationDataset(output_file)
        assert len(dataset) == 12


def test_create_sample_data_generation():
    """Test creating sample generation data"""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = Path(tmpdir) / "generation.jsonl"
        
        create_sample_data(output_file, "generation", num_samples=20)
        
        dataset = GenerationDataset(output_file)
        assert len(dataset) == 20


def test_get_dataset_statistics_verification():
    """Test statistics computation for verification dataset"""
    with tempfile.TemporaryDirectory() as tmpdir:
        create_sample_data(
            Path(tmpdir) / "verification.jsonl",
            dataset_type="verification",
            num_samples=9  # Will have 3 of each score: 0, 0.5, 1
        )
        
        dataset = VerificationDataset(Path(tmpdir) / "verification.jsonl")
        stats = get_dataset_statistics(dataset)
        
        assert stats['num_samples'] == 9
        assert stats['dataset_type'] == 'VerificationDataset'
        assert 'score_distribution' in stats
        assert stats['score_distribution']['0.0'] == 3
        assert stats['score_distribution']['0.5'] == 3
        assert stats['score_distribution']['1.0'] == 3
        assert stats['mean_score'] == 0.5


def test_get_dataset_statistics_generation():
    """Test statistics computation for generation dataset"""
    with tempfile.TemporaryDirectory() as tmpdir:
        create_sample_data(
            Path(tmpdir) / "generation.jsonl",
            dataset_type="generation",
            num_samples=12
        )
        
        dataset = GenerationDataset(Path(tmpdir) / "generation.jsonl")
        stats = get_dataset_statistics(dataset)
        
        assert stats['num_samples'] == 12
        assert stats['dataset_type'] == 'GenerationDataset'
        assert 'category_distribution' in stats
        # Each category should appear 4 times (12 samples, 3 categories)
        assert stats['category_distribution']['algebra'] == 4
        assert stats['category_distribution']['geometry'] == 4
        assert stats['category_distribution']['number_theory'] == 4


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])