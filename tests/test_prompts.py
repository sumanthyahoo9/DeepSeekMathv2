"""
tests/test_prompts.py

Unit tests for prompt templates and utility functions.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.prompts import (
    get_proof_generation_prompt,
    get_proof_verification_prompt,
    get_meta_verification_prompt,
    get_proof_refinement_prompt,
    extract_score_from_response,
    extract_sections_from_response,
    check_format_compliance,
    VERIFICATION_RUBRICS
)


# ============================================================================
# Test Prompt Generation
# ============================================================================

def test_proof_generation_prompt():
    """Test proof generation prompt creation"""
    problem = "Prove that the sum of angles in a triangle is 180 degrees."
    prompt = get_proof_generation_prompt(problem)
    
    # Check key components are present
    assert problem in prompt
    assert "## Problem" in prompt
    assert "## Solution" in prompt
    assert "## Self Evaluation" in prompt
    assert VERIFICATION_RUBRICS in prompt
    assert "\\boxed{" in prompt


def test_proof_verification_prompt():
    """Test proof verification prompt creation"""
    problem = "Prove that 2+2=4"
    proof = "By definition of addition, 2+2=4. QED."
    prompt = get_proof_verification_prompt(problem, proof)
    
    assert problem in prompt
    assert proof in prompt
    assert "## Problem" in prompt
    assert "## Solution" in prompt
    assert VERIFICATION_RUBRICS in prompt
    assert "Here is my evaluation of the solution:" in prompt


def test_meta_verification_prompt():
    """Test meta-verification prompt creation"""
    problem = "Test problem"
    proof = "Test proof"
    analysis = "The proof is correct. Score: 1"
    prompt = get_meta_verification_prompt(problem, proof, analysis)
    
    assert problem in prompt
    assert proof in prompt
    assert analysis in prompt
    assert "solution evaluation" in prompt
    assert "defect analysis" in prompt.lower()


def test_proof_refinement_prompt():
    """Test proof refinement prompt creation"""
    problem = "Test problem"
    prev_proof = "## Solution\nIncorrect attempt"
    analyses = "Issues: Missing steps"
    prompt = get_proof_refinement_prompt(problem, prev_proof, analyses)
    
    assert problem in prompt
    assert prev_proof in prompt
    assert analyses in prompt
    assert "Candidate Solution" in prompt


# ============================================================================
# Test Score Extraction
# ============================================================================

def test_extract_score_valid():
    """Test extracting valid scores"""
    response1 = "Analysis here... \\boxed{1}"
    response2 = "More analysis... \\boxed{0.5}"
    response3 = "Failed proof... \\boxed{0}"
    
    assert extract_score_from_response(response1) == 1
    assert extract_score_from_response(response2) == 0.5
    assert extract_score_from_response(response3) == 0


def test_extract_score_invalid():
    """Test score extraction with invalid inputs"""
    response1 = "No score here"
    response2 = "\\boxed{2.5}"  # Invalid score
    response3 = "\\boxed{abc}"  # Non-numeric
    
    assert extract_score_from_response(response1) is None
    assert extract_score_from_response(response2) is None
    assert extract_score_from_response(response3) is None


def test_extract_score_multiple():
    """Test extraction when multiple boxed values exist"""
    response = "First: \\boxed{0.5} ... Final: \\boxed{1}"
    # Should return the last score
    assert extract_score_from_response(response) == 1


# ============================================================================
# Test Section Extraction
# ============================================================================

def test_extract_sections():
    """Test extracting solution and evaluation sections"""
    response = """
## Solution
This is the proof.
Step 1: ...
Step 2: ...

## Self Evaluation
Here is my evaluation of the solution:
The proof is correct.
Based on my evaluation, the final overall score should be: \\boxed{1}
"""
    
    sections = extract_sections_from_response(response)
    
    assert 'solution' in sections
    assert 'evaluation' in sections
    assert 'This is the proof' in sections['solution']
    assert 'correct' in sections['evaluation']


def test_extract_sections_partial():
    """Test extraction with only some sections"""
    response = "## Solution\nJust a solution, no evaluation"
    sections = extract_sections_from_response(response)
    
    assert 'solution' in sections
    assert 'evaluation' not in sections


# ============================================================================
# Test Format Compliance
# ============================================================================

def test_format_compliance_generation_valid():
    """Test format checking for valid generation response"""
    response = """
## Solution
Proof goes here...

## Self Evaluation
Here is my evaluation of the solution:
Analysis here...
Based on my evaluation, the final overall score should be: \\boxed{1}
"""
    
    assert check_format_compliance(response, "generation") is True


def test_format_compliance_generation_invalid():
    """Test format checking for invalid generation response"""
    # Missing self evaluation
    response1 = "## Solution\nJust a proof"
    assert check_format_compliance(response1, "generation") is False
    
    # Missing score
    response2 = """
## Solution
Proof here

## Self Evaluation
Here is my evaluation of the solution:
No score provided
"""
    assert check_format_compliance(response2, "generation") is False


def test_format_compliance_verification_valid():
    """Test format checking for valid verification response"""
    response = """
Here is my evaluation of the solution:
The proof has some issues...
Based on my evaluation, the final overall score should be: \\boxed{0.5}
"""
    
    assert check_format_compliance(response, "verification") is True


def test_format_compliance_verification_invalid():
    """Test format checking for invalid verification response"""
    response = "Just some text without proper format"
    assert check_format_compliance(response, "verification") is False


def test_format_compliance_meta_verification():
    """Test format checking for meta-verification"""
    valid_response = """
Here is my analysis of the "solution evaluation":
The evaluation correctly identifies the issues...
Based on my analysis, I will rate the "solution evaluation" as: \\boxed{1}
"""
    
    assert check_format_compliance(valid_response, "meta_verification") is True
    
    invalid_response = "No proper format"
    assert check_format_compliance(invalid_response, "meta_verification") is False


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])