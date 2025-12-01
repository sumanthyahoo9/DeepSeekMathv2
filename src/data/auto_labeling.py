"""
src/data/auto_labeling.py

Automatic proof labeling using scaled verification and meta-verification.

Implements the auto-labeling pipeline from Section 2.3 of the paper:
1. Generate n verification analyses per proof
2. For analyses reporting issues, run m meta-verifications
3. Determine consensus: if >=k valid analyses agree, label with lowest score
4. Otherwise, label as correct (score=1.0)

This enables verifier improvement without human annotation by leveraging
scaled verification compute.
"""

import logging
import random
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

logger = logging.getLogger(__name__)


@dataclass
class VerificationAnalysis:
    """
    Result of a single verification attempt.
    
    Attributes:
        proof_id: Identifier for the proof
        analysis_text: Full verification analysis text
        score: Predicted score (0, 0.5, or 1.0)
        has_issues: Whether issues were identified
    """
    proof_id: str
    analysis_text: str
    score: float
    has_issues: bool
    
    def __post_init__(self):
        """Validate score and has_issues consistency."""
        if self.score < 1.0 and not self.has_issues:
            logger.warning(f"Proof {self.proof_id}: score < 1.0 but has_issues=False")
        if self.score == 1.0 and self.has_issues:
            logger.warning(f"Proof {self.proof_id}: score = 1.0 but has_issues=True")


@dataclass
class MetaVerificationResult:
    """
    Result of meta-verifying a verification analysis.
    
    Attributes:
        analysis_id: Identifier for the analysis being checked
        is_valid: Whether the analysis is faithful (issues are real)
        quality_score: Meta-verification quality score (0, 0.5, or 1.0)
        meta_analysis: Meta-verification analysis text
    """
    analysis_id: str
    is_valid: bool
    quality_score: float
    meta_analysis: str


@dataclass
class AutoLabelResult:
    """
    Result of auto-labeling a single proof.
    
    Attributes:
        proof_id: Identifier for the proof
        label: Assigned score (0, 0.5, or 1.0)
        confidence: Confidence in the label (0-1)
        num_analyses: Total verification analyses generated
        num_valid_analyses: Analyses with valid issues
        consensus_reached: Whether k-threshold was met
        reasoning: Explanation of how label was determined
    """
    proof_id: str
    label: float
    confidence: float
    num_analyses: int
    num_valid_analyses: int
    consensus_reached: bool
    reasoning: str


class ScaledVerification:
    """
    Scaled verification for automatic proof labeling.
    
    Uses multiple verification attempts and meta-verification to
    automatically assign labels to proofs without human annotation.
    """
    
    def __init__(
        self,
        verifier: Any,
        meta_verifier: Optional[Any] = None,
        n_verifications: int = 64,
        m_meta_checks: int = 32,
        k_threshold: int = 8,
        meta_consensus_threshold: float = 0.5,
        device: Optional[str] = None
    ):
        """
        Initialize scaled verification.
        
        Args:
            verifier: Verifier model for proof analysis
            meta_verifier: Meta-verifier model (uses verifier if None)
            n_verifications: Number of verification analyses per proof
            m_meta_checks: Number of meta-verifications per analysis
            k_threshold: Minimum valid analyses for consensus
            meta_consensus_threshold: Threshold for meta-verification majority
            device: Device to run on ('cuda' or 'cpu')
        """
        if not TORCH_AVAILABLE:
            self.mock_mode = True
            logger.warning("PyTorch not available, running in mock mode")
            return
        
        self.mock_mode = False
        self.verifier = verifier
        self.meta_verifier = meta_verifier if meta_verifier else verifier
        self.n_verifications = n_verifications
        self.m_meta_checks = m_meta_checks
        self.k_threshold = k_threshold
        self.meta_consensus_threshold = meta_consensus_threshold
        
        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Move models to device and set to eval mode
        if hasattr(self.verifier, 'to'):
            self.verifier = self.verifier.to(self.device)
        if hasattr(self.verifier, 'eval'):
            self.verifier.eval()
        
        if hasattr(self.meta_verifier, 'to'):
            self.meta_verifier = self.meta_verifier.to(self.device)
        if hasattr(self.meta_verifier, 'eval'):
            self.meta_verifier.eval()
        
        logger.info(
            f"Initialized ScaledVerification: n={n_verifications}, "
            f"m={m_meta_checks}, k={k_threshold}"
        )
    
    def generate_verification_analyses(
        self,
        problem: str,
        proof: str,
        proof_id: str = "proof_0"
    ) -> List[VerificationAnalysis]:
        """
        Generate multiple verification analyses for a proof.
        
        Args:
            problem: Problem statement
            proof: Proof to verify
            proof_id: Identifier for the proof
            
        Returns:
            List of verification analyses
        """
        if self.mock_mode:
            # Return mock analyses
            analyses = []
            for i in range(self.n_verifications):
                score = random.choice([0.0, 0.5, 1.0])
                analyses.append(VerificationAnalysis(
                    proof_id=f"{proof_id}_analysis_{i}",
                    analysis_text=f"Mock analysis {i} for {proof_id}",
                    score=score,
                    has_issues=(score < 1.0)
                ))
            return analyses
        
        analyses = []
        
        # Generate n verification analyses
        for i in range(self.n_verifications):
            # TODO: Implement actual verification generation
            # This would call verifier.generate() with proper prompting
            
            # Placeholder
            analysis_text = f"Analysis {i}"
            score = 0.5
            has_issues = (score < 1.0)
            
            analyses.append(VerificationAnalysis(
                proof_id=f"{proof_id}_analysis_{i}",
                analysis_text=analysis_text,
                score=score,
                has_issues=has_issues
            ))
        
        logger.debug(f"Generated {len(analyses)} verification analyses for {proof_id}")
        return analyses
    
    def meta_verify_analysis(
        self,
        problem: str,
        proof: str,
        analysis: VerificationAnalysis
    ) -> List[MetaVerificationResult]:
        """
        Run meta-verification on a verification analysis.
        
        Args:
            problem: Problem statement
            proof: Original proof
            analysis: Verification analysis to check
            
        Returns:
            List of meta-verification results
        """
        if self.mock_mode:
            # Return mock meta-verifications
            results = []
            for i in range(self.m_meta_checks):
                is_valid = random.choice([True, False])
                quality = 1.0 if is_valid else 0.0
                results.append(MetaVerificationResult(
                    analysis_id=f"{analysis.proof_id}_meta_{i}",
                    is_valid=is_valid,
                    quality_score=quality,
                    meta_analysis=f"Mock meta-analysis {i}"
                ))
            return results
        
        results = []
        
        # Run m meta-verifications
        for i in range(self.m_meta_checks):
            # TODO: Implement actual meta-verification
            # This would call meta_verifier.generate() with meta-verification prompt
            
            # Placeholder
            is_valid = True
            quality = 1.0
            meta_analysis = f"Meta-analysis {i}"
            
            results.append(MetaVerificationResult(
                analysis_id=f"{analysis.proof_id}_meta_{i}",
                is_valid=is_valid,
                quality_score=quality,
                meta_analysis=meta_analysis
            ))
        
        return results
    
    def compute_meta_consensus(
        self,
        meta_results: List[MetaVerificationResult]
    ) -> Tuple[bool, float]:
        """
        Compute consensus from meta-verification results.
        
        Args:
            meta_results: List of meta-verification results
            
        Returns:
            Tuple of (is_valid, confidence)
            - is_valid: True if majority says analysis is valid
            - confidence: Fraction of meta-checks agreeing with majority
        """
        if not meta_results:
            return False, 0.0
        
        # Count valid vs invalid
        num_valid = sum(1 for r in meta_results if r.is_valid)
        total = len(meta_results)
        
        # Majority vote
        is_valid = (num_valid / total) >= self.meta_consensus_threshold
        confidence = num_valid / total if is_valid else (total - num_valid) / total
        
        return is_valid, confidence
    
    def label_proof(
        self,
        problem: str,
        proof: str,
        proof_id: str = "proof_0"
    ) -> AutoLabelResult:
        """
        Automatically label a proof using scaled verification.
        
        Algorithm:
        1. Generate n verification analyses
        2. Filter analyses that report issues (score < 1.0)
        3. For each issue analysis, run m meta-verifications
        4. Keep analyses where majority of meta-checks say issues are valid
        5. If >= k valid analyses exist, label with lowest score
        6. Otherwise, label as correct (score = 1.0)
        
        Args:
            problem: Problem statement
            proof: Proof to label
            proof_id: Identifier for the proof
            
        Returns:
            AutoLabelResult with assigned label and metadata
        """
        logger.info(f"Auto-labeling proof {proof_id}")
        
        # Step 1: Generate verification analyses
        analyses = self.generate_verification_analyses(problem, proof, proof_id)
        
        # Step 2: Filter analyses that found issues
        issue_analyses = [a for a in analyses if a.has_issues]
        
        logger.debug(
            f"Proof {proof_id}: {len(issue_analyses)}/{len(analyses)} "
            f"analyses found issues"
        )
        
        # Step 3: Meta-verify each issue analysis
        valid_analyses = []
        analysis_scores = []
        
        for analysis in issue_analyses:
            # Run meta-verification
            meta_results = self.meta_verify_analysis(problem, proof, analysis)
            
            # Check consensus
            is_valid, confidence = self.compute_meta_consensus(meta_results)
            
            if is_valid:
                valid_analyses.append(analysis)
                analysis_scores.append(analysis.score)
                logger.debug(
                    f"Analysis {analysis.proof_id}: valid (confidence={confidence:.2f})"
                )
        
        # Step 4: Determine label based on consensus
        num_valid = len(valid_analyses)
        consensus_reached = (num_valid >= self.k_threshold)
        
        if consensus_reached:
            # At least k analyses found valid issues
            label = min(analysis_scores)
            reasoning = (
                f"{num_valid} valid analyses found issues "
                f"(>= threshold {self.k_threshold}). "
                f"Lowest score: {label}"
            )
            confidence = num_valid / self.n_verifications
        else:
            # No consensus on issues
            label = 1.0
            reasoning = (
                f"Only {num_valid} valid analyses found issues "
                f"(< threshold {self.k_threshold}). "
                f"Labeling as correct."
            )
            confidence = (self.n_verifications - len(issue_analyses)) / self.n_verifications
        
        logger.info(
            f"Proof {proof_id}: label={label}, confidence={confidence:.2f}, "
            f"valid_analyses={num_valid}/{len(issue_analyses)}"
        )
        
        return AutoLabelResult(
            proof_id=proof_id,
            label=label,
            confidence=confidence,
            num_analyses=len(analyses),
            num_valid_analyses=num_valid,
            consensus_reached=consensus_reached,
            reasoning=reasoning
        )
    
    def batch_label_proofs(
        self,
        problems: List[str],
        proofs: List[str],
        proof_ids: Optional[List[str]] = None
    ) -> List[AutoLabelResult]:
        """
        Auto-label multiple proofs.
        
        Args:
            problems: List of problem statements
            proofs: List of proofs
            proof_ids: Optional list of proof identifiers
            
        Returns:
            List of auto-label results
        """
        if proof_ids is None:
            proof_ids = [f"proof_{i}" for i in range(len(proofs))]
        
        if len(problems) != len(proofs) or len(proofs) != len(proof_ids):
            raise ValueError("problems, proofs, and proof_ids must have same length")
        
        logger.info(f"Auto-labeling {len(proofs)} proofs")
        
        results = []
        for i, (problem, proof, proof_id) in enumerate(zip(problems, proofs, proof_ids)):
            logger.info(f"Processing {i+1}/{len(proofs)}: {proof_id}")
            result = self.label_proof(problem, proof, proof_id)
            results.append(result)
        
        # Summary statistics
        labels = [r.label for r in results]
        avg_confidence = sum(r.confidence for r in results) / len(results)
        consensus_rate = sum(r.consensus_reached for r in results) / len(results)
        
        logger.info("\n" + "="*50)
        logger.info("AUTO-LABELING SUMMARY")
        logger.info("="*50)
        logger.info(f"Total proofs: {len(results)}")
        logger.info(f"Label distribution: {dict(zip(*zip(*[(l, labels.count(l)) for l in set(labels)])))}") 
        logger.info(f"Average confidence: {avg_confidence:.2f}")
        logger.info(f"Consensus rate: {consensus_rate:.2%}")
        logger.info("="*50 + "\n")
        
        return results
    
    def get_statistics(
        self,
        results: List[AutoLabelResult]
    ) -> Dict[str, Any]:
        """
        Compute statistics from auto-labeling results.
        
        Args:
            results: List of auto-label results
            
        Returns:
            Dictionary of statistics
        """
        if not results:
            return {}
        
        labels = [r.label for r in results]
        confidences = [r.confidence for r in results]
        
        # Label distribution
        label_counts = defaultdict(int)
        for label in labels:
            label_counts[label] += 1
        
        stats = {
            'total_proofs': len(results),
            'label_distribution': dict(label_counts),
            'avg_confidence': sum(confidences) / len(confidences),
            'min_confidence': min(confidences),
            'max_confidence': max(confidences),
            'consensus_rate': sum(r.consensus_reached for r in results) / len(results),
            'avg_analyses_per_proof': sum(r.num_analyses for r in results) / len(results),
            'avg_valid_analyses': sum(r.num_valid_analyses for r in results) / len(results)
        }
        
        return stats


def save_labeled_dataset(
    results: List[AutoLabelResult],
    problems: List[str],
    proofs: List[str],
    output_path: str
):
    """
    Save auto-labeled dataset to file.
    
    Args:
        results: Auto-labeling results
        problems: Problem statements
        proofs: Proofs
        output_path: Path to save dataset (JSONL format)
    """
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for result, problem, proof in zip(results, problems, proofs):
            entry = {
                'proof_id': result.proof_id,
                'problem': problem,
                'proof': proof,
                'score': result.label,
                'confidence': result.confidence,
                'consensus_reached': result.consensus_reached,
                'num_valid_analyses': result.num_valid_analyses,
                'reasoning': result.reasoning
            }
            f.write(json.dumps(entry) + '\n')
    
    logger.info(f"Saved {len(results)} labeled proofs to {output_path}")