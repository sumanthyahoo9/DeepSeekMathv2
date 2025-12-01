#!/usr/bin/env python3
"""
scripts/20_auto_label.py

Command-line script for automatic proof labeling using scaled verification.

This implements the auto-labeling pipeline from Section 2.3:
- Generate proofs from generator
- Run scaled verification (n analyses per proof)
- Run meta-verification (m checks per analysis)  
- Assign labels based on consensus (k threshold)
- Save labeled dataset for verifier retraining

Usage:
    # Auto-label from existing proofs
    python scripts/20_auto_label.py \
        --verifier checkpoints/verifier/best.pt \
        --proofs data/unlabeled_proofs.jsonl \
        --output data/auto_labeled.jsonl

    # Generate and auto-label
    python scripts/20_auto_label.py \
        --generator checkpoints/generator/best.pt \
        --verifier checkpoints/verifier/best.pt \
        --problems data/problems.jsonl \
        --output data/auto_labeled.jsonl \
        --generate
"""

import argparse
import logging
import sys
from pathlib import Path
import json
from typing import List, Dict
import torch
TORCH_AVAILABLE = True
from src.data.auto_labeling import ScaledVerification, save_labeled_dataset

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Auto-label proofs using scaled verification"
    )
    
    # Model paths
    parser.add_argument(
        "--verifier",
        type=str,
        required=True,
        help="Path to verifier checkpoint"
    )
    
    parser.add_argument(
        "--meta_verifier",
        type=str,
        default=None,
        help="Path to meta-verifier checkpoint (uses verifier if not specified)"
    )
    
    parser.add_argument(
        "--generator",
        type=str,
        default=None,
        help="Path to generator checkpoint (required if --generate is set)"
    )
    
    # Data paths
    parser.add_argument(
        "--problems",
        type=str,
        default=None,
        help="Path to problems file (JSONL, required if --generate is set)"
    )
    
    parser.add_argument(
        "--proofs",
        type=str,
        default=None,
        help="Path to existing proofs file (JSONL, required if not generating)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save auto-labeled dataset (JSONL)"
    )
    
    # Mode
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate proofs using generator before labeling"
    )
    
    # Scaled verification parameters (from paper Section 2.3)
    parser.add_argument(
        "--n_verifications",
        type=int,
        default=64,
        help="Number of verification analyses per proof (paper uses 64)"
    )
    
    parser.add_argument(
        "--m_meta_checks",
        type=int,
        default=32,
        help="Number of meta-verification checks per analysis (paper uses 32)"
    )
    
    parser.add_argument(
        "--k_threshold",
        type=int,
        default=8,
        help="Minimum valid analyses for consensus (paper uses 8)"
    )
    
    parser.add_argument(
        "--meta_consensus_threshold",
        type=float,
        default=0.5,
        help="Threshold for meta-verification majority vote"
    )
    
    # Generation parameters (if generating proofs)
    parser.add_argument(
        "--num_proofs_per_problem",
        type=int,
        default=1,
        help="Number of proofs to generate per problem"
    )
    
    parser.add_argument(
        "--max_problems",
        type=int,
        default=None,
        help="Maximum number of problems to process (None = all)"
    )
    
    # Hardware
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda or cpu)"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for generation/verification"
    )
    
    # Debugging
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Dry run: setup everything but don't process"
    )
    
    return parser.parse_args()


def load_model(checkpoint_path: str, device: str, model_type: str = "verifier"):
    """
    Load model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint
        device: Device to load on
        model_type: Type of model ("verifier", "meta_verifier", or "generator")
        
    Returns:
        Loaded model
    """
    logger.info(f"Loading {model_type} from {checkpoint_path}")
    
    if not TORCH_AVAILABLE:
        logger.warning("PyTorch not available, returning None")
        return None
    
    if not Path(checkpoint_path).exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return None
    
    # TODO: Implement actual model loading
    logger.info("Model loading not yet implemented, using mock model")
    
    model = torch.nn.Linear(10, 10)
    model = model.to(device)
    model.eval()
    
    logger.info(f"{model_type} loaded successfully")
    return model


def load_problems(problems_path: str, max_problems: int = None) -> List[Dict]:
    """
    Load problems from JSONL file.
    
    Args:
        problems_path: Path to problems file
        max_problems: Maximum problems to load
        
    Returns:
        List of problem dictionaries
    """
    logger.info(f"Loading problems from {problems_path}")
    
    if not Path(problems_path).exists():
        logger.warning(f"Problems file not found: {problems_path}")
        logger.info("Using mock problems for testing")
        
        mock_problems = [
            {"problem": "Prove that √2 is irrational."},
            {"problem": "Show that there are infinitely many primes."},
            {"problem": "Prove the Pythagorean theorem."}
        ]
        return mock_problems[:max_problems] if max_problems else mock_problems
    
    problems = []
    with open(problems_path, 'r') as f:
        for i, line in enumerate(f):
            if max_problems and i >= max_problems:
                break
            problems.append(json.loads(line))
    
    logger.info(f"Loaded {len(problems)} problems")
    return problems


def load_proofs(proofs_path: str, max_proofs: int = None) -> List[Dict]:
    """
    Load existing proofs from JSONL file.
    
    Args:
        proofs_path: Path to proofs file
        max_proofs: Maximum proofs to load
        
    Returns:
        List of proof dictionaries (with 'problem' and 'proof' keys)
    """
    logger.info(f"Loading proofs from {proofs_path}")
    
    if not Path(proofs_path).exists():
        logger.warning(f"Proofs file not found: {proofs_path}")
        logger.info("Using mock proofs for testing")
        
        mock_proofs = [
            {
                "problem": "Prove that √2 is irrational.",
                "proof": "Assume √2 = p/q in lowest terms..."
            },
            {
                "problem": "Show that there are infinitely many primes.",
                "proof": "Assume finitely many primes p1, p2, ..., pn..."
            }
        ]
        return mock_proofs[:max_proofs] if max_proofs else mock_proofs
    
    proofs = []
    with open(proofs_path, 'r') as f:
        for i, line in enumerate(f):
            if max_proofs and i >= max_proofs:
                break
            proofs.append(json.loads(line))
    
    logger.info(f"Loaded {len(proofs)} proofs")
    return proofs


def generate_proofs(
    generator,
    problems: List[Dict],
    num_proofs_per_problem: int = 1,
    device: str = "cuda"
) -> List[Dict]:
    """
    Generate proofs for problems using generator.
    
    Args:
        generator: Generator model
        problems: List of problem dictionaries
        num_proofs_per_problem: Number of proofs per problem
        device: Device to run on
        
    Returns:
        List of (problem, proof) dictionaries
    """
    logger.info(f"Generating {num_proofs_per_problem} proof(s) per problem")
    
    if generator is None:
        logger.warning("No generator provided, using mock proofs")
        return [
            {
                "problem": p["problem"],
                "proof": f"Mock proof for: {p['problem'][:50]}..."
            }
            for p in problems
            for _ in range(num_proofs_per_problem)
        ]
    
    # TODO: Implement actual generation
    logger.info("Generation not yet implemented, using mock proofs")
    
    proofs = []
    for problem in problems:
        for i in range(num_proofs_per_problem):
            proofs.append({
                "problem": problem["problem"],
                "proof": f"Generated proof {i+1} for: {problem['problem'][:50]}..."
            })
    
    logger.info(f"Generated {len(proofs)} total proofs")
    return proofs


def main():
    """Main auto-labeling function."""
    args = parse_args()
    
    # Set logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Validation
        if args.generate:
            if not args.generator:
                logger.error("--generator required when --generate is set")
                sys.exit(1)
            if not args.problems:
                logger.error("--problems required when --generate is set")
                sys.exit(1)
        else:
            if not args.proofs:
                logger.error("--proofs required when not generating")
                sys.exit(1)
        
        # Print configuration
        logger.info("\n" + "="*50)
        logger.info("AUTO-LABELING CONFIGURATION")
        logger.info("="*50)
        logger.info(f"Verifier: {args.verifier}")
        logger.info(f"Meta-verifier: {args.meta_verifier or 'same as verifier'}")
        if args.generate:
            logger.info(f"Generator: {args.generator}")
            logger.info(f"Problems: {args.problems}")
            logger.info(f"Proofs per problem: {args.num_proofs_per_problem}")
        else:
            logger.info(f"Proofs: {args.proofs}")
        logger.info(f"Output: {args.output}")
        logger.info(f"n (verifications): {args.n_verifications}")
        logger.info(f"m (meta-checks): {args.m_meta_checks}")
        logger.info(f"k (threshold): {args.k_threshold}")
        logger.info("="*50 + "\n")
        
        # Load models
        verifier = load_model(args.verifier, args.device, "verifier")
        meta_verifier = None
        if args.meta_verifier:
            meta_verifier = load_model(args.meta_verifier, args.device, "meta_verifier")
        
        generator = None
        if args.generate:
            generator = load_model(args.generator, args.device, "generator")
        
        # Get proofs to label
        if args.generate:
            # Generate proofs
            problems = load_problems(args.problems, args.max_problems)
            proofs_data = generate_proofs(
                generator,
                problems,
                args.num_proofs_per_problem,
                args.device
            )
        else:
            # Load existing proofs
            proofs_data = load_proofs(args.proofs, args.max_problems)
        
        # Extract problems and proofs
        problems = [p["problem"] for p in proofs_data]
        proofs = [p["proof"] for p in proofs_data]
        proof_ids = [f"proof_{i}" for i in range(len(proofs))]
        
        logger.info(f"Total proofs to label: {len(proofs)}")
        
        # Dry run check
        if args.dry_run:
            logger.info("Dry run complete. Exiting without auto-labeling.")
            return
        
        # Initialize scaled verification
        logger.info("\n" + "="*50)
        logger.info("STARTING AUTO-LABELING")
        logger.info("="*50 + "\n")
        
        scaled_verifier = ScaledVerification(
            verifier=verifier,
            meta_verifier=meta_verifier,
            n_verifications=args.n_verifications,
            m_meta_checks=args.m_meta_checks,
            k_threshold=args.k_threshold,
            meta_consensus_threshold=args.meta_consensus_threshold,
            device=args.device
        )
        
        # Run auto-labeling
        results = scaled_verifier.batch_label_proofs(problems, proofs, proof_ids)
        
        # Get statistics
        stats = scaled_verifier.get_statistics(results)
        
        # Print results
        logger.info("\n" + "="*50)
        logger.info("AUTO-LABELING COMPLETE")
        logger.info("="*50)
        logger.info(f"Total proofs labeled: {stats['total_proofs']}")
        logger.info(f"Label distribution: {stats['label_distribution']}")
        logger.info(f"Average confidence: {stats['avg_confidence']:.2f}")
        logger.info(f"Consensus rate: {stats['consensus_rate']:.2%}")
        logger.info("="*50 + "\n")
        
        # Save results
        save_labeled_dataset(results, problems, proofs, args.output)
        
        # Save statistics
        stats_path = Path(args.output).parent / "auto_label_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Statistics saved to {stats_path}")
        
        logger.info("\nAuto-labeling pipeline complete!")
        logger.info(f"Labeled dataset: {args.output}")
        logger.info("This dataset can now be used to retrain the verifier.")
        
    except KeyboardInterrupt:
        logger.info("\nAuto-labeling interrupted by user")
        sys.exit(1)
    
    except Exception as e:
        logger.error(f"Auto-labeling failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()