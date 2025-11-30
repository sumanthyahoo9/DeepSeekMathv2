#!/usr/bin/env python3
"""
scripts/10_train_verifier.py

Command-line script for training proof verification models.

Usage:
    python scripts/10_train_verifier.py --config configs/verifier_config.yaml
    python scripts/10_train_verifier.py --config configs/verifier_config.yaml --resume checkpoints/verifier/checkpoint_1000.pt
"""

import argparse
import logging
import sys
from pathlib import Path
import json
import signal

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.training.verifier_trainer import VerifierTrainer

try:
    import torch
    import yaml
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    yaml = None
# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class GracefulKiller:
    """Handle graceful shutdown on SIGINT/SIGTERM."""
    
    def __init__(self):
        self.kill_now = False
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)
    
    def exit_gracefully(self, *args):
        logger.info("\nReceived interrupt signal. Saving checkpoint and exiting...")
        self.kill_now = True


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train proof verification model using GRPO"
    )
    
    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration YAML file"
    )
    
    # Resume training
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    
    # Override config options
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override output directory from config"
    )
    
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=None,
        help="Override number of epochs from config"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override batch size from config"
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Override learning rate from config"
    )
    
    # Debugging
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Dry run: setup everything but don't train"
    )
    
    return parser.parse_args()


def load_and_validate_config(config_path: str, args) -> dict:
    """
    Load configuration and validate required fields.
    
    Args:
        config_path: Path to YAML config file
        args: Command-line arguments (for overrides)
        
    Returns:
        Configuration dictionary
    """
    logger.info(f"Loading configuration from {config_path}")
    
    if not TORCH_AVAILABLE:
        logger.warning("PyTorch not available, running in mock mode")
        return {}
    
    # Load YAML config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Apply command-line overrides
    if args.output_dir is not None:
        config['logging']['output_dir'] = args.output_dir
    
    if args.num_epochs is not None:
        config['training']['num_epochs'] = args.num_epochs
    
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    
    if args.learning_rate is not None:
        config['training']['learning_rate'] = args.learning_rate
    
    if args.debug:
        config['advanced']['debug_mode'] = True
        config['logging']['logging_steps'] = 1
    
    # Validate required fields
    required_fields = [
        'model.name',
        'data.train_dataset',
        'grpo.group_size',
        'reward.alpha_format',
        'reward.alpha_score',
        'training.learning_rate',
        'training.batch_size',
        'training.num_epochs'
    ]
    
    for field in required_fields:
        keys = field.split('.')
        value = config
        for key in keys:
            if key not in value:
                raise ValueError(f"Missing required config field: {field}")
            value = value[key]
    
    logger.info("Configuration loaded and validated successfully")
    return config


def create_datasets(config: dict):
    """
    Create training and validation datasets.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    logger.info("Creating datasets...")
    
    train_path = config['data']['train_dataset']
    val_path = config['data'].get('val_dataset', None)
    
    # Check if data files exist
    if not Path(train_path).exists():
        logger.warning(f"Training data not found at {train_path}")
        logger.info("Creating mock dataset for testing")
        
        # Create mock dataset
        class MockVerificationDataset:
            def __init__(self, size=100):
                self.size = size
            
            def __len__(self):
                return self.size
            
            def __getitem__(self, idx):
                return {
                    'problem': f'Problem {idx}',
                    'proof': f'Proof for problem {idx}',
                    'score': 0.5 if idx % 2 == 0 else 1.0
                }
        
        train_dataset = MockVerificationDataset(100)
        val_dataset = MockVerificationDataset(20) if val_path else None
        
        logger.info(f"Created mock datasets: train={len(train_dataset)}, val={len(val_dataset) if val_dataset else 0}")
        return train_dataset, val_dataset
    
    # Load real datasets
    # TODO: Implement actual dataset loading from JSONL files
    # For now, use mock datasets
    logger.info("Real dataset loading not yet implemented, using mock datasets")
    
    class MockVerificationDataset:
        def __init__(self, size=100):
            self.size = size
        
        def __len__(self):
            return self.size
        
        def __getitem__(self, idx):
            return {
                'problem': f'Problem {idx}',
                'proof': f'Proof for problem {idx}',
                'score': 0.5 if idx % 2 == 0 else 1.0
            }
    
    train_dataset = MockVerificationDataset(100)
    val_dataset = MockVerificationDataset(20) if val_path else None
    
    logger.info(f"Datasets created: train={len(train_dataset)}, val={len(val_dataset) if val_dataset else 0}")
    return train_dataset, val_dataset


def create_model_and_tokenizer(config: dict):
    """
    Create model and tokenizer.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info("Creating model and tokenizer...")
    
    if not TORCH_AVAILABLE:
        logger.warning("PyTorch not available, returning None")
        return None, None
    
    model_name = config['model']['name']
    checkpoint = config['model'].get('checkpoint', None)
    
    # TODO: Load actual model and tokenizer
    # For now, create mock model
    logger.info(f"Model loading not yet implemented, creating mock model for {model_name}")
    
    class MockTokenizer:
        def __call__(self, text, **kwargs):
            return {'input_ids': torch.randint(0, 1000, (len(text), 10))}
    
    model = torch.nn.Linear(10, 10)
    tokenizer = MockTokenizer()
    
    logger.info(f"Model created: {model_name}")
    
    if checkpoint:
        logger.info(f"Loading checkpoint from {checkpoint}")
        # TODO: Load checkpoint
    
    return model, tokenizer


def setup_trainer(config: dict, model, tokenizer, train_dataset, val_dataset):
    """
    Setup VerifierTrainer with configuration.
    
    Args:
        config: Configuration dictionary
        model: Model to train
        tokenizer: Tokenizer
        train_dataset: Training dataset
        val_dataset: Validation dataset
        
    Returns:
        VerifierTrainer instance
    """
    logger.info("Setting up trainer...")
    
    output_dir = config['logging']['output_dir']
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save config to output directory
    config_save_path = Path(output_dir) / "config.yaml"
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    logger.info(f"Saved configuration to {config_save_path}")
    
    # Initialize trainer
    trainer = VerifierTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=output_dir,
        # GRPO parameters
        group_size=config['grpo']['group_size'],
        kl_coef=config['grpo']['kl_coef'],
        clip_range=config['grpo']['clip_range'],
        # Reward parameters
        alpha_format=config['reward']['alpha_format'],
        alpha_score=config['reward']['alpha_score'],
        use_meta_verification=config['reward'].get('use_meta_verification', False),
        # Training parameters
        learning_rate=config['training']['learning_rate'],
        batch_size=config['training']['batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        num_epochs=config['training']['num_epochs'],
        max_grad_norm=config['training']['max_grad_norm'],
        warmup_steps=config['training']['warmup_steps'],
        logging_steps=config['logging']['logging_steps'],
        eval_steps=config['logging']['eval_steps'],
        save_steps=config['logging']['save_steps'],
        # Generation parameters
        max_new_tokens=config['generation']['max_new_tokens'],
        temperature=config['generation']['temperature'],
        top_p=config['generation']['top_p']
    )
    
    logger.info("Trainer initialized successfully")
    return trainer


def main():
    """Main training function."""
    args = parse_args()
    
    # Setup graceful shutdown
    killer = GracefulKiller()
    
    try:
        # Load configuration
        config = load_and_validate_config(args.config, args)
        
        # Print configuration summary
        logger.info("\n" + "="*50)
        logger.info("VERIFIER TRAINING CONFIGURATION")
        logger.info("="*50)
        logger.info(f"Model: {config['model']['name']}")
        logger.info(f"Output directory: {config['logging']['output_dir']}")
        logger.info(f"Batch size: {config['training']['batch_size']}")
        logger.info(f"Group size (K): {config['grpo']['group_size']}")
        logger.info(f"Learning rate: {config['training']['learning_rate']}")
        logger.info(f"Epochs: {config['training']['num_epochs']}")
        logger.info(f"Reward weights: α_format={config['reward']['alpha_format']}, α_score={config['reward']['alpha_score']}")
        logger.info("="*50 + "\n")
        
        # Create datasets
        train_dataset, val_dataset = create_datasets(config)
        
        # Create model and tokenizer
        model, tokenizer = create_model_and_tokenizer(config)
        
        # Setup trainer
        trainer = setup_trainer(config, model, tokenizer, train_dataset, val_dataset)
        
        # Resume from checkpoint if specified
        if args.resume:
            logger.info(f"Resuming from checkpoint: {args.resume}")
            trainer.load_checkpoint(args.resume)
        
        # Dry run check
        if args.dry_run:
            logger.info("Dry run complete. Exiting without training.")
            return
        
        # Start training
        logger.info("\n" + "="*50)
        logger.info("STARTING TRAINING")
        logger.info("="*50 + "\n")
        
        training_history = trainer.train()
        
        # Check for interruption
        if killer.kill_now:
            logger.info("Training interrupted. Saving final checkpoint...")
            trainer.save_checkpoint(
                epoch=training_history['epochs'][-1]['epoch'],
                is_best=False
            )
            return
        
        # Training complete
        logger.info("\n" + "="*50)
        logger.info("TRAINING COMPLETE")
        logger.info("="*50)
        logger.info(f"Best reward: {training_history['best_reward']:.4f}")
        logger.info(f"Total epochs: {len(training_history['epochs'])}")
        logger.info(f"Checkpoints saved to: {config['logging']['output_dir']}")
        logger.info("="*50 + "\n")
        
        # Save final results
        results_path = Path(config['logging']['output_dir']) / "training_results.json"
        with open(results_path, 'w') as f:
            json.dump(training_history, f, indent=2)
        logger.info(f"Training results saved to {results_path}")
        
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
        sys.exit(1)
    
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()