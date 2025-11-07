"""
Train Quote Overlay Model

This script leverages the elem2design (LLaVA) training infrastructure
to train a model for quote overlay generation.
"""

import os
import sys
import argparse
import yaml
from pathlib import Path

# Add parent directory to path to import llava
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from llava.train.train import train


def load_config(config_path: str) -> dict:
    """Load training configuration from YAML."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_training_args(config: dict) -> list:
    """
    Convert YAML config to command-line arguments for LLaVA trainer.

    Args:
        config: Configuration dictionary

    Returns:
        List of command-line arguments
    """
    args = []

    # Model arguments
    args.extend([
        f"--model_name_or_path={config['model']['base_model']}",
        f"--vision_tower={config['model']['vision_tower']}",
        f"--mm_vision_select_layer={config['model']['mm_vision_select_layer']}",
        f"--mm_vision_select_feature={config['model']['mm_vision_select_feature']}",
        f"--mm_projector_type={config['model']['mm_projector_type']}",
    ])

    # Data arguments
    args.extend([
        f"--data_path={config['data']['train_data_path']}",
        f"--image_folder={config['data']['image_folder']}",
        f"--image_aspect_ratio={config['data']['image_aspect_ratio']}",
    ])

    # Training arguments
    train_cfg = config['training']
    args.extend([
        f"--output_dir={train_cfg['output_dir']}",
        f"--num_train_epochs={train_cfg['num_train_epochs']}",
        f"--per_device_train_batch_size={train_cfg['per_device_train_batch_size']}",
        f"--per_device_eval_batch_size={train_cfg['per_device_eval_batch_size']}",
        f"--gradient_accumulation_steps={train_cfg['gradient_accumulation_steps']}",
        f"--learning_rate={train_cfg['learning_rate']}",
        f"--weight_decay={train_cfg['weight_decay']}",
        f"--warmup_steps={train_cfg['warmup_steps']}",
        f"--max_grad_norm={train_cfg['max_grad_norm']}",
        f"--logging_steps={train_cfg['logging_steps']}",
        f"--save_strategy={train_cfg['save_strategy']}",
        f"--save_steps={train_cfg['save_steps']}",
        f"--save_total_limit={train_cfg['save_total_limit']}",
        f"--evaluation_strategy={train_cfg['evaluation_strategy']}",
        f"--eval_steps={train_cfg['eval_steps']}",
        f"--optim={train_cfg['optim']}",
        f"--model_max_length={config['data']['model_max_length']}",
        f"--dataloader_num_workers={train_cfg['dataloader_num_workers']}",
        f"--seed={train_cfg['seed']}",
    ])

    # LoRA arguments
    if config['lora']['enable']:
        args.extend([
            "--lora_enable",
            f"--lora_r={config['lora']['r']}",
            f"--lora_alpha={config['lora']['alpha']}",
            f"--lora_dropout={config['lora']['dropout']}",
            f"--lora_bias={config['lora']['bias']}",
        ])

    # Mixed precision
    if train_cfg['fp16']:
        args.append("--fp16")
    if train_cfg['bf16']:
        args.append("--bf16")

    # Gradient checkpointing
    if train_cfg['gradient_checkpointing']:
        args.append("--gradient_checkpointing")

    # Remove unused columns
    if not train_cfg['remove_unused_columns']:
        args.append("--remove_unused_columns=False")

    # MM projector learning rate
    if 'mm_projector_lr' in train_cfg:
        args.append(f"--mm_projector_lr={train_cfg['mm_projector_lr']}")

    # DeepSpeed
    if train_cfg.get('deepspeed'):
        args.append(f"--deepspeed={train_cfg['deepspeed']}")

    # Weights & Biases
    if config['wandb']['enabled']:
        args.extend([
            "--report_to=wandb",
            f"--run_name={config['wandb']['name']}",
        ])
        if config['wandb']['project']:
            os.environ['WANDB_PROJECT'] = config['wandb']['project']
        if config['wandb']['entity']:
            os.environ['WANDB_ENTITY'] = config['wandb']['entity']
    else:
        args.append("--report_to=none")

    return args


def main():
    parser = argparse.ArgumentParser(description="Train Quote Overlay Model")
    parser.add_argument("--config", type=str, default="./training/training_config.yaml",
                       help="Path to training configuration YAML file")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Override output directory from config")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                       help="Path to checkpoint to resume from")
    args = parser.parse_args()

    print("=" * 80)
    print("Quote Overlay Model Training")
    print("=" * 80)

    # Load configuration
    print(f"\nLoading configuration from {args.config}...")
    config = load_config(args.config)

    # Override output dir if specified
    if args.output_dir:
        config['training']['output_dir'] = args.output_dir

    # Print configuration summary
    print("\nConfiguration Summary:")
    print(f"  Base Model: {config['model']['base_model']}")
    print(f"  Vision Tower: {config['model']['vision_tower']}")
    print(f"  LoRA: {'Enabled' if config['lora']['enable'] else 'Disabled'}")
    if config['lora']['enable']:
        print(f"    - Rank: {config['lora']['r']}")
        print(f"    - Alpha: {config['lora']['alpha']}")
    print(f"  Training Data: {config['data']['train_data_path']}")
    print(f"  Output Directory: {config['training']['output_dir']}")
    print(f"  Epochs: {config['training']['num_train_epochs']}")
    print(f"  Batch Size: {config['training']['per_device_train_batch_size']} (per device)")
    print(f"  Gradient Accumulation: {config['training']['gradient_accumulation_steps']}")
    print(f"  Effective Batch Size: {config['training']['per_device_train_batch_size'] * config['training']['gradient_accumulation_steps']}")
    print(f"  Learning Rate: {config['training']['learning_rate']}")

    # Create output directory
    output_dir = Path(config['training']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config to output directory
    config_save_path = output_dir / "training_config.yaml"
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"\n✓ Configuration saved to {config_save_path}")

    # Convert config to command-line arguments
    training_args = create_training_args(config)

    # Add resume checkpoint if specified
    if args.resume_from_checkpoint:
        print(f"\n⚠️  Resuming from checkpoint: {args.resume_from_checkpoint}")
        # LLaVA trainer will automatically resume if checkpoint exists

    print("\n" + "=" * 80)
    print("Starting Training...")
    print("=" * 80)
    print()

    # Override sys.argv for HfArgumentParser
    sys.argv = ['train_quote_model.py'] + training_args

    # Call LLaVA train function
    try:
        train()
    except Exception as e:
        print(f"\n✗ Training failed with error: {e}")
        raise

    print("\n" + "=" * 80)
    print("✓ Training Complete!")
    print("=" * 80)
    print(f"\nModel saved to: {config['training']['output_dir']}")
    print("\nNext steps:")
    print("1. Evaluate the model: python evaluation/evaluate_readability.py")
    print("2. Test inference: python inference/generate_quote.py")
    print("3. Launch demo: python app/quote_app.py")


if __name__ == "__main__":
    main()
