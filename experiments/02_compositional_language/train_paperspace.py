#!/usr/bin/env python3
"""
Full-scale training script for Paperspace GPU

Run this on Paperspace A4000 for complete progressive curriculum training.
"""

import os

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from train_progressive_curriculum import train_progressive_curriculum

# Full training configuration
config = {
    # Model parameters
    "d_model": 256,
    "batch_size": 64,  # Larger batch for GPU
    # Full training epochs
    "stage1_epochs": 50,
    "stage2_epochs": 50,
    "stage3_epochs": 50,
    "stage4_epochs": 50,
    # Learning rates
    "stage1_lr": 1e-3,
    "stage2_lr": 5e-4,
    "stage3_lr": 2e-4,
    "stage4_lr": 1e-4,
    # Output and logging
    "output_dir": "outputs/full_training",
    "use_wandb": True,
    "wandb_project": "compositional-language-invention",
}

print("=" * 60)
print("Compositional Language Progressive Curriculum Training")
print("=" * 60)
print(f"Model size: ~50M parameters")
print(f"Total epochs: 200 (50 per stage)")
print(f"Batch size: {config['batch_size']}")
print(f"Estimated time: 6-8 hours on A4000")
print("=" * 60)

# Run training
train_progressive_curriculum(config)

print("\nTraining complete!")
