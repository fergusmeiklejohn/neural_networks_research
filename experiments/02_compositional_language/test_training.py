#!/usr/bin/env python3
"""Quick test of the compositional language training pipeline"""

import os

os.environ["KERAS_BACKEND"] = "tensorflow"

from train_progressive_curriculum import train_progressive_curriculum

# Minimal config for testing
config = {
    # Model parameters
    "d_model": 128,  # Smaller model
    "batch_size": 16,
    # Very short training for testing
    "stage1_epochs": 1,
    "stage2_epochs": 1,
    "stage3_epochs": 1,
    "stage4_epochs": 1,
    "stage1_lr": 1e-3,
    "stage2_lr": 5e-4,
    "stage3_lr": 2e-4,
    "stage4_lr": 1e-4,
    # Output
    "output_dir": "outputs/test_run",
    "use_wandb": False,
}

print("Running minimal test of compositional language training...")
train_progressive_curriculum(config)
print("\nTest complete!")
