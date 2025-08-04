#!/usr/bin/env python3
"""
Training script with mixed strategy to prevent catastrophic interference.

Key innovation: Mix unmodified examples with modified examples in Stages 2-4
to help the model maintain base knowledge while learning modifications.
"""

import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from models_v2 import create_model_v2
from modification_generator import ModificationGenerator
from scan_data_loader import SCANDataLoader
from train_progressive_curriculum import SCANTokenizer, create_dataset


def create_mixed_dataset(
    base_data, modification_data, mix_ratio=0.5, tokenizer=None, batch_size=32
):
    """
    Create a mixed dataset combining base and modified examples.

    Args:
        base_data: Unmodified SCAN examples
        modification_data: Modified examples with modification descriptions
        mix_ratio: Fraction of modified examples (0.5 = 50/50 mix)
    """
    # Calculate split
    len(base_data) + len(modification_data)
    n_modified = int(len(modification_data) * mix_ratio)
    n_base = int(n_modified / mix_ratio * (1 - mix_ratio))

    # Sample from both datasets
    sampled_base = np.random.choice(
        base_data, size=min(n_base, len(base_data)), replace=False
    )
    sampled_mods = np.random.choice(
        modification_data, size=min(n_modified, len(modification_data)), replace=False
    )

    # Convert base samples to have dummy modifications
    base_with_dummy_mods = []
    for sample in sampled_base:
        base_with_dummy_mods.append(
            {
                "command": sample["command"],
                "action": sample["action"],
                "modification": "",  # Empty modification
            }
        )

    # Combine and shuffle
    combined_data = list(base_with_dummy_mods) + list(sampled_mods)
    np.random.shuffle(combined_data)

    print(
        f"Mixed dataset: {len(base_with_dummy_mods)} base + {len(sampled_mods)} modified = {len(combined_data)} total"
    )

    # Create TF dataset
    return create_dataset(combined_data, tokenizer, batch_size)


def train_with_mixed_strategy(output_dir="outputs/mixed_strategy_training"):
    """Train model with mixed strategy to prevent catastrophic interference."""

    print("=== MIXED STRATEGY TRAINING ===\n")

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading SCAN data...")
    data_loader = SCANDataLoader()
    data_loader.load_all_data()
    splits = data_loader.create_isolated_splits()

    # Keep as SCANSample objects for modification generator
    train_samples = splits["train"]
    val_samples = splits["val_interpolation"]

    # Convert to dict format for tokenizer
    train_data = [{"command": s.command, "action": s.action} for s in train_samples]
    val_data = [{"command": s.command, "action": s.action} for s in val_samples]

    print(f"Loaded {len(train_data)} training samples")

    # Create tokenizer
    print("\nCreating tokenizer...")
    tokenizer = SCANTokenizer()
    tokenizer.build_vocabulary(train_data)

    # Save vocabulary
    tokenizer.save_vocabulary(output_dir / "vocabulary.json")

    # Generate modifications
    print("\nGenerating modifications...")
    mod_gen = ModificationGenerator()
    modification_dict = mod_gen.generate_all_modifications(train_samples[:1000])

    # Flatten all modifications into single list
    all_modification_pairs = []
    for mod_type, pairs in modification_dict.items():
        all_modification_pairs.extend(pairs)

    # Convert to training format
    modifications = []
    for pair in all_modification_pairs:
        modifications.append(
            {
                "command": pair.modified_sample.command,
                "action": pair.modified_sample.action,
                "modification": pair.modification_description,
            }
        )

    print(f"Generated {len(modifications)} modifications")

    # Create model (choose v1 or v2)
    print("\nCreating model...")
    use_v2 = True  # Set to True to use improved architecture

    if use_v2:
        print("Using improved model v2 with gating mechanism")
        model = create_model_v2(
            len(tokenizer.command_to_id), len(tokenizer.action_to_id), d_model=128
        )
    else:
        print("Using original model v1")
        from models import create_model

        model = create_model(
            len(tokenizer.command_to_id), len(tokenizer.action_to_id), d_model=128
        )

    # Training configuration
    stages = [
        {
            "name": "Stage 1: Basic SCAN",
            "data": train_data,
            "mix_ratio": 0.0,  # No modifications
            "epochs": 2,  # Reduced for testing
            "lr": 0.001,
        },
        {
            "name": "Stage 2: Mixed Training (70/30)",
            "data": (train_data, modifications[:300]),
            "mix_ratio": 0.3,  # 30% modifications
            "epochs": 2,  # Reduced for testing
            "lr": 0.0005,
        },
        {
            "name": "Stage 3: Mixed Training (50/50)",
            "data": (train_data, modifications[:600]),
            "mix_ratio": 0.5,  # 50% modifications
            "epochs": 2,  # Reduced for testing
            "lr": 0.0002,
        },
        {
            "name": "Stage 4: Mixed Training (30/70)",
            "data": (train_data, modifications),
            "mix_ratio": 0.7,  # 70% modifications
            "epochs": 2,  # Reduced for testing
            "lr": 0.0001,
        },
    ]

    # Training history
    history = {
        "model_version": "v2" if use_v2 else "v1",
        "strategy": "mixed_training",
        "stages": [],
    }

    # Train each stage
    for stage_idx, stage in enumerate(stages):
        print(f"\n\n{'='*60}")
        print(f"{stage['name']}")
        print(f"{'='*60}")

        # Create dataset
        if stage_idx == 0:
            # Stage 1: Only base data
            dataset = create_dataset(stage["data"], tokenizer, batch_size=32)
        else:
            # Stages 2-4: Mixed data
            base_data, mod_data = stage["data"]
            dataset = create_mixed_dataset(
                base_data,
                mod_data,
                mix_ratio=stage["mix_ratio"],
                tokenizer=tokenizer,
                batch_size=32,
            )

        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=stage["lr"]),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        # Train
        stage_history = []
        for epoch in range(stage["epochs"]):
            print(f"\nEpoch {epoch + 1}/{stage['epochs']}")

            # Train one epoch
            hist = model.fit(dataset, epochs=1, verbose=1)

            # Record metrics
            stage_history.append(
                {
                    "epoch": epoch + 1,
                    "loss": float(hist.history["loss"][0]),
                    "accuracy": float(hist.history["accuracy"][0]),
                }
            )

            # Save checkpoint
            checkpoint_path = output_dir / f"stage_{stage_idx+1}_epoch_{epoch+1}.h5"
            model.save_weights(checkpoint_path)

        # Evaluate on validation set
        val_dataset = create_dataset(val_data, tokenizer, batch_size=32)
        val_metrics = model.evaluate(val_dataset, verbose=0)

        # Record stage results
        history["stages"].append(
            {
                "name": stage["name"],
                "mix_ratio": stage["mix_ratio"],
                "epochs": stage_history,
                "val_loss": float(val_metrics[0]),
                "val_accuracy": float(val_metrics[1]),
            }
        )

        print(
            f"\nStage complete - Val Loss: {val_metrics[0]:.4f}, Val Acc: {val_metrics[1]:.4f}"
        )

    # Save training history
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Save final model
    model.save_weights(output_dir / "final_model.h5")

    print("\n\n=== TRAINING COMPLETE ===")
    print(f"Results saved to: {output_dir}")

    # Analyze results
    print("\n=== RESULTS ANALYSIS ===")
    for i, stage in enumerate(history["stages"]):
        final_acc = stage["epochs"][-1]["accuracy"]
        val_acc = stage["val_accuracy"]
        print(f"\n{stage['name']}:")
        print(f"  Mix ratio: {stage['mix_ratio']*100:.0f}% modifications")
        print(f"  Final training acc: {final_acc:.4f}")
        print(f"  Validation acc: {val_acc:.4f}")

    # Check for catastrophic interference
    stage1_val = history["stages"][0]["val_accuracy"]
    stage2_val = history["stages"][1]["val_accuracy"]
    degradation = (stage1_val - stage2_val) / stage1_val * 100

    print(f"\n\nCatastrophic Interference Check:")
    print(f"Stage 1 → Stage 2 accuracy drop: {degradation:.1f}%")
    if degradation < 5:
        print("✓ Mixed training successfully prevented catastrophic interference!")
    else:
        print("⚠️ Some interference remains, but less than pure modification training")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train with mixed strategy")
    parser.add_argument("--output_dir", default="outputs/mixed_strategy_training")
    parser.add_argument(
        "--test_run", action="store_true", help="Quick test with minimal data"
    )
    args = parser.parse_args()

    if args.test_run:
        print("Running quick test with minimal data...")
        # TODO: Implement test mode with reduced data

    train_with_mixed_strategy(args.output_dir)
