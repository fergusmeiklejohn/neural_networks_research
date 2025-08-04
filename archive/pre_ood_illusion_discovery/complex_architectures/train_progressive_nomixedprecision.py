#!/usr/bin/env python3
"""
Progressive training without mixed precision to avoid type errors
Keeps all other optimizations: GPU memory growth, periodic clearing
"""

import os

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import gc
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict

import tensorflow as tf
import wandb
from tensorflow import keras
from tqdm import tqdm

# Configure GPU memory growth
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU memory growth enabled for {len(gpus)} GPU(s)")
    except RuntimeError as e:
        print(f"GPU memory growth setting failed: {e}")

from scan_data_loader import SCANDataLoader
from train_progressive_curriculum import (
    ProgressiveCurriculum,
    SCANTokenizer,
    compute_accuracy,
    create_dataset,
)

# Import from current directory
from models import create_model


def train_progressive_curriculum_no_mixed_precision(config: Dict):
    """Training without mixed precision but with other memory optimizations"""

    # Setup
    print("Setting up progressive curriculum training (no mixed precision)...")
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize wandb if enabled
    if config.get("use_wandb", False):
        wandb.init(
            project=config.get("wandb_project", "compositional-language"),
            name=f"no_mixed_precision_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=config,
        )

    # Load data
    print("\nLoading data...")
    loader = SCANDataLoader(data_dir="data")
    splits = loader.load_processed_splits()

    # Load modifications
    mod_path = Path("data/processed/modification_pairs.pkl")
    if mod_path.exists():
        with open(mod_path, "rb") as f:
            modifications = pickle.load(f)
    else:
        modifications = []

    # Create tokenizer
    print("Building vocabulary...")
    tokenizer = SCANTokenizer()
    tokenizer.build_vocabulary(splits["train"])
    tokenizer.save_vocabulary(output_dir / "vocabulary.json")

    print(f"Command vocabulary size: {len(tokenizer.command_to_id)}")
    print(f"Action vocabulary size: {len(tokenizer.action_to_id)}")

    # Create model
    print("\nCreating model...")
    model = create_model(
        command_vocab_size=len(tokenizer.command_to_id),
        action_vocab_size=len(tokenizer.action_to_id),
        d_model=config.get("d_model", 128),
    )

    # Build model
    print("Building model...")
    dummy_command = tf.zeros((1, 50), dtype=tf.int32)
    dummy_target = tf.zeros((1, 99), dtype=tf.int32)
    dummy_modification = tf.zeros((1, 20), dtype=tf.int32)
    _ = model(
        {
            "command": dummy_command,
            "target": dummy_target,
            "modification": dummy_modification,
        },
        training=False,
    )
    print("Model built successfully")

    # Initialize curriculum
    curriculum = ProgressiveCurriculum(config)

    # Loss function
    loss_fn = keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction=tf.keras.losses.Reduction.NONE
    )

    # Training loop
    for stage in range(1, 5):
        stage_config = curriculum.get_stage_config(stage)
        print(f"\n{'='*60}")
        print(f"{stage_config['name']}")
        print(f"{stage_config['description']}")
        print(f"{'='*60}")

        # Clear memory before each stage
        tf.keras.backend.clear_session()
        gc.collect()

        # Create stage-specific dataset
        if stage_config.get("use_modifications", False):
            dataset = create_dataset(
                splits["train"],
                tokenizer,
                modifications=modifications,
                modification_ratio=stage_config.get("modification_ratio", 0.0),
                batch_size=config["batch_size"],
            )
        else:
            dataset = create_dataset(
                splits["train"], tokenizer, batch_size=config["batch_size"]
            )

        # Create validation dataset
        val_dataset = create_dataset(
            splits["val_interpolation"], tokenizer, batch_size=config["batch_size"]
        )

        # Setup optimizer
        optimizer = keras.optimizers.Adam(learning_rate=stage_config["lr"])

        # Training epochs
        for epoch in range(stage_config["epochs"]):
            print(f"\nEpoch {epoch + 1}/{stage_config['epochs']}")

            # Training metrics
            train_loss = 0
            num_batches = 0

            # Progress bar
            pbar = tqdm(dataset, desc="Training")

            for i, batch in enumerate(pbar):
                with tf.GradientTape() as tape:
                    # Prepare inputs
                    inputs = {
                        "command": batch["command"],
                        "target": batch["action"][:, :-1],
                        "modification": batch["modification"],
                    }

                    # Forward pass
                    outputs = model(inputs, training=True)

                    # Compute loss
                    logits = outputs["logits"]
                    targets = batch["action"][:, 1:]

                    # Mask padding
                    mask = tf.cast(
                        targets != tokenizer.action_to_id["<PAD>"], tf.float32
                    )
                    loss_per_token = loss_fn(targets, logits)
                    masked_loss = loss_per_token * mask
                    loss = tf.reduce_sum(masked_loss) / tf.reduce_sum(mask)

                # Compute gradients
                gradients = tape.gradient(loss, model.trainable_variables)

                # Filter None gradients
                valid_grads_and_vars = [
                    (g, v)
                    for g, v in zip(gradients, model.trainable_variables)
                    if g is not None
                ]

                # Apply gradients
                optimizer.apply_gradients(valid_grads_and_vars)

                # Update metrics
                train_loss += loss.numpy()
                num_batches += 1

                # Update progress bar
                pbar.set_postfix({"loss": f"{train_loss/num_batches:.4f}"})

                # Periodic memory clearing
                if i > 0 and i % 200 == 0:
                    gc.collect()
                    # Note: We don't clear TF session during training as it would reset the model

            # Validation every 5 epochs
            if epoch % 5 == 0:
                val_acc = compute_accuracy(
                    model, val_dataset, tokenizer, max_samples=100
                )
                print(f"  Train Loss: {train_loss/num_batches:.4f}")
                print(f"  Val Accuracy: {val_acc:.2%}")

                if config.get("use_wandb", False):
                    wandb.log(
                        {
                            f"stage_{stage}/train_loss": train_loss / num_batches,
                            f"stage_{stage}/val_accuracy": val_acc,
                            "epoch": sum(
                                [
                                    curriculum.get_stage_config(s)["epochs"]
                                    for s in range(1, stage)
                                ]
                            )
                            + epoch,
                        }
                    )

        # Save stage checkpoint
        checkpoint_path = output_dir / f"stage_{stage}_model.h5"
        model.save_weights(str(checkpoint_path))
        print(f"Saved checkpoint: {checkpoint_path}")

    # Final evaluation
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)

    # Test on different splits
    test_results = {}

    for split_name in ["test_interpolation", "test_primitive_extrap"]:
        if split_name in splits:
            test_dataset = create_dataset(
                splits[split_name], tokenizer, batch_size=config["batch_size"]
            )
            accuracy = compute_accuracy(model, test_dataset, tokenizer)
            test_results[split_name] = accuracy
            print(f"{split_name}: {accuracy:.2%}")

    # Save results
    results_path = output_dir / "training_results.json"
    with open(results_path, "w") as f:
        json.dump(
            {
                "config": config,
                "test_results": test_results,
                "vocabulary_sizes": {
                    "command": len(tokenizer.command_to_id),
                    "action": len(tokenizer.action_to_id),
                },
                "optimizations": {
                    "mixed_precision": False,
                    "memory_growth": True,
                    "tf_function": False,
                    "periodic_clearing": True,
                },
            },
            f,
            indent=2,
        )

    if config.get("use_wandb", False):
        wandb.finish()

    print(f"\nTraining complete! Results saved to {output_dir}")


def main():
    """Run training without mixed precision"""

    config = {
        # Model parameters
        "d_model": 128,
        "batch_size": 8,
        # Training epochs (reduced for memory)
        "stage1_epochs": 10,
        "stage2_epochs": 10,
        "stage3_epochs": 10,
        "stage4_epochs": 10,
        # Learning rates
        "stage1_lr": 1e-3,
        "stage2_lr": 5e-4,
        "stage3_lr": 2e-4,
        "stage4_lr": 1e-4,
        # Output
        "output_dir": "outputs/no_mixed_precision",
        "use_wandb": True,
        "wandb_project": "compositional-language-nomixedprecision",
    }

    train_progressive_curriculum_no_mixed_precision(config)


if __name__ == "__main__":
    main()
