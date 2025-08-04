#!/usr/bin/env python3
"""
Train the Distribution Modification Component.

This script loads the modification pairs data and trains the model to learn
physics parameter modifications from natural language requests.
"""

import os

os.environ["KERAS_BACKEND"] = "jax"

import json
import pickle
from datetime import datetime
from pathlib import Path

import jax
import keras
import numpy as np
import wandb
from distribution_modifier import DistributionModifier, ModificationDataProcessor


def load_modification_data(data_path: Path):
    """Load modification pairs from pickle file."""
    with open(data_path, "rb") as f:
        return pickle.load(f)


def create_train_val_split(data, val_ratio=0.1):
    """Split data into training and validation sets."""
    n_samples = len(data[0])
    n_val = int(n_samples * val_ratio)

    # Shuffle indices
    indices = np.random.permutation(n_samples)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    # Split data
    train_data = tuple(d[train_indices] for d in data)
    val_data = tuple(d[val_indices] for d in data)

    return train_data, val_data


class ModificationTrainer:
    """Handles training of the distribution modifier."""

    def __init__(self, model, optimizer, wandb_run=None):
        self.model = model
        self.optimizer = optimizer
        self.wandb_run = wandb_run

    def train_step(self, batch):
        """Single training step."""
        params, descriptions, target_params, _ = batch

        # JAX-based gradient computation
        def compute_loss_fn(trainable_variables):
            # Temporarily set the model's trainable variables
            for var, new_val in zip(
                self.model.trainable_variables, trainable_variables
            ):
                var.assign(new_val)

            losses = self.model.compute_loss(params, descriptions, target_params, None)
            return losses["total_loss"], losses

        # Get current trainable variables
        trainable_vars = [var.numpy() for var in self.model.trainable_variables]

        # Compute gradients using JAX
        grad_fn = jax.value_and_grad(compute_loss_fn, has_aux=True)
        (loss, losses), grads = grad_fn(trainable_vars)

        # Apply gradients
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        return losses

    def val_step(self, batch):
        """Single validation step."""
        params, descriptions, target_params, _ = batch

        # Forward pass
        pred_params, _, _ = self.model([params, descriptions], training=False)

        # Compute metrics
        param_error = keras.ops.mean(keras.ops.abs(pred_params - target_params))
        relative_error = keras.ops.mean(
            keras.ops.abs(pred_params - target_params)
            / (keras.ops.abs(target_params) + 1e-6)
        )

        # Check which parameters changed correctly
        actual_changes = keras.ops.abs(target_params - params) / (
            keras.ops.abs(params) + 1e-6
        )
        pred_changes = keras.ops.abs(pred_params - params) / (
            keras.ops.abs(params) + 1e-6
        )

        # Direction accuracy: did we change params in the right direction?
        target_dirs = keras.ops.sign(target_params - params)
        pred_dirs = keras.ops.sign(pred_params - params)
        direction_acc = keras.ops.mean(
            keras.ops.cast(target_dirs == pred_dirs, "float32")
        )

        return {
            "param_error": param_error,
            "relative_error": relative_error,
            "direction_accuracy": direction_acc,
        }

    def train(self, train_data, val_data, epochs=50, batch_size=32):
        """Train the model."""
        n_train = len(train_data[0])
        n_val = len(val_data[0])

        print(f"Training on {n_train} samples, validating on {n_val} samples")
        print(f"Batch size: {batch_size}, Epochs: {epochs}")

        best_val_error = float("inf")

        for epoch in range(epochs):
            # Training
            train_losses = []
            n_batches = (n_train + batch_size - 1) // batch_size

            for i in range(n_batches):
                start = i * batch_size
                end = min(start + batch_size, n_train)

                batch = tuple(d[start:end] for d in train_data)
                losses = self.train_step(batch)
                train_losses.append({k: float(v) for k, v in losses.items()})

            # Average training losses
            avg_train_losses = {}
            for key in train_losses[0].keys():
                avg_train_losses[key] = np.mean([l[key] for l in train_losses])

            # Validation
            val_metrics = []
            n_val_batches = (n_val + batch_size - 1) // batch_size

            for i in range(n_val_batches):
                start = i * batch_size
                end = min(start + batch_size, n_val)

                batch = tuple(d[start:end] for d in val_data)
                metrics = self.val_step(batch)
                val_metrics.append({k: float(v) for k, v in metrics.items()})

            # Average validation metrics
            avg_val_metrics = {}
            for key in val_metrics[0].keys():
                avg_val_metrics[key] = np.mean([m[key] for m in val_metrics])

            # Print progress
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print(
                f"Train - Loss: {avg_train_losses['total_loss']:.4f}, "
                f"Param: {avg_train_losses['param_loss']:.4f}, "
                f"Change: {avg_train_losses['change_loss']:.4f}"
            )
            print(
                f"Val - Param Error: {avg_val_metrics['param_error']:.4f}, "
                f"Relative: {avg_val_metrics['relative_error']:.4f}, "
                f"Direction: {avg_val_metrics['direction_accuracy']:.4f}"
            )

            # Log to wandb
            if self.wandb_run:
                log_dict = {
                    "epoch": epoch + 1,
                    "train/total_loss": avg_train_losses["total_loss"],
                    "train/param_loss": avg_train_losses["param_loss"],
                    "train/change_loss": avg_train_losses["change_loss"],
                    "train/magnitude_loss": avg_train_losses["magnitude_loss"],
                    "val/param_error": avg_val_metrics["param_error"],
                    "val/relative_error": avg_val_metrics["relative_error"],
                    "val/direction_accuracy": avg_val_metrics["direction_accuracy"],
                }
                wandb.log(log_dict)

            # Save best model
            if avg_val_metrics["param_error"] < best_val_error:
                best_val_error = avg_val_metrics["param_error"]
                self.model.save("outputs/checkpoints/distribution_modifier_best.keras")
                print(f"  -> New best model saved (error: {best_val_error:.4f})")


def evaluate_modification_types(model, processor, test_data):
    """Evaluate performance on different modification types."""
    mod_types = {}

    # Group by modification type
    for i, pair in enumerate(test_data):
        mod_type = pair["modification_type"]
        if mod_type not in mod_types:
            mod_types[mod_type] = []
        mod_types[mod_type].append(i)

    print("\nEvaluation by Modification Type:")
    print("-" * 60)

    results = {}

    for mod_type, indices in mod_types.items():
        if len(indices) == 0:
            continue

        # Process test samples
        test_pairs = [test_data[i] for i in indices]
        params, descs, targets, _ = processor.process_modification_pairs(test_pairs)

        # Predict
        pred_params, _, _ = model([params, descs], training=False)

        # Compute metrics
        param_error = np.mean(np.abs(pred_params - targets))
        relative_error = np.mean(
            np.abs(pred_params - targets) / (np.abs(targets) + 1e-6)
        )

        # Direction accuracy
        target_dirs = np.sign(targets - params)
        pred_dirs = np.sign(pred_params - params)
        direction_acc = np.mean(target_dirs == pred_dirs)

        results[mod_type] = {
            "count": len(indices),
            "param_error": float(param_error),
            "relative_error": float(relative_error),
            "direction_accuracy": float(direction_acc),
        }

        print(
            f"{mod_type:25s} | n={len(indices):4d} | "
            f"Error: {param_error:.4f} | "
            f"Relative: {relative_error:.4f} | "
            f"Direction: {direction_acc:.4f}"
        )

    return results


def main():
    """Main training function."""
    # Set random seed
    np.random.seed(42)
    keras.utils.set_random_seed(42)

    # Initialize wandb
    use_wandb = False  # Set to True to enable wandb logging
    wandb_run = None
    if use_wandb:
        wandb_run = wandb.init(
            project="distribution-invention",
            name=f"physics_modifier_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "vocab_size": 500,
                "n_params": 4,
                "learning_rate": 1e-3,
                "batch_size": 32,
                "epochs": 75,
            },
        )

    # Load data
    print("Loading modification pairs data...")
    data_path = Path("data/processed/physics_worlds/modification_pairs.pkl")
    mod_pairs = load_modification_data(data_path)
    print(f"Loaded {len(mod_pairs)} modification pairs")

    # Initialize processor
    processor = ModificationDataProcessor(vocab_size=500, max_length=15)

    # Process all data
    print("\nProcessing modification pairs...")
    all_data = processor.process_modification_pairs(mod_pairs)

    # Save vocabulary
    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/checkpoints").mkdir(exist_ok=True)
    processor.save_vocabulary(Path("outputs/modifier_vocabulary.json"))

    # Create train/val split
    train_data, val_data = create_train_val_split(all_data, val_ratio=0.15)

    # Initialize model
    print("\nInitializing model...")
    model = DistributionModifier(vocab_size=processor.vocab_size, n_params=4)

    # Build model with dummy input
    dummy_params = np.ones((1, 4), dtype=np.float32)
    dummy_tokens = np.ones((1, processor.max_length), dtype=np.int32)
    _ = model([dummy_params, dummy_tokens], training=False)

    print(f"Model parameters: {model.count_params():,}")

    # Initialize optimizer
    optimizer = keras.optimizers.Adam(learning_rate=1e-3)

    # Train
    trainer = ModificationTrainer(model, optimizer, wandb_run)
    trainer.train(train_data, val_data, epochs=75, batch_size=32)

    # Save final model
    model.save("outputs/checkpoints/distribution_modifier_final.keras")

    # Evaluate on different modification types
    print("\nEvaluating on different modification types...")
    eval_results = evaluate_modification_types(model, processor, mod_pairs[:1000])

    # Save evaluation results
    with open("outputs/modifier_evaluation_results.json", "w") as f:
        json.dump(eval_results, f, indent=2)

    print("\nTraining complete!")

    if wandb_run:
        wandb.finish()


if __name__ == "__main__":
    main()
