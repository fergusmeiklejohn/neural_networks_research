#!/usr/bin/env python3
"""
Train the Distribution Modification Component using Keras 3.
"""

import os

os.environ["KERAS_BACKEND"] = "jax"

import json
import pickle
from pathlib import Path

import keras
import matplotlib.pyplot as plt
import numpy as np
from distribution_modifier import DistributionModifier, ModificationDataProcessor


class ModifierTrainingModel(keras.Model):
    """Training wrapper for distribution modifier."""

    def __init__(self, modifier):
        super().__init__()
        self.modifier = modifier
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.param_error_tracker = keras.metrics.Mean(name="param_error")

    def call(self, inputs, training=None):
        return self.modifier(inputs, training=training)

    @property
    def metrics(self):
        return [self.loss_tracker, self.param_error_tracker]

    def compute_loss(self, x, y, sample_weight=None):
        params, descriptions = x
        target_params = y

        # Get predictions
        pred_params, _, _ = self.modifier([params, descriptions], training=True)

        # Compute loss
        param_loss = keras.ops.mean(keras.ops.square(pred_params - target_params))

        # Track metrics
        self.loss_tracker.update_state(param_loss)
        param_error = keras.ops.mean(keras.ops.abs(pred_params - target_params))
        self.param_error_tracker.update_state(param_error)

        return param_loss


def load_and_process_data():
    """Load and process modification data."""
    # Load data
    print("Loading modification pairs data...")
    data_path = Path("data/processed/physics_worlds/modification_pairs.pkl")
    with open(data_path, "rb") as f:
        mod_pairs = pickle.load(f)

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

    return all_data, processor, mod_pairs


def create_datasets(data, batch_size=32, val_ratio=0.15):
    """Create training and validation datasets."""
    n_samples = len(data[0])
    n_val = int(n_samples * val_ratio)

    # Shuffle indices
    indices = np.random.permutation(n_samples)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    # Split data
    train_params = data[0][train_indices]
    train_descs = data[1][train_indices]
    train_targets = data[2][train_indices]

    val_params = data[0][val_indices]
    val_descs = data[1][val_indices]
    val_targets = data[2][val_indices]

    print(
        f"Training samples: {len(train_indices)}, Validation samples: {len(val_indices)}"
    )

    return ((train_params, train_descs), train_targets), (
        (val_params, val_descs),
        val_targets,
    )


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


def simple_training_loop(
    model, optimizer, train_data, val_data, epochs=50, batch_size=32
):
    """Simple training loop for Keras 3."""
    train_x, train_y = train_data
    val_x, val_y = val_data

    n_train = len(train_y)
    n_val = len(val_y)

    history = {"loss": [], "val_loss": [], "val_error": []}
    best_val_error = float("inf")

    for epoch in range(epochs):
        # Training
        train_losses = []
        indices = np.random.permutation(n_train)

        for i in range(0, n_train, batch_size):
            batch_idx = indices[i : i + batch_size]

            # Get batch
            batch_x = (train_x[0][batch_idx], train_x[1][batch_idx])
            batch_y = train_y[batch_idx]

            # Forward and backward pass
            with keras.tf.GradientTape() as tape:
                loss = model.compute_loss(batch_x, batch_y)

            # Compute gradients and update
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            train_losses.append(float(loss))

        # Validation
        val_losses = []
        val_errors = []

        for i in range(0, n_val, batch_size):
            batch_x = (val_x[0][i : i + batch_size], val_x[1][i : i + batch_size])
            batch_y = val_y[i : i + batch_size]

            # Forward pass only
            pred_y, _, _ = model(batch_x, training=False)
            val_loss = keras.ops.mean(keras.ops.square(pred_y - batch_y))
            val_error = keras.ops.mean(keras.ops.abs(pred_y - batch_y))

            val_losses.append(float(val_loss))
            val_errors.append(float(val_error))

        # Average metrics
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        avg_val_error = np.mean(val_errors)

        history["loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_error"].append(avg_val_error)

        # Print progress
        print(
            f"Epoch {epoch + 1}/{epochs} - "
            f"Loss: {avg_train_loss:.4f} - "
            f"Val Loss: {avg_val_loss:.4f} - "
            f"Val Error: {avg_val_error:.4f}"
        )

        # Save best model
        if avg_val_error < best_val_error:
            best_val_error = avg_val_error
            model.modifier.save("outputs/checkpoints/distribution_modifier_best.keras")
            print(f"  -> Saved best model (error: {best_val_error:.4f})")

    return history


def main():
    """Main training function."""
    # Set random seed
    np.random.seed(42)
    keras.utils.set_random_seed(42)

    # Load and process data
    all_data, processor, mod_pairs = load_and_process_data()

    # Create datasets
    train_data, val_data = create_datasets(all_data, batch_size=32)

    # Initialize model
    print("\nInitializing model...")
    base_modifier = DistributionModifier(vocab_size=processor.vocab_size, n_params=4)

    # Build model
    dummy_params = np.ones((1, 4), dtype=np.float32)
    dummy_tokens = np.ones((1, processor.max_length), dtype=np.int32)
    _ = base_modifier([dummy_params, dummy_tokens], training=False)

    print(f"Model parameters: {base_modifier.count_params():,}")

    # Create training model
    model = ModifierTrainingModel(base_modifier)

    # Initialize optimizer
    optimizer = keras.optimizers.Adam(learning_rate=1e-3)

    # Train
    print("\nStarting training...")
    history = simple_training_loop(
        model, optimizer, train_data, val_data, epochs=50, batch_size=32
    )

    # Save final model
    base_modifier.save("outputs/checkpoints/distribution_modifier_final.keras")

    # Plot training history
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    ax.plot(history["loss"], label="Train Loss")
    ax.plot(history["val_loss"], label="Val Loss")
    ax.plot(history["val_error"], label="Val Error")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss/Error")
    ax.legend()
    ax.set_title("Training Progress")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("outputs/modifier_training_history.png")
    plt.close()

    # Evaluate on different modification types
    print("\nEvaluating on different modification types...")
    eval_results = evaluate_modification_types(
        base_modifier, processor, mod_pairs[:1000]
    )

    # Save evaluation results
    with open("outputs/modifier_evaluation_results.json", "w") as f:
        json.dump(eval_results, f, indent=2)

    # Analyze per-parameter performance
    print("\nParameter-wise Performance:")
    print("-" * 40)

    # Get a sample for analysis
    sample_data = processor.process_modification_pairs(mod_pairs[:500])
    params, descs, targets, _ = sample_data
    pred_params, _, _ = base_modifier([params, descs], training=False)

    # Calculate per-parameter metrics
    for i, param_name in enumerate(processor.param_names):
        param_error = np.mean(np.abs(pred_params[:, i] - targets[:, i]))
        relative_error = np.mean(
            np.abs(pred_params[:, i] - targets[:, i]) / (np.abs(targets[:, i]) + 1e-6)
        )

        # Direction accuracy
        target_dir = np.sign(targets[:, i] - params[:, i])
        pred_dir = np.sign(pred_params[:, i] - params[:, i])
        dir_acc = np.mean(target_dir == pred_dir)

        print(
            f"{param_name:12s} | Error: {param_error:.4f} | "
            f"Relative: {relative_error:.4f} | Direction: {dir_acc:.4f}"
        )

    print("\nTraining complete!")
    print(f"Best validation error: {min(history['val_error']):.4f}")


if __name__ == "__main__":
    main()
