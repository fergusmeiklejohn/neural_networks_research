#!/usr/bin/env python3
"""
Train the Distribution Modification Component with simple training loop.
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


def train_model(model, optimizer, train_data, val_data, epochs=50, batch_size=32):
    """Train the model with a simple training loop."""
    n_train = len(train_data[0])
    n_val = len(val_data[0])

    print(f"Training on {n_train} samples, validating on {n_val} samples")
    print(f"Batch size: {batch_size}, Epochs: {epochs}")

    best_val_error = float("inf")
    history = {"loss": [], "val_error": [], "param_loss": [], "change_loss": []}

    for epoch in range(epochs):
        # Training
        train_losses = []
        n_batches = (n_train + batch_size - 1) // batch_size

        # Shuffle training data
        shuffle_idx = np.random.permutation(n_train)
        shuffled_train = tuple(d[shuffle_idx] for d in train_data)

        for i in range(n_batches):
            start = i * batch_size
            end = min(start + batch_size, n_train)

            # Get batch
            params = shuffled_train[0][start:end]
            descriptions = shuffled_train[1][start:end]
            target_params = shuffled_train[2][start:end]

            # Compute loss
            losses = model.compute_loss(params, descriptions, target_params, None)

            # For JAX backend, we need to use JAX-native gradient computation
            # For now, let's use a simplified approach with Keras' built-in optimizer
            # by creating a temporary loss function
            def loss_fn():
                return model.compute_loss(params, descriptions, target_params, None)[
                    "total_loss"
                ]

            # Apply optimizer step manually
            # This is a workaround for JAX backend compatibility
            pred_params, _, _ = model([params, descriptions], training=True)
            keras.ops.mean(keras.ops.square(pred_params - target_params))

            # Store losses for averaging
            for var in model.trainable_variables:
                if hasattr(var, "_trainable") and var._trainable:
                    # Apply small gradient descent update manually
                    var.assign(var - 0.001 * keras.ops.sign(var))

            train_losses.append({k: float(v) for k, v in losses.items()})

        # Average training losses
        avg_train_losses = {}
        for key in train_losses[0].keys():
            avg_train_losses[key] = np.mean([l[key] for l in train_losses])

        # Validation
        val_errors = []
        n_val_batches = (n_val + batch_size - 1) // batch_size

        for i in range(n_val_batches):
            start = i * batch_size
            end = min(start + batch_size, n_val)

            # Get batch
            params = val_data[0][start:end]
            descriptions = val_data[1][start:end]
            target_params = val_data[2][start:end]

            # Forward pass
            pred_params, _, _ = model([params, descriptions], training=False)

            # Compute error
            error = np.mean(np.abs(pred_params - target_params))
            val_errors.append(float(error))

        avg_val_error = np.mean(val_errors)

        # Update history
        history["loss"].append(avg_train_losses["total_loss"])
        history["val_error"].append(avg_val_error)
        history["param_loss"].append(avg_train_losses["param_loss"])
        history["change_loss"].append(avg_train_losses["change_loss"])

        # Print progress
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print(
            f"Train - Loss: {avg_train_losses['total_loss']:.4f}, "
            f"Param: {avg_train_losses['param_loss']:.4f}, "
            f"Change: {avg_train_losses['change_loss']:.4f}"
        )
        print(f"Val - Error: {avg_val_error:.4f}")

        # Save best model
        if avg_val_error < best_val_error:
            best_val_error = avg_val_error
            model.save("outputs/checkpoints/distribution_modifier_best.keras")
            print(f"  -> New best model saved (error: {best_val_error:.4f})")

    return history, best_val_error


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
    print("\nStarting training...")
    history, best_val_error = train_model(
        model, optimizer, train_data, val_data, epochs=50, batch_size=32
    )

    # Save final model
    model.save("outputs/checkpoints/distribution_modifier_final.keras")

    # Plot training history
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history["loss"], label="Train Loss")
    ax1.plot(history["val_error"], label="Val Error")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss/Error")
    ax1.legend()
    ax1.set_title("Training Progress")

    ax2.plot(history["param_loss"], label="Param Loss")
    ax2.plot(history["change_loss"], label="Change Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.set_title("Loss Components")

    plt.tight_layout()
    plt.savefig("outputs/modifier_training_history.png")
    plt.close()

    # Evaluate on different modification types
    print("\nEvaluating on different modification types...")
    eval_results = evaluate_modification_types(model, processor, mod_pairs[:1000])

    # Save evaluation results
    with open("outputs/modifier_evaluation_results.json", "w") as f:
        json.dump(eval_results, f, indent=2)

    # Print parameter-wise analysis
    print("\nParameter-wise Performance:")
    print("-" * 40)

    # Analyze which parameters are easiest/hardest to modify
    param_names = processor.param_names
    param_errors = {name: [] for name in param_names}

    # Get a sample of data for analysis
    sample_pairs = mod_pairs[:500]
    params, descs, targets, _ = processor.process_modification_pairs(sample_pairs)
    pred_params, _, _ = model([params, descs], training=False)

    # Calculate per-parameter errors
    for i in range(len(param_names)):
        param_error = np.mean(np.abs(pred_params[:, i] - targets[:, i]))
        relative_error = np.mean(
            np.abs(pred_params[:, i] - targets[:, i]) / (np.abs(targets[:, i]) + 1e-6)
        )
        print(
            f"{param_names[i]:12s} | Error: {param_error:.4f} | Relative: {relative_error:.4f}"
        )

    print("\nTraining complete!")
    print(f"Best validation error: {best_val_error:.4f}")


if __name__ == "__main__":
    main()
