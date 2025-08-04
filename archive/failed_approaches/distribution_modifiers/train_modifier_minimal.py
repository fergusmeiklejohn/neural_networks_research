#!/usr/bin/env python3
"""
Minimal training script for Distribution Modification Component.
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


def load_and_process_data():
    """Load and process modification data."""
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


def split_data(data, val_ratio=0.15):
    """Split data into train and validation sets."""
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


def train_epoch(model, optimizer, train_data, batch_size=32):
    """Train for one epoch."""
    n_samples = len(train_data[0])
    indices = np.random.permutation(n_samples)

    losses = []

    for i in range(0, n_samples, batch_size):
        batch_idx = indices[i : i + batch_size]

        # Get batch
        params = train_data[0][batch_idx]
        descriptions = train_data[1][batch_idx]
        target_params = train_data[2][batch_idx]

        # Define loss function for this batch
        def loss_fn():
            pred_params, _, _ = model([params, descriptions], training=True)
            return keras.ops.mean(keras.ops.square(pred_params - target_params))

        # Compute loss and gradients
        loss, grads = keras.ops.value_and_grad(loss_fn)(model.trainable_variables)

        # Update weights
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        losses.append(float(loss))

    return np.mean(losses)


def validate(model, val_data, batch_size=32):
    """Validate the model."""
    n_samples = len(val_data[0])
    errors = []

    for i in range(0, n_samples, batch_size):
        # Get batch
        params = val_data[0][i : i + batch_size]
        descriptions = val_data[1][i : i + batch_size]
        target_params = val_data[2][i : i + batch_size]

        # Predict
        pred_params, _, _ = model([params, descriptions], training=False)

        # Compute error
        error = keras.ops.mean(keras.ops.abs(pred_params - target_params))
        errors.append(float(error))

    return np.mean(errors)


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

    # Load and process data
    all_data, processor, mod_pairs = load_and_process_data()

    # Split data
    train_data, val_data = split_data(all_data)
    print(
        f"\nTraining samples: {len(train_data[0])}, Validation samples: {len(val_data[0])}"
    )

    # Initialize model
    print("\nInitializing model...")
    model = DistributionModifier(vocab_size=processor.vocab_size, n_params=4)

    # Build model
    dummy_params = np.ones((1, 4), dtype=np.float32)
    dummy_tokens = np.ones((1, processor.max_length), dtype=np.int32)
    _ = model([dummy_params, dummy_tokens], training=False)

    print(f"Model parameters: {model.count_params():,}")

    # Initialize optimizer
    optimizer = keras.optimizers.Adam(learning_rate=1e-3)

    # Training loop
    epochs = 50
    batch_size = 32
    best_val_error = float("inf")
    history = {"loss": [], "val_error": []}

    print("\nStarting training...")
    for epoch in range(epochs):
        # Train
        train_loss = train_epoch(model, optimizer, train_data, batch_size)

        # Validate
        val_error = validate(model, val_data, batch_size)

        # Store history
        history["loss"].append(train_loss)
        history["val_error"].append(val_error)

        # Print progress
        print(
            f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f} - Val Error: {val_error:.4f}"
        )

        # Save best model
        if val_error < best_val_error:
            best_val_error = val_error
            model.save("outputs/checkpoints/distribution_modifier_best.keras")
            print(f"  -> Saved best model (error: {best_val_error:.4f})")

    # Save final model
    model.save("outputs/checkpoints/distribution_modifier_final.keras")

    # Plot training history
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history["loss"])
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(history["val_error"])
    plt.title("Validation Error")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Absolute Error")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("outputs/modifier_training_history.png", dpi=150)
    plt.close()

    # Evaluate on different modification types
    print("\nEvaluating on different modification types...")
    eval_results = evaluate_modification_types(model, processor, mod_pairs[:1000])

    # Save evaluation results
    with open("outputs/modifier_evaluation_results.json", "w") as f:
        json.dump(eval_results, f, indent=2)

    # Test examples
    print("\n" + "=" * 60)
    print("Testing modification examples:")
    print("=" * 60)

    test_examples = [
        ("increase gravity by 20%", [800.0, 0.5, 0.7, 0.95]),
        ("decrease friction significantly", [600.0, 0.8, 0.5, 0.9]),
        ("make objects more bouncy", [700.0, 0.3, 0.4, 0.92]),
    ]

    for desc, base_params in test_examples:
        test_params = np.array([base_params], dtype=np.float32)
        test_desc = processor.encode_description(desc).reshape(1, -1)

        pred, mod_factors, change_mask = model([test_params, test_desc], training=False)

        print(f"\nRequest: '{desc}'")
        print(f"Base params:     {test_params[0]}")
        print(f"Predicted:       {pred[0].numpy()}")
        print(f"Mod factors:     {mod_factors[0].numpy()}")
        print(f"Change mask:     {change_mask[0].numpy()}")

        # Show which parameters changed
        for i, param_name in enumerate(processor.param_names):
            if change_mask[0, i] > 0.5:
                print(f"  -> {param_name} changed by factor {mod_factors[0, i]:.3f}")

    print("\nTraining complete!")
    print(f"Best validation error: {best_val_error:.4f}")


if __name__ == "__main__":
    main()
