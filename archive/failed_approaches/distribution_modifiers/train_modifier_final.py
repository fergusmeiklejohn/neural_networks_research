#!/usr/bin/env python3
"""
Train the Distribution Modification Component using standard Keras fit.
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


class ModifierWrapper(keras.Model):
    """Wrapper to make the modifier work with Keras fit()."""

    def __init__(self, modifier):
        super().__init__()
        self.modifier = modifier

    def call(self, inputs, training=None):
        # Unpack inputs - Keras will pass them as a list/tuple
        if isinstance(inputs, (list, tuple)) and len(inputs) == 2:
            params, descriptions = inputs
        else:
            # During fit(), Keras might concatenate inputs
            # Split them back
            params = inputs[:, :4]  # First 4 columns are parameters
            descriptions = inputs[:, 4:]  # Rest are description tokens
            descriptions = keras.ops.cast(descriptions, "int32")

        pred_params, _, _ = self.modifier([params, descriptions], training=training)
        return pred_params

    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None):
        """Override compute_loss for custom loss calculation."""
        if y_pred is None:
            y_pred = self(x, training=True)

        loss = keras.ops.mean(keras.ops.square(y - y_pred))
        return loss


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


def prepare_data_for_keras(data, val_ratio=0.15):
    """Prepare data in format suitable for Keras fit()."""
    n_samples = len(data[0])
    n_val = int(n_samples * val_ratio)

    # Shuffle indices
    indices = np.random.permutation(n_samples)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]

    # Combine params and descriptions into single input array
    params = data[0]
    descriptions = data[1]
    targets = data[2]

    # Create combined input (params + descriptions)
    combined_input = np.concatenate([params, descriptions], axis=1)

    # Split into train/val
    x_train = combined_input[train_indices]
    y_train = targets[train_indices]
    x_val = combined_input[val_indices]
    y_val = targets[val_indices]

    return (x_train, y_train), (x_val, y_val)


def create_callbacks():
    """Create training callbacks."""
    return [
        keras.callbacks.ModelCheckpoint(
            "outputs/checkpoints/distribution_modifier_best.keras",
            monitor="val_mae",
            save_best_only=True,
            mode="min",
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_mae", patience=10, restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_mae", factor=0.5, patience=5, min_lr=1e-6, verbose=1
        ),
    ]


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
    base_modifier = model.modifier

    for mod_type, indices in mod_types.items():
        if len(indices) == 0:
            continue

        # Process test samples
        test_pairs = [test_data[i] for i in indices]
        params, descs, targets, _ = processor.process_modification_pairs(test_pairs)

        # Predict
        pred_params, _, _ = base_modifier([params, descs], training=False)

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

    # Prepare data for Keras
    (x_train, y_train), (x_val, y_val) = prepare_data_for_keras(all_data)
    print(f"\nTraining samples: {len(x_train)}, Validation samples: {len(x_val)}")

    # Initialize base modifier
    print("\nInitializing model...")
    base_modifier = DistributionModifier(vocab_size=processor.vocab_size, n_params=4)

    # Build model
    dummy_params = np.ones((1, 4), dtype=np.float32)
    dummy_tokens = np.ones((1, processor.max_length), dtype=np.int32)
    _ = base_modifier([dummy_params, dummy_tokens], training=False)

    print(f"Model parameters: {base_modifier.count_params():,}")

    # Create wrapper model
    model = ModifierWrapper(base_modifier)

    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse", metrics=["mae"]
    )

    # Train
    print("\nStarting training...")
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=create_callbacks(),
        verbose=1,
    )

    # Save final model
    base_modifier.save("outputs/checkpoints/distribution_modifier_final.keras")

    # Plot training history
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history.history["loss"], label="Train Loss")
    ax1.plot(history.history["val_loss"], label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss (MSE)")
    ax1.legend()
    ax1.set_title("Loss History")
    ax1.grid(True, alpha=0.3)

    ax2.plot(history.history["mae"], label="Train MAE")
    ax2.plot(history.history["val_mae"], label="Val MAE")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Mean Absolute Error")
    ax2.legend()
    ax2.set_title("Error History")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("outputs/modifier_training_history.png", dpi=150)
    plt.close()

    # Evaluate on different modification types
    print("\nEvaluating on different modification types...")
    eval_results = evaluate_modification_types(model, processor, mod_pairs[:1000])

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
        mask = np.abs(targets[:, i] - params[:, i]) > 0.01  # Only count actual changes
        if np.sum(mask) > 0:
            target_dir = np.sign(targets[mask, i] - params[mask, i])
            pred_dir = np.sign(pred_params[mask, i] - params[mask, i])
            dir_acc = np.mean(target_dir == pred_dir)
        else:
            dir_acc = 0.0

        print(
            f"{param_name:12s} | Error: {param_error:.4f} | "
            f"Relative: {relative_error:.4f} | Direction: {dir_acc:.4f}"
        )

    print("\nTraining complete!")
    print(f"Best validation MAE: {min(history.history['val_mae']):.4f}")

    # Test a few examples
    print("\n" + "=" * 60)
    print("Testing a few modification examples:")
    print("=" * 60)

    test_cases = [
        ("increase gravity by 20%", {"gravity": 1.2}),
        ("decrease friction significantly", {"friction": 0.5}),
        ("make objects more bouncy", {"elasticity": 1.5}),
        ("reduce damping", {"damping": 0.8}),
    ]

    for desc, expected_change in test_cases:
        # Create test input
        test_params = np.array(
            [[800.0, 0.5, 0.7, 0.95]], dtype=np.float32
        )  # Example params
        test_desc = processor.encode_description(desc).reshape(1, -1)

        # Predict
        pred, _, change_mask = base_modifier([test_params, test_desc], training=False)

        print(f"\nRequest: '{desc}'")
        print(f"Original params: {test_params[0]}")
        print(f"Predicted params: {pred[0].numpy()}")
        print(f"Change mask: {change_mask[0].numpy()}")

        # Check if changes match expected
        for param, factor in expected_change.items():
            if param in processor.param_indices:
                idx = processor.param_indices[param]
                original = test_params[0, idx]
                predicted = pred[0, idx].numpy()
                actual_factor = predicted / original
                print(
                    f"  {param}: {original:.3f} -> {predicted:.3f} (factor: {actual_factor:.3f})"
                )


if __name__ == "__main__":
    main()
