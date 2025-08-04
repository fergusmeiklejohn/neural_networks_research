"""Test to diagnose why TTA predictions aren't changing."""

import pickle
import sys
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.imports import setup_project_paths

setup_project_paths()

import keras

from models.test_time_adaptation.tta_wrappers import TTAWrapper
from utils.config import setup_environment
from utils.paths import get_data_path


def create_simple_model():
    """Create a simple test model."""
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(1, 8)),
            keras.layers.Flatten(),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(80),
            keras.layers.Reshape((10, 8)),
        ]
    )

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
    return model


def main():
    """Test TTA prediction behavior."""
    print("Testing TTA Prediction Issue")
    print("=" * 70)

    # Setup
    setup_environment()

    # Load data
    data_dir = get_data_path() / "true_ood_physics"
    varying_files = sorted(data_dir.glob("time_varying_gravity_*.pkl"))
    with open(varying_files[-1], "rb") as f:
        ood_data = pickle.load(f)

    # Create and setup model
    model = create_simple_model()

    # Get test trajectory
    test_traj = ood_data["trajectories"][0]
    X = test_traj[0:1].reshape(1, 1, 8)
    y_true = test_traj[1:11]

    # Initialize with random weights
    _ = model(X)  # Build model

    print("\n1. Testing baseline predictions:")
    y_baseline = model.predict(X, verbose=0)
    print(f"   Baseline prediction shape: {y_baseline.shape}")
    print(f"   First prediction: {y_baseline[0, 0, :2]}")

    # Create TTA wrapper
    print("\n2. Creating TTA wrapper...")
    tta_model = TTAWrapper(
        model,
        tta_method="regression_v2",
        adaptation_steps=5,
        learning_rate=1e-3,  # Higher LR to force changes
        consistency_loss_weight=0.0,
        smoothness_loss_weight=0.0,
    )

    # Method 1: Using predict with adapt=True
    print("\n3. Testing TTA predictions (method 1: predict with adapt=True):")
    y_tta_1 = tta_model.predict(X, adapt=True)
    if len(y_tta_1.shape) == 3:
        y_tta_1 = y_tta_1[0]
    print(f"   TTA prediction shape: {y_tta_1.shape}")
    print(f"   First prediction: {y_tta_1[0, :2]}")

    # Check if predictions changed
    diff_1 = np.abs(y_tta_1 - y_baseline[0]).mean()
    print(f"   Mean absolute difference from baseline: {diff_1:.6f}")

    # Reset model
    tta_model.reset()

    # Method 2: Direct adaptation then prediction
    print("\n4. Testing direct adaptation (method 2):")
    # Adapt
    adapted_pred = tta_model.tta_adapter.adapt(X)
    print(f"   Adapted prediction shape: {adapted_pred.shape}")
    print(f"   First prediction: {adapted_pred[0, :2]}")

    # Now check model's prediction after adaptation
    y_after_adapt = model.predict(X, verbose=0)
    print(f"\n   Model prediction after adapt: {y_after_adapt[0, 0, :2]}")

    diff_2 = np.abs(adapted_pred - y_baseline).mean()
    diff_3 = np.abs(y_after_adapt - y_baseline).mean()
    print(f"   Adapted pred diff from baseline: {diff_2:.6f}")
    print(f"   Model pred diff from baseline: {diff_3:.6f}")

    # Test weight changes
    print("\n5. Checking weight changes:")
    # Get first layer weights before
    first_dense = [l for l in model.layers if isinstance(l, keras.layers.Dense)][0]
    weights_before = first_dense.get_weights()[0].copy()

    # Reset and adapt again
    tta_model.reset()
    _ = tta_model.predict(X, adapt=True)

    # Get weights after
    weights_after = first_dense.get_weights()[0]
    weight_diff = np.abs(weights_after - weights_before).mean()
    print(f"   Mean weight change: {weight_diff:.6f}")

    # Test BatchNorm statistics
    bn_layer = [
        l for l in model.layers if isinstance(l, keras.layers.BatchNormalization)
    ][0]
    print(f"\n6. BatchNorm statistics:")
    print(f"   Moving mean shape: {bn_layer.moving_mean.shape}")
    print(f"   Moving mean (first 5): {bn_layer.moving_mean.numpy()[:5]}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY:")
    print(f"- Predictions changed (method 1): {'YES' if diff_1 > 1e-6 else 'NO'}")
    print(f"- Predictions changed (method 2): {'YES' if diff_2 > 1e-6 else 'NO'}")
    print(
        f"- Model predictions changed after adapt: {'YES' if diff_3 > 1e-6 else 'NO'}"
    )
    print(f"- Weights changed: {'YES' if weight_diff > 1e-6 else 'NO'}")

    if diff_1 < 1e-6 and diff_2 > 1e-6:
        print(
            "\n⚠️  ISSUE: adapt() returns different predictions but predict() doesn't use them!"
        )
        print("    The adapted state may not be properly propagated.")


if __name__ == "__main__":
    main()
