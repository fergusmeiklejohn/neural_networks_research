"""Diagnose why TTA is returning zero predictions."""

import pickle
import sys
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.imports import setup_project_paths

setup_project_paths()

import jax.numpy as jnp
import keras
from keras import ops

from models.test_time_adaptation.regression_tta_v2 import RegressionTTAV2
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


def prepare_data(trajectories, input_steps=1, output_steps=10):
    """Prepare trajectory data for training."""
    X, y = [], []
    for traj in trajectories:
        for i in range(len(traj) - input_steps - output_steps + 1):
            X.append(traj[i : i + input_steps])
            y.append(traj[i + input_steps : i + input_steps + output_steps])
    return np.array(X), np.array(y)


def main():
    """Diagnose zero predictions in TTA."""
    print("Diagnosing TTA Zero Predictions")
    print("=" * 70)

    # Setup
    setup_environment()

    # Load data
    data_dir = get_data_path() / "true_ood_physics"
    const_files = sorted(data_dir.glob("constant_gravity_*.pkl"))
    with open(const_files[-1], "rb") as f:
        const_data = pickle.load(f)

    # Create and train model properly
    model = create_simple_model()

    print("1. Training model on constant gravity data...")
    X_train, y_train = prepare_data(const_data["trajectories"][:50])
    model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)

    # Test on training data
    test_traj = const_data["trajectories"][60]  # Different from training
    X_test = test_traj[0:1].reshape(1, 1, 8)

    print("\n2. Testing trained model predictions:")
    y_baseline = model.predict(X_test, verbose=0)
    print(f"   Baseline prediction shape: {y_baseline.shape}")
    print(f"   First values: {y_baseline[0, 0, :4]}")
    print(f"   Prediction range: [{y_baseline.min():.2f}, {y_baseline.max():.2f}]")

    # Create TTA adapter directly
    print("\n3. Creating RegressionTTAV2 adapter...")
    adapter = RegressionTTAV2(
        model,
        adaptation_steps=1,
        learning_rate=1e-4,
        consistency_weight=0.0,
        smoothness_weight=0.0,
    )

    # Test stateless call
    print("\n4. Testing model stateless_call:")
    trainable_vars = [ops.convert_to_numpy(v) for v in model.trainable_variables]
    non_trainable_vars = [
        ops.convert_to_numpy(v) for v in model.non_trainable_variables
    ]

    # Convert to JAX
    trainable_vars_jax = [jnp.array(v) for v in trainable_vars]
    non_trainable_vars_jax = [jnp.array(v) for v in non_trainable_vars]
    X_test_jax = jnp.array(X_test)

    # Try stateless call
    try:
        y_stateless, _ = model.stateless_call(
            trainable_vars_jax, non_trainable_vars_jax, X_test_jax, training=False
        )
        print(f"   Stateless prediction shape: {y_stateless.shape}")
        print(f"   First values: {np.array(y_stateless)[0, 0, :4]}")
    except Exception as e:
        print(f"   ERROR in stateless_call: {e}")

    # Test adaptation
    print("\n5. Testing adaptation:")
    y_adapted = adapter.adapt(X_test)
    print(f"   Adapted prediction shape: {y_adapted.shape}")
    print(
        f"   First values: {y_adapted[0, 0, :4] if y_adapted.ndim > 2 else y_adapted[0, :4]}"
    )
    print(f"   Adapted range: [{y_adapted.min():.2f}, {y_adapted.max():.2f}]")

    # Check if all zeros
    if np.allclose(y_adapted, 0):
        print("\n   ⚠️  WARNING: Adapted predictions are all zeros!")

        # Check model state
        print("\n6. Checking model state after adaptation:")
        y_after = model.predict(X_test, verbose=0)
        print(f"   Model prediction after adapt: {y_after[0, 0, :4]}")

        # Check if weights changed
        first_layer = model.layers[1]  # First Dense layer
        weights = first_layer.get_weights()[0]
        print(f"   First layer weight sum: {weights.sum():.4f}")
        print(
            f"   First layer weight range: [{weights.min():.4f}, {weights.max():.4f}]"
        )

    # Test TTA wrapper
    print("\n7. Testing TTAWrapper:")
    tta_wrapper = TTAWrapper(model, tta_method="regression_v2", adaptation_steps=1)
    tta_wrapper.reset()  # Ensure clean state

    y_tta = tta_wrapper.predict(X_test, adapt=True)
    print(f"   TTA wrapper prediction shape: {y_tta.shape}")
    print(f"   First values: {y_tta[0, :4] if y_tta.ndim == 2 else y_tta[0, 0, :4]}")

    # Summary
    print("\n" + "=" * 70)
    print("DIAGNOSIS:")
    baseline_ok = not np.allclose(y_baseline, 0)
    adapted_ok = not np.allclose(y_adapted, 0)

    print(f"- Baseline predictions OK: {'YES' if baseline_ok else 'NO'}")
    print(f"- Adapted predictions OK: {'YES' if adapted_ok else 'NO'}")

    if baseline_ok and not adapted_ok:
        print("\n🔍 Issue: Adaptation is producing zero outputs")
        print("   Possible causes:")
        print("   1. Stateless call not working correctly")
        print("   2. Gradient computation issues")
        print("   3. Model state not properly initialized in JAX")


if __name__ == "__main__":
    main()
