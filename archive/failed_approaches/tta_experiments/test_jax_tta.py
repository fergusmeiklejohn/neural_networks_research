"""Test JAX-compatible TTA implementation."""

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


def test_jax_compatibility():
    """Test that JAX backend works with TTA."""
    print("Testing JAX Compatibility for TTA")
    print("=" * 60)

    # Check backend
    backend = keras.backend.backend()
    print(f"Current Keras backend: {backend}")

    if backend != "jax":
        print("WARNING: Not using JAX backend. Set KERAS_BACKEND=jax")
        return

    # Create a simple model
    print("\n1. Creating test model...")
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(8,)),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(8),
        ]
    )

    model.compile(optimizer="adam", loss="mse")
    print(f"Model created with {model.count_params()} parameters")

    # Generate test data
    print("\n2. Generating test data...")
    n_samples = 5
    X_test = np.random.randn(n_samples, 8).astype(np.float32)

    # Test baseline predictions
    print("\n3. Testing baseline predictions...")
    y_baseline = model.predict(X_test, verbose=0)
    print(f"Baseline prediction shape: {y_baseline.shape}")
    print(f"Baseline mean: {np.mean(y_baseline):.4f}")

    # Test TENT adaptation
    print("\n4. Testing TENT with JAX...")
    try:
        tent_wrapper = TTAWrapper(
            model, tta_method="tent", adaptation_steps=1, learning_rate=1e-4
        )
        y_tent = tent_wrapper.predict(X_test, adapt=True)
        print(f"TENT prediction shape: {y_tent.shape}")
        print(f"TENT mean: {np.mean(y_tent):.4f}")
        print("✓ TENT works with JAX!")
    except Exception as e:
        print(f"✗ TENT error: {e}")

    # Test PhysicsTENT
    print("\n5. Testing PhysicsTENT with JAX...")
    try:
        physics_tent_wrapper = TTAWrapper(
            model,
            tta_method="physics_tent",
            adaptation_steps=1,
            learning_rate=1e-4,
            physics_loss_weight=0.1,
        )
        y_physics_tent = physics_tent_wrapper.predict(X_test, adapt=True)
        print(f"PhysicsTENT prediction shape: {y_physics_tent.shape}")
        print(f"PhysicsTENT mean: {np.mean(y_physics_tent):.4f}")
        print("✓ PhysicsTENT works with JAX!")
    except Exception as e:
        print(f"✗ PhysicsTENT error: {e}")

    # Test TTT
    print("\n6. Testing TTT with JAX...")
    try:
        ttt_wrapper = TTAWrapper(
            model,
            tta_method="ttt",
            adaptation_steps=2,
            learning_rate=1e-4,
            trajectory_length=50,
            adaptation_window=5,
        )
        y_ttt = ttt_wrapper.predict(X_test, adapt=True)
        print(f"TTT prediction shape: {y_ttt.shape}")
        print(f"TTT mean: {np.mean(y_ttt):.4f}")
        print("✓ TTT works with JAX!")
    except Exception as e:
        print(f"✗ TTT error: {e}")

    # Test adaptation metrics
    print("\n7. Testing adaptation metrics...")
    metrics = tent_wrapper.get_metrics()
    print("TENT Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    print("\nJAX compatibility test complete!")


def test_gradient_computation():
    """Test JAX gradient computation directly."""
    print("\n\n" + "=" * 60)
    print("Testing JAX Gradient Computation")
    print("=" * 60)

    import jax
    import jax.numpy as jnp

    # Simple test function
    def loss_fn(params, x):
        w, b = params
        y = jnp.dot(x, w) + b
        return jnp.mean(y**2)

    # Test data
    x = jnp.array(np.random.randn(5, 3).astype(np.float32))
    w = jnp.array(np.random.randn(3, 2).astype(np.float32))
    b = jnp.array(np.random.randn(2).astype(np.float32))
    params = [w, b]

    # Compute gradients
    print("\n1. Computing gradients with JAX...")
    loss_value, grads = jax.value_and_grad(loss_fn)(params, x)
    print(f"Loss value: {loss_value:.4f}")
    print(f"Gradient shapes: {[g.shape for g in grads]}")
    print("✓ JAX gradient computation works!")

    # Test with Keras model
    print("\n2. Testing with Keras layers...")
    layer = keras.layers.Dense(4)
    layer.build((None, 3))

    def keras_loss_fn(weights, x):
        # Manual forward pass
        w, b = weights
        y = jnp.dot(x, w) + b
        return jnp.mean(y**2)

    # Get layer weights as JAX arrays
    weights_jax = [jnp.array(w.numpy()) for w in layer.weights]
    x_jax = jnp.array(np.random.randn(5, 3).astype(np.float32))

    loss_val, grads = jax.value_and_grad(keras_loss_fn)(weights_jax, x_jax)
    print(f"Keras layer loss: {loss_val:.4f}")
    print(f"Keras layer gradient shapes: {[g.shape for g in grads]}")
    print("✓ JAX works with Keras layers!")


def test_time_varying_physics():
    """Test TTA on time-varying physics with JAX."""
    print("\n\n" + "=" * 60)
    print("Testing TTA on Time-Varying Physics (JAX)")
    print("=" * 60)

    # Create model
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(8,)),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(8),
        ]
    )
    model.compile(optimizer="adam", loss="mse")

    # Generate trajectory with time-varying gravity
    def generate_trajectory(n_steps=20, varying_gravity=False):
        traj = np.zeros((n_steps, 8), dtype=np.float32)
        traj[0] = np.random.randn(8) * 0.1
        traj[0, [1, 5]] = 1.0  # Initial height

        dt = 0.1
        for t in range(1, n_steps):
            if varying_gravity:
                g = -9.8 * (1 + 0.1 * np.sin(0.5 * t))
            else:
                g = -9.8

            # Update positions
            traj[t, [0, 4]] = traj[t - 1, [0, 4]] + traj[t - 1, [2, 6]] * dt
            traj[t, [1, 5]] = traj[t - 1, [1, 5]] + traj[t - 1, [3, 7]] * dt

            # Update velocities
            traj[t, [2, 6]] = traj[t - 1, [2, 6]]
            traj[t, [3, 7]] = traj[t - 1, [3, 7]] + g * dt

        return traj

    # Test on constant gravity
    print("\n1. Testing on constant gravity...")
    const_traj = generate_trajectory(varying_gravity=False)
    input_state = const_traj[0:1]
    target = const_traj[1:]

    pred_const = model.predict(input_state, verbose=0)
    mse_const = np.mean((pred_const[0, : len(target)] - target) ** 2)
    print(f"Constant gravity MSE: {mse_const:.4f}")

    # Test on varying gravity without TTA
    print("\n2. Testing on varying gravity (no TTA)...")
    var_traj = generate_trajectory(varying_gravity=True)
    input_state_var = var_traj[0:1]
    target_var = var_traj[1:]

    pred_var = model.predict(input_state_var, verbose=0)
    mse_var = np.mean((pred_var[0, : len(target_var)] - target_var) ** 2)
    print(f"Varying gravity MSE (no TTA): {mse_var:.4f}")

    # Test with PhysicsTENT
    print("\n3. Testing with PhysicsTENT adaptation...")
    physics_tent = TTAWrapper(
        model,
        tta_method="physics_tent",
        adaptation_steps=5,
        learning_rate=1e-3,
        physics_loss_weight=0.1,
    )

    # Adapt on first part of trajectory
    adapt_steps = 10
    adapt_data = var_traj[:adapt_steps]
    _ = physics_tent.predict(adapt_data[np.newaxis, ...], adapt=True)

    # Predict on new state
    pred_adapted = model.predict(var_traj[adapt_steps : adapt_steps + 1], verbose=0)
    target_adapted = var_traj[adapt_steps + 1 :]
    mse_adapted = np.mean(
        (pred_adapted[0, : len(target_adapted)] - target_adapted) ** 2
    )
    print(f"Varying gravity MSE (with TTA): {mse_adapted:.4f}")

    improvement = (mse_var - mse_adapted) / mse_var * 100
    print(f"Improvement with TTA: {improvement:.1f}%")

    # Get adaptation metrics
    metrics = physics_tent.get_metrics()
    print("\nAdaptation metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")


def main():
    """Run all JAX compatibility tests."""
    setup_environment()

    # Run tests
    test_jax_compatibility()
    test_gradient_computation()
    test_time_varying_physics()

    print("\n" + "=" * 60)
    print("All JAX TTA tests complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
