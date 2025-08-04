"""Simple test of TTA functionality with fresh models."""

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


def create_simple_physics_model(input_dim=8, output_dim=8, sequence_length=49):
    """Create a simple model for testing."""
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(1, input_dim)),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.BatchNormalization(),  # Important for TENT
            keras.layers.Dense(64, activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(output_dim * sequence_length),
            keras.layers.Reshape((sequence_length, output_dim)),
        ]
    )
    return model


def generate_test_trajectory(n_steps=50, noise_level=0.1):
    """Generate a simple test trajectory."""
    # Simple falling ball
    trajectory = np.zeros((n_steps, 8))

    # Initial conditions
    trajectory[0] = np.random.randn(8) * 0.1
    trajectory[0, [1, 5]] = 1.0  # Start at y=1

    # Simulate with gravity
    g = -9.8
    dt = 0.1

    for t in range(1, n_steps):
        # Update positions
        trajectory[t, [0, 4]] = (
            trajectory[t - 1, [0, 4]] + trajectory[t - 1, [2, 6]] * dt
        )
        trajectory[t, [1, 5]] = (
            trajectory[t - 1, [1, 5]] + trajectory[t - 1, [3, 7]] * dt
        )

        # Update velocities (gravity on y)
        trajectory[t, [2, 6]] = trajectory[t - 1, [2, 6]]
        trajectory[t, [3, 7]] = trajectory[t - 1, [3, 7]] + g * dt

        # Add noise
        trajectory[t] += np.random.randn(8) * noise_level

    return trajectory


def test_tta_methods():
    """Test different TTA methods."""
    print("Creating test model...")
    model = create_simple_physics_model()

    # Compile model
    model.compile(optimizer="adam", loss="mse")

    print(f"Model created with {model.count_params()} parameters")
    print(f"Input shape: {model.input_shape}")
    print(f"Output shape: {model.output_shape}")

    # Generate test data
    print("\nGenerating test trajectories...")
    n_test = 5
    test_trajectories = [generate_test_trajectory() for _ in range(n_test)]

    # Test 1: Baseline (no TTA)
    print("\n1. Testing baseline (no TTA)...")
    mse_baseline = []
    for traj in test_trajectories:
        input_state = traj[0:1]
        target = traj[1:]
        pred = model.predict(input_state[np.newaxis, ...], verbose=0)
        mse = np.mean((pred[0] - target) ** 2)
        mse_baseline.append(mse)

    print(f"   Baseline MSE: {np.mean(mse_baseline):.4f} ± {np.std(mse_baseline):.4f}")

    # Test 2: TENT
    print("\n2. Testing TENT...")
    tent_wrapper = TTAWrapper(
        model, tta_method="tent", adaptation_steps=1, learning_rate=1e-4
    )
    mse_tent = []

    for traj in test_trajectories:
        input_state = traj[0:1]
        target = traj[1:]
        pred = tent_wrapper.predict(input_state[np.newaxis, ...], adapt=True)
        mse = np.mean((pred[0] - target) ** 2)
        mse_tent.append(mse)
        tent_wrapper.reset()

    print(f"   TENT MSE: {np.mean(mse_tent):.4f} ± {np.std(mse_tent):.4f}")
    improvement = (
        (np.mean(mse_baseline) - np.mean(mse_tent)) / np.mean(mse_baseline) * 100
    )
    print(f"   Improvement: {improvement:.1f}%")

    # Test 3: PhysicsTENT
    print("\n3. Testing PhysicsTENT...")
    physics_tent_wrapper = TTAWrapper(
        model,
        tta_method="physics_tent",
        adaptation_steps=1,
        learning_rate=1e-4,
        physics_loss_weight=0.1,
    )
    mse_physics_tent = []

    for traj in test_trajectories:
        input_state = traj[0:1]
        target = traj[1:]
        pred = physics_tent_wrapper.predict(input_state[np.newaxis, ...], adapt=True)
        mse = np.mean((pred[0] - target) ** 2)
        mse_physics_tent.append(mse)
        physics_tent_wrapper.reset()

    print(
        f"   PhysicsTENT MSE: {np.mean(mse_physics_tent):.4f} ± {np.std(mse_physics_tent):.4f}"
    )
    improvement = (
        (np.mean(mse_baseline) - np.mean(mse_physics_tent))
        / np.mean(mse_baseline)
        * 100
    )
    print(f"   Improvement: {improvement:.1f}%")

    # Test 4: TTT
    print("\n4. Testing TTT (Test-Time Training)...")
    ttt_wrapper = TTAWrapper(
        model,
        tta_method="ttt",
        adaptation_steps=5,
        learning_rate=1e-4,
        trajectory_length=50,
        adaptation_window=5,
    )
    mse_ttt = []

    for traj in test_trajectories:
        # For TTT, we can use more of the trajectory for adaptation
        context_length = 10
        input_context = traj[:context_length]
        target = traj[context_length:]

        # Adapt on context
        _ = ttt_wrapper.predict(input_context[np.newaxis, ...], adapt=True)

        # Predict future
        pred = model.predict(input_context[-1:][np.newaxis, ...], verbose=0)

        # Compare with remaining trajectory
        min_len = min(pred.shape[1], len(target))
        mse = np.mean((pred[0, :min_len] - target[:min_len]) ** 2)
        mse_ttt.append(mse)
        ttt_wrapper.reset()

    print(f"   TTT MSE: {np.mean(mse_ttt):.4f} ± {np.std(mse_ttt):.4f}")
    improvement = (
        (np.mean(mse_baseline) - np.mean(mse_ttt)) / np.mean(mse_baseline) * 100
    )
    print(f"   Improvement: {improvement:.1f}%")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Baseline:     {np.mean(mse_baseline):.4f}")
    print(
        f"TENT:         {np.mean(mse_tent):.4f} ({(np.mean(mse_baseline) - np.mean(mse_tent)) / np.mean(mse_baseline) * 100:.1f}% improvement)"
    )
    print(
        f"PhysicsTENT:  {np.mean(mse_physics_tent):.4f} ({(np.mean(mse_baseline) - np.mean(mse_physics_tent)) / np.mean(mse_baseline) * 100:.1f}% improvement)"
    )
    print(
        f"TTT:          {np.mean(mse_ttt):.4f} ({(np.mean(mse_baseline) - np.mean(mse_ttt)) / np.mean(mse_baseline) * 100:.1f}% improvement)"
    )

    # Test adaptation metrics
    print("\nTENT Adaptation Metrics:")
    metrics = tent_wrapper.get_metrics()
    for key, value in metrics.items():
        print(f"  {key}: {value}")


def test_time_varying_gravity():
    """Test TTA on time-varying gravity scenario."""
    print("\n" + "=" * 60)
    print("Testing on Time-Varying Gravity")
    print("=" * 60)

    model = create_simple_physics_model()
    model.compile(optimizer="adam", loss="mse")

    # Generate trajectory with time-varying gravity
    def generate_varying_gravity_trajectory(n_steps=50):
        trajectory = np.zeros((n_steps, 8))
        trajectory[0] = np.random.randn(8) * 0.1
        trajectory[0, [1, 5]] = 1.0

        dt = 0.1
        for t in range(1, n_steps):
            # Time-varying gravity!
            g = -9.8 * (1 + 0.1 * np.sin(0.5 * t))

            # Update positions
            trajectory[t, [0, 4]] = (
                trajectory[t - 1, [0, 4]] + trajectory[t - 1, [2, 6]] * dt
            )
            trajectory[t, [1, 5]] = (
                trajectory[t - 1, [1, 5]] + trajectory[t - 1, [3, 7]] * dt
            )

            # Update velocities
            trajectory[t, [2, 6]] = trajectory[t - 1, [2, 6]]
            trajectory[t, [3, 7]] = trajectory[t - 1, [3, 7]] + g * dt

        return trajectory

    # Test on time-varying gravity
    test_traj = generate_varying_gravity_trajectory()

    # Baseline
    input_state = test_traj[0:1]
    target = test_traj[1:]
    pred_baseline = model.predict(input_state[np.newaxis, ...], verbose=0)
    mse_baseline = np.mean((pred_baseline[0] - target) ** 2)
    print(f"\nBaseline MSE on time-varying gravity: {mse_baseline:.4f}")

    # With TTT
    ttt_wrapper = TTAWrapper(
        model,
        tta_method="ttt",
        adaptation_steps=10,
        learning_rate=1e-3,
        trajectory_length=50,
        adaptation_window=10,
    )

    # Use first 20 steps for adaptation
    adapt_context = test_traj[:20]
    _ = ttt_wrapper.predict(adapt_context[np.newaxis, ...], adapt=True)

    # Predict remaining
    pred_ttt = model.predict(adapt_context[-1:][np.newaxis, ...], verbose=0)
    target_ttt = test_traj[20:]
    min_len = min(pred_ttt.shape[1], len(target_ttt))
    mse_ttt = np.mean((pred_ttt[0, :min_len] - target_ttt[:min_len]) ** 2)

    print(f"TTT MSE on time-varying gravity: {mse_ttt:.4f}")
    print(f"Improvement: {(mse_baseline - mse_ttt) / mse_baseline * 100:.1f}%")

    # Check if TTT detected time-varying gravity
    if hasattr(ttt_wrapper.tta_adapter, "adaptation_state"):
        state = ttt_wrapper.tta_adapter.adaptation_state
        if state.get("estimated_physics"):
            print(f"\nEstimated physics parameters:")
            print(f"  Gravity: {state['estimated_physics']['gravity']:.2f}")
            print(f"  Time-varying: {state['estimated_physics']['time_varying']}")


def main():
    """Run all tests."""
    setup_environment()

    print("Testing Test-Time Adaptation Methods")
    print("=" * 60)

    # Basic TTA tests
    test_tta_methods()

    # Time-varying gravity test
    test_time_varying_gravity()

    print("\nAll tests complete!")


if __name__ == "__main__":
    main()
