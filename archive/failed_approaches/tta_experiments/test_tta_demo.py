"""Simplified TTA demonstration without gradient issues."""

import sys
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.imports import setup_project_paths

setup_project_paths()

import keras

from utils.config import setup_environment


def create_simple_model():
    """Create a simple model for testing."""
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
    return model


def simple_tent_adaptation(model, x_test, learning_rate=1e-3):
    """Simplified TENT adaptation focusing on the concept."""
    # TENT concept: minimize entropy of predictions

    # Get initial prediction
    y_pred_initial = model(x_test, training=False)

    # In TENT, we would:
    # 1. Compute entropy of predictions
    # 2. Update BatchNorm statistics to minimize entropy
    # 3. Optionally update BatchNorm parameters

    # For demonstration, let's just update the model with test data
    # This simulates the BatchNorm adaptation
    _ = model(x_test, training=True)  # This updates BN statistics

    # Get adapted prediction
    y_pred_adapted = model(x_test, training=False)

    return y_pred_initial, y_pred_adapted


def demonstrate_tta_concept():
    """Demonstrate the concept of test-time adaptation."""
    print("Test-Time Adaptation Concept Demonstration")
    print("=" * 60)

    # Create model
    model = create_simple_model()
    model.compile(optimizer="adam", loss="mse")

    # Generate synthetic training data (constant gravity)
    print("\n1. Training on constant gravity data...")
    n_train = 100
    X_train = np.random.randn(n_train, 8)
    # Simple linear transformation representing constant physics
    W_constant = np.random.randn(8, 8) * 0.1
    W_constant[3, 3] = 0.9  # y-velocity decay (gravity effect)
    W_constant[7, 7] = 0.9
    y_train = X_train @ W_constant

    # Train model
    model.fit(X_train, y_train, epochs=10, verbose=0)
    print("   Training complete!")

    # Test on constant gravity (in-distribution)
    print("\n2. Testing on constant gravity (in-distribution)...")
    X_test_constant = np.random.randn(10, 8)
    y_test_constant = X_test_constant @ W_constant

    pred_constant = model.predict(X_test_constant, verbose=0)
    mse_constant = np.mean((pred_constant - y_test_constant) ** 2)
    print(f"   MSE on constant gravity: {mse_constant:.4f}")

    # Test on time-varying gravity (out-of-distribution)
    print("\n3. Testing on time-varying gravity (OOD)...")
    X_test_varying = np.random.randn(10, 8)
    # Different transformation representing time-varying physics
    W_varying = np.random.randn(8, 8) * 0.1
    W_varying[3, 3] = 0.7  # Different gravity!
    W_varying[7, 7] = 0.7
    y_test_varying = X_test_varying @ W_varying

    pred_varying_before = model.predict(X_test_varying, verbose=0)
    mse_varying_before = np.mean((pred_varying_before - y_test_varying) ** 2)
    print(f"   MSE on varying gravity (before adaptation): {mse_varying_before:.4f}")

    # Simulate adaptation
    print("\n4. Applying test-time adaptation...")
    print("   (In real TENT: minimizing prediction entropy)")
    print("   (In real TTT: using auxiliary self-supervised tasks)")

    # Simple adaptation: expose model to test distribution
    # In practice, this would be done through entropy minimization
    for _ in range(5):
        _ = model(X_test_varying, training=True)  # Update BN stats

    pred_varying_after = model.predict(X_test_varying, verbose=0)
    mse_varying_after = np.mean((pred_varying_after - y_test_varying) ** 2)
    print(f"   MSE on varying gravity (after adaptation): {mse_varying_after:.4f}")

    # Calculate improvement
    improvement = (mse_varying_before - mse_varying_after) / mse_varying_before * 100
    print(f"   Improvement: {improvement:.1f}%")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Constant gravity (in-distribution):  {mse_constant:.4f}")
    print(f"Varying gravity (before TTA):        {mse_varying_before:.4f}")
    print(f"Varying gravity (after TTA):         {mse_varying_after:.4f}")
    print(f"TTA Improvement:                     {improvement:.1f}%")

    print("\nKey Insights:")
    print("- Models trained on constant physics fail on time-varying physics")
    print("- Test-time adaptation can help by updating model statistics")
    print("- Real TTA methods (TENT, TTT) use principled objectives")
    print("- Our implementation provides these methods for physics tasks")


def explain_tta_methods():
    """Explain the different TTA methods implemented."""
    print("\n" + "=" * 60)
    print("Test-Time Adaptation Methods Explained")
    print("=" * 60)

    print("\n1. TENT (Test-time Entropy Minimization)")
    print("   - Minimizes entropy of predictions during test time")
    print("   - Updates only BatchNorm parameters")
    print("   - Fast and effective for distribution shifts")
    print("   - Paper: Wang et al., ICLR 2021")

    print("\n2. PhysicsTENT")
    print("   - Extends TENT with physics-specific losses")
    print("   - Adds energy conservation and momentum checks")
    print("   - Better for physics prediction tasks")

    print("\n3. TTT (Test-Time Training)")
    print("   - Uses auxiliary self-supervised tasks")
    print("   - Can adapt to larger distribution shifts")
    print("   - Reconstruction, consistency, smoothness tasks")
    print("   - More compute but better adaptation")

    print("\n4. Key Applications:")
    print("   - Time-varying gravity scenarios")
    print("   - Novel physics regimes")
    print("   - Sensor drift compensation")
    print("   - Domain adaptation without source data")


def main():
    """Run demonstration."""
    setup_environment()

    # Demonstrate TTA concept
    demonstrate_tta_concept()

    # Explain methods
    explain_tta_methods()

    print("\nDemo complete! See full implementation in models/test_time_adaptation/")


if __name__ == "__main__":
    main()
