"""Simple TTA evaluation test with minimal dependencies."""

import pickle
import time
from pathlib import Path

import numpy as np

# Skip matplotlib if not available
try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.imports import setup_project_paths

setup_project_paths()

import keras

from models.test_time_adaptation.tta_wrappers import TTAWrapper
from utils.config import setup_environment
from utils.paths import get_data_path, get_output_path


def create_simple_physics_model():
    """Create a simple model for testing."""
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(1, 8)),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(8 * 49),  # Predict next 49 timesteps
            keras.layers.Reshape((49, 8)),
        ]
    )

    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def evaluate_on_trajectories(model, trajectories, with_tta=False, tta_method="tent"):
    """Evaluate model on trajectory prediction."""
    errors = []
    times = []

    # Create TTA wrapper if needed
    if with_tta:
        model = TTAWrapper(
            model,
            tta_method=tta_method,
            adaptation_steps=5,
            learning_rate=1e-4,
            physics_loss_weight=0.1 if tta_method == "physics_tent" else None,
        )

    for traj in trajectories[:20]:  # Limit to 20 for speed
        start_time = time.time()

        # Use first timestep to predict rest
        X = traj[0:1].reshape(1, 1, 8)
        y_true = traj[1:]

        # Predict
        if with_tta:
            y_pred = model.predict(X, adapt=True)
            model.reset()  # Reset after each trajectory
        else:
            y_pred = model.predict(X, verbose=0)

        elapsed = time.time() - start_time
        times.append(elapsed)

        # Compute error
        if len(y_pred.shape) == 3:
            y_pred = y_pred[0]

        min_len = min(len(y_true), len(y_pred))
        mse = np.mean((y_true[:min_len] - y_pred[:min_len]) ** 2)
        errors.append(mse)

    return {
        "mse": np.mean(errors),
        "mse_std": np.std(errors),
        "time": np.mean(times),
        "n_samples": len(errors),
    }


def main():
    """Run simple TTA evaluation."""
    print("Simple TTA Evaluation on True OOD Physics")
    print("=" * 60)

    # Load data
    data_dir = get_data_path() / "true_ood_physics"

    # Load constant gravity
    const_files = sorted(data_dir.glob("constant_gravity_*.pkl"))
    with open(const_files[-1], "rb") as f:
        const_data = pickle.load(f)

    # Load time-varying gravity
    varying_files = sorted(data_dir.glob("time_varying_gravity_*.pkl"))
    with open(varying_files[-1], "rb") as f:
        varying_data = pickle.load(f)

    print(f"\nLoaded data:")
    print(f"  Constant gravity: {const_data['trajectories'].shape}")
    print(f"  Time-varying gravity: {varying_data['trajectories'].shape}")

    # Create and train simple model
    print("\n1. Creating and training simple model...")
    model = create_simple_physics_model()

    # Quick training on constant gravity
    X_train = const_data["trajectories"][:80, 0:1]
    y_train = const_data["trajectories"][:80, 1:]

    print("   Training on constant gravity...")
    history = model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=0)
    print(f"   Final training loss: {history.history['loss'][-1]:.4f}")

    # Evaluate on constant gravity (baseline)
    print("\n2. Evaluating on constant gravity (in-distribution):")
    const_results = evaluate_on_trajectories(model, const_data["trajectories"][80:])
    print(f"   MSE: {const_results['mse']:.4f} ± {const_results['mse_std']:.4f}")

    # Evaluate on time-varying gravity without TTA
    print("\n3. Evaluating on time-varying gravity (no TTA):")
    varying_results = evaluate_on_trajectories(model, varying_data["trajectories"])
    print(f"   MSE: {varying_results['mse']:.4f} ± {varying_results['mse_std']:.4f}")
    degradation = (
        (varying_results["mse"] - const_results["mse"]) / const_results["mse"] * 100
    )
    print(f"   Degradation: {degradation:.1f}%")

    # Evaluate with different TTA methods
    tta_methods = ["tent", "physics_tent"]
    results = {"none": varying_results}

    for method in tta_methods:
        print(f"\n4. Evaluating with {method.upper()}:")
        tta_results = evaluate_on_trajectories(
            model, varying_data["trajectories"], with_tta=True, tta_method=method
        )
        results[method] = tta_results

        print(f"   MSE: {tta_results['mse']:.4f} ± {tta_results['mse_std']:.4f}")
        improvement = (
            (varying_results["mse"] - tta_results["mse"]) / varying_results["mse"] * 100
        )
        print(f"   Improvement: {improvement:.1f}%")
        print(f"   Time per sample: {tta_results['time']:.3f}s")

    # Visualize results
    if HAS_MATPLOTLIB:
        print("\n5. Creating visualization...")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # MSE comparison
        methods = list(results.keys())
        mses = [results[m]["mse"] for m in methods]
        stds = [results[m]["mse_std"] for m in methods]

        ax1.bar(methods, mses, yerr=stds, capsize=5, alpha=0.7)
        ax1.axhline(
            y=const_results["mse"],
            color="green",
            linestyle="--",
            label="Constant gravity baseline",
        )
        ax1.set_ylabel("Mean Squared Error")
        ax1.set_title("TTA Performance on Time-Varying Gravity")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Time comparison
        times = [results[m]["time"] for m in methods]
        ax2.bar(methods, times, alpha=0.7, color="orange")
        ax2.set_ylabel("Time per sample (seconds)")
        ax2.set_title("Computational Cost")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save results
        output_dir = get_output_path() / "tta_evaluation"
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / "simple_tta_results.png", dpi=150, bbox_inches="tight")
        plt.close()
    else:
        print("\n5. Skipping visualization (matplotlib not available)")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Constant gravity MSE: {const_results['mse']:.4f}")
    print(
        f"Time-varying gravity (no TTA): {varying_results['mse']:.4f} ({degradation:.1f}% worse)"
    )

    for method in tta_methods:
        improvement = (
            (varying_results["mse"] - results[method]["mse"])
            / varying_results["mse"]
            * 100
        )
        print(
            f"{method.upper()}: {results[method]['mse']:.4f} ({improvement:.1f}% improvement)"
        )

    best_method = min(results.items(), key=lambda x: x[1]["mse"])[0]
    print(f"\nBest method: {best_method}")

    # Check if TTA helps with true OOD
    if any(results[m]["mse"] < varying_results["mse"] for m in tta_methods):
        print("✓ TTA successfully improves performance on true OOD scenarios!")
    else:
        print("✗ TTA did not improve performance in this test")


if __name__ == "__main__":
    config = setup_environment()
    main()
