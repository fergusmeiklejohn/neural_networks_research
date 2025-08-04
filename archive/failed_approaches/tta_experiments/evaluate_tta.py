"""Evaluate Test-Time Adaptation methods on physics experiments."""

import json
import os
import time
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from utils.imports import setup_project_paths

setup_project_paths()

from models.baseline_models import create_baseline_model
from models.test_time_adaptation.tta_wrappers import TTAWrapper
from utils.config import setup_environment
from utils.paths import get_data_path, get_output_path


def load_test_data(data_type: str = "time_varying") -> Dict:
    """Load test data for evaluation.

    Args:
        data_type: Type of test data ('constant', 'time_varying', 'true_ood')

    Returns:
        Dictionary with test data
    """
    data_path = get_data_path("physics_worlds", "processed", "physics_worlds")

    if data_type == "time_varying":
        # Load or generate time-varying gravity data
        # This would be implemented based on generate_true_ood_data.py
        print(f"Loading time-varying gravity test data...")
        # Placeholder - implement actual loading
        return {
            "trajectories": np.random.randn(
                100, 50, 8
            ),  # 100 trajectories, 50 timesteps, 8 features
            "gravity_fn": lambda t: -9.8 * (1 + 0.1 * np.sin(0.5 * t)),
            "metadata": {"type": "time_varying", "num_samples": 100},
        }
    else:
        # Load standard test data
        import pickle

        test_file = os.path.join(data_path, "test_data.pkl")
        with open(test_file, "rb") as f:
            return pickle.load(f)


def evaluate_model_with_tta(
    model_name: str,
    test_data: Dict,
    tta_methods: List[str] = ["none", "tent", "physics_tent", "ttt"],
    verbose: bool = True,
) -> Dict[str, Dict]:
    """Evaluate a model with different TTA methods.

    Args:
        model_name: Name of baseline model
        test_data: Test data dictionary
        tta_methods: List of TTA methods to evaluate
        verbose: Whether to print progress

    Returns:
        Results dictionary
    """
    results = {}

    # Load base model
    if verbose:
        print(f"\nEvaluating {model_name}...")

    try:
        # Create base model
        base_model = create_baseline_model(
            model_name,
            input_dim=8,
            output_dim=8,
            sequence_length=49,  # Predict 49 future steps from 1 input
        )

        # Load pretrained weights if available
        model_path = get_output_path("models", f"{model_name}_model.keras")
        if os.path.exists(model_path):
            base_model.model.load_weights(model_path)
            if verbose:
                print(f"Loaded pretrained weights from {model_path}")
    except Exception as e:
        print(f"Error creating {model_name}: {e}")
        return results

    # Evaluate each TTA method
    for tta_method in tta_methods:
        if verbose:
            print(f"  Testing {tta_method}...")

        try:
            if tta_method == "none":
                # No adaptation - baseline
                model = base_model
            else:
                # Create TTA wrapper
                tta_kwargs = {
                    "adaptation_steps": 5 if tta_method == "ttt" else 1,
                    "learning_rate": 1e-4,
                    "reset_after_batch": True,
                }
                model = TTAWrapper(base_model, tta_method, **tta_kwargs)

            # Evaluate
            start_time = time.time()
            metrics = evaluate_trajectory_prediction(
                model, test_data["trajectories"], adapt=(tta_method != "none")
            )
            metrics["time"] = time.time() - start_time

            # Add TTA-specific metrics
            if tta_method != "none":
                tta_metrics = model.get_metrics()
                metrics.update(tta_metrics)

            results[tta_method] = metrics

            if verbose:
                print(f"    MSE: {metrics['mse']:.4f}, Time: {metrics['time']:.2f}s")

        except Exception as e:
            print(f"    Error with {tta_method}: {e}")
            results[tta_method] = {"error": str(e)}

    return results


def evaluate_trajectory_prediction(
    model, trajectories: np.ndarray, adapt: bool = False
) -> Dict[str, float]:
    """Evaluate trajectory prediction performance.

    Args:
        model: Model to evaluate (with or without TTA)
        trajectories: Test trajectories (batch, time, features)
        adapt: Whether to use test-time adaptation

    Returns:
        Dictionary of metrics
    """
    mse_scores = []
    physics_consistency_scores = []

    for traj in trajectories:
        # Use first timestep as input
        input_state = traj[0:1]  # Shape: (1, features)
        target = traj[1:]  # Rest of trajectory

        # Predict
        if hasattr(model, "predict"):
            pred = model.predict(input_state[np.newaxis, ...], adapt=adapt)
        else:
            pred = model(input_state[np.newaxis, ...], training=False)

        # Ensure prediction shape matches target
        if pred.shape[1] > len(target):
            pred = pred[:, : len(target)]
        elif pred.shape[1] < len(target):
            target = target[: pred.shape[1]]

        # Compute MSE
        mse = np.mean((pred[0] - target) ** 2)
        mse_scores.append(mse)

        # Compute physics consistency
        physics_score = compute_physics_consistency(pred[0])
        physics_consistency_scores.append(physics_score)

    return {
        "mse": np.mean(mse_scores),
        "mse_std": np.std(mse_scores),
        "physics_consistency": np.mean(physics_consistency_scores),
        "physics_consistency_std": np.std(physics_consistency_scores),
    }


def compute_physics_consistency(trajectory: np.ndarray) -> float:
    """Compute physics consistency score for trajectory.

    Args:
        trajectory: Predicted trajectory (time, features)

    Returns:
        Consistency score (lower is better)
    """
    # Extract positions and velocities
    positions = trajectory[:, [0, 1, 4, 5]]
    velocities = trajectory[:, [2, 3, 6, 7]]

    # Check velocity consistency with position changes
    if len(trajectory) > 1:
        dt = 0.1  # Assumed timestep
        computed_velocities = np.diff(positions, axis=0) / dt
        velocity_error = np.mean(np.abs(computed_velocities - velocities[:-1]))
    else:
        velocity_error = 0.0

    # Check energy conservation (simplified)
    kinetic_energy = 0.5 * np.sum(velocities**2, axis=1)
    if len(kinetic_energy) > 1:
        energy_variation = np.std(kinetic_energy) / (np.mean(kinetic_energy) + 1e-8)
    else:
        energy_variation = 0.0

    return velocity_error + energy_variation


def plot_tta_comparison(results: Dict[str, Dict], save_path: Optional[str] = None):
    """Plot comparison of TTA methods.

    Args:
        results: Results dictionary from evaluation
        save_path: Path to save plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # MSE comparison
    models = list(results.keys())
    methods = list(next(iter(results.values())).keys())

    x = np.arange(len(methods))
    width = 0.8 / len(models)

    for i, model in enumerate(models):
        mse_values = [
            results[model].get(method, {}).get("mse", np.nan) for method in methods
        ]
        ax1.bar(x + i * width, mse_values, width, label=model)

    ax1.set_xlabel("TTA Method")
    ax1.set_ylabel("MSE")
    ax1.set_title("Trajectory Prediction Error")
    ax1.set_xticks(x + width * (len(models) - 1) / 2)
    ax1.set_xticklabels(methods)
    ax1.legend()
    ax1.set_yscale("log")

    # Time comparison
    for i, model in enumerate(models):
        time_values = [
            results[model].get(method, {}).get("time", np.nan) for method in methods
        ]
        ax2.bar(x + i * width, time_values, width, label=model)

    ax2.set_xlabel("TTA Method")
    ax2.set_ylabel("Time (seconds)")
    ax2.set_title("Inference Time")
    ax2.set_xticks(x + width * (len(models) - 1) / 2)
    ax2.set_xticklabels(methods)
    ax2.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def main():
    """Main evaluation script."""
    # Setup
    setup_environment()

    # Models to evaluate
    models = ["graph_extrap", "maml", "gflownet", "erm"]

    # TTA methods to test
    tta_methods = ["none", "tent", "physics_tent", "ttt"]

    # Load test data
    print("Loading test data...")
    test_data_types = {
        "constant": load_test_data("constant"),
        "time_varying": load_test_data("time_varying"),
    }

    # Evaluate each model
    all_results = {}

    for data_type, test_data in test_data_types.items():
        print(f"\n{'='*60}")
        print(f"Evaluating on {data_type} gravity data")
        print(f"{'='*60}")

        results = {}
        for model_name in models:
            model_results = evaluate_model_with_tta(
                model_name, test_data, tta_methods, verbose=True
            )
            results[model_name] = model_results

        all_results[data_type] = results

        # Save results
        output_dir = get_output_path("tta_results")
        os.makedirs(output_dir, exist_ok=True)

        results_file = os.path.join(output_dir, f"tta_results_{data_type}.json")
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        # Plot comparison
        plot_path = os.path.join(output_dir, f"tta_comparison_{data_type}.png")
        plot_tta_comparison(results, plot_path)

    # Summary report
    print("\n" + "=" * 60)
    print("SUMMARY REPORT")
    print("=" * 60)

    for data_type, results in all_results.items():
        print(f"\n{data_type.upper()} GRAVITY:")
        print("-" * 40)

        # Find best method for each model
        for model in models:
            if model not in results:
                continue

            best_method = None
            best_mse = float("inf")

            for method, metrics in results[model].items():
                if "mse" in metrics and metrics["mse"] < best_mse:
                    best_mse = metrics["mse"]
                    best_method = method

            baseline_mse = results[model].get("none", {}).get("mse", float("inf"))
            improvement = (
                (baseline_mse - best_mse) / baseline_mse * 100
                if baseline_mse > 0
                else 0
            )

            print(
                f"{model:12s}: Best method = {best_method:12s}, "
                f"MSE = {best_mse:.4f}, Improvement = {improvement:.1f}%"
            )

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
