"""Quick hyperparameter tuning for JAX gradient-based Test-Time Adaptation V2."""

import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

import keras
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.imports import setup_project_paths

setup_project_paths()

from models.test_time_adaptation.tta_wrappers import TTAWrapper
from utils.config import setup_environment
from utils.paths import get_data_path, get_output_path


def create_physics_model(input_steps=1, output_steps=10):
    """Create a physics prediction model."""
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(input_steps, 8)),
            keras.layers.Flatten(),
            keras.layers.Dense(256, activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(128, activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(8 * output_steps),
            keras.layers.Reshape((output_steps, 8)),
        ]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse", metrics=["mae"]
    )
    return model


def prepare_data(trajectories, input_steps=1, output_steps=10):
    """Prepare trajectory data for training/evaluation."""
    X, y = [], []
    for traj in trajectories:
        for i in range(len(traj) - input_steps - output_steps + 1):
            X.append(traj[i : i + input_steps])
            y.append(traj[i + input_steps : i + input_steps + output_steps])
    return np.array(X), np.array(y)


def evaluate_tta_config(
    model, test_trajectories, tta_method="regression_v2", **tta_kwargs
):
    """Evaluate a specific TTA configuration."""
    # Create TTA wrapper
    tta_model = TTAWrapper(model, tta_method=tta_method, **tta_kwargs)

    # Evaluate on test trajectories (only 10 for speed)
    errors = []
    for i, traj in enumerate(test_trajectories[:10]):
        X = traj[0:1].reshape(1, 1, 8)
        y_true = traj[1:11]

        # Adapt and predict
        y_pred = tta_model.predict(X, adapt=True)
        if len(y_pred.shape) == 3:
            y_pred = y_pred[0]

        # Compute error
        min_len = min(len(y_true), len(y_pred))
        mse = np.mean((y_true[:min_len] - y_pred[:min_len]) ** 2)
        errors.append(mse)

        # Reset model for next trajectory
        tta_model.reset()

    return {"mean_mse": np.mean(errors), "std_mse": np.std(errors), "errors": errors}


def run_quick_search():
    """Run quick hyperparameter search for TTA V2."""
    print("Quick TTA V2 Hyperparameter Tuning")
    print("=" * 50)

    # Setup
    config = setup_environment()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load data
    data_dir = get_data_path() / "true_ood_physics"

    # Load constant gravity (training)
    print("Loading data...")
    const_files = sorted(data_dir.glob("constant_gravity_*.pkl"))
    if not const_files:
        print("No constant gravity data found. Please generate data first.")
        return

    with open(const_files[-1], "rb") as f:
        const_data = pickle.load(f)

    # Load time-varying gravity (OOD test)
    varying_files = sorted(data_dir.glob("time_varying_gravity_*.pkl"))
    if not varying_files:
        print("No time-varying gravity data found. Please generate data first.")
        return

    with open(varying_files[-1], "rb") as f:
        ood_data = pickle.load(f)

    print(f"Loaded {len(const_data['trajectories'])} constant gravity trajectories")
    print(f"Loaded {len(ood_data['trajectories'])} time-varying gravity trajectories")

    # Train base model
    print("\nTraining base model...")
    model = create_physics_model()

    X_train, y_train = prepare_data(const_data["trajectories"][:80])
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

    # Baseline evaluation (no TTA)
    print("\nBaseline evaluation (no TTA)...")
    X_test_ood, y_test_ood = prepare_data(ood_data["trajectories"][:10])
    baseline_ood = model.evaluate(X_test_ood, y_test_ood, verbose=0)
    print(f"Time-varying gravity MSE (no TTA): {baseline_ood[0]:.4f}")

    # Test a few key configurations
    configs = [
        # RegressionTTAV2 - BN only with different learning rates
        {
            "name": "RegressionV2-BN-lr1e-5",
            "tta_method": "regression_v2",
            "learning_rate": 1e-5,
            "adaptation_steps": 10,
            "consistency_loss_weight": 0.1,
            "smoothness_loss_weight": 0.05,
            "bn_only_mode": True,
        },
        {
            "name": "RegressionV2-BN-lr1e-6",
            "tta_method": "regression_v2",
            "learning_rate": 1e-6,
            "adaptation_steps": 10,
            "consistency_loss_weight": 0.1,
            "smoothness_loss_weight": 0.05,
            "bn_only_mode": True,
        },
        # RegressionTTAV2 - All params
        {
            "name": "RegressionV2-All-lr1e-5",
            "tta_method": "regression_v2",
            "learning_rate": 1e-5,
            "adaptation_steps": 10,
            "consistency_loss_weight": 0.1,
            "smoothness_loss_weight": 0.05,
            "bn_only_mode": False,
        },
        {
            "name": "RegressionV2-All-lr1e-6",
            "tta_method": "regression_v2",
            "learning_rate": 1e-6,
            "adaptation_steps": 20,
            "consistency_loss_weight": 0.1,
            "smoothness_loss_weight": 0.05,
            "bn_only_mode": False,
        },
        # PhysicsRegressionTTAV2
        {
            "name": "PhysicsV2-BN-lr1e-5",
            "tta_method": "physics_regression_v2",
            "learning_rate": 1e-5,
            "adaptation_steps": 10,
            "consistency_loss_weight": 0.1,
            "smoothness_loss_weight": 0.05,
            "energy_loss_weight": 0.1,
            "momentum_loss_weight": 0.1,
            "bn_only_mode": True,
        },
        {
            "name": "PhysicsV2-All-lr1e-6",
            "tta_method": "physics_regression_v2",
            "learning_rate": 1e-6,
            "adaptation_steps": 20,
            "consistency_loss_weight": 0.1,
            "smoothness_loss_weight": 0.05,
            "energy_loss_weight": 0.1,
            "momentum_loss_weight": 0.1,
            "bn_only_mode": False,
        },
    ]

    # Test each configuration
    results = []
    best_config = None
    best_mse = float("inf")

    print("\nTesting configurations...")
    for config in configs:
        print(f"\n{config['name']}:")

        # Extract config params
        name = config.pop("name")
        result = evaluate_tta_config(model, ood_data["trajectories"], **config)

        improvement = (1 - result["mean_mse"] / baseline_ood[0]) * 100
        print(f"  MSE: {result['mean_mse']:.4f} ± {result['std_mse']:.4f}")
        print(f"  Improvement: {improvement:.1f}%")

        result_dict = {
            "name": name,
            **config,
            "mean_mse": result["mean_mse"],
            "std_mse": result["std_mse"],
            "improvement": improvement,
        }
        results.append(result_dict)

        if result["mean_mse"] < best_mse:
            best_mse = result["mean_mse"]
            best_config = result_dict

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"\nBaseline OOD MSE: {baseline_ood[0]:.4f}")
    print(f"\nBest configuration: {best_config['name']}")
    print(f"  MSE: {best_config['mean_mse']:.4f}")
    print(f"  Improvement: {best_config['improvement']:.1f}%")

    # Sort by improvement
    results.sort(key=lambda x: x["mean_mse"])
    print("\nAll results (sorted by MSE):")
    for r in results:
        print(
            f"  {r['name']}: MSE={r['mean_mse']:.4f}, Improvement={r['improvement']:.1f}%"
        )

    # Save results
    output_dir = get_output_path() / "tta_tuning"
    output_dir.mkdir(exist_ok=True)

    results_file = output_dir / f"quick_tta_v2_results_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "baseline_ood_mse": baseline_ood[0],
                "results": results,
                "best_config": best_config,
            },
            f,
            indent=2,
        )

    print(f"\nResults saved to: {results_file}")
    print("\n✓ Quick V2 tuning complete!")


def main():
    """Run quick hyperparameter tuning."""
    run_quick_search()


if __name__ == "__main__":
    main()
