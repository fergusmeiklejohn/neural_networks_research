"""Analyze why TTA degrades performance on OOD data."""

import pickle
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.imports import setup_project_paths

setup_project_paths()

import keras

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
    """Prepare trajectory data for training."""
    X, y = [], []
    for traj in trajectories:
        for i in range(len(traj) - input_steps - output_steps + 1):
            X.append(traj[i : i + input_steps])
            y.append(traj[i + input_steps : i + input_steps + output_steps])
    return np.array(X), np.array(y)


def analyze_tta_behavior(model, trajectory, tta_kwargs, name="TTA Analysis"):
    """Analyze how TTA behaves during adaptation."""
    print(f"\n{name}")
    print("=" * 60)

    # Prepare data
    X = trajectory[0:1].reshape(1, 1, 8)
    y_true = trajectory[1:11]

    # Baseline prediction
    y_baseline = model.predict(X, verbose=0)[0]
    baseline_mse = np.mean((y_true - y_baseline) ** 2)

    # Create TTA wrapper
    tta_model = TTAWrapper(model, **tta_kwargs)

    # Track predictions over adaptation steps
    adapter = tta_model.tta_adapter
    predictions_per_step = []
    losses_per_step = []

    # Get predictions at each adaptation step
    all_preds = adapter.adapt(X, return_all_steps=True)

    # Calculate MSE for each step
    mses = []
    for i, pred in enumerate(all_preds):
        if pred.ndim == 3:
            pred = pred[0]
        mse = np.mean((y_true - pred) ** 2)
        mses.append(mse)
        predictions_per_step.append(pred)

    # Get adaptation losses
    if (
        hasattr(adapter, "adaptation_metrics")
        and adapter.adaptation_metrics["adaptation_loss"]
    ):
        losses_per_step = adapter.adaptation_metrics["adaptation_loss"][-1]

    # Reset for next test
    tta_model.reset()

    # Print results
    print(f"Baseline MSE: {baseline_mse:.2f}")
    print(f"MSE per adaptation step: {[f'{m:.2f}' for m in mses]}")
    if losses_per_step:
        print(f"Adaptation loss per step: {[f'{l:.4f}' for l in losses_per_step]}")

    # Analyze prediction changes
    pred_changes = []
    for i in range(1, len(predictions_per_step)):
        change = np.mean(np.abs(predictions_per_step[i] - predictions_per_step[i - 1]))
        pred_changes.append(change)

    if pred_changes:
        print(f"Mean prediction change per step: {[f'{c:.4f}' for c in pred_changes]}")

    # Check if MSE is getting worse
    if mses[-1] > baseline_mse:
        print(f"⚠️  MSE increased by {((mses[-1]/baseline_mse - 1) * 100):.1f}%")
    else:
        print(f"✓ MSE decreased by {((1 - mses[-1]/baseline_mse) * 100):.1f}%")

    return {
        "baseline_mse": baseline_mse,
        "mses": mses,
        "losses": losses_per_step,
        "predictions": predictions_per_step,
        "pred_changes": pred_changes,
    }


def visualize_adaptation_trajectory(results, output_dir, name="adaptation"):
    """Visualize how predictions change during adaptation."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. MSE over adaptation steps
    ax = axes[0, 0]
    steps = list(range(len(results["mses"])))
    ax.plot(steps, results["mses"], "o-", label="Adapted MSE")
    ax.axhline(results["baseline_mse"], color="r", linestyle="--", label="Baseline MSE")
    ax.set_xlabel("Adaptation Step")
    ax.set_ylabel("MSE")
    ax.set_title("MSE During Adaptation")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Adaptation loss over steps
    if results["losses"]:
        ax = axes[0, 1]
        ax.plot(results["losses"], "o-")
        ax.set_xlabel("Adaptation Step")
        ax.set_ylabel("Adaptation Loss")
        ax.set_title("Self-Supervised Loss During Adaptation")
        ax.grid(True, alpha=0.3)

    # 3. Prediction changes
    if results["pred_changes"]:
        ax = axes[1, 0]
        ax.plot(results["pred_changes"], "o-")
        ax.set_xlabel("Step Transition")
        ax.set_ylabel("Mean Absolute Change")
        ax.set_title("Prediction Changes Between Steps")
        ax.grid(True, alpha=0.3)

    # 4. First timestep predictions
    ax = axes[1, 1]
    # Plot x and y coordinates for first timestep
    for i, pred in enumerate(results["predictions"]):
        if i % 2 == 0:  # Plot every other step for clarity
            ax.plot(pred[0, 0], pred[0, 1], "o", label=f"Step {i}")
    ax.set_xlabel("X position")
    ax.set_ylabel("Y position")
    ax.set_title("First Timestep Position Predictions")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f"{name}_trajectory.png", dpi=150)
    plt.close()


def main():
    """Analyze TTA degradation."""
    print("Analyzing TTA Performance Degradation")
    print("=" * 70)

    # Setup
    config = setup_environment()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create output directory
    output_dir = get_output_path() / "tta_analysis"
    output_dir.mkdir(exist_ok=True)

    # Load data
    data_dir = get_data_path() / "true_ood_physics"

    print("Loading data...")
    const_files = sorted(data_dir.glob("constant_gravity_*.pkl"))
    with open(const_files[-1], "rb") as f:
        const_data = pickle.load(f)

    varying_files = sorted(data_dir.glob("time_varying_gravity_*.pkl"))
    with open(varying_files[-1], "rb") as f:
        ood_data = pickle.load(f)

    # Train model properly
    print("\nTraining model on constant gravity...")
    model = create_physics_model()
    X_train, y_train = prepare_data(const_data["trajectories"][:100])
    history = model.fit(
        X_train, y_train, epochs=30, batch_size=32, verbose=1, validation_split=0.2
    )

    # Test different TTA configurations
    configs = [
        {
            "name": "Minimal TTA (1 step, lr=1e-6)",
            "kwargs": {
                "tta_method": "regression_v2",
                "adaptation_steps": 1,
                "learning_rate": 1e-6,
                "consistency_loss_weight": 0.0,
                "smoothness_loss_weight": 0.0,
            },
        },
        {
            "name": "Standard TTA (5 steps, lr=1e-5)",
            "kwargs": {
                "tta_method": "regression_v2",
                "adaptation_steps": 5,
                "learning_rate": 1e-5,
                "consistency_loss_weight": 0.1,
                "smoothness_loss_weight": 0.05,
            },
        },
        {
            "name": "Many steps TTA (20 steps, lr=1e-6)",
            "kwargs": {
                "tta_method": "regression_v2",
                "adaptation_steps": 20,
                "learning_rate": 1e-6,
                "consistency_loss_weight": 0.1,
                "smoothness_loss_weight": 0.05,
            },
        },
        {
            "name": "Physics TTA (10 steps, lr=1e-5)",
            "kwargs": {
                "tta_method": "physics_regression_v2",
                "adaptation_steps": 10,
                "learning_rate": 1e-5,
                "consistency_loss_weight": 0.05,
                "smoothness_loss_weight": 0.05,
                "energy_loss_weight": 0.1,
                "momentum_loss_weight": 0.1,
            },
        },
    ]

    # Test on different scenarios
    scenarios = [
        (
            "In-distribution",
            const_data["trajectories"][50],
        ),  # Use a trajectory not in training
        ("Time-varying gravity", ood_data["trajectories"][0]),
    ]

    all_results = {}

    for scenario_name, trajectory in scenarios:
        print(f"\n\n{'='*70}")
        print(f"Testing on: {scenario_name}")
        print("=" * 70)

        scenario_results = {}

        for config in configs:
            results = analyze_tta_behavior(
                model, trajectory, config["kwargs"], config["name"]
            )
            scenario_results[config["name"]] = results

            # Visualize the most interesting cases
            if config["name"] == "Standard TTA (5 steps, lr=1e-5)":
                visualize_adaptation_trajectory(
                    results,
                    output_dir,
                    f"{scenario_name.lower().replace(' ', '_')}_{timestamp}",
                )

        all_results[scenario_name] = scenario_results

    # Summary analysis
    print("\n\n" + "=" * 70)
    print("SUMMARY ANALYSIS")
    print("=" * 70)

    for scenario_name, scenario_results in all_results.items():
        print(f"\n{scenario_name}:")

        for config_name, results in scenario_results.items():
            baseline = results["baseline_mse"]
            final = results["mses"][-1]
            change = (final / baseline - 1) * 100

            print(f"  {config_name}: {change:+.1f}%")

    # Key insights
    print("\n\nKEY INSIGHTS:")
    print("-" * 50)

    # Check if adaptation helps on in-distribution
    id_results = all_results["In-distribution"]
    id_improvements = []
    for config_name, results in id_results.items():
        improvement = (1 - results["mses"][-1] / results["baseline_mse"]) * 100
        id_improvements.append(improvement)

    if max(id_improvements) > 0:
        print("✓ TTA can improve on in-distribution data")
    else:
        print("✗ TTA degrades even on in-distribution data")

    # Check OOD performance
    ood_results = all_results["Time-varying gravity"]
    ood_improvements = []
    for config_name, results in ood_results.items():
        improvement = (1 - results["mses"][-1] / results["baseline_mse"]) * 100
        ood_improvements.append(improvement)

    print(f"\nBest OOD improvement: {max(ood_improvements):.1f}%")

    # Analyze adaptation behavior
    print("\nAdaptation behavior:")
    for config_name in ["Standard TTA (5 steps, lr=1e-5)"]:
        if config_name in ood_results:
            results = ood_results[config_name]
            # Check if MSE increases monotonically
            mses = results["mses"]
            if all(mses[i] >= mses[i - 1] for i in range(1, len(mses))):
                print("- MSE increases monotonically during adaptation")
            else:
                print("- MSE fluctuates during adaptation")

            # Check prediction stability
            if results["pred_changes"]:
                avg_change = np.mean(results["pred_changes"])
                print(f"- Average prediction change per step: {avg_change:.4f}")


if __name__ == "__main__":
    main()
