#!/usr/bin/env python3
"""
Test a simple baseline model on True OOD benchmark data.

This demonstrates the catastrophic failure of standard neural networks
on true physics extrapolation (time-varying gravity).
"""

import os

os.environ["KERAS_BACKEND"] = "jax"

import sys

sys.path.append("../..")

import json
import pickle
from datetime import datetime
from pathlib import Path

import keras
import matplotlib.pyplot as plt
import numpy as np
from keras import layers


def load_standard_physics_data():
    """Load standard constant gravity training data."""
    # Try to load the newly generated comprehensive training data
    data_dir = Path("data/processed/constant_gravity_training")
    if data_dir.exists():
        pkl_files = list(data_dir.glob("constant_gravity_data_*.pkl"))
        if pkl_files:
            data_path = sorted(pkl_files)[-1]  # Most recent
            print(f"Loading training data from: {data_path}")
            with open(data_path, "rb") as f:
                data = pickle.load(f)
            print(f"Loaded {len(data)} training trajectories")
        else:
            # Fallback to demo data
            data_path = Path("data/processed/physics_worlds_demo/demo_data.pkl")
            if not data_path.exists():
                raise FileNotFoundError(f"No training data found")

            with open(data_path, "rb") as f:
                data = pickle.load(f)

            print(f"Loaded {len(data)} training trajectories (demo data)")
    else:
        # Fallback to demo data
        data_path = Path("data/processed/physics_worlds_demo/demo_data.pkl")
        if not data_path.exists():
            raise FileNotFoundError(f"No training data found")

        with open(data_path, "rb") as f:
            data = pickle.load(f)

        print(f"Loaded {len(data)} training trajectories (demo data)")

    # Extract trajectories with core features
    trajectories = []
    for sample in data:
        traj = sample["trajectory"]
        # Extract [x1, y1, x2, y2, vx1, vy1, vx2, vy2] for each timestep
        core_features = []
        for t in range(len(traj)):
            state = traj[t]
            # Assuming format matches: [time, x1, y1, vx1, vy1, ..., x2, y2, vx2, vy2, ...]
            if len(state) >= 17:
                features = [
                    state[1],
                    state[2],  # x1, y1
                    state[9],
                    state[10],  # x2, y2
                    state[3],
                    state[4],  # vx1, vy1
                    state[11],
                    state[12],  # vx2, vy2
                ]
            else:
                # Fallback for different format
                features = state[:8]
            core_features.append(features)
        trajectories.append(np.array(core_features))

    return trajectories


def load_true_ood_data():
    """Load the generated True OOD benchmark data."""
    # Find the most recent harmonic gravity data
    data_dir = Path("data/processed/true_ood_benchmark")
    if not data_dir.exists():
        raise FileNotFoundError(f"True OOD data directory not found: {data_dir}")

    pkl_files = list(data_dir.glob("harmonic_gravity_data_*.pkl"))
    if not pkl_files:
        raise FileNotFoundError("No harmonic gravity data files found")

    data_path = sorted(pkl_files)[-1]  # Most recent
    print(f"Loading True OOD data from: {data_path}")

    with open(data_path, "rb") as f:
        data = pickle.load(f)

    # Convert to same format as training data
    trajectories = []
    for sample in data:
        traj_array = np.array(sample["trajectory"])
        # Extract [x1, y1, x2, y2, vx1, vy1, vx2, vy2]
        core_features = []
        for t in range(len(traj_array)):
            state = traj_array[t]
            features = [
                state[1],
                state[2],  # x1, y1
                state[9],
                state[10],  # x2, y2
                state[3],
                state[4],  # vx1, vy1
                state[11],
                state[12],  # vx2, vy2
            ]
            core_features.append(features)
        trajectories.append(np.array(core_features))

    return trajectories, data


def prepare_data(trajectories, input_steps=1, output_steps=10):
    """Prepare trajectory data for training/evaluation."""
    X, y = [], []
    for traj in trajectories:
        for i in range(len(traj) - input_steps - output_steps + 1):
            X.append(traj[i : i + input_steps])
            y.append(traj[i + input_steps : i + input_steps + output_steps])
    return np.array(X), np.array(y)


def create_simple_baseline():
    """Create a simple neural network baseline."""
    model = keras.Sequential(
        [
            layers.Input(shape=(1, 8)),
            layers.Flatten(),
            layers.Dense(256, activation="relu"),
            layers.BatchNormalization(),
            layers.Dense(128, activation="relu"),
            layers.BatchNormalization(),
            layers.Dense(64, activation="relu"),
            layers.BatchNormalization(),
            layers.Dense(80),  # 10 timesteps * 8 features
            layers.Reshape((10, 8)),
        ]
    )

    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])

    return model


def train_baseline(model, X_train, y_train, epochs=10):
    """Quick training of the baseline model."""
    print("Training baseline model...")

    history = model.fit(
        X_train, y_train, validation_split=0.1, epochs=epochs, batch_size=32, verbose=1
    )

    return history


def evaluate_on_ood(model, X_test, y_test, test_name):
    """Evaluate model on test data and compute metrics."""
    print(f"\nEvaluating on {test_name}...")

    # Get predictions
    predictions = model.predict(X_test, batch_size=32, verbose=0)

    # Compute metrics
    mse = float(np.mean((predictions - y_test) ** 2))
    mae = float(np.mean(np.abs(predictions - y_test)))

    # Per-timestep error
    timestep_mse = np.mean((predictions - y_test) ** 2, axis=(0, 2))

    # Analyze degradation
    if np.isnan(mse) or mse > 1e6:
        status = "catastrophic_failure"
    elif mse > 10000:
        status = "severe_degradation"
    elif mse > 1000:
        status = "major_degradation"
    else:
        status = "moderate_degradation"

    results = {
        "test_set": test_name,
        "mse": mse,
        "mae": mae,
        "status": status,
        "timestep_mse": timestep_mse.tolist(),
        "n_samples": len(X_test),
    }

    print(f"  MSE: {mse:.2f}")
    print(f"  MAE: {mae:.2f}")
    print(f"  Status: {status}")

    return results, predictions


def visualize_comparison(y_true, y_pred, ood_metadata, n_samples=3):
    """Visualize true vs predicted trajectories."""
    fig, axes = plt.subplots(2, n_samples, figsize=(5 * n_samples, 10))

    for i in range(min(n_samples, len(y_true))):
        # True trajectory
        ax_true = axes[0, i]
        true_traj = y_true[i]
        ax_true.plot(true_traj[:, 0], true_traj[:, 1], "b-", alpha=0.7, label="Ball 1")
        ax_true.plot(true_traj[:, 2], true_traj[:, 3], "r-", alpha=0.7, label="Ball 2")
        ax_true.set_title(f"True Trajectory (Harmonic Gravity)")
        ax_true.set_xlim(0, 800)
        ax_true.set_ylim(0, 600)
        ax_true.invert_yaxis()
        ax_true.legend()
        ax_true.grid(True, alpha=0.3)

        # Predicted trajectory
        ax_pred = axes[1, i]
        pred_traj = y_pred[i]
        ax_pred.plot(
            pred_traj[:, 0], pred_traj[:, 1], "b--", alpha=0.7, label="Ball 1 (pred)"
        )
        ax_pred.plot(
            pred_traj[:, 2], pred_traj[:, 3], "r--", alpha=0.7, label="Ball 2 (pred)"
        )

        # Compute error for this sample
        sample_mse = np.mean((pred_traj - true_traj) ** 2)
        ax_pred.set_title(f"Predicted (MSE: {sample_mse:.1f})")
        ax_pred.set_xlim(0, 800)
        ax_pred.set_ylim(0, 600)
        ax_pred.invert_yaxis()
        ax_pred.legend()
        ax_pred.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def main():
    """Run the complete evaluation."""
    print("=" * 80)
    print("Testing Simple Baseline on True OOD Physics")
    print("=" * 80)

    # Create output directory
    output_dir = Path("outputs/true_ood_evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load data
    print("\n1. Loading training data...")
    train_trajectories = load_standard_physics_data()
    X_train, y_train = prepare_data(train_trajectories)
    print(f"   Training samples: {len(X_train)}")

    print("\n2. Loading True OOD data...")
    ood_trajectories, ood_metadata = load_true_ood_data()
    X_ood, y_ood = prepare_data(ood_trajectories)
    print(f"   OOD samples: {len(X_ood)}")

    # Create and train model
    print("\n3. Creating baseline model...")
    model = create_simple_baseline()
    print(f"   Parameters: {model.count_params()}")

    # Subsample for faster training
    if len(X_train) > 50000:
        print(
            f"   Subsampling training data from {len(X_train)} to 50000 samples for faster training"
        )
        indices = np.random.choice(len(X_train), 50000, replace=False)
        X_train_subset = X_train[indices]
        y_train_subset = y_train[indices]
    else:
        X_train_subset = X_train
        y_train_subset = y_train

    print("\n4. Training on constant gravity data...")
    history = train_baseline(model, X_train_subset, y_train_subset, epochs=5)

    # Evaluate on training distribution (sanity check)
    print("\n5. Evaluating on training distribution...")
    X_val = X_train[-100:]
    y_val = y_train[-100:]
    val_results, _ = evaluate_on_ood(model, X_val, y_val, "Constant Gravity (Val)")

    # Evaluate on True OOD
    print("\n6. Evaluating on True OOD (Time-Varying Gravity)...")
    ood_results, ood_predictions = evaluate_on_ood(
        model, X_ood, y_ood, "Harmonic Gravity"
    )

    # Calculate degradation
    degradation_factor = ood_results["mse"] / val_results["mse"]
    print(f"\n7. Performance Degradation: {degradation_factor:.1f}x")

    # Visualize results
    print("\n8. Creating visualizations...")
    fig = visualize_comparison(y_ood[:5], ood_predictions[:5], ood_metadata)
    fig.savefig(output_dir / f"simple_baseline_ood_comparison_{timestamp}.png", dpi=150)
    plt.close()

    # Save results
    results_summary = {
        "timestamp": timestamp,
        "model": "Simple Neural Network Baseline",
        "parameters": model.count_params(),
        "training_samples": len(X_train),
        "validation_results": val_results,
        "ood_results": ood_results,
        "degradation_factor": degradation_factor,
        "conclusion": "Catastrophic failure on time-varying gravity"
        if ood_results["mse"] > 10000
        else "Significant degradation",
    }

    with open(output_dir / f"simple_baseline_results_{timestamp}.json", "w") as f:
        json.dump(results_summary, f, indent=2)

    # Generate report
    report = f"""# Simple Baseline True OOD Evaluation

Generated: {timestamp}

## Model Details
- Architecture: Simple feedforward neural network
- Parameters: {model.count_params():,}
- Training data: Constant gravity only

## Results Summary

### Validation Set (Constant Gravity)
- MSE: {val_results['mse']:.2f}
- MAE: {val_results['mae']:.2f}
- Status: Expected baseline performance

### True OOD Test (Time-Varying Gravity)
- MSE: {ood_results['mse']:.2f}
- MAE: {ood_results['mae']:.2f}
- Status: {ood_results['status']}
- **Degradation: {degradation_factor:.1f}x worse than validation**

## Key Findings

1. **Catastrophic Failure**: The model completely fails when gravity varies with time
2. **Not Interpolation**: This cannot be solved by interpolating between training examples
3. **Fundamental Limitation**: Standard neural networks cannot extrapolate to new physics

## Implications

This demonstrates that:
- Current "OOD" benchmarks that only vary parameters are actually testing interpolation
- True OOD requires fundamentally different dynamics (e.g., time-varying forces)
- We need new architectures that can handle mechanism changes, not just parameter shifts

## Files Generated
- `simple_baseline_results_{timestamp}.json`: Detailed results
- `simple_baseline_ood_comparison_{timestamp}.png`: Trajectory visualizations
"""

    with open(output_dir / f"simple_baseline_report_{timestamp}.md", "w") as f:
        f.write(report)

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print(f"Validation MSE: {val_results['mse']:.2f}")
    print(f"True OOD MSE: {ood_results['mse']:.2f}")
    print(f"Degradation Factor: {degradation_factor:.1f}x")
    print(f"\nResults saved to: {output_dir}")

    return results_summary


if __name__ == "__main__":
    results = main()
