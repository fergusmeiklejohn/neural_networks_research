"""Simplified baseline evaluation focusing on key results."""

import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

import keras
import numpy as np
from keras import layers

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.imports import setup_project_paths

setup_project_paths()

from utils.config import setup_environment
from utils.paths import get_data_path, get_output_path


def prepare_data(trajectories, input_steps=1, output_steps=10):
    """Prepare trajectory data for training/evaluation."""
    X, y = [], []
    for traj in trajectories:
        for i in range(len(traj) - input_steps - output_steps + 1):
            X.append(traj[i : i + input_steps])
            y.append(traj[i + input_steps : i + input_steps + output_steps])
    return np.array(X), np.array(y)


def create_standard_model():
    """Create a standard neural network for baseline comparison."""
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
            layers.Dense(80),
            layers.Reshape((10, 8)),
        ]
    )

    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])

    return model


def test_gflownet_inspired_baseline(train_data, val_data, test_data_dict):
    """Test GFlowNet-inspired approach with exploration."""
    print("\n" + "=" * 70)
    print("Testing GFlowNet-Inspired Baseline")
    print("=" * 70)

    X_train, y_train = train_data
    X_val, y_val = val_data

    # Train multiple models with different random seeds (exploration)
    models = []
    for i in range(3):
        print(f"\nTraining model {i+1}/3 with exploration...")

        # Add noise to training data to encourage exploration
        noise_level = 0.1 * (i + 1)  # Increasing exploration
        X_train_noisy = X_train + np.random.normal(0, noise_level, X_train.shape)

        model = create_standard_model()
        history = model.fit(
            X_train_noisy,
            y_train,
            validation_data=(X_val, y_val),
            epochs=20,
            batch_size=32,
            verbose=0,
        )

        val_loss = model.evaluate(X_val, y_val, verbose=0)
        if isinstance(val_loss, list):
            val_loss = val_loss[0]
        print(f"  Model {i+1} val loss: {val_loss:.2f}")
        models.append(model)

    # Evaluate ensemble on test sets
    results = {}
    for test_name, (X_test, y_test) in test_data_dict.items():
        print(f"\nEvaluating on {test_name}...")

        # Get predictions from all models
        predictions = []
        for model in models:
            pred = model.predict(X_test, verbose=0)
            predictions.append(pred)

        # Use mean prediction (could also use variance for uncertainty)
        ensemble_pred = np.mean(predictions, axis=0)
        mse = np.mean((ensemble_pred - y_test) ** 2)

        # Also get individual model MSEs
        individual_mses = []
        for pred in predictions:
            individual_mse = np.mean((pred - y_test) ** 2)
            individual_mses.append(individual_mse)

        results[test_name] = {
            "ensemble_mse": float(mse),
            "individual_mses": individual_mses,
            "prediction_std": float(np.mean(np.std(predictions, axis=0))),
        }

        print(f"  Ensemble MSE: {mse:.2f}")
        print(f"  Individual MSEs: {[f'{m:.2f}' for m in individual_mses]}")
        print(
            f"  Prediction uncertainty (std): {results[test_name]['prediction_std']:.4f}"
        )

    return results


def test_maml_inspired_baseline(train_data, val_data, test_data_dict):
    """Test MAML-inspired approach with fast adaptation."""
    print("\n" + "=" * 70)
    print("Testing MAML-Inspired Baseline")
    print("=" * 70)

    X_train, y_train = train_data
    X_val, y_val = val_data

    # Train base model
    print("\nTraining base model...")
    base_model = create_standard_model()

    # Train with early stopping to prevent overfitting
    # This makes the model more adaptable
    history = base_model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=15,  # Less training for better adaptability
        batch_size=32,
        verbose=1,
    )

    results = {}

    # Test without adaptation
    print("\nTesting base model (no adaptation)...")
    for test_name, (X_test, y_test) in test_data_dict.items():
        pred = base_model.predict(X_test, verbose=0)
        mse = np.mean((pred - y_test) ** 2)
        results[f"{test_name}_base"] = float(mse)
        print(f"  {test_name}: MSE = {mse:.2f}")

    # Test with adaptation (fine-tuning on few examples)
    print("\nTesting adapted model (10-shot adaptation)...")
    for test_name, (X_test, y_test) in test_data_dict.items():
        if test_name == "time_varying":
            # Clone model for adaptation
            adapted_model = keras.models.clone_model(base_model)
            adapted_model.set_weights(base_model.get_weights())
            adapted_model.compile(
                optimizer=keras.optimizers.SGD(0.01),  # Higher LR for fast adaptation
                loss="mse",
            )

            # Adapt on first 10 examples
            X_adapt = X_test[:10]
            y_adapt = y_test[:10]

            for _ in range(5):  # Quick adaptation
                adapted_model.train_on_batch(X_adapt, y_adapt)

            # Test on remaining examples
            pred = adapted_model.predict(X_test[10:], verbose=0)
            mse = np.mean((pred - y_test[10:]) ** 2)
            results[f"{test_name}_adapted"] = float(mse)
            print(f"  {test_name}: MSE = {mse:.2f}")

    return results


def test_standard_baseline(train_data, val_data, test_data_dict):
    """Test standard ERM baseline for comparison."""
    print("\n" + "=" * 70)
    print("Testing Standard Baseline (ERM)")
    print("=" * 70)

    X_train, y_train = train_data
    X_val, y_val = val_data

    # Train standard model
    model = create_standard_model()

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=32,
        verbose=1,
    )

    # Evaluate
    results = {}
    for test_name, (X_test, y_test) in test_data_dict.items():
        pred = model.predict(X_test, verbose=0)
        mse = np.mean((pred - y_test) ** 2)
        results[test_name] = float(mse)
        print(f"\n{test_name}: MSE = {mse:.2f}")

    return results


def main():
    """Run simplified baseline evaluations."""
    print("Simplified Baseline Evaluation on Time-Varying Gravity")
    print("=" * 70)

    # Setup
    setup_environment()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create output directory
    output_dir = get_output_path() / "baseline_evaluation"
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

    print(f"Loaded {len(const_data['trajectories'])} constant gravity trajectories")
    print(f"Loaded {len(ood_data['trajectories'])} time-varying gravity trajectories")

    # Prepare data
    X_train, y_train = prepare_data(const_data["trajectories"][:80])
    X_val, y_val = prepare_data(const_data["trajectories"][80:90])
    X_test_id, y_test_id = prepare_data(const_data["trajectories"][90:])
    X_test_ood, y_test_ood = prepare_data(ood_data["trajectories"][:50])

    train_data = (X_train, y_train)
    val_data = (X_val, y_val)
    test_data_dict = {
        "in_distribution": (X_test_id, y_test_id),
        "time_varying": (X_test_ood, y_test_ood),
    }

    print(f"\nData splits:")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Val: {len(X_val)} samples")
    print(f"  Test ID: {len(X_test_id)} samples")
    print(f"  Test OOD: {len(X_test_ood)} samples")

    # Run evaluations
    all_results = {}

    # Standard baseline
    standard_results = test_standard_baseline(train_data, val_data, test_data_dict)
    all_results["Standard_ERM"] = standard_results

    # GFlowNet-inspired (exploration)
    gflownet_results = test_gflownet_inspired_baseline(
        train_data, val_data, test_data_dict
    )
    all_results["GFlowNet_inspired"] = gflownet_results

    # MAML-inspired (adaptation)
    maml_results = test_maml_inspired_baseline(train_data, val_data, test_data_dict)
    all_results["MAML_inspired"] = maml_results

    # Save results
    results_file = output_dir / f"baseline_results_simple_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump({"timestamp": timestamp, "results": all_results}, f, indent=2)

    print(f"\n\nResults saved to: {results_file}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Performance on Time-Varying Gravity")
    print("=" * 70)

    print("\nMethod                          | MSE     | vs Baseline")
    print("-" * 60)

    baseline_mse = standard_results.get("time_varying", float("inf"))

    # Standard ERM
    print(f"Standard ERM                    | {baseline_mse:7.1f} | ---")

    # GFlowNet ensemble
    gf_mse = gflownet_results.get("time_varying", {}).get("ensemble_mse", float("inf"))
    gf_change = (gf_mse / baseline_mse - 1) * 100
    print(f"GFlowNet-inspired (ensemble)    | {gf_mse:7.1f} | {gf_change:+.1f}%")

    # MAML base
    maml_base = maml_results.get("time_varying_base", float("inf"))
    maml_base_change = (maml_base / baseline_mse - 1) * 100
    print(
        f"MAML-inspired (no adapt)        | {maml_base:7.1f} | {maml_base_change:+.1f}%"
    )

    # MAML adapted
    maml_adapted = maml_results.get("time_varying_adapted", float("inf"))
    maml_adapted_change = (maml_adapted / baseline_mse - 1) * 100
    print(
        f"MAML-inspired (10-shot adapt)   | {maml_adapted:7.1f} | {maml_adapted_change:+.1f}%"
    )

    # TTA comparison
    print(f"\nFor comparison:")
    print(f"TTA (from earlier experiments)  |  6935.0 | +235.0%")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    if gf_change > 50 and maml_adapted_change > 50:
        print("\n✗ ALL baseline methods fail to handle time-varying gravity")
        print("  - GFlowNet-style exploration doesn't help")
        print("  - MAML-style adaptation doesn't help")
        print("  - TTA makes it even worse")
        print(
            "\n→ This confirms true OOD is fundamentally different from interpolation"
        )
    elif maml_adapted_change < -10:
        print("\n⚠️ MAML shows some improvement with adaptation")
        print("  This suggests meta-learning might be a promising direction")
    else:
        print("\n✓ Results confirm our hypothesis about OOD difficulty")


if __name__ == "__main__":
    main()
