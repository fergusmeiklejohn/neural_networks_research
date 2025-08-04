"""Evaluate TTA with multi-timestep inputs for better adaptation signal."""

import os
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent.parent.parent))
os.environ["KERAS_BACKEND"] = "jax"

import keras

from models.test_time_adaptation.tta_wrappers import TTAWrapper


def create_multistep_model(input_steps=5, output_steps=10):
    """Create model that takes multiple timesteps as input."""
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(input_steps, 8)),
            # Process temporal information
            keras.layers.LSTM(128, return_sequences=True),
            keras.layers.LSTM(64),
            keras.layers.BatchNormalization(),
            # Predict future
            keras.layers.Dense(128, activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(8 * output_steps),
            keras.layers.Reshape((output_steps, 8)),
        ]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse", metrics=["mae"]
    )

    return model


def prepare_multistep_data(trajectories, input_steps=5, output_steps=10):
    """Prepare data with multiple input timesteps."""
    X, y = [], []

    for traj in trajectories:
        for i in range(len(traj) - input_steps - output_steps + 1):
            X.append(traj[i : i + input_steps])
            y.append(traj[i + input_steps : i + input_steps + output_steps])

    return np.array(X), np.array(y)


def evaluate_tta_multistep(
    model,
    test_trajectories,
    tta_config=None,
    input_steps=5,
    output_steps=10,
    n_samples=20,
):
    """Evaluate with multi-timestep inputs."""
    errors = []
    improvements = []

    if tta_config:
        tta_model = TTAWrapper(model, **tta_config)

    for i, traj in enumerate(test_trajectories[:n_samples]):
        if len(traj) < input_steps + output_steps:
            continue

        # Prepare input
        X = traj[:input_steps].reshape(1, input_steps, 8)
        y_true = traj[input_steps : input_steps + output_steps]

        # Baseline prediction (no adaptation)
        y_baseline = model.predict(X, verbose=0)[0]
        mse_baseline = np.mean((y_true - y_baseline) ** 2)

        if tta_config:
            # Reset model to initial weights
            tta_model.reset()

            # Adapted prediction
            y_adapted = tta_model.predict(X, adapt=True)[0]
            mse_adapted = np.mean((y_true - y_adapted) ** 2)

            improvement = (mse_baseline - mse_adapted) / mse_baseline * 100
            improvements.append(improvement)
            errors.append(mse_adapted)
        else:
            errors.append(mse_baseline)

    result = {
        "mse": np.mean(errors),
        "mse_std": np.std(errors),
        "n_samples": len(errors),
    }

    if improvements:
        result["improvement_mean"] = np.mean(improvements)
        result["improvement_std"] = np.std(improvements)
        result["improved_count"] = sum(1 for imp in improvements if imp > 0)

    return result


def test_different_hyperparameters(model, test_data, input_steps=5):
    """Test TTA with different hyperparameters."""
    results = {}

    # Different learning rates
    learning_rates = [1e-5, 1e-4, 1e-3, 1e-2]

    # Different adaptation steps
    adapt_steps = [1, 5, 10, 20]

    print("\nHyperparameter Grid Search:")
    print("-" * 60)

    for lr in learning_rates:
        for steps in adapt_steps:
            config_name = f"lr_{lr}_steps_{steps}"
            print(f"\nTesting {config_name}...")

            tta_config = {
                "tta_method": "tent",
                "adaptation_steps": steps,
                "learning_rate": lr,
            }

            result = evaluate_tta_multistep(
                model, test_data, tta_config, input_steps=input_steps, n_samples=10
            )

            results[config_name] = result
            print(f"  MSE: {result['mse']:.2f} (±{result['mse_std']:.2f})")
            if "improvement_mean" in result:
                print(
                    f"  Improvement: {result['improvement_mean']:.1f}% (±{result['improvement_std']:.1f}%)"
                )
                print(f"  Improved samples: {result['improved_count']}/10")

    return results


def main():
    print("Multi-Step TTA Evaluation")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load data
    data_dir = Path(__file__).parent.parent.parent / "data" / "true_ood_physics"

    print("\nLoading data...")
    const_files = sorted(data_dir.glob("constant_gravity_*.pkl"))
    varying_files = sorted(data_dir.glob("time_varying_gravity_*.pkl"))

    with open(const_files[-1], "rb") as f:
        const_data = pickle.load(f)
    with open(varying_files[-1], "rb") as f:
        varying_data = pickle.load(f)

    # Test different input lengths
    input_lengths = [1, 3, 5, 10]

    for input_steps in input_lengths:
        print(f"\n{'='*70}")
        print(f"Testing with {input_steps} input timesteps")
        print("=" * 70)

        # Create model
        model = create_multistep_model(input_steps=input_steps, output_steps=10)

        # Prepare training data
        X_train, y_train = prepare_multistep_data(
            const_data["trajectories"][:80], input_steps=input_steps
        )

        if len(X_train) < 10:
            print(f"Not enough data for {input_steps} timesteps, skipping...")
            continue

        print(f"\nTraining data shape: X={X_train.shape}, y={y_train.shape}")

        # Train model
        print("Training model...")
        history = model.fit(
            X_train, y_train, epochs=20, batch_size=32, verbose=0, validation_split=0.2
        )

        print(f"Final training loss: {history.history['loss'][-1]:.4f}")

        # Baseline evaluation
        print("\nBaseline evaluation (no TTA):")
        baseline_result = evaluate_tta_multistep(
            model, varying_data["trajectories"], None, input_steps=input_steps
        )
        print(
            f"  MSE: {baseline_result['mse']:.2f} (±{baseline_result['mse_std']:.2f})"
        )

        # Test TTA methods
        tta_methods = {
            "TENT": {
                "tta_method": "tent",
                "adaptation_steps": 10,
                "learning_rate": 1e-3,
            },
            "PhysicsTENT": {
                "tta_method": "physics_tent",
                "adaptation_steps": 10,
                "learning_rate": 1e-3,
                "physics_loss_weight": 0.1,
            },
            "TTT": {
                "tta_method": "ttt",
                "adaptation_steps": 10,
                "learning_rate": 5e-4,
                "auxiliary_weight": 0.3,
            },
        }

        print("\nTTA Results:")
        for method_name, config in tta_methods.items():
            result = evaluate_tta_multistep(
                model, varying_data["trajectories"], config, input_steps=input_steps
            )

            print(f"\n{method_name}:")
            print(f"  MSE: {result['mse']:.2f} (±{result['mse_std']:.2f})")

            if "improvement_mean" in result:
                print(f"  Average improvement: {result['improvement_mean']:.1f}%")
                print(
                    f"  Samples improved: {result['improved_count']}/{result['n_samples']}"
                )

                if result["improvement_mean"] > 0:
                    print(f"  ✓ TTA is helping!")

    # Best configuration search for 5-step input
    print("\n" + "=" * 70)
    print("Hyperparameter Search (5-step input)")
    print("=" * 70)

    if input_steps == 5:  # Use the last model
        hp_results = test_different_hyperparameters(
            model, varying_data["trajectories"], input_steps=5
        )

        # Find best configuration
        best_config = min(hp_results.items(), key=lambda x: x[1]["mse"])
        print(f"\nBest configuration: {best_config[0]}")
        print(f"Best MSE: {best_config[1]['mse']:.2f}")

    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)
    print("\n1. Multi-step inputs should provide richer adaptation signal")
    print("2. LSTM can capture temporal patterns better")
    print("3. Higher learning rates (1e-3) may work better with more information")
    print("4. Look for configurations where >50% of samples improve")


if __name__ == "__main__":
    main()
