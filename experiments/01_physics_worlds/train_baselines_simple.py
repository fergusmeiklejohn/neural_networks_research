"""
Simplified Baseline Training for Physics Worlds

A minimal version that trains baselines without complex dependencies.
"""

import os

os.environ["KERAS_BACKEND"] = "jax"

import json
import sys
from datetime import datetime
from pathlib import Path

import keras
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))


class SimplePhysicsBaseline:
    """Simple baseline model for physics prediction."""

    def __init__(self, input_shape, output_shape, name="baseline"):
        self.name = name
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.model = None
        self.history = None

    def build_model(self):
        """Build a simple feedforward model."""
        self.model = keras.Sequential(
            [
                keras.layers.Input(shape=self.input_shape),
                keras.layers.Flatten(),
                keras.layers.Dense(256, activation="relu"),
                keras.layers.Dense(256, activation="relu"),
                keras.layers.Dense(128, activation="relu"),
                keras.layers.Dense(np.prod(self.output_shape)),
                keras.layers.Reshape(self.output_shape),
            ]
        )

        self.model.compile(optimizer="adam", loss="mse", metrics=["mae"])

        return self.model

    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """Train the model."""
        print(f"\nTraining {self.name} baseline...")

        self.history = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
        )

        return self.history

    def evaluate(self, X_test, y_test):
        """Evaluate the model."""
        loss, mae = self.model.evaluate(X_test, y_test, verbose=0)
        return {"mse": loss, "mae": mae}


def generate_synthetic_physics_data(n_samples=1000):
    """Generate simple synthetic physics data."""
    print("Generating synthetic physics data...")

    # Simple 2D physics: position and velocity
    # Input: [x, y, vx, vy] at time t
    # Output: [x, y, vx, vy] at time t+1

    dt = 0.1
    gravity = -9.8

    # Generate random initial conditions
    X = np.random.randn(n_samples, 4)
    X[:, 0] *= 10  # x position
    X[:, 1] = np.abs(X[:, 1]) * 10 + 5  # y position (above ground)
    X[:, 2] *= 5  # x velocity
    X[:, 3] *= 5  # y velocity

    # Simple physics update
    y = X.copy()
    y[:, 0] += X[:, 2] * dt  # x = x + vx * dt
    y[:, 1] += X[:, 3] * dt  # y = y + vy * dt
    y[:, 3] += gravity * dt  # vy = vy + g * dt

    # Add some noise
    y += 0.01 * np.random.randn(*y.shape)

    return X, y


def train_baselines_simple():
    """Train simple baseline models on physics data."""

    # Generate data
    X_all, y_all = generate_synthetic_physics_data(2000)

    # Split data
    n_train = 1200
    n_val = 400

    X_train, y_train = X_all[:n_train], y_all[:n_train]
    X_val, y_val = X_all[n_train : n_train + n_val], y_all[n_train : n_train + n_val]
    X_test, y_test = X_all[n_train + n_val :], y_all[n_train + n_val :]

    print(f"\nData shapes:")
    print(f"Train: {X_train.shape}")
    print(f"Val: {X_val.shape}")
    print(f"Test: {X_test.shape}")

    # Define baseline configurations
    baselines = [
        {
            "name": "Standard_NN",
            "description": "Standard neural network (like ERM baseline)",
        },
        {"name": "Augmented_NN", "description": "NN with data augmentation"},
    ]

    results = {}

    for config in baselines:
        print(f"\n{'='*50}")
        print(f"Training {config['name']}")
        print(f"Description: {config['description']}")
        print(f"{'='*50}")

        # Create and build model
        baseline = SimplePhysicsBaseline(
            input_shape=(4,), output_shape=(4,), name=config["name"]
        )
        baseline.build_model()

        # Apply augmentation if needed
        if config["name"] == "Augmented_NN":
            # Simple augmentation: add noisy versions
            X_aug = np.vstack(
                [X_train, X_train + 0.1 * np.random.randn(*X_train.shape)]
            )
            y_aug = np.vstack(
                [y_train, y_train + 0.1 * np.random.randn(*y_train.shape)]
            )
        else:
            X_aug = X_train
            y_aug = y_train

        # Train
        start_time = datetime.now()
        history = baseline.train(X_aug, y_aug, X_val, y_val, epochs=20)
        train_time = (datetime.now() - start_time).total_seconds()

        # Evaluate
        test_results = baseline.evaluate(X_test, y_test)

        # Store results
        results[config["name"]] = {
            "description": config["description"],
            "train_time": train_time,
            "final_train_loss": history.history["loss"][-1],
            "final_val_loss": history.history["val_loss"][-1],
            "test_mse": test_results["mse"],
            "test_mae": test_results["mae"],
            "model_params": baseline.model.count_params(),
        }

        print(f"\nResults for {config['name']}:")
        print(f"Test MSE: {test_results['mse']:.4f}")
        print(f"Test MAE: {test_results['mae']:.4f}")
        print(f"Training time: {train_time:.1f}s")

    # Save results
    output_dir = Path("outputs/baseline_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / "simple_baseline_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    print(f"{'Model':<20} {'Test MSE':>10} {'Test MAE':>10} {'Time (s)':>10}")
    print("-" * 50)

    for name, res in results.items():
        print(
            f"{name:<20} {res['test_mse']:>10.4f} {res['test_mae']:>10.4f} {res['train_time']:>10.1f}"
        )

    print(f"\nResults saved to: {results_file}")

    return results


if __name__ == "__main__":
    print("Simple Physics Baseline Training")
    print("================================\n")

    results = train_baselines_simple()

    print("\n\nTraining complete!")
