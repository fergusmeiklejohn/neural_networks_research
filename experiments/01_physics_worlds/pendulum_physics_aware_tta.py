"""
Physics-aware Test-Time Adaptation for pendulum using energy consistency.
Implements energy-based and Hamiltonian consistency losses as suggested by reviewer.
"""

import json
import pickle
import time
from pathlib import Path
from typing import Dict

import keras
import matplotlib.pyplot as plt
import numpy as np
from keras import ops


class PhysicsAwareTTA:
    """Test-Time Adaptation using physics-informed losses"""

    def __init__(self, base_model: keras.Model, adaptation_type: str = "energy"):
        """
        Args:
            base_model: Pre-trained model to adapt
            adaptation_type: 'energy', 'hamiltonian', or 'prediction' (baseline)
        """
        self.base_model = base_model
        self.adaptation_type = adaptation_type

        # Clone model for adaptation (preserve original)
        self.adapted_model = keras.models.clone_model(base_model)
        self.adapted_model.set_weights(base_model.get_weights())

        # TTA optimizer
        self.tta_optimizer = keras.optimizers.Adam(learning_rate=1e-4)

    def compute_energy_consistency_loss(self, states: np.ndarray) -> float:
        """Compute energy consistency loss for pendulum"""
        # States: [batch, time, features] where features = [x, y, theta, theta_dot, length]

        # Extract components
        x = states[:, :, 0]
        y = states[:, :, 1]
        theta = states[:, :, 2]
        theta_dot = states[:, :, 3]
        length = states[:, :, 4]

        # Constants
        g = 9.8
        m = 1.0  # Assume unit mass

        # Kinetic energy: KE = 0.5 * m * L^2 * theta_dot^2
        KE = 0.5 * m * (length**2) * (theta_dot**2)

        # Potential energy: PE = m * g * L * (1 - cos(theta))
        # Using small angle approximation: 1 - cos(theta) ≈ theta^2/2
        PE = m * g * length * (1 - ops.cos(theta))

        # Total energy
        E_total = KE + PE

        # Energy consistency loss: minimize variance of energy over time
        # For fixed-length pendulum, energy should be constant
        energy_mean = ops.mean(E_total, axis=1, keepdims=True)
        energy_variance = ops.mean((E_total - energy_mean) ** 2, axis=1)

        return ops.mean(energy_variance)

    def compute_hamiltonian_consistency_loss(self, states: np.ndarray) -> float:
        """Compute Hamiltonian consistency loss"""
        # For pendulum, Hamiltonian H = KE + PE (same as total energy)
        # But we can add additional constraints based on Hamilton's equations

        # Extract components
        theta = states[:, :, 2]
        theta_dot = states[:, :, 3]
        length = states[:, :, 4]

        g = 9.8

        # Compute time derivatives using finite differences
        dt = 1 / 60.0  # Assuming 60 fps
        theta_dot_numerical = (theta[:, 1:] - theta[:, :-1]) / dt
        theta_ddot_numerical = (theta_dot[:, 1:] - theta_dot[:, :-1]) / dt

        # Hamilton's equation for pendulum: theta_ddot = -(g/L) * sin(theta)
        # (ignoring damping and length variations for simplicity)
        theta_ddot_physics = -(g / length[:, :-1]) * ops.sin(theta[:, :-1])

        # Consistency loss: predicted dynamics should match physics
        dynamics_error = ops.mean((theta_ddot_numerical - theta_ddot_physics) ** 2)

        # Also include energy conservation
        energy_loss = self.compute_energy_consistency_loss(states)

        return dynamics_error + 0.1 * energy_loss

    def compute_prediction_consistency_loss(self, predictions: np.ndarray) -> float:
        """Standard prediction consistency loss (baseline)"""
        # Minimize variance of predictions across time
        pred_mean = ops.mean(predictions, axis=1, keepdims=True)
        pred_variance = ops.mean((predictions - pred_mean) ** 2)

        return pred_variance

    def adapt_batch(self, X_batch: np.ndarray, num_steps: int = 10):
        """Adapt model on a batch using selected loss"""

        # Create a simple adaptation function
        @keras.saving.register_keras_serializable()
        def tta_loss_fn(y_true, y_pred):
            """Custom loss for TTA"""
            if self.adaptation_type == "energy":
                return self.compute_energy_consistency_loss(y_pred)
            elif self.adaptation_type == "hamiltonian":
                return self.compute_hamiltonian_consistency_loss(y_pred)
            else:  # prediction
                return self.compute_prediction_consistency_loss(y_pred)

        # Compile adapted model with TTA loss
        self.adapted_model.compile(optimizer=self.tta_optimizer, loss=tta_loss_fn)

        # Adapt using dummy targets (we don't use them in the loss)
        dummy_targets = self.adapted_model.predict(X_batch, verbose=0)

        history = self.adapted_model.fit(
            X_batch, dummy_targets, epochs=num_steps, batch_size=len(X_batch), verbose=0
        )

        # Print progress
        for step in range(0, num_steps, 5):
            if step < len(history.history["loss"]):
                print(f"  TTA Step {step}: Loss = {history.history['loss'][step]:.6f}")

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate adapted model"""
        # Predictions from original model
        y_pred_original = self.base_model.predict(X_test, verbose=0)
        mse_original = np.mean((y_pred_original - y_test) ** 2)

        # Predictions from adapted model
        y_pred_adapted = self.adapted_model.predict(X_test, verbose=0)
        mse_adapted = np.mean((y_pred_adapted - y_test) ** 2)

        # Compute degradation/improvement
        adaptation_factor = mse_adapted / mse_original

        return {
            "mse_original": float(mse_original),
            "mse_adapted": float(mse_adapted),
            "adaptation_factor": float(adaptation_factor),
            "adaptation_type": self.adaptation_type,
        }


def test_physics_aware_tta():
    """Test different TTA variants on pendulum data"""
    print("Testing Physics-Aware TTA on Pendulum Data")
    print("=" * 50)

    # Load pre-trained model (check both locations)
    model_path = Path("outputs/pendulum_baselines/erm_best.keras")
    if not model_path.exists():
        # Try test location
        model_path = Path("outputs/pendulum_test_quick/erm_best.keras")
        if not model_path.exists():
            print(
                "Error: No pre-trained model found. Run train_pendulum_baselines.py first."
            )
            return

    base_model = keras.models.load_model(model_path)
    print(f"Loaded pre-trained model from {model_path}")

    # Load OOD test data (check both locations)
    data_path = Path("data/processed/pendulum/pendulum_test_ood.pkl")
    if not data_path.exists():
        # Try test location
        data_path = Path("data/processed/pendulum_test_quick/pendulum_test_ood.pkl")
        if not data_path.exists():
            print("Error: No OOD test data found. Generate pendulum data first.")
            return

    with open(data_path, "rb") as f:
        ood_data = pickle.load(f)

    # Prepare test data
    X_test, y_test = [], []
    for traj_data in ood_data["trajectories"][:100]:  # Use first 100 trajectories
        traj = traj_data["trajectory"]
        states = np.column_stack(
            [traj["x"], traj["y"], traj["theta"], traj["theta_dot"], traj["length"]]
        )

        for i in range(len(states) - 11):
            X_test.append(states[i : i + 1])
            y_test.append(states[i + 1 : i + 11])

    X_test = np.array(X_test)
    y_test = np.array(y_test)
    print(f"\nTest data shape: X={X_test.shape}, y={y_test.shape}")

    # Test different adaptation methods
    results = {}

    for adaptation_type in ["prediction", "energy", "hamiltonian"]:
        print(f"\n\nTesting {adaptation_type.upper()} consistency TTA...")
        print("-" * 40)

        # Create TTA instance
        tta = PhysicsAwareTTA(base_model, adaptation_type=adaptation_type)

        # Adapt on first batch
        adapt_batch_size = 32
        X_adapt = X_test[:adapt_batch_size]

        print(f"Adapting on batch of {adapt_batch_size} samples...")
        tta.adapt_batch(X_adapt, num_steps=20)

        # Evaluate on full test set
        print("\nEvaluating on full OOD test set...")
        eval_results = tta.evaluate(X_test, y_test)
        results[adaptation_type] = eval_results

        print(f"\nResults for {adaptation_type} TTA:")
        print(f"  Original MSE: {eval_results['mse_original']:.6f}")
        print(f"  Adapted MSE:  {eval_results['mse_adapted']:.6f}")
        print(
            f"  Factor: {eval_results['adaptation_factor']:.2f}x "
            + ("(improved)" if eval_results["adaptation_factor"] < 1 else "(degraded)")
        )

    # Save results
    output_dir = Path("outputs/pendulum_tta")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"physics_aware_tta_results_{timestamp}.json"

    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    # Create comparison plot
    plt.figure(figsize=(10, 6))

    methods = list(results.keys())
    original_mse = results["prediction"]["mse_original"]
    adapted_mses = [results[m]["mse_adapted"] for m in methods]
    factors = [results[m]["adaptation_factor"] for m in methods]

    x = np.arange(len(methods))
    width = 0.35

    plt.subplot(1, 2, 1)
    plt.bar(
        x - width / 2,
        [original_mse] * len(methods),
        width,
        label="Original",
        color="blue",
        alpha=0.7,
    )
    plt.bar(x + width / 2, adapted_mses, width, label="Adapted", color="red", alpha=0.7)
    plt.xlabel("TTA Method")
    plt.ylabel("MSE")
    plt.title("Pendulum OOD Prediction Error")
    plt.xticks(
        x,
        ["Prediction\nConsistency", "Energy\nConsistency", "Hamiltonian\nConsistency"],
    )
    plt.legend()
    plt.yscale("log")

    plt.subplot(1, 2, 2)
    colors = ["green" if f < 1 else "red" for f in factors]
    plt.bar(methods, factors, color=colors, alpha=0.7)
    plt.axhline(y=1, color="k", linestyle="--", alpha=0.5)
    plt.xlabel("TTA Method")
    plt.ylabel("Adaptation Factor")
    plt.title("TTA Performance (lower is better)")
    plt.xticks(range(len(methods)), ["Prediction", "Energy", "Hamiltonian"])

    plt.tight_layout()
    plot_file = output_dir / f"physics_aware_tta_comparison_{timestamp}.png"
    plt.savefig(plot_file, dpi=150)
    print(f"\n\nResults saved to {results_file}")
    print(f"Plot saved to {plot_file}")

    # Summary
    print("\n\n=== SUMMARY ===")
    print(f"{'Method':<20} {'Original MSE':<15} {'Adapted MSE':<15} {'Factor':<10}")
    print("-" * 60)
    for method in methods:
        r = results[method]
        print(
            f"{method.capitalize():<20} {r['mse_original']:<15.6f} "
            + f"{r['mse_adapted']:<15.6f} {r['adaptation_factor']:<10.2f}x"
        )


if __name__ == "__main__":
    test_physics_aware_tta()
