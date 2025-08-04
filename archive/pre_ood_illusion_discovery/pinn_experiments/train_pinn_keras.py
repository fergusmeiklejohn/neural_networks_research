"""
Training script for Physics-Informed Neural Network using Keras built-in training.
This version uses compile() and fit() for proper gradient computation.
"""

import os

os.environ["KERAS_BACKEND"] = "jax"

import sys

sys.path.append("../..")

import json
from datetime import datetime
from pathlib import Path

import keras
import numpy as np
from keras import ops

from models.physics_simple_model import SimplePhysicsInformedModel


class PhysicsDataset:
    """Generate physics trajectory data."""

    def __init__(self, n_samples=1000, gravity_range=(-9.8, -3.7), seq_len=50):
        self.n_samples = n_samples
        self.gravity_range = gravity_range
        self.seq_len = seq_len

    def generate_trajectory(
        self, gravity=-9.8, friction=0.5, elasticity=0.8, damping=0.95
    ):
        """Generate a single trajectory."""
        dt = 1 / 30.0

        # Initial conditions
        pos1 = np.array([200.0, 300.0])
        pos2 = np.array([600.0, 400.0])
        vel1 = np.array([50.0, -20.0])
        vel2 = np.array([-30.0, 10.0])

        trajectory = []

        for t in range(self.seq_len):
            # Record state
            state = np.concatenate([pos1, pos2, vel1, vel2])
            trajectory.append(state)

            # Update physics
            vel1[1] += gravity * dt
            vel2[1] += gravity * dt
            vel1 *= damping
            vel2 *= damping

            pos1 += vel1 * dt
            pos2 += vel2 * dt

            # Boundaries
            for pos, vel in [(pos1, vel1), (pos2, vel2)]:
                if pos[0] < 20 or pos[0] > 780:
                    vel[0] *= -elasticity
                    pos[0] = np.clip(pos[0], 20, 780)
                if pos[1] < 20 or pos[1] > 580:
                    vel[1] *= -elasticity
                    pos[1] = np.clip(pos[1], 20, 580)

        return np.array(trajectory, dtype=np.float32)

    def generate(self):
        """Generate dataset."""
        trajectories = []
        params = []

        for i in range(self.n_samples):
            gravity = np.random.uniform(*self.gravity_range)
            friction = np.random.uniform(0.3, 0.7)
            elasticity = np.random.uniform(0.6, 0.9)
            damping = np.random.uniform(0.92, 0.98)

            traj = self.generate_trajectory(gravity, friction, elasticity, damping)
            trajectories.append(traj)
            params.append([gravity, friction, elasticity, damping])

        return np.array(trajectories), np.array(params)


class PINNWrapper(keras.Model):
    """Wrapper to make PINN work with Keras compile/fit."""

    def __init__(self, pinn_model, **kwargs):
        super().__init__(**kwargs)
        self.pinn = pinn_model

    def call(self, inputs, training=None):
        # Get full predictions from PINN
        predictions = self.pinn(inputs, training=training)
        # Return only trajectory for standard training
        return predictions["trajectory"]

    def train_step(self, state, data):
        # JAX backend passes state as first argument
        x, y = data

        # Forward pass
        with self.compute_gradient_tape() as tape:
            # Get trajectory prediction (this calls our modified call method)
            y_pred = self(x, training=True)

            # Also get full predictions for physics loss
            full_predictions = self.pinn(x, training=True)

            # Custom loss computation
            loss = ops.mean(ops.square(y_pred - y))

            # Add physics loss if enabled
            if self.pinn.use_physics_loss:
                # Extract states for physics loss
                positions, velocities = self.pinn.extract_physics_state(y_pred)
                physics_targets = {"positions": positions, "velocities": velocities}
                physics_losses = self.pinn.compute_physics_loss(
                    full_predictions, physics_targets
                )
                loss = loss + 0.01 * physics_losses["total_physics_loss"]

        # Compute gradients
        gradients = tape.gradient(loss, self.trainable_variables)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Update metrics
        self.compiled_metrics.update_state(y, y_pred)
        metrics = {m.name: m.result() for m in self.metrics}
        metrics["loss"] = loss

        return metrics


def evaluate_model(model, test_data, test_params, name):
    """Evaluate model on test data."""
    # Use next timestep prediction
    X_test = test_data[:, :-1]
    y_test = test_data[:, 1:]

    # Get trajectory prediction from wrapper
    pred_traj = model(X_test, training=False).numpy()

    # Get full predictions from underlying PINN
    full_predictions = model.pinn(X_test, training=False)

    # Compute MSE
    mse = np.mean((pred_traj - y_test) ** 2)

    # Check gravity prediction
    pred_params = full_predictions["physics_params"].numpy()
    gravity_error = np.mean(np.abs(pred_params[:, 0] - test_params[:, 0]))

    return {
        "name": name,
        "mse": float(mse),
        "gravity_error": float(gravity_error),
        "avg_pred_gravity": float(np.mean(pred_params[:, 0])),
        "true_gravity": float(test_params[0, 0]),
    }


def main():
    """Main training function."""
    print("=" * 80)
    print("Physics-Informed Neural Network Training (Keras Version)")
    print("=" * 80)

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"outputs/pinn_keras_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate datasets
    print("\nGenerating datasets...")

    # Training data - progressive stages
    earth_mars_data = PhysicsDataset(n_samples=1000, gravity_range=(-9.8, -3.7))
    moon_data = PhysicsDataset(n_samples=300, gravity_range=(-1.6, -1.6))
    jupiter_data = PhysicsDataset(n_samples=300, gravity_range=(-24.8, -24.8))

    # Test data
    test_earth = PhysicsDataset(n_samples=100, gravity_range=(-9.8, -9.8))
    test_moon = PhysicsDataset(n_samples=100, gravity_range=(-1.6, -1.6))
    test_jupiter = PhysicsDataset(n_samples=100, gravity_range=(-24.8, -24.8))

    # Generate all data
    print("Generating Earth-Mars training data...")
    train_em, params_em = earth_mars_data.generate()

    print("Generating Moon training data...")
    train_moon, params_moon = moon_data.generate()

    print("Generating Jupiter training data...")
    train_jupiter, params_jupiter = jupiter_data.generate()

    print("Generating test data...")
    test_earth_data, test_earth_params = test_earth.generate()
    test_moon_data, test_moon_params = test_moon.generate()
    test_jupiter_data, test_jupiter_params = test_jupiter.generate()

    # Create model
    print("\nCreating Physics-Informed Model...")
    base_model = SimplePhysicsInformedModel(
        sequence_length=49,  # After taking differences
        hidden_dim=128,
        num_layers=3,
        use_physics_loss=True,
    )

    # Wrap for Keras training
    model = PINNWrapper(base_model)

    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse", metrics=["mae"]
    )

    # Progressive training
    results = {"stages": []}

    print("\n" + "=" * 60)
    print("Stage 1: Earth-Mars Training")
    print("=" * 60)

    # Prepare data for next-step prediction
    X_train = train_em[:, :-1]
    y_train = train_em[:, 1:]

    # Train
    history1 = model.fit(
        X_train, y_train, batch_size=32, epochs=20, validation_split=0.2, verbose=1
    )

    # Evaluate
    stage1_results = {"stage": "Earth-Mars Only", "test_results": {}}

    for name, data, params in [
        ("Earth", test_earth_data, test_earth_params),
        ("Moon", test_moon_data, test_moon_params),
        ("Jupiter", test_jupiter_data, test_jupiter_params),
    ]:
        eval_result = evaluate_model(model, data, params, name)
        stage1_results["test_results"][name] = eval_result
        print(
            f"{name}: MSE={eval_result['mse']:.4f}, Gravity Error={eval_result['gravity_error']:.2f}"
        )

    results["stages"].append(stage1_results)

    print("\n" + "=" * 60)
    print("Stage 2: Add Moon Data")
    print("=" * 60)

    # Combine Earth-Mars and Moon data
    X_train_2 = np.concatenate([train_em[:, :-1], train_moon[:, :-1]])
    y_train_2 = np.concatenate([train_em[:, 1:], train_moon[:, 1:]])

    # Continue training with lower learning rate
    model.optimizer.learning_rate = 5e-4

    history2 = model.fit(
        X_train_2, y_train_2, batch_size=32, epochs=15, validation_split=0.2, verbose=1
    )

    # Evaluate
    stage2_results = {"stage": "Earth-Mars + Moon", "test_results": {}}

    for name, data, params in [
        ("Earth", test_earth_data, test_earth_params),
        ("Moon", test_moon_data, test_moon_params),
        ("Jupiter", test_jupiter_data, test_jupiter_params),
    ]:
        eval_result = evaluate_model(model, data, params, name)
        stage2_results["test_results"][name] = eval_result
        print(
            f"{name}: MSE={eval_result['mse']:.4f}, Gravity Error={eval_result['gravity_error']:.2f}"
        )

    results["stages"].append(stage2_results)

    print("\n" + "=" * 60)
    print("Stage 3: Add Jupiter Data")
    print("=" * 60)

    # Combine all data
    X_train_3 = np.concatenate(
        [train_em[:, :-1], train_moon[:, :-1], train_jupiter[:, :-1]]
    )
    y_train_3 = np.concatenate(
        [train_em[:, 1:], train_moon[:, 1:], train_jupiter[:, 1:]]
    )

    # Final training stage
    model.optimizer.learning_rate = 2e-4

    history3 = model.fit(
        X_train_3, y_train_3, batch_size=32, epochs=15, validation_split=0.2, verbose=1
    )

    # Final evaluation
    stage3_results = {"stage": "All Gravity Conditions", "test_results": {}}

    print("\nFinal Evaluation:")
    for name, data, params in [
        ("Earth", test_earth_data, test_earth_params),
        ("Moon", test_moon_data, test_moon_params),
        ("Jupiter", test_jupiter_data, test_jupiter_params),
    ]:
        eval_result = evaluate_model(model, data, params, name)
        stage3_results["test_results"][name] = eval_result
        print(
            f"{name}: MSE={eval_result['mse']:.4f}, Gravity Error={eval_result['gravity_error']:.2f}"
        )

    results["stages"].append(stage3_results)

    # Compare with baselines
    print("\n" + "=" * 80)
    print("PINN vs Baselines Comparison")
    print("=" * 80)

    baseline_jupiter_mse = {
        "ERM+Aug": 1.1284,
        "GFlowNet": 0.8500,
        "GraphExtrap": 0.7663,
        "MAML": 0.8228,
    }

    pinn_jupiter_mse = stage3_results["test_results"]["Jupiter"]["mse"]
    best_baseline = min(baseline_jupiter_mse.values())

    print(f"\nJupiter Gravity Performance:")
    print(f"PINN:          {pinn_jupiter_mse:.4f}")
    print(f"Best Baseline: {best_baseline:.4f}")
    print(f"Improvement:   {(1 - pinn_jupiter_mse/best_baseline)*100:.1f}%")

    # Save results
    results["comparison"] = {
        "pinn_jupiter_mse": pinn_jupiter_mse,
        "best_baseline_mse": best_baseline,
        "improvement_percent": (1 - pinn_jupiter_mse / best_baseline) * 100,
    }

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Save model
    model_path = output_dir / "pinn_model.keras"
    model.save(str(model_path))

    print(f"\nResults saved to: {output_dir}")
    print("\nTraining complete!")

    return model, results


if __name__ == "__main__":
    model, results = main()
