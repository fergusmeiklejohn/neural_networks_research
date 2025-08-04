"""
Train baseline models on actual physics world data.
Uses the same data loading as train_minimal_pinn.py
"""

import os

os.environ["KERAS_BACKEND"] = "jax"

import sys

sys.path.append("../..")

import argparse
import json
from datetime import datetime
from pathlib import Path

import keras
import numpy as np
from keras import layers, ops


def load_physics_data():
    """Load existing physics datasets - same as train_minimal_pinn.py"""
    import pickle

    # Use the most recent complete dataset
    data_dir = Path("data/processed/physics_worlds_v2")

    # Load training data
    with open(data_dir / "train_data.pkl", "rb") as f:
        train_data = pickle.load(f)

    # Filter for 2-ball trajectories only
    print(f"Total training trajectories: {len(train_data)}")

    # Filter by trajectory shape
    train_data_2ball = []
    for traj in train_data:
        traj_array = np.array(traj["trajectory"])
        if traj_array.shape[1] >= 17:  # Has enough features for 2 balls
            train_data_2ball.append(traj)
    print(f"Filtered to {len(train_data_2ball)} 2-ball trajectories")

    # Extract trajectories
    trajectories = []
    physics_params = []

    for sample in train_data_2ball:
        traj_array = np.array(sample["trajectory"])
        if traj_array.shape[1] >= 17:
            # Extract core features: positions and velocities
            x1 = traj_array[:, 1:2]
            y1 = traj_array[:, 2:3]
            vx1 = traj_array[:, 3:4]
            vy1 = traj_array[:, 4:5]
            x2 = traj_array[:, 9:10]
            y2 = traj_array[:, 10:11]
            vx2 = traj_array[:, 11:12]
            vy2 = traj_array[:, 12:13]

            core_features = np.concatenate([x1, y1, x2, y2, vx1, vy1, vx2, vy2], axis=1)

            # Truncate to 50 timesteps
            if core_features.shape[0] > 50:
                core_features = core_features[:50]
            elif core_features.shape[0] < 50:
                padding = np.zeros((50 - core_features.shape[0], 8))
                core_features = np.concatenate([core_features, padding], axis=0)

            trajectories.append(core_features)

            # Extract physics parameters
            if "physics_config" in sample:
                config = sample["physics_config"]
                pixel_to_meter = 40.0
                params = [
                    config.get("gravity", -392.0) / pixel_to_meter,
                    config.get("friction", 0.1),
                    config.get("elasticity", 0.8),
                ]
                physics_params.append(params)

    X_train = np.array(trajectories)
    params_train = np.array(physics_params) if physics_params else None

    # Load validation data
    with open(data_dir / "val_in_dist_data.pkl", "rb") as f:
        val_data = pickle.load(f)

    # Process validation data similarly
    val_trajectories = []
    for sample in val_data:
        traj_array = np.array(sample["trajectory"])
        if traj_array.shape[1] >= 17:
            x1 = traj_array[:, 1:2]
            y1 = traj_array[:, 2:3]
            vx1 = traj_array[:, 3:4]
            vy1 = traj_array[:, 4:5]
            x2 = traj_array[:, 9:10]
            y2 = traj_array[:, 10:11]
            vx2 = traj_array[:, 11:12]
            vy2 = traj_array[:, 12:13]

            core_features = np.concatenate([x1, y1, x2, y2, vx1, vy1, vx2, vy2], axis=1)

            if core_features.shape[0] > 50:
                core_features = core_features[:50]
            elif core_features.shape[0] < 50:
                padding = np.zeros((50 - core_features.shape[0], 8))
                core_features = np.concatenate([core_features, padding], axis=0)

            val_trajectories.append(core_features)

    X_val = (
        np.array(val_trajectories)
        if val_trajectories
        else np.array([]).reshape(0, 50, 8)
    )

    # Load test data

    # Jupiter test set
    with open(data_dir / "test_extrapolation_data.pkl", "rb") as f:
        test_extrap = pickle.load(f)

    jupiter_trajectories = []
    jupiter_params = []

    for sample in test_extrap:
        if "physics_config" in sample:
            config = sample["physics_config"]
            gravity_ms2 = config.get("gravity", -392.0) / 40.0

            # Filter for Jupiter-like gravity
            if gravity_ms2 < -20.0:
                traj_array = np.array(sample["trajectory"])
                if traj_array.shape[1] >= 17:
                    x1 = traj_array[:, 1:2]
                    y1 = traj_array[:, 2:3]
                    vx1 = traj_array[:, 3:4]
                    vy1 = traj_array[:, 4:5]
                    x2 = traj_array[:, 9:10]
                    y2 = traj_array[:, 10:11]
                    vx2 = traj_array[:, 11:12]
                    vy2 = traj_array[:, 12:13]

                    core_features = np.concatenate(
                        [x1, y1, x2, y2, vx1, vy1, vx2, vy2], axis=1
                    )

                    if core_features.shape[0] > 50:
                        core_features = core_features[:50]
                    elif core_features.shape[0] < 50:
                        padding = np.zeros((50 - core_features.shape[0], 8))
                        core_features = np.concatenate([core_features, padding], axis=0)

                    jupiter_trajectories.append(core_features)
                    jupiter_params.append(
                        [
                            gravity_ms2,
                            config.get("friction", 0.1),
                            config.get("elasticity", 0.8),
                        ]
                    )

    X_test_jupiter = np.array(jupiter_trajectories) if jupiter_trajectories else None

    print(f"Loaded data shapes:")
    print(f"  Train: {X_train.shape}")
    print(f"  Val: {X_val.shape}")
    if X_test_jupiter is not None:
        print(f"  Test (Jupiter): {X_test_jupiter.shape}")
        print(f"  Jupiter gravity range: {[p[0] for p in jupiter_params[:5]]}")

    return X_train, X_val, X_test_jupiter, params_train


class GFlowNetBaseline(keras.Model):
    """Simplified GFlowNet for physics prediction."""

    def __init__(self, hidden_dim=128):
        super().__init__()
        self.encoder = keras.Sequential(
            [
                layers.Flatten(),
                layers.Dense(hidden_dim, activation="relu"),
                layers.Dense(hidden_dim, activation="relu"),
            ]
        )

        self.flow_net = keras.Sequential(
            [
                layers.Dense(hidden_dim, activation="relu"),
                layers.Dense(hidden_dim, activation="relu"),
                layers.Dense(50 * 8),  # Output full trajectory
            ]
        )

        self.exploration_bonus = 0.1
        # Create seed generator for JAX
        self.seed_generator = keras.random.SeedGenerator(seed=42)

    def call(self, inputs, training=None):
        # Encode initial state
        encoded = self.encoder(inputs)

        # Add exploration noise during training
        if training:
            noise = keras.random.normal(
                shape=ops.shape(encoded),
                stddev=self.exploration_bonus,
                seed=self.seed_generator,
            )
            encoded = encoded + noise

        # Generate trajectory
        output = self.flow_net(encoded)
        output = ops.reshape(output, (-1, 50, 8))

        return output


class MAMLBaseline(keras.Model):
    """Simplified MAML for physics prediction."""

    def __init__(self, hidden_dim=64):
        super().__init__()
        self.base_model = keras.Sequential(
            [
                layers.Flatten(),
                layers.Dense(hidden_dim, activation="relu"),
                layers.Dense(hidden_dim, activation="relu"),
                layers.Dense(50 * 8),
                layers.Reshape((50, 8)),
            ]
        )

        self.inner_lr = 0.01
        self.inner_steps = 5

    def call(self, inputs, training=None):
        return self.base_model(inputs)

    def adapt(self, support_x, support_y, steps=None):
        """Adapt model on support set."""
        if steps is None:
            steps = self.inner_steps

        # Clone weights for adaptation
        adapted_weights = [w.numpy().copy() for w in self.base_model.trainable_weights]

        # Inner loop adaptation
        for _ in range(steps):
            with keras.ops.GradientTape() as tape:
                predictions = self.base_model(support_x)
                loss = ops.mean(ops.square(predictions - support_y))

            gradients = tape.gradient(loss, self.base_model.trainable_weights)

            # Update adapted weights
            for i, (w, g) in enumerate(zip(adapted_weights, gradients)):
                if g is not None:
                    adapted_weights[i] = w - self.inner_lr * g.numpy()

        return adapted_weights


def train_baseline(model_name, X_train, X_val, X_test_jupiter):
    """Train a specific baseline model."""
    print(f"\n{'='*60}")
    print(f"Training {model_name} Baseline")
    print(f"{'='*60}")

    # Create model based on name
    if model_name == "gflownet":
        model = GFlowNetBaseline()
    elif model_name == "maml":
        model = MAMLBaseline()
    else:
        # Default feedforward model
        model = keras.Sequential(
            [
                layers.Input(shape=(50, 8)),
                layers.Flatten(),
                layers.Dense(256, activation="relu"),
                layers.Dense(256, activation="relu"),
                layers.Dense(128, activation="relu"),
                layers.Dense(50 * 8),
                layers.Reshape((50, 8)),
            ]
        )

    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse", metrics=["mae"]
    )

    # Build model
    dummy_input = ops.zeros((1, 50, 8))
    _ = model(dummy_input)
    print(f"Model parameters: {model.count_params():,}")

    # Train
    print("\nTraining...")
    history = model.fit(
        X_train,
        X_train,  # Autoencoder style
        validation_data=(X_val, X_val),
        epochs=50,
        batch_size=32,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5),
        ],
        verbose=1,
    )

    # Evaluate on Jupiter if available
    if X_test_jupiter is not None:
        print("\nEvaluating on Jupiter gravity...")
        jupiter_loss = model.evaluate(X_test_jupiter, X_test_jupiter, verbose=0)
        print(f"Jupiter MSE: {jupiter_loss[0]:.4f}")

        # Compare with known baselines
        print("\nComparison with other models:")
        print(f"  {model_name}: {jupiter_loss[0]:.4f}")
        print(f"  GraphExtrap: 0.766")
        print(f"  Failed PINN: 880.879")

    # Save model and results
    output_dir = Path("outputs/baseline_results")
    output_dir.mkdir(exist_ok=True, parents=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = output_dir / f"{model_name}_model_{timestamp}.keras"
    model.save(model_path)

    results = {
        "model": model_name,
        "parameters": model.count_params(),
        "training_loss": float(history.history["loss"][-1]),
        "validation_loss": float(history.history["val_loss"][-1]),
        "jupiter_mse": float(jupiter_loss[0]) if X_test_jupiter is not None else None,
        "timestamp": timestamp,
    }

    results_path = output_dir / f"{model_name}_results_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nModel saved to: {model_path}")
    print(f"Results saved to: {results_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train baseline models on physics data"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["gflownet", "maml", "simple"],
        default="gflownet",
        help="Which baseline to train",
    )
    args = parser.parse_args()

    # Load data
    print("Loading physics world data...")
    X_train, X_val, X_test_jupiter, params_train = load_physics_data()

    # Train baseline
    results = train_baseline(args.model, X_train, X_val, X_test_jupiter)

    print("\nTraining complete!")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
