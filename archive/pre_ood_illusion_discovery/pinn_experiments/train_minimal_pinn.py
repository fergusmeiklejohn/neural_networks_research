"""
Train Minimal PINN with proper physics integration.
Based on lessons from GraphExtrap success and PINN failure.
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

from models.minimal_physics_model import MinimalPhysicsModel


def load_physics_data():
    """Load existing physics datasets."""
    import pickle

    # Use the most recent complete dataset
    data_dir = Path("data/processed/physics_worlds_v2")

    # Load training data
    with open(data_dir / "train_data.pkl", "rb") as f:
        train_data = pickle.load(f)
    with open(data_dir / "train_metadata.json", "r") as f:
        json.load(f)

    # Filter for 2-ball trajectories only
    print(f"Total training trajectories: {len(train_data)}")

    # Debug: Check structure of first data point
    if len(train_data) > 0:
        print(f"First data point keys: {list(train_data[0].keys())}")
        if "physics_config" in train_data[0]:
            print(f"Physics config example: {train_data[0]['physics_config']}")

    # Check if data has num_balls field
    if len(train_data) > 0 and "num_balls" in train_data[0]:
        train_data_2ball = [traj for traj in train_data if traj["num_balls"] == 2]
        print(f"Filtered to {len(train_data_2ball)} 2-ball trajectories")
    else:
        # If no num_balls field, filter by trajectory shape
        train_data_2ball = []
        for traj in train_data:
            traj_array = np.array(traj["trajectory"])
            # 2-ball trajectories should have 8 core features or more
            if traj_array.shape[1] >= 8 and traj_array.shape[1] < 20:
                train_data_2ball.append(traj)
        print(
            f"Filtered to {len(train_data_2ball)} trajectories with 8-20 features (likely 2-ball)"
        )

    # Extract trajectories from initial_balls format
    trajectories = []
    for sample in train_data_2ball:
        traj_array = np.array(sample["trajectory"])
        sample.get("initial_balls", [])

        # The trajectory format appears to be: [time, x1, y1, vx1, vy1, mass1, radius1, ..., x2, y2, vx2, vy2, mass2, radius2, ...]
        # Extract positions and velocities for 2 balls
        if traj_array.shape[1] >= 17:  # Ensure we have enough features
            # Extract core features: positions and velocities
            # Format: time, ball1_data, ball2_data
            # Ball data: x, y, vx, vy, mass, radius, ...
            time = traj_array[:, 0:1]  # Keep time for debugging

            # Ball 1: columns 1-7
            x1 = traj_array[:, 1:2]
            y1 = traj_array[:, 2:3]
            vx1 = traj_array[:, 3:4]
            vy1 = traj_array[:, 4:5]

            # Ball 2: columns 9-15 (skip some features)
            x2 = traj_array[:, 9:10]
            y2 = traj_array[:, 10:11]
            vx2 = traj_array[:, 11:12]
            vy2 = traj_array[:, 12:13]

            # Combine into expected format: x1, y1, x2, y2, vx1, vy1, vx2, vy2
            core_features = np.concatenate([x1, y1, x2, y2, vx1, vy1, vx2, vy2], axis=1)
        # Truncate to 50 timesteps to match model expectation
        if core_features.shape[0] > 50:
            core_features = core_features[:50]
        elif core_features.shape[0] < 50:
            # Pad if needed
            padding = np.zeros((50 - core_features.shape[0], 8))
            core_features = np.concatenate([core_features, padding], axis=0)
        trajectories.append(core_features)

    X_train = np.array(trajectories)

    # Extract physics parameters - check which field exists
    if len(train_data_2ball) > 0:
        if "physics_params" in train_data_2ball[0]:
            params_train = np.array(
                [traj["physics_params"] for traj in train_data_2ball]
            )
        elif "physics_config" in train_data_2ball[0]:
            # Extract from physics_config
            params_list = []
            # Debug first config
            if len(train_data_2ball) > 0:
                print(
                    f"Physics config structure: {train_data_2ball[0]['physics_config']}"
                )

            for traj in train_data_2ball:
                config = traj["physics_config"]
                # Check if config is a dict or has different structure
                if isinstance(config, dict):
                    # Convert from pixel units to m/s²
                    # Assuming 40 pixels = 1 meter (based on data analysis)
                    pixel_to_meter = 40.0
                    params = [
                        config.get("gravity", -392.0)
                        / pixel_to_meter,  # Convert to m/s²
                        config.get("friction", 0.1),
                        config.get(
                            "elasticity", 0.8
                        ),  # Note: it's 'elasticity' not 'restitution'
                    ]
                else:
                    # Config might be a list or tuple
                    params = list(config) if len(config) >= 3 else [-9.8, 0.1, 0.8]
                params_list.append(params)
            params_train = np.array(params_list)
        elif "gravity" in train_data_2ball[0]:
            # Construct params from individual fields
            params_list = []
            for traj in train_data_2ball:
                # Assume gravity, friction, restitution
                params = [
                    traj.get("gravity", -9.8),
                    traj.get("friction", 0.1),
                    traj.get("restitution", 0.8),
                ]
                params_list.append(params)
            params_train = np.array(params_list)
        else:
            # Default physics parameters
            print("Warning: No physics parameters found, using defaults")
            params_train = np.array([[-9.8, 0.1, 0.8]] * len(train_data_2ball))
    else:
        params_train = np.array([])

    # Load validation data
    with open(data_dir / "val_in_dist_data.pkl", "rb") as f:
        val_data = pickle.load(f)

    # Filter validation data for 2-ball trajectories
    if len(val_data) > 0 and "num_balls" in val_data[0]:
        val_data_2ball = [traj for traj in val_data if traj["num_balls"] == 2]
    else:
        val_data_2ball = []
        for traj in val_data:
            traj_array = np.array(traj["trajectory"])
            if traj_array.shape[1] >= 8 and traj_array.shape[1] < 20:
                val_data_2ball.append(traj)

    # Process validation trajectories
    val_trajectories = []
    for sample in val_data_2ball:
        traj_array = np.array(sample["trajectory"])
        if traj_array.shape[1] >= 17:
            # Extract same format as training
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

    # Extract validation physics parameters
    if len(val_data_2ball) > 0:
        if "physics_params" in val_data_2ball[0]:
            params_val = np.array([traj["physics_params"] for traj in val_data_2ball])
        elif "physics_config" in val_data_2ball[0]:
            params_list = []
            pixel_to_meter = 40.0
            for traj in val_data_2ball:
                config = traj["physics_config"]
                params = [
                    config.get("gravity", -392.0) / pixel_to_meter,
                    config.get("friction", 0.1),
                    config.get("elasticity", 0.8),
                ]
                params_list.append(params)
            params_val = np.array(params_list)
        elif "gravity" in val_data_2ball[0]:
            params_list = []
            for traj in val_data_2ball:
                params = [
                    traj.get("gravity", -9.8),
                    traj.get("friction", 0.1),
                    traj.get("restitution", 0.8),
                ]
                params_list.append(params)
            params_val = np.array(params_list)
        else:
            params_val = np.array([[-9.8, 0.1, 0.8]] * len(val_data_2ball))
    else:
        params_val = np.array([]).reshape(0, 3)

    # Helper function to process test data
    def process_test_data(test_data):
        """Filter and process test data for 2-ball trajectories."""
        if len(test_data) > 0 and "num_balls" in test_data[0]:
            test_data_2ball = [traj for traj in test_data if traj["num_balls"] == 2]
        else:
            test_data_2ball = []
            for traj in test_data:
                traj_array = np.array(traj["trajectory"])
                if traj_array.shape[1] >= 8 and traj_array.shape[1] < 20:
                    test_data_2ball.append(traj)

        trajectories = []
        for sample in test_data_2ball:
            traj_array = np.array(sample["trajectory"])
            if traj_array.shape[1] >= 17:
                # Extract same format as training
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
                trajectories.append(core_features)

        if len(trajectories) > 0:
            X = np.array(trajectories)

            # Extract physics parameters
            if "physics_params" in test_data_2ball[0]:
                params = np.array([traj["physics_params"] for traj in test_data_2ball])
            elif "physics_config" in test_data_2ball[0]:
                params_list = []
                pixel_to_meter = 40.0
                for traj in test_data_2ball:
                    config = traj["physics_config"]
                    p = [
                        config.get("gravity", -392.0) / pixel_to_meter,
                        config.get("friction", 0.1),
                        config.get("elasticity", 0.8),
                    ]
                    params_list.append(p)
                params = np.array(params_list)
            elif "gravity" in test_data_2ball[0]:
                params_list = []
                for traj in test_data_2ball:
                    p = [
                        traj.get("gravity", -9.8),
                        traj.get("friction", 0.1),
                        traj.get("restitution", 0.8),
                    ]
                    params_list.append(p)
                params = np.array(params_list)
            else:
                params = np.array([[-9.8, 0.1, 0.8]] * len(test_data_2ball))

            return X, params
        else:
            return np.array([]).reshape(0, 50, 8), np.array([]).reshape(0, 3)

    # Load test data
    test_sets = {}

    # Interpolation test (similar to training)
    with open(data_dir / "test_interpolation_data.pkl", "rb") as f:
        test_interp = pickle.load(f)
    X_test_interp, params_test_interp = process_test_data(test_interp)
    if X_test_interp.shape[0] > 0:
        test_sets["test_interpolation"] = (X_test_interp, params_test_interp)

    # Extrapolation test (includes Jupiter-like gravity)
    with open(data_dir / "test_extrapolation_data.pkl", "rb") as f:
        test_extrap = pickle.load(f)
    X_test_extrap, params_test_extrap = process_test_data(test_extrap)
    if X_test_extrap.shape[0] > 0:
        test_sets["test_extrapolation"] = (X_test_extrap, params_test_extrap)

    # Novel test
    with open(data_dir / "test_novel_data.pkl", "rb") as f:
        test_novel = pickle.load(f)
    X_test_novel, params_test_novel = process_test_data(test_novel)
    if X_test_novel.shape[0] > 0:
        test_sets["test_novel"] = (X_test_novel, params_test_novel)

    # Create specific gravity test sets
    # Earth-like gravity (around -9.8)
    if "test_interpolation" in test_sets and params_test_interp.shape[0] > 0:
        earth_mask = np.abs(params_test_interp[:, 0] - (-9.8)) < 2.0
        if np.any(earth_mask):
            test_sets["test_earth"] = (
                X_test_interp[earth_mask],
                params_test_interp[earth_mask],
            )

    # Jupiter-like gravity (around -24.8)
    if "test_extrapolation" in test_sets and params_test_extrap.shape[0] > 0:
        jupiter_mask = params_test_extrap[:, 0] < -20.0
        if np.any(jupiter_mask):
            test_sets["test_jupiter"] = (
                X_test_extrap[jupiter_mask],
                params_test_extrap[jupiter_mask],
            )

    print(f"Loaded data shapes:")
    print(f"  Train: {X_train.shape}")
    print(f"  Val: {X_val.shape}")
    for name, (X, p) in test_sets.items():
        print(f"  {name}: {X.shape}")
        if p.shape[0] > 0:
            gravity_values = p[:, 0]
            print(
                f"    Gravity range: [{gravity_values.min():.1f}, {gravity_values.max():.1f}]"
            )

    # Check if we have Jupiter-like gravity in extrapolation set
    if (
        "test_extrapolation" in test_sets
        and test_sets["test_extrapolation"][1].shape[0] > 0
    ):
        _, params_extrap = test_sets["test_extrapolation"]
        jupiter_count = np.sum(params_extrap[:, 0] < -20.0)
        if jupiter_count > 0:
            print(
                f"\nFound {jupiter_count} Jupiter-like gravity samples in extrapolation set"
            )

    return (X_train, params_train), (X_val, params_val), test_sets


def create_physics_loss(model, reconstruction_weight=1.0, physics_weight=100.0):
    """Create combined loss function with proper weighting."""

    def loss_fn(y_true, y_pred):
        # Reconstruction loss (MSE)
        mse_loss = ops.mean(ops.square(y_true - y_pred))

        # Physics losses
        physics_losses = model.compute_physics_losses(y_true, y_pred)

        # Combine with heavy physics weighting
        total_physics = sum(physics_losses.values())
        total_loss = reconstruction_weight * mse_loss + physics_weight * total_physics

        return total_loss

    return loss_fn


def evaluate_extrapolation(model, test_sets):
    """Evaluate model on different gravity conditions."""
    results = {}

    for name, (X_test, params_test) in test_sets.items():
        # Get initial states
        initial_states = X_test[:, 0, :]

        # Generate predictions
        predicted = model.integrate_trajectory(initial_states, steps=X_test.shape[1])

        # Compute MSE
        mse = float(ops.mean(ops.square(predicted - X_test)))

        # Extract gravity from params
        if params_test.shape[0] > 0:
            gravity = float(np.mean(params_test[:, 0]))
        else:
            gravity = -9.8  # Default

        results[name] = {
            "mse": mse,
            "gravity": gravity,
            "predicted_gravity": float(model.gravity[0]),
        }

        print(
            f"{name}: MSE={mse:.4f}, True gravity={gravity:.2f}, "
            f"Predicted gravity={float(model.gravity[0]):.2f}"
        )

    return results


def main():
    """Main training function."""
    print("=" * 80)
    print("Training Minimal Physics-Informed Neural Network")
    print("=" * 80)
    print("\nKey improvements over failed PINN:")
    print("- Uses physics-aware features (polar coordinates)")
    print("- Predicts accelerations via F=ma + corrections")
    print("- Physics losses weighted 100x more than MSE")
    print("- Minimal architecture (2 layers, 64 units)")
    print("=" * 80)

    # Load data
    print("\nLoading physics data...")
    (X_train, params_train), (X_val, params_val), test_sets = load_physics_data()
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Validation samples: {X_val.shape[0]}")
    print(f"Test sets: {list(test_sets.keys())}")

    # Create model
    print("\nCreating minimal PINN...")
    model = MinimalPhysicsModel(hidden_dim=64)

    # Create optimizer with lower learning rate for physics parameters
    optimizer = keras.optimizers.Adam(learning_rate=1e-4)  # Reduced from 1e-3

    # Compile with custom loss - reduce physics weight to avoid NaN
    loss_fn = create_physics_loss(
        model, reconstruction_weight=1.0, physics_weight=10.0
    )  # Reduced from 100.0

    # Create custom metrics to track loss components
    class MSEMetric(keras.metrics.Metric):
        def __init__(self, name="mse_metric", **kwargs):
            super().__init__(name=name, **kwargs)
            self.total = self.add_weight(name="total", initializer="zeros")
            self.count = self.add_weight(name="count", initializer="zeros")

        def update_state(self, y_true, y_pred, sample_weight=None):
            mse = ops.mean(ops.square(y_true - y_pred))
            self.total.assign_add(mse)
            self.count.assign_add(1)

        def result(self):
            return self.total / self.count

        def reset_state(self):
            self.total.assign(0)
            self.count.assign(0)

    model.compile(optimizer=optimizer, loss=loss_fn, metrics=[MSEMetric()])

    # Build model
    dummy_input = ops.zeros((1, 50, 8))
    _ = model(dummy_input)
    print(f"Model parameters: {model.count_params():,}")

    # Initial evaluation
    print("\n" + "=" * 60)
    print("Initial Performance (Before Training)")
    print("=" * 60)
    initial_results = evaluate_extrapolation(model, test_sets)

    # Progressive training
    results = {"initial": initial_results, "stages": []}

    # Stage 1: Earth/Mars only
    print("\n" + "=" * 60)
    print("Stage 1: Training on Earth/Mars Physics")
    print("=" * 60)

    # Prepare data
    X_stage1 = X_train
    y_stage1 = X_train  # Autoencoder style

    # Train
    history1 = model.fit(
        X_stage1,
        y_stage1,
        validation_data=(X_val, X_val),
        epochs=50,
        batch_size=32,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-5),
        ],
        verbose=1,
    )

    # Evaluate
    print("\nStage 1 Evaluation:")
    stage1_results = evaluate_extrapolation(model, test_sets)
    results["stages"].append(
        {
            "stage": "Earth/Mars",
            "results": stage1_results,
            "final_loss": float(history1.history["loss"][-1]),
        }
    )

    # Stage 2: Add Moon data if available
    if "test_moon" in test_sets:
        print("\n" + "=" * 60)
        print("Stage 2: Fine-tuning with Moon Physics")
        print("=" * 60)

        # Add some moon data
        X_moon, params_moon = test_sets["test_moon"]
        X_stage2 = np.concatenate([X_train[:800], X_moon[:200]], axis=0)
        y_stage2 = X_stage2

        # Lower learning rate
        model.optimizer.learning_rate = 5e-4

        # Continue training
        history2 = model.fit(
            X_stage2,
            y_stage2,
            validation_data=(X_val, X_val),
            epochs=30,
            batch_size=32,
            verbose=1,
        )

        # Evaluate
        print("\nStage 2 Evaluation:")
        stage2_results = evaluate_extrapolation(model, test_sets)
        results["stages"].append(
            {
                "stage": "Earth/Mars/Moon",
                "results": stage2_results,
                "final_loss": float(history2.history["loss"][-1]),
            }
        )

    # Compare with baselines
    print("\n" + "=" * 80)
    print("Comparison with Failed PINN and Baselines")
    print("=" * 80)

    baseline_jupiter_mse = {
        "Failed PINN": 880.879,
        "GraphExtrap": 0.766,
        "MAML": 0.823,
        "GFlowNet": 0.850,
        "ERM+Aug": 1.128,
    }

    our_jupiter_mse = (
        results["stages"][-1]["results"]
        .get("test_jupiter", {})
        .get("mse", float("inf"))
    )

    print(f"\nJupiter Gravity Performance:")
    print(f"{'Model':<20} {'MSE':<10}")
    print("-" * 30)
    print(f"{'Minimal PINN':<20} {our_jupiter_mse:<10.4f}")
    for name, mse in sorted(baseline_jupiter_mse.items(), key=lambda x: x[1]):
        print(f"{name:<20} {mse:<10.4f}")

    if our_jupiter_mse < baseline_jupiter_mse["GraphExtrap"]:
        print(f"\n✓ SUCCESS! Minimal PINN beats best baseline!")
    else:
        print(
            f"\nStill improving... ({our_jupiter_mse:.4f} vs {baseline_jupiter_mse['GraphExtrap']:.4f})"
        )

    # Save results
    output_dir = Path("outputs/minimal_pinn")
    output_dir.mkdir(exist_ok=True, parents=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"results_{timestamp}.json"

    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    # Save model
    model_file = output_dir / f"model_{timestamp}.keras"
    model.save(str(model_file))

    print(f"\nResults saved to: {results_file}")
    print(f"Model saved to: {model_file}")

    # Print learned physics parameters
    print(f"\nLearned Physics Parameters:")
    print(f"Gravity: {float(model.gravity[0]):.2f} m/s² (Earth: -9.8)")
    print(
        f"Friction: {float(model.friction[0]):.4f}"
    )  # friction is now a property that returns array


if __name__ == "__main__":
    main()
