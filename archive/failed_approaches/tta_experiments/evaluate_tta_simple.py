"""Simplified TTA evaluation on true OOD physics data."""

import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

# Add project to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Set backend
os.environ["KERAS_BACKEND"] = "jax"


def main():
    try:
        # Import after setting backend
        import keras

        from models.test_time_adaptation.tta_wrappers import TTAWrapper

        print("TTA Evaluation on True OOD Physics")
        print("=" * 60)

        # Load data
        data_dir = Path(__file__).parent.parent.parent / "data" / "true_ood_physics"

        # Find most recent files
        const_files = sorted(data_dir.glob("constant_gravity_*.pkl"))
        varying_files = sorted(data_dir.glob("time_varying_gravity_*.pkl"))

        if not const_files or not varying_files:
            print("Error: Could not find data files")
            print(f"Looking in: {data_dir}")
            return

        print(f"\nLoading data from {data_dir}")

        # Load constant gravity (training data)
        with open(const_files[-1], "rb") as f:
            const_data = pickle.load(f)
        print(f"Loaded constant gravity: {const_data['trajectories'].shape}")

        # Load time-varying gravity (OOD test)
        with open(varying_files[-1], "rb") as f:
            varying_data = pickle.load(f)
        print(f"Loaded time-varying gravity: {varying_data['trajectories'].shape}")

        # Create simple model
        print("\nCreating simple physics model...")
        model = keras.Sequential(
            [
                keras.layers.Input(shape=(1, 8)),
                keras.layers.Flatten(),
                keras.layers.Dense(64, activation="relu"),
                keras.layers.BatchNormalization(),
                keras.layers.Dense(32, activation="relu"),
                keras.layers.BatchNormalization(),
                keras.layers.Dense(8 * 10),  # Predict next 10 timesteps
                keras.layers.Reshape((10, 8)),
            ]
        )

        model.compile(optimizer="adam", loss="mse", metrics=["mae"])

        # Quick training
        print("\nTraining on constant gravity...")
        X_train = const_data["trajectories"][:80, 0:1]
        y_train = const_data["trajectories"][:80, 1:11]  # Next 10 steps

        history = model.fit(
            X_train, y_train, epochs=20, batch_size=16, verbose=0, validation_split=0.2
        )
        print(f"Final training loss: {history.history['loss'][-1]:.4f}")

        # Evaluate on test sets
        def evaluate_model(model, trajectories, name, adapt=False, tta_method=None):
            """Evaluate model on trajectories."""
            errors = []
            times = []

            # Use TTA wrapper if needed
            if adapt:
                model = TTAWrapper(
                    model, tta_method=tta_method, adaptation_steps=5, learning_rate=1e-4
                )

            # Test on subset for speed
            for i, traj in enumerate(trajectories[:10]):
                start_time = time.time()

                X = traj[0:1].reshape(1, 1, 8)
                y_true = traj[1:11]

                if adapt:
                    y_pred = model.predict(X, adapt=True)
                    model.reset()
                else:
                    y_pred = model.predict(X, verbose=0)

                elapsed = time.time() - start_time
                times.append(elapsed)

                # Compute error
                y_pred = y_pred[0] if len(y_pred.shape) == 3 else y_pred
                mse = np.mean((y_true - y_pred[: len(y_true)]) ** 2)
                errors.append(mse)

            avg_mse = np.mean(errors)
            avg_time = np.mean(times)

            print(f"\n{name}:")
            print(f"  MSE: {avg_mse:.4f} (±{np.std(errors):.4f})")
            print(f"  Time: {avg_time:.3f}s per sample")

            return avg_mse, avg_time

        # Run evaluations
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)

        # Baseline on constant gravity
        const_mse, _ = evaluate_model(
            model, const_data["trajectories"][80:], "Constant Gravity (baseline)"
        )

        # Time-varying without TTA
        varying_mse, _ = evaluate_model(
            model, varying_data["trajectories"], "Time-Varying Gravity (no TTA)"
        )

        degradation = (varying_mse - const_mse) / const_mse * 100
        print(f"\nDegradation on OOD: {degradation:.1f}%")

        # Time-varying with TENT
        tent_mse, tent_time = evaluate_model(
            model,
            varying_data["trajectories"],
            "Time-Varying Gravity (TENT)",
            adapt=True,
            tta_method="tent",
        )

        tent_improvement = (varying_mse - tent_mse) / varying_mse * 100
        print(f"TENT improvement: {tent_improvement:.1f}%")

        # Time-varying with PhysicsTENT
        physics_tent_mse, physics_tent_time = evaluate_model(
            model,
            varying_data["trajectories"],
            "Time-Varying Gravity (PhysicsTENT)",
            adapt=True,
            tta_method="physics_tent",
        )

        physics_tent_improvement = (varying_mse - physics_tent_mse) / varying_mse * 100
        print(f"PhysicsTENT improvement: {physics_tent_improvement:.1f}%")

        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Baseline (constant gravity): {const_mse:.4f}")
        print(f"OOD (no adaptation): {varying_mse:.4f} ({degradation:.1f}% worse)")
        print(f"OOD (TENT): {tent_mse:.4f} ({tent_improvement:.1f}% improvement)")
        print(
            f"OOD (PhysicsTENT): {physics_tent_mse:.4f} ({physics_tent_improvement:.1f}% improvement)"
        )

        if tent_improvement > 0 or physics_tent_improvement > 0:
            print("\n✓ TTA successfully improves performance on true OOD scenarios!")
        else:
            print("\n✗ TTA did not improve performance in this test")

    except ImportError as e:
        print(f"Import error: {e}")
        print("\nPlease activate the conda environment:")
        print("  conda activate dist-invention")
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
