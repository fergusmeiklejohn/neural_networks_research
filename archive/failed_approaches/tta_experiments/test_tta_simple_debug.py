"""Simple test to understand TTA convergence issue."""

import os

os.environ["KERAS_BACKEND"] = "jax"

import sys
from pathlib import Path

# Minimal imports
import numpy as np

sys.path.append(str(Path(__file__).parent.parent.parent))


def simple_test():
    """Run simple TTA test to see what's happening."""
    print("Simple TTA Debug Test")
    print("=" * 50)

    try:
        import keras

        print(f"Keras backend: {keras.backend.backend()}")

        # Create simple model
        model = keras.Sequential(
            [
                keras.layers.Input(shape=(8,)),
                keras.layers.Dense(32, activation="relu"),
                keras.layers.BatchNormalization(),
                keras.layers.Dense(16, activation="relu"),
                keras.layers.BatchNormalization(),
                keras.layers.Dense(10),
            ]
        )

        model.compile(optimizer="adam", loss="mse")
        print("Model created successfully")

        # Create dummy data
        X = np.random.randn(5, 8).astype(np.float32)
        y = np.random.randn(5, 10).astype(np.float32)

        # Quick train
        print("\nTraining...")
        model.fit(X, y, epochs=5, verbose=0)

        # Test prediction
        test_X = np.random.randn(1, 8).astype(np.float32)

        # Get predictions at different points
        print("\nTesting predictions...")
        pred1 = model.predict(test_X, verbose=0)
        print(f"Prediction 1 mean: {np.mean(pred1):.6f}")

        # Import TTA
        from models.test_time_adaptation.tta_wrappers import TTAWrapper

        # Test different TTA methods
        methods = ["tent", "physics_tent", "ttt"]

        for method in methods:
            print(f"\n{method.upper()} Test:")

            # Reset model weights
            model_copy = keras.models.clone_model(model)
            model_copy.set_weights(model.get_weights())
            model_copy.compile(optimizer="adam", loss="mse")

            # Create TTA wrapper
            tta = TTAWrapper(
                model_copy, tta_method=method, adaptation_steps=5, learning_rate=1e-3
            )

            # Get adapted prediction
            pred_adapted = tta.predict(test_X.reshape(1, 1, 8), adapt=True)
            print(f"  Adapted prediction mean: {np.mean(pred_adapted):.6f}")
            print(
                f"  Change from original: {np.mean(np.abs(pred_adapted - pred1)):.6f}"
            )

        # Test with higher learning rate
        print("\nTesting with higher learning rate (1e-2):")
        model_copy = keras.models.clone_model(model)
        model_copy.set_weights(model.get_weights())
        model_copy.compile(optimizer="adam", loss="mse")

        tta_high_lr = TTAWrapper(
            model_copy, tta_method="tent", adaptation_steps=5, learning_rate=1e-2
        )

        pred_high_lr = tta_high_lr.predict(test_X.reshape(1, 1, 8), adapt=True)
        print(f"  Prediction mean: {np.mean(pred_high_lr):.6f}")
        print(f"  Change: {np.mean(np.abs(pred_high_lr - pred1)):.6f}")

        # Check if BatchNorm is being updated
        print("\nChecking BatchNorm updates...")

        # Get initial BN stats
        bn_stats_before = []
        for layer in model_copy.layers:
            if isinstance(layer, keras.layers.BatchNormalization):
                bn_stats_before.append(
                    {
                        "mean": layer.moving_mean.numpy().copy(),
                        "var": layer.moving_variance.numpy().copy(),
                    }
                )

        # Adapt
        _ = tta_high_lr.predict(test_X.reshape(1, 1, 8), adapt=True)

        # Check BN stats after
        bn_changed = False
        for i, layer in enumerate(model_copy.layers):
            if isinstance(layer, keras.layers.BatchNormalization):
                if i < len(bn_stats_before):
                    mean_change = np.mean(
                        np.abs(layer.moving_mean.numpy() - bn_stats_before[i]["mean"])
                    )
                    var_change = np.mean(
                        np.abs(
                            layer.moving_variance.numpy() - bn_stats_before[i]["var"]
                        )
                    )

                    if mean_change > 1e-6 or var_change > 1e-6:
                        print(
                            f"  BN layer {i}: mean changed by {mean_change:.6f}, var by {var_change:.6f}"
                        )
                        bn_changed = True

        if not bn_changed:
            print("  WARNING: BatchNorm stats did not change!")

        print("\nConclusion:")
        print("If all predictions are similar, TTA is not working properly.")
        print("This could be due to:")
        print("1. Learning rate too small")
        print("2. Not enough adaptation steps")
        print("3. Single sample not providing enough signal")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    simple_test()
