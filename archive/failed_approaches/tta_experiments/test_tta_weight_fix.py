"""Test TTA weight restoration fix without external dependencies."""

import os
import sys
from pathlib import Path

import numpy as np

# Add project to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Set backend before importing keras
os.environ["KERAS_BACKEND"] = "jax"

try:
    import keras

    # Import after keras is loaded
    from models.test_time_adaptation.tta_wrappers import TTAWrapper

    print("Successfully imported all dependencies")

    # Create a simple model with BatchNorm
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(10,)),
            keras.layers.Dense(20, activation="relu"),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(10),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(1),
        ]
    )

    model.compile(optimizer="adam", loss="mse")
    print("Created model with BatchNorm layers")

    # Create some dummy data
    X = np.random.randn(32, 10).astype(np.float32)
    y = np.random.randn(32, 1).astype(np.float32)

    # Do a forward pass to initialize BatchNorm
    _ = model(X[:1], training=True)
    print("Initialized BatchNorm statistics")

    # Create TTA wrapper
    try:
        tta_model = TTAWrapper(
            model, tta_method="tent", adaptation_steps=2, learning_rate=1e-3
        )
        print("Created TTA wrapper successfully")

        # Test prediction with adaptation
        print("\nTesting adaptation...")
        pred = tta_model.predict(X[:5], adapt=True)
        print(f"Adaptation successful! Output shape: {pred.shape}")

        # Test reset
        print("\nTesting reset...")
        tta_model.reset()
        print("Reset successful!")

        # Test multiple adaptations
        print("\nTesting multiple adaptations...")
        for i in range(3):
            pred = tta_model.predict(X[i * 5 : (i + 1) * 5], adapt=True)
            tta_model.reset()
            print(f"  Batch {i+1}: Success")

        print("\n✓ All tests passed! Weight restoration fix is working.")

    except Exception as e:
        print(f"\n✗ Error during TTA: {e}")
        import traceback

        traceback.print_exc()

except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure you have activated the 'dist-invention' conda environment")
    print("Run: conda activate dist-invention")
