"""Simple test focusing on JAX TTA core functionality."""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.imports import setup_project_paths
setup_project_paths()

from utils.config import setup_environment
import keras


def test_basic_jax_tta():
    """Test basic TTA functionality with JAX backend."""
    print("Testing Basic JAX TTA")
    print("="*60)
    
    # Check backend
    backend = keras.backend.backend()
    print(f"Keras backend: {backend}")
    
    # Create simple model
    model = keras.Sequential([
        keras.layers.Input(shape=(4,)),
        keras.layers.Dense(8, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(4)
    ])
    model.compile(optimizer='adam', loss='mse')
    print(f"Model created: {model.count_params()} parameters")
    
    # Test data
    X = np.random.randn(5, 4).astype(np.float32)
    y = np.random.randn(5, 4).astype(np.float32)
    
    # Train briefly
    print("\nTraining model...")
    model.fit(X, y, epochs=5, verbose=0)
    
    # Test predictions
    print("\nTesting predictions...")
    pred_before = model.predict(X, verbose=0)
    print(f"Predictions shape: {pred_before.shape}")
    print(f"MSE before adaptation: {np.mean((pred_before - y)**2):.4f}")
    
    # Test BatchNorm adaptation (simple version)
    print("\nTesting BatchNorm adaptation...")
    # Running in training mode updates BN statistics
    _ = model(X, training=True)
    
    pred_after = model.predict(X, verbose=0)
    print(f"MSE after BN update: {np.mean((pred_after - y)**2):.4f}")
    
    # Test with our TTA wrapper
    print("\nTesting TTA wrapper...")
    from models.test_time_adaptation.tta_wrappers import TTAWrapper
    
    try:
        # Reset model
        model = keras.Sequential([
            keras.layers.Input(shape=(4,)),
            keras.layers.Dense(8, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(4)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X, y, epochs=5, verbose=0)
        
        # Create TTA wrapper
        tta_model = TTAWrapper(model, tta_method='tent', adaptation_steps=1, learning_rate=1e-4)
        
        # Test adaptation
        X_test = np.random.randn(3, 4).astype(np.float32)
        pred_tta = tta_model.predict(X_test, adapt=True)
        print(f"TTA predictions shape: {pred_tta.shape}")
        print("✓ TTA wrapper works!")
        
    except Exception as e:
        print(f"✗ TTA wrapper error: {e}")
    
    print("\nDone!")


def test_entropy_minimization():
    """Test entropy minimization directly."""
    print("\n\nTesting Entropy Minimization")
    print("="*60)
    
    # Create classification model
    model = keras.Sequential([
        keras.layers.Input(shape=(10,)),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(3, activation='softmax')  # 3-class
    ])
    
    # Test data - ambiguous predictions
    X_test = np.random.randn(5, 10).astype(np.float32)
    
    # Get initial predictions
    pred_initial = model(X_test, training=False)
    print(f"Initial predictions:\n{np.array(pred_initial)}")
    
    # Compute entropy
    entropy = -np.sum(pred_initial * np.log(pred_initial + 1e-8), axis=-1)
    print(f"Initial entropy: {entropy.mean():.4f}")
    
    # Run through training mode to update BN
    for _ in range(5):
        _ = model(X_test, training=True)
    
    # Get updated predictions
    pred_updated = model(X_test, training=False)
    entropy_updated = -np.sum(pred_updated * np.log(pred_updated + 1e-8), axis=-1)
    print(f"Updated entropy: {entropy_updated.mean():.4f}")
    print(f"Entropy reduction: {(entropy.mean() - entropy_updated.mean()):.4f}")


def main():
    """Run tests."""
    config = setup_environment()
    
    test_basic_jax_tta()
    test_entropy_minimization()
    
    print("\n" + "="*60)
    print("JAX TTA testing complete!")


if __name__ == "__main__":
    main()