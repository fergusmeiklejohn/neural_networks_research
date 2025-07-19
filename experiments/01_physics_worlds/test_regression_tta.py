"""Test the regression-specific TTA implementation."""

import os
os.environ['KERAS_BACKEND'] = 'jax'

import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))


def test_regression_tta():
    """Test TTA for regression tasks."""
    print("Testing Regression-Specific TTA")
    print("="*60)
    
    try:
        import keras
        from models.test_time_adaptation.tta_wrappers import TTAWrapper
        
        # Create a regression model
        model = keras.Sequential([
            keras.layers.Input(shape=(1, 8)),
            keras.layers.Flatten(),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(10 * 8),
            keras.layers.Reshape((10, 8))
        ])
        
        model.compile(optimizer='adam', loss='mse')
        print("Model created successfully")
        
        # Create dummy training data
        X_train = np.random.randn(50, 1, 8).astype(np.float32)
        y_train = np.random.randn(50, 10, 8).astype(np.float32)
        
        # Quick training
        print("\nTraining model...")
        model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=0)
        
        # Test data
        X_test = np.random.randn(5, 1, 8).astype(np.float32)
        
        # Baseline predictions
        print("\nBaseline predictions:")
        y_baseline = model.predict(X_test, verbose=0)
        print(f"  Shape: {y_baseline.shape}")
        print(f"  Mean: {np.mean(y_baseline):.4f}")
        print(f"  Std: {np.std(y_baseline):.4f}")
        
        # Test different configurations
        configs = [
            {'learning_rate': 1e-4, 'adaptation_steps': 5},
            {'learning_rate': 1e-3, 'adaptation_steps': 10},
            {'learning_rate': 1e-2, 'adaptation_steps': 10},
            {'learning_rate': 1e-2, 'adaptation_steps': 20},
        ]
        
        print("\nTesting TTA configurations:")
        print("-" * 60)
        
        for config in configs:
            print(f"\nConfig: lr={config['learning_rate']}, steps={config['adaptation_steps']}")
            
            # Test TENT
            tta_tent = TTAWrapper(
                model,
                tta_method='tent',
                task_type='regression',
                **config
            )
            
            # Reset to original weights
            tta_tent.reset()
            
            # Adapt and predict
            y_tent = tta_tent.predict(X_test[0:1], adapt=True)
            
            change = np.mean(np.abs(y_tent - y_baseline[0:1]))
            print(f"  TENT change: {change:.6f}")
            
            # Test PhysicsTENT
            tta_physics = TTAWrapper(
                model,
                tta_method='physics_tent',
                task_type='regression',
                physics_loss_weight=0.1,
                **config
            )
            
            tta_physics.reset()
            y_physics = tta_physics.predict(X_test[0:1], adapt=True)
            
            change_physics = np.mean(np.abs(y_physics - y_baseline[0:1]))
            print(f"  PhysicsTENT change: {change_physics:.6f}")
            
            # Check if adaptation is actually happening
            if change < 1e-6 and change_physics < 1e-6:
                print("  WARNING: No adaptation occurred!")
        
        # Test with multiple samples (batch adaptation)
        print("\n\nBatch adaptation test:")
        print("-" * 60)
        
        tta_batch = TTAWrapper(
            model,
            tta_method='physics_tent',
            task_type='regression',
            learning_rate=1e-2,
            adaptation_steps=20,
            physics_loss_weight=0.1
        )
        
        # Adapt on batch
        y_batch = tta_batch.predict(X_test, adapt=True)
        
        batch_changes = [np.mean(np.abs(y_batch[i] - y_baseline[i])) for i in range(len(X_test))]
        print(f"Average change per sample: {np.mean(batch_changes):.6f}")
        print(f"Max change: {np.max(batch_changes):.6f}")
        
        # Conclusion
        print("\n" + "="*60)
        print("CONCLUSION")
        print("="*60)
        
        if np.max(batch_changes) > 0.01:
            print("✓ TTA is working! Predictions are being adapted.")
            print("  Higher learning rates and more steps = more adaptation")
        else:
            print("✗ TTA is not working effectively.")
            print("  Possible issues:")
            print("  1. Learning rate still too small")
            print("  2. Loss function not appropriate")
            print("  3. Model architecture limitations")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_regression_tta()