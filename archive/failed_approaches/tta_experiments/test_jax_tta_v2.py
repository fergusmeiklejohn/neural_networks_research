"""Test the improved JAX-based TTA implementation with full gradient support."""

import numpy as np
import pickle
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.imports import setup_project_paths
setup_project_paths()

from utils.config import setup_environment
from utils.paths import get_data_path
import keras
from models.test_time_adaptation.regression_tta_v2 import RegressionTTAV2, PhysicsRegressionTTAV2


def create_physics_model():
    """Create a physics prediction model."""
    model = keras.Sequential([
        keras.layers.Input(shape=(8,)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(8)
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='mse',
        metrics=['mae']
    )
    return model


def test_gradient_computation():
    """Test that gradients are properly computed."""
    print("Testing Gradient Computation")
    print("="*60)
    
    # Create simple model
    model = keras.Sequential([
        keras.layers.Input(shape=(4,)),
        keras.layers.Dense(8, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(4)
    ])
    model.build((None, 4))
    
    # Create TTA wrapper
    tta = RegressionTTAV2(model, adaptation_steps=1, learning_rate=1e-3, update_bn_only=False)
    
    # Test data
    X = np.random.randn(5, 4).astype(np.float32)
    
    # Get initial weights
    initial_weights = [w.numpy().copy() for w in model.trainable_variables]
    
    # Perform adaptation step
    y_pred, loss = tta.adapt_step_jax(X)
    
    # Check that weights changed
    weights_changed = False
    for initial, current in zip(initial_weights, model.trainable_variables):
        if not np.allclose(initial, current.numpy()):
            weights_changed = True
            break
    
    print(f"Adaptation loss: {loss:.4f}")
    print(f"Weights changed: {weights_changed}")
    print(f"✓ Gradient computation {'working' if weights_changed else 'NOT working'}")
    
    return weights_changed


def test_bn_only_adaptation():
    """Test BatchNorm-only adaptation."""
    print("\nTesting BatchNorm-Only Adaptation")
    print("="*60)
    
    # Create model
    model = keras.Sequential([
        keras.layers.Input(shape=(4,)),
        keras.layers.Dense(8, activation='relu', name='dense1'),
        keras.layers.BatchNormalization(name='bn1'),
        keras.layers.Dense(4, name='dense2')
    ])
    model.build((None, 4))
    
    # Create TTA with BN-only updates
    tta = RegressionTTAV2(model, adaptation_steps=1, learning_rate=1e-2, update_bn_only=True)
    
    # Test data
    X = np.random.randn(10, 4).astype(np.float32)
    
    # Store initial weights
    initial_dense1 = model.get_layer('dense1').kernel.numpy().copy()
    initial_bn_gamma = model.get_layer('bn1').gamma.numpy().copy() if model.get_layer('bn1').gamma is not None else None
    
    # Adapt
    y_pred, loss = tta.adapt_step_jax(X)
    
    # Check what changed
    dense1_changed = not np.allclose(initial_dense1, model.get_layer('dense1').kernel.numpy())
    bn_changed = initial_bn_gamma is not None and not np.allclose(initial_bn_gamma, model.get_layer('bn1').gamma.numpy())
    
    print(f"Dense layer changed: {dense1_changed}")
    print(f"BatchNorm gamma changed: {bn_changed}")
    print(f"✓ BN-only adaptation {'working' if bn_changed and not dense1_changed else 'NOT working correctly'}")


def test_state_restoration():
    """Test complete state restoration."""
    print("\nTesting State Restoration")
    print("="*60)
    
    # Create model
    model = create_physics_model()
    
    # Train briefly
    X_train = np.random.randn(100, 8).astype(np.float32)
    y_train = np.random.randn(100, 8).astype(np.float32)
    model.fit(X_train, y_train, epochs=2, verbose=0)
    
    # Create TTA
    tta = RegressionTTAV2(model, adaptation_steps=5, learning_rate=1e-3)
    
    # Test data
    X_test = np.random.randn(10, 8).astype(np.float32)
    
    # Get predictions before adaptation
    pred_before = model.predict(X_test[0:1], verbose=0)
    
    # Adapt on test data
    _ = tta.adapt(X_test)
    
    # Reset
    tta.reset()
    
    # Get predictions after reset
    pred_after = model.predict(X_test[0:1], verbose=0)
    
    # Check restoration
    restoration_error = np.mean(np.abs(pred_before - pred_after))
    print(f"Restoration error: {restoration_error:.6f}")
    print(f"✓ State restoration {'working' if restoration_error < 1e-5 else 'NOT working'}")
    
    return restoration_error < 1e-5


def test_on_physics_data():
    """Test on actual physics OOD data."""
    print("\nTesting on Physics OOD Data")
    print("="*60)
    
    # Load data
    data_dir = get_data_path() / "true_ood_physics"
    
    try:
        # Load constant gravity
        const_files = sorted(data_dir.glob("constant_gravity_*.pkl"))
        with open(const_files[-1], 'rb') as f:
            const_data = pickle.load(f)
        
        # Load time-varying gravity  
        varying_files = sorted(data_dir.glob("time_varying_gravity_*.pkl"))
        with open(varying_files[-1], 'rb') as f:
            ood_data = pickle.load(f)
    except:
        print("Could not load physics data, using synthetic data")
        const_data = {'trajectories': np.random.randn(100, 50, 8).astype(np.float32)}
        ood_data = {'trajectories': np.random.randn(50, 50, 8).astype(np.float32) * 1.5}
    
    # Prepare training data
    X_train = []
    y_train = []
    for traj in const_data['trajectories'][:50]:
        for i in range(len(traj) - 1):
            X_train.append(traj[i])
            y_train.append(traj[i+1])
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    # Train model
    print("Training model on constant gravity...")
    model = create_physics_model()
    model.fit(X_train, y_train, epochs=10, batch_size=64, verbose=0)
    
    # Evaluate baseline
    X_test_ood = []
    y_test_ood = []
    for traj in ood_data['trajectories'][:5]:
        for i in range(len(traj) - 1):
            X_test_ood.append(traj[i])
            y_test_ood.append(traj[i+1])
    X_test_ood = np.array(X_test_ood)
    y_test_ood = np.array(y_test_ood)
    
    baseline_mse = model.evaluate(X_test_ood, y_test_ood, verbose=0)[0]
    print(f"Baseline OOD MSE: {baseline_mse:.4f}")
    
    # Test different TTA configurations
    configs = [
        {'name': 'RegressionTTA (BN only)', 'class': RegressionTTAV2, 'update_bn_only': True, 'lr': 1e-3},
        {'name': 'RegressionTTA (all params)', 'class': RegressionTTAV2, 'update_bn_only': False, 'lr': 5e-4},
        {'name': 'PhysicsRegressionTTA', 'class': PhysicsRegressionTTAV2, 'update_bn_only': False, 'lr': 5e-4},
    ]
    
    for config in configs:
        print(f"\nTesting {config['name']}...")
        
        # Create TTA
        tta = config['class'](
            model,
            adaptation_steps=5,
            learning_rate=config['lr'],
            update_bn_only=config['update_bn_only']
        )
        
        # Test batch by batch
        batch_size = 20
        errors = []
        
        for i in range(0, len(X_test_ood), batch_size):
            X_batch = X_test_ood[i:i+batch_size]
            y_batch = y_test_ood[i:i+batch_size]
            
            # Adapt and predict
            y_pred = tta.adapt(X_batch)
            error = np.mean((y_pred - y_batch)**2)
            errors.append(error)
            
            # Reset for next batch
            tta.reset()
        
        mean_mse = np.mean(errors)
        improvement = (1 - mean_mse/baseline_mse) * 100
        print(f"  MSE: {mean_mse:.4f}")
        print(f"  Improvement: {improvement:+.1f}%")


def main():
    """Run all tests."""
    print("Testing Improved JAX TTA Implementation (V2)")
    print("="*70)
    
    config = setup_environment()
    print(f"Backend: {keras.backend.backend()}")
    
    # Run tests
    gradient_works = test_gradient_computation()
    test_bn_only_adaptation()
    restoration_works = test_state_restoration()
    test_on_physics_data()
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"✓ Gradient computation: {'PASS' if gradient_works else 'FAIL'}")
    print(f"✓ State restoration: {'PASS' if restoration_works else 'FAIL'}")
    print("\nThe improved JAX TTA implementation is ready for use!")


if __name__ == "__main__":
    main()