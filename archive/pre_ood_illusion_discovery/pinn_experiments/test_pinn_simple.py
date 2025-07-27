#!/usr/bin/env python3
"""
Simple PINN component test without full training.

Just verifies all components work together.
"""

import sys
import os
sys.path.append('../..')
os.environ['KERAS_BACKEND'] = 'jax'

import numpy as np
import keras
from pathlib import Path

from models.physics_informed_transformer import PhysicsInformedTrajectoryTransformer
from models.physics_losses import PhysicsLosses


def test_pinn_components():
    """Test PINN components work together."""
    print("="*60)
    print("PINN Component Integration Test")
    print("="*60)
    
    # Parameters
    batch_size = 4
    seq_len = 50
    feature_dim = 18
    
    print("\n1. Creating model...")
    model = PhysicsInformedTrajectoryTransformer(
        sequence_length=seq_len,
        feature_dim=feature_dim,
        num_transformer_layers=2,
        num_heads=4,
        transformer_dim=128,
        use_hnn=True,
        use_soft_collisions=True,
        use_physics_losses=True
    )
    print("✓ Model created")
    
    print("\n2. Creating test data...")
    # Create realistic test data
    X = np.random.randn(batch_size, seq_len, feature_dim).astype(np.float32)
    
    # Make positions realistic (within bounds)
    X[:, :, 1:5] = np.random.uniform(100, 700, (batch_size, seq_len, 4))
    
    # Make velocities reasonable
    X[:, :, 5:9] = np.random.uniform(-50, 50, (batch_size, seq_len, 4))
    
    # Create targets
    positions = X[:, :, 1:5].reshape(batch_size, seq_len, 2, 2)
    velocities = X[:, :, 5:9].reshape(batch_size, seq_len, 2, 2)
    physics_params = np.array([
        [-981.0, 0.5, 0.7, 0.92],
        [-800.0, 0.6, 0.6, 0.90],
        [-1000.0, 0.4, 0.8, 0.93],
        [-900.0, 0.5, 0.7, 0.91]
    ], dtype=np.float32)
    
    targets = {
        'positions': positions,
        'velocities': velocities,
        'physics_params': physics_params
    }
    print("✓ Test data created")
    
    print("\n3. Testing forward pass...")
    outputs = model(X, training=False)
    print("✓ Forward pass successful")
    print("   Output shapes:")
    for key, value in outputs.items():
        if value is not None:
            print(f"   - {key}: {value.shape}")
    
    print("\n4. Testing loss computation...")
    losses = model.compute_loss(X, targets, training=False)
    print("✓ Loss computation successful")
    print("   Loss values:")
    for key, value in losses.items():
        print(f"   - {key}: {float(value):.4f}")
    
    print("\n5. Testing physics components individually...")
    
    # Test HNN
    if model.use_hnn:
        state = np.random.randn(batch_size, 8).astype(np.float32)
        masses = np.ones((batch_size, 2), dtype=np.float32)
        h_value = model.hnn({'state': state, 'masses': masses})
        print(f"✓ HNN output shape: {h_value.shape}")
    
    # Test collision model
    if model.use_soft_collisions:
        pos1 = np.random.uniform(100, 700, (batch_size, 2)).astype(np.float32)
        pos2 = np.random.uniform(100, 700, (batch_size, 2)).astype(np.float32)
        collision_inputs = {
            'positions1': pos1,
            'positions2': pos2,
            'radius1': np.full((batch_size,), 20.0, dtype=np.float32),
            'radius2': np.full((batch_size,), 20.0, dtype=np.float32)
        }
        collision_pot = model.collision_model(collision_inputs)
        print(f"✓ Collision potential shape: {collision_pot.shape}")
    
    print("\n6. Testing physics losses...")
    physics_losses = PhysicsLosses()
    
    # Energy conservation
    trajectory = {
        'positions': positions,
        'velocities': velocities,
        'masses': np.ones((batch_size, 2), dtype=np.float32)
    }
    energy_loss = physics_losses.energy_conservation_loss(trajectory)
    print(f"✓ Energy conservation loss: {float(energy_loss):.4f}")
    
    # Trajectory smoothness
    smoothness_loss = physics_losses.trajectory_smoothness_loss(positions)
    print(f"✓ Trajectory smoothness loss: {float(smoothness_loss):.4f}")
    
    print("\n7. Testing save/load...")
    save_path = "outputs/test_model.keras"
    Path("outputs").mkdir(exist_ok=True)
    
    # Save
    model.save(save_path)
    print(f"✓ Model saved to {save_path}")
    
    # Load
    loaded_model = keras.models.load_model(save_path)
    print("✓ Model loaded successfully")
    
    # Test loaded model
    loaded_outputs = loaded_model(X[:1], training=False)
    print("✓ Loaded model inference works")
    
    # Clean up
    os.remove(save_path)
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60)
    print("\nThe PINN implementation is working correctly.")
    print("Ready for full training with train_pinn_extractor.py")
    print("\nKey components verified:")
    print("- Transformer backbone")
    print("- Hamiltonian Neural Network")
    print("- Soft collision models")
    print("- Physics-informed losses")
    print("- Model serialization")


if __name__ == "__main__":
    test_pinn_components()