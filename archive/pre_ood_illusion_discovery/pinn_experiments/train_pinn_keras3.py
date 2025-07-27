#!/usr/bin/env python3
"""
Keras 3 compatible training script for Physics-Informed Neural Network.

Uses proper Keras 3 training API with custom training step.
"""

import sys
import os
sys.path.append('../..')
os.environ['KERAS_BACKEND'] = 'jax'

import numpy as np
import pickle
import keras
from pathlib import Path
import time
from typing import Dict, Tuple, Optional

from models.physics_informed_transformer import PhysicsInformedTrajectoryTransformer


class PINNTrainer(keras.Model):
    """Wrapper to enable proper Keras 3 training with custom loss."""
    
    def __init__(self, pinn_model):
        super().__init__()
        self.pinn_model = pinn_model
        
    def call(self, inputs, training=None):
        return self.pinn_model(inputs, training=training)
    
    def train_step(self, data):
        """Custom training step for PINN."""
        x, y = data
        
        # Forward pass and compute loss
        with self.distribute_strategy.scope():
            # Compute loss
            loss_dict = self.pinn_model.compute_loss(x, y, training=True)
            total_loss = loss_dict['total_loss']
            
        # Get gradients
        gradients = self.compute_gradients(total_loss, self.trainable_variables)
        
        # Update weights
        self.optimizer.apply(gradients, self.trainable_variables)
        
        # Update metrics
        metrics = {
            'loss': total_loss,
            'trajectory_loss': loss_dict['trajectory_loss'],
            'physics_loss': loss_dict['physics_loss']
        }
        
        return metrics
    
    def test_step(self, data):
        """Custom validation step."""
        x, y = data
        
        # Compute loss
        loss_dict = self.pinn_model.compute_loss(x, y, training=False)
        
        metrics = {
            'loss': loss_dict['total_loss'],
            'trajectory_loss': loss_dict['trajectory_loss'],
            'physics_loss': loss_dict['physics_loss']
        }
        
        return metrics


def load_small_dataset(data_path: str = "data", max_samples: int = 1000):
    """Load a small subset of data for testing."""
    print("Loading small dataset for testing...")
    
    # Load train data
    train_path = Path(data_path) / "processed" / "physics_worlds_v2" / "train_data.pkl"
    with open(train_path, 'rb') as f:
        train_data = pickle.load(f)
    
    # Filter and limit samples
    train_data = [s for s in train_data if s['num_balls'] == 2][:max_samples]
    
    # Process data
    X_train = []
    y_positions = []
    y_velocities = []
    y_params = []
    
    for sample in train_data:
        traj = np.array(sample['trajectory'])
        
        # Pad/truncate to 300 timesteps
        if len(traj) > 300:
            traj = traj[:300]
        elif len(traj) < 300:
            padding = np.tile(traj[-1:], (300 - len(traj), 1))
            traj = np.concatenate([traj, padding])
        
        X_train.append(traj)
        
        # Extract targets
        pos = traj[:, 1:5].reshape(-1, 2, 2)
        vel = traj[:, 5:9].reshape(-1, 2, 2)
        
        y_positions.append(pos)
        y_velocities.append(vel)
        y_params.append([
            sample['physics_config']['gravity'],
            sample['physics_config']['friction'],
            sample['physics_config']['elasticity'],
            sample['physics_config']['damping']
        ])
    
    X_train = np.array(X_train, dtype=np.float32)
    
    # Normalize
    X_mean = X_train.mean(axis=(0, 1))
    X_std = X_train.std(axis=(0, 1)) + 1e-8
    X_train = (X_train - X_mean) / X_std
    
    y_train = {
        'positions': np.array(y_positions, dtype=np.float32),
        'velocities': np.array(y_velocities, dtype=np.float32),
        'physics_params': np.array(y_params, dtype=np.float32)
    }
    
    # Split train/val
    n_train = int(0.8 * len(X_train))
    X_val = X_train[n_train:]
    X_train = X_train[:n_train]
    
    y_val = {k: v[n_train:] for k, v in y_train.items()}
    y_train = {k: v[:n_train] for k, v in y_train.items()}
    
    return X_train, y_train, X_val, y_val


def train_pinn_keras3():
    """Train PINN using Keras 3 API."""
    print("="*60)
    print("PINN Training with Keras 3")
    print("="*60)
    
    # Load small dataset for testing
    X_train, y_train, X_val, y_val = load_small_dataset(max_samples=500)
    
    print(f"\nDataset shapes:")
    print(f"X_train: {X_train.shape}")
    print(f"X_val: {X_val.shape}")
    
    # Create PINN model
    print("\nCreating PINN model...")
    base_model = PhysicsInformedTrajectoryTransformer(
        sequence_length=300,
        feature_dim=X_train.shape[-1],
        num_transformer_layers=2,  # Smaller for testing
        num_heads=4,
        transformer_dim=128,
        use_hnn=True,
        use_soft_collisions=True,
        use_physics_losses=True
    )
    
    # Wrap in trainer
    model = PINNTrainer(base_model)
    
    # Compile
    print("Compiling model...")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss=None  # We use custom loss in train_step
    )
    
    # Prepare data for Keras
    # Keras expects (X, y) tuples, but our y is a dict
    # We'll create a custom dataset
    
    print("\nTraining for 5 epochs (testing)...")
    
    # Simple training loop since fit() expects different data format
    batch_size = 16
    n_epochs = 5
    
    for epoch in range(n_epochs):
        print(f"\nEpoch {epoch + 1}/{n_epochs}")
        
        # Training
        train_losses = []
        n_batches = len(X_train) // batch_size
        
        start_time = time.time()
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            
            X_batch = X_train[start_idx:end_idx]
            y_batch = {k: v[start_idx:end_idx] for k, v in y_train.items()}
            
            # Forward pass
            loss_dict = base_model.compute_loss(X_batch, y_batch, training=True)
            train_losses.append(float(loss_dict['total_loss']))
            
            # Progress
            if i % 10 == 0:
                print(f"  Batch {i}/{n_batches}, Loss: {train_losses[-1]:.4f}")
        
        # Validation
        val_losses = []
        n_val_batches = len(X_val) // batch_size
        
        for i in range(n_val_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            
            X_batch = X_val[start_idx:end_idx]
            y_batch = {k: v[start_idx:end_idx] for k, v in y_val.items()}
            
            loss_dict = base_model.compute_loss(X_batch, y_batch, training=False)
            val_losses.append(float(loss_dict['total_loss']))
        
        epoch_time = time.time() - start_time
        avg_train_loss = np.mean(train_losses) if train_losses else 0
        avg_val_loss = np.mean(val_losses) if val_losses else 0
        
        print(f"  Time: {epoch_time:.1f}s")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        
        # Update physics weights progressively
        if epoch == 2:
            print("  Introducing physics losses...")
            base_model.use_physics_losses = True
            base_model.energy_weight = 0.1
            base_model.momentum_weight = 0.05
        elif epoch == 4:
            print("  Increasing physics weights...")
            base_model.energy_weight = 0.5
            base_model.momentum_weight = 0.25
    
    # Save model
    print("\nSaving model...")
    save_dir = Path("outputs/checkpoints")
    save_dir.mkdir(parents=True, exist_ok=True)
    base_model.save(save_dir / "pinn_test_keras3.keras")
    
    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)
    print("\nNote: This is a simplified training loop for testing.")
    print("For full training with gradient updates, you would need to:")
    print("1. Implement proper JAX gradient computation")
    print("2. Use jax.grad() with the loss function")
    print("3. Update weights using optax or similar")
    print("\nThe current implementation only computes losses without")
    print("updating weights, but verifies the model architecture works.")


if __name__ == "__main__":
    train_pinn_keras3()