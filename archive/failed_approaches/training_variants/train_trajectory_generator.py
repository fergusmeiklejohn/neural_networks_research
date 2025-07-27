"""
Trajectory Generator Training - Reconstruct physics trajectories from rules
"""

import sys
import os
sys.path.append('../..')

import numpy as np
import pickle
import keras
from keras import layers, callbacks, ops
from tqdm import tqdm
from typing import Dict, List, Tuple

from models.core.trajectory_generator import TrajectoryConfig


class TrajectoryReconstructor(keras.Model):
    """Simplified trajectory generator for reconstruction training"""
    
    def __init__(self, config: TrajectoryConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        
        # Physics parameters input
        self.physics_projection = layers.Dense(64, activation='relu')
        
        # LSTM for sequence generation
        self.lstm1 = layers.LSTM(128, return_sequences=True)
        self.lstm2 = layers.LSTM(64, return_sequences=True)
        
        # Output trajectory features (positions, velocities for 2 balls)
        self.trajectory_output = layers.Dense(config.feature_dim, name='trajectory')
        
    def call(self, inputs, training=None):
        # inputs: physics_params tensor
        physics_params = inputs
        batch_size = ops.shape(physics_params)[0]
        seq_length = self.config.sequence_length
        
        # Project physics parameters
        physics_features = self.physics_projection(physics_params)
        
        # Expand physics features to sequence length
        physics_seq = ops.repeat(
            ops.expand_dims(physics_features, axis=1), 
            seq_length, 
            axis=1
        )
        
        # Generate trajectory sequence
        x = self.lstm1(physics_seq, training=training)
        x = self.lstm2(x, training=training)
        
        # Output trajectory features
        trajectory = self.trajectory_output(x, training=training)
        
        return trajectory


def load_and_prepare_trajectory_data(data_path: str, max_samples: int = None):
    """Load and prepare data for trajectory reconstruction"""
    print(f"Loading data from {data_path}...")
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    # Filter to 2-ball samples only
    filtered_data = [sample for sample in data if sample['num_balls'] == 2]
    
    if max_samples:
        filtered_data = filtered_data[:max_samples]
    
    print(f"Using {len(filtered_data)} samples with 2 balls")
    
    # Prepare arrays
    physics_params = []
    trajectories = []
    
    for sample in tqdm(filtered_data, desc="Processing"):
        traj = np.array(sample['trajectory'])
        
        # Use first 50 frames for training
        if len(traj) > 50:
            traj = traj[:50]
        elif len(traj) < 50:
            # Pad with last frame
            padding = np.tile(traj[-1:], (50 - len(traj), 1))
            traj = np.concatenate([traj, padding], axis=0)
        
        trajectories.append(traj)
        
        # Extract physics parameters
        physics = sample['physics_config']
        physics_params.append([
            physics['gravity'],
            physics['friction'], 
            physics['elasticity'],
            physics['damping']
        ])
    
    X_physics = np.array(physics_params)
    y_trajectories = np.array(trajectories)
    
    # Normalize physics parameters (same as rule extractor)
    X_physics_norm = normalize_physics_params(X_physics)
    
    # Normalize trajectories (z-score)
    y_traj_flat = y_trajectories.reshape(len(y_trajectories), -1)
    y_traj_norm = (y_traj_flat - y_traj_flat.mean()) / (y_traj_flat.std() + 1e-8)
    y_traj_norm = y_traj_norm.reshape(y_trajectories.shape)
    
    print(f"Data shapes: X_physics={X_physics_norm.shape}, y_trajectories={y_traj_norm.shape}")
    print(f"Physics range: {X_physics_norm.min():.3f} to {X_physics_norm.max():.3f}")
    print(f"Trajectory range: {y_traj_norm.min():.3f} to {y_traj_norm.max():.3f}")
    
    return X_physics_norm, y_traj_norm, X_physics, y_trajectories


def normalize_physics_params(params):
    """Normalize physics parameters to similar scales"""
    normalized = params.copy()
    
    # Gravity: scale to 0-1 range
    gravity_min, gravity_max = -1500, -200
    normalized[:, 0] = (params[:, 0] - gravity_min) / (gravity_max - gravity_min)
    
    # Friction: already 0-1, but ensure it
    normalized[:, 1] = np.clip(params[:, 1], 0, 1)
    
    # Elasticity: already 0-1, but ensure it  
    normalized[:, 2] = np.clip(params[:, 2], 0, 1)
    
    # Damping: already close to 0-1, normalize to 0-1
    damping_min, damping_max = 0.8, 1.0
    normalized[:, 3] = (params[:, 3] - damping_min) / (damping_max - damping_min)
    
    return normalized


def calculate_reconstruction_metrics(y_true, y_pred):
    """Calculate reconstruction quality metrics"""
    
    # MSE (main metric)
    mse = np.mean((y_pred - y_true) ** 2)
    
    # MAE
    mae = np.mean(np.abs(y_pred - y_true))
    
    # Position accuracy (first half of features are positions)
    pos_features = y_true.shape[-1] // 2
    pos_mse = np.mean((y_pred[:, :, :pos_features] - y_true[:, :, :pos_features]) ** 2)
    
    # Velocity accuracy (second half are velocities)
    vel_mse = np.mean((y_pred[:, :, pos_features:] - y_true[:, :, pos_features:]) ** 2)
    
    # Trajectory smoothness (check for sudden jumps)
    pred_diffs = np.diff(y_pred, axis=1)
    true_diffs = np.diff(y_true, axis=1)
    smoothness_error = np.mean((pred_diffs - true_diffs) ** 2)
    
    return {
        'mse': mse,
        'mae': mae,
        'position_mse': pos_mse,
        'velocity_mse': vel_mse,
        'smoothness_error': smoothness_error
    }


def train_trajectory_generator():
    """Train trajectory reconstruction model"""
    
    # Configuration
    config = TrajectoryConfig(
        sequence_length=50,
        feature_dim=17,  # 2-ball data has 17 features
        max_balls=2,
        learning_rate=1e-3
    )
    
    # Load data
    X_train, y_train, X_train_orig, y_train_orig = load_and_prepare_trajectory_data(
        'data/processed/physics_worlds/train_data.pkl', 
        max_samples=1000
    )
    X_val, y_val, X_val_orig, y_val_orig = load_and_prepare_trajectory_data(
        'data/processed/physics_worlds/val_data.pkl',
        max_samples=200
    )
    
    # Create model
    model = TrajectoryReconstructor(config)
    
    # Build model with dummy input
    dummy_physics = X_train[:1]
    _ = model(dummy_physics)
    
    print(f"Model parameters: {model.count_params():,}")
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss='mse',
        metrics=['mae']
    )
    
    # Data generator for training
    def data_generator(X_physics, y_traj, batch_size=32, shuffle=True):
        n_samples = len(X_physics)
        indices = np.arange(n_samples)
        
        while True:
            if shuffle:
                np.random.shuffle(indices)
            
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]
                
                batch_physics = X_physics[batch_indices]
                batch_traj = y_traj[batch_indices]
                
                # Input is just physics parameters
                yield batch_physics, batch_traj
    
    # Setup callbacks
    callbacks_list = [
        callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6)
    ]
    
    # Calculate steps
    batch_size = 32
    steps_per_epoch = len(X_train) // batch_size
    validation_steps = len(X_val) // batch_size
    
    # Create generators
    train_gen = data_generator(X_train, y_train, batch_size, shuffle=True)
    val_gen = data_generator(X_val, y_val, batch_size, shuffle=False)
    
    # Train
    print("Starting trajectory generation training...")
    history = model.fit(
        train_gen,
        epochs=100,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_gen,
        validation_steps=validation_steps,
        callbacks=callbacks_list,
        verbose=1
    )
    
    # Evaluate
    print("Evaluating trajectory reconstruction...")
    
    # Generate predictions
    y_pred = np.array(model(X_val))
    
    # Calculate metrics
    metrics = calculate_reconstruction_metrics(y_val, y_pred)
    
    print(f"\nTrajectory Reconstruction Results:")
    print(f"MSE: {metrics['mse']:.6f}")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"Position MSE: {metrics['position_mse']:.6f}")
    print(f"Velocity MSE: {metrics['velocity_mse']:.6f}")
    print(f"Smoothness Error: {metrics['smoothness_error']:.6f}")
    
    # Check if target achieved
    target_mse = 0.1
    achieved = metrics['mse'] <= target_mse
    
    print(f"\n{'='*60}")
    print(f"TRAJECTORY GENERATOR PRE-TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Target MSE: {target_mse:.1f}")
    print(f"Achieved MSE: {metrics['mse']:.4f}")
    print(f"Target achieved: {'✅ YES' if achieved else '❌ NO'}")
    
    if achieved:
        print(f"\n🎉 Trajectory generation pre-training successful!")
        
        # Save model
        os.makedirs('outputs/checkpoints', exist_ok=True)
        model.save('outputs/checkpoints/trajectory_generator.keras')
        print(f"Model saved to: outputs/checkpoints/trajectory_generator.keras")
    else:
        print(f"\n📈 Training completed but target not yet reached.")
        print(f"Best MSE so far: {metrics['mse']:.4f}")
    
    # Show some sample predictions
    print(f"\nSample trajectory predictions (first 3 time steps):")
    for i in range(min(3, len(y_val))):
        print(f"Sample {i}:")
        print(f"  True shape: {y_val[i][:3, :].shape}")
        print(f"  Pred shape: {y_pred[i][:3, :].shape}")
        print(f"  True first 3 features: {y_val[i][0, :3]}")
        print(f"  Pred first 3 features: {y_pred[i][0, :3]}")
    
    return model, history, metrics


if __name__ == "__main__":
    train_trajectory_generator()