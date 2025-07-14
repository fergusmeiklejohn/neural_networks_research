"""
Simplified Physics-Informed Neural Network for trajectory prediction.
Focuses on essential physics constraints for demonstrating causal understanding.
"""

import os
os.environ['KERAS_BACKEND'] = 'jax'

import keras
from keras import layers, ops
import numpy as np
from typing import Dict, Tuple, Optional


@keras.saving.register_keras_serializable()
class SimplePhysicsInformedModel(keras.Model):
    """Simplified PINN that learns physics rules for extrapolation.
    
    Key features:
    - Explicit gravity parameter prediction
    - Energy conservation loss
    - Momentum conservation loss
    - Lightweight architecture for fast training
    """
    
    def __init__(self,
                 sequence_length: int = 50,  # Shorter sequences for efficiency
                 hidden_dim: int = 128,
                 num_layers: int = 3,
                 use_physics_loss: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_physics_loss = use_physics_loss
        
        # Build model layers
        self._build_layers()
        
    def _build_layers(self):
        """Build model layers."""
        # Input processing
        self.input_norm = layers.LayerNormalization()
        
        # Feature extraction with LSTM
        self.lstm_layers = []
        for i in range(self.num_layers):
            self.lstm_layers.append(
                layers.LSTM(self.hidden_dim, return_sequences=True)
            )
            
        # Physics parameter prediction head
        self.physics_head = keras.Sequential([
            layers.GlobalAveragePooling1D(),
            layers.Dense(64, activation='relu'),
            layers.Dense(4)  # gravity, friction, elasticity, damping
        ])
        
        # Trajectory prediction head
        self.trajectory_head = keras.Sequential([
            layers.Dense(64, activation='relu'),
            layers.Dense(8)  # 2 balls * (x, y, vx, vy)
        ])
        
    def get_config(self):
        config = super().get_config()
        config.update({
            'sequence_length': self.sequence_length,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'use_physics_loss': self.use_physics_loss
        })
        return config
        
    def extract_physics_state(self, trajectory):
        """Extract position and velocity states from trajectory.
        
        Args:
            trajectory: Shape (batch, seq_len, 8) with [x1, y1, x2, y2, vx1, vy1, vx2, vy2]
            
        Returns:
            positions: Shape (batch, seq_len, 2, 2)
            velocities: Shape (batch, seq_len, 2, 2)
        """
        # Positions for both balls
        positions = ops.stack([
            trajectory[..., 0:2],  # Ball 1 position
            trajectory[..., 2:4]   # Ball 2 position
        ], axis=-2)
        
        # Velocities for both balls
        velocities = ops.stack([
            trajectory[..., 4:6],  # Ball 1 velocity
            trajectory[..., 6:8]   # Ball 2 velocity
        ], axis=-2)
        
        return positions, velocities
        
    def compute_energy(self, positions, velocities, gravity, mass=1.0):
        """Compute total energy of the system.
        
        Args:
            positions: Shape (batch, seq_len, 2, 2) - 2 balls, 2D positions
            velocities: Shape (batch, seq_len, 2, 2) - 2 balls, 2D velocities
            gravity: Scalar or shape (batch,)
            mass: Mass of each ball
            
        Returns:
            Total energy per timestep: Shape (batch, seq_len)
        """
        # Kinetic energy: 0.5 * m * v^2
        kinetic = 0.5 * mass * ops.sum(velocities**2, axis=[-2, -1])
        
        # Potential energy: m * g * h (where h is y-coordinate)
        # Note: gravity is negative, so we use -gravity
        heights = positions[..., 1]  # y-coordinates, shape (batch, seq_len, 2)
        total_height = ops.sum(heights, axis=-1)  # shape (batch, seq_len)
        
        # Expand gravity to match shape if needed
        if len(ops.shape(gravity)) == 1:  # shape (batch,)
            gravity = ops.expand_dims(gravity, axis=1)  # shape (batch, 1)
        
        potential = mass * (-gravity) * total_height
        
        return kinetic + potential
        
    def compute_momentum(self, velocities, mass=1.0):
        """Compute total momentum of the system.
        
        Args:
            velocities: Shape (batch, seq_len, 2, 2)
            mass: Mass of each ball
            
        Returns:
            Total momentum: Shape (batch, seq_len, 2)
        """
        # Sum momentum across both balls
        return mass * ops.sum(velocities, axis=-2)
        
    def call(self, inputs, training=None):
        """Forward pass through the model.
        
        Args:
            inputs: Trajectory features (batch, seq_len, 8)
            
        Returns:
            Dictionary with predictions
        """
        # Normalize inputs
        x = self.input_norm(inputs)
        
        # Extract features through LSTM layers
        features = x
        for lstm in self.lstm_layers:
            features = lstm(features)
            
        # Predict physics parameters
        physics_params = self.physics_head(features)
        
        # Predict trajectory evolution
        trajectory_pred = self.trajectory_head(features)
        
        # Extract states for physics computations
        positions, velocities = self.extract_physics_state(trajectory_pred)
        
        return {
            'trajectory': trajectory_pred,
            'positions': positions,
            'velocities': velocities,
            'physics_params': physics_params,
            'features': features
        }
        
    def compute_physics_loss(self, predictions, targets):
        """Compute physics-informed losses.
        
        Args:
            predictions: Model predictions dict
            targets: Ground truth dict
            
        Returns:
            Dictionary of physics losses
        """
        # Get predicted states
        pred_pos = predictions['positions']
        pred_vel = predictions['velocities']
        pred_gravity = predictions['physics_params'][..., 0]  # First param is gravity
        
        # Get true states
        true_pos = targets['positions']
        true_vel = targets['velocities']
        
        # Energy conservation loss
        # Energy should be approximately conserved (with some damping)
        pred_energy = self.compute_energy(pred_pos, pred_vel, pred_gravity)
        energy_diff = ops.diff(pred_energy, axis=1)  # Change in energy over time
        energy_loss = ops.mean(ops.square(energy_diff))
        
        # Momentum conservation loss
        # In absence of external forces, momentum should be conserved
        pred_momentum = self.compute_momentum(pred_vel)
        momentum_diff = ops.diff(pred_momentum, axis=1)
        momentum_loss = ops.mean(ops.square(momentum_diff))
        
        # Gravity consistency loss
        # Predicted gravity should be consistent across the sequence
        gravity_std = ops.std(pred_gravity)
        gravity_loss = gravity_std
        
        return {
            'energy_loss': energy_loss,
            'momentum_loss': momentum_loss,
            'gravity_loss': gravity_loss,
            'total_physics_loss': energy_loss + 0.5 * momentum_loss + 0.1 * gravity_loss
        }
        
    def compute_loss(self, inputs, targets, training=None):
        """Compute combined loss.
        
        Args:
            inputs: Input trajectories
            targets: Target dict with 'trajectory' and optionally 'physics_params'
            
        Returns:
            Total loss
        """
        # Get predictions
        predictions = self(inputs, training=training)
        
        # Trajectory prediction loss
        traj_loss = ops.mean(ops.square(predictions['trajectory'] - targets['trajectory']))
        
        # Physics parameter loss (if available)
        if 'physics_params' in targets:
            param_loss = ops.mean(ops.square(
                predictions['physics_params'] - targets['physics_params']
            ))
        else:
            param_loss = 0.0
            
        # Physics-informed losses
        if self.use_physics_loss and training:
            physics_losses = self.compute_physics_loss(predictions, targets)
            physics_loss = physics_losses['total_physics_loss']
        else:
            physics_loss = 0.0
            
        # Total loss
        total_loss = traj_loss + 0.1 * param_loss + 0.01 * physics_loss
        
        return total_loss


def create_physics_informed_model(config: Dict) -> SimplePhysicsInformedModel:
    """Create a physics-informed model for trajectory prediction.
    
    Args:
        config: Configuration dict with model parameters
        
    Returns:
        Compiled model ready for training
    """
    model = SimplePhysicsInformedModel(
        sequence_length=config.get('sequence_length', 50),
        hidden_dim=config.get('hidden_dim', 128),
        num_layers=config.get('num_layers', 3),
        use_physics_loss=config.get('use_physics_loss', True)
    )
    
    # Build model with dummy input
    dummy_input = ops.zeros((1, config.get('sequence_length', 50), 8))
    _ = model(dummy_input)
    
    return model


def test_physics_model():
    """Test the physics-informed model."""
    print("Testing Simplified Physics-Informed Model...")
    
    # Create model
    config = {
        'sequence_length': 50,
        'hidden_dim': 128,
        'num_layers': 2,
        'use_physics_loss': True
    }
    
    model = create_physics_informed_model(config)
    
    # Test data
    batch_size = 16
    seq_len = 50
    
    # Create dummy trajectory
    inputs = np.random.randn(batch_size, seq_len, 8).astype(np.float32)
    
    # Forward pass
    outputs = model(inputs, training=True)
    
    print(f"\nOutput shapes:")
    for key, value in outputs.items():
        if hasattr(value, 'shape'):
            print(f"  {key}: {value.shape}")
    
    # Test loss computation
    targets = {
        'trajectory': inputs + np.random.normal(0, 0.1, inputs.shape),
        'positions': outputs['positions'].numpy() + np.random.normal(0, 0.1, outputs['positions'].shape),
        'velocities': outputs['velocities'].numpy() + np.random.normal(0, 0.1, outputs['velocities'].shape),
        'physics_params': np.array([[-9.8, 0.5, 0.8, 0.95]] * batch_size, dtype=np.float32)
    }
    
    loss = model.compute_loss(inputs, targets, training=True)
    print(f"\nTotal loss: {float(loss):.4f}")
    
    # Test physics losses
    physics_losses = model.compute_physics_loss(outputs, targets)
    print(f"\nPhysics losses:")
    for key, value in physics_losses.items():
        print(f"  {key}: {float(value):.4f}")
    
    print("\nModel test passed!")
    
    # Print model summary
    total_params = sum(np.prod(p.shape) for p in model.trainable_variables)
    print(f"\nModel parameters: {total_params:,}")
    
    return model


if __name__ == "__main__":
    test_physics_model()