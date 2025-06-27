#!/usr/bin/env python3
"""
Physics-Informed Transformer for Ball Trajectory Prediction.

Combines transformer architecture with Hamiltonian Neural Networks
and physics constraints for improved extrapolation.
"""

import os
os.environ['KERAS_BACKEND'] = 'jax'

import keras
from keras import layers, ops
import numpy as np
from typing import Dict, Tuple, Optional

from models.physics_informed_components import (
    HamiltonianNN, PhysicsGuidedAttention, NonDimensionalizer, FourierFeatures
)
from models.collision_models import SoftCollisionPotential, WallBounceModel
from models.physics_losses import PhysicsLosses, ReLoBRaLo


@keras.saving.register_keras_serializable()
class PhysicsInformedTrajectoryTransformer(keras.Model):
    """Hybrid model combining transformer with physics-informed components.
    
    Architecture:
    1. Feature extraction with transformer encoder
    2. Physics constraints via Hamiltonian Neural Network
    3. Physics-guided attention for refinement
    4. Adaptive fusion of learned and physics features
    """
    
    def __init__(self,
                 # Transformer config
                 sequence_length: int = 300,
                 feature_dim: int = 18,  # 2 balls * 9 features
                 num_transformer_layers: int = 4,
                 num_heads: int = 8,
                 transformer_dim: int = 256,
                 # Physics config
                 use_hnn: bool = True,
                 hnn_hidden_dim: int = 256,
                 hnn_layers: int = 3,
                 # Collision config
                 use_soft_collisions: bool = True,
                 collision_stiffness: float = 1000.0,
                 # Loss config
                 use_physics_losses: bool = True,
                 energy_weight: float = 1.0,
                 momentum_weight: float = 0.5,
                 # Other
                 dropout_rate: float = 0.1,
                 **kwargs):
        super().__init__(**kwargs)
        
        # Save config
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.num_transformer_layers = num_transformer_layers
        self.num_heads = num_heads
        self.transformer_dim = transformer_dim
        self.use_hnn = use_hnn
        self.hnn_hidden_dim = hnn_hidden_dim
        self.hnn_layers = hnn_layers
        self.use_soft_collisions = use_soft_collisions
        self.collision_stiffness = collision_stiffness
        self.use_physics_losses = use_physics_losses
        self.energy_weight = energy_weight
        self.momentum_weight = momentum_weight
        self.dropout_rate = dropout_rate
        
        # Build components
        self._build_components()
        
    def _build_components(self):
        """Build all model components."""
        # Non-dimensionalizer
        self.non_dimensionalizer = NonDimensionalizer()
        
        # Input projection
        self.input_projection = layers.Dense(self.transformer_dim)
        self.positional_embedding = layers.Embedding(
            self.sequence_length, self.transformer_dim
        )
        
        # Transformer encoder
        self.transformer_blocks = []
        for _ in range(self.num_transformer_layers):
            self.transformer_blocks.append(
                TransformerBlock(
                    d_model=self.transformer_dim,
                    num_heads=self.num_heads,
                    dropout=self.dropout_rate
                )
            )
            
        # Physics components
        if self.use_hnn:
            self.hnn = HamiltonianNN(
                hidden_dim=self.hnn_hidden_dim,
                num_layers=self.hnn_layers,
                use_fourier_features=True
            )
            
        # Collision models
        if self.use_soft_collisions:
            self.collision_model = SoftCollisionPotential(
                stiffness=self.collision_stiffness
            )
            self.wall_model = WallBounceModel()
            
        # Physics-guided attention
        self.physics_attention = PhysicsGuidedAttention(
            hidden_dim=self.transformer_dim,
            num_heads=4
        )
        
        # Fusion layer
        self.fusion_gate = layers.Dense(self.transformer_dim, activation='sigmoid')
        self.fusion_transform = layers.Dense(self.transformer_dim)
        
        # Output heads
        self.position_head = layers.Dense(4)  # 2 balls * 2D positions
        self.velocity_head = layers.Dense(4)  # 2 balls * 2D velocities
        self.physics_param_head = layers.Dense(4)  # gravity, friction, elasticity, damping
        
        # Physics loss computer
        if self.use_physics_losses:
            self.physics_losses = PhysicsLosses(
                energy_weight=self.energy_weight,
                momentum_weight=self.momentum_weight
            )
            
    def get_config(self):
        return {
            'sequence_length': self.sequence_length,
            'feature_dim': self.feature_dim,
            'num_transformer_layers': self.num_transformer_layers,
            'num_heads': self.num_heads,
            'transformer_dim': self.transformer_dim,
            'use_hnn': self.use_hnn,
            'hnn_hidden_dim': self.hnn_hidden_dim,
            'hnn_layers': self.hnn_layers,
            'use_soft_collisions': self.use_soft_collisions,
            'collision_stiffness': self.collision_stiffness,
            'use_physics_losses': self.use_physics_losses,
            'energy_weight': self.energy_weight,
            'momentum_weight': self.momentum_weight,
            'dropout_rate': self.dropout_rate
        }
    
    def extract_physics_state(self, trajectory_features):
        """Extract position and momentum from trajectory features.
        
        Args:
            trajectory_features: Shape (batch, seq_len, features)
            
        Returns:
            Dictionary with positions, velocities, masses
        """
        # Assuming features are [time, x1, y1, x2, y2, vx1, vy1, vx2, vy2, ...]
        positions = ops.stack([
            trajectory_features[..., 1:3],  # Ball 1 position
            trajectory_features[..., 3:5]   # Ball 2 position
        ], axis=-2)  # (batch, seq_len, 2 balls, 2)
        
        velocities = ops.stack([
            trajectory_features[..., 5:7],  # Ball 1 velocity
            trajectory_features[..., 7:9]   # Ball 2 velocity
        ], axis=-2)
        
        # Extract masses (assuming they're in features or use default)
        masses = ops.ones((ops.shape(trajectory_features)[0], 2))  # Default mass 1.0
        
        return {
            'positions': positions,
            'velocities': velocities,
            'masses': masses
        }
    
    def compute_physics_features(self, physics_state, training=None):
        """Compute physics-based features using HNN and collision models."""
        features_list = []
        
        # HNN features
        if self.use_hnn:
            # Flatten for HNN (expects [q, p] concatenated)
            batch_size = ops.shape(physics_state['positions'])[0]
            seq_len = ops.shape(physics_state['positions'])[1]
            
            # Compute momenta from velocities (p = m * v)
            momenta = physics_state['velocities'] * ops.expand_dims(
                ops.expand_dims(physics_state['masses'], 1), -1
            )
            
            # Reshape for HNN: concatenate [x1, y1, x2, y2, px1, py1, px2, py2]
            positions_flat = ops.reshape(physics_state['positions'], (batch_size, seq_len, -1))
            momenta_flat = ops.reshape(momenta, (batch_size, seq_len, -1))
            hnn_state = ops.concatenate([positions_flat, momenta_flat], axis=-1)
            
            # Apply HNN to each timestep
            hnn_features = []
            for t in range(seq_len):
                state_t = hnn_state[:, t]
                h_value = self.hnn({'state': state_t, 'masses': physics_state['masses']})
                hnn_features.append(h_value)
                
            hnn_features = ops.stack(hnn_features, axis=1)  # (batch, seq_len)
            hnn_features = ops.expand_dims(hnn_features, -1)  # (batch, seq_len, 1)
            features_list.append(hnn_features)
            
        # Collision features
        if self.use_soft_collisions:
            collision_features = []
            
            for t in range(seq_len):
                pos1 = physics_state['positions'][:, t, 0]
                pos2 = physics_state['positions'][:, t, 1]
                
                # Ball-ball collision potential
                collision_inputs = {
                    'positions1': pos1,
                    'positions2': pos2,
                    'radius1': ops.full((batch_size,), 20.0),
                    'radius2': ops.full((batch_size,), 20.0)
                }
                collision_pot = self.collision_model(collision_inputs)
                
                # Wall collision potentials
                wall_inputs1 = {
                    'position': pos1,
                    'velocity': physics_state['velocities'][:, t, 0],
                    'radius': ops.full((batch_size,), 20.0)
                }
                wall_pot1, _ = self.wall_model(wall_inputs1)
                
                wall_inputs2 = {
                    'position': pos2,
                    'velocity': physics_state['velocities'][:, t, 1],
                    'radius': ops.full((batch_size,), 20.0)
                }
                wall_pot2, _ = self.wall_model(wall_inputs2)
                
                total_collision = collision_pot + wall_pot1 + wall_pot2
                collision_features.append(total_collision)
                
            collision_features = ops.stack(collision_features, axis=1)
            features_list.append(ops.expand_dims(collision_features, -1))
            
        # Concatenate all physics features
        if features_list:
            physics_features = ops.concatenate(features_list, axis=-1)
            # Project to transformer dimension
            physics_features = layers.Dense(self.transformer_dim)(physics_features)
        else:
            physics_features = None
            
        return physics_features
    
    def call(self, inputs, training=None):
        """Forward pass through the model.
        
        Args:
            inputs: Trajectory features (batch, seq_len, features)
            
        Returns:
            Dictionary with predictions and physics quantities
        """
        # Stage 1: Input processing
        batch_size = ops.shape(inputs)[0]
        seq_len = ops.shape(inputs)[1]
        
        # Project input features
        x = self.input_projection(inputs)
        
        # Add positional embeddings
        positions = ops.arange(seq_len)
        pos_embeddings = self.positional_embedding(positions)
        x = x + pos_embeddings
        
        # Stage 2: Transformer encoding
        transformer_features = x
        for transformer_block in self.transformer_blocks:
            transformer_features = transformer_block(transformer_features, training=training)
            
        # Stage 3: Extract physics state and compute physics features
        physics_state = self.extract_physics_state(inputs)
        physics_features = self.compute_physics_features(physics_state, training=training)
        
        # Stage 4: Physics-guided attention
        if physics_features is not None:
            refined_features = self.physics_attention(
                transformer_features, physics_features, training=training
            )
        else:
            refined_features = transformer_features
            
        # Stage 5: Adaptive fusion
        if physics_features is not None:
            fusion_input = ops.concatenate([refined_features, physics_features], axis=-1)
            gate = self.fusion_gate(fusion_input)
            transformed = self.fusion_transform(refined_features)
            fused_features = gate * transformed + (1 - gate) * refined_features
        else:
            fused_features = refined_features
            
        # Stage 6: Output predictions
        # Pool over sequence for physics parameters
        pooled_features = ops.mean(fused_features, axis=1)
        physics_params = self.physics_param_head(pooled_features)
        
        # Predict trajectories
        position_preds = self.position_head(fused_features)
        velocity_preds = self.velocity_head(fused_features)
        
        # Reshape predictions
        position_preds = ops.reshape(position_preds, (batch_size, seq_len, 2, 2))
        velocity_preds = ops.reshape(velocity_preds, (batch_size, seq_len, 2, 2))
        
        return {
            'positions': position_preds,
            'velocities': velocity_preds,
            'physics_params': physics_params,
            'transformer_features': transformer_features,
            'physics_features': physics_features,
            'fused_features': fused_features
        }
    
    def compute_loss(self, inputs, targets, training=None):
        """Compute combined loss with physics constraints."""
        predictions = self(inputs, training=training)
        
        # Trajectory prediction loss
        position_loss = ops.mean(ops.square(predictions['positions'] - targets['positions']))
        velocity_loss = ops.mean(ops.square(predictions['velocities'] - targets['velocities']))
        trajectory_loss = position_loss + velocity_loss
        
        # Physics parameter loss (if available)
        if 'physics_params' in targets:
            param_loss = ops.mean(ops.square(predictions['physics_params'] - targets['physics_params']))
        else:
            param_loss = 0.0
            
        # Physics-informed losses
        if self.use_physics_losses:
            physics_state = {
                'positions': predictions['positions'],
                'velocities': predictions['velocities'],
                'masses': ops.ones((ops.shape(inputs)[0], 2))  # Default masses
            }
            
            physics_loss_dict = self.physics_losses.combined_physics_loss(
                predictions=physics_state,
                targets={},  # No specific physics targets
                config={'damping': 0.95}  # From targets if available
            )
            
            physics_loss = physics_loss_dict['total']
        else:
            physics_loss = 0.0
            
        # Total loss
        total_loss = trajectory_loss + 0.1 * param_loss + physics_loss
        
        return {
            'total_loss': total_loss,
            'trajectory_loss': trajectory_loss,
            'param_loss': param_loss,
            'physics_loss': physics_loss
        }


@keras.saving.register_keras_serializable()
class TransformerBlock(layers.Layer):
    """Single transformer encoder block."""
    
    def __init__(self, d_model, num_heads, dff=None, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff or 4 * d_model
        self.dropout_rate = dropout
        
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        
        self.ffn = keras.Sequential([
            layers.Dense(self.dff, activation='relu'),
            layers.Dropout(dropout),
            layers.Dense(d_model)
        ])
        
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)
        
    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dff': self.dff,
            'dropout': self.dropout_rate
        })
        return config
        
    def call(self, inputs, training=None):
        # Multi-head attention
        attn_output = self.attention(inputs, inputs, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        # Feed-forward network
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2


def test_physics_informed_transformer():
    """Test the Physics-Informed Transformer implementation."""
    print("Testing Physics-Informed Transformer...")
    
    # Create model
    model = PhysicsInformedTrajectoryTransformer(
        sequence_length=50,  # Shorter for testing
        feature_dim=18,
        num_transformer_layers=2,
        num_heads=4,
        transformer_dim=128,
        use_hnn=True,
        use_soft_collisions=True,
        use_physics_losses=True
    )
    
    # Test data
    batch_size = 8
    seq_len = 50
    features = 18  # 2 balls * 9 features
    
    # Create dummy trajectory data
    inputs = np.random.randn(batch_size, seq_len, features).astype(np.float32)
    
    # Make realistic positions and velocities
    inputs[:, :, 1:5] = np.random.uniform(100, 700, (batch_size, seq_len, 4))  # Positions
    inputs[:, :, 5:9] = np.random.uniform(-50, 50, (batch_size, seq_len, 4))  # Velocities
    
    # Forward pass
    outputs = model(inputs, training=True)
    
    print(f"Output shapes:")
    for key, value in outputs.items():
        if value is not None:
            print(f"  {key}: {value.shape}")
    
    # Test loss computation
    targets = {
        'positions': outputs['positions'] + np.random.normal(0, 0.1, outputs['positions'].shape),
        'velocities': outputs['velocities'] + np.random.normal(0, 0.1, outputs['velocities'].shape),
        'physics_params': np.array([[-981.0, 0.7, 0.8, 0.95]] * batch_size, dtype=np.float32)
    }
    
    losses = model.compute_loss(inputs, targets, training=True)
    print(f"\nLosses:")
    for key, value in losses.items():
        print(f"  {key}: {float(value):.4f}")
    
    print("\nTest passed!")


if __name__ == "__main__":
    test_physics_informed_transformer()