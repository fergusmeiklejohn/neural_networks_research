"""
Physics Training with MMD Loss Integration

Integrates Maximum Mean Discrepancy loss into physics model training
to ensure generated distributions match target statistical properties.
"""

import os
import sys
import numpy as np
import keras
from pathlib import Path
from typing import Dict, Tuple, Optional

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from models.mmd_loss import MMDLoss, create_physics_loss_with_mmd
from models.physics_informed_transformer import PhysicsInformedTrajectoryTransformer
from models.physics_losses import PhysicsLosses


class PhysicsMMDTrainer:
    """Trainer that combines physics losses with MMD regularization."""
    
    def __init__(self,
                 model: keras.Model,
                 mmd_weight: float = 0.1,
                 mmd_bandwidths: Optional[list] = None,
                 physics_loss_weights: Optional[Dict[str, float]] = None):
        """
        Args:
            model: Physics model to train
            mmd_weight: Weight for MMD loss term
            mmd_bandwidths: Bandwidths for RBF kernel
            physics_loss_weights: Weights for different physics losses
        """
        self.model = model
        self.mmd_weight = mmd_weight
        self.mmd_bandwidths = mmd_bandwidths or [0.01, 0.1, 1.0, 10.0, 100.0]
        
        # Default physics loss weights
        self.physics_loss_weights = physics_loss_weights or {
            'trajectory': 1.0,
            'energy': 0.1,
            'momentum': 0.1,
            'collision': 0.01
        }
        
        # Create losses
        self.mmd_loss = MMDLoss(bandwidths=self.mmd_bandwidths)
        self.physics_losses = PhysicsLosses()
        
        # Optimizer
        self.optimizer = keras.optimizers.Adam(learning_rate=1e-3)
        
    def compute_total_loss(self,
                          y_true: keras.KerasTensor,
                          y_pred: keras.KerasTensor,
                          initial_conditions: Optional[keras.KerasTensor] = None) -> Tuple[keras.KerasTensor, Dict]:
        """
        Compute combined physics + MMD loss.
        
        Args:
            y_true: True trajectories
            y_pred: Predicted trajectories
            initial_conditions: Initial physics conditions
            
        Returns:
            total_loss, loss_components dictionary
        """
        loss_components = {}
        
        # Trajectory prediction loss
        trajectory_loss = keras.losses.mean_squared_error(y_true, y_pred)
        loss_components['trajectory'] = trajectory_loss
        
        # Physics constraint losses
        if hasattr(self.physics_losses, 'energy_conservation_loss'):
            energy_loss = self.physics_losses.energy_conservation_loss(y_pred, initial_conditions)
            loss_components['energy'] = energy_loss
        
        if hasattr(self.physics_losses, 'momentum_conservation_loss'):
            momentum_loss = self.physics_losses.momentum_conservation_loss(y_pred)
            loss_components['momentum'] = momentum_loss
            
        # MMD loss for distribution matching
        mmd_loss = self.mmd_loss(y_true, y_pred)
        loss_components['mmd'] = mmd_loss
        
        # Combine losses
        total_loss = self.physics_loss_weights['trajectory'] * trajectory_loss
        
        if 'energy' in loss_components:
            total_loss += self.physics_loss_weights['energy'] * loss_components['energy']
            
        if 'momentum' in loss_components:
            total_loss += self.physics_loss_weights['momentum'] * loss_components['momentum']
            
        total_loss += self.mmd_weight * mmd_loss
        
        return total_loss, loss_components
    
    @keras.utils.register_keras_serializable()
    def train_step(self, data):
        """Single training step with MMD loss."""
        x, y = data
        
        with keras.ops.GradientTape() as tape:
            y_pred = self.model(x, training=True)
            
            # Extract initial conditions from input if available
            initial_conditions = x if len(x.shape) > 2 else None
            
            # Compute total loss
            loss, loss_components = self.compute_total_loss(y, y_pred, initial_conditions)
            
        # Compute gradients and update
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        # Return metrics
        metrics = {'loss': loss}
        metrics.update({f'loss_{k}': v for k, v in loss_components.items()})
        
        return metrics


def create_physics_model_with_mmd(input_shape: Tuple[int, ...],
                                 output_shape: Tuple[int, ...],
                                 hidden_dim: int = 128) -> keras.Model:
    """
    Create a simple physics model for demonstration.
    
    Args:
        input_shape: Input trajectory shape
        output_shape: Output trajectory shape
        hidden_dim: Hidden layer dimension
        
    Returns:
        Keras model
    """
    inputs = keras.Input(shape=input_shape)
    
    # Simple feedforward for demonstration
    x = keras.layers.Flatten()(inputs)
    x = keras.layers.Dense(hidden_dim, activation='relu')(x)
    x = keras.layers.Dense(hidden_dim, activation='relu')(x)
    x = keras.layers.Dense(np.prod(output_shape))(x)
    outputs = keras.layers.Reshape(output_shape)(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def train_with_mmd(train_data: Tuple[np.ndarray, np.ndarray],
                  val_data: Tuple[np.ndarray, np.ndarray],
                  epochs: int = 50,
                  batch_size: int = 32,
                  mmd_weight: float = 0.1) -> Dict:
    """
    Train physics model with MMD loss.
    
    Args:
        train_data: (X_train, y_train) tuple
        val_data: (X_val, y_val) tuple
        epochs: Number of epochs
        batch_size: Batch size
        mmd_weight: Weight for MMD term
        
    Returns:
        Training history
    """
    X_train, y_train = train_data
    X_val, y_val = val_data
    
    # Create model
    input_shape = X_train.shape[1:]
    output_shape = y_train.shape[1:]
    model = create_physics_model_with_mmd(input_shape, output_shape)
    
    # Create trainer
    trainer = PhysicsMMDTrainer(model, mmd_weight=mmd_weight)
    
    # Compile model with composite loss
    model.compile(
        optimizer=trainer.optimizer,
        loss=lambda y_true, y_pred: trainer.compute_total_loss(y_true, y_pred)[0],
        metrics=['mae']
    )
    
    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    
    return history.history


# Example usage
if __name__ == "__main__":
    # Create synthetic data for testing
    n_samples = 1000
    trajectory_length = 10
    n_features = 4  # x, y, vx, vy
    
    # Generate synthetic trajectories
    X_train = np.random.randn(n_samples, trajectory_length, n_features)
    y_train = X_train + 0.1 * np.random.randn(n_samples, trajectory_length, n_features)
    
    X_val = np.random.randn(200, trajectory_length, n_features)
    y_val = X_val + 0.1 * np.random.randn(200, trajectory_length, n_features)
    
    print("Training physics model with MMD loss...")
    
    # Train with different MMD weights
    for mmd_weight in [0.0, 0.1, 0.5]:
        print(f"\nMMD weight: {mmd_weight}")
        history = train_with_mmd(
            (X_train, y_train),
            (X_val, y_val),
            epochs=10,
            mmd_weight=mmd_weight
        )
        
        final_loss = history['loss'][-1]
        final_val_loss = history['val_loss'][-1]
        print(f"Final loss: {final_loss:.4f}, Val loss: {final_val_loss:.4f}")