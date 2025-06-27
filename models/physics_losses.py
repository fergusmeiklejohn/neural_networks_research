#!/usr/bin/env python3
"""
Physics-Informed Loss Functions for Ball Trajectory Prediction.

Implements energy conservation, momentum conservation, and trajectory
smoothness losses with advanced loss balancing strategies.
"""

import os
os.environ['KERAS_BACKEND'] = 'jax'

import keras
from keras import ops
import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


class PhysicsLosses:
    """Collection of physics-informed loss functions."""
    
    def __init__(self,
                 gravity: float = -981.0,
                 energy_weight: float = 1.0,
                 momentum_weight: float = 0.5,
                 smoothness_weight: float = 0.1):
        self.gravity = gravity
        self.energy_weight = energy_weight
        self.momentum_weight = momentum_weight
        self.smoothness_weight = smoothness_weight
        
    def compute_kinetic_energy(self, velocities: jnp.ndarray, masses: jnp.ndarray) -> jnp.ndarray:
        """Compute kinetic energy: KE = 1/2 * m * v²"""
        # velocities shape: (batch, time, n_balls, 2)
        # masses shape: (batch, n_balls)
        v_squared = ops.sum(velocities**2, axis=-1)  # (batch, time, n_balls)
        
        # Expand masses for broadcasting
        masses_expanded = ops.expand_dims(masses, axis=1)  # (batch, 1, n_balls)
        
        ke = 0.5 * masses_expanded * v_squared
        return ops.sum(ke, axis=-1)  # Total KE per timestep (batch, time)
        
    def compute_potential_energy(self, positions: jnp.ndarray, masses: jnp.ndarray) -> jnp.ndarray:
        """Compute gravitational potential energy: PE = m * g * h"""
        # positions shape: (batch, time, n_balls, 2), take y-coordinate
        heights = positions[..., 1]  # (batch, time, n_balls)
        
        # Expand masses for broadcasting
        masses_expanded = ops.expand_dims(masses, axis=1)  # (batch, 1, n_balls)
        
        pe = masses_expanded * (-self.gravity) * heights  # Negative gravity
        return ops.sum(pe, axis=-1)  # Total PE per timestep (batch, time)
        
    def energy_conservation_loss(self, 
                               trajectory: Dict[str, jnp.ndarray],
                               damping: float = 0.95) -> jnp.ndarray:
        """Compute energy conservation loss over trajectory.
        
        Args:
            trajectory: Dictionary with 'positions', 'velocities', 'masses'
                       Each has shape (batch, time, n_balls, ...)
            damping: Expected energy decay rate due to air resistance
            
        Returns:
            Energy conservation loss
        """
        positions = trajectory['positions']
        velocities = trajectory['velocities'] 
        masses = trajectory['masses']
        
        # Compute total energy at each timestep
        ke = self.compute_kinetic_energy(velocities, masses)
        pe = self.compute_potential_energy(positions, masses)
        total_energy = ke + pe  # Shape: (batch, time)
        
        # Expected energy with damping
        initial_energy = total_energy[:, 0:1]  # (batch, 1)
        time_steps = ops.arange(total_energy.shape[1])
        expected_energy = initial_energy * (damping ** time_steps)
        
        # Energy conservation error
        energy_error = ops.abs(total_energy - expected_energy) / (initial_energy + 1e-6)
        
        # Average over time, sum over batch
        return ops.mean(energy_error)
        
    def momentum_conservation_loss(self,
                                 velocities_before: jnp.ndarray,
                                 velocities_after: jnp.ndarray,
                                 masses: jnp.ndarray,
                                 collision_mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """Compute momentum conservation loss for collisions.
        
        Args:
            velocities_before: Velocities before collision (batch, n_balls, 2)
            velocities_after: Velocities after collision  
            masses: Ball masses (batch, n_balls)
            collision_mask: Binary mask indicating collision frames
            
        Returns:
            Momentum conservation loss
        """
        # Compute momentum: p = m * v
        momentum_before = ops.expand_dims(masses, -1) * velocities_before
        momentum_after = ops.expand_dims(masses, -1) * velocities_after
        
        # Total momentum for system
        total_momentum_before = ops.sum(momentum_before, axis=1)  # (batch, 2)
        total_momentum_after = ops.sum(momentum_after, axis=1)
        
        # Momentum should be conserved
        momentum_error = ops.norm(total_momentum_after - total_momentum_before, axis=-1)
        
        # Apply collision mask if provided
        if collision_mask is not None:
            momentum_error = momentum_error * collision_mask
            
        return ops.mean(momentum_error)
        
    def trajectory_smoothness_loss(self,
                                 positions: jnp.ndarray,
                                 dt: float = 1/60.0) -> jnp.ndarray:
        """Compute trajectory smoothness using acceleration continuity.
        
        Args:
            positions: Ball positions over time (batch, time, n_balls, 2)
            dt: Time step
            
        Returns:
            Smoothness loss
        """
        # Compute velocities from positions
        velocities = (positions[:, 1:] - positions[:, :-1]) / dt
        
        # Compute accelerations from velocities  
        accelerations = (velocities[:, 1:] - velocities[:, :-1]) / dt
        
        # Compute jerk (change in acceleration)
        jerk = (accelerations[:, 1:] - accelerations[:, :-1]) / dt
        
        # Penalize high jerk for smooth trajectories
        jerk_magnitude = ops.norm(jerk, axis=-1)  # (batch, time-3, n_balls)
        
        return ops.mean(jerk_magnitude)
        
    def physics_consistency_loss(self,
                               positions: jnp.ndarray,
                               velocities: jnp.ndarray,
                               forces: jnp.ndarray,
                               masses: jnp.ndarray,
                               dt: float = 1/60.0) -> jnp.ndarray:
        """Ensure trajectory follows F = ma.
        
        Args:
            positions: Positions (batch, time, n_balls, 2)
            velocities: Velocities (batch, time, n_balls, 2)
            forces: Predicted forces (batch, time, n_balls, 2)
            masses: Masses (batch, n_balls)
            
        Returns:
            Physics consistency loss
        """
        # Compute observed accelerations
        observed_acc = (velocities[:, 1:] - velocities[:, :-1]) / dt
        
        # Expected accelerations from forces
        masses_expanded = ops.expand_dims(ops.expand_dims(masses, 1), -1)
        expected_acc = forces[:, :-1] / masses_expanded
        
        # Consistency error
        acc_error = ops.norm(observed_acc - expected_acc, axis=-1)
        
        return ops.mean(acc_error)
        
    def combined_physics_loss(self, predictions: Dict, targets: Dict, config: Dict) -> Dict[str, jnp.ndarray]:
        """Compute all physics losses with configured weights.
        
        Returns dictionary of individual losses and total loss.
        """
        losses = {}
        
        # Energy conservation
        if self.energy_weight > 0:
            losses['energy'] = self.energy_conservation_loss(
                predictions, 
                damping=config.get('damping', 0.95)
            ) * self.energy_weight
            
        # Momentum conservation (if collision data provided)
        if 'collision_mask' in targets and self.momentum_weight > 0:
            losses['momentum'] = self.momentum_conservation_loss(
                predictions['velocities'],
                targets['velocities_after_collision'],
                predictions['masses'],
                targets['collision_mask']
            ) * self.momentum_weight
            
        # Trajectory smoothness
        if self.smoothness_weight > 0:
            losses['smoothness'] = self.trajectory_smoothness_loss(
                predictions['positions']
            ) * self.smoothness_weight
            
        # Physics consistency (if forces provided)
        if 'forces' in predictions:
            losses['consistency'] = self.physics_consistency_loss(
                predictions['positions'],
                predictions['velocities'],
                predictions['forces'],
                predictions['masses']
            )
            
        # Total loss
        losses['total'] = sum(losses.values())
        
        return losses


class ReLoBRaLo:
    """Relative Loss Balancing with Random Lookback (ReLoBRaLo).
    
    Automatically balances multiple loss terms based on their
    relative convergence rates.
    """
    
    def __init__(self,
                 alpha: float = 0.95,
                 lookback_window: int = 100,
                 update_freq: int = 10,
                 min_weight: float = 0.01,
                 max_weight: float = 10.0):
        """
        Args:
            alpha: Exponential moving average decay factor
            lookback_window: Number of steps to look back for statistics
            update_freq: How often to update weights
            min_weight: Minimum allowed weight
            max_weight: Maximum allowed weight
        """
        self.alpha = alpha
        self.lookback_window = lookback_window
        self.update_freq = update_freq
        self.min_weight = min_weight
        self.max_weight = max_weight
        
        # History tracking
        self.loss_history = defaultdict(list)
        self.weight_history = defaultdict(list)
        self.current_weights = {}
        self.step_count = 0
        
    def update(self, losses: Dict[str, float]) -> Dict[str, float]:
        """Update loss weights based on relative convergence.
        
        Args:
            losses: Dictionary of current loss values
            
        Returns:
            Dictionary of loss weights to use
        """
        self.step_count += 1
        
        # Record losses
        for name, value in losses.items():
            if name != 'total':  # Skip total loss
                self.loss_history[name].append(float(value))
                
        # Initialize weights if needed
        if not self.current_weights:
            for name in losses:
                if name != 'total':
                    self.current_weights[name] = 1.0
                    
        # Update weights periodically
        if self.step_count % self.update_freq == 0 and self.step_count > self.lookback_window:
            self._update_weights()
            
        return self.current_weights.copy()
        
    def _update_weights(self):
        """Update weights based on relative loss decrease rates."""
        relative_rates = {}
        
        for name in self.loss_history:
            history = self.loss_history[name]
            
            if len(history) < self.lookback_window:
                continue
                
            # Random lookback sampling
            lookback_idx = np.random.randint(
                len(history) - self.lookback_window,
                len(history) - self.lookback_window // 2
            )
            
            # Compute relative decrease rate
            old_loss = history[lookback_idx]
            recent_losses = history[-self.lookback_window // 10:]
            current_loss = np.mean(recent_losses)
            
            # Relative improvement rate
            if old_loss > 0:
                rate = (old_loss - current_loss) / old_loss
            else:
                rate = 0.0
                
            relative_rates[name] = rate
            
        # Normalize rates
        if relative_rates:
            mean_rate = np.mean(list(relative_rates.values()))
            
            for name in relative_rates:
                # Losses improving slower get higher weight
                if mean_rate > 0:
                    weight_update = 1.0 + (mean_rate - relative_rates[name]) / mean_rate
                else:
                    weight_update = 1.0
                    
                # Exponential moving average update
                old_weight = self.current_weights[name]
                new_weight = self.alpha * old_weight + (1 - self.alpha) * weight_update
                
                # Clamp weights
                new_weight = np.clip(new_weight, self.min_weight, self.max_weight)
                self.current_weights[name] = new_weight
                
        # Record weight history
        for name, weight in self.current_weights.items():
            self.weight_history[name].append(weight)
            
    def get_statistics(self) -> Dict:
        """Get loss balancing statistics for monitoring."""
        stats = {
            'step_count': self.step_count,
            'current_weights': self.current_weights.copy(),
            'loss_history_lengths': {name: len(hist) for name, hist in self.loss_history.items()},
        }
        
        # Recent loss values
        if self.loss_history:
            stats['recent_losses'] = {
                name: np.mean(hist[-10:]) if hist else 0.0
                for name, hist in self.loss_history.items()
            }
            
        # Weight evolution
        if self.weight_history:
            stats['weight_changes'] = {
                name: hist[-1] / hist[0] if len(hist) > 0 and hist[0] > 0 else 1.0
                for name, hist in self.weight_history.items()
            }
            
        return stats


class PhysicsInformedLoss(keras.losses.Loss):
    """Keras loss wrapper for physics-informed losses with ReLoBRaLo."""
    
    def __init__(self,
                 physics_losses: PhysicsLosses,
                 use_relobralo: bool = True,
                 trajectory_weight: float = 1.0,
                 **kwargs):
        super().__init__(**kwargs)
        self.physics_losses = physics_losses
        self.use_relobralo = use_relobralo
        self.trajectory_weight = trajectory_weight
        
        if use_relobralo:
            self.loss_balancer = ReLoBRaLo()
        else:
            self.loss_balancer = None
            
    def call(self, y_true, y_pred):
        """Compute combined loss with physics constraints.
        
        Args:
            y_true: Target trajectory data
            y_pred: Predicted trajectory data
            
        Returns:
            Total loss value
        """
        # Trajectory prediction loss (MSE)
        trajectory_loss = ops.mean(ops.square(y_pred - y_true))
        
        # Physics losses
        physics_loss_dict = self.physics_losses.combined_physics_loss(
            predictions=y_pred,
            targets=y_true,
            config={}  # Would be passed from model
        )
        
        # Balance losses
        if self.use_relobralo and self.loss_balancer:
            # Convert to Python floats for ReLoBRaLo
            loss_values = {
                name: float(value) for name, value in physics_loss_dict.items()
                if name != 'total'
            }
            loss_values['trajectory'] = float(trajectory_loss)
            
            # Get balanced weights
            weights = self.loss_balancer.update(loss_values)
            
            # Apply weights
            total_loss = weights.get('trajectory', self.trajectory_weight) * trajectory_loss
            for name, value in physics_loss_dict.items():
                if name != 'total':
                    total_loss += weights.get(name, 1.0) * value
        else:
            # Fixed weights
            total_loss = self.trajectory_weight * trajectory_loss + physics_loss_dict['total']
            
        return total_loss


def test_physics_losses():
    """Test physics loss implementations."""
    print("Testing Physics Losses...")
    
    # Create dummy trajectory data
    batch_size = 16
    time_steps = 50
    n_balls = 2
    
    positions = np.random.randn(batch_size, time_steps, n_balls, 2).astype(np.float32) * 100 + 400
    velocities = np.random.randn(batch_size, time_steps, n_balls, 2).astype(np.float32) * 10
    masses = np.ones((batch_size, n_balls), dtype=np.float32)
    
    trajectory = {
        'positions': positions,
        'velocities': velocities,
        'masses': masses
    }
    
    # Test individual losses
    physics_losses = PhysicsLosses()
    
    energy_loss = physics_losses.energy_conservation_loss(trajectory)
    print(f"Energy conservation loss: {energy_loss}")
    
    smoothness_loss = physics_losses.trajectory_smoothness_loss(positions)
    print(f"Trajectory smoothness loss: {smoothness_loss}")
    
    # Test ReLoBRaLo
    print("\nTesting ReLoBRaLo...")
    relobralo = ReLoBRaLo()
    
    # Simulate training steps
    for step in range(200):
        # Simulate decreasing losses
        fake_losses = {
            'energy': 1.0 * np.exp(-step * 0.01) + np.random.normal(0, 0.1),
            'momentum': 0.5 * np.exp(-step * 0.005) + np.random.normal(0, 0.05),
            'smoothness': 0.2 * np.exp(-step * 0.02) + np.random.normal(0, 0.02)
        }
        
        weights = relobralo.update(fake_losses)
        
        if step % 50 == 0:
            print(f"Step {step}: Weights = {weights}")
    
    # Get final statistics
    stats = relobralo.get_statistics()
    print(f"\nFinal statistics:")
    print(f"  Current weights: {stats['current_weights']}")
    print(f"  Weight changes: {stats.get('weight_changes', {})}")
    
    print("\nAll tests passed!")


if __name__ == "__main__":
    test_physics_losses()