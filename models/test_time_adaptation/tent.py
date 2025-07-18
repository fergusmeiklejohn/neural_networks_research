"""TENT: Test-time Entropy Minimization implementation.

Based on "Tent: Fully Test-Time Adaptation by Entropy Minimization" (Wang et al., 2021)
https://arxiv.org/abs/2006.10726
"""

from typing import Any, List, Optional
import keras
from keras import ops

from .base_tta import BaseTTA
from .utils.entropy import entropy_loss, confidence_selection


class TENT(BaseTTA):
    """Test-time Entropy Minimization (TENT) adaptation.
    
    TENT adapts by minimizing the entropy of model predictions during test time.
    It focuses on updating BatchNorm statistics and affine parameters.
    """
    
    def __init__(
        self,
        model: keras.Model,
        adaptation_steps: int = 1,
        learning_rate: float = 1e-3,
        reset_after_batch: bool = True,
        update_bn_only: bool = True,
        confidence_threshold: Optional[float] = None,
        **kwargs
    ):
        """Initialize TENT.
        
        Args:
            model: Base model to adapt
            adaptation_steps: Number of adaptation steps
            learning_rate: Learning rate for updates
            reset_after_batch: Whether to reset after each batch
            update_bn_only: If True, only update BatchNorm parameters
            confidence_threshold: Optional threshold for confident sample selection
        """
        super().__init__(model, adaptation_steps, learning_rate, reset_after_batch, **kwargs)
        
        self.update_bn_only = update_bn_only
        self.confidence_threshold = confidence_threshold
        
        # Configure model for TENT
        self._configure_model()
    
    def _configure_model(self):
        """Configure model for TENT adaptation."""
        # Set model to training mode for BatchNorm updates
        self.model.trainable = True
        
        if self.update_bn_only:
            # Only update BatchNorm parameters
            for layer in self.model.layers:
                if isinstance(layer, (keras.layers.BatchNormalization,)):
                    layer.trainable = True
                else:
                    layer.trainable = False
    
    def compute_adaptation_loss(self, x: Any, y_pred: Any) -> Any:
        """Compute entropy loss for adaptation.
        
        Args:
            x: Input data (not used in TENT)
            y_pred: Model predictions (logits or probabilities)
            
        Returns:
            Entropy loss
        """
        # Convert logits to probabilities if needed
        if ops.shape(y_pred)[-1] > 1:  # Multi-class
            probs = ops.softmax(y_pred, axis=-1)
        else:  # Binary
            probs = ops.sigmoid(y_pred)
        
        # Compute entropy
        ent_loss = entropy_loss(probs)
        
        # Optional: Filter by confidence
        if self.confidence_threshold is not None:
            mask = confidence_selection(probs, self.confidence_threshold)
            ent_loss = ops.mean(ent_loss * mask)
        else:
            ent_loss = ops.mean(ent_loss)
        
        return ent_loss


class PhysicsTENT(TENT):
    """Physics-aware TENT adaptation.
    
    Extends TENT with physics-specific losses for better adaptation
    on physics prediction tasks.
    """
    
    def __init__(
        self,
        model: keras.Model,
        adaptation_steps: int = 1,
        learning_rate: float = 1e-3,
        reset_after_batch: bool = True,
        update_bn_only: bool = True,
        confidence_threshold: Optional[float] = None,
        physics_loss_weight: float = 0.1,
        **kwargs
    ):
        """Initialize PhysicsTENT.
        
        Args:
            model: Base model
            physics_loss_weight: Weight for physics consistency loss
            **kwargs: Other arguments passed to TENT
        """
        super().__init__(
            model, adaptation_steps, learning_rate, 
            reset_after_batch, update_bn_only, confidence_threshold,
            **kwargs
        )
        self.physics_loss_weight = physics_loss_weight
    
    def compute_physics_loss(self, x: Any, y_pred: Any) -> Any:
        """Compute physics-based consistency loss.
        
        Args:
            x: Input trajectory data
            y_pred: Predicted trajectory
            
        Returns:
            Physics consistency loss
        """
        # Extract positions and velocities
        # Assuming y_pred shape: (batch, time, features)
        # where features = [x1, y1, vx1, vy1, x2, y2, vx2, vy2]
        
        if len(ops.shape(y_pred)) == 3:
            # Trajectory prediction
            positions = y_pred[..., [0, 1, 4, 5]]  # x,y for both balls
            velocities = y_pred[..., [2, 3, 6, 7]]  # vx,vy for both balls
            
            # Energy conservation check (simplified)
            # KE = 0.5 * m * v^2 (assuming unit mass)
            kinetic_energy = 0.5 * ops.sum(velocities**2, axis=-1)
            
            # Energy should be approximately conserved
            energy_diff = ops.diff(kinetic_energy, axis=1)
            energy_loss = ops.mean(ops.abs(energy_diff))
            
            # Momentum conservation (assuming closed system)
            # p = m * v (assuming unit mass)
            total_momentum = ops.sum(velocities.reshape((-1, ops.shape(velocities)[1], 2, 2)), axis=2)
            momentum_diff = ops.diff(total_momentum, axis=1)
            momentum_loss = ops.mean(ops.abs(momentum_diff))
            
            return energy_loss + momentum_loss
        else:
            # Single state prediction - no physics loss
            return ops.zeros(())
    
    def compute_adaptation_loss(self, x: Any, y_pred: Any) -> Any:
        """Compute combined entropy and physics loss.
        
        Args:
            x: Input data
            y_pred: Model predictions
            
        Returns:
            Combined loss
        """
        # Get entropy loss from parent
        ent_loss = super().compute_adaptation_loss(x, y_pred)
        
        # Add physics loss
        phys_loss = self.compute_physics_loss(x, y_pred)
        
        return ent_loss + self.physics_loss_weight * phys_loss