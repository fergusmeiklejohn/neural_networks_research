"""Test-Time Adaptation methods specifically designed for regression tasks."""

from typing import Any

import keras
from keras import ops

from .base_tta_jax import BaseTTAJax


class RegressionTTA(BaseTTAJax):
    """TTA for regression tasks using self-supervised objectives.

    This adapts models by minimizing prediction consistency loss and
    temporal smoothness for time-series/trajectory data.
    """

    def __init__(
        self,
        model: keras.Model,
        adaptation_steps: int = 5,
        learning_rate: float = 1e-4,
        reset_after_batch: bool = True,
        consistency_weight: float = 1.0,
        smoothness_weight: float = 0.1,
        update_bn_only: bool = True,
        **kwargs,
    ):
        """Initialize RegressionTTA.

        Args:
            model: Base model to adapt
            adaptation_steps: Number of adaptation steps
            learning_rate: Learning rate for updates
            reset_after_batch: Whether to reset after each batch
            consistency_weight: Weight for prediction consistency loss
            smoothness_weight: Weight for temporal smoothness loss
            update_bn_only: If True, only update BatchNorm parameters
        """
        super().__init__(
            model, adaptation_steps, learning_rate, reset_after_batch, **kwargs
        )

        self.consistency_weight = consistency_weight
        self.smoothness_weight = smoothness_weight
        self.update_bn_only = update_bn_only

        # Store which parameters to adapt
        if self.update_bn_only:
            self.adaptable_params = []
            for layer in self.model.layers:
                if isinstance(layer, keras.layers.BatchNormalization):
                    # Only adapt scale and offset, not moving stats
                    if layer.scale is not None:
                        self.adaptable_params.append(layer.scale)
                    if layer.center is not None:
                        self.adaptable_params.append(layer.center)
        else:
            self.adaptable_params = self.model.trainable_variables

    def compute_adaptation_loss(self, x: Any, y_pred: Any) -> Any:
        """Compute self-supervised loss for regression adaptation.

        Args:
            x: Input data
            y_pred: Model predictions

        Returns:
            Combined adaptation loss
        """
        losses = []

        # 1. Prediction consistency loss
        # Encourage consistent predictions by minimizing variance
        if len(ops.shape(x)) > 0 and ops.shape(x)[0] > 1:
            # Compute variance across batch
            pred_mean = ops.mean(y_pred, axis=0, keepdims=True)
            consistency_loss = ops.mean(ops.square(y_pred - pred_mean))
            losses.append(self.consistency_weight * consistency_loss)

        # 2. Temporal smoothness loss (for trajectory data)
        if len(ops.shape(y_pred)) == 3:  # (batch, time, features)
            # Minimize changes between consecutive timesteps
            diff = y_pred[:, 1:, :] - y_pred[:, :-1, :]
            smoothness_loss = ops.mean(ops.square(diff))
            losses.append(self.smoothness_weight * smoothness_loss)
        elif len(ops.shape(y_pred)) == 2 and ops.shape(x)[0] > 1:
            # For single-step predictions, encourage smooth predictions
            # by minimizing differences between similar inputs
            # (This is a simplified version)
            if ops.shape(x)[0] >= 2:
                pred_diff = y_pred[1:] - y_pred[:-1]
                smoothness_loss = ops.mean(ops.square(pred_diff))
                losses.append(self.smoothness_weight * smoothness_loss)

        # Combine losses
        if losses:
            total_loss = ops.sum(ops.stack(losses))
        else:
            # Fallback: just use L2 regularization on predictions
            total_loss = 0.01 * ops.mean(ops.square(y_pred))

        return total_loss


class PhysicsRegressionTTA(RegressionTTA):
    """Physics-aware TTA for regression on physics simulations.

    Adds physics-based constraints to the adaptation loss.
    """

    def __init__(
        self,
        model: keras.Model,
        adaptation_steps: int = 5,
        learning_rate: float = 1e-4,
        reset_after_batch: bool = True,
        consistency_weight: float = 0.5,
        smoothness_weight: float = 0.1,
        physics_weight: float = 1.0,
        update_bn_only: bool = True,
        **kwargs,
    ):
        """Initialize PhysicsRegressionTTA."""
        super().__init__(
            model,
            adaptation_steps,
            learning_rate,
            reset_after_batch,
            consistency_weight,
            smoothness_weight,
            update_bn_only,
            **kwargs,
        )
        self.physics_weight = physics_weight

    def compute_physics_constraints(self, x: Any, y_pred: Any) -> Any:
        """Compute physics-based constraint violations.

        Args:
            x: Input states
            y_pred: Predicted next states

        Returns:
            Physics constraint loss
        """
        # For 2-ball physics: y_pred contains [x1, y1, vx1, vy1, x2, y2, vx2, vy2]

        if len(ops.shape(y_pred)) == 2 and ops.shape(y_pred)[-1] >= 8:
            # Extract positions and velocities
            pos1 = y_pred[..., 0:2]  # Ball 1 position
            vel1 = y_pred[..., 2:4]  # Ball 1 velocity
            pos2 = y_pred[..., 4:6]  # Ball 2 position
            vel2 = y_pred[..., 6:8]  # Ball 2 velocity

            # 1. Momentum conservation (in absence of external forces)
            # Total momentum should remain relatively constant
            total_momentum = vel1 + vel2  # Assuming unit mass
            momentum_var = ops.var(total_momentum, axis=0)
            momentum_loss = ops.mean(momentum_var)

            # 2. Energy dissipation constraint
            # Kinetic energy should not increase
            ke = 0.5 * (ops.sum(vel1**2, axis=-1) + ops.sum(vel2**2, axis=-1))
            if ops.shape(x)[-1] >= 8:
                # Compare with input velocities
                vel1_in = x[..., 2:4]
                vel2_in = x[..., 6:8]
                ke_in = 0.5 * (
                    ops.sum(vel1_in**2, axis=-1) + ops.sum(vel2_in**2, axis=-1)
                )
                # Penalize energy increase
                energy_increase = ops.maximum(ke - ke_in, 0.0)
                energy_loss = ops.mean(energy_increase)
            else:
                energy_loss = 0.0

            # 3. Collision constraint
            # Balls shouldn't overlap too much
            distance = ops.sqrt(ops.sum((pos1 - pos2) ** 2, axis=-1))
            min_distance = 1.0  # Assuming unit radius
            overlap = ops.maximum(min_distance - distance, 0.0)
            collision_loss = ops.mean(overlap**2)

            return momentum_loss + energy_loss + collision_loss

        return ops.zeros(())

    def compute_adaptation_loss(self, x: Any, y_pred: Any) -> Any:
        """Compute combined adaptation loss with physics constraints."""
        # Get base regression losses
        base_loss = super().compute_adaptation_loss(x, y_pred)

        # Add physics constraints
        physics_loss = self.compute_physics_constraints(x, y_pred)

        return base_loss + self.physics_weight * physics_loss
