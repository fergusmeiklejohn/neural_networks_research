"""TENT adaptation modified for regression tasks like physics prediction."""

from typing import Any

import keras
from keras import ops

from . import BaseTTA


class TENTRegression(BaseTTA):
    """Test-time Entropy Minimization adapted for regression.

    Since entropy doesn't apply to continuous outputs, we use:
    1. Prediction variance minimization (confident predictions)
    2. Temporal consistency for sequences
    3. Smoothness regularization
    """

    def __init__(
        self,
        model: keras.Model,
        adaptation_steps: int = 1,
        learning_rate: float = 1e-3,
        reset_after_batch: bool = True,
        update_bn_only: bool = True,
        consistency_weight: float = 0.1,
        **kwargs,
    ):
        """Initialize TENT for regression.

        Args:
            model: Model to adapt
            adaptation_steps: Number of adaptation steps
            learning_rate: Learning rate for adaptation
            reset_after_batch: Whether to reset after each batch
            update_bn_only: Only update BatchNorm parameters
            consistency_weight: Weight for temporal consistency loss
        """
        super().__init__(
            model, adaptation_steps, learning_rate, reset_after_batch, **kwargs
        )

        self.update_bn_only = update_bn_only
        self.consistency_weight = consistency_weight

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
                    # Also ensure the layer is in training mode
                    if hasattr(layer, "training"):
                        layer.training = True
                else:
                    layer.trainable = False

    def compute_adaptation_loss(self, x: Any, y_pred: Any) -> Any:
        """Compute adaptation loss for regression.

        For regression, we minimize:
        1. Prediction variance (uncertainty)
        2. Temporal inconsistency
        3. Output roughness

        Args:
            x: Input data
            y_pred: Model predictions

        Returns:
            Adaptation loss
        """
        # 1. Variance minimization (confidence)
        # For trajectory prediction, minimize variance across different dimensions
        if len(ops.shape(y_pred)) == 3:  # (batch, time, features)
            # Variance across time dimension
            pred_mean = ops.mean(y_pred, axis=1, keepdims=True)
            variance = ops.mean((y_pred - pred_mean) ** 2)
        else:
            # Simple variance
            variance = ops.var(y_pred)

        # 2. Temporal consistency (if sequence)
        consistency_loss = 0.0
        if len(ops.shape(y_pred)) == 3 and ops.shape(y_pred)[1] > 1:
            # Penalize large changes between consecutive timesteps
            diff = y_pred[:, 1:] - y_pred[:, :-1]
            consistency_loss = ops.mean(diff**2)

        # 3. Smoothness regularization
        # Penalize very large values (physics should be bounded)
        smoothness = ops.mean(ops.abs(y_pred))

        # Combine losses
        total_loss = (
            variance + self.consistency_weight * consistency_loss + 0.01 * smoothness
        )

        return total_loss

    def adapt_step_augmented(self, x: Any) -> Any:
        """Adaptation step with augmented samples.

        For regression, we can create augmented views by:
        1. Adding small noise
        2. Temporal shifts
        3. Interpolation
        """
        # Get base prediction
        y_pred = self.model(x, training=True)

        # Create augmented versions
        noise_scale = 0.01
        x_aug = x + ops.random.normal(ops.shape(x)) * noise_scale

        # Get augmented prediction
        y_aug = self.model(x_aug, training=True)

        # Consistency between original and augmented
        aug_loss = ops.mean((y_pred - y_aug) ** 2)

        # Regular adaptation loss
        adapt_loss = self.compute_adaptation_loss(x, y_pred)

        return adapt_loss + 0.5 * aug_loss


class PhysicsTENTRegression(TENTRegression):
    """Physics-aware TENT for regression."""

    def __init__(
        self,
        model: keras.Model,
        adaptation_steps: int = 5,
        learning_rate: float = 1e-3,
        physics_loss_weight: float = 0.1,
        **kwargs,
    ):
        super().__init__(model, adaptation_steps, learning_rate, **kwargs)
        self.physics_loss_weight = physics_loss_weight

    def compute_adaptation_loss(self, x: Any, y_pred: Any) -> Any:
        """Add physics constraints to adaptation loss."""
        # Base regression loss
        base_loss = super().compute_adaptation_loss(x, y_pred)

        # Physics consistency
        physics_loss = self.compute_physics_loss(x, y_pred)

        return base_loss + self.physics_loss_weight * physics_loss

    def compute_physics_loss(self, x: Any, y_pred: Any) -> Any:
        """Compute physics-based consistency loss.

        For 2-ball trajectories:
        - Positions should follow smooth paths
        - Velocities should change smoothly (acceleration bounds)
        - Energy should be approximately conserved
        """
        if len(ops.shape(y_pred)) != 3:
            return 0.0

        # Extract positions and velocities
        # Assuming format: [x1, y1, vx1, vy1, x2, y2, vx2, vy2]
        pos1 = y_pred[:, :, 0:2]  # Ball 1 position
        vel1 = y_pred[:, :, 2:4]  # Ball 1 velocity
        pos2 = y_pred[:, :, 4:6]  # Ball 2 position
        vel2 = y_pred[:, :, 6:8]  # Ball 2 velocity

        # 1. Velocity consistency (position derivative should match velocity)
        if ops.shape(y_pred)[1] > 1:
            dt = 1 / 60.0  # Assuming 60 FPS
            pos1_diff = (pos1[:, 1:] - pos1[:, :-1]) / dt
            pos2_diff = (pos2[:, 1:] - pos2[:, :-1]) / dt

            vel_consistency = ops.mean((pos1_diff - vel1[:, :-1]) ** 2)
            vel_consistency += ops.mean((pos2_diff - vel2[:, :-1]) ** 2)
        else:
            vel_consistency = 0.0

        # 2. Acceleration bounds (physics constraint)
        if ops.shape(y_pred)[1] > 1:
            acc1 = (vel1[:, 1:] - vel1[:, :-1]) / dt
            acc2 = (vel2[:, 1:] - vel2[:, :-1]) / dt

            # Acceleration should be reasonable (gravity + some variance)
            max_acc = 1000.0  # pixels/s^2
            acc_penalty = ops.mean(ops.maximum(ops.abs(acc1) - max_acc, 0))
            acc_penalty += ops.mean(ops.maximum(ops.abs(acc2) - max_acc, 0))
        else:
            acc_penalty = 0.0

        # 3. Distance constraint (balls shouldn't overlap or go too far)
        ball_dist = ops.sqrt(ops.sum((pos1 - pos2) ** 2, axis=-1))
        min_dist = 20.0  # 2 * ball_radius
        max_dist = 800.0  # screen width

        dist_penalty = ops.mean(ops.maximum(min_dist - ball_dist, 0))
        dist_penalty += ops.mean(ops.maximum(ball_dist - max_dist, 0))

        # Combine physics losses
        physics_loss = vel_consistency + 0.1 * acc_penalty + 0.1 * dist_penalty

        return physics_loss
