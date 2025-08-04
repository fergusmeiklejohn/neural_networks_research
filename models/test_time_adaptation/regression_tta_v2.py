"""Improved Test-Time Adaptation methods for regression with full JAX gradient support."""

from typing import Any, List

import keras
from keras import ops

from .base_tta_jax_v2 import BaseTTAJaxV2


class RegressionTTAV2(BaseTTAJaxV2):
    """Improved TTA for regression tasks with full gradient support.

    This version properly computes gradients using JAX and can update
    all parameters, not just BatchNorm statistics.
    """

    def __init__(
        self,
        model: keras.Model,
        adaptation_steps: int = 5,
        learning_rate: float = 1e-4,
        reset_after_batch: bool = True,
        consistency_weight: float = 1.0,
        smoothness_weight: float = 0.1,
        update_bn_only: bool = False,  # Changed default to False
        **kwargs,
    ):
        """Initialize RegressionTTAV2.

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

        # Identify which parameters to adapt
        self._setup_adaptable_params()

    def _setup_adaptable_params(self):
        """Setup which parameters should be adapted."""
        self.bn_param_indices = []
        self.adaptable_param_indices = []

        if self.update_bn_only:
            # Find BatchNorm parameters
            for i, var in enumerate(self.model.trainable_variables):
                # Check if this variable belongs to a BatchNorm layer
                for layer in self.model.layers:
                    if isinstance(layer, keras.layers.BatchNormalization):
                        if (hasattr(layer, "gamma") and var is layer.gamma) or (
                            hasattr(layer, "beta") and var is layer.beta
                        ):
                            self.bn_param_indices.append(i)
                            self.adaptable_param_indices.append(i)
                            break
        else:
            # Adapt all trainable parameters
            self.adaptable_param_indices = list(
                range(len(self.model.trainable_variables))
            )

    def _get_adaptable_params_mask(self) -> List[bool]:
        """Get mask indicating which parameters should be adapted."""
        mask = [False] * len(self.model.trainable_variables)
        for idx in self.adaptable_param_indices:
            mask[idx] = True
        return mask

    def compute_adaptation_loss(self, x: Any, y_pred: Any) -> Any:
        """Compute self-supervised loss for regression adaptation.

        Args:
            x: Input data
            y_pred: Model predictions

        Returns:
            Combined adaptation loss
        """
        total_loss = 0.0

        # 1. Prediction consistency loss
        if len(ops.shape(x)) > 0 and ops.shape(x)[0] > 1:
            # Multiple samples: encourage consistent predictions
            pred_mean = ops.mean(y_pred, axis=0, keepdims=True)
            # Use JAX operations for compatibility
            consistency_loss = ops.mean(ops.square(y_pred - pred_mean))
            total_loss = total_loss + self.consistency_weight * consistency_loss

        # 2. Temporal smoothness loss (for trajectory data)
        if len(ops.shape(y_pred)) == 3:  # (batch, time, features)
            # Minimize changes between consecutive timesteps
            diff = y_pred[:, 1:, :] - y_pred[:, :-1, :]
            smoothness_loss = ops.mean(ops.square(diff))
            total_loss = total_loss + self.smoothness_weight * smoothness_loss
        elif len(ops.shape(y_pred)) == 2 and ops.shape(x)[0] > 1:
            # For single-step predictions, minimize variance
            pred_std = ops.std(y_pred, axis=0)
            smoothness_loss = ops.mean(pred_std)
            total_loss = total_loss + self.smoothness_weight * smoothness_loss

        # 3. L2 regularization on predictions (prevents drift)
        pred_norm = ops.mean(ops.square(y_pred))
        total_loss = total_loss + 0.001 * pred_norm

        return total_loss


class PhysicsRegressionTTAV2(RegressionTTAV2):
    """Physics-aware TTA for regression with full gradient support.

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
        update_bn_only: bool = False,
        **kwargs,
    ):
        """Initialize PhysicsRegressionTTAV2."""
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
            x: Input states [x1, y1, vx1, vy1, x2, y2, vx2, vy2]
            y_pred: Predicted next states

        Returns:
            Physics constraint loss
        """
        total_physics_loss = 0.0

        # For 2-ball physics
        if ops.shape(y_pred)[-1] >= 8:
            # Extract components
            pos1 = y_pred[..., 0:2]  # Ball 1 position
            vel1 = y_pred[..., 2:4]  # Ball 1 velocity
            pos2 = y_pred[..., 4:6]  # Ball 2 position
            vel2 = y_pred[..., 6:8]  # Ball 2 velocity

            # 1. Momentum conservation
            # In a closed system, total momentum should be conserved
            total_momentum = vel1 + vel2  # Assuming unit masses

            if ops.shape(x)[-1] >= 8:
                # Compare with input momentum
                vel1_in = x[..., 2:4]
                vel2_in = x[..., 6:8]
                total_momentum_in = vel1_in + vel2_in

                # Momentum change should be minimal
                momentum_change = ops.mean(
                    ops.square(total_momentum - total_momentum_in)
                )
                total_physics_loss = total_physics_loss + momentum_change

            # 2. Energy conservation/dissipation
            # Kinetic energy should not increase (can decrease due to friction)
            ke = 0.5 * (ops.sum(vel1**2, axis=-1) + ops.sum(vel2**2, axis=-1))

            if ops.shape(x)[-1] >= 8:
                vel1_in = x[..., 2:4]
                vel2_in = x[..., 6:8]
                ke_in = 0.5 * (
                    ops.sum(vel1_in**2, axis=-1) + ops.sum(vel2_in**2, axis=-1)
                )

                # Penalize energy increase
                energy_increase = ops.maximum(ke - ke_in, 0.0)
                energy_loss = ops.mean(energy_increase)
                total_physics_loss = total_physics_loss + energy_loss

            # 3. Minimum separation constraint
            # Balls shouldn't overlap (assuming unit radius)
            distance = ops.sqrt(ops.sum((pos1 - pos2) ** 2, axis=-1) + 1e-6)
            min_distance = 2.0  # Two unit radius balls
            overlap = ops.maximum(min_distance - distance, 0.0)
            collision_loss = ops.mean(overlap**2)
            total_physics_loss = total_physics_loss + 10.0 * collision_loss

            # 4. Bounded positions (balls should stay in reasonable bounds)
            max_position = 10.0  # Reasonable bound
            out_of_bounds = ops.maximum(
                ops.abs(pos1) - max_position, 0.0
            ) + ops.maximum(ops.abs(pos2) - max_position, 0.0)
            bounds_loss = ops.mean(out_of_bounds**2)
            total_physics_loss = total_physics_loss + bounds_loss

        return total_physics_loss

    def compute_adaptation_loss(self, x: Any, y_pred: Any) -> Any:
        """Compute combined adaptation loss with physics constraints."""
        # Get base regression losses
        base_loss = super().compute_adaptation_loss(x, y_pred)

        # Add physics constraints
        physics_loss = self.compute_physics_constraints(x, y_pred)

        return base_loss + self.physics_weight * physics_loss
