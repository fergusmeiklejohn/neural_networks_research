"""Test-Time Training (TTT) for physics prediction tasks.

Implements test-time training specifically designed for physics trajectory prediction,
with focus on adapting to time-varying physical parameters.
"""

from typing import Any, Dict, List, Tuple

import keras
from keras import ops

from . import BaseTTA


class PhysicsTTT(BaseTTA):
    """Test-Time Training for physics prediction.

    This implementation uses auxiliary self-supervised tasks during test time
    to adapt the model to new physics regimes (e.g., time-varying gravity).
    """

    def __init__(
        self,
        model: keras.Model,
        adaptation_steps: int = 10,
        learning_rate: float = 1e-4,
        reset_after_batch: bool = False,
        auxiliary_tasks: List[str] = None,
        trajectory_length: int = 10,
        adaptation_window: int = 5,
        **kwargs,
    ):
        """Initialize PhysicsTTT.

        Args:
            model: Base physics prediction model
            adaptation_steps: Number of TTT steps
            learning_rate: Learning rate for adaptation
            reset_after_batch: Whether to reset (False for online adaptation)
            auxiliary_tasks: List of auxiliary tasks to use
            trajectory_length: Expected trajectory length
            adaptation_window: Number of timesteps to use for adaptation
        """
        super().__init__(
            model, adaptation_steps, learning_rate, reset_after_batch, **kwargs
        )

        self.auxiliary_tasks = auxiliary_tasks or [
            "reconstruction",
            "consistency",
            "smoothness",
        ]
        self.trajectory_length = trajectory_length
        self.adaptation_window = adaptation_window

        # Track adaptation state for online learning
        self.adaptation_state = {
            "observed_trajectory": [],
            "estimated_physics": None,
            "adaptation_count": 0,
        }

    def compute_reconstruction_loss(
        self, trajectory: Any, start_idx: int = 0
    ) -> Tuple[Any, Any]:
        """Reconstruction task: predict masked timesteps.

        Args:
            trajectory: Observed trajectory segment
            start_idx: Starting index for masking

        Returns:
            Tuple of (loss, predictions)
        """
        # Check if we have enough timesteps for reconstruction
        n_timesteps = ops.shape(trajectory)[1]

        if n_timesteps < 2:
            # Not enough timesteps for reconstruction - return dummy loss
            # This happens when adapting on single timestep inputs
            predictions = self.model(trajectory, training=True)
            # Return small dummy loss to avoid NaN
            return ops.ones((1,)) * 0.01, predictions

        # Use first half to predict second half
        mid_point = n_timesteps // 2

        context = trajectory[:, :mid_point]
        target = trajectory[:, mid_point:]

        # Predict future from context
        predictions = self.model(context, training=True)

        # Ensure predictions match target shape
        if ops.shape(predictions)[1] > ops.shape(target)[1]:
            predictions = predictions[:, : ops.shape(target)[1]]

        # MSE loss
        loss = ops.mean((predictions - target) ** 2)

        return loss, predictions

    def compute_consistency_loss(self, trajectory: Any) -> Any:
        """Physics consistency loss.

        Ensures predictions follow physical laws.
        """
        # Check if we have enough timesteps
        if ops.shape(trajectory)[1] < 2:
            # Not enough timesteps for consistency check
            return ops.ones((1,)) * 0.01

        # Forward prediction
        pred_forward = self.model(trajectory[:, :-1], training=True)

        # Reverse prediction (if model supports it)
        try:
            # Reverse time direction
            reversed_traj = trajectory[:, ::-1]
            pred_reverse = self.model(reversed_traj[:, :-1], training=True)
            pred_reverse = pred_reverse[:, ::-1]

            # Consistency: forward and reverse should match
            consistency_loss = ops.mean((pred_forward - pred_reverse) ** 2)
        except:
            # Fallback: just check smoothness
            consistency_loss = ops.mean(ops.abs(ops.diff(pred_forward, axis=1)))

        return consistency_loss

    def compute_smoothness_loss(self, trajectory: Any) -> Any:
        """Trajectory smoothness loss.

        Penalizes non-smooth trajectories.
        """
        # Compute accelerations (second derivative)
        positions = trajectory[..., [0, 1, 4, 5]]  # x,y positions
        velocities = ops.diff(positions, axis=1)
        accelerations = ops.diff(velocities, axis=1)

        # Smooth trajectories have small acceleration changes
        jerk = ops.diff(accelerations, axis=1)
        smoothness_loss = ops.mean(ops.abs(jerk))

        return smoothness_loss

    def estimate_physics_parameters(self, trajectory: Any) -> Dict[str, float]:
        """Estimate physical parameters from observed trajectory.

        Args:
            trajectory: Observed trajectory

        Returns:
            Dictionary of estimated parameters
        """
        # Extract vertical positions and velocities
        y_positions = trajectory[..., [1, 5]]  # y for both balls
        y_velocities = ops.diff(y_positions, axis=1) / 0.1  # assuming dt=0.1
        y_accelerations = ops.diff(y_velocities, axis=1) / 0.1

        # Estimate gravity from average vertical acceleration
        # (simplified - assumes no air resistance)
        avg_gravity = -ops.mean(y_accelerations)

        # Estimate if gravity is time-varying
        gravity_std = ops.std(y_accelerations)
        is_time_varying = gravity_std > 0.5  # threshold

        return {
            "gravity": float(avg_gravity),
            "gravity_std": float(gravity_std),
            "time_varying": bool(is_time_varying),
        }

    def compute_adaptation_loss(self, x: Any, y_pred: Any) -> Any:
        """Compute combined loss for test-time training.

        Args:
            x: Input trajectory segment
            y_pred: Model predictions (not used directly)

        Returns:
            Combined auxiliary task loss
        """
        total_loss = 0.0
        loss_weights = {"reconstruction": 1.0, "consistency": 0.5, "smoothness": 0.1}

        if "reconstruction" in self.auxiliary_tasks:
            recon_loss, _ = self.compute_reconstruction_loss(x)
            total_loss = total_loss + loss_weights["reconstruction"] * recon_loss

        if "consistency" in self.auxiliary_tasks:
            cons_loss = self.compute_consistency_loss(x)
            total_loss = total_loss + loss_weights["consistency"] * cons_loss

        if "smoothness" in self.auxiliary_tasks:
            smooth_loss = self.compute_smoothness_loss(x)
            total_loss = total_loss + loss_weights["smoothness"] * smooth_loss

        return total_loss

    def adapt_online(self, new_observation: Any) -> Dict[str, Any]:
        """Online adaptation with new observations.

        Args:
            new_observation: New timestep observation

        Returns:
            Dictionary with adaptation results
        """
        # Add to observation buffer
        self.adaptation_state["observed_trajectory"].append(new_observation)

        # Only adapt when we have enough observations
        if len(self.adaptation_state["observed_trajectory"]) >= self.adaptation_window:
            # Create trajectory tensor
            trajectory = ops.stack(
                self.adaptation_state["observed_trajectory"][-self.adaptation_window :]
            )
            trajectory = ops.expand_dims(trajectory, axis=0)  # Add batch dimension

            # Estimate physics parameters
            physics_params = self.estimate_physics_parameters(trajectory)
            self.adaptation_state["estimated_physics"] = physics_params

            # Perform adaptation
            self.adapt(trajectory)
            self.adaptation_state["adaptation_count"] += 1

            # Slide window
            if (
                len(self.adaptation_state["observed_trajectory"])
                > self.trajectory_length
            ):
                self.adaptation_state["observed_trajectory"].pop(0)

        return self.adaptation_state

    def predict_next(self, context: Any, adapt: bool = True) -> Any:
        """Predict next timestep with optional adaptation.

        Args:
            context: Context trajectory
            adapt: Whether to perform test-time adaptation

        Returns:
            Next timestep prediction
        """
        if adapt and ops.shape(context)[1] >= self.adaptation_window:
            # Use recent context for adaptation
            adapt_context = context[:, -self.adaptation_window :]
            self.adapt(adapt_context)

        # Make prediction
        prediction = self.model(context, training=False)

        # Extract next timestep
        if len(ops.shape(prediction)) == 3:
            next_step = prediction[:, 0]  # First future timestep
        else:
            next_step = prediction

        return next_step
