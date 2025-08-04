"""
Minimal Physics-Informed Neural Network starting from F=ma.
Based on lessons learned from PINN failure and GraphExtrap success.
"""

import keras
from keras import layers, ops


class MinimalPhysicsModel(keras.Model):
    """
    Minimal PINN that starts with Newton's laws and adds learnable corrections.

    Key principles:
    1. Use physics-aware features (polar coordinates)
    2. Predict accelerations, not positions
    3. Start with F=ma, add small corrections
    4. Conserve energy and angular momentum
    """

    def __init__(self, hidden_dim=64, **kwargs):
        super().__init__(**kwargs)

        # Small correction network
        self.correction_net = keras.Sequential(
            [
                layers.Dense(hidden_dim, activation="tanh"),
                layers.Dense(hidden_dim, activation="tanh"),
                layers.Dense(2),  # Correction to acceleration
            ]
        )

        # Learnable physics parameters (initialized to Earth gravity)
        self.gravity = self.add_weight(
            name="gravity",
            shape=(1,),
            initializer=keras.initializers.Constant(-9.8),
            trainable=True,
        )

        # Use absolute value to ensure friction is always positive
        self.friction_raw = self.add_weight(
            name="friction_raw",
            shape=(1,),
            initializer=keras.initializers.Constant(0.1),
            trainable=True,
        )

    @property
    def friction(self):
        """Ensure friction is always positive."""
        return ops.abs(self.friction_raw)

    def extract_physics_features(self, states):
        """Convert Cartesian to physics-aware features."""
        # states shape: (batch, time, 8) for 2 balls
        # Extract positions and velocities
        pos1 = states[..., :2]  # Ball 1 position
        pos2 = states[..., 2:4]  # Ball 2 position
        vel1 = states[..., 4:6]  # Ball 1 velocity
        vel2 = states[..., 6:8]  # Ball 2 velocity

        # Convert to polar coordinates
        r1 = ops.sqrt(ops.sum(pos1**2, axis=-1, keepdims=True))
        theta1 = ops.arctan2(pos1[..., 1:2], pos1[..., 0:1])

        r2 = ops.sqrt(ops.sum(pos2**2, axis=-1, keepdims=True))
        theta2 = ops.arctan2(pos2[..., 1:2], pos2[..., 0:1])

        # Radial and angular velocities
        v_r1 = ops.sum(pos1 * vel1, axis=-1, keepdims=True) / (r1 + 1e-6)
        v_theta1 = (
            pos1[..., 0:1] * vel1[..., 1:2] - pos1[..., 1:2] * vel1[..., 0:1]
        ) / (r1**2 + 1e-6)

        v_r2 = ops.sum(pos2 * vel2, axis=-1, keepdims=True) / (r2 + 1e-6)
        v_theta2 = (
            pos2[..., 0:1] * vel2[..., 1:2] - pos2[..., 1:2] * vel2[..., 0:1]
        ) / (r2**2 + 1e-6)

        # Angular momentum (conserved quantity)
        L1 = r1 * v_theta1
        L2 = r2 * v_theta2

        # Ball-to-ball distance
        r12 = ops.sqrt(ops.sum((pos2 - pos1) ** 2, axis=-1, keepdims=True))

        # Combine features
        features = ops.concatenate(
            [r1, theta1, v_r1, v_theta1, L1, r2, theta2, v_r2, v_theta2, L2, r12],
            axis=-1,
        )

        return features, (pos1, pos2, vel1, vel2)

    def compute_accelerations(self, states):
        """Compute accelerations using F=ma + learned corrections."""
        features, (pos1, pos2, vel1, vel2) = self.extract_physics_features(states)

        # Base physics: gravity and friction
        # F = mg (gravity) - bv (friction)
        # Convert gravity from m/s² to pixels/s²
        pixel_gravity = self.gravity[0] * 40.0  # 40 pixels per meter

        acc1_base = ops.stack(
            [
                ops.zeros_like(pos1[..., 0]),  # No x acceleration from gravity
                pixel_gravity
                * ops.ones_like(pos1[..., 1]),  # y acceleration = g (in pixels)
            ],
            axis=-1,
        )

        acc2_base = ops.stack(
            [ops.zeros_like(pos2[..., 0]), pixel_gravity * ops.ones_like(pos2[..., 1])],
            axis=-1,
        )

        # Add friction (opposes velocity)
        acc1_base = acc1_base - self.friction * vel1
        acc2_base = acc2_base - self.friction * vel2

        # Learned corrections (small)
        corrections = self.correction_net(features)

        # Split corrections for both balls
        corr1 = corrections[..., :2] * 0.1  # Scale down corrections
        corr2 = corrections[..., :2] * 0.1

        # Total acceleration
        acc1 = acc1_base + corr1
        acc2 = acc2_base + corr2

        return acc1, acc2

    def integrate_trajectory(self, initial_state, steps=50):
        """Integrate trajectory using computed accelerations."""
        dt = 1 / 30.0  # 30 FPS
        pixel_to_meter = 40.0  # Conversion factor

        # Initialize
        states = []
        state = initial_state

        for _ in range(steps):
            states.append(state)

            # Get positions and velocities
            pos1 = state[..., :2]
            pos2 = state[..., 2:4]
            vel1 = state[..., 4:6]
            vel2 = state[..., 6:8]

            # Compute accelerations
            acc1, acc2 = self.compute_accelerations(state[..., None, :])
            acc1 = ops.squeeze(acc1, axis=-2)
            acc2 = ops.squeeze(acc2, axis=-2)

            # Symplectic integration (preserves energy better)
            # Update velocities first
            new_vel1 = vel1 + acc1 * dt
            new_vel2 = vel2 + acc2 * dt

            # Then update positions
            new_pos1 = pos1 + new_vel1 * dt
            new_pos2 = pos2 + new_vel2 * dt

            # Handle boundaries (elastic collision)
            # Process ball 1
            mask_x1 = ops.logical_or(new_pos1[..., 0] < 20, new_pos1[..., 0] > 780)
            new_vel1_x = ops.where(mask_x1, -0.8 * new_vel1[..., 0], new_vel1[..., 0])
            new_pos1_x = ops.clip(new_pos1[..., 0], 20, 780)

            mask_y1 = ops.logical_or(new_pos1[..., 1] < 20, new_pos1[..., 1] > 580)
            new_vel1_y = ops.where(mask_y1, -0.8 * new_vel1[..., 1], new_vel1[..., 1])
            new_pos1_y = ops.clip(new_pos1[..., 1], 20, 580)

            # Create updated arrays
            new_pos1 = ops.stack([new_pos1_x, new_pos1_y], axis=-1)
            new_vel1 = ops.stack([new_vel1_x, new_vel1_y], axis=-1)

            # Process ball 2
            mask_x2 = ops.logical_or(new_pos2[..., 0] < 20, new_pos2[..., 0] > 780)
            new_vel2_x = ops.where(mask_x2, -0.8 * new_vel2[..., 0], new_vel2[..., 0])
            new_pos2_x = ops.clip(new_pos2[..., 0], 20, 780)

            mask_y2 = ops.logical_or(new_pos2[..., 1] < 20, new_pos2[..., 1] > 580)
            new_vel2_y = ops.where(mask_y2, -0.8 * new_vel2[..., 1], new_vel2[..., 1])
            new_pos2_y = ops.clip(new_pos2[..., 1], 20, 580)

            # Create updated arrays
            new_pos2 = ops.stack([new_pos2_x, new_pos2_y], axis=-1)
            new_vel2 = ops.stack([new_vel2_x, new_vel2_y], axis=-1)

            # Combine new state
            new_state = ops.concatenate(
                [new_pos1, new_pos2, new_vel1, new_vel2], axis=-1
            )
            state = new_state

        return ops.stack(states, axis=-2)

    def call(self, inputs, training=None):
        """Forward pass: integrate trajectory from initial state."""
        # inputs shape: (batch, time, 8)
        # Use first timestep as initial condition
        initial_state = inputs[:, 0, :]

        # Integrate forward
        predicted_trajectory = self.integrate_trajectory(
            initial_state, steps=inputs.shape[1]
        )

        return predicted_trajectory

    def compute_physics_losses(self, y_true, y_pred):
        """Compute physics-based losses."""
        losses = {}

        # 1. Energy conservation
        _, (pos1_true, pos2_true, vel1_true, vel2_true) = self.extract_physics_features(
            y_true
        )
        _, (pos1_pred, pos2_pred, vel1_pred, vel2_pred) = self.extract_physics_features(
            y_pred
        )

        # Kinetic energy
        KE_true = 0.5 * (
            ops.sum(vel1_true**2, axis=-1) + ops.sum(vel2_true**2, axis=-1)
        )
        KE_pred = 0.5 * (
            ops.sum(vel1_pred**2, axis=-1) + ops.sum(vel2_pred**2, axis=-1)
        )

        # Potential energy (gravity) - use pixel units
        pixel_gravity = self.gravity[0] * 40.0
        PE_true = -pixel_gravity * (pos1_true[..., 1] + pos2_true[..., 1])
        PE_pred = -pixel_gravity * (pos1_pred[..., 1] + pos2_pred[..., 1])

        # Total energy
        KE_true + PE_true
        E_pred = KE_pred + PE_pred

        # Energy should be approximately conserved (allowing for friction)
        energy_loss = ops.mean(ops.square(E_pred[:, 1:] - E_pred[:, :-1]))
        losses["energy"] = energy_loss

        # 2. Angular momentum conservation (without external torques)
        features_true, _ = self.extract_physics_features(y_true)
        features_pred, _ = self.extract_physics_features(y_pred)

        L1_true = features_true[..., 4]  # Angular momentum ball 1
        L2_true = features_true[..., 9]  # Angular momentum ball 2
        L1_pred = features_pred[..., 4]
        L2_pred = features_pred[..., 9]

        # Total angular momentum
        L1_true + L2_true
        L_total_pred = L1_pred + L2_pred

        # Should be approximately conserved
        momentum_loss = ops.mean(ops.square(L_total_pred[:, 1:] - L_total_pred[:, :-1]))
        losses["angular_momentum"] = momentum_loss

        # 3. Trajectory smoothness
        acc1_pred, acc2_pred = self.compute_accelerations(y_pred)
        jerk1 = acc1_pred[:, 1:] - acc1_pred[:, :-1]
        jerk2 = acc2_pred[:, 1:] - acc2_pred[:, :-1]

        smoothness_loss = ops.mean(ops.square(jerk1)) + ops.mean(ops.square(jerk2))
        losses["smoothness"] = smoothness_loss

        return losses
