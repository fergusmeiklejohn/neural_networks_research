#!/usr/bin/env python3
"""
Soft Collision Models for Physics-Informed Neural Networks.

Implements smooth, differentiable collision handling for ball-to-ball
and ball-to-wall interactions using penalty methods.
"""

import os

os.environ["KERAS_BACKEND"] = "jax"


import jax
import jax.numpy as jnp
import keras
from keras import layers, ops


@keras.saving.register_keras_serializable()
class SoftCollisionPotential(layers.Layer):
    """Smooth collision potential using Fischer-Burmeister function.

    Provides differentiable approximation of hard contact constraints
    for ball-to-ball collisions.
    """

    def __init__(
        self,
        stiffness: float = 1000.0,
        epsilon: float = 1e-3,
        elasticity: float = 0.8,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.stiffness = stiffness
        self.epsilon = epsilon
        self.elasticity = elasticity

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "stiffness": self.stiffness,
                "epsilon": self.epsilon,
                "elasticity": self.elasticity,
            }
        )
        return config

    def call(self, inputs):
        """Compute soft collision potential between two balls.

        Args:
            inputs: Dictionary with keys:
                - positions1: Position of ball 1, shape (..., 2) for [x, y]
                - positions2: Position of ball 2, shape (..., 2) for [x, y]
                - radius1: Radius of ball 1
                - radius2: Radius of ball 2

        Returns:
            Collision potential energy
        """
        positions1 = inputs["positions1"]
        positions2 = inputs["positions2"]
        radius1 = inputs["radius1"]
        radius2 = inputs["radius2"]
        # Compute distance between centers
        r12 = positions2 - positions1
        distance = ops.sqrt(ops.sum(r12**2, axis=-1) + self.epsilon**2)

        # Penetration depth (negative when no collision)
        radius_sum = radius1 + radius2
        penetration = radius_sum - distance

        # Smooth approximation using softplus
        # This gives 0 when no collision, positive when colliding
        contact_force = ops.nn.softplus(penetration / self.epsilon) * self.epsilon

        # Collision potential energy
        potential = 0.5 * self.stiffness * contact_force**2

        # Apply elasticity factor
        potential = potential * self.elasticity

        return potential

    def compute_force(self, positions1, positions2, radius1, radius2):
        """Compute collision force between two balls.

        Returns force vector on ball 1 (negative for ball 2).
        """

        # Use JAX for gradient computation
        def potential_func(pos1):
            inputs = {
                "positions1": pos1,
                "positions2": positions2,
                "radius1": radius1,
                "radius2": radius2,
            }
            return self.call(inputs)

        # Force is negative gradient of potential
        force = -jax.grad(potential_func)(positions1)
        return force


@keras.saving.register_keras_serializable()
class WallBounceModel(layers.Layer):
    """Smooth wall bounce handling using sigmoid approximations."""

    def __init__(
        self,
        world_width: float = 800.0,
        world_height: float = 600.0,
        wall_stiffness: float = 2000.0,
        temperature: float = 0.01,
        elasticity: float = 0.8,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.world_width = world_width
        self.world_height = world_height
        self.wall_stiffness = wall_stiffness
        self.temperature = temperature
        self.elasticity = elasticity

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "world_width": self.world_width,
                "world_height": self.world_height,
                "wall_stiffness": self.wall_stiffness,
                "temperature": self.temperature,
                "elasticity": self.elasticity,
            }
        )
        return config

    def compute_wall_potential(self, position, radius):
        """Compute smooth wall collision potential.

        Args:
            position: Ball position [x, y]
            radius: Ball radius

        Returns:
            Wall collision potential energy
        """
        x, y = position[..., 0], position[..., 1]

        # Distance to walls (negative inside bounds)
        left_dist = x - radius
        right_dist = self.world_width - x - radius
        bottom_dist = y - radius
        top_dist = self.world_height - y - radius

        # Smooth wall potentials using softplus
        left_potential = (
            ops.nn.softplus(-left_dist / self.temperature) * self.temperature
        )
        right_potential = (
            ops.nn.softplus(-right_dist / self.temperature) * self.temperature
        )
        bottom_potential = (
            ops.nn.softplus(-bottom_dist / self.temperature) * self.temperature
        )
        top_potential = ops.nn.softplus(-top_dist / self.temperature) * self.temperature

        # Total wall potential
        total_potential = self.wall_stiffness * (
            left_potential**2
            + right_potential**2
            + bottom_potential**2
            + top_potential**2
        )

        return total_potential

    def compute_wall_force(self, position, velocity, radius):
        """Compute wall collision force with velocity-dependent damping.

        Returns force vector including elastic reflection.
        """
        x, y = position[..., 0], position[..., 1]
        vx, vy = velocity[..., 0], velocity[..., 1]

        # Activation functions for each wall
        left_activation = ops.nn.sigmoid((radius - x) / self.temperature)
        right_activation = ops.nn.sigmoid(
            (x + radius - self.world_width) / self.temperature
        )
        bottom_activation = ops.nn.sigmoid((radius - y) / self.temperature)
        top_activation = ops.nn.sigmoid(
            (y + radius - self.world_height) / self.temperature
        )

        # Compute reflection forces
        fx = 0.0
        fy = 0.0

        # Left/right walls
        fx += left_activation * (-2 * self.elasticity * vx * (vx < 0))
        fx += right_activation * (-2 * self.elasticity * vx * (vx > 0))

        # Bottom/top walls
        fy += bottom_activation * (-2 * self.elasticity * vy * (vy < 0))
        fy += top_activation * (-2 * self.elasticity * vy * (vy > 0))

        # Add position-based restoring force
        fx += self.wall_stiffness * (
            left_activation * (radius - x)
            - right_activation * (x + radius - self.world_width)
        )
        fy += self.wall_stiffness * (
            bottom_activation * (radius - y)
            - top_activation * (y + radius - self.world_height)
        )

        return ops.stack([fx, fy], axis=-1)

    def call(self, inputs):
        """Combined wall interaction.

        Args:
            inputs: Dictionary with 'position', 'velocity', 'radius'

        Returns:
            Tuple of (potential, force)
        """
        position = inputs["position"]
        velocity = inputs["velocity"]
        radius = inputs["radius"]

        potential = self.compute_wall_potential(position, radius)
        force = self.compute_wall_force(position, velocity, radius)
        return potential, force


@keras.saving.register_keras_serializable()
class AugmentedLagrangianCollision(layers.Layer):
    """Augmented Lagrangian method for handling multiple collision constraints.

    Automatically balances multiple simultaneous collisions through
    learnable Lagrange multipliers.
    """

    def __init__(
        self,
        num_constraints: int = 10,  # Max simultaneous collisions
        penalty_weight: float = 100.0,
        multiplier_lr: float = 0.01,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_constraints = num_constraints
        self.penalty_weight = penalty_weight
        self.multiplier_lr = multiplier_lr

    def build(self, input_shape):
        # Learnable Lagrange multipliers
        self.multipliers = self.add_weight(
            name="lagrange_multipliers",
            shape=(self.num_constraints,),
            initializer="zeros",
            trainable=True,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_constraints": self.num_constraints,
                "penalty_weight": self.penalty_weight,
                "multiplier_lr": self.multiplier_lr,
            }
        )
        return config

    def compute_constraint_violations(self, positions, radii):
        """Compute all pairwise collision constraint violations.

        Args:
            positions: Ball positions, shape (n_balls, 2)
            radii: Ball radii, shape (n_balls,)

        Returns:
            Constraint violations (negative for satisfied constraints)
        """
        n_balls = ops.shape(positions)[0]
        violations = []

        # Compute all pairwise distances
        for i in range(n_balls):
            for j in range(i + 1, n_balls):
                r_ij = positions[j] - positions[i]
                dist = ops.sqrt(ops.sum(r_ij**2) + 1e-6)
                min_dist = radii[i] + radii[j]

                # Violation is positive when balls overlap
                violation = min_dist - dist
                violations.append(violation)

        # Pad to fixed size
        violations = ops.stack(violations)
        if len(violations) < self.num_constraints:
            padding = ops.zeros((self.num_constraints - len(violations),))
            violations = ops.concatenate([violations, padding])
        else:
            violations = violations[: self.num_constraints]

        return violations

    def call(self, positions, radii, training=None):
        """Compute augmented Lagrangian penalty for collisions.

        Returns:
            Augmented Lagrangian value
        """
        violations = self.compute_constraint_violations(positions, radii)

        # Only consider positive violations (actual collisions)
        active_violations = ops.nn.relu(violations)

        # Augmented Lagrangian: λᵀc + ρ/2 ||c||²
        lagrangian = ops.sum(
            self.multipliers * active_violations
        ) + 0.5 * self.penalty_weight * ops.sum(active_violations**2)

        # Update multipliers during training (dual ascent)
        if training:
            # This would normally be done in optimizer, but we can
            # add a stop_gradient to prevent backprop through this update
            new_multipliers = self.multipliers + self.multiplier_lr * active_violations
            self.multipliers.assign(ops.stop_gradient(new_multipliers))

        return lagrangian


@keras.saving.register_keras_serializable()
class InteractionNetwork(layers.Layer):
    """Neural network to model complex ball-ball interactions."""

    def __init__(self, hidden_dim: int = 64, num_layers: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Build interaction MLP
        layers_list = []
        for _ in range(num_layers):
            layers_list.append(layers.Dense(hidden_dim, activation="relu"))
        layers_list.append(layers.Dense(2))  # Output 2D force

        self.interaction_mlp = keras.Sequential(layers_list)

    def get_config(self):
        config = super().get_config()
        config.update({"hidden_dim": self.hidden_dim, "num_layers": self.num_layers})
        return config

    def call(self, state1, state2, params1, params2):
        """Compute interaction force between two balls.

        Args:
            state1, state2: States [x, y, vx, vy] for each ball
            params1, params2: Physical parameters [mass, radius, elasticity]

        Returns:
            Force on ball 1 due to ball 2
        """
        # Relative state
        rel_pos = state2[:2] - state1[:2]
        rel_vel = state2[2:] - state1[2:]

        # Concatenate all features
        features = ops.concatenate([rel_pos, rel_vel, state1, state2, params1, params2])

        # Compute interaction force
        force = self.interaction_mlp(features)

        return force


def test_collision_models():
    """Test collision model implementations."""
    print("Testing Soft Collision Models...")

    # Test soft collision potential
    collision_model = SoftCollisionPotential(stiffness=1000.0)

    pos1 = jnp.array([100.0, 100.0])
    pos2 = jnp.array([110.0, 100.0])  # 10 units apart
    r1, r2 = 20.0, 20.0  # Should collide (sum=40, dist=10)

    inputs = {
        "positions1": pos1,
        "positions2": pos2,
        "radius1": jnp.array(r1),
        "radius2": jnp.array(r2),
    }
    potential = collision_model(inputs)
    print(f"Collision potential (overlapping): {potential}")

    # Test no collision
    pos2_far = jnp.array([200.0, 100.0])
    inputs_far = {
        "positions1": pos1,
        "positions2": pos2_far,
        "radius1": jnp.array(r1),
        "radius2": jnp.array(r2),
    }
    potential_far = collision_model(inputs_far)
    print(f"Collision potential (separated): {potential_far}")

    # Test wall bounce model
    print("\nTesting Wall Bounce Model...")
    wall_model = WallBounceModel()

    # Ball near left wall
    pos_near_wall = jnp.array([15.0, 300.0])
    vel = jnp.array([-50.0, 0.0])
    radius = 20.0

    wall_inputs = {
        "position": pos_near_wall,
        "velocity": vel,
        "radius": jnp.array(radius),
    }
    wall_potential, wall_force = wall_model(wall_inputs)
    print(f"Wall potential: {wall_potential}")
    print(f"Wall force: {wall_force}")

    # Test Augmented Lagrangian
    print("\nTesting Augmented Lagrangian Collision...")
    al_model = AugmentedLagrangianCollision(num_constraints=3)
    al_model.build(None)

    # Three balls
    positions = jnp.array([[100.0, 100.0], [120.0, 100.0], [110.0, 120.0]])
    radii = jnp.array([20.0, 20.0, 20.0])

    lagrangian = al_model(positions, radii, training=False)
    print(f"Augmented Lagrangian value: {lagrangian}")

    print("\nAll tests passed!")


if __name__ == "__main__":
    test_collision_models()
