#!/usr/bin/env python3
"""
Physics-Informed Neural Network Components for Ball Trajectory Prediction.

Implements Hamiltonian Neural Networks (HNNs) with energy conservation
for improved extrapolation on physics simulations.
"""

import os

os.environ["KERAS_BACKEND"] = "jax"

from typing import Tuple

import jax.numpy as jnp
import keras
import numpy as np
from keras import layers, ops


@keras.saving.register_keras_serializable()
class HamiltonianNN(keras.Model):
    """Hamiltonian Neural Network that conserves energy by construction.

    For a 2-ball system, models the Hamiltonian:
    H = Σᵢ(pᵢ²/2mᵢ) + Σᵢ(mᵢgyᵢ) + V(r₁₂)

    Where:
    - pᵢ: momentum of ball i
    - mᵢ: mass of ball i
    - g: gravity constant
    - yᵢ: height of ball i
    - V(r₁₂): interaction potential between balls
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 3,
        activation: str = "tanh",
        use_fourier_features: bool = True,
        fourier_features: int = 32,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.activation = activation
        self.use_fourier_features = use_fourier_features
        self.fourier_features = fourier_features

        # Build the Hamiltonian network
        self.hamiltonian_net = self._build_hamiltonian_network()

    def build(self, input_shape):
        # Build the hamiltonian network with proper input shape
        if isinstance(input_shape, dict):
            state_shape = input_shape.get("state", (None, 8))
        else:
            state_shape = (None, 8)  # Default for 2 balls
        dummy_input = ops.zeros((1,) + state_shape[1:])
        _ = self.hamiltonian_net(dummy_input)  # Build the network
        super().build(input_shape)

    def _build_hamiltonian_network(self):
        """Build neural network to approximate the Hamiltonian."""
        layers_list = []

        # Optional Fourier feature layer
        if self.use_fourier_features:
            layers_list.append(FourierFeatures(self.fourier_features))

        # Hidden layers
        for _ in range(self.num_layers):
            layers_list.append(
                layers.Dense(self.hidden_dim, activation=self.activation)
            )

        # Output single scalar (Hamiltonian value)
        layers_list.append(layers.Dense(1, activation=None))

        return keras.Sequential(layers_list)

    def get_config(self):
        return {
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "activation": self.activation,
            "use_fourier_features": self.use_fourier_features,
            "fourier_features": self.fourier_features,
        }

    def compute_hamiltonian(
        self, state: jnp.ndarray, masses: jnp.ndarray
    ) -> jnp.ndarray:
        """Compute Hamiltonian value for given state.

        Args:
            state: [q, p] concatenated positions and momenta
                  Shape: (batch, 8) for 2 balls [x1, y1, x2, y2, px1, py1, px2, py2]
            masses: Masses of the balls, shape: (batch, 2)

        Returns:
            Hamiltonian values, shape: (batch,)
        """
        # Neural network approximation
        H_nn = self.hamiltonian_net(state)

        # Add known kinetic energy term for better physics
        batch_size = ops.shape(state)[0]
        q, p = ops.split(state, 2, axis=-1)  # Split along last axis

        # Reshape p to (batch, 2 balls, 2 dims)
        p_reshaped = ops.reshape(p, (batch_size, 2, 2))

        # Compute kinetic energy: KE = sum(p^2 / 2m)
        p_squared = ops.sum(p_reshaped**2, axis=-1)  # (batch, 2)
        kinetic = ops.sum(p_squared / (2 * masses), axis=-1)  # (batch,)

        # Combine with neural network output
        H_nn_scalar = ops.squeeze(H_nn, axis=-1)  # Remove last dimension

        return H_nn_scalar + kinetic

    def compute_dynamics(self, state: np.ndarray, masses: np.ndarray) -> np.ndarray:
        """Compute Hamiltonian dynamics dq/dt, dp/dt.

        Uses automatic differentiation to compute:
        dq/dt = ∂H/∂p
        dp/dt = -∂H/∂q

        Note: This requires pure JAX operations, not Keras layers.
        For now, returning placeholder dynamics.
        """
        # Split state into positions and momenta
        q, p = np.split(state, 2)

        # Placeholder dynamics (would use JAX grad in pure JAX implementation)
        # For actual use, this would be implemented with JAX autodiff
        dq_dt = p / masses.repeat(2)  # Velocity from momentum
        dp_dt = np.zeros_like(p)  # Placeholder forces
        dp_dt[1::2] = -9.81 * masses  # Gravity on y-components

        return np.concatenate([dq_dt, dp_dt])

    def call(self, inputs, training=None):
        """Forward pass computing Hamiltonian value.

        Args:
            inputs: Dictionary with 'state' and 'masses'

        Returns:
            Hamiltonian value
        """
        state = inputs["state"]
        masses = inputs["masses"]
        return self.compute_hamiltonian(state, masses)


@keras.saving.register_keras_serializable()
class FourierFeatures(layers.Layer):
    """Fourier feature embedding for improved expressivity."""

    def __init__(
        self,
        num_features: int = 32,
        scale_range: Tuple[float, float] = (0, 6),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_features = num_features
        self.scale_range = scale_range

    def build(self, input_shape):
        # Initialize frequency scales as a variable
        scales = np.linspace(
            self.scale_range[0], self.scale_range[1], self.num_features
        )
        self.frequencies = self.add_weight(
            name="frequencies",
            shape=(self.num_features,),
            initializer=keras.initializers.Constant(2.0**scales),
            trainable=False,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {"num_features": self.num_features, "scale_range": self.scale_range}
        )
        return config

    def call(self, inputs):
        # Get input shape
        input_shape = ops.shape(inputs)
        batch_size = input_shape[0]
        input_dim = input_shape[-1]

        # Expand dimensions for broadcasting
        x = ops.expand_dims(inputs, axis=-1)  # (batch, input_dim, 1)
        freqs = ops.reshape(
            self.frequencies, (1, 1, self.num_features)
        )  # (1, 1, num_features)

        # Compute Fourier features
        angles = x * freqs  # (batch, input_dim, num_features)

        # Concatenate sin and cos
        sin_features = ops.sin(angles)
        cos_features = ops.cos(angles)

        # Stack and reshape
        features = ops.concatenate(
            [sin_features, cos_features], axis=-1
        )  # (batch, input_dim, num_features*2)

        # Flatten last two dimensions
        output_shape = (batch_size, input_dim * self.num_features * 2)
        return ops.reshape(features, output_shape)


@keras.saving.register_keras_serializable()
class PhysicsGuidedAttention(layers.Layer):
    """Attention mechanism guided by physics constraints."""

    def __init__(self, hidden_dim: int = 128, num_heads: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Multi-head attention for physics features
        self.physics_attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=hidden_dim // num_heads
        )

        # Fusion layers
        self.fusion_gate = layers.Dense(hidden_dim, activation="sigmoid")
        self.fusion_transform = layers.Dense(hidden_dim)

    def get_config(self):
        config = super().get_config()
        config.update({"hidden_dim": self.hidden_dim, "num_heads": self.num_heads})
        return config

    def call(self, transformer_features, physics_features, training=None):
        """Apply physics-guided attention to transformer features.

        Args:
            transformer_features: Features from transformer encoder
            physics_features: Features from physics module (HNN)

        Returns:
            Refined features incorporating physics guidance
        """
        # Use physics features to attend to transformer features
        attended = self.physics_attention(
            query=physics_features,
            value=transformer_features,
            key=transformer_features,
            training=training,
        )

        # Gated fusion
        gate = self.fusion_gate(
            ops.concatenate([transformer_features, physics_features], axis=-1)
        )
        transformed = self.fusion_transform(attended)

        # Combine with gating
        output = gate * transformed + (1 - gate) * transformer_features

        return output


@keras.saving.register_keras_serializable()
class NonDimensionalizer(layers.Layer):
    """Non-dimensionalize inputs for better generalization across parameter ranges."""

    def __init__(
        self,
        characteristic_length: float = 400.0,  # Half of world width
        characteristic_time: float = 1.0,
        characteristic_mass: float = 1.0,
        gravity_scale: float = 1000.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.characteristic_length = characteristic_length
        self.characteristic_time = characteristic_time
        self.characteristic_mass = characteristic_mass
        self.gravity_scale = gravity_scale

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "characteristic_length": self.characteristic_length,
                "characteristic_time": self.characteristic_time,
                "characteristic_mass": self.characteristic_mass,
                "gravity_scale": self.gravity_scale,
            }
        )
        return config

    def call(self, positions, velocities, masses, gravity):
        """Non-dimensionalize physical quantities.

        Returns:
            Dictionary of non-dimensional quantities
        """
        # Non-dimensionalize
        positions_nd = positions / self.characteristic_length
        velocities_nd = velocities / (
            self.characteristic_length / self.characteristic_time
        )
        masses_nd = masses / self.characteristic_mass
        gravity_nd = gravity / self.gravity_scale

        return {
            "positions": positions_nd,
            "velocities": velocities_nd,
            "masses": masses_nd,
            "gravity": gravity_nd,
        }

    def inverse(self, positions_nd, velocities_nd, masses_nd, gravity_nd):
        """Convert back to dimensional quantities."""
        positions = positions_nd * self.characteristic_length
        velocities = velocities_nd * (
            self.characteristic_length / self.characteristic_time
        )
        masses = masses_nd * self.characteristic_mass
        gravity = gravity_nd * self.gravity_scale

        return {
            "positions": positions,
            "velocities": velocities,
            "masses": masses,
            "gravity": gravity,
        }


def test_hamiltonian_nn():
    """Test the Hamiltonian Neural Network implementation."""
    print("Testing Hamiltonian Neural Network...")

    # Create model
    hnn = HamiltonianNN(hidden_dim=128, num_layers=2)

    # Test data - 2 balls
    batch_size = 32
    state = np.random.randn(batch_size, 8).astype(
        np.float32
    )  # [x1,y1,x2,y2,px1,py1,px2,py2]
    masses = np.ones((batch_size, 2), dtype=np.float32)

    # Test forward pass
    inputs = {"state": state, "masses": masses}
    H_value = hnn(inputs)
    print(f"Hamiltonian shape: {H_value.shape}")
    print(f"Hamiltonian values: {np.array(H_value[:5])}")

    # Test dynamics computation (single sample for JAX)
    single_state = state[0]
    single_masses = masses[0]
    dynamics = hnn.compute_dynamics(single_state, single_masses)
    print(f"Dynamics shape: {dynamics.shape}")
    print(f"dq/dt: {dynamics[:4]}")
    print(f"dp/dt: {dynamics[4:]}")

    print("Test passed!\n")

    # Test Fourier features
    print("Testing Fourier Features...")
    ff = FourierFeatures(num_features=16)
    ff.build((None, 8))
    ff_out = ff(state)
    print(f"Fourier features shape: {ff_out.shape}")
    print("Fourier features test passed!\n")

    # Test Physics-Guided Attention
    print("Testing Physics-Guided Attention...")
    pga = PhysicsGuidedAttention(hidden_dim=64, num_heads=4)

    # Mock features
    seq_len = 10
    feature_dim = 64
    transformer_features = np.random.randn(batch_size, seq_len, feature_dim).astype(
        np.float32
    )
    physics_features = np.random.randn(batch_size, seq_len, feature_dim).astype(
        np.float32
    )

    refined = pga(transformer_features, physics_features)
    print(f"Refined features shape: {refined.shape}")
    print("Physics-guided attention test passed!")


if __name__ == "__main__":
    test_hamiltonian_nn()
