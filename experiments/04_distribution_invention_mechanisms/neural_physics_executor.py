#!/usr/bin/env python3
"""Neural Physics Executor - Stage 2 of Two-Stage Physics Compiler.

Executes physics simulations with explicit parameter context, similar to how
we execute commands with variable bindings. Uses cross-attention to learn
how physics parameters affect state evolution.
"""

from utils.imports import setup_project_paths

setup_project_paths()

from typing import Dict

import mlx.core as mx
import mlx.nn as nn


class PhysicsEncoder(nn.Module):
    """Encodes physics parameters into a neural representation."""

    def __init__(self, num_params: int = 4, hidden_dim: int = 64):
        super().__init__()
        self.num_params = num_params
        self.hidden_dim = hidden_dim

        # Parameter names to indices
        self.param_indices = {
            "gravity": 0,
            "friction": 1,
            "elasticity": 2,
            "damping": 3,
        }

        # Encode each parameter with its metadata
        self.param_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),  # value, normalized_value, is_active
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Combine all parameters
        self.combiner = nn.Sequential(
            nn.Linear(num_params * hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def __call__(self, parameters: Dict[str, float], timestep: int = 0) -> mx.array:
        """Encode physics parameters for neural processing."""
        # Default values
        defaults = {
            "gravity": 9.8,
            "friction": 0.3,
            "elasticity": 0.8,
            "damping": 0.99,
        }

        # Encode each parameter
        encoded_params = []
        for param_name, idx in self.param_indices.items():
            if param_name in parameters:
                value = parameters[param_name]
                normalized = value / defaults[param_name]  # Normalize by default
                is_active = 1.0
            else:
                value = defaults[param_name]
                normalized = 1.0
                is_active = 0.0

            # Create parameter vector
            param_vec = mx.array([value, normalized, is_active])
            encoded = self.param_encoder(param_vec)
            encoded_params.append(encoded)

        # Combine all parameters
        combined = mx.concatenate(encoded_params, axis=-1)
        return self.combiner(combined)


class NeuralPhysicsExecutor(nn.Module):
    """Executes physics simulation with explicit parameter context.

    Key insight: Like executing "do X" with binding context, we execute
    physics steps with parameter context.
    """

    def __init__(self, state_dim: int = 4, param_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.state_dim = state_dim
        self.param_dim = param_dim
        self.hidden_dim = hidden_dim

        # State encoder (position and velocity)
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Cross-attention: state attends to parameters
        self.cross_attention = nn.MultiHeadAttention(
            dims=hidden_dim,
            num_heads=8,
            query_input_dims=hidden_dim,
            key_input_dims=param_dim,
            value_input_dims=param_dim,
            value_dims=hidden_dim // 8,
            value_output_dims=hidden_dim,
        )

        # Physics predictor with residual connection
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, state_dim),
        )

        # Time embedding for temporal physics
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim),
        )

    def __call__(
        self, state: mx.array, param_encoding: mx.array, timestep: float = 0.0
    ) -> mx.array:
        """Execute one physics step with parameter context.

        Args:
            state: Current state [x, y, vx, vy]
            param_encoding: Encoded physics parameters
            timestep: Current time (for time-varying physics)

        Returns:
            Next state [x, y, vx, vy]
        """
        # Encode current state
        state_encoded = self.state_encoder(state)

        # Add time information
        time_vec = mx.array([timestep])
        time_encoded = self.time_embed(time_vec)
        state_encoded = state_encoded + time_encoded

        # Cross-attention: how do parameters affect this state?
        # Add batch dimension for attention
        state_batch = state_encoded[None, None, :]  # [1, 1, hidden]
        param_batch = param_encoding[None, None, :]  # [1, 1, param_dim]

        attended = self.cross_attention(
            state_batch,  # queries
            param_batch,  # keys
            param_batch,  # values
        )
        attended = attended[0, 0, :]  # Remove batch dimensions

        # Predict state change (residual)
        delta_state = self.predictor(attended)

        # Physics-informed integration (semi-implicit Euler)
        dt = 1.0 / 60.0  # 60 FPS

        # Extract position and velocity
        pos = state[:2]
        vel = state[2:]

        # Update velocity first (forces affect velocity)
        new_vel = vel + delta_state[2:] * dt

        # Then update position with new velocity
        new_pos = pos + new_vel * dt

        # Combine into new state
        new_state = mx.concatenate([new_pos, new_vel])

        return new_state


class PhysicsAwareTransformer(nn.Module):
    """Transformer that processes trajectories with physics parameter awareness."""

    def __init__(self, state_dim: int = 4, hidden_dim: int = 128, num_heads: int = 8):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        # Trajectory encoder
        self.input_proj = nn.Linear(state_dim, hidden_dim)

        # Self-attention layers
        self.attention = nn.MultiHeadAttention(
            dims=hidden_dim,
            num_heads=num_heads,
            query_input_dims=hidden_dim,
            key_input_dims=hidden_dim,
            value_input_dims=hidden_dim,
            value_dims=hidden_dim // num_heads,
            value_output_dims=hidden_dim,
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, state_dim)

    def __call__(self, trajectory: mx.array, param_encoding: mx.array) -> mx.array:
        """Process trajectory with parameter awareness."""
        # Project states
        x = self.input_proj(trajectory)

        # Add parameter information to each timestep
        param_broadcast = param_encoding[None, :].repeat(x.shape[0], axis=0)
        x = x + param_broadcast[:, : self.hidden_dim]

        # Self-attention
        x_batch = x[None, :, :]  # Add batch dimension
        attn_out = self.attention(x_batch, x_batch, x_batch)
        attn_out = attn_out[0]  # Remove batch dimension

        x = self.norm1(x + attn_out)

        # FFN
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        # Project back to state space
        return self.output_proj(x)


def test_neural_executor():
    """Test the neural physics executor."""
    print("Testing Neural Physics Executor\n" + "=" * 50)

    # Create components
    param_encoder = PhysicsEncoder()
    executor = NeuralPhysicsExecutor()

    # Test scenarios
    test_scenarios = [
        {
            "name": "Standard gravity",
            "params": {"gravity": 9.8, "friction": 0.3},
            "state": mx.array([0.0, 10.0, 5.0, 0.0]),  # x=0, y=10m, vx=5m/s, vy=0
        },
        {
            "name": "Low gravity",
            "params": {"gravity": 1.6, "friction": 0.3},
            "state": mx.array([0.0, 10.0, 5.0, 0.0]),
        },
        {
            "name": "High friction",
            "params": {"gravity": 9.8, "friction": 0.9},
            "state": mx.array([0.0, 10.0, 5.0, 0.0]),
        },
        {
            "name": "Space physics",
            "params": {"gravity": 0.1, "friction": 0.0, "damping": 1.0},
            "state": mx.array([0.0, 10.0, 5.0, 0.0]),
        },
    ]

    for scenario in test_scenarios:
        print(f"\n{scenario['name']}:")
        print(f"Parameters: {scenario['params']}")
        print(f"Initial state: {scenario['state']}")

        # Encode parameters
        param_enc = param_encoder(scenario["params"])
        print(f"Parameter encoding shape: {param_enc.shape}")

        # Run one physics step
        next_state = executor(scenario["state"], param_enc, timestep=0.0)
        print(f"Next state: {next_state}")

        # Compute change
        delta = next_state - scenario["state"]
        print(f"State change: {delta}")

    # Test trajectory processing
    print("\n" + "=" * 50)
    print("Testing trajectory generation:")

    # Generate a simple trajectory
    state = mx.array([0.0, 10.0, 5.0, 0.0])
    params = {"gravity": 9.8, "friction": 0.3}
    param_enc = param_encoder(params)

    trajectory = [state]
    for t in range(60):  # 1 second at 60 FPS
        state = executor(state, param_enc, timestep=t / 60.0)
        trajectory.append(state)

    trajectory = mx.stack(trajectory)
    print(f"Generated trajectory shape: {trajectory.shape}")
    print(f"Final position: {trajectory[-1, :2]}")
    print(f"Final velocity: {trajectory[-1, 2:]}")

    # Test if ball fell (y should decrease)
    y_positions = trajectory[:, 1]
    print(f"Y position changed from {y_positions[0]:.2f} to {y_positions[-1]:.2f}")
    print(
        f"Physics behavior: {'✓ Correct' if y_positions[-1] < y_positions[0] else '✗ Wrong'}"
    )


if __name__ == "__main__":
    test_neural_executor()
