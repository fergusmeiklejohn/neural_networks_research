"""
Debug minimal PINN to understand why it's failing.
"""

import os

os.environ["KERAS_BACKEND"] = "jax"

import sys

sys.path.append("../..")

import numpy as np
from keras import ops

from models.minimal_physics_model import MinimalPhysicsModel


# Create a simple test case
def create_simple_test():
    """Create a simple 2-ball falling scenario."""
    # Initial positions (pixels)
    pos1 = np.array([100.0, 100.0])
    pos2 = np.array([200.0, 100.0])

    # Initial velocities (pixels/s)
    vel1 = np.array([10.0, 0.0])
    vel2 = np.array([-10.0, 0.0])

    # Create trajectory with simple physics
    dt = 1 / 30.0
    gravity = -9.8 * 40  # m/s² to pixels/s²

    trajectory = []
    for t in range(50):
        # Update velocities (gravity only)
        vel1[1] += gravity * dt
        vel2[1] += gravity * dt

        # Update positions
        pos1 += vel1 * dt
        pos2 += vel2 * dt

        # Bounce off ground
        if pos1[1] > 580:
            pos1[1] = 580
            vel1[1] = -0.8 * vel1[1]
        if pos2[1] > 580:
            pos2[1] = 580
            vel2[1] = -0.8 * vel2[1]

        # Store state
        state = np.concatenate([pos1, pos2, vel1, vel2])
        trajectory.append(state)

    return np.array(trajectory)


# Test the model
print("Creating simple test trajectory...")
true_trajectory = create_simple_test()
print(f"Trajectory shape: {true_trajectory.shape}")
print(f"First state: {true_trajectory[0]}")
print(f"Last state: {true_trajectory[-1]}")

# Create model
print("\nCreating minimal PINN...")
model = MinimalPhysicsModel(hidden_dim=32)

# Get initial state
initial_state = true_trajectory[0:1]  # Shape (1, 8)

# Test forward pass
print("\nTesting forward pass...")
dummy_input = true_trajectory[np.newaxis, :, :]  # Add batch dimension
predicted = model(dummy_input)
print(f"Predicted shape: {predicted.shape}")

# Check if predictions are reasonable
print(f"\nPrediction statistics:")
print(f"  Min: {ops.min(predicted)}")
print(f"  Max: {ops.max(predicted)}")
print(f"  Mean: {ops.mean(predicted)}")
print(f"  Contains NaN: {ops.any(ops.isnan(predicted))}")

# Test physics losses
print("\nTesting physics losses...")
losses = model.compute_physics_losses(dummy_input, predicted)
for name, loss in losses.items():
    print(f"  {name}: {float(loss):.4f}")

# Check learned parameters
print(f"\nInitial physics parameters:")
print(f"  Gravity: {float(model.gravity[0]):.2f} m/s²")
print(f"  Friction: {float(model.friction[0]):.4f}")

# Test with extreme gravity
print("\n\nTesting with Jupiter gravity...")
model.gravity.assign([-24.8])
predicted_jupiter = model(dummy_input)
mse_jupiter = float(ops.mean(ops.square(predicted_jupiter - dummy_input)))
print(f"MSE with Jupiter gravity: {mse_jupiter:.4f}")

# Test trajectory integration directly
print("\nTesting trajectory integration...")
integrated = model.integrate_trajectory(initial_state[0], steps=50)
print(f"Integrated shape: {integrated.shape}")
print(f"First integrated state: {integrated[0]}")
print(f"Last integrated state: {integrated[-1]}")

# Compare with true trajectory
mse_integrated = float(ops.mean(ops.square(integrated - true_trajectory)))
print(f"\nMSE vs true trajectory: {mse_integrated:.4f}")
