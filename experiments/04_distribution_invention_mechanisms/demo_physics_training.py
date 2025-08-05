#!/usr/bin/env python3
"""Demo training of Neural Physics Executor - simplified for proof of concept."""

from utils.imports import setup_project_paths

setup_project_paths()

import os

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
from neural_physics_executor import NeuralPhysicsExecutor, PhysicsEncoder


def demo_training():
    """Demonstrate that the neural physics executor can learn simple physics."""
    print("Demo Training of Neural Physics Executor")
    print("=" * 60)

    # Create models
    param_encoder = PhysicsEncoder(num_params=4, hidden_dim=64)
    physics_executor = NeuralPhysicsExecutor(state_dim=4, param_dim=64, hidden_dim=128)

    # Initialize with dummy forward pass
    dummy_state = mx.zeros((4,))
    dummy_params = {"gravity": 9.8, "friction": 0.3, "elasticity": 0.8, "damping": 0.99}
    dummy_encoding = param_encoder(dummy_params)
    _ = physics_executor(dummy_state, dummy_encoding)

    print("Models initialized successfully!")

    # Create simple training data - just falling balls
    print("\nCreating simple training data...")
    training_examples = []

    for gravity in [5.0, 7.0, 9.8, 12.0]:
        for initial_height in [5.0, 10.0, 15.0]:
            example = {
                "initial_state": [0.0, initial_height, 0.0, 0.0],  # x, y, vx, vy
                "physics_params": {
                    "gravity": gravity,
                    "friction": 0.3,
                    "elasticity": 0.8,
                    "damping": 0.99,
                },
                "expected_y_after_1s": initial_height
                - 0.5 * gravity * 1.0**2,  # Simple physics
            }
            training_examples.append(example)

    print(f"Created {len(training_examples)} training examples")

    # Simple training loop
    optimizer = optim.Adam(learning_rate=1e-3)

    print("\nTraining for 100 iterations...")
    for iteration in range(100):
        total_loss = 0.0

        # Shuffle examples
        np.random.shuffle(training_examples)

        for example in training_examples[:4]:  # Mini-batch of 4
            # Get data
            initial_state = mx.array(example["initial_state"])
            physics_params = example["physics_params"]
            expected_y = example["expected_y_after_1s"]

            def loss_fn(executor, encoder):
                # Encode physics
                param_encoding = encoder(physics_params)

                # Simulate for 60 steps (1 second at 60 FPS)
                state = initial_state
                for t in range(60):
                    state = executor(state, param_encoding, timestep=t / 60.0)

                # Loss: how close is final y position to expected?
                predicted_y = state[1]
                loss = (predicted_y - expected_y) ** 2
                return loss

            # Compute gradients
            loss, grads = mx.value_and_grad(loss_fn, argnums=[0, 1])(
                physics_executor, param_encoder
            )

            # Update
            optimizer.update(physics_executor, grads[0])
            optimizer.update(param_encoder, grads[1])

            total_loss += float(loss)

        # Evaluate parameters
        mx.eval(physics_executor.parameters(), param_encoder.parameters())

        # Print progress
        if iteration % 20 == 0:
            avg_loss = total_loss / 4
            print(f"Iteration {iteration}: Average loss = {avg_loss:.4f}")

    print("\nTraining complete!")

    # Test the trained model
    print("\nTesting trained model:")
    test_cases = [
        {"gravity": 9.8, "height": 10.0, "name": "Training case"},
        {"gravity": 20.0, "height": 10.0, "name": "OOD gravity (20)"},
        {"gravity": 2.0, "height": 10.0, "name": "OOD gravity (2)"},
    ]

    for test in test_cases:
        initial_state = mx.array([0.0, test["height"], 0.0, 0.0])
        physics_params = {
            "gravity": test["gravity"],
            "friction": 0.3,
            "elasticity": 0.8,
            "damping": 0.99,
        }

        # Encode and simulate
        param_encoding = param_encoder(physics_params)
        state = initial_state

        for t in range(60):  # 1 second
            state = physics_executor(state, param_encoding, timestep=t / 60.0)

        final_y = float(state[1])
        expected_y = test["height"] - 0.5 * test["gravity"] * 1.0**2
        error = abs(final_y - expected_y)

        print(f"\n{test['name']}:")
        print(f"  Initial height: {test['height']:.1f}m")
        print(f"  Gravity: {test['gravity']:.1f} m/s²")
        print(f"  Final Y (predicted): {final_y:.2f}m")
        print(f"  Final Y (expected): {expected_y:.2f}m")
        print(f"  Error: {error:.2f}m")

    # Save the model
    os.makedirs("outputs", exist_ok=True)
    mx.save_weights(
        "outputs/physics_executor_demo.npz", [physics_executor, param_encoder]
    )
    print("\nModel saved to outputs/physics_executor_demo.npz")

    return physics_executor, param_encoder


if __name__ == "__main__":
    demo_training()
