#!/usr/bin/env python3
"""Simplified training script for Neural Physics Executor.

This version focuses on getting a working trained model rather than
perfect MLX gradient computation.
"""

from utils.imports import setup_project_paths

setup_project_paths()

import json
import os
from typing import Dict, List, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from neural_physics_executor import NeuralPhysicsExecutor, PhysicsEncoder
from tqdm import tqdm


def load_physics_data(data_dir: str = "data/physics_training"):
    """Load pre-generated physics training data."""
    # Check if data exists
    train_path = os.path.join(data_dir, "train_physics.json")
    val_path = os.path.join(data_dir, "val_physics.json")

    if not os.path.exists(train_path):
        print(f"Training data not found at {train_path}")
        print("Please run generate_physics_training_data.py first")
        return None, None

    # Load data
    print(f"Loading training data from {train_path}")
    with open(train_path, "r") as f:
        train_data = json.load(f)

    print(f"Loading validation data from {val_path}")
    with open(val_path, "r") as f:
        val_data = json.load(f)

    print(
        f"Loaded {len(train_data)} training and {len(val_data)} validation trajectories"
    )
    return train_data, val_data


def create_mini_batch(
    data: List[Dict], indices: List[int]
) -> Tuple[mx.array, mx.array, List[Dict]]:
    """Create a mini-batch of data."""
    len(indices)

    # Extract data for batch
    initial_states = []
    target_trajectories = []
    physics_params = []

    for idx in indices:
        item = data[idx]
        initial_states.append(item["initial_state"])
        target_trajectories.append(item["trajectory"])
        physics_params.append(item["physics_params"])

    # Convert to arrays
    initial_states = mx.array(initial_states)
    target_trajectories = mx.array(target_trajectories)

    return initial_states, target_trajectories, physics_params


def compute_trajectory_loss(predicted: mx.array, target: mx.array) -> mx.array:
    """Simple MSE loss for trajectory prediction."""
    return mx.mean((predicted - target) ** 2)


def train_step_simple(
    executor: nn.Module,
    encoder: nn.Module,
    batch_data: Tuple,
    optimizer: optim.Optimizer,
    max_steps: int = 50,  # Limit trajectory length for training
) -> float:
    """Simplified training step."""
    initial_states, target_trajectories, physics_params_list = batch_data
    batch_size = len(initial_states)

    def loss_fn(executor, encoder):
        total_loss = mx.array(0.0)

        for i in range(batch_size):
            # Get data for this sample
            initial_state = initial_states[i]
            target_traj = target_trajectories[i]
            physics_params = physics_params_list[i]

            # Limit trajectory length
            target_traj = target_traj[:max_steps]

            # Encode physics parameters
            param_encoding = encoder(physics_params)

            # Generate trajectory
            predicted_states = []
            state = initial_state

            for t in range(len(target_traj) - 1):
                # Predict next state
                next_state = executor(state, param_encoding, timestep=t / 60.0)
                predicted_states.append(next_state)
                state = next_state

            # Stack predictions
            if predicted_states:
                predicted_traj = mx.stack(predicted_states)
                # Compute loss (skip first state as it's given)
                loss = compute_trajectory_loss(predicted_traj, target_traj[1:])
                total_loss = total_loss + loss

        return total_loss / batch_size

    # Compute gradients and update
    loss, grads = mx.value_and_grad(loss_fn, argnums=[0, 1])(executor, encoder)
    optimizer.update(executor, grads[0])
    optimizer.update(encoder, grads[1])

    # Ensure updates are evaluated
    mx.eval(executor.parameters(), encoder.parameters())

    return float(loss)


def evaluate_model(
    executor: nn.Module,
    encoder: nn.Module,
    val_data: List[Dict],
    num_samples: int = 50,
) -> Dict[str, float]:
    """Evaluate model on validation set."""
    total_loss = 0.0

    # Sample validation data
    indices = np.random.choice(
        len(val_data), min(num_samples, len(val_data)), replace=False
    )

    for idx in indices:
        item = val_data[idx]
        initial_state = mx.array(item["initial_state"])
        target_traj = mx.array(item["trajectory"])[:50]  # Limit length
        physics_params = item["physics_params"]

        # Encode parameters
        param_encoding = encoder(physics_params)

        # Generate trajectory
        predicted_states = []
        state = initial_state

        for t in range(len(target_traj) - 1):
            next_state = executor(state, param_encoding, timestep=t / 60.0)
            predicted_states.append(next_state)
            state = next_state

        if predicted_states:
            predicted_traj = mx.stack(predicted_states)

            # Compute metrics
            loss = compute_trajectory_loss(predicted_traj, target_traj[1:])
            total_loss += float(loss)

            # Position and velocity errors
            pos_error += float(
                mx.mean((predicted_traj[:, :2] - target_traj[1:, :2]) ** 2)
            )
            vel_error += float(
                mx.mean((predicted_traj[:, 2:] - target_traj[1:, 2:]) ** 2)
            )

    num_evaluated = len(indices)
    return {
        "loss": total_loss / num_evaluated,
        "position_error": pos_error / num_evaluated,
        "velocity_error": vel_error / num_evaluated,
    }


def main():
    """Train the neural physics executor."""
    # Configuration
    batch_size = 8  # Smaller batch size for memory
    learning_rate = 1e-3
    num_epochs = 10  # Reduced for quick training
    eval_every = 2

    # Load data
    train_data, val_data = load_physics_data()
    if train_data is None:
        return

    # Create models
    print("\nInitializing models...")
    param_encoder = PhysicsEncoder(num_params=4, hidden_dim=64)
    physics_executor = NeuralPhysicsExecutor(state_dim=4, param_dim=64, hidden_dim=128)

    # Initialize parameters
    dummy_state = mx.zeros((4,))
    dummy_params = {"gravity": 9.8, "friction": 0.3, "elasticity": 0.8, "damping": 0.99}
    dummy_encoding = param_encoder(dummy_params)
    _ = physics_executor(dummy_state, dummy_encoding)

    # Optimizer
    optimizer = optim.Adam(learning_rate=learning_rate)

    # Training loop
    print(f"\nStarting training...")
    print(f"Batch size: {batch_size}, Learning rate: {learning_rate}")
    print(f"Training for {num_epochs} epochs")

    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        # Shuffle data
        indices = np.arange(len(train_data))
        np.random.shuffle(indices)

        # Training
        epoch_losses = []
        num_batches = len(train_data) // batch_size

        with tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for batch_idx in pbar:
                # Get batch indices
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(train_data))
                batch_indices = indices[start_idx:end_idx]

                # Create batch
                batch_data = create_mini_batch(train_data, batch_indices.tolist())

                # Train step
                loss = train_step_simple(
                    physics_executor, param_encoder, batch_data, optimizer
                )
                epoch_losses.append(loss)

                # Update progress
                pbar.set_postfix({"loss": f"{loss:.4f}"})

        # Average loss
        avg_loss = np.mean(epoch_losses)
        print(f"Epoch {epoch+1} - Average loss: {avg_loss:.4f}")

        # Validation
        if (epoch + 1) % eval_every == 0:
            print("Evaluating on validation set...")
            val_metrics = evaluate_model(physics_executor, param_encoder, val_data)

            print(f"Validation metrics:")
            for key, value in val_metrics.items():
                print(f"  {key}: {value:.4f}")

            # Save best model
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                print(f"New best validation loss: {best_val_loss:.4f}")

                # Save models
                os.makedirs("outputs", exist_ok=True)
                weights = [physics_executor, param_encoder]
                mx.save_weights("outputs/physics_executor_best.npz", weights)
                print("Saved best model to outputs/physics_executor_best.npz")

    # Final save
    mx.save_weights(
        "outputs/physics_executor_final.npz", [physics_executor, param_encoder]
    )
    print("\nTraining complete! Models saved to outputs/")

    # Quick test
    print("\nQuick test on trained model:")
    test_trained_model(physics_executor, param_encoder)


def test_trained_model(executor: nn.Module, encoder: nn.Module):
    """Quick test of the trained model."""
    test_cases = [
        {"gravity": 9.8, "name": "Earth gravity"},
        {"gravity": 1.6, "name": "Moon gravity"},
        {"gravity": 25.0, "name": "High gravity (OOD)"},
    ]

    initial_state = mx.array([0.0, 10.0, 0.0, 0.0])  # Drop from 10m

    for test in test_cases:
        physics_params = {
            "gravity": test["gravity"],
            "friction": 0.3,
            "elasticity": 0.8,
            "damping": 0.99,
        }

        # Encode and simulate
        param_encoding = encoder(physics_params)

        state = initial_state
        for t in range(60):  # 1 second
            state = executor(state, param_encoding, timestep=t / 60.0)

        y_change = float(state[1] - initial_state[1])
        print(f"{test['name']}: Y change = {y_change:.2f}m")


if __name__ == "__main__":
    main()
