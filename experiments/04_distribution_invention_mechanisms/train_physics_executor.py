#!/usr/bin/env python3
"""Train the Neural Physics Executor component of Two-Stage Physics Compiler.

This trains Stage 2 to execute physics simulations given explicit parameter context.
Stage 1 (rule extraction) is already perfect, so we focus on the neural component.
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


class PhysicsDataset:
    """Dataset for physics trajectory training."""

    def __init__(self, data_path: str):
        """Load physics training data."""
        with open(data_path, "r") as f:
            self.data = json.load(f)

        print(f"Loaded {len(self.data)} trajectories from {data_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Get a single trajectory with physics context."""
        item = self.data[idx]

        # Convert to MLX arrays
        trajectory = mx.array(item["trajectory"])
        initial_state = mx.array(item["initial_state"])
        physics_params = item["physics_params"]

        return {
            "trajectory": trajectory,
            "initial_state": initial_state,
            "physics_params": physics_params,
            "command": item["command"],
        }

    def batch_generator(self, batch_size: int = 32, shuffle: bool = True):
        """Generate batches of data."""
        indices = np.arange(len(self.data))
        if shuffle:
            np.random.shuffle(indices)

        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i : i + batch_size]
            batch = [self[idx] for idx in batch_indices]
            yield batch


def compute_physics_loss(
    predicted_trajectory: mx.array,
    target_trajectory: mx.array,
    physics_params: Dict[str, float],
) -> Tuple[mx.array, Dict[str, float]]:
    """Compute physics-informed loss with multiple components."""
    # Trajectory prediction loss (MSE)
    traj_loss = mx.mean((predicted_trajectory - target_trajectory) ** 2)

    # Position and velocity losses separately
    pos_loss = mx.mean((predicted_trajectory[:, :2] - target_trajectory[:, :2]) ** 2)
    vel_loss = mx.mean((predicted_trajectory[:, 2:] - target_trajectory[:, 2:]) ** 2)

    # Physics consistency losses
    dt = 1.0 / 60.0
    gravity = physics_params.get("gravity", 9.8)

    # Check if acceleration matches gravity when in free fall
    # (simplified - only when y > 0 and not bouncing)
    pred_acc_y = (predicted_trajectory[1:, 3] - predicted_trajectory[:-1, 3]) / dt
    expected_acc_y = -gravity

    # Only apply gravity consistency when ball is in air (y > 0.5)
    in_air = predicted_trajectory[:-1, 1] > 0.5
    gravity_loss = mx.mean(in_air * (pred_acc_y - expected_acc_y) ** 2)

    # Energy conservation (approximate)
    # KE + PE should be roughly constant (with some loss due to friction/damping)
    pred_ke = 0.5 * mx.sum(predicted_trajectory[:, 2:] ** 2, axis=1)
    pred_pe = gravity * predicted_trajectory[:, 1]
    pred_energy = pred_ke + pred_pe

    # Energy should decrease slightly over time (damping)
    energy_diff = pred_energy[1:] - pred_energy[:-1]
    energy_loss = mx.mean(mx.maximum(energy_diff, 0) ** 2)  # Penalize energy increases

    # Total loss
    total_loss = traj_loss + 0.1 * gravity_loss + 0.05 * energy_loss

    # Store components for logging
    loss_components = {
        "total": float(total_loss),
        "trajectory": float(traj_loss),
        "position": float(pos_loss),
        "velocity": float(vel_loss),
        "gravity": float(gravity_loss),
        "energy": float(energy_loss),
    }

    return total_loss, loss_components


def train_step(
    model: nn.Module,
    param_encoder: nn.Module,
    batch: List[Dict],
    optimizer: optim.Optimizer,
) -> Dict[str, float]:
    """Single training step."""

    # Process batch to extract arrays
    target_trajectories = []
    initial_states = []
    physics_params_list = []

    for item in batch:
        target_trajectories.append(item["trajectory"])
        initial_states.append(item["initial_state"])
        physics_params_list.append(item["physics_params"])

    # Stack into arrays
    target_trajectories = mx.stack(target_trajectories)
    initial_states = mx.stack(initial_states)

    def loss_fn(model, param_encoder):
        total_loss = mx.array(0.0)
        batch_size = len(initial_states)

        for i in range(batch_size):
            # Get data for this sample
            target_traj = target_trajectories[i]
            initial_state = initial_states[i]
            physics_params = physics_params_list[i]

            # Encode physics parameters
            param_encoding = param_encoder(physics_params)

            # Generate trajectory step by step
            predicted_traj = []
            state = initial_state

            for t in range(len(target_traj) - 1):
                # Predict next state
                next_state = model(state, param_encoding, timestep=t / 60.0)
                predicted_traj.append(next_state)
                state = next_state

            # Stack predictions
            predicted_traj = mx.stack(predicted_traj)

            # Simple MSE loss for now
            loss = mx.mean((predicted_traj - target_traj[1:]) ** 2)
            total_loss = total_loss + loss

        # Average over batch
        return total_loss / batch_size

    # Compute loss and gradients
    loss, grads = mx.value_and_grad(loss_fn, argnums=[0, 1])(model, param_encoder)

    # Update parameters
    optimizer.update(model, grads[0])
    optimizer.update(param_encoder, grads[1])

    mx.eval(model.parameters(), param_encoder.parameters())

    # Return simple loss dict
    return {"trajectory": float(loss), "position": float(loss), "velocity": float(loss)}


def evaluate(
    model: nn.Module,
    param_encoder: nn.Module,
    dataset: PhysicsDataset,
    num_samples: int = 100,
) -> Dict[str, float]:
    """Evaluate model on validation set."""
    model.eval()
    param_encoder.eval()

    total_loss = 0
    loss_components = {
        "trajectory": 0,
        "position": 0,
        "velocity": 0,
        "gravity": 0,
        "energy": 0,
    }

    # Evaluate on subset
    num_samples = min(num_samples, len(dataset))

    for i in range(num_samples):
        item = dataset[i]

        # Generate trajectory
        target_traj = item["trajectory"]
        initial_state = item["initial_state"]
        physics_params = item["physics_params"]

        # Encode parameters
        param_encoding = param_encoder(physics_params)

        # Generate trajectory
        predicted_traj = []
        state = initial_state

        for t in range(len(target_traj) - 1):
            next_state = model(state, param_encoding, timestep=t / 60.0)
            predicted_traj.append(next_state)
            state = next_state

        predicted_traj = mx.stack(predicted_traj)

        # Compute loss
        loss, components = compute_physics_loss(
            predicted_traj, target_traj[1:], physics_params
        )

        total_loss += float(loss)
        for key, value in components.items():
            if key != "total":
                loss_components[key] += value

    # Average
    total_loss /= num_samples
    for key in loss_components:
        loss_components[key] /= num_samples

    loss_components["total"] = total_loss

    model.train()
    param_encoder.train()

    return loss_components


def main():
    """Train the neural physics executor."""
    # Configuration
    batch_size = 16
    learning_rate = 1e-3
    num_epochs = 5  # Reduced for quick demo
    eval_every = 2
    save_every = 5

    # Load datasets
    print("Loading datasets...")
    train_dataset = PhysicsDataset("data/physics_training/train_physics.json")
    val_dataset = PhysicsDataset("data/physics_training/val_physics.json")

    # Create models
    print("\nInitializing models...")
    param_encoder = PhysicsEncoder(num_params=4, hidden_dim=64)
    physics_executor = NeuralPhysicsExecutor(state_dim=4, param_dim=64, hidden_dim=128)

    # Optimizer
    optimizer = optim.Adam(learning_rate=learning_rate)

    # Training loop
    print("\nStarting training...")
    print(f"Batch size: {batch_size}, Learning rate: {learning_rate}")
    print(f"Training for {num_epochs} epochs")

    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        # Training
        train_losses = []

        with tqdm(
            train_dataset.batch_generator(batch_size),
            total=len(train_dataset) // batch_size,
            desc=f"Epoch {epoch+1}/{num_epochs}",
        ) as pbar:
            for batch in pbar:
                loss_components = train_step(
                    physics_executor, param_encoder, batch, optimizer
                )
                train_losses.append(loss_components["trajectory"])

                # Update progress bar
                pbar.set_postfix(
                    {
                        "loss": f"{loss_components['trajectory']:.4f}",
                        "pos": f"{loss_components['position']:.4f}",
                        "vel": f"{loss_components['velocity']:.4f}",
                    }
                )

        # Average training loss
        avg_train_loss = np.mean(train_losses)

        # Validation
        if (epoch + 1) % eval_every == 0:
            print(f"\nEvaluating on validation set...")
            val_components = evaluate(physics_executor, param_encoder, val_dataset)

            print(f"Validation losses:")
            for key, value in val_components.items():
                print(f"  {key}: {value:.4f}")

            # Save best model
            if val_components["total"] < best_val_loss:
                best_val_loss = val_components["total"]
                print(f"New best validation loss: {best_val_loss:.4f}")

                # Save models
                mx.save_weights(
                    "outputs/physics_executor_best.npz",
                    [physics_executor, param_encoder],
                )

        # Regular checkpoint
        if (epoch + 1) % save_every == 0:
            mx.save_weights(
                f"outputs/physics_executor_epoch_{epoch+1}.npz",
                [physics_executor, param_encoder],
            )
            print(f"Saved checkpoint at epoch {epoch+1}")

        print(f"Epoch {epoch+1} - Train loss: {avg_train_loss:.4f}\n")

    # Final save
    mx.save_weights(
        "outputs/physics_executor_final.npz",
        [physics_executor, param_encoder],
    )
    print("\nTraining complete! Models saved to outputs/")

    # Test on specific scenarios
    print("\nTesting on specific scenarios...")
    test_physics_executor(physics_executor, param_encoder)


def test_physics_executor(executor: nn.Module, encoder: nn.Module):
    """Test the trained executor on specific scenarios."""
    # Load test scenarios
    with open("data/physics_training/test_scenarios.json", "r") as f:
        test_scenarios = json.load(f)

    print("\nTesting trained physics executor:")
    print("=" * 50)

    for i, scenario in enumerate(test_scenarios):
        print(f"\nTest {i+1}: {scenario['command']}")

        # Setup
        initial_state = mx.array(scenario["initial_state"])
        physics_params = scenario["physics_params"]

        # Encode parameters
        param_encoding = encoder(physics_params)

        # Generate short trajectory
        trajectory = [initial_state]
        state = initial_state

        for t in range(60):  # 1 second
            state = executor(state, param_encoding, timestep=t / 60.0)
            trajectory.append(state)

        trajectory = mx.stack(trajectory)

        # Analyze
        final_pos = trajectory[-1, :2]
        final_vel = trajectory[-1, 2:]
        y_change = float(trajectory[-1, 1] - trajectory[0, 1])

        print(f"  Initial: pos=({initial_state[0]:.1f}, {initial_state[1]:.1f})")
        print(f"  Final: pos=({final_pos[0]:.1f}, {final_pos[1]:.1f})")
        print(f"  Y change: {y_change:.2f}m")

        # Check physics behavior
        if physics_params["gravity"] > 8:
            expected = "fall quickly"
        elif physics_params["gravity"] < 3:
            expected = "fall slowly"
        else:
            expected = "moderate fall"

        actual = (
            "fall quickly"
            if y_change < -8
            else "fall slowly"
            if y_change > -3
            else "moderate fall"
        )
        print(f"  Expected: {expected}, Actual: {actual}")


if __name__ == "__main__":
    # Create output directory
    os.makedirs("outputs", exist_ok=True)

    # Train the model
    main()
