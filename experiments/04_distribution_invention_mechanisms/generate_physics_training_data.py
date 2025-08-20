#!/usr/bin/env python3
"""Generate physics training data for Two-Stage Physics Compiler.

Creates diverse physics trajectories with explicit parameter tracking,
enabling our neural executor to learn physics dynamics.
"""

from utils.imports import setup_project_paths

setup_project_paths()

import json
import os
from typing import Dict, List

import numpy as np
from tqdm import tqdm


class PhysicsSimulator:
    """Simulates 2D physics with explicit parameter control."""

    def __init__(self, dt: float = 1.0 / 60.0):
        self.dt = dt

    def simulate_trajectory(
        self,
        initial_state: np.ndarray,
        physics_params: Dict[str, float],
        timesteps: int = 300,
        ground_y: float = 0.0,
    ) -> np.ndarray:
        """Simulate a physics trajectory with given parameters.

        Args:
            initial_state: [x, y, vx, vy]
            physics_params: Dict with gravity, friction, elasticity, damping
            timesteps: Number of simulation steps
            ground_y: Ground level for collision detection

        Returns:
            Trajectory array of shape (timesteps, 4)
        """
        # Extract parameters
        gravity = physics_params.get("gravity", 9.8)
        friction = physics_params.get("friction", 0.3)
        elasticity = physics_params.get("elasticity", 0.8)
        damping = physics_params.get("damping", 0.99)

        # Initialize trajectory
        trajectory = np.zeros((timesteps, 4))
        state = initial_state.copy()

        for t in range(timesteps):
            # Store current state
            trajectory[t] = state

            # Extract position and velocity
            x, y, vx, vy = state

            # Apply forces
            # Gravity (negative y is down)
            ay = -gravity

            # Air resistance (simplified damping)
            vx *= damping
            vy *= damping

            # Ground collision
            if y <= ground_y and vy < 0:
                # Bounce with elasticity
                vy = -vy * elasticity
                y = ground_y  # Prevent penetration

                # Apply friction to horizontal velocity
                vx *= 1 - friction

            # Update velocity
            vx_new = vx
            vy_new = vy + ay * self.dt

            # Update position
            x_new = x + vx_new * self.dt
            y_new = y + vy_new * self.dt

            # Ensure ball stays above ground
            if y_new < ground_y:
                y_new = ground_y
                vy_new = max(0, vy_new)  # Prevent sinking

            # Update state
            state = np.array([x_new, y_new, vx_new, vy_new])

        return trajectory


def generate_training_scenarios() -> List[Dict]:
    """Generate diverse training scenarios with varied physics."""
    scenarios = []

    # Standard Earth-like physics with variations
    gravity_values = [7.0, 8.0, 9.0, 9.8, 10.0, 11.0, 12.0]
    friction_values = [0.1, 0.2, 0.3, 0.4, 0.5]
    elasticity_values = [0.6, 0.7, 0.8, 0.85, 0.9]

    # Initial conditions
    heights = [5.0, 10.0, 15.0, 20.0]
    velocities = [
        (0.0, 0.0),  # Drop
        (5.0, 0.0),  # Horizontal throw
        (-5.0, 0.0),  # Backward throw
        (3.0, 5.0),  # Diagonal up
        (3.0, -5.0),  # Diagonal down
    ]

    # Generate combinations
    for gravity in gravity_values:
        for friction in friction_values:
            for elasticity in elasticity_values:
                for height in heights:
                    for vx, vy in velocities:
                        scenario = {
                            "physics_params": {
                                "gravity": gravity,
                                "friction": friction,
                                "elasticity": elasticity,
                                "damping": 0.99,  # Keep constant for now
                            },
                            "initial_state": [0.0, height, vx, vy],
                            "command": generate_command(gravity, friction, elasticity),
                        }
                        scenarios.append(scenario)

    return scenarios


def generate_command(gravity: float, friction: float, elasticity: float) -> str:
    """Generate natural language command for physics parameters."""
    commands = []

    # Gravity description
    if abs(gravity - 9.8) < 0.1:
        commands.append("standard Earth gravity")
    else:
        commands.append(f"set gravity to {gravity:.1f} m/s²")

    # Friction description
    if friction < 0.2:
        commands.append("low friction")
    elif friction > 0.4:
        commands.append("high friction")
    else:
        commands.append(f"friction coefficient {friction:.1f}")

    # Elasticity description
    if elasticity > 0.85:
        commands.append("highly elastic")
    elif elasticity < 0.7:
        commands.append("low bounce")

    # Combine with various connectors
    if len(commands) == 1:
        return commands[0]
    elif len(commands) == 2:
        return f"{commands[0]} and {commands[1]}"
    else:
        return f"{commands[0]}, {commands[1]}, and {commands[2]}"


def save_training_data(scenarios: List[Dict], output_dir: str):
    """Save training data in format compatible with Two-Stage Compiler."""
    os.makedirs(output_dir, exist_ok=True)

    # Initialize simulator
    simulator = PhysicsSimulator()

    # Process each scenario
    all_data = []
    for i, scenario in enumerate(tqdm(scenarios, desc="Generating trajectories")):
        # Simulate trajectory
        trajectory = simulator.simulate_trajectory(
            initial_state=np.array(scenario["initial_state"]),
            physics_params=scenario["physics_params"],
            timesteps=300,  # 5 seconds at 60 FPS
        )

        # Create data entry
        data_entry = {
            "id": i,
            "command": scenario["command"],
            "physics_params": scenario["physics_params"],
            "initial_state": scenario["initial_state"],
            "trajectory": trajectory.tolist(),
        }
        all_data.append(data_entry)

    # Split into train/val
    np.random.shuffle(all_data)
    split_idx = int(0.8 * len(all_data))
    train_data = all_data[:split_idx]
    val_data = all_data[split_idx:]

    # Save to files
    with open(os.path.join(output_dir, "train_physics.json"), "w") as f:
        json.dump(train_data, f, indent=2)

    with open(os.path.join(output_dir, "val_physics.json"), "w") as f:
        json.dump(val_data, f, indent=2)

    print(
        f"\nGenerated {len(train_data)} training and {len(val_data)} validation trajectories"
    )
    print(f"Saved to {output_dir}/")

    # Save some statistics
    stats = {
        "num_train": len(train_data),
        "num_val": len(val_data),
        "gravity_range": [7.0, 12.0],
        "friction_range": [0.1, 0.5],
        "elasticity_range": [0.6, 0.9],
        "trajectory_length": 300,
        "dt": 1.0 / 60.0,
    }

    with open(os.path.join(output_dir, "dataset_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)


def generate_test_scenarios() -> List[Dict]:
    """Generate test scenarios for quick validation."""
    return [
        {
            "physics_params": {
                "gravity": 9.8,
                "friction": 0.3,
                "elasticity": 0.8,
                "damping": 0.99,
            },
            "initial_state": [0.0, 10.0, 0.0, 0.0],
            "command": "standard Earth gravity",
        },
        {
            "physics_params": {
                "gravity": 5.0,
                "friction": 0.3,
                "elasticity": 0.8,
                "damping": 0.99,
            },
            "initial_state": [0.0, 10.0, 0.0, 0.0],
            "command": "set gravity to 5.0 m/s²",
        },
        {
            "physics_params": {
                "gravity": 1.6,
                "friction": 0.3,
                "elasticity": 0.8,
                "damping": 0.99,
            },
            "initial_state": [0.0, 10.0, 5.0, 0.0],
            "command": "moon gravity",
        },
    ]


def visualize_sample_trajectories(output_dir: str, num_samples: int = 5):
    """Create simple visualization of sample trajectories."""
    import matplotlib.pyplot as plt

    # Load training data
    with open(os.path.join(output_dir, "train_physics.json"), "r") as f:
        train_data = json.load(f)

    # Plot random samples
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    if num_samples == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        if i >= len(train_data):
            break

        sample = train_data[i]
        trajectory = np.array(sample["trajectory"])

        # Plot trajectory
        ax.plot(trajectory[:, 0], trajectory[:, 1], "b-", alpha=0.7)
        ax.scatter(trajectory[0, 0], trajectory[0, 1], c="green", s=50, label="Start")
        ax.scatter(trajectory[-1, 0], trajectory[-1, 1], c="red", s=50, label="End")

        # Add ground line
        ax.axhline(y=0, color="brown", linestyle="--", alpha=0.5)

        # Labels
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(f"g={sample['physics_params']['gravity']:.1f}")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-1, max(trajectory[:, 1].max() + 1, 15))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sample_trajectories.png"))
    print(f"Saved visualization to {output_dir}/sample_trajectories.png")


if __name__ == "__main__":
    # Generate training data
    output_dir = "data/physics_training"

    print("Generating physics training scenarios...")
    scenarios = generate_training_scenarios()
    print(f"Created {len(scenarios)} training scenarios")

    print("\nSimulating trajectories and saving data...")
    save_training_data(scenarios, output_dir)

    print("\nGenerating test scenarios...")
    test_scenarios = generate_test_scenarios()
    test_output = os.path.join(output_dir, "test_scenarios.json")
    with open(test_output, "w") as f:
        json.dump(test_scenarios, f, indent=2)
    print(f"Saved {len(test_scenarios)} test scenarios to {test_output}")

    print("\nCreating visualizations...")
    visualize_sample_trajectories(output_dir)

    print("\nDone! Training data ready for neural physics executor.")
