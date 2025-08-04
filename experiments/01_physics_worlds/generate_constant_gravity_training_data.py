#!/usr/bin/env python3
"""
Generate comprehensive constant gravity training data for baseline models.
"""

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np


class ConstantGravityPhysicsSimulator:
    """Simulates 2D physics with constant gravity for training data."""

    def __init__(self, world_width=800, world_height=600, dt=1 / 60.0):
        self.world_width = world_width
        self.world_height = world_height
        self.dt = dt
        self.pixel_to_meter = 40.0  # 40 pixels = 1 meter

    def simulate_trajectory(
        self,
        initial_state: np.ndarray,
        gravity: float = -9.8,
        friction: float = 0.1,
        elasticity: float = 0.8,
        duration: float = 3.0,
    ) -> np.ndarray:
        """
        Simulate a 2-ball trajectory with constant physics.

        Args:
            initial_state: [x1, y1, x2, y2, vx1, vy1, vx2, vy2, m1, m2, r1, r2]
            gravity: Gravity in m/s² (negative for downward)
            friction: Linear friction coefficient
            elasticity: Collision restitution
            duration: Simulation duration in seconds

        Returns:
            trajectory: Array of shape (timesteps, 17) with full state
        """
        state = initial_state.copy()
        trajectory = []

        # Convert to pixel units
        gravity_pixels = gravity * self.pixel_to_meter
        timesteps = int(duration / self.dt)

        for t in range(timesteps):
            # Extract current state
            x1, y1, x2, y2 = state[0], state[1], state[2], state[3]
            vx1, vy1, vx2, vy2 = state[4], state[5], state[6], state[7]
            m1, m2 = state[8], state[9]
            r1, r2 = state[10], state[11]

            # Apply forces
            # Gravity
            ay1 = (
                -gravity_pixels
            )  # Negative because y-axis points down in screen coords
            ay2 = -gravity_pixels

            # Friction
            ax1 = -friction * vx1
            ax2 = -friction * vx2
            ay1 -= friction * vy1
            ay2 -= friction * vy2

            # Update velocities
            vx1 += ax1 * self.dt
            vy1 += ay1 * self.dt
            vx2 += ax2 * self.dt
            vy2 += ay2 * self.dt

            # Update positions
            x1 += vx1 * self.dt
            y1 += vy1 * self.dt
            x2 += vx2 * self.dt
            y2 += vy2 * self.dt

            # Wall collisions
            if x1 - r1 <= 0 or x1 + r1 >= self.world_width:
                vx1 = -vx1 * elasticity
                x1 = np.clip(x1, r1, self.world_width - r1)
            if y1 - r1 <= 0 or y1 + r1 >= self.world_height:
                vy1 = -vy1 * elasticity
                y1 = np.clip(y1, r1, self.world_height - r1)

            if x2 - r2 <= 0 or x2 + r2 >= self.world_width:
                vx2 = -vx2 * elasticity
                x2 = np.clip(x2, r2, self.world_width - r2)
            if y2 - r2 <= 0 or y2 + r2 >= self.world_height:
                vy2 = -vy2 * elasticity
                y2 = np.clip(y2, r2, self.world_height - r2)

            # Ball-ball collision
            dx = x2 - x1
            dy = y2 - y1
            dist = np.sqrt(dx**2 + dy**2)

            if dist < r1 + r2 and dist > 0:
                # Collision detected
                nx = dx / dist
                ny = dy / dist

                # Relative velocity
                dvx = vx2 - vx1
                dvy = vy2 - vy1
                dvn = dvx * nx + dvy * ny

                if dvn < 0:  # Balls moving towards each other
                    # Impulse
                    impulse = 2 * dvn / (1 / m1 + 1 / m2) * elasticity

                    # Update velocities
                    vx1 += impulse * nx / m1
                    vy1 += impulse * ny / m1
                    vx2 -= impulse * nx / m2
                    vy2 -= impulse * ny / m2

                    # Separate balls
                    overlap = r1 + r2 - dist
                    separate = overlap / 2
                    x1 -= separate * nx
                    y1 -= separate * ny
                    x2 += separate * nx
                    y2 += separate * ny

            # Update state
            state[0], state[1], state[2], state[3] = x1, y1, x2, y2
            state[4], state[5], state[6], state[7] = vx1, vy1, vx2, vy2

            # Record trajectory (full state for compatibility)
            trajectory_point = [
                t * self.dt,  # time
                x1,
                y1,
                vx1,
                vy1,
                ax1,
                ay1,
                m1,
                r1,  # ball 1
                x2,
                y2,
                vx2,
                vy2,
                ax2,
                ay2,
                m2,
                r2,  # ball 2
            ]
            trajectory.append(trajectory_point)

        return np.array(trajectory)


def generate_diverse_initial_conditions(n_samples: int) -> List[np.ndarray]:
    """Generate diverse initial conditions for training."""
    initial_conditions = []

    for i in range(n_samples):
        # Random masses and radii
        m1 = np.random.uniform(1.0, 5.0)
        m2 = np.random.uniform(1.0, 5.0)
        r1 = np.random.uniform(10, 30)
        r2 = np.random.uniform(10, 30)

        # Random positions (ensure no overlap)
        while True:
            x1 = np.random.uniform(r1 + 50, 750 - r1)
            y1 = np.random.uniform(r1 + 50, 550 - r1)
            x2 = np.random.uniform(r2 + 50, 750 - r2)
            y2 = np.random.uniform(r2 + 50, 550 - r2)

            dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if dist > r1 + r2 + 10:  # Ensure separation
                break

        # Random velocities
        vx1 = np.random.uniform(-200, 200)
        vy1 = np.random.uniform(-200, 200)
        vx2 = np.random.uniform(-200, 200)
        vy2 = np.random.uniform(-200, 200)

        initial_state = np.array([x1, y1, x2, y2, vx1, vy1, vx2, vy2, m1, m2, r1, r2])
        initial_conditions.append(initial_state)

    return initial_conditions


def generate_constant_gravity_data(
    n_samples: int = 5000, gravity_range=(-12, -8)
) -> List[Dict]:
    """Generate training data with constant gravity values."""
    print(f"Generating {n_samples} constant gravity trajectories...")

    simulator = ConstantGravityPhysicsSimulator()
    data = []

    # Generate diverse initial conditions
    initial_conditions = generate_diverse_initial_conditions(n_samples)

    for i, initial_state in enumerate(initial_conditions):
        # Sample gravity from range
        gravity = np.random.uniform(*gravity_range)

        # Random physics parameters
        friction = np.random.uniform(0.05, 0.15)
        elasticity = np.random.uniform(0.7, 0.9)

        # Simulate trajectory
        trajectory = simulator.simulate_trajectory(
            initial_state,
            gravity=gravity,
            friction=friction,
            elasticity=elasticity,
            duration=3.0,
        )

        # Extract initial ball info
        x1, y1, x2, y2 = initial_state[:4]
        vx1, vy1, vx2, vy2 = initial_state[4:8]
        m1, m2, r1, r2 = initial_state[8:12]

        # Create data entry
        data_entry = {
            "sample_id": f"constant_gravity_{i:05d}",
            "physics_type": "constant_gravity",
            "physics_config": {
                "gravity": float(gravity),
                "friction": float(friction),
                "elasticity": float(elasticity),
                "world_width": 800,
                "world_height": 600,
                "dt": 1 / 60.0,
            },
            "initial_balls": [
                {
                    "position": [float(x1), float(y1)],
                    "velocity": [float(vx1), float(vy1)],
                    "mass": float(m1),
                    "radius": float(r1),
                },
                {
                    "position": [float(x2), float(y2)],
                    "velocity": [float(vx2), float(vy2)],
                    "mass": float(m2),
                    "radius": float(r2),
                },
            ],
            "trajectory": trajectory.tolist(),
            "num_balls": 2,
            "sequence_length": len(trajectory),
            "true_ood": False,
            "ood_type": "in_distribution",
        }

        data.append(data_entry)

        if (i + 1) % 500 == 0:
            print(f"  Generated {i + 1}/{n_samples} trajectories")

    return data


def main():
    """Generate and save constant gravity training data."""
    print("=" * 60)
    print("Generating Constant Gravity Training Data")
    print("=" * 60)

    # Create output directory
    output_dir = Path("data/processed/constant_gravity_training")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate data
    training_data = generate_constant_gravity_data(n_samples=5000)

    print(f"\nGenerated {len(training_data)} training trajectories")

    # Save data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save trajectories
    with open(output_dir / f"constant_gravity_data_{timestamp}.pkl", "wb") as f:
        pickle.dump(training_data, f)

    # Save metadata
    metadata = {
        "generation_time": timestamp,
        "n_samples": len(training_data),
        "physics_type": "constant_gravity",
        "gravity_range": [-12, -8],
        "description": "Training data with constant gravity values",
    }

    with open(output_dir / f"constant_gravity_metadata_{timestamp}.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✓ Data saved to: {output_dir}")
    print(f"✓ Generated {len(training_data)} trajectories")

    return training_data


if __name__ == "__main__":
    data = main()
