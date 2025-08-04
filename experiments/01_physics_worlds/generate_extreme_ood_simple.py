"""Simplified extreme OOD physics generator without environment dependencies."""

import pickle
from datetime import datetime
from pathlib import Path

import numpy as np


class SimpleExtremeOODGenerator:
    """Generator for extreme OOD physics scenarios."""

    def __init__(self, width=800, height=600, ball_radius=10):
        self.width = width
        self.height = height
        self.ball_radius = ball_radius
        self.dt = 1 / 60  # 60 FPS

    def generate_rotating_frame(self, omega=0.5, n_steps=50):
        """Rotating reference frame with Coriolis forces."""
        # Initial positions and velocities
        pos1 = np.array([200.0, 300.0], dtype=np.float32)
        pos2 = np.array([600.0, 300.0], dtype=np.float32)
        vel1 = np.array([50.0, -100.0], dtype=np.float32)
        vel2 = np.array([-50.0, -80.0], dtype=np.float32)

        trajectory = []
        center = np.array([400.0, 300.0])  # Center of rotation

        for _ in range(n_steps):
            # Store state
            trajectory.append(np.concatenate([pos1, vel1, pos2, vel2]))

            # Coriolis: a = -2ω × v (in 2D: rotate v by 90° and scale)
            coriolis1 = 2 * omega * np.array([-vel1[1], vel1[0]])
            coriolis2 = 2 * omega * np.array([-vel2[1], vel2[0]])

            # Centrifugal: a = ω²r (outward from center)
            r1 = pos1 - center
            r2 = pos2 - center
            centrifugal1 = omega**2 * r1
            centrifugal2 = omega**2 * r2

            # Total acceleration (gravity + fictitious forces)
            acc1 = np.array([0.0, -392.0]) + coriolis1 + centrifugal1
            acc2 = np.array([0.0, -392.0]) + coriolis2 + centrifugal2

            # Update
            vel1 += acc1 * self.dt
            vel2 += acc2 * self.dt
            pos1 += vel1 * self.dt
            pos2 += vel2 * self.dt

            # Boundaries
            for pos, vel in [(pos1, vel1), (pos2, vel2)]:
                if pos[0] < 10 or pos[0] > 790:
                    vel[0] *= -0.8
                    pos[0] = np.clip(pos[0], 10, 790)
                if pos[1] < 10 or pos[1] > 590:
                    vel[1] *= -0.8
                    pos[1] = np.clip(pos[1], 10, 590)

        return np.array(trajectory)

    def generate_spring_coupled(self, k=50.0, rest_length=200.0, n_steps=50):
        """Spring coupling between balls."""
        # Start stretched
        pos1 = np.array([200.0, 300.0], dtype=np.float32)
        pos2 = np.array([600.0, 300.0], dtype=np.float32)
        vel1 = np.array([0.0, -50.0], dtype=np.float32)
        vel2 = np.array([0.0, -50.0], dtype=np.float32)

        trajectory = []

        for _ in range(n_steps):
            trajectory.append(np.concatenate([pos1, vel1, pos2, vel2]))

            # Spring force
            delta = pos2 - pos1
            dist = np.linalg.norm(delta)
            if dist > 0:
                direction = delta / dist
                spring_force = k * (dist - rest_length) * direction
            else:
                spring_force = np.zeros(2)

            # Total forces
            gravity = np.array([0.0, -392.0])
            force1 = gravity + spring_force
            force2 = gravity - spring_force

            # Update with damping
            vel1 = 0.99 * vel1 + force1 * self.dt
            vel2 = 0.99 * vel2 + force2 * self.dt
            pos1 += vel1 * self.dt
            pos2 += vel2 * self.dt

            # Boundaries
            for pos, vel in [(pos1, vel1), (pos2, vel2)]:
                if pos[0] < 10 or pos[0] > 790:
                    vel[0] *= -0.8
                    pos[0] = np.clip(pos[0], 10, 790)
                if pos[1] < 10 or pos[1] > 590:
                    vel[1] *= -0.8
                    pos[1] = np.clip(pos[1], 10, 590)

        return np.array(trajectory)


def main():
    print("Generating Extreme OOD Physics (Simplified)")
    print("=" * 50)

    generator = SimpleExtremeOODGenerator()
    output_dir = Path("data/true_ood_physics")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Generate rotating frame physics
    print("\n1. Generating rotating frame physics...")
    rotating_trajectories = []
    for i in range(100):
        omega = 0.3 + 0.4 * np.random.random()  # 0.3 to 0.7 rad/s
        traj = generator.generate_rotating_frame(omega=omega)
        rotating_trajectories.append(traj)

    rotating_data = {
        "trajectories": np.array(rotating_trajectories),
        "metadata": {
            "physics_type": "rotating_frame",
            "description": "Coriolis and centrifugal forces",
            "omega_range": [0.3, 0.7],
            "timestamp": timestamp,
        },
    }

    filename = f"rotating_frame_physics_{timestamp}.pkl"
    with open(output_dir / filename, "wb") as f:
        pickle.dump(rotating_data, f)
    print(f"   Saved: {filename}")
    print(f"   Shape: {rotating_data['trajectories'].shape}")

    # Generate spring coupled physics
    print("\n2. Generating spring coupled physics...")
    spring_trajectories = []
    for i in range(100):
        k = 30 + 40 * np.random.random()  # 30 to 70 N/m
        rest_length = 150 + 100 * np.random.random()  # 150 to 250 pixels
        traj = generator.generate_spring_coupled(k=k, rest_length=rest_length)
        spring_trajectories.append(traj)

    spring_data = {
        "trajectories": np.array(spring_trajectories),
        "metadata": {
            "physics_type": "spring_coupled",
            "description": "Spring force between balls",
            "k_range": [30, 70],
            "rest_length_range": [150, 250],
            "timestamp": timestamp,
        },
    }

    filename = f"spring_coupled_physics_{timestamp}.pkl"
    with open(output_dir / filename, "wb") as f:
        pickle.dump(spring_data, f)
    print(f"   Saved: {filename}")
    print(f"   Shape: {spring_data['trajectories'].shape}")

    print(f"\nExtreme OOD data generated successfully!")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
