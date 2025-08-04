"""Generate extreme OOD physics scenarios: rotating frames and spring coupling."""

import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.imports import setup_project_paths

setup_project_paths()

from utils.config import setup_environment
from utils.paths import get_data_path


class ExtremeOODPhysicsGenerator:
    """Generator for extreme out-of-distribution physics scenarios."""

    def __init__(self, width=800, height=600, ball_radius=10):
        self.width = width
        self.height = height
        self.ball_radius = ball_radius
        self.dt = 1 / 60  # 60 FPS
        self.pixels_per_meter = 40  # 40 pixels = 1 meter

    def generate_rotating_frame_trajectory(self, omega=0.5, n_steps=50):
        """Generate trajectory in rotating reference frame with Coriolis forces.

        Args:
            omega: Angular velocity of rotating frame (rad/s)
            n_steps: Number of timesteps
        """
        # Initial conditions
        pos1 = np.array([self.width / 4, self.height / 2], dtype=np.float32)
        pos2 = np.array([3 * self.width / 4, self.height / 2], dtype=np.float32)
        vel1 = np.array([50.0, -100.0], dtype=np.float32)  # pixels/s
        vel2 = np.array([-50.0, -80.0], dtype=np.float32)

        trajectory = []

        for t in range(n_steps):
            # Store current state
            state = np.concatenate([pos1, vel1, pos2, vel2])
            trajectory.append(state)

            # Gravity (standard)
            g = -392.0  # pixels/s² (9.8 m/s²)

            # Coriolis acceleration: a_cor = -2 * omega × v
            # For 2D rotation about z-axis: a_cor = 2*omega*[-vy, vx]
            coriolis1 = 2 * omega * np.array([-vel1[1], vel1[0]])
            coriolis2 = 2 * omega * np.array([-vel2[1], vel2[0]])

            # Centrifugal acceleration: a_cen = omega² * r
            # In rotating frame, objects experience outward force
            center = np.array([self.width / 2, self.height / 2])
            r1 = pos1 - center
            r2 = pos2 - center
            centrifugal1 = omega**2 * r1
            centrifugal2 = omega**2 * r2

            # Total acceleration
            acc1 = np.array([0, g]) + coriolis1 + centrifugal1
            acc2 = np.array([0, g]) + coriolis2 + centrifugal2

            # Update velocities
            vel1 += acc1 * self.dt
            vel2 += acc2 * self.dt

            # Update positions
            pos1 += vel1 * self.dt
            pos2 += vel2 * self.dt

            # Handle wall collisions with damping
            for pos, vel in [(pos1, vel1), (pos2, vel2)]:
                if pos[0] <= self.ball_radius:
                    pos[0] = self.ball_radius
                    vel[0] = -vel[0] * 0.8
                elif pos[0] >= self.width - self.ball_radius:
                    pos[0] = self.width - self.ball_radius
                    vel[0] = -vel[0] * 0.8

                if pos[1] <= self.ball_radius:
                    pos[1] = self.ball_radius
                    vel[1] = -vel[1] * 0.8
                elif pos[1] >= self.height - self.ball_radius:
                    pos[1] = self.height - self.ball_radius
                    vel[1] = -vel[1] * 0.8

        return np.array(trajectory)

    def generate_spring_coupled_trajectory(self, k=50.0, rest_length=200.0, n_steps=50):
        """Generate trajectory with spring coupling between balls.

        Args:
            k: Spring constant (N/m in physics units)
            rest_length: Rest length of spring (pixels)
            n_steps: Number of timesteps
        """
        # Initial conditions - start stretched
        pos1 = np.array([self.width / 4, self.height / 2], dtype=np.float32)
        pos2 = np.array([3 * self.width / 4, self.height / 2], dtype=np.float32)
        vel1 = np.array([0.0, -50.0], dtype=np.float32)
        vel2 = np.array([0.0, -50.0], dtype=np.float32)

        # Mass of balls (arbitrary units)
        m1 = m2 = 1.0

        trajectory = []

        for t in range(n_steps):
            # Store current state
            state = np.concatenate([pos1, vel1, pos2, vel2])
            trajectory.append(state)

            # Gravity
            g = -392.0  # pixels/s²
            gravity_force = np.array([0, m1 * g])

            # Spring force
            displacement = pos2 - pos1
            distance = np.linalg.norm(displacement)
            direction = displacement / distance if distance > 0 else np.array([1, 0])

            # Hooke's law: F = -k * (x - x0)
            spring_magnitude = k * (distance - rest_length)
            spring_force = spring_magnitude * direction

            # Forces on each ball
            # Ball 1: gravity + spring pulling toward ball 2
            force1 = gravity_force + spring_force
            # Ball 2: gravity - spring (pulling toward ball 1)
            force2 = gravity_force - spring_force

            # Accelerations
            acc1 = force1 / m1
            acc2 = force2 / m2

            # Update velocities with damping
            damping = 0.99
            vel1 = vel1 * damping + acc1 * self.dt
            vel2 = vel2 * damping + acc2 * self.dt

            # Update positions
            pos1 += vel1 * self.dt
            pos2 += vel2 * self.dt

            # Handle wall collisions
            for pos, vel in [(pos1, vel1), (pos2, vel2)]:
                if pos[0] <= self.ball_radius:
                    pos[0] = self.ball_radius
                    vel[0] = -vel[0] * 0.8
                elif pos[0] >= self.width - self.ball_radius:
                    pos[0] = self.width - self.ball_radius
                    vel[0] = -vel[0] * 0.8

                if pos[1] <= self.ball_radius:
                    pos[1] = self.ball_radius
                    vel[1] = -vel[1] * 0.8
                elif pos[1] >= self.height - self.ball_radius:
                    pos[1] = self.height - self.ball_radius
                    vel[1] = -vel[1] * 0.8

        return np.array(trajectory)

    def generate_magnetic_field_trajectory(self, B=0.001, charge=1.0, n_steps=50):
        """Generate trajectory in uniform magnetic field (Lorentz force).

        Args:
            B: Magnetic field strength (Tesla equivalent)
            charge: Charge of particles
            n_steps: Number of timesteps
        """
        # Initial conditions
        pos1 = np.array([self.width / 4, self.height / 3], dtype=np.float32)
        pos2 = np.array([3 * self.width / 4, self.height / 3], dtype=np.float32)
        vel1 = np.array([100.0, 0.0], dtype=np.float32)
        vel2 = np.array([-100.0, 0.0], dtype=np.float32)

        # Mass and charge
        m = 1.0
        q1 = charge
        q2 = -charge  # Opposite charges

        trajectory = []

        for t in range(n_steps):
            # Store current state
            state = np.concatenate([pos1, vel1, pos2, vel2])
            trajectory.append(state)

            # Gravity
            g = -392.0
            gravity_force = np.array([0, m * g])

            # Lorentz force: F = q(v × B)
            # For B in z-direction: F = qB[-vy, vx]
            lorentz1 = q1 * B * self.pixels_per_meter * np.array([-vel1[1], vel1[0]])
            lorentz2 = q2 * B * self.pixels_per_meter * np.array([-vel2[1], vel2[0]])

            # Total force
            force1 = gravity_force + lorentz1
            force2 = gravity_force + lorentz2

            # Update velocities
            vel1 += (force1 / m) * self.dt
            vel2 += (force2 / m) * self.dt

            # Update positions
            pos1 += vel1 * self.dt
            pos2 += vel2 * self.dt

            # Handle collisions
            for pos, vel in [(pos1, vel1), (pos2, vel2)]:
                if (
                    pos[0] <= self.ball_radius
                    or pos[0] >= self.width - self.ball_radius
                ):
                    vel[0] *= -0.8
                    pos[0] = np.clip(
                        pos[0], self.ball_radius, self.width - self.ball_radius
                    )
                if (
                    pos[1] <= self.ball_radius
                    or pos[1] >= self.height - self.ball_radius
                ):
                    vel[1] *= -0.8
                    pos[1] = np.clip(
                        pos[1], self.ball_radius, self.height - self.ball_radius
                    )

        return np.array(trajectory)

    def generate_dataset(self, physics_type, n_trajectories=100, **kwargs):
        """Generate dataset for specific physics type."""
        trajectories = []

        generators = {
            "rotating_frame": self.generate_rotating_frame_trajectory,
            "spring_coupled": self.generate_spring_coupled_trajectory,
            "magnetic_field": self.generate_magnetic_field_trajectory,
        }

        if physics_type not in generators:
            raise ValueError(f"Unknown physics type: {physics_type}")

        generator = generators[physics_type]

        for i in range(n_trajectories):
            # Add some randomness to parameters
            if physics_type == "rotating_frame":
                omega = kwargs.get("omega", 0.5) * (0.8 + 0.4 * np.random.random())
                traj = generator(omega=omega, n_steps=50)
            elif physics_type == "spring_coupled":
                k = kwargs.get("k", 50.0) * (0.7 + 0.6 * np.random.random())
                rest_length = kwargs.get("rest_length", 200.0) * (
                    0.8 + 0.4 * np.random.random()
                )
                traj = generator(k=k, rest_length=rest_length, n_steps=50)
            elif physics_type == "magnetic_field":
                B = kwargs.get("B", 0.001) * (0.5 + 1.0 * np.random.random())
                charge = kwargs.get("charge", 1.0) * np.random.choice([-1, 1])
                traj = generator(B=B, charge=charge, n_steps=50)

            trajectories.append(traj)

        return np.array(trajectories)


def main():
    """Generate extreme OOD datasets."""
    config = setup_environment()

    print("Generating Extreme OOD Physics Data")
    print("=" * 60)

    generator = ExtremeOODPhysicsGenerator()
    output_dir = get_data_path() / "true_ood_physics"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Physics configurations
    configs = {
        "rotating_frame": {
            "omega": 0.5,  # rad/s
            "description": "Rotating reference frame with Coriolis and centrifugal forces",
        },
        "spring_coupled": {
            "k": 50.0,  # Spring constant
            "rest_length": 200.0,  # pixels
            "description": "Spring coupling between balls",
        },
        "magnetic_field": {
            "B": 0.001,  # Tesla-like units
            "charge": 1.0,
            "description": "Charged particles in magnetic field (Lorentz force)",
        },
    }

    # Generate each type
    for physics_type, config in configs.items():
        print(f"\nGenerating {physics_type} physics...")
        print(f"  {config['description']}")

        # Generate trajectories
        trajectories = generator.generate_dataset(
            physics_type,
            n_trajectories=100,
            **{k: v for k, v in config.items() if k != "description"},
        )

        # Create metadata
        metadata = {
            "physics_type": physics_type,
            "description": config["description"],
            "parameters": {k: v for k, v in config.items() if k != "description"},
            "n_trajectories": len(trajectories),
            "trajectory_length": trajectories.shape[1],
            "state_dim": trajectories.shape[2],
            "timestamp": timestamp,
            "generator_info": {
                "width": generator.width,
                "height": generator.height,
                "dt": generator.dt,
                "pixels_per_meter": generator.pixels_per_meter,
            },
        }

        # Save data
        filename = f"{physics_type}_physics_{timestamp}.pkl"
        filepath = output_dir / filename

        with open(filepath, "wb") as f:
            pickle.dump({"trajectories": trajectories, "metadata": metadata}, f)

        print(f"  Saved: {filename}")
        print(f"  Shape: {trajectories.shape}")

        # Quick stats
        print(
            f"  Avg displacement: {np.mean([np.linalg.norm(t[-1, :2] - t[0, :2]) for t in trajectories]):.1f} pixels"
        )

    # Create summary
    summary = {
        "timestamp": timestamp,
        "physics_types": list(configs.keys()),
        "descriptions": {k: v["description"] for k, v in configs.items()},
        "output_dir": str(output_dir),
    }

    with open(output_dir / f"extreme_ood_summary_{timestamp}.pkl", "wb") as f:
        pickle.dump(summary, f)

    print(f"\nAll extreme OOD data generated successfully!")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
