"""Generate true OOD physics data with time-varying gravity and other modifications."""

import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Callable, Optional
import matplotlib.pyplot as plt
from datetime import datetime

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.imports import setup_project_paths
setup_project_paths()

from utils.config import setup_environment
from utils.paths import get_data_path, get_output_path


class TrueOODPhysicsGenerator:
    """Generate physics trajectories with true out-of-distribution modifications."""
    
    def __init__(self, 
                 n_balls: int = 2,
                 timesteps: int = 50,
                 dt: float = 0.1,
                 image_size: int = 128):
        """Initialize the generator.
        
        Args:
            n_balls: Number of balls in simulation
            timesteps: Number of time steps per trajectory
            dt: Time step size
            image_size: Size of the image (for pixel coordinates)
        """
        self.n_balls = n_balls
        self.timesteps = timesteps
        self.dt = dt
        self.image_size = image_size
        
        # Physics parameters (pixels/s^2 for acceleration)
        self.standard_gravity = -9.8 * 40  # Convert m/s^2 to pixels/s^2
        self.friction = 0.1
        
    def generate_standard_physics(self, n_samples: int) -> Dict[str, np.ndarray]:
        """Generate trajectories with standard constant gravity.
        
        Args:
            n_samples: Number of trajectories to generate
            
        Returns:
            Dictionary with 'trajectories' and 'metadata'
        """
        trajectories = []
        
        for i in range(n_samples):
            traj = self._simulate_trajectory(
                gravity_fn=lambda t: self.standard_gravity,
                initial_state=self._random_initial_state()
            )
            trajectories.append(traj)
        
        return {
            'trajectories': np.array(trajectories),
            'metadata': {
                'physics_type': 'standard',
                'gravity': self.standard_gravity,
                'n_balls': self.n_balls,
                'timesteps': self.timesteps,
                'dt': self.dt
            }
        }
    
    def generate_time_varying_gravity(self, 
                                    n_samples: int,
                                    amplitude: float = 0.3,
                                    frequency: float = 0.5) -> Dict[str, np.ndarray]:
        """Generate trajectories with time-varying gravity.
        
        Args:
            n_samples: Number of trajectories
            amplitude: Amplitude of gravity variation (fraction of base gravity)
            frequency: Frequency of variation (Hz)
            
        Returns:
            Dictionary with trajectories and metadata
        """
        trajectories = []
        
        # Define time-varying gravity function
        def gravity_fn(t):
            return self.standard_gravity * (1 + amplitude * np.sin(2 * np.pi * frequency * t))
        
        for i in range(n_samples):
            traj = self._simulate_trajectory(
                gravity_fn=gravity_fn,
                initial_state=self._random_initial_state()
            )
            trajectories.append(traj)
        
        return {
            'trajectories': np.array(trajectories),
            'metadata': {
                'physics_type': 'time_varying_gravity',
                'gravity_fn': f'g(t) = {self.standard_gravity} * (1 + {amplitude}*sin(2π*{frequency}*t))',
                'amplitude': amplitude,
                'frequency': frequency,
                'n_balls': self.n_balls,
                'timesteps': self.timesteps,
                'dt': self.dt
            }
        }
    
    def generate_height_dependent_gravity(self, 
                                        n_samples: int,
                                        scale_height: float = 100.0) -> Dict[str, np.ndarray]:
        """Generate trajectories with height-dependent gravity.
        
        Args:
            n_samples: Number of trajectories
            scale_height: Height scale for gravity variation (pixels)
            
        Returns:
            Dictionary with trajectories and metadata
        """
        trajectories = []
        
        # Define height-dependent gravity
        def gravity_fn_y(y):
            # Gravity weakens with height
            return self.standard_gravity * np.exp(-y / scale_height)
        
        for i in range(n_samples):
            traj = self._simulate_trajectory_height_dependent(
                gravity_fn_y=gravity_fn_y,
                initial_state=self._random_initial_state()
            )
            trajectories.append(traj)
        
        return {
            'trajectories': np.array(trajectories),
            'metadata': {
                'physics_type': 'height_dependent_gravity',
                'gravity_fn': f'g(y) = {self.standard_gravity} * exp(-y/{scale_height})',
                'scale_height': scale_height,
                'n_balls': self.n_balls,
                'timesteps': self.timesteps,
                'dt': self.dt
            }
        }
    
    def generate_rotating_frame(self, 
                              n_samples: int,
                              omega: float = 0.1) -> Dict[str, np.ndarray]:
        """Generate trajectories in rotating reference frame.
        
        Args:
            n_samples: Number of trajectories
            omega: Angular velocity of rotation (rad/s)
            
        Returns:
            Dictionary with trajectories and metadata
        """
        trajectories = []
        
        for i in range(n_samples):
            traj = self._simulate_rotating_frame(
                omega=omega,
                initial_state=self._random_initial_state()
            )
            trajectories.append(traj)
        
        return {
            'trajectories': np.array(trajectories),
            'metadata': {
                'physics_type': 'rotating_frame',
                'omega': omega,
                'fictitious_forces': ['centrifugal', 'coriolis'],
                'n_balls': self.n_balls,
                'timesteps': self.timesteps,
                'dt': self.dt
            }
        }
    
    def generate_spring_coupled(self, 
                              n_samples: int,
                              spring_constant: float = 100.0,
                              rest_length: float = 50.0) -> Dict[str, np.ndarray]:
        """Generate trajectories with springs between balls.
        
        Args:
            n_samples: Number of trajectories
            spring_constant: Spring constant (N/m in pixel units)
            rest_length: Rest length of spring (pixels)
            
        Returns:
            Dictionary with trajectories and metadata
        """
        if self.n_balls < 2:
            raise ValueError("Need at least 2 balls for spring coupling")
        
        trajectories = []
        
        for i in range(n_samples):
            traj = self._simulate_spring_system(
                k=spring_constant,
                rest_length=rest_length,
                initial_state=self._random_initial_state()
            )
            trajectories.append(traj)
        
        return {
            'trajectories': np.array(trajectories),
            'metadata': {
                'physics_type': 'spring_coupled',
                'spring_constant': spring_constant,
                'rest_length': rest_length,
                'n_balls': self.n_balls,
                'timesteps': self.timesteps,
                'dt': self.dt
            }
        }
    
    def _random_initial_state(self) -> np.ndarray:
        """Generate random initial state for balls.
        
        Returns:
            Initial state array of shape (n_balls * 4,)
            Format: [x1, y1, vx1, vy1, x2, y2, vx2, vy2, ...]
        """
        state = np.zeros(self.n_balls * 4)
        
        for i in range(self.n_balls):
            # Random position (keep away from edges)
            state[i*4] = np.random.uniform(20, self.image_size - 20)      # x
            state[i*4 + 1] = np.random.uniform(20, self.image_size - 20)  # y
            
            # Random velocity (pixels/s)
            state[i*4 + 2] = np.random.uniform(-50, 50)  # vx
            state[i*4 + 3] = np.random.uniform(-20, 50)  # vy
        
        return state
    
    def _simulate_trajectory(self, 
                           gravity_fn: Callable[[float], float],
                           initial_state: np.ndarray) -> np.ndarray:
        """Simulate trajectory with time-dependent gravity.
        
        Args:
            gravity_fn: Function mapping time to gravity value
            initial_state: Initial positions and velocities
            
        Returns:
            Trajectory array of shape (timesteps, n_balls * 4)
        """
        trajectory = np.zeros((self.timesteps, len(initial_state)))
        trajectory[0] = initial_state
        
        for t in range(1, self.timesteps):
            state = trajectory[t-1].copy()
            
            # Get current gravity
            g = gravity_fn(t * self.dt)
            
            # Update each ball
            for i in range(self.n_balls):
                idx = i * 4
                
                # Update positions
                state[idx] += state[idx + 2] * self.dt      # x += vx * dt
                state[idx + 1] += state[idx + 3] * self.dt  # y += vy * dt
                
                # Update velocities
                state[idx + 2] *= (1 - self.friction * self.dt)  # Apply friction
                state[idx + 3] += g * self.dt                    # Apply gravity
                state[idx + 3] *= (1 - self.friction * self.dt)  # Apply friction
                
                # Handle collisions with walls
                if state[idx] < 10 or state[idx] > self.image_size - 10:
                    state[idx + 2] *= -0.8  # Bounce with damping
                    state[idx] = np.clip(state[idx], 10, self.image_size - 10)
                
                if state[idx + 1] < 10 or state[idx + 1] > self.image_size - 10:
                    state[idx + 3] *= -0.8  # Bounce with damping
                    state[idx + 1] = np.clip(state[idx + 1], 10, self.image_size - 10)
            
            trajectory[t] = state
        
        return trajectory
    
    def _simulate_trajectory_height_dependent(self,
                                            gravity_fn_y: Callable[[float], float],
                                            initial_state: np.ndarray) -> np.ndarray:
        """Simulate with height-dependent gravity."""
        trajectory = np.zeros((self.timesteps, len(initial_state)))
        trajectory[0] = initial_state
        
        for t in range(1, self.timesteps):
            state = trajectory[t-1].copy()
            
            for i in range(self.n_balls):
                idx = i * 4
                
                # Get gravity based on current height
                y = state[idx + 1]
                g = gravity_fn_y(y)
                
                # Update positions
                state[idx] += state[idx + 2] * self.dt
                state[idx + 1] += state[idx + 3] * self.dt
                
                # Update velocities
                state[idx + 2] *= (1 - self.friction * self.dt)
                state[idx + 3] += g * self.dt
                state[idx + 3] *= (1 - self.friction * self.dt)
                
                # Collisions
                if state[idx] < 10 or state[idx] > self.image_size - 10:
                    state[idx + 2] *= -0.8
                    state[idx] = np.clip(state[idx], 10, self.image_size - 10)
                
                if state[idx + 1] < 10 or state[idx + 1] > self.image_size - 10:
                    state[idx + 3] *= -0.8
                    state[idx + 1] = np.clip(state[idx + 1], 10, self.image_size - 10)
            
            trajectory[t] = state
        
        return trajectory
    
    def _simulate_rotating_frame(self,
                               omega: float,
                               initial_state: np.ndarray) -> np.ndarray:
        """Simulate in rotating reference frame with fictitious forces."""
        trajectory = np.zeros((self.timesteps, len(initial_state)))
        trajectory[0] = initial_state
        
        center = np.array([self.image_size / 2, self.image_size / 2])
        
        for t in range(1, self.timesteps):
            state = trajectory[t-1].copy()
            
            for i in range(self.n_balls):
                idx = i * 4
                
                # Position and velocity
                r = np.array([state[idx], state[idx + 1]]) - center
                v = np.array([state[idx + 2], state[idx + 3]])
                
                # Fictitious forces in rotating frame
                # Centrifugal: omega^2 * r (outward)
                f_centrifugal = omega**2 * r
                
                # Coriolis: -2 * omega × v (perpendicular to velocity)
                f_coriolis = np.array([2 * omega * v[1], -2 * omega * v[0]])
                
                # Update positions
                state[idx] += state[idx + 2] * self.dt
                state[idx + 1] += state[idx + 3] * self.dt
                
                # Update velocities (including fictitious forces)
                state[idx + 2] += (f_centrifugal[0] + f_coriolis[0]) * self.dt
                state[idx + 2] *= (1 - self.friction * self.dt)
                
                state[idx + 3] += (self.standard_gravity + f_centrifugal[1] + f_coriolis[1]) * self.dt
                state[idx + 3] *= (1 - self.friction * self.dt)
                
                # Collisions
                if state[idx] < 10 or state[idx] > self.image_size - 10:
                    state[idx + 2] *= -0.8
                    state[idx] = np.clip(state[idx], 10, self.image_size - 10)
                
                if state[idx + 1] < 10 or state[idx + 1] > self.image_size - 10:
                    state[idx + 3] *= -0.8
                    state[idx + 1] = np.clip(state[idx + 1], 10, self.image_size - 10)
            
            trajectory[t] = state
        
        return trajectory
    
    def _simulate_spring_system(self,
                              k: float,
                              rest_length: float,
                              initial_state: np.ndarray) -> np.ndarray:
        """Simulate balls connected by springs."""
        trajectory = np.zeros((self.timesteps, len(initial_state)))
        trajectory[0] = initial_state
        
        for t in range(1, self.timesteps):
            state = trajectory[t-1].copy()
            forces = np.zeros_like(state)
            
            # Calculate spring forces between all pairs
            for i in range(self.n_balls):
                for j in range(i + 1, self.n_balls):
                    # Positions
                    r1 = state[i*4:i*4+2]
                    r2 = state[j*4:j*4+2]
                    
                    # Spring force
                    dr = r2 - r1
                    distance = np.linalg.norm(dr)
                    if distance > 0:
                        direction = dr / distance
                        spring_force = k * (distance - rest_length) * direction
                        
                        # Apply equal and opposite forces
                        forces[i*4:i*4+2] += spring_force
                        forces[j*4:j*4+2] -= spring_force
            
            # Update dynamics
            for i in range(self.n_balls):
                idx = i * 4
                
                # Update positions
                state[idx] += state[idx + 2] * self.dt
                state[idx + 1] += state[idx + 3] * self.dt
                
                # Update velocities (gravity + spring forces)
                state[idx + 2] += forces[idx] * self.dt
                state[idx + 2] *= (1 - self.friction * self.dt)
                
                state[idx + 3] += (self.standard_gravity + forces[idx + 1]) * self.dt
                state[idx + 3] *= (1 - self.friction * self.dt)
                
                # Collisions
                if state[idx] < 10 or state[idx] > self.image_size - 10:
                    state[idx + 2] *= -0.8
                    state[idx] = np.clip(state[idx], 10, self.image_size - 10)
                
                if state[idx + 1] < 10 or state[idx + 1] > self.image_size - 10:
                    state[idx + 3] *= -0.8
                    state[idx + 1] = np.clip(state[idx + 1], 10, self.image_size - 10)
            
            trajectory[t] = state
        
        return trajectory


def visualize_trajectory(trajectory: np.ndarray, 
                        title: str = "Physics Trajectory",
                        save_path: Optional[Path] = None):
    """Visualize a single trajectory."""
    n_balls = trajectory.shape[1] // 4
    
    plt.figure(figsize=(8, 8))
    colors = plt.cm.rainbow(np.linspace(0, 1, n_balls))
    
    for i in range(n_balls):
        x = trajectory[:, i*4]
        y = trajectory[:, i*4 + 1]
        plt.plot(x, y, color=colors[i], alpha=0.7, linewidth=2, label=f'Ball {i+1}')
        plt.scatter(x[0], y[0], color=colors[i], s=100, marker='o', edgecolor='black', label=f'Start {i+1}')
        plt.scatter(x[-1], y[-1], color=colors[i], s=100, marker='s', edgecolor='black', label=f'End {i+1}')
    
    plt.xlim(0, 128)
    plt.ylim(0, 128)
    plt.xlabel('X Position (pixels)')
    plt.ylabel('Y Position (pixels)')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.gca().invert_yaxis()  # Invert y-axis to match image coordinates
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def generate_all_datasets():
    """Generate all true OOD datasets."""
    generator = TrueOODPhysicsGenerator(n_balls=2, timesteps=50)
    
    # Create output directory
    output_dir = get_data_path() / "true_ood_physics"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate datasets
    datasets = {}
    
    print("Generating standard physics (baseline)...")
    datasets['standard'] = generator.generate_standard_physics(n_samples=1000)
    
    print("Generating time-varying gravity...")
    datasets['time_varying'] = generator.generate_time_varying_gravity(
        n_samples=500, amplitude=0.3, frequency=0.5
    )
    
    print("Generating height-dependent gravity...")
    datasets['height_dependent'] = generator.generate_height_dependent_gravity(
        n_samples=500, scale_height=100.0
    )
    
    print("Generating rotating frame physics...")
    datasets['rotating_frame'] = generator.generate_rotating_frame(
        n_samples=500, omega=0.1
    )
    
    print("Generating spring-coupled system...")
    datasets['spring_coupled'] = generator.generate_spring_coupled(
        n_samples=500, spring_constant=50.0, rest_length=40.0
    )
    
    # Save datasets
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for name, data in datasets.items():
        filename = output_dir / f"{name}_physics_{timestamp}.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved {name} dataset to {filename}")
    
    # Create visualizations
    viz_dir = get_output_path() / "true_ood_visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nCreating visualizations...")
    for name, data in datasets.items():
        # Visualize first trajectory
        traj = data['trajectories'][0]
        save_path = viz_dir / f"{name}_example_{timestamp}.png"
        visualize_trajectory(traj, title=f"{name.replace('_', ' ').title()} Physics", save_path=save_path)
    
    # Save metadata summary
    summary = {
        'timestamp': timestamp,
        'datasets': list(datasets.keys()),
        'samples_per_dataset': {name: len(data['trajectories']) for name, data in datasets.items()},
        'physics_parameters': {name: data['metadata'] for name, data in datasets.items()}
    }
    
    summary_path = output_dir / f"dataset_summary_{timestamp}.pkl"
    with open(summary_path, 'wb') as f:
        pickle.dump(summary, f)
    
    print(f"\nAll datasets generated successfully!")
    print(f"Output directory: {output_dir}")
    print(f"Visualizations: {viz_dir}")
    
    return datasets


if __name__ == "__main__":
    config = setup_environment()
    datasets = generate_all_datasets()