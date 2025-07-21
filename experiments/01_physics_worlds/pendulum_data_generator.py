"""
Pendulum data generation for mechanism shift experiments.
Generates fixed-length (training) and variable-length (test) pendulum trajectories.
"""

import numpy as np
import pickle
import json
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass, asdict
import time
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

@dataclass
class PendulumConfig:
    """Configuration for pendulum physics"""
    length: float = 1.0  # Pendulum length in meters
    gravity: float = 9.8  # Gravitational acceleration
    damping: float = 0.01  # Air resistance coefficient
    
    # For variable length pendulum
    length_variation: float = 0.0  # Amplitude of length variation (0 = fixed)
    length_frequency: float = 0.1  # Frequency of length variation
    
    # Time settings
    dt: float = 0.01  # Time step
    fps: int = 60  # Frames per second for output
    
@dataclass
class PendulumDataConfig:
    """Configuration for data generation"""
    num_samples: int = 10000
    sequence_length: int = 300  # 5 seconds at 60fps
    
    # Initial condition ranges
    theta_range: Tuple[float, float] = (-np.pi/6, np.pi/6)  # ±30 degrees
    theta_dot_range: Tuple[float, float] = (-1.0, 1.0)  # rad/s
    
    # Physics variations for training data
    length_range: Tuple[float, float] = (0.8, 1.2)  # 80-120cm
    gravity_range: Tuple[float, float] = (9.0, 10.6)  # Slight variations
    damping_range: Tuple[float, float] = (0.005, 0.02)  # Light damping
    
    # Test mechanism shift parameters
    test_length_variation: float = 0.2  # 20% variation amplitude
    test_length_frequency: float = 0.1  # Frequency of variation
    
    # Output settings
    output_dir: str = "data/processed/pendulum"
    
    # Quality validation
    energy_threshold: float = 0.15  # 15% energy variance allowed
    min_trajectory_length: int = 50

class PendulumSimulator:
    """Simulate pendulum dynamics with RK4 integration"""
    
    def __init__(self, config: PendulumConfig):
        self.config = config
        
    def _compute_acceleration(self, theta: float, theta_dot: float, t: float) -> float:
        """Compute angular acceleration including variable length effects"""
        L = self._get_length(t)
        L_dot = self._get_length_derivative(t)
        
        # Standard pendulum term
        gravity_term = -(self.config.gravity / L) * np.sin(theta)
        
        # Damping term
        damping_term = -self.config.damping * theta_dot
        
        # Variable length term (from conservation of angular momentum)
        if abs(L_dot) > 1e-10:  # Only if length is changing
            variable_term = -2 * (L_dot / L) * theta_dot
        else:
            variable_term = 0
            
        return gravity_term + damping_term + variable_term
    
    def _get_length(self, t: float) -> float:
        """Get pendulum length at time t"""
        if self.config.length_variation > 0:
            return self.config.length * (1 + self.config.length_variation * 
                                       np.sin(self.config.length_frequency * t))
        return self.config.length
    
    def _get_length_derivative(self, t: float) -> float:
        """Get rate of change of pendulum length"""
        if self.config.length_variation > 0:
            return (self.config.length * self.config.length_variation * 
                    self.config.length_frequency * 
                    np.cos(self.config.length_frequency * t))
        return 0.0
    
    def simulate(self, theta0: float, theta_dot0: float, duration: float) -> Dict[str, np.ndarray]:
        """Simulate pendulum motion using RK4 integration"""
        dt = self.config.dt
        steps = int(duration / dt)
        
        # Arrays to store trajectory
        t_arr = np.zeros(steps)
        theta_arr = np.zeros(steps)
        theta_dot_arr = np.zeros(steps)
        x_arr = np.zeros(steps)
        y_arr = np.zeros(steps)
        length_arr = np.zeros(steps)
        energy_arr = np.zeros(steps)
        
        # Initial conditions
        theta = theta0
        theta_dot = theta_dot0
        
        for i in range(steps):
            t = i * dt
            t_arr[i] = t
            theta_arr[i] = theta
            theta_dot_arr[i] = theta_dot
            
            # Get current length
            L = self._get_length(t)
            length_arr[i] = L
            
            # Convert to Cartesian coordinates (origin at pivot)
            x_arr[i] = L * np.sin(theta)
            y_arr[i] = -L * np.cos(theta)  # Negative because y increases downward
            
            # Compute energy (for validation)
            KE = 0.5 * L**2 * theta_dot**2
            PE = -self.config.gravity * L * np.cos(theta)
            energy_arr[i] = KE + PE
            
            # RK4 integration step
            if i < steps - 1:
                k1v = self._compute_acceleration(theta, theta_dot, t)
                k1x = theta_dot
                
                k2v = self._compute_acceleration(theta + k1x*dt/2, theta_dot + k1v*dt/2, t + dt/2)
                k2x = theta_dot + k1v*dt/2
                
                k3v = self._compute_acceleration(theta + k2x*dt/2, theta_dot + k2v*dt/2, t + dt/2)
                k3x = theta_dot + k2v*dt/2
                
                k4v = self._compute_acceleration(theta + k3x*dt, theta_dot + k3v*dt, t + dt)
                k4x = theta_dot + k3v*dt
                
                theta_dot += (k1v + 2*k2v + 2*k3v + k4v) * dt / 6
                theta += (k1x + 2*k2x + 2*k3x + k4x) * dt / 6
        
        # Downsample to desired FPS
        frame_interval = int(1.0 / (self.config.fps * dt))
        indices = np.arange(0, steps, frame_interval)
        
        return {
            't': t_arr[indices],
            'theta': theta_arr[indices],
            'theta_dot': theta_dot_arr[indices],
            'x': x_arr[indices],
            'y': y_arr[indices],
            'length': length_arr[indices],
            'energy': energy_arr[indices]
        }

class PendulumDataGenerator:
    """Generate pendulum simulation data for training and testing"""
    
    def __init__(self, config: PendulumDataConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def _generate_random_pendulum_config(self, mechanism: str = 'fixed') -> PendulumConfig:
        """Generate random pendulum configuration"""
        if mechanism == 'fixed':
            # Training distribution: fixed length with slight parameter variations
            return PendulumConfig(
                length=np.random.uniform(*self.config.length_range),
                gravity=np.random.uniform(*self.config.gravity_range),
                damping=np.random.uniform(*self.config.damping_range),
                length_variation=0.0,  # No variation for training
                length_frequency=0.0
            )
        else:  # time-varying
            # Test distribution: time-varying length (mechanism shift)
            return PendulumConfig(
                length=1.0,  # Base length
                gravity=9.8,  # Standard gravity
                damping=0.01,  # Standard damping
                length_variation=self.config.test_length_variation,
                length_frequency=self.config.test_length_frequency
            )
    
    def _generate_trajectory(self, args: Tuple[int, str]) -> Optional[Dict[str, Any]]:
        """Generate a single trajectory (for parallel processing)"""
        idx, mechanism = args
        
        # Random initial conditions
        theta0 = np.random.uniform(*self.config.theta_range)
        theta_dot0 = np.random.uniform(*self.config.theta_dot_range)
        
        # Random physics config
        physics_config = self._generate_random_pendulum_config(mechanism)
        
        # Simulate
        simulator = PendulumSimulator(physics_config)
        duration = self.config.sequence_length / 60.0  # 60 fps
        
        try:
            trajectory = simulator.simulate(theta0, theta_dot0, duration)
            
            # Validate trajectory
            if self._validate_trajectory(trajectory):
                return {
                    'trajectory': trajectory,
                    'physics_config': asdict(physics_config),
                    'initial_conditions': {
                        'theta': theta0,
                        'theta_dot': theta_dot0
                    },
                    'mechanism': mechanism
                }
        except Exception as e:
            print(f"Failed to generate trajectory {idx}: {e}")
            
        return None
    
    def _validate_trajectory(self, trajectory: Dict[str, np.ndarray]) -> bool:
        """Validate trajectory quality"""
        # Check minimum length
        if len(trajectory['t']) < self.config.min_trajectory_length:
            return False
            
        # Check energy conservation (for fixed length only)
        if trajectory['length'][0] == trajectory['length'][-1]:  # Fixed length
            energy = trajectory['energy']
            energy_var = np.std(energy) / np.mean(np.abs(energy))
            if energy_var > self.config.energy_threshold:
                return False
                
        return True
    
    def generate_dataset(self, mechanism: str = 'fixed', num_samples: Optional[int] = None) -> Dict[str, Any]:
        """Generate complete dataset"""
        if num_samples is None:
            num_samples = self.config.num_samples
            
        print(f"Generating {num_samples} {mechanism} pendulum trajectories...")
        
        # Prepare arguments for parallel processing
        args_list = [(i, mechanism) for i in range(num_samples)]
        
        # Generate trajectories in parallel
        with mp.Pool() as pool:
            trajectories = list(tqdm(
                pool.imap(self._generate_trajectory, args_list),
                total=num_samples
            ))
        
        # Filter out failed trajectories
        valid_trajectories = [t for t in trajectories if t is not None]
        print(f"Generated {len(valid_trajectories)} valid trajectories")
        
        # Organize data
        dataset = {
            'trajectories': valid_trajectories,
            'metadata': {
                'mechanism': mechanism,
                'num_samples': len(valid_trajectories),
                'config': asdict(self.config),
                'generation_time': time.strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
        return dataset
    
    def save_dataset(self, dataset: Dict[str, Any], filename: str):
        """Save dataset to disk"""
        filepath = self.output_dir / filename
        
        # Save pickle for fast loading
        with open(filepath.with_suffix('.pkl'), 'wb') as f:
            pickle.dump(dataset, f)
            
        # Save metadata as JSON for inspection
        metadata_path = filepath.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(dataset['metadata'], f, indent=2)
            
        print(f"Saved dataset to {filepath}")

def main():
    """Generate pendulum datasets for mechanism shift experiment"""
    
    # Create data generator
    config = PendulumDataConfig(
        num_samples=10000,  # Full dataset
        sequence_length=300  # 5 seconds at 60fps
    )
    generator = PendulumDataGenerator(config)
    
    # Generate training data (fixed length)
    print("Generating training data (fixed-length pendulum)...")
    train_data = generator.generate_dataset(mechanism='fixed', num_samples=8000)
    generator.save_dataset(train_data, 'pendulum_train')
    
    # Generate validation data (fixed length, different parameters)
    print("Generating validation data (fixed-length pendulum)...")
    val_data = generator.generate_dataset(mechanism='fixed', num_samples=1000)
    generator.save_dataset(val_data, 'pendulum_val')
    
    # Generate test data (fixed length, for interpolation testing)
    print("Generating test data (fixed-length pendulum)...")
    test_fixed = generator.generate_dataset(mechanism='fixed', num_samples=1000)
    generator.save_dataset(test_fixed, 'pendulum_test_fixed')
    
    # Generate OOD test data (time-varying length)
    print("Generating OOD test data (time-varying pendulum)...")
    test_ood = generator.generate_dataset(mechanism='time-varying', num_samples=1000)
    generator.save_dataset(test_ood, 'pendulum_test_ood')
    
    print("Dataset generation complete!")

if __name__ == "__main__":
    main()