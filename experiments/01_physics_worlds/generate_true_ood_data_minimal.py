"""Generate minimal true OOD physics data for testing."""

import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.imports import setup_project_paths
setup_project_paths()

from utils.config import setup_environment
from utils.paths import get_data_path


def generate_time_varying_gravity_simple(n_samples=100, timesteps=50, dt=0.1):
    """Generate simple time-varying gravity trajectories."""
    
    trajectories = []
    
    for i in range(n_samples):
        # Initial state: [x1, y1, vx1, vy1, x2, y2, vx2, vy2]
        state = np.zeros((timesteps, 8))
        
        # Random initial positions (pixels)
        state[0, 0] = np.random.uniform(20, 108)  # x1
        state[0, 1] = np.random.uniform(20, 108)  # y1
        state[0, 4] = np.random.uniform(20, 108)  # x2
        state[0, 5] = np.random.uniform(20, 108)  # y2
        
        # Random initial velocities (pixels/s)
        state[0, 2] = np.random.uniform(-50, 50)   # vx1
        state[0, 3] = np.random.uniform(-20, 50)   # vy1
        state[0, 6] = np.random.uniform(-50, 50)   # vx2
        state[0, 7] = np.random.uniform(-20, 50)   # vy2
        
        # Simulate with time-varying gravity
        for t in range(1, timesteps):
            # Time-varying gravity: g(t) = -9.8 * 40 * (1 + 0.3*sin(0.5*t))
            # 40 is pixel conversion factor
            g = -9.8 * 40 * (1 + 0.3 * np.sin(0.5 * t * dt))
            
            # Update positions
            state[t, 0] = state[t-1, 0] + state[t-1, 2] * dt  # x1
            state[t, 1] = state[t-1, 1] + state[t-1, 3] * dt  # y1
            state[t, 4] = state[t-1, 4] + state[t-1, 6] * dt  # x2
            state[t, 5] = state[t-1, 5] + state[t-1, 7] * dt  # y2
            
            # Update velocities (with friction)
            friction = 0.1
            state[t, 2] = state[t-1, 2] * (1 - friction * dt)  # vx1
            state[t, 3] = state[t-1, 3] * (1 - friction * dt) + g * dt  # vy1
            state[t, 6] = state[t-1, 6] * (1 - friction * dt)  # vx2
            state[t, 7] = state[t-1, 7] * (1 - friction * dt) + g * dt  # vy2
            
            # Simple boundary conditions
            for idx in [0, 1, 4, 5]:  # positions
                state[t, idx] = np.clip(state[t, idx], 10, 118)
            
            # Bounce at walls
            if state[t, 0] <= 10 or state[t, 0] >= 118:
                state[t, 2] *= -0.8
            if state[t, 1] <= 10 or state[t, 1] >= 118:
                state[t, 3] *= -0.8
            if state[t, 4] <= 10 or state[t, 4] >= 118:
                state[t, 6] *= -0.8
            if state[t, 5] <= 10 or state[t, 5] >= 118:
                state[t, 7] *= -0.8
        
        trajectories.append(state)
    
    return {
        'trajectories': np.array(trajectories),
        'metadata': {
            'physics_type': 'time_varying_gravity',
            'gravity_fn': 'g(t) = -392 * (1 + 0.3*sin(0.5*t))',
            'n_samples': n_samples,
            'timesteps': timesteps,
            'dt': dt
        }
    }


def generate_constant_gravity_simple(n_samples=100, timesteps=50, dt=0.1):
    """Generate simple constant gravity trajectories for comparison."""
    
    trajectories = []
    g = -9.8 * 40  # Constant gravity in pixels/s^2
    
    for i in range(n_samples):
        state = np.zeros((timesteps, 8))
        
        # Same initial conditions as time-varying
        state[0, 0] = np.random.uniform(20, 108)
        state[0, 1] = np.random.uniform(20, 108)
        state[0, 4] = np.random.uniform(20, 108)
        state[0, 5] = np.random.uniform(20, 108)
        state[0, 2] = np.random.uniform(-50, 50)
        state[0, 3] = np.random.uniform(-20, 50)
        state[0, 6] = np.random.uniform(-50, 50)
        state[0, 7] = np.random.uniform(-20, 50)
        
        # Simulate with constant gravity
        for t in range(1, timesteps):
            # Update positions
            state[t, 0] = state[t-1, 0] + state[t-1, 2] * dt
            state[t, 1] = state[t-1, 1] + state[t-1, 3] * dt
            state[t, 4] = state[t-1, 4] + state[t-1, 6] * dt
            state[t, 5] = state[t-1, 5] + state[t-1, 7] * dt
            
            # Update velocities
            friction = 0.1
            state[t, 2] = state[t-1, 2] * (1 - friction * dt)
            state[t, 3] = state[t-1, 3] * (1 - friction * dt) + g * dt
            state[t, 6] = state[t-1, 6] * (1 - friction * dt)
            state[t, 7] = state[t-1, 7] * (1 - friction * dt) + g * dt
            
            # Boundaries
            for idx in [0, 1, 4, 5]:
                state[t, idx] = np.clip(state[t, idx], 10, 118)
            
            # Bounces
            if state[t, 0] <= 10 or state[t, 0] >= 118:
                state[t, 2] *= -0.8
            if state[t, 1] <= 10 or state[t, 1] >= 118:
                state[t, 3] *= -0.8
            if state[t, 4] <= 10 or state[t, 4] >= 118:
                state[t, 6] *= -0.8
            if state[t, 5] <= 10 or state[t, 5] >= 118:
                state[t, 7] *= -0.8
        
        trajectories.append(state)
    
    return {
        'trajectories': np.array(trajectories),
        'metadata': {
            'physics_type': 'constant_gravity',
            'gravity': g,
            'n_samples': n_samples,
            'timesteps': timesteps,
            'dt': dt
        }
    }


def main():
    """Generate minimal datasets for testing."""
    print("Generating true OOD physics data (minimal version)...")
    
    # Setup
    output_dir = get_data_path() / "true_ood_physics"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Generate constant gravity baseline
    print("\n1. Generating constant gravity data...")
    const_data = generate_constant_gravity_simple(n_samples=100)
    const_path = output_dir / f"constant_gravity_{timestamp}.pkl"
    with open(const_path, 'wb') as f:
        pickle.dump(const_data, f)
    print(f"   Saved to: {const_path}")
    print(f"   Shape: {const_data['trajectories'].shape}")
    
    # Generate time-varying gravity
    print("\n2. Generating time-varying gravity data...")
    varying_data = generate_time_varying_gravity_simple(n_samples=100)
    varying_path = output_dir / f"time_varying_gravity_{timestamp}.pkl"
    with open(varying_path, 'wb') as f:
        pickle.dump(varying_data, f)
    print(f"   Saved to: {varying_path}")
    print(f"   Shape: {varying_data['trajectories'].shape}")
    
    # Quick statistics
    print("\n3. Data Statistics:")
    print(f"   Constant gravity range: y ∈ [{const_data['trajectories'][:,:,[1,5]].min():.1f}, {const_data['trajectories'][:,:,[1,5]].max():.1f}]")
    print(f"   Time-varying range: y ∈ [{varying_data['trajectories'][:,:,[1,5]].min():.1f}, {varying_data['trajectories'][:,:,[1,5]].max():.1f}]")
    
    # Save summary
    summary = {
        'timestamp': timestamp,
        'constant_gravity_path': str(const_path),
        'time_varying_gravity_path': str(varying_path),
        'n_samples': 100,
        'timesteps': 50,
        'features': 8,
        'description': 'Minimal true OOD dataset with time-varying gravity'
    }
    
    summary_path = output_dir / f"minimal_summary_{timestamp}.pkl"
    with open(summary_path, 'wb') as f:
        pickle.dump(summary, f)
    
    print(f"\n✓ Data generation complete!")
    print(f"  Output directory: {output_dir}")
    
    return const_data, varying_data


if __name__ == "__main__":
    config = setup_environment()
    const_data, varying_data = main()