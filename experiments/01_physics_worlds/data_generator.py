"""
Data generation pipeline for physics world trajectories.
Generates training data with various physics configurations.
"""

import numpy as np
import pickle
import json
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass, asdict
import random
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

from physics_env import PhysicsWorld, PhysicsConfig, Ball


@dataclass 
class DataConfig:
    """Configuration for data generation"""
    num_samples: int = 10000
    sequence_length: int = 300  # Number of time steps (5 seconds at 60fps)
    min_balls: int = 1
    max_balls: int = 4
    
    # Physics variation ranges
    gravity_range: Tuple[float, float] = (-1500, -500)  # Normal: -981
    friction_range: Tuple[float, float] = (0.1, 0.9)    # Normal: 0.7
    elasticity_range: Tuple[float, float] = (0.3, 0.95) # Normal: 0.8
    damping_range: Tuple[float, float] = (0.85, 0.99)   # Normal: 0.95
    
    # Ball property ranges
    ball_radius_range: Tuple[float, float] = (10, 30)
    ball_mass_range: Tuple[float, float] = (0.5, 2.0)
    ball_velocity_range: Tuple[float, float] = (-200, 200)
    
    # World dimensions
    world_width: int = 800
    world_height: int = 600
    
    # Output settings
    output_dir: str = "data/processed/physics_worlds"
    save_visualizations: bool = False


class PhysicsDataGenerator:
    """Generate physics simulation data for training"""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def _generate_random_physics_config(self) -> PhysicsConfig:
        """Generate random physics configuration within ranges"""
        return PhysicsConfig(
            gravity=random.uniform(*self.config.gravity_range),
            friction=random.uniform(*self.config.friction_range),
            elasticity=random.uniform(*self.config.elasticity_range),
            damping=random.uniform(*self.config.damping_range),
            world_width=self.config.world_width,
            world_height=self.config.world_height
        )
    
    def _generate_random_balls(self) -> List[Ball]:
        """Generate random ball configurations"""
        num_balls = random.randint(self.config.min_balls, self.config.max_balls)
        balls = []
        
        for _ in range(num_balls):
            # Ensure balls don't start too close to edges or each other
            x = random.uniform(50, self.config.world_width - 50)
            y = random.uniform(100, self.config.world_height - 50)
            
            ball = Ball(
                x=x,
                y=y,
                vx=random.uniform(*self.config.ball_velocity_range),
                vy=random.uniform(*self.config.ball_velocity_range),
                radius=random.uniform(*self.config.ball_radius_range),
                mass=random.uniform(*self.config.ball_mass_range)
            )
            balls.append(ball)
        
        return balls
    
    def _generate_single_sample(self, sample_id: int) -> Dict[str, Any]:
        """Generate a single physics simulation sample"""
        # Create random configuration
        physics_config = self._generate_random_physics_config()
        balls = self._generate_random_balls()
        
        # Create world and add balls
        world = PhysicsWorld(physics_config)
        for ball in balls:
            world.add_ball(ball)
        
        # Run simulation
        for _ in range(self.config.sequence_length):
            world.step()
        
        # Extract data
        trajectory_array = world.get_trajectory_array()
        metrics = world.get_physics_metrics()
        
        # Create sample data structure
        sample_data = {
            'sample_id': sample_id,
            'physics_config': asdict(physics_config),
            'initial_balls': [asdict(ball) for ball in balls],
            'trajectory': trajectory_array.tolist(),
            'metrics': metrics,
            'num_balls': len(balls),
            'sequence_length': len(world.trajectory_data)
        }
        
        # Save visualization if requested
        if self.config.save_visualizations and sample_id % 100 == 0:
            viz_path = self.output_dir / f"visualization_{sample_id:06d}.png"
            world.visualize_trajectory(str(viz_path))
        
        return sample_data
    
    def generate_dataset(self, split: str = "train") -> str:
        """Generate a complete dataset"""
        print(f"Generating {self.config.num_samples} {split} samples...")
        
        samples = []
        failed_samples = 0
        
        for sample_id in tqdm(range(self.config.num_samples), desc=f"Generating {split} data"):
            try:
                sample = self._generate_single_sample(sample_id)
                samples.append(sample)
            except Exception as e:
                print(f"Failed to generate sample {sample_id}: {e}")
                failed_samples += 1
                continue
        
        print(f"Successfully generated {len(samples)} samples ({failed_samples} failed)")
        
        # Save dataset
        output_file = self.output_dir / f"{split}_data.pkl"
        with open(output_file, 'wb') as f:
            pickle.dump(samples, f)
        
        # Save metadata
        metadata = {
            'config': asdict(self.config),
            'split': split,
            'num_samples': len(samples),
            'failed_samples': failed_samples,
            'file_path': str(output_file)
        }
        
        metadata_file = self.output_dir / f"{split}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Dataset saved to {output_file}")
        print(f"Metadata saved to {metadata_file}")
        
        return str(output_file)
    
    def generate_modification_pairs(self, base_samples: int = 1000) -> str:
        """Generate paired samples for rule modification training"""
        print(f"Generating {base_samples} modification pairs...")
        
        modification_data = []
        
        for sample_id in tqdm(range(base_samples), desc="Generating modification pairs"):
            try:
                # Generate base configuration
                base_config = self._generate_random_physics_config()
                balls = self._generate_random_balls()
                
                # Run base simulation
                base_world = PhysicsWorld(base_config)
                for ball in balls:
                    base_world.add_ball(ball)
                
                for _ in range(self.config.sequence_length):
                    base_world.step()
                
                base_trajectory = base_world.get_trajectory_array()
                base_metrics = base_world.get_physics_metrics()
                
                # Generate modified versions
                modifications = [
                    {'gravity': base_config.gravity * 1.2},  # 20% stronger gravity
                    {'gravity': base_config.gravity * 0.8},  # 20% weaker gravity
                    {'friction': 0.1},                       # Very low friction
                    {'friction': 0.9},                       # High friction
                    {'elasticity': 0.3},                     # Low bounce
                    {'elasticity': 0.95},                    # High bounce
                ]
                
                for mod_idx, modification in enumerate(modifications):
                    # Create modified world
                    modified_world = PhysicsWorld(base_config)
                    for ball in balls:
                        modified_world.add_ball(ball)
                    
                    # Apply modification
                    modified_world.modify_physics(modification)
                    
                    # Run modified simulation
                    for _ in range(self.config.sequence_length):
                        modified_world.step()
                    
                    modified_trajectory = modified_world.get_trajectory_array()
                    modified_metrics = modified_world.get_physics_metrics()
                    
                    # Create modification pair
                    pair_data = {
                        'pair_id': f"{sample_id}_{mod_idx}",
                        'base_config': asdict(base_config),
                        'modification': modification,
                        'initial_balls': [asdict(ball) for ball in balls],
                        'base_trajectory': base_trajectory.tolist(),
                        'modified_trajectory': modified_trajectory.tolist(),
                        'base_metrics': base_metrics,
                        'modified_metrics': modified_metrics,
                        'modification_type': list(modification.keys())[0],
                        'modification_value': list(modification.values())[0]
                    }
                    
                    modification_data.append(pair_data)
                
            except Exception as e:
                print(f"Failed to generate modification pair {sample_id}: {e}")
                continue
        
        # Save modification dataset
        output_file = self.output_dir / "modification_pairs.pkl"
        with open(output_file, 'wb') as f:
            pickle.dump(modification_data, f)
        
        print(f"Generated {len(modification_data)} modification pairs")
        print(f"Modification data saved to {output_file}")
        
        return str(output_file)


def generate_physics_datasets():
    """Generate all physics datasets for training"""
    
    # Base configuration
    config = DataConfig(
        num_samples=10000,
        output_dir="data/processed/physics_worlds"
    )
    
    generator = PhysicsDataGenerator(config)
    
    # Generate training data
    print("=== Generating Training Data ===")
    train_file = generator.generate_dataset("train")
    
    # Generate validation data
    print("\n=== Generating Validation Data ===")
    val_config = DataConfig(
        num_samples=2000,
        output_dir="data/processed/physics_worlds"
    )
    val_generator = PhysicsDataGenerator(val_config)
    val_file = val_generator.generate_dataset("val")
    
    # Generate test data
    print("\n=== Generating Test Data ===")
    test_config = DataConfig(
        num_samples=1000,
        output_dir="data/processed/physics_worlds"
    )
    test_generator = PhysicsDataGenerator(test_config)
    test_file = test_generator.generate_dataset("test")
    
    # Generate modification pairs
    print("\n=== Generating Modification Pairs ===")
    mod_file = generator.generate_modification_pairs(1000)
    
    print("\n=== Data Generation Complete ===")
    print(f"Training data: {train_file}")
    print(f"Validation data: {val_file}")
    print(f"Test data: {test_file}")
    print(f"Modification pairs: {mod_file}")
    
    return {
        'train': train_file,
        'val': val_file,
        'test': test_file,
        'modifications': mod_file
    }


def load_physics_dataset(file_path: str) -> List[Dict[str, Any]]:
    """Load a physics dataset from file"""
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def analyze_dataset(file_path: str):
    """Analyze dataset statistics"""
    data = load_physics_dataset(file_path)
    
    print(f"Dataset: {file_path}")
    print(f"Number of samples: {len(data)}")
    
    if not data:
        return
    
    # Analyze physics parameters
    gravities = [sample['physics_config']['gravity'] for sample in data]
    frictions = [sample['physics_config']['friction'] for sample in data]
    elasticities = [sample['physics_config']['elasticity'] for sample in data]
    
    print(f"Gravity range: {min(gravities):.1f} to {max(gravities):.1f}")
    print(f"Friction range: {min(frictions):.3f} to {max(frictions):.3f}")
    print(f"Elasticity range: {min(elasticities):.3f} to {max(elasticities):.3f}")
    
    # Analyze trajectory lengths
    traj_lengths = [sample['sequence_length'] for sample in data]
    print(f"Trajectory lengths: {min(traj_lengths)} to {max(traj_lengths)} (avg: {np.mean(traj_lengths):.1f})")
    
    # Analyze ball counts
    ball_counts = [sample['num_balls'] for sample in data]
    print(f"Ball counts: {min(ball_counts)} to {max(ball_counts)} (avg: {np.mean(ball_counts):.1f})")


if __name__ == "__main__":
    # Test data generation
    print("Testing physics data generation...")
    
    # Generate small test dataset
    test_config = DataConfig(
        num_samples=10,
        output_dir="data/processed/physics_worlds_test",
        save_visualizations=True
    )
    
    generator = PhysicsDataGenerator(test_config)
    test_file = generator.generate_dataset("test")
    
    # Analyze the test dataset
    analyze_dataset(test_file)
    
    print("\nTest data generation complete!")
    
    # Generate full datasets (uncomment to run)
    # datasets = generate_physics_datasets()