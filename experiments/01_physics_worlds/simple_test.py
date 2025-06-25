#!/usr/bin/env python
"""
Simple test script for Physics Worlds Experiment 1
Tests the basic components without complex imports.
"""

import sys
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

def test_physics_environment():
    """Test the physics simulation environment"""
    print("=" * 50)
    print("Testing Physics Environment")
    print("=" * 50)
    
    try:
        from physics_env import PhysicsWorld, PhysicsConfig, Ball
        
        # Create physics world
        config = PhysicsConfig()
        world = PhysicsWorld(config)
        
        # Add a ball
        ball = Ball(x=100, y=500, vx=150, vy=0, radius=15)
        world.add_ball(ball)
        
        print(f"Created physics world with 1 ball")
        print("Running simulation for 60 steps...")
        
        # Run simulation
        for _ in range(60):
            world.step()
        
        print(f"Simulation complete. Generated {len(world.trajectory_data)} frames")
        
        # Test physics modification
        world.modify_physics({'gravity': -1500})
        print("Modified gravity to -1500 (stronger)")
        
        # Run more steps
        for _ in range(60):
            world.step()
        
        print(f"Total frames after modification: {len(world.trajectory_data)}")
        print("✓ Physics environment test passed!")
        
    except Exception as e:
        print(f"❌ Physics environment test failed: {e}")
        return False
    
    return True


def test_data_generation():
    """Test the data generation pipeline"""
    print("\n" + "=" * 50)
    print("Testing Data Generation")
    print("=" * 50)
    
    try:
        from data_generator import PhysicsDataGenerator, DataConfig
        
        # Create small test dataset
        config = DataConfig(
            num_samples=5,
            sequence_length=50,
            output_dir="data/processed/physics_test"
        )
        
        generator = PhysicsDataGenerator(config)
        print("Created data generator")
        
        # Generate a single sample
        sample = generator._generate_single_sample(0)
        print(f"Generated sample with {len(sample['trajectory'])} trajectory points")
        print(f"Sample has {sample['num_balls']} balls")
        
        physics_config = sample['physics_config']
        print("Physics parameters:")
        print(f"  Gravity: {physics_config['gravity']:.1f}")
        print(f"  Friction: {physics_config['friction']:.3f}")
        print(f"  Elasticity: {physics_config['elasticity']:.3f}")
        print(f"  Damping: {physics_config['damping']:.3f}")
        
        print("✓ Data generation test passed!")
        
    except Exception as e:
        print(f"❌ Data generation test failed: {e}")
        return False
    
    return True


def test_model_creation():
    """Test model creation (without training)"""
    print("\n" + "=" * 50)
    print("Testing Model Creation")
    print("=" * 50)
    
    try:
        # Test individual components
        print("Testing PhysicsRuleExtractor...")
        sys.path.append(str(project_root / "models" / "core"))
        
        from physics_rule_extractor import PhysicsRuleConfig, PhysicsRuleExtractor
        
        config = PhysicsRuleConfig()
        rule_extractor = PhysicsRuleExtractor(config)
        
        # Test with dummy data
        dummy_input = np.random.random((2, config.sequence_length, config.feature_dim))
        outputs = rule_extractor(dummy_input)
        
        print(f"Rule extractor output shapes:")
        for key, value in outputs.items():
            if hasattr(value, 'shape'):
                print(f"  {key}: {value.shape}")
        
        print("✓ PhysicsRuleExtractor created successfully!")
        
        # Test distribution modifier
        print("\nTesting DistributionModifier...")
        from distribution_modifier import ModifierConfig, DistributionModifier
        
        mod_config = ModifierConfig()
        modifier = DistributionModifier(mod_config)
        
        # Test with dummy data
        base_rules = {
            'gravity': np.random.normal(-981, 100, (2, 1)),
            'friction': np.random.uniform(0.3, 0.9, (2, 1)),
            'elasticity': np.random.uniform(0.5, 0.9, (2, 1)),
            'damping': np.random.uniform(0.9, 0.98, (2, 1))
        }
        modification_text = np.random.randint(0, 100, (2, 20))
        
        inputs = {
            'base_rules': base_rules,
            'modification_text': modification_text
        }
        
        mod_outputs = modifier(inputs)
        print(f"Distribution modifier output shapes:")
        for key, value in mod_outputs.items():
            if isinstance(value, dict):
                print(f"  {key}: {type(value)} with {len(value)} items")
            elif hasattr(value, 'shape'):
                print(f"  {key}: {value.shape}")
        
        print("✓ DistributionModifier created successfully!")
        
        # Test trajectory generator
        print("\nTesting TrajectoryGenerator...")
        from trajectory_generator import TrajectoryConfig, TrajectoryGenerator
        
        traj_config = TrajectoryConfig()
        generator = TrajectoryGenerator(traj_config)
        
        # Test with dummy data
        initial_conditions = np.random.random((2, 3, traj_config.feature_dim))
        physics_rules = base_rules
        
        gen_inputs = {
            'initial_conditions': initial_conditions,
            'physics_rules': physics_rules,
            'sequence_length': 20
        }
        
        gen_outputs = generator(gen_inputs)
        print(f"Trajectory generator output shapes:")
        for key, value in gen_outputs.items():
            if hasattr(value, 'shape'):
                print(f"  {key}: {value.shape}")
        
        print("✓ TrajectoryGenerator created successfully!")
        print("✓ All model components test passed!")
        
    except Exception as e:
        print(f"❌ Model creation test failed: {e}")
        import traceback
        print(traceback.format_exc())
        return False
    
    return True


def main():
    """Run all tests"""
    print("🚀 PHYSICS WORLDS EXPERIMENT 1 - SIMPLE TEST")
    print("=" * 60)
    
    all_passed = True
    
    # Test physics environment
    if not test_physics_environment():
        all_passed = False
    
    # Test data generation
    if not test_data_generation():
        all_passed = False
    
    # Test model creation
    if not test_model_creation():
        all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL TESTS PASSED!")
        print("The physics worlds experiment components are working correctly.")
        print("\nNext steps:")
        print("1. Generate full dataset: python data_generator.py")
        print("2. Train model: python train_physics.py --epochs 10 --batch_size 4")
    else:
        print("❌ SOME TESTS FAILED!")
        print("Check the error messages above and ensure all dependencies are installed.")
    
    print("=" * 60)


if __name__ == "__main__":
    main()