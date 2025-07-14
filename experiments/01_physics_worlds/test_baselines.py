"""
Quick test of baseline training setup
"""

import os
os.environ['KERAS_BACKEND'] = 'jax'

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from train_baselines import PhysicsBaselineTrainer

def test_baseline_setup():
    """Test that baselines can be initialized and data loaded."""
    print("Testing baseline training setup...")
    
    # Initialize trainer
    trainer = PhysicsBaselineTrainer()
    
    # Check data shapes
    print(f"\nData shapes:")
    print(f"Train X: {trainer.train_data['x'].shape}")
    print(f"Train Y: {trainer.train_data['y'].shape}")
    print(f"Test categories: {trainer.test_data['categories'] is not None}")
    
    # Test config creation
    configs = trainer.create_baseline_configs()
    print(f"\nCreated configs for: {list(configs.keys())}")
    
    # Try initializing one baseline
    from models.baseline_models import ERMWithAugmentation
    
    print("\nTesting ERM baseline initialization...")
    erm = ERMWithAugmentation(configs['erm'])
    erm.build_model()
    print(f"Model built successfully! Parameters: {erm.model.count_params():,}")
    
    print("\nSetup test passed! ✓")

if __name__ == "__main__":
    test_baseline_setup()