#!/usr/bin/env python3
"""
Complete pipeline for Paperspace: Generate data and train
This ensures all data is created on the GPU instance before training.
"""

import os
import sys
from pathlib import Path

# Set backend before any keras imports
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def setup_environment():
    """Setup the environment and paths"""
    print("Setting up environment...")
    
    # Detect base path
    if os.path.exists('/notebooks/neural_networks_research'):
        base_path = '/notebooks/neural_networks_research'
    elif os.path.exists('/workspace/neural_networks_research'):
        base_path = '/workspace/neural_networks_research'
    else:
        base_path = os.path.abspath('../..')
    
    # Change to experiment directory
    exp_dir = os.path.join(base_path, 'experiments/02_compositional_language')
    os.chdir(exp_dir)
    
    print(f"Working directory: {os.getcwd()}")
    return exp_dir

def generate_data():
    """Generate SCAN data and modifications"""
    print("\n" + "="*60)
    print("STEP 1: Generating SCAN Dataset")
    print("="*60)
    
    # Check if data already exists
    data_path = Path("data/processed/train.pkl")
    if data_path.exists():
        print("Processed data already exists. Skipping generation.")
        return True
    
    print("Generating SCAN data...")
    
    # Import and run data loader
    from scan_data_loader import SCANDataLoader
    
    loader = SCANDataLoader(data_dir='data')
    
    # Load all data
    print("Downloading SCAN dataset...")
    all_data = loader.load_all_data()
    
    # Create isolated splits
    print("Creating train/test splits with proper isolation...")
    splits = loader.create_isolated_splits()
    
    # Save processed data
    print("Saving processed data...")
    loader.save_processed_data(splits)
    
    # Generate modifications
    print("\nGenerating modification pairs...")
    from modification_generator import ModificationGenerator
    
    generator = ModificationGenerator()
    
    # Load the processed splits
    processed_splits = loader.load_processed_splits()
    
    # Convert back to SCANSample objects for modification generation
    from scan_data_loader import SCANSample
    train_samples = []
    for data in processed_splits['train'][:5000]:  # Use subset for faster generation
        sample = SCANSample(
            command=data['command'],
            action=data['action'],
            primitives=set(data['primitives']),
            modifiers=set(data['modifiers']),
            connectors=set(data['connectors']),
            length=data['length']
        )
        train_samples.append(sample)
    
    # Generate modifications
    modifications = generator.generate_all_modifications(train_samples)
    
    # Save modifications
    generator.save_modifications(modifications)
    
    print("Data generation complete!")
    return True

def train_model():
    """Run the progressive curriculum training with optimizations"""
    print("\n" + "="*60)
    print("STEP 2: Running Optimized Progressive Curriculum Training")
    print("="*60)
    
    # Check if optimized version exists, otherwise fall back to original
    try:
        from train_progressive_optimized import train_progressive_curriculum_optimized
        print("Using memory-optimized training...")
        
        # Optimized configuration
        config = {
            # Model parameters
            'd_model': 128,
            'batch_size': 8,
            'gradient_accumulation_steps': 2,  # Effective batch size of 16
            
            # Training epochs
            'stage1_epochs': 20,
            'stage2_epochs': 20,
            'stage3_epochs': 20,
            'stage4_epochs': 20,
            
            # Learning rates
            'stage1_lr': 1e-3,
            'stage2_lr': 5e-4,
            'stage3_lr': 2e-4,
            'stage4_lr': 1e-4,
            
            # Output and logging
            'output_dir': 'outputs/optimized_training',
            'use_wandb': True,
            'wandb_project': 'compositional-language-optimized'
        }
        
        # Run optimized training
        train_progressive_curriculum_optimized(config)
        
    except ImportError:
        print("Optimized version not found, using standard training...")
        from train_progressive_curriculum import train_progressive_curriculum
        
        # Standard configuration
        config = {
            'd_model': 128,
            'batch_size': 8,
            'stage1_epochs': 20,
            'stage2_epochs': 20,
            'stage3_epochs': 20,
            'stage4_epochs': 20,
            'stage1_lr': 1e-3,
            'stage2_lr': 5e-4,
            'stage3_lr': 2e-4,
            'stage4_lr': 1e-4,
            'output_dir': 'outputs/full_training',
            'use_wandb': True,
            'wandb_project': 'compositional-language-invention'
        }
        
        # Run standard training
        train_progressive_curriculum(config)
    
    return True

def save_final_results():
    """Save results before shutdown"""
    print("\n" + "="*60)
    print("STEP 3: Saving Results")
    print("="*60)
    
    try:
        from save_results import save_critical_results
        results_dir = save_critical_results()
        return results_dir
    except Exception as e:
        print(f"Error saving results: {e}")
        return None

def main():
    """Main pipeline"""
    print("Compositional Language Progressive Curriculum Pipeline")
    print("=" * 60)
    
    # Setup
    exp_dir = setup_environment()
    
    # Generate data
    if not generate_data():
        print("Data generation failed!")
        return 1
    
    # Train model
    if not train_model():
        print("Training failed!")
        return 1
    
    # Save results
    results_dir = save_final_results()
    
    print("\n" + "="*60)
    print("Pipeline completed successfully!")
    if results_dir:
        print(f"Results saved in: {results_dir}")
        print(f"Download with: zip -r results.zip {results_dir}/")
    print("="*60)
    
    return 0

if __name__ == "__main__":
    # Check for GPU
    import tensorflow as tf
    print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")
    
    sys.exit(main())