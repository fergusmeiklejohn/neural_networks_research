#!/usr/bin/env python3
"""
Phase 1 Data Generation Script for Physics Worlds Experiment

This script generates the complete dataset required for Phase 1:
- Training data: 10,000 samples
- Validation data: 2,000 samples 
- Test data: 1,000 samples
- Modification pairs: ~9,000 pairs (1,000 base × 9 modifications)

Run with: python generate_full_dataset.py [--quick]
"""

import argparse
import time
from pathlib import Path
from data_generator import generate_physics_datasets, DataConfig, PhysicsDataGenerator

def generate_experiment_datasets(quick_mode: bool = False):
    """Generate complete datasets for Phase 1 of the physics experiment"""
    
    print("🚀 PHYSICS WORLDS EXPERIMENT 1 - PHASE 1 DATA GENERATION")
    print("=" * 60)
    print("Generating complete datasets for distribution invention training...")
    
    if quick_mode:
        print("⚡ QUICK MODE: Generating smaller datasets for testing")
        print("   - Training: 100 samples (instead of 10,000)")
        print("   - Validation: 50 samples (instead of 2,000)")
        print("   - Test: 25 samples (instead of 1,000)")
        print("   - Modification pairs: ~225 pairs")
    else:
        print("📊 FULL MODE: Generating complete experiment datasets")
        print("   - Training: 10,000 samples")
        print("   - Validation: 2,000 samples")
        print("   - Test: 1,000 samples")
        print("   - Modification pairs: ~9,000 pairs")
    
    print()
    start_time = time.time()
    
    try:
        if quick_mode:
            # Quick mode for testing
            datasets = generate_quick_datasets()
        else:
            # Full dataset generation
            datasets = generate_physics_datasets()
        
        # Calculate total time
        total_time = time.time() - start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        
        print("\n🎉 DATA GENERATION COMPLETE!")
        print("=" * 40)
        print(f"⏱️  Total time: {hours:02d}h {minutes:02d}m {seconds:02d}s")
        print(f"📁 Output directory: data/processed/physics_worlds")
        print()
        print("Generated files:")
        for split, path in datasets.items():
            if Path(path).exists():
                size_mb = Path(path).stat().st_size / (1024 * 1024)
                print(f"   {split:>13}: {Path(path).name} ({size_mb:.1f} MB)")
        
        print()
        print("✅ Ready for Phase 2: Model Training!")
        print("   Next steps:")
        print("   1. Update experiment plan with Phase 1 results")
        print("   2. Begin model component training")
        print("   3. Run quick_start.py to verify data loading")
        
        return datasets
        
    except Exception as e:
        print(f"\n❌ DATA GENERATION FAILED: {e}")
        print("Check the error above and retry with --quick for debugging")
        raise

def generate_quick_datasets():
    """Generate smaller datasets for quick testing"""
    
    # Training data (reduced)
    print("=== Generating Quick Training Data ===")
    train_config = DataConfig(
        num_samples=100,
        sequence_length=200,  # Keep full sequence length
        output_dir="data/processed/physics_worlds",
        save_visualizations=False
    )
    train_generator = PhysicsDataGenerator(train_config)
    train_file = train_generator.generate_dataset("train")
    
    # Validation data (reduced)
    print("\n=== Generating Quick Validation Data ===")
    val_config = DataConfig(
        num_samples=50,
        sequence_length=200,
        output_dir="data/processed/physics_worlds"
    )
    val_generator = PhysicsDataGenerator(val_config)
    val_file = val_generator.generate_dataset("val")
    
    # Test data (reduced)
    print("\n=== Generating Quick Test Data ===")
    test_config = DataConfig(
        num_samples=25,
        sequence_length=200,
        output_dir="data/processed/physics_worlds"
    )
    test_generator = PhysicsDataGenerator(test_config)
    test_file = test_generator.generate_dataset("test")
    
    # Modification pairs (reduced)
    print("\n=== Generating Quick Modification Pairs ===")
    mod_file = train_generator.generate_modification_pairs(25)  # 25 * 9 = 225 pairs
    
    return {
        'train': train_file,
        'val': val_file,
        'test': test_file,
        'modifications': mod_file
    }

def verify_generation_environment():
    """Verify that the environment is ready for data generation"""
    print("🔍 Verifying environment...")
    
    try:
        # Test imports
        import pygame
        import pymunk
        import numpy as np
        from physics_env import PhysicsWorld, PhysicsConfig, Ball
        print("✅ All required packages available")
        
        # Test physics simulation
        config = PhysicsConfig()
        world = PhysicsWorld(config)
        world.add_ball(Ball(x=100, y=100))
        world.step()
        print("✅ Physics simulation working")
        
        # Check output directory
        output_dir = Path("data/processed/physics_worlds")
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"✅ Output directory ready: {output_dir}")
        
        return True
        
    except Exception as e:
        print(f"❌ Environment check failed: {e}")
        print("Please ensure all dependencies are installed:")
        print("   pip install pygame pymunk numpy matplotlib tqdm")
        return False

def main():
    parser = argparse.ArgumentParser(description='Generate complete Physics Worlds dataset')
    parser.add_argument('--quick', action='store_true',
                       help='Generate smaller datasets for quick testing')
    parser.add_argument('--skip-verify', action='store_true',
                       help='Skip environment verification')
    
    args = parser.parse_args()
    
    # Verify environment unless skipped
    if not args.skip_verify:
        if not verify_generation_environment():
            return 1
        print()
    
    # Generate datasets
    try:
        datasets = generate_experiment_datasets(quick_mode=args.quick)
        
        # Save generation summary
        summary_path = Path("data/processed/physics_worlds/generation_summary.txt")
        with open(summary_path, 'w') as f:
            f.write(f"Physics Worlds Data Generation Summary\n")
            f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Mode: {'Quick' if args.quick else 'Full'}\n")
            f.write(f"Files:\n")
            for split, path in datasets.items():
                if Path(path).exists():
                    size_mb = Path(path).stat().st_size / (1024 * 1024)
                    f.write(f"  {split}: {Path(path).name} ({size_mb:.1f} MB)\n")
        
        print(f"📝 Generation summary saved to: {summary_path}")
        return 0
        
    except KeyboardInterrupt:
        print("\n⏹️  Generation cancelled by user")
        return 1
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())