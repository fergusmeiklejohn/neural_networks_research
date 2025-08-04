#!/usr/bin/env python
"""
Quick start script for Physics Worlds Experiment 1
Demonstrates the complete pipeline from data generation to evaluation.
"""

import sys
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from data_generator import DataConfig, PhysicsDataGenerator
from physics_env import create_sample_scenario

from models.core.distribution_inventor import (
    DistributionInventorConfig,
    create_distribution_inventor,
)


def demo_physics_simulation():
    """Demonstrate the physics simulation environment"""
    print("=" * 60)
    print("DEMO 1: Physics Simulation Environment")
    print("=" * 60)

    # Create sample scenario
    world, balls = create_sample_scenario()

    print(f"Created physics world with {len(balls)} balls")
    print("Running simulation for 3 seconds...")

    # Run simulation
    for _ in range(180):  # 3 seconds at 60 FPS
        world.step()

    # Get metrics
    metrics = world.get_physics_metrics()
    print("Physics metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    # Test physics modification
    print("\nTesting physics modification (stronger gravity)...")
    world.reset()

    # Add balls again
    for ball in balls:
        world.add_ball(ball)

    # Modify physics
    world.modify_physics({"gravity": -1500})  # 50% stronger gravity

    # Run modified simulation
    for _ in range(180):
        world.step()

    modified_metrics = world.get_physics_metrics()
    print("Modified physics metrics:")
    for key, value in modified_metrics.items():
        print(f"  {key}: {value:.4f}")

    print("✓ Physics simulation demo complete!\n")


def demo_data_generation():
    """Demonstrate data generation pipeline"""
    print("=" * 60)
    print("DEMO 2: Data Generation Pipeline")
    print("=" * 60)

    # Create small test dataset
    config = DataConfig(
        num_samples=20,
        sequence_length=100,
        output_dir="data/processed/physics_worlds_demo",
        save_visualizations=False,
    )

    generator = PhysicsDataGenerator(config)

    print("Generating small test dataset...")
    dataset_file = generator.generate_dataset("demo")

    print(f"Generated dataset saved to: {dataset_file}")

    # Generate modification pairs
    print("Generating modification pairs...")
    mod_file = generator.generate_modification_pairs(5)

    print(f"Modification pairs saved to: {mod_file}")
    print("✓ Data generation demo complete!\n")


def demo_model_components():
    """Demonstrate individual model components"""
    print("=" * 60)
    print("DEMO 3: Model Components")
    print("=" * 60)

    # Test rule extractor
    print("Testing PhysicsRuleExtractor...")
    from models.core.physics_rule_extractor import create_physics_rule_extractor

    rule_extractor = create_physics_rule_extractor()

    # Create dummy trajectory data
    dummy_trajectory = np.random.random((1, 100, 9))
    rules = rule_extractor.extract_rules(dummy_trajectory)

    print("Extracted rules:")
    for rule, value in rules.items():
        if isinstance(value, np.ndarray):
            print(f"  {rule}: {value.shape} - {value.mean():.4f}")

    # Test distribution modifier
    print("\nTesting DistributionModifier...")
    from models.core.distribution_modifier import create_distribution_modifier

    modifier = create_distribution_modifier()

    test_rules = {
        "gravity": -981.0,
        "friction": 0.7,
        "elasticity": 0.8,
        "damping": 0.95,
    }

    modified = modifier.modify_distribution(test_rules, "increase gravity by 20%")

    print("Original vs Modified rules:")
    for rule in test_rules.keys():
        if rule in modified:
            orig_val = test_rules[rule]
            mod_val = (
                modified[rule].item()
                if isinstance(modified[rule], np.ndarray)
                else modified[rule]
            )
            change = (mod_val - orig_val) / orig_val * 100
            print(f"  {rule}: {orig_val:.2f} → {mod_val:.2f} ({change:+.1f}%)")

    # Test trajectory generator
    print("\nTesting TrajectoryGenerator...")
    from models.core.trajectory_generator import create_trajectory_generator

    generator = create_trajectory_generator()

    initial_conditions = np.random.random((2, 9))
    physics_rules = {
        "gravity": -981.0,
        "friction": 0.7,
        "elasticity": 0.8,
        "damping": 0.95,
    }

    generated = generator.generate_trajectory(
        initial_conditions, physics_rules, sequence_length=50
    )

    print("Generated trajectory:")
    for key, value in generated.items():
        print(f"  {key}: {value.shape}")

    print("✓ Model components demo complete!\n")


def demo_full_pipeline():
    """Demonstrate the complete distribution invention pipeline"""
    print("=" * 60)
    print("DEMO 4: Complete Distribution Invention Pipeline")
    print("=" * 60)

    # Create the full model
    print("Creating DistributionInventor model...")
    config = DistributionInventorConfig()
    model = create_distribution_inventor(config)

    print(f"Model created with {model.count_params():,} parameters")

    # Create test data
    print("\nTesting distribution invention...")

    # Base trajectory (from which to extract rules)
    base_trajectory = np.random.random((100, 9))

    # Initial conditions for new trajectory
    initial_conditions = np.random.random((2, 9))

    # Test different modification requests
    modification_requests = [
        "increase gravity by 20%",
        "reduce friction significantly",
        "make objects more bouncy",
        "remove air resistance",
    ]

    for request in modification_requests:
        print(f"\nTesting: '{request}'")

        try:
            result = model.invent_distribution(
                base_trajectory, request, initial_conditions, return_details=True
            )

            print(f"  Success: {result['success']}")
            print(f"  New trajectory shape: {result['new_trajectory'].shape}")

            if "quality_scores" in result:
                print("  Quality scores:")
                for metric, score in result["quality_scores"].items():
                    print(f"    {metric}: {score:.4f}")

            print("  Rule changes:")
            orig_rules = result.get("original_rules", {})
            mod_rules = result.get("modified_rules", {})

            for rule in ["gravity", "friction", "elasticity", "damping"]:
                if rule in orig_rules and rule in mod_rules:
                    orig_val = (
                        orig_rules[rule].item()
                        if isinstance(orig_rules[rule], np.ndarray)
                        else orig_rules[rule]
                    )
                    mod_val = (
                        mod_rules[rule].item()
                        if isinstance(mod_rules[rule], np.ndarray)
                        else mod_rules[rule]
                    )
                    change = (
                        (mod_val - orig_val) / abs(orig_val) * 100
                        if abs(orig_val) > 1e-6
                        else 0
                    )
                    print(
                        f"    {rule}: {orig_val:.3f} → {mod_val:.3f} ({change:+.1f}%)"
                    )

        except Exception as e:
            print(f"  Error: {e}")

    print("✓ Complete pipeline demo complete!\n")


def main():
    """Run all demonstrations"""
    print("🚀 PHYSICS WORLDS EXPERIMENT 1 - QUICK START DEMO")
    print("=" * 60)
    print("This script demonstrates the complete distribution invention pipeline")
    print("for the physics worlds experiment.\n")

    try:
        # Run demonstrations
        demo_physics_simulation()
        demo_data_generation()
        demo_model_components()
        demo_full_pipeline()

        print("=" * 60)
        print("✅ ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nNext steps:")
        print(
            "1. Generate full dataset: python experiments/01_physics_worlds/data_generator.py"
        )
        print("2. Train model: python experiments/01_physics_worlds/train_physics.py")
        print(
            "3. Evaluate model: python experiments/01_physics_worlds/evaluate_physics.py"
        )
        print("\nFor detailed training:")
        print(
            "python experiments/01_physics_worlds/train_physics.py --epochs 50 --batch_size 16"
        )

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\nMake sure you've activated the conda environment:")
        print("conda activate dist-invention")
        print("\nAnd installed all dependencies.")

    except Exception as e:
        print(f"❌ Error during demo: {e}")
        print("\nThis might be expected for the model components demo,")
        print("as the models need proper training data to work effectively.")


if __name__ == "__main__":
    main()
