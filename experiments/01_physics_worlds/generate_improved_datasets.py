#!/usr/bin/env python3
"""
Generate Improved Physics Datasets with Proper Train/Test Isolation

This script implements the data isolation fix plan by generating datasets with:
- Proper train/validation/test splits
- No parameter overlap between train and test
- Systematic coverage for interpolation testing
- Extrapolation ranges for true generalization testing
- Novel physics regimes for distribution invention testing

Run with: python generate_improved_datasets.py [--quick]
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from improved_data_generator import (
    ImprovedDataConfig,
    ImprovedPhysicsDataGenerator,
    generate_improved_physics_datasets,
)


def verify_environment():
    """Verify that the environment is ready for improved data generation"""
    print("🔍 Verifying environment for improved data generation...")

    try:
        # Test imports
        from physics_env import Ball, PhysicsConfig, PhysicsWorld

        print("✅ All required packages available")

        # Test physics simulation
        config = PhysicsConfig()
        world = PhysicsWorld(config)
        world.add_ball(Ball(x=100, y=100))
        world.step()
        print("✅ Physics simulation working")

        # Check output directory
        output_dir = Path("data/processed/physics_worlds_v2")
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"✅ Output directory ready: {output_dir}")

        return True

    except Exception as e:
        print(f"❌ Environment check failed: {e}")
        print("Please ensure all dependencies are installed:")
        print("   pip install pygame pymunk numpy matplotlib tqdm scikit-learn scipy")
        return False


def generate_quick_improved_datasets():
    """Generate smaller datasets for quick testing of improved approach"""
    print("⚡ QUICK MODE: Generating smaller improved datasets for testing")

    config = ImprovedDataConfig(
        total_samples=500,  # Much smaller for quick testing
        sequence_length=150,  # Shorter sequences
        output_dir="data/processed/physics_worlds_v2_quick",
        save_visualizations=False,
    )

    generator = ImprovedPhysicsDataGenerator(config)
    datasets = generator.generate_all_datasets()

    return datasets


def analyze_improved_datasets(datasets_path: Dict[str, str]):
    """Analyze the improved datasets to verify proper isolation"""
    print("\n📊 ANALYZING IMPROVED DATASETS")
    print("=" * 50)

    import pickle

    import numpy as np

    # Load and analyze each dataset
    dataset_stats = {}

    for split_name, file_path in datasets_path.items():
        if not Path(file_path).exists():
            continue

        print(f"\nAnalyzing {split_name}...")

        with open(file_path, "rb") as f:
            data = pickle.load(f)

        # Extract physics parameters
        params = {
            "gravity": [sample["physics_config"]["gravity"] for sample in data],
            "friction": [sample["physics_config"]["friction"] for sample in data],
            "elasticity": [sample["physics_config"]["elasticity"] for sample in data],
            "damping": [sample["physics_config"]["damping"] for sample in data],
        }

        # Calculate statistics
        stats = {}
        for param_name, values in params.items():
            stats[param_name] = {
                "min": np.min(values),
                "max": np.max(values),
                "mean": np.mean(values),
                "std": np.std(values),
            }

        dataset_stats[split_name] = {"num_samples": len(data), "params": stats}

        # Print summary
        print(f"  Samples: {len(data)}")
        for param_name, param_stats in stats.items():
            print(
                f"  {param_name:>10}: [{param_stats['min']:>7.1f}, {param_stats['max']:>7.1f}] (μ={param_stats['mean']:.1f}, σ={param_stats['std']:.1f})"
            )

    # Verify no overlap between train and test sets
    print(f"\n🔍 VERIFYING TRAIN/TEST ISOLATION")
    print("-" * 40)

    if "train" in dataset_stats and "test_extrapolation" in dataset_stats:
        train_stats = dataset_stats["train"]["params"]
        test_stats = dataset_stats["test_extrapolation"]["params"]

        for param in ["gravity", "friction", "elasticity", "damping"]:
            train_range = [train_stats[param]["min"], train_stats[param]["max"]]
            test_range = [test_stats[param]["min"], test_stats[param]["max"]]

            # Check for overlap
            overlap_min = max(train_range[0], test_range[0])
            overlap_max = min(train_range[1], test_range[1])
            has_overlap = overlap_min < overlap_max

            if has_overlap:
                overlap_size = overlap_max - overlap_min
                test_size = test_range[1] - test_range[0]
                overlap_pct = (overlap_size / test_size) * 100
                status = f"⚠️  {overlap_pct:.1f}% overlap"
            else:
                if test_range[1] < train_range[0]:
                    status = "✅ Extrapolating BELOW training range"
                elif test_range[0] > train_range[1]:
                    status = "✅ Extrapolating ABOVE training range"
                else:
                    status = "✅ Complex extrapolation pattern"

            print(f"  {param:>10}: Train{train_range} vs Test{test_range} - {status}")

    return dataset_stats


def compare_old_vs_new_approach():
    """Compare parameter distributions between old and new approaches"""
    print(f"\n🔄 COMPARING OLD VS NEW APPROACH")
    print("-" * 40)

    # Check if old data exists
    old_train_path = Path("data/processed/physics_worlds/train_data.pkl")
    new_train_path = Path("data/processed/physics_worlds_v2/train_data.pkl")

    if old_train_path.exists() and new_train_path.exists():
        import pickle

        import numpy as np

        # Load old data
        with open(old_train_path, "rb") as f:
            old_data = pickle.load(f)[:1000]  # Sample for comparison

        # Load new data
        with open(new_train_path, "rb") as f:
            new_data = pickle.load(f)[:1000]

        print(
            f"Comparing {len(old_data)} old samples vs {len(new_data)} new samples..."
        )

        for param in ["gravity", "friction", "elasticity", "damping"]:
            old_values = [sample["physics_config"][param] for sample in old_data]
            new_values = [sample["physics_config"][param] for sample in new_data]

            old_range = [np.min(old_values), np.max(old_values)]
            new_range = [np.min(new_values), np.max(new_values)]

            print(f"  {param:>10}: Old{old_range} vs New{new_range}")
    else:
        print("  Old datasets not found - cannot compare")


def main():
    parser = argparse.ArgumentParser(
        description="Generate improved Physics Worlds datasets with proper isolation"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Generate smaller datasets for quick testing",
    )
    parser.add_argument(
        "--skip-verify", action="store_true", help="Skip environment verification"
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only analyze existing datasets without generating new ones",
    )

    args = parser.parse_args()

    # Verify environment unless skipped
    if not args.skip_verify and not args.analyze_only:
        if not verify_environment():
            return 1
        print()

    # Generate datasets unless analyze-only
    if not args.analyze_only:
        try:
            start_time = time.time()

            if args.quick:
                datasets = generate_quick_improved_datasets()
            else:
                datasets = generate_improved_physics_datasets()

            # Calculate total time
            total_time = time.time() - start_time
            hours = int(total_time // 3600)
            minutes = int((total_time % 3600) // 60)
            seconds = int(total_time % 60)

            print(
                f"\n⏱️  Generation completed in: {hours:02d}h {minutes:02d}m {seconds:02d}s"
            )

            # Analyze the generated datasets
            analyze_improved_datasets(datasets)

            # Compare with old approach if available
            compare_old_vs_new_approach()

            # Save generation summary
            summary_path = Path(
                f"data/processed/physics_worlds_v2{'_quick' if args.quick else ''}/generation_summary.txt"
            )
            with open(summary_path, "w") as f:
                f.write(f"Improved Physics Worlds Data Generation Summary\n")
                f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Mode: {'Quick' if args.quick else 'Full'}\n")
                f.write(
                    f"Generation time: {hours:02d}h {minutes:02d}m {seconds:02d}s\n"
                )
                f.write(f"Files:\n")
                for split, path in datasets.items():
                    if Path(path).exists():
                        size_mb = Path(path).stat().st_size / (1024 * 1024)
                        f.write(f"  {split}: {Path(path).name} ({size_mb:.1f} MB)\n")

            print(f"\n📝 Generation summary saved to: {summary_path}")

            print(f"\n✅ IMPROVED DATA GENERATION COMPLETE!")
            print(f"Next steps:")
            print(f"1. Update training scripts to use new data splits")
            print(f"2. Implement new evaluation metrics")
            print(f"3. Re-run experiments with proper isolation")

            return 0

        except KeyboardInterrupt:
            print("\n⏹️  Generation cancelled by user")
            return 1
        except Exception as e:
            print(f"\n💥 Unexpected error: {e}")
            import traceback

            traceback.print_exc()
            return 1

    else:
        # Analyze existing datasets
        datasets_dir = Path("data/processed/physics_worlds_v2")
        if datasets_dir.exists():
            datasets = {}
            for split in [
                "train",
                "val_in_dist",
                "val_near_dist",
                "test_interpolation",
                "test_extrapolation",
                "test_novel",
            ]:
                data_file = datasets_dir / f"{split}_data.pkl"
                if data_file.exists():
                    datasets[split] = str(data_file)

            if datasets:
                analyze_improved_datasets(datasets)
                compare_old_vs_new_approach()
            else:
                print("No improved datasets found to analyze")
        else:
            print("Improved datasets directory not found")

        return 0


if __name__ == "__main__":
    exit(main())
