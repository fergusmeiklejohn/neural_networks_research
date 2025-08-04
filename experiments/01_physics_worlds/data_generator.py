"""
Data generation pipeline for physics world trajectories.
Generates training data with various physics configurations.
"""

import json
import pickle
import random
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from physics_env import Ball, PhysicsConfig, PhysicsWorld
from tqdm import tqdm


@dataclass
class DataConfig:
    """Configuration for data generation"""

    num_samples: int = 10000
    sequence_length: int = 300  # Number of time steps (5 seconds at 60fps)
    min_balls: int = 1
    max_balls: int = 4

    # Physics variation ranges (enhanced for better coverage)
    gravity_range: Tuple[float, float] = (
        -1500,
        -200,
    )  # Wide range including low gravity
    friction_range: Tuple[float, float] = (
        0.05,
        0.95,
    )  # Near-frictionless to high friction
    elasticity_range: Tuple[float, float] = (
        0.1,
        0.99,
    )  # Completely inelastic to super bouncy
    damping_range: Tuple[float, float] = (
        0.8,
        0.99,
    )  # High air resistance to vacuum-like

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

    # Quality validation settings (relaxed for physics realism)
    energy_conservation_threshold: float = 0.25  # 25% variance allowed
    min_trajectory_length: int = 30  # Minimum valid trajectory length
    max_velocity_threshold: float = (
        1500  # Maximum reasonable velocity (increased for collisions)
    )
    max_jump_threshold: float = 150  # Maximum position jump per frame


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
            world_height=self.config.world_height,
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
                mass=random.uniform(*self.config.ball_mass_range),
            )
            balls.append(ball)

        return balls

    def _validate_sample_quality(self, sample_data: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate the quality of a generated sample"""
        try:
            trajectory = np.array(sample_data["trajectory"])
            metrics = sample_data["metrics"]

            # Check trajectory length
            if len(trajectory) < self.config.min_trajectory_length:
                return (
                    False,
                    f"Trajectory too short: {len(trajectory)} < {self.config.min_trajectory_length}",
                )

            # Check energy conservation
            energy_conservation = metrics.get("energy_conservation", 0)
            if abs(energy_conservation) > self.config.energy_conservation_threshold:
                return False, f"Poor energy conservation: {energy_conservation:.4f}"

            # Check for reasonable velocities (no physics explosions)
            # Trajectory is 2D: (time_steps, features_per_ball * num_balls)
            # Features per ball: [time, x, y, vx, vy, mass, radius, ke, pe]
            if len(trajectory) > 0 and trajectory.shape[-1] >= 5:
                # Extract velocities (features 3,4 for first ball, 12,13 for second ball, etc.)
                num_features_per_ball = 9
                num_balls = trajectory.shape[1] // num_features_per_ball

                velocities = []
                for ball_idx in range(num_balls):
                    start_idx = ball_idx * num_features_per_ball
                    if start_idx + 5 <= trajectory.shape[1]:
                        vx = trajectory[:, start_idx + 3]  # vx feature
                        vy = trajectory[:, start_idx + 4]  # vy feature
                        velocities.extend([vx, vy])

                if velocities:
                    max_velocity = np.max(np.abs(velocities))
                    if max_velocity > self.config.max_velocity_threshold:
                        return False, f"Unreasonable velocity: {max_velocity:.2f}"

            # Check for trajectory smoothness (no sudden jumps)
            if len(trajectory) > 1 and trajectory.shape[-1] >= 3:
                # Extract positions for all balls
                num_features_per_ball = 9
                num_balls = trajectory.shape[1] // num_features_per_ball

                max_jump = 0
                for ball_idx in range(num_balls):
                    start_idx = ball_idx * num_features_per_ball
                    if start_idx + 3 <= trajectory.shape[1]:
                        x_positions = trajectory[:, start_idx + 1]  # x feature
                        y_positions = trajectory[:, start_idx + 2]  # y feature

                        # Calculate position jumps
                        x_diffs = np.diff(x_positions)
                        y_diffs = np.diff(y_positions)
                        jumps = np.sqrt(x_diffs**2 + y_diffs**2)
                        max_jump = max(max_jump, np.max(jumps))

                if max_jump > self.config.max_jump_threshold:
                    return False, f"Trajectory too jumpy: {max_jump:.2f}"

            return True, "Valid"

        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def _generate_single_sample(self, sample_id: int) -> Dict[str, Any]:
        """Generate a single physics simulation sample with quality validation"""
        max_retries = 3

        for retry in range(max_retries):
            try:
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
                    "sample_id": sample_id,
                    "physics_config": asdict(physics_config),
                    "initial_balls": [asdict(ball) for ball in balls],
                    "trajectory": trajectory_array.tolist(),
                    "metrics": metrics,
                    "num_balls": len(balls),
                    "sequence_length": len(world.trajectory_data),
                }

                # Validate sample quality
                is_valid, validation_msg = self._validate_sample_quality(sample_data)
                if not is_valid:
                    if retry < max_retries - 1:
                        warnings.warn(
                            f"Sample {sample_id} retry {retry + 1}: {validation_msg}"
                        )
                        continue
                    else:
                        raise ValueError(
                            f"Failed to generate valid sample after {max_retries} retries: {validation_msg}"
                        )

                # Add quality validation info
                sample_data["quality_validation"] = {
                    "is_valid": is_valid,
                    "validation_message": validation_msg,
                    "retry_count": retry,
                }

                # Save visualization if requested
                if self.config.save_visualizations and sample_id % 100 == 0:
                    viz_path = self.output_dir / f"visualization_{sample_id:06d}.png"
                    world.visualize_trajectory(str(viz_path))

                return sample_data

            except Exception as e:
                if retry < max_retries - 1:
                    warnings.warn(f"Sample {sample_id} retry {retry + 1}: {str(e)}")
                    continue
                else:
                    raise e

    def generate_dataset(self, split: str = "train") -> str:
        """Generate a complete dataset"""
        print(f"Generating {self.config.num_samples} {split} samples...")

        samples = []
        failed_samples = 0

        for sample_id in tqdm(
            range(self.config.num_samples), desc=f"Generating {split} data"
        ):
            try:
                sample = self._generate_single_sample(sample_id)
                samples.append(sample)
            except Exception as e:
                print(f"Failed to generate sample {sample_id}: {e}")
                failed_samples += 1
                continue

        print(
            f"Successfully generated {len(samples)} samples ({failed_samples} failed)"
        )

        # Save dataset
        output_file = self.output_dir / f"{split}_data.pkl"
        with open(output_file, "wb") as f:
            pickle.dump(samples, f)

        # Save metadata
        metadata = {
            "config": asdict(self.config),
            "split": split,
            "num_samples": len(samples),
            "failed_samples": failed_samples,
            "file_path": str(output_file),
        }

        metadata_file = self.output_dir / f"{split}_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Dataset saved to {output_file}")
        print(f"Metadata saved to {metadata_file}")

        return str(output_file)

    def generate_modification_pairs(self, base_samples: int = 1000) -> str:
        """Generate paired samples for rule modification training"""
        print(f"Generating {base_samples} modification pairs...")

        modification_data = []

        for sample_id in tqdm(
            range(base_samples), desc="Generating modification pairs"
        ):
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

                # Generate modified versions with natural language descriptions
                modifications = [
                    {
                        "params": {"gravity": base_config.gravity * 1.2},
                        "description": "increase gravity by 20%",
                        "type": "gravity_increase",
                    },
                    {
                        "params": {"gravity": base_config.gravity * 0.8},
                        "description": "decrease gravity by 20%",
                        "type": "gravity_decrease",
                    },
                    {
                        "params": {"friction": max(0.05, base_config.friction * 0.3)},
                        "description": "reduce friction significantly",
                        "type": "friction_decrease",
                    },
                    {
                        "params": {"friction": min(0.95, base_config.friction * 1.5)},
                        "description": "increase friction",
                        "type": "friction_increase",
                    },
                    {
                        "params": {
                            "elasticity": max(0.1, base_config.elasticity * 0.5)
                        },
                        "description": "make objects less bouncy",
                        "type": "elasticity_decrease",
                    },
                    {
                        "params": {
                            "elasticity": min(0.99, base_config.elasticity * 1.2)
                        },
                        "description": "make objects more bouncy",
                        "type": "elasticity_increase",
                    },
                    {
                        "params": {"damping": max(0.8, base_config.damping * 0.9)},
                        "description": "increase air resistance",
                        "type": "damping_increase",
                    },
                    {
                        "params": {
                            "gravity": base_config.gravity * 0.5,
                            "friction": base_config.friction * 0.5,
                        },
                        "description": "space-like physics with low gravity and friction",
                        "type": "space_physics",
                    },
                    {
                        "params": {
                            "damping": 0.85,
                            "gravity": base_config.gravity * 0.7,
                        },
                        "description": "underwater-like physics with high resistance",
                        "type": "underwater_physics",
                    },
                ]

                for mod_idx, modification in enumerate(modifications):
                    # Create modified world
                    modified_world = PhysicsWorld(base_config)
                    for ball in balls:
                        modified_world.add_ball(ball)

                    # Apply modification
                    modified_world.modify_physics(modification["params"])

                    # Run modified simulation
                    for _ in range(self.config.sequence_length):
                        modified_world.step()

                    modified_trajectory = modified_world.get_trajectory_array()
                    modified_metrics = modified_world.get_physics_metrics()

                    # Create modification pair with enhanced information
                    pair_data = {
                        "pair_id": f"{sample_id}_{mod_idx}",
                        "base_config": asdict(base_config),
                        "modification_params": modification["params"],
                        "modification_description": modification["description"],
                        "modification_type": modification["type"],
                        "initial_balls": [asdict(ball) for ball in balls],
                        "base_trajectory": base_trajectory.tolist(),
                        "modified_trajectory": modified_trajectory.tolist(),
                        "base_metrics": base_metrics,
                        "modified_metrics": modified_metrics,
                        "parameter_changes": {
                            param: {
                                "original": getattr(base_config, param),
                                "modified": value,
                                "change_ratio": value / getattr(base_config, param)
                                if getattr(base_config, param) != 0
                                else 0,
                            }
                            for param, value in modification["params"].items()
                        },
                    }

                    modification_data.append(pair_data)

            except Exception as e:
                print(f"Failed to generate modification pair {sample_id}: {e}")
                continue

        # Save modification dataset
        output_file = self.output_dir / "modification_pairs.pkl"
        with open(output_file, "wb") as f:
            pickle.dump(modification_data, f)

        print(f"Generated {len(modification_data)} modification pairs")
        print(f"Modification data saved to {output_file}")

        return str(output_file)


def generate_physics_datasets():
    """Generate all physics datasets for training"""

    # Base configuration
    config = DataConfig(num_samples=10000, output_dir="data/processed/physics_worlds")

    generator = PhysicsDataGenerator(config)

    # Generate training data
    print("=== Generating Training Data ===")
    train_file = generator.generate_dataset("train")

    # Generate validation data
    print("\n=== Generating Validation Data ===")
    val_config = DataConfig(
        num_samples=2000, output_dir="data/processed/physics_worlds"
    )
    val_generator = PhysicsDataGenerator(val_config)
    val_file = val_generator.generate_dataset("val")

    # Generate test data
    print("\n=== Generating Test Data ===")
    test_config = DataConfig(
        num_samples=1000, output_dir="data/processed/physics_worlds"
    )
    test_generator = PhysicsDataGenerator(test_config)
    test_file = test_generator.generate_dataset("test")

    # Generate modification pairs
    print("\n=== Generating Modification Pairs ===")
    mod_file = generator.generate_modification_pairs(
        1000
    )  # 1000 base samples * 9 modifications = 9000 pairs

    print("\n=== Data Generation Complete ===")
    print(f"Training data: {train_file}")
    print(f"Validation data: {val_file}")
    print(f"Test data: {test_file}")
    print(f"Modification pairs: {mod_file}")

    # Generate quality report
    print("\n=== Generating Quality Report ===")
    datasets = {
        "train": train_file,
        "val": val_file,
        "test": test_file,
        "modifications": mod_file,
    }

    quality_report = generate_quality_report(datasets, config.output_dir)

    print(f"\nOverall Statistics:")
    print(f"  Total samples: {quality_report['overall_statistics']['total_samples']:,}")
    print(
        f"  Quality rate: {quality_report['overall_statistics']['overall_quality_rate']:.2%}"
    )

    return datasets


def load_physics_dataset(file_path: str) -> List[Dict[str, Any]]:
    """Load a physics dataset from file"""
    with open(file_path, "rb") as f:
        return pickle.load(f)


def generate_quality_report(
    datasets: Dict[str, str], output_dir: str = "data/processed/physics_worlds"
) -> Dict[str, Any]:
    """Generate comprehensive quality report for all datasets"""
    print("Generating data quality report...")

    report = {
        "generation_timestamp": time.time(),
        "datasets": {},
        "overall_statistics": {},
        "quality_metrics": {},
    }

    all_quality_scores = []
    total_samples = 0

    for split, file_path in datasets.items():
        if not Path(file_path).exists():
            continue

        data = load_physics_dataset(file_path)
        dataset_stats = analyze_dataset_detailed(data, split)
        report["datasets"][split] = dataset_stats

        # Collect quality scores
        if data and "quality_validation" in data[0]:
            quality_scores = [
                sample.get("quality_validation", {}).get("is_valid", False)
                for sample in data
            ]
            all_quality_scores.extend(quality_scores)

        total_samples += len(data)

    # Overall statistics
    report["overall_statistics"] = {
        "total_samples": total_samples,
        "overall_quality_rate": np.mean(all_quality_scores)
        if all_quality_scores
        else 0.0,
        "datasets_generated": len(datasets),
    }

    # Save report
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    report_path = output_path / "data_quality_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"Quality report saved to {report_path}")
    return report


def analyze_dataset_detailed(data: List[Dict[str, Any]], split: str) -> Dict[str, Any]:
    """Detailed analysis of a dataset"""
    if not data:
        return {}

    # Basic statistics
    stats = {"split": split, "num_samples": len(data), "failed_samples": 0}

    # Physics parameter analysis (only for samples with physics_config)
    physics_samples = [sample for sample in data if "physics_config" in sample]
    if physics_samples:
        gravities = [sample["physics_config"]["gravity"] for sample in physics_samples]
        frictions = [sample["physics_config"]["friction"] for sample in physics_samples]
        elasticities = [
            sample["physics_config"]["elasticity"] for sample in physics_samples
        ]
        dampings = [sample["physics_config"]["damping"] for sample in physics_samples]
    else:
        gravities = frictions = elasticities = dampings = []

    if gravities:  # Only calculate stats if we have physics data
        stats["physics_parameters"] = {
            "gravity": {
                "min": min(gravities),
                "max": max(gravities),
                "mean": np.mean(gravities),
                "std": np.std(gravities),
            },
            "friction": {
                "min": min(frictions),
                "max": max(frictions),
                "mean": np.mean(frictions),
                "std": np.std(frictions),
            },
            "elasticity": {
                "min": min(elasticities),
                "max": max(elasticities),
                "mean": np.mean(elasticities),
                "std": np.std(elasticities),
            },
            "damping": {
                "min": min(dampings),
                "max": max(dampings),
                "mean": np.mean(dampings),
                "std": np.std(dampings),
            },
        }
    else:
        stats["physics_parameters"] = {
            "note": "No physics_config data found in samples"
        }

    # Trajectory analysis
    traj_samples = [sample for sample in data if "sequence_length" in sample]
    if traj_samples:
        traj_lengths = [sample["sequence_length"] for sample in traj_samples]
        ball_counts = [
            sample["num_balls"] for sample in traj_samples if "num_balls" in sample
        ]
    else:
        traj_lengths = ball_counts = []

    if traj_lengths:
        stats["trajectory_stats"] = {
            "length": {
                "min": min(traj_lengths),
                "max": max(traj_lengths),
                "mean": np.mean(traj_lengths),
                "std": np.std(traj_lengths),
            },
            "ball_count": {
                "min": min(ball_counts),
                "max": max(ball_counts),
                "mean": np.mean(ball_counts),
                "std": np.std(ball_counts),
            }
            if ball_counts
            else {"note": "No ball_count data"},
        }
    else:
        stats["trajectory_stats"] = {"note": "No trajectory data found"}

    # Quality metrics (if available)
    if data and "quality_validation" in data[0]:
        quality_validations = [sample.get("quality_validation", {}) for sample in data]
        valid_samples = [q.get("is_valid", False) for q in quality_validations]
        retry_counts = [q.get("retry_count", 0) for q in quality_validations]

        stats["quality_metrics"] = {
            "validation_rate": np.mean(valid_samples),
            "avg_retry_count": np.mean(retry_counts),
            "max_retry_count": max(retry_counts) if retry_counts else 0,
        }

    # Energy conservation analysis (if available)
    if "metrics" in data[0]:
        energy_conservations = [
            sample["metrics"].get("energy_conservation", 0) for sample in data
        ]
        stats["energy_conservation"] = {
            "mean": np.mean(energy_conservations),
            "std": np.std(energy_conservations),
            "within_threshold": np.mean([abs(e) < 0.15 for e in energy_conservations]),
        }

    return stats


def analyze_dataset(file_path: str):
    """Analyze dataset statistics (simple version for backward compatibility)"""
    data = load_physics_dataset(file_path)

    print(f"Dataset: {file_path}")
    print(f"Number of samples: {len(data)}")

    if not data:
        return

    # Analyze physics parameters
    physics_samples = [sample for sample in data if "physics_config" in sample]
    if physics_samples:
        gravities = [sample["physics_config"]["gravity"] for sample in physics_samples]
        frictions = [sample["physics_config"]["friction"] for sample in physics_samples]
        elasticities = [
            sample["physics_config"]["elasticity"] for sample in physics_samples
        ]

        print(f"Gravity range: {min(gravities):.1f} to {max(gravities):.1f}")
        print(f"Friction range: {min(frictions):.3f} to {max(frictions):.3f}")
        print(f"Elasticity range: {min(elasticities):.3f} to {max(elasticities):.3f}")
    else:
        print("No physics_config data found in samples")

    # Analyze trajectory lengths
    traj_lengths = [sample["sequence_length"] for sample in data]
    print(
        f"Trajectory lengths: {min(traj_lengths)} to {max(traj_lengths)} (avg: {np.mean(traj_lengths):.1f})"
    )

    # Analyze ball counts
    ball_counts = [sample["num_balls"] for sample in data]
    print(
        f"Ball counts: {min(ball_counts)} to {max(ball_counts)} (avg: {np.mean(ball_counts):.1f})"
    )

    # Quality metrics if available
    if data and "quality_validation" in data[0]:
        valid_samples = [
            sample.get("quality_validation", {}).get("is_valid", False)
            for sample in data
        ]
        print(f"Quality validation rate: {np.mean(valid_samples):.2%}")


if __name__ == "__main__":
    # Test data generation
    print("Testing physics data generation...")

    # Generate small test dataset
    test_config = DataConfig(
        num_samples=10,
        output_dir="data/processed/physics_worlds_test",
        save_visualizations=True,
    )

    generator = PhysicsDataGenerator(test_config)
    test_file = generator.generate_dataset("test")

    # Analyze the test dataset
    analyze_dataset(test_file)

    print("\nTest data generation complete!")

    # Generate full datasets (uncomment to run)
    # datasets = generate_physics_datasets()
