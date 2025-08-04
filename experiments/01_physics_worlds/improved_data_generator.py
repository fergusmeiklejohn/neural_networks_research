"""
Improved Data Generator with Proper Train/Test Isolation
Implements the data isolation fix plan to ensure valid distribution invention testing.
"""

import json
import pickle
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from physics_env import Ball, PhysicsConfig, PhysicsWorld
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

# Define parameter ranges for proper data splits
TRAIN_RANGES = {
    "gravity": (-1200, -300),  # Narrower than original
    "friction": (0.1, 0.8),  # Exclude extremes
    "elasticity": (0.2, 0.9),  # Exclude extremes
    "damping": (0.85, 0.98),  # Typical air resistance
}

NEAR_VAL_RANGES = {
    "gravity": (-1400, -250),  # 15% extension beyond training
    "friction": (0.05, 0.9),  # Include more extremes
    "elasticity": (0.15, 0.95),
    "damping": (0.82, 0.99),
}

EXTRAP_TEST_RANGES = {
    "gravity": [(-2000, -1500), (-200, -50)],  # Very high/low gravity
    "friction": [(0.0, 0.05), (0.9, 1.0)],  # Frictionless/sticky
    "elasticity": [(0.05, 0.15), (0.95, 1.1)],  # No bounce/super bounce
    "damping": [(0.7, 0.82), (0.99, 1.0)],  # Thick air/vacuum
}

NOVEL_REGIMES = {
    "moon": {"gravity": -162, "friction": 0.7, "elasticity": 0.8, "damping": 1.0},
    "jupiter": {"gravity": -2479, "friction": 0.5, "elasticity": 0.6, "damping": 0.9},
    "underwater": {"gravity": -600, "friction": 0.3, "elasticity": 0.4, "damping": 0.7},
    "ice_rink": {"gravity": -981, "friction": 0.02, "elasticity": 0.9, "damping": 0.98},
    "rubber_room": {
        "gravity": -981,
        "friction": 0.9,
        "elasticity": 0.99,
        "damping": 0.95,
    },
    "space_station": {
        "gravity": -20,
        "friction": 0.8,
        "elasticity": 0.7,
        "damping": 1.0,
    },
    "thick_atmosphere": {
        "gravity": -981,
        "friction": 0.6,
        "elasticity": 0.8,
        "damping": 0.75,
    },
    "super_earth": {
        "gravity": -1962,
        "friction": 0.4,
        "elasticity": 0.7,
        "damping": 0.92,
    },
}


@dataclass
class ImprovedDataConfig:
    """Configuration for improved data generation with proper splits"""

    total_samples: int = 13000  # Will be split appropriately
    sequence_length: int = 300
    min_balls: int = 1
    max_balls: int = 4

    # Ball property ranges (unchanged)
    ball_radius_range: Tuple[float, float] = (10, 30)
    ball_mass_range: Tuple[float, float] = (0.5, 2.0)
    ball_velocity_range: Tuple[float, float] = (-200, 200)

    # World dimensions
    world_width: int = 800
    world_height: int = 600

    # Output settings
    output_dir: str = "data/processed/physics_worlds_v2"
    save_visualizations: bool = False

    # Quality validation settings
    energy_conservation_threshold: float = 0.25
    min_trajectory_length: int = 30
    max_velocity_threshold: float = 1500
    max_jump_threshold: float = 150

    # Data split proportions
    train_proportion: float = 0.7
    val_in_dist_proportion: float = 0.1
    val_near_dist_proportion: float = 0.05
    test_interpolation_proportion: float = 0.05
    test_extrapolation_proportion: float = 0.05
    test_novel_proportion: float = 0.05


class ImprovedPhysicsDataGenerator:
    """Generate physics simulation data with proper train/test isolation"""

    def __init__(self, config: ImprovedDataConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Calculate sample sizes for each split
        self.split_sizes = self._calculate_split_sizes()

        # Define sampling strategies for each split
        self.split_definitions = self._define_data_splits()

    def _calculate_split_sizes(self) -> Dict[str, int]:
        """Calculate number of samples for each data split"""
        total = self.config.total_samples
        return {
            "train": int(total * self.config.train_proportion),
            "val_in_dist": int(total * self.config.val_in_dist_proportion),
            "val_near_dist": int(total * self.config.val_near_dist_proportion),
            "test_interpolation": int(
                total * self.config.test_interpolation_proportion
            ),
            "test_extrapolation": int(
                total * self.config.test_extrapolation_proportion
            ),
            "test_novel": int(total * self.config.test_novel_proportion),
        }

    def _define_data_splits(self) -> Dict[str, Dict]:
        """Define parameter ranges and sampling strategies for each split"""
        return {
            "train": {
                "ranges": TRAIN_RANGES,
                "sampling": "random_uniform",
                "description": "Training data with core parameter ranges",
            },
            "val_in_dist": {
                "ranges": TRAIN_RANGES,
                "sampling": "random_uniform",
                "description": "In-distribution validation for hyperparameter tuning",
            },
            "val_near_dist": {
                "ranges": NEAR_VAL_RANGES,
                "sampling": "random_uniform",
                "description": "Near-distribution validation for early generalization testing",
            },
            "test_interpolation": {
                "ranges": TRAIN_RANGES,
                "sampling": "grid",
                "description": "Systematic coverage of training parameter space",
            },
            "test_extrapolation": {
                "ranges": EXTRAP_TEST_RANGES,
                "sampling": "systematic_extrapolation",
                "description": "Out-of-distribution testing for true generalization",
            },
            "test_novel": {
                "regimes": NOVEL_REGIMES,
                "sampling": "predefined_regimes",
                "description": "Predefined physics scenarios for distribution invention testing",
            },
        }

    def _generate_physics_config_random(
        self, ranges: Dict[str, Tuple[float, float]]
    ) -> PhysicsConfig:
        """Generate random physics configuration within specified ranges"""
        return PhysicsConfig(
            gravity=random.uniform(*ranges["gravity"]),
            friction=random.uniform(*ranges["friction"]),
            elasticity=random.uniform(*ranges["elasticity"]),
            damping=random.uniform(*ranges["damping"]),
            world_width=self.config.world_width,
            world_height=self.config.world_height,
        )

    def _generate_physics_config_grid(
        self, ranges: Dict[str, Tuple[float, float]], num_samples: int
    ) -> List[PhysicsConfig]:
        """Generate physics configurations using grid sampling for systematic coverage"""
        # Calculate grid dimensions (approximately equal spacing)
        n_dims = len(ranges)
        samples_per_dim = int(np.ceil(num_samples ** (1 / n_dims)))

        # Create parameter grid
        param_values = {}
        for param, (min_val, max_val) in ranges.items():
            param_values[param] = np.linspace(min_val, max_val, samples_per_dim)

        # Generate all combinations
        grid = list(ParameterGrid(param_values))

        # Randomly sample if we have too many combinations
        if len(grid) > num_samples:
            grid = random.sample(grid, num_samples)

        # Convert to PhysicsConfig objects
        configs = []
        for params in grid:
            config = PhysicsConfig(
                gravity=params["gravity"],
                friction=params["friction"],
                elasticity=params["elasticity"],
                damping=params["damping"],
                world_width=self.config.world_width,
                world_height=self.config.world_height,
            )
            configs.append(config)

        return configs

    def _generate_physics_config_extrapolation(
        self, ranges: Dict[str, List[Tuple[float, float]]], num_samples: int
    ) -> List[PhysicsConfig]:
        """Generate physics configurations for extrapolation testing"""
        configs = []

        # For each parameter, sample from out-of-distribution ranges
        param_names = list(ranges.keys())
        samples_per_param = num_samples // len(param_names)

        for param_name in param_names:
            param_ranges = ranges[param_name]

            for range_tuple in param_ranges:
                range_samples = samples_per_param // len(param_ranges)

                for _ in range(range_samples):
                    # Start with baseline values from training range center
                    baseline_params = {
                        "gravity": -750,  # Middle of training range
                        "friction": 0.45,
                        "elasticity": 0.55,
                        "damping": 0.915,
                    }

                    # Override the specific parameter with extrapolation value
                    baseline_params[param_name] = random.uniform(*range_tuple)

                    config = PhysicsConfig(
                        gravity=baseline_params["gravity"],
                        friction=baseline_params["friction"],
                        elasticity=baseline_params["elasticity"],
                        damping=baseline_params["damping"],
                        world_width=self.config.world_width,
                        world_height=self.config.world_height,
                    )
                    configs.append(config)

        # Fill remaining samples with random combinations of extrapolation ranges
        remaining = num_samples - len(configs)
        for _ in range(remaining):
            params = {}
            for param_name, param_ranges in ranges.items():
                chosen_range = random.choice(param_ranges)
                params[param_name] = random.uniform(*chosen_range)

            config = PhysicsConfig(
                **params,
                world_width=self.config.world_width,
                world_height=self.config.world_height,
            )
            configs.append(config)

        return configs

    def _generate_physics_config_novel(
        self, regimes: Dict[str, Dict], num_samples: int
    ) -> List[PhysicsConfig]:
        """Generate physics configurations for novel regime testing"""
        configs = []
        regime_names = list(regimes.keys())
        samples_per_regime = num_samples // len(regime_names)

        for regime_name in regime_names:
            regime_params = regimes[regime_name]

            for _ in range(samples_per_regime):
                config = PhysicsConfig(
                    gravity=regime_params["gravity"],
                    friction=regime_params["friction"],
                    elasticity=regime_params["elasticity"],
                    damping=regime_params["damping"],
                    world_width=self.config.world_width,
                    world_height=self.config.world_height,
                )
                configs.append(config)

        # Fill remaining samples by randomly selecting regimes
        remaining = num_samples - len(configs)
        for _ in range(remaining):
            regime_name = random.choice(regime_names)
            regime_params = regimes[regime_name]

            config = PhysicsConfig(
                gravity=regime_params["gravity"],
                friction=regime_params["friction"],
                elasticity=regime_params["elasticity"],
                damping=regime_params["damping"],
                world_width=self.config.world_width,
                world_height=self.config.world_height,
            )
            configs.append(config)

        return configs

    def _generate_random_balls(self) -> List[Ball]:
        """Generate random ball configurations (unchanged from original)"""
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
        """Validate the quality of a generated sample (unchanged from original)"""
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

            # Check for reasonable velocities
            if len(trajectory) > 0 and trajectory.shape[-1] >= 5:
                num_features_per_ball = 9
                num_balls = trajectory.shape[1] // num_features_per_ball

                velocities = []
                for ball_idx in range(num_balls):
                    start_idx = ball_idx * num_features_per_ball
                    if start_idx + 5 <= trajectory.shape[1]:
                        vx = trajectory[:, start_idx + 3]
                        vy = trajectory[:, start_idx + 4]
                        velocities.extend([vx, vy])

                if velocities:
                    max_velocity = np.max(np.abs(velocities))
                    if max_velocity > self.config.max_velocity_threshold:
                        return False, f"Unreasonable velocity: {max_velocity:.2f}"

            # Check for trajectory smoothness
            if len(trajectory) > 1 and trajectory.shape[-1] >= 3:
                num_features_per_ball = 9
                num_balls = trajectory.shape[1] // num_features_per_ball

                max_jump = 0
                for ball_idx in range(num_balls):
                    start_idx = ball_idx * num_features_per_ball
                    if start_idx + 3 <= trajectory.shape[1]:
                        x_positions = trajectory[:, start_idx + 1]
                        y_positions = trajectory[:, start_idx + 2]

                        x_diffs = np.diff(x_positions)
                        y_diffs = np.diff(y_positions)
                        jumps = np.sqrt(x_diffs**2 + y_diffs**2)
                        max_jump = max(max_jump, np.max(jumps))

                if max_jump > self.config.max_jump_threshold:
                    return False, f"Trajectory too jumpy: {max_jump:.2f}"

            return True, "Valid"

        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def _generate_single_sample(
        self, sample_id: int, physics_config: PhysicsConfig, split_name: str
    ) -> Dict[str, Any]:
        """Generate a single physics simulation sample"""
        max_retries = 3

        for retry in range(max_retries):
            try:
                # Create random ball configuration
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
                    "split_name": split_name,
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

                # Add split metadata
                sample_data["split_metadata"] = {
                    "split_name": split_name,
                    "sampling_method": self.split_definitions[split_name]["sampling"],
                    "description": self.split_definitions[split_name]["description"],
                }

                return sample_data

            except Exception as e:
                if retry < max_retries - 1:
                    continue
                else:
                    raise e

    def generate_split_dataset(self, split_name: str) -> str:
        """Generate dataset for a specific split with proper isolation"""
        print(
            f"Generating {split_name} dataset ({self.split_sizes[split_name]} samples)..."
        )

        split_def = self.split_definitions[split_name]
        num_samples = self.split_sizes[split_name]

        # Generate physics configurations based on split type
        if split_def["sampling"] == "random_uniform":
            physics_configs = [
                self._generate_physics_config_random(split_def["ranges"])
                for _ in range(num_samples)
            ]
        elif split_def["sampling"] == "grid":
            physics_configs = self._generate_physics_config_grid(
                split_def["ranges"], num_samples
            )
        elif split_def["sampling"] == "systematic_extrapolation":
            physics_configs = self._generate_physics_config_extrapolation(
                split_def["ranges"], num_samples
            )
        elif split_def["sampling"] == "predefined_regimes":
            physics_configs = self._generate_physics_config_novel(
                split_def["regimes"], num_samples
            )
        else:
            raise ValueError(f"Unknown sampling method: {split_def['sampling']}")

        # Generate samples
        samples = []
        failed_samples = 0

        for sample_id, physics_config in enumerate(
            tqdm(physics_configs, desc=f"Generating {split_name}")
        ):
            try:
                sample = self._generate_single_sample(
                    sample_id, physics_config, split_name
                )
                samples.append(sample)
            except Exception as e:
                print(f"Failed to generate {split_name} sample {sample_id}: {e}")
                failed_samples += 1
                continue

        print(
            f"Successfully generated {len(samples)} {split_name} samples ({failed_samples} failed)"
        )

        # Save dataset
        output_file = self.output_dir / f"{split_name}_data.pkl"
        with open(output_file, "wb") as f:
            pickle.dump(samples, f)

        # Save metadata
        metadata = {
            "config": asdict(self.config),
            "split_name": split_name,
            "split_definition": split_def,
            "num_samples": len(samples),
            "failed_samples": failed_samples,
            "file_path": str(output_file),
            "parameter_coverage": self._analyze_parameter_coverage(samples),
        }

        metadata_file = self.output_dir / f"{split_name}_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        print(f"Dataset saved to {output_file}")
        print(f"Metadata saved to {metadata_file}")

        return str(output_file)

    def _analyze_parameter_coverage(self, samples: List[Dict]) -> Dict[str, Dict]:
        """Analyze parameter space coverage for a dataset split"""
        if not samples:
            return {}

        params = ["gravity", "friction", "elasticity", "damping"]
        coverage = {}

        for param in params:
            values = [sample["physics_config"][param] for sample in samples]
            coverage[param] = {
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "q25": float(np.percentile(values, 25)),
                "q75": float(np.percentile(values, 75)),
            }

        return coverage

    def generate_all_datasets(self) -> Dict[str, str]:
        """Generate all dataset splits with proper isolation"""
        print("🚀 IMPROVED PHYSICS WORLDS DATA GENERATION")
        print("=" * 60)
        print("Generating datasets with proper train/test isolation...")
        print()

        # Print split information
        for split_name, split_def in self.split_definitions.items():
            size = self.split_sizes[split_name]
            print(f"{split_name:>18}: {size:>5} samples - {split_def['description']}")
        print()

        start_time = time.time()
        datasets = {}

        # Generate each split
        for split_name in self.split_definitions.keys():
            try:
                file_path = self.generate_split_dataset(split_name)
                datasets[split_name] = file_path
                print()
            except Exception as e:
                print(f"❌ Failed to generate {split_name}: {e}")
                continue

        # Calculate total time
        total_time = time.time() - start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)

        print("🎉 IMPROVED DATA GENERATION COMPLETE!")
        print("=" * 50)
        print(f"⏱️  Total time: {hours:02d}h {minutes:02d}m {seconds:02d}s")
        print(f"📁 Output directory: {self.output_dir}")
        print()
        print("Generated files:")
        for split, path in datasets.items():
            if Path(path).exists():
                size_mb = Path(path).stat().st_size / (1024 * 1024)
                print(f"   {split:>18}: {Path(path).name} ({size_mb:.1f} MB)")

        # Generate comprehensive report
        self._generate_isolation_report(datasets)

        print()
        print("✅ Ready for Phase 2 with proper data isolation!")
        print("   Next steps:")
        print("   1. Update training scripts to use new data splits")
        print("   2. Implement new evaluation metrics")
        print("   3. Re-run experiments with proper isolation testing")

        return datasets

    def _generate_isolation_report(self, datasets: Dict[str, str]):
        """Generate comprehensive report on data isolation"""
        report = {
            "generation_timestamp": time.time(),
            "config": asdict(self.config),
            "split_definitions": self.split_definitions,
            "split_sizes": self.split_sizes,
            "datasets": {},
            "isolation_verification": {},
            "parameter_range_analysis": {},
        }

        # Analyze each dataset
        for split_name, file_path in datasets.items():
            if Path(file_path).exists():
                with open(file_path, "rb") as f:
                    data = pickle.load(f)

                report["datasets"][split_name] = {
                    "file_path": file_path,
                    "num_samples": len(data),
                    "parameter_coverage": self._analyze_parameter_coverage(data),
                }

        # Verify no parameter overlap between train and test
        train_coverage = (
            report["datasets"].get("train", {}).get("parameter_coverage", {})
        )

        for test_split in ["test_interpolation", "test_extrapolation", "test_novel"]:
            if test_split in report["datasets"]:
                test_coverage = report["datasets"][test_split]["parameter_coverage"]
                overlap_analysis = self._analyze_parameter_overlap(
                    train_coverage, test_coverage
                )
                report["isolation_verification"][
                    f"train_vs_{test_split}"
                ] = overlap_analysis

        # Save report
        report_path = self.output_dir / "isolation_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        print(f"📊 Isolation report saved to: {report_path}")

    def _analyze_parameter_overlap(
        self, train_coverage: Dict, test_coverage: Dict
    ) -> Dict:
        """Analyze parameter overlap between train and test sets"""
        overlap_analysis = {}

        for param in ["gravity", "friction", "elasticity", "damping"]:
            if param in train_coverage and param in test_coverage:
                train_min, train_max = (
                    train_coverage[param]["min"],
                    train_coverage[param]["max"],
                )
                test_min, test_max = (
                    test_coverage[param]["min"],
                    test_coverage[param]["max"],
                )

                # Check for range overlap
                overlap_min = max(train_min, test_min)
                overlap_max = min(train_max, test_max)
                has_overlap = overlap_min < overlap_max

                if has_overlap:
                    overlap_fraction = (overlap_max - overlap_min) / (
                        test_max - test_min
                    )
                else:
                    overlap_fraction = 0.0

                overlap_analysis[param] = {
                    "train_range": [train_min, train_max],
                    "test_range": [test_min, test_max],
                    "has_overlap": has_overlap,
                    "overlap_fraction": overlap_fraction,
                    "extrapolation_direction": "both"
                    if test_min < train_min and test_max > train_max
                    else "lower"
                    if test_max < train_min
                    else "higher"
                    if test_min > train_max
                    else "interpolation",
                }

        return overlap_analysis


def generate_improved_physics_datasets():
    """Generate all physics datasets with proper train/test isolation"""

    config = ImprovedDataConfig(
        total_samples=13000,  # Will be split: 9100 train, 1300 val_in, 650 val_near, 650 test_interp, 650 test_extrap, 650 test_novel
        output_dir="data/processed/physics_worlds_v2",
    )

    generator = ImprovedPhysicsDataGenerator(config)
    datasets = generator.generate_all_datasets()

    return datasets


if __name__ == "__main__":
    print("Testing improved physics data generation...")
    datasets = generate_improved_physics_datasets()
    print("Improved data generation complete!")
