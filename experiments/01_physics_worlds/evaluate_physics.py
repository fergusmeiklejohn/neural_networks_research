"""
Evaluation framework for Physics Distribution Invention
Comprehensive testing and analysis tools.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

import keras

from experiments.physics_worlds.data_generator import load_physics_dataset


class PhysicsEvaluator:
    """Comprehensive evaluation framework for physics distribution invention"""

    def __init__(self, model_path: str):
        self.model = keras.models.load_model(model_path)
        self.results = {}

    def evaluate_rule_extraction(
        self, test_data: List[Dict], sample_size: int = 100
    ) -> Dict[str, float]:
        """Evaluate how well the model extracts physics rules from trajectories"""
        print("Evaluating rule extraction accuracy...")

        extraction_errors = {
            "gravity": [],
            "friction": [],
            "elasticity": [],
            "damping": [],
        }

        for i, sample in enumerate(
            tqdm(test_data[:sample_size], desc="Rule extraction")
        ):
            try:
                trajectory = np.array(sample["trajectory"])
                true_physics = sample["physics_config"]

                # Extract rules using the model
                extracted = self.model.rule_extractor.extract_rules(
                    np.expand_dims(trajectory, 0)
                )

                # Calculate errors
                for rule in extraction_errors.keys():
                    if rule in true_physics:
                        true_value = true_physics[rule]
                        pred_value = extracted[rule][0]  # Remove batch dimension

                        # Normalize error by true value magnitude
                        if abs(true_value) > 1e-6:
                            error = abs(pred_value - true_value) / abs(true_value)
                        else:
                            error = abs(pred_value - true_value)

                        extraction_errors[rule].append(error)

            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue

        # Calculate metrics
        extraction_metrics = {}
        for rule, errors in extraction_errors.items():
            if errors:
                extraction_metrics[f"{rule}_mae"] = np.mean(errors)
                extraction_metrics[f"{rule}_std"] = np.std(errors)
                extraction_metrics[f"{rule}_accuracy_10%"] = np.mean(
                    np.array(errors) < 0.1
                )
                extraction_metrics[f"{rule}_accuracy_20%"] = np.mean(
                    np.array(errors) < 0.2
                )

        return extraction_metrics

    def evaluate_modification_consistency(
        self, test_data: List[Dict], sample_size: int = 50
    ) -> Dict[str, float]:
        """Evaluate consistency of rule modifications"""
        print("Evaluating modification consistency...")

        modification_tests = [
            ("increase gravity by 20%", "gravity", 1.2),
            ("decrease gravity by 20%", "gravity", 0.8),
            ("increase friction", "friction", lambda x: x + 0.2),
            ("decrease friction", "friction", lambda x: max(0, x - 0.2)),
            ("make more bouncy", "elasticity", lambda x: min(1.0, x + 0.1)),
            ("reduce bounce", "elasticity", lambda x: max(0, x - 0.1)),
        ]

        consistency_scores = []
        directional_accuracy = []

        for i, sample in enumerate(
            tqdm(test_data[:sample_size], desc="Modification consistency")
        ):
            try:
                trajectory = np.array(sample["trajectory"])
                original_physics = sample["physics_config"]
                initial_conditions = np.random.random((2, 9))  # Simplified

                for request, target_rule, expected_change in modification_tests:
                    result = self.model.invent_distribution(
                        trajectory, request, initial_conditions
                    )

                    if result["success"]:
                        original_value = original_physics[target_rule]
                        modified_value = result["modified_rules"][target_rule]

                        # Calculate expected value
                        if callable(expected_change):
                            expected_value = expected_change(original_value)
                        else:
                            expected_value = original_value * expected_change

                        # Consistency score (how close to expected)
                        if abs(expected_value) > 1e-6:
                            consistency = 1.0 - abs(
                                modified_value - expected_value
                            ) / abs(expected_value)
                        else:
                            consistency = 1.0 - abs(modified_value - expected_value)

                        consistency_scores.append(max(0, consistency))

                        # Directional accuracy (is change in right direction?)
                        if isinstance(expected_change, float):
                            if expected_change > 1.0:  # Should increase
                                directional_accuracy.append(
                                    modified_value > original_value
                                )
                            elif expected_change < 1.0:  # Should decrease
                                directional_accuracy.append(
                                    modified_value < original_value
                                )

            except Exception as e:
                print(f"Error in modification test {i}: {e}")
                continue

        return {
            "modification_consistency": np.mean(consistency_scores)
            if consistency_scores
            else 0.0,
            "directional_accuracy": np.mean(directional_accuracy)
            if directional_accuracy
            else 0.0,
            "success_rate": len(consistency_scores)
            / (sample_size * len(modification_tests)),
        }

    def evaluate_trajectory_quality(
        self, test_data: List[Dict], sample_size: int = 50
    ) -> Dict[str, float]:
        """Evaluate quality of generated trajectories"""
        print("Evaluating trajectory quality...")

        quality_metrics = {
            "energy_conservation": [],
            "smoothness": [],
            "physical_plausibility": [],
            "collision_realism": [],
        }

        for i, sample in enumerate(
            tqdm(test_data[:sample_size], desc="Trajectory quality")
        ):
            try:
                trajectory = np.array(sample["trajectory"])
                physics_config = sample["physics_config"]
                initial_conditions = np.random.random((2, 9))

                # Generate new trajectory
                result = self.model.invent_distribution(
                    trajectory, "slightly increase gravity", initial_conditions
                )

                if result["success"]:
                    generated_traj = result["new_trajectory"]

                    # Energy conservation check
                    energy_conservation = self._check_energy_conservation(
                        generated_traj
                    )
                    quality_metrics["energy_conservation"].append(energy_conservation)

                    # Trajectory smoothness
                    smoothness = self._check_trajectory_smoothness(generated_traj)
                    quality_metrics["smoothness"].append(smoothness)

                    # Physical plausibility
                    plausibility = self._check_physical_plausibility(
                        generated_traj, physics_config
                    )
                    quality_metrics["physical_plausibility"].append(plausibility)

                    # Collision realism (simplified)
                    collision_realism = self._check_collision_realism(generated_traj)
                    quality_metrics["collision_realism"].append(collision_realism)

            except Exception as e:
                print(f"Error in quality evaluation {i}: {e}")
                continue

        # Calculate averages
        avg_metrics = {}
        for metric, values in quality_metrics.items():
            if values:
                avg_metrics[f"{metric}_avg"] = np.mean(values)
                avg_metrics[f"{metric}_std"] = np.std(values)

        return avg_metrics

    def evaluate_distribution_space(
        self, test_data: List[Dict], sample_size: int = 30
    ) -> Dict[str, Any]:
        """Evaluate the space of possible distributions the model can create"""
        print("Evaluating distribution space coverage...")

        modification_requests = [
            "increase gravity by 50%",
            "decrease gravity by 50%",
            "remove all friction",
            "maximum friction",
            "perfectly elastic collisions",
            "completely inelastic collisions",
            "zero air resistance",
            "high air resistance",
            "anti-gravity",
            "sideways gravity",
        ]

        generated_physics = []
        novelty_scores = []
        success_rates = {}

        for request in modification_requests:
            successes = 0
            attempts = 0
            request_physics = []

            for i, sample in enumerate(test_data[:sample_size]):
                try:
                    trajectory = np.array(sample["trajectory"])
                    initial_conditions = np.random.random((2, 9))

                    result = self.model.invent_distribution(
                        trajectory, request, initial_conditions
                    )

                    attempts += 1
                    if result["success"]:
                        successes += 1
                        request_physics.append(result["modified_rules"])

                        if "quality_scores" in result:
                            novelty_scores.append(
                                result["quality_scores"].get("modification_novelty", 0)
                            )

                except Exception:
                    continue

            if attempts > 0:
                success_rates[request] = successes / attempts
                generated_physics.extend(request_physics)

        # Analyze coverage of physics parameter space
        if generated_physics:
            param_ranges = {}
            for param in ["gravity", "friction", "elasticity", "damping"]:
                values = [p[param] for p in generated_physics if param in p]
                if values:
                    param_ranges[param] = {
                        "min": np.min(values),
                        "max": np.max(values),
                        "range": np.max(values) - np.min(values),
                        "std": np.std(values),
                    }

        return {
            "success_rates": success_rates,
            "parameter_ranges": param_ranges if generated_physics else {},
            "average_novelty": np.mean(novelty_scores) if novelty_scores else 0.0,
            "total_unique_distributions": len(generated_physics),
        }

    def _check_energy_conservation(self, trajectory: np.ndarray) -> float:
        """Check how well energy is conserved in a trajectory"""
        if len(trajectory.shape) < 3 or trajectory.shape[-1] < 9:
            return 0.0

        # Extract kinetic and potential energy (assuming last 2 features)
        ke = trajectory[:, :, -2]  # [time, balls]
        pe = trajectory[:, :, -1]
        total_energy = ke + pe

        # Calculate energy conservation score
        energy_variance = np.var(total_energy.flatten())
        conservation_score = 1.0 / (1.0 + energy_variance)

        return conservation_score

    def _check_trajectory_smoothness(self, trajectory: np.ndarray) -> float:
        """Check trajectory smoothness (no sudden jumps)"""
        if len(trajectory.shape) < 3 or trajectory.shape[0] < 2:
            return 0.0

        # Extract positions (assuming features 1:3 are x, y)
        positions = trajectory[:, :, 1:3]  # [time, balls, xy]

        # Calculate position differences between consecutive frames
        position_diffs = np.diff(positions, axis=0)

        # Smoothness is inverse of average position jump magnitude
        jump_magnitudes = np.sqrt(np.sum(position_diffs**2, axis=-1))
        avg_jump = np.mean(jump_magnitudes)

        smoothness_score = 1.0 / (1.0 + avg_jump)
        return smoothness_score

    def _check_physical_plausibility(
        self, trajectory: np.ndarray, physics_config: Dict
    ) -> float:
        """Check if trajectory follows expected physics"""
        # Simplified plausibility check
        # In a real implementation, this would run the trajectory through
        # a physics simulator and compare

        if len(trajectory.shape) < 3:
            return 0.0

        # Check if objects fall downward under gravity
        positions_y = trajectory[:, :, 2]  # y positions

        if physics_config.get("gravity", 0) < 0:  # Downward gravity
            # Objects should generally move downward over time
            y_trend = np.mean(np.diff(positions_y, axis=0))
            plausibility = 1.0 if y_trend < 0 else 0.5
        else:
            plausibility = 0.8  # Neutral for unusual gravity

        return plausibility

    def _check_collision_realism(self, trajectory: np.ndarray) -> float:
        """Check if collisions look realistic"""
        # Simplified collision realism check
        # Would need more sophisticated analysis in practice

        if len(trajectory.shape) < 3 or trajectory.shape[1] < 2:
            return 0.8  # Default score for single ball

        # Check if balls don't overlap (simplified)
        positions = trajectory[:, :, 1:3]  # [time, balls, xy]

        overlaps = 0
        total_checks = 0

        for t in range(len(positions)):
            for i in range(positions.shape[1]):
                for j in range(i + 1, positions.shape[1]):
                    dist = np.sqrt(np.sum((positions[t, i] - positions[t, j]) ** 2))
                    # Assuming radius ~20 pixels
                    if dist < 40:  # Balls overlapping
                        overlaps += 1
                    total_checks += 1

        if total_checks > 0:
            collision_realism = 1.0 - (overlaps / total_checks)
        else:
            collision_realism = 1.0

        return collision_realism

    def generate_report(self, output_dir: str = "outputs/evaluation"):
        """Generate comprehensive evaluation report"""
        os.makedirs(output_dir, exist_ok=True)

        # Combine all results
        full_report = {
            "model_info": {
                "parameters": self.model.count_params(),
                "architecture": "DistributionInventor",
            },
            "evaluation_results": self.results,
        }

        # Save JSON report
        report_path = os.path.join(output_dir, "evaluation_report.json")
        with open(report_path, "w") as f:
            json.dump(full_report, f, indent=2, default=str)

        # Generate visualizations
        self._create_visualizations(output_dir)

        print(f"Evaluation report saved to {report_path}")
        return full_report

    def _create_visualizations(self, output_dir: str):
        """Create evaluation visualizations"""

        # Rule extraction accuracy plot
        if "rule_extraction" in self.results:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))

            rules = ["gravity", "friction", "elasticity", "damping"]
            for i, rule in enumerate(rules):
                ax = axes[i // 2, i % 2]

                mae_key = f"{rule}_mae"
                acc_10_key = f"{rule}_accuracy_10%"

                if mae_key in self.results["rule_extraction"]:
                    mae = self.results["rule_extraction"][mae_key]
                    acc = self.results["rule_extraction"].get(acc_10_key, 0)

                    ax.bar(["MAE", "Accuracy@10%"], [mae, acc])
                    ax.set_title(f"{rule.capitalize()} Extraction")
                    ax.set_ylim(0, 1)

            plt.tight_layout()
            plt.savefig(
                os.path.join(output_dir, "rule_extraction_accuracy.png"), dpi=300
            )
            plt.close()

        # Success rates by modification type
        if "distribution_space" in self.results:
            success_rates = self.results["distribution_space"].get("success_rates", {})

            if success_rates:
                plt.figure(figsize=(12, 6))
                requests = list(success_rates.keys())
                rates = list(success_rates.values())

                plt.bar(range(len(requests)), rates)
                plt.xticks(range(len(requests)), requests, rotation=45, ha="right")
                plt.ylabel("Success Rate")
                plt.title("Success Rates by Modification Type")
                plt.tight_layout()
                plt.savefig(
                    os.path.join(output_dir, "modification_success_rates.png"), dpi=300
                )
                plt.close()

        print(f"Visualizations saved to {output_dir}")

    def run_full_evaluation(
        self, test_data_path: str, sample_sizes: Dict[str, int] = None
    ) -> Dict[str, Any]:
        """Run complete evaluation suite"""

        if sample_sizes is None:
            sample_sizes = {
                "rule_extraction": 100,
                "modification_consistency": 50,
                "trajectory_quality": 50,
                "distribution_space": 30,
            }

        # Load test data
        test_data = load_physics_dataset(test_data_path)
        print(f"Loaded {len(test_data)} test samples")

        # Run all evaluations
        print("Starting comprehensive evaluation...")

        self.results["rule_extraction"] = self.evaluate_rule_extraction(
            test_data, sample_sizes["rule_extraction"]
        )

        self.results[
            "modification_consistency"
        ] = self.evaluate_modification_consistency(
            test_data, sample_sizes["modification_consistency"]
        )

        self.results["trajectory_quality"] = self.evaluate_trajectory_quality(
            test_data, sample_sizes["trajectory_quality"]
        )

        self.results["distribution_space"] = self.evaluate_distribution_space(
            test_data, sample_sizes["distribution_space"]
        )

        # Generate report
        report = self.generate_report()

        print("Evaluation complete!")
        return report


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Physics Distribution Inventor"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to trained model"
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default="data/processed/physics_worlds/test_data.pkl",
        help="Path to test data",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/evaluation",
        help="Output directory for evaluation results",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick evaluation with smaller sample sizes",
    )

    args = parser.parse_args()

    # Create evaluator
    evaluator = PhysicsEvaluator(args.model_path)

    # Set sample sizes
    if args.quick:
        sample_sizes = {
            "rule_extraction": 20,
            "modification_consistency": 10,
            "trajectory_quality": 10,
            "distribution_space": 5,
        }
    else:
        sample_sizes = None  # Use defaults

    # Run evaluation
    report = evaluator.run_full_evaluation(args.test_data, sample_sizes)

    # Print summary
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)

    for category, results in report["evaluation_results"].items():
        print(f"\n{category.upper()}:")
        for metric, value in results.items():
            if isinstance(value, (int, float)):
                print(f"  {metric}: {value:.4f}")
            elif isinstance(value, dict) and len(value) < 10:
                print(f"  {metric}: {value}")


if __name__ == "__main__":
    main()
