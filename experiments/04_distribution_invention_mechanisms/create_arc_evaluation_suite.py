#!/usr/bin/env python3
"""
Create evaluation suite using REAL ARC tasks.

CRITICAL: We must use actual ARC tasks to avoid biasing the evaluation
toward our approach. No synthetic tasks that we might unconsciously make easier!
"""

from utils.imports import setup_project_paths

setup_project_paths()

import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np


class ARCImaginationEvaluator:
    """
    Evaluates imagination requirements using real ARC tasks.

    Key principle: Use ONLY real ARC tasks, categorized by whether
    they require imagination vs pattern matching.
    """

    def __init__(self):
        self.data_dir = (
            Path(__file__).parent.parent.parent
            / "data"
            / "arc_agi_official"
            / "ARC-AGI"
            / "data"
            / "training"
        )

        # Categories based on our analysis
        self.imagination_required_tasks = [
            "05269061",  # Background removal - inconsistent patterns
            "007bbfb7",  # Position-dependent modifications
            "00d62c1b",  # Object-relative rules
            "0ca9ddb6",  # Complex emergent patterns
            "0d3d703e",  # Spatial transformations
            "025d127b",  # Rule composition
            "0520fde7",  # Abstract relationships
            "06df4c85",  # Multi-step reasoning
        ]

        self.pattern_matching_tasks = [
            "08ed6ac7",  # Simple color mapping (V7 solved)
            "0a938d79",  # Direct pattern application
            "09629e4f",  # Regular tiling
            "0b148d64",  # Symmetry operations
            "0e206a2e",  # Basic transformations
            "103f5b53",  # Color substitution
            "11852cab",  # Grid scaling
            "1190e5a7",  # Rotation/reflection
        ]

        self.mixed_tasks = [
            "05f2a901",  # Mostly pattern but needs small imagination
            "045e512c",  # Can be solved either way
            "04290ef0",  # Depends on approach
            "0692e18c",  # Borderline case
        ]

    def load_task(self, task_id: str) -> Dict:
        """Load a specific ARC task."""
        task_path = self.data_dir / f"{task_id}.json"
        if not task_path.exists():
            print(f"Warning: Task {task_id} not found")
            return None

        with open(task_path) as f:
            return json.load(f)

    def categorize_task(self, task_id: str) -> str:
        """Categorize a task as requiring imagination, pattern matching, or mixed."""
        if task_id in self.imagination_required_tasks:
            return "imagination"
        elif task_id in self.pattern_matching_tasks:
            return "pattern_matching"
        elif task_id in self.mixed_tasks:
            return "mixed"
        else:
            return "unknown"

    def evaluate_solver(self, solver, task_ids: List[str] = None) -> Dict:
        """
        Evaluate a solver on ARC tasks.

        Returns metrics broken down by category.
        """
        if task_ids is None:
            # Use a mix of all categories
            task_ids = (
                random.sample(
                    self.imagination_required_tasks,
                    min(3, len(self.imagination_required_tasks)),
                )
                + random.sample(
                    self.pattern_matching_tasks,
                    min(3, len(self.pattern_matching_tasks)),
                )
                + random.sample(self.mixed_tasks, min(2, len(self.mixed_tasks)))
            )

        results = {
            "imagination": {"total": 0, "solved": 0, "accuracies": []},
            "pattern_matching": {"total": 0, "solved": 0, "accuracies": []},
            "mixed": {"total": 0, "solved": 0, "accuracies": []},
            "overall": {"total": 0, "solved": 0, "accuracies": []},
        }

        for task_id in task_ids:
            task = self.load_task(task_id)
            if task is None:
                continue

            category = self.categorize_task(task_id)
            if category == "unknown":
                category = "mixed"  # Default to mixed for unknown tasks

            # Get training examples
            train_examples = [
                (np.array(ex["input"]), np.array(ex["output"])) for ex in task["train"]
            ]

            # Test on first test example
            test_input = np.array(task["test"][0]["input"])
            expected_output = np.array(task["test"][0]["output"])

            # Run solver
            try:
                predicted = solver.solve(train_examples, test_input)

                # Calculate accuracy
                if predicted is not None and predicted.shape == expected_output.shape:
                    accuracy = (
                        np.sum(predicted == expected_output) / expected_output.size
                    )
                    is_solved = accuracy == 1.0
                else:
                    accuracy = 0.0
                    is_solved = False
            except Exception as e:
                print(f"Error on task {task_id}: {e}")
                accuracy = 0.0
                is_solved = False

            # Update results
            results[category]["total"] += 1
            results[category]["solved"] += int(is_solved)
            results[category]["accuracies"].append(accuracy)

            results["overall"]["total"] += 1
            results["overall"]["solved"] += int(is_solved)
            results["overall"]["accuracies"].append(accuracy)

            print(
                f"Task {task_id} ({category}): {accuracy:.1%} {'✓' if is_solved else '✗'}"
            )

        # Calculate averages
        for category in results:
            if results[category]["accuracies"]:
                results[category]["avg_accuracy"] = np.mean(
                    results[category]["accuracies"]
                )
                results[category]["solve_rate"] = (
                    results[category]["solved"] / results[category]["total"]
                )
            else:
                results[category]["avg_accuracy"] = 0.0
                results[category]["solve_rate"] = 0.0

        return results

    def compare_approaches(self, solvers: Dict[str, any]) -> None:
        """Compare multiple solving approaches."""
        print("=" * 60)
        print("COMPARING APPROACHES ON REAL ARC TASKS")
        print("=" * 60)

        # Use same tasks for fair comparison
        test_tasks = (
            random.sample(self.imagination_required_tasks, 3)
            + random.sample(self.pattern_matching_tasks, 3)
            + random.sample(self.mixed_tasks, 2)
        )

        all_results = {}

        for name, solver in solvers.items():
            print(f"\nTesting {name}...")
            print("-" * 40)
            results = self.evaluate_solver(solver, test_tasks)
            all_results[name] = results

        # Summary comparison
        print("\n" + "=" * 60)
        print("SUMMARY COMPARISON")
        print("=" * 60)

        # Header
        print(f"{'Approach':<20} {'Overall':>10} {'Imagination':>12} {'Pattern':>10}")
        print("-" * 52)

        # Results for each solver
        for name, results in all_results.items():
            overall = f"{results['overall']['solve_rate']:.1%}"
            imagination = f"{results['imagination']['solve_rate']:.1%}"
            pattern = f"{results['pattern_matching']['solve_rate']:.1%}"
            print(f"{name:<20} {overall:>10} {imagination:>12} {pattern:>10}")

        # Analysis
        print("\n" + "=" * 60)
        print("KEY INSIGHTS")
        print("=" * 60)

        # Find best for each category
        best_overall = max(
            all_results.items(), key=lambda x: x[1]["overall"]["solve_rate"]
        )
        best_imagination = max(
            all_results.items(), key=lambda x: x[1]["imagination"]["solve_rate"]
        )
        best_pattern = max(
            all_results.items(), key=lambda x: x[1]["pattern_matching"]["solve_rate"]
        )

        print(
            f"Best overall: {best_overall[0]} ({best_overall[1]['overall']['solve_rate']:.1%})"
        )
        print(
            f"Best at imagination tasks: {best_imagination[0]} ({best_imagination[1]['imagination']['solve_rate']:.1%})"
        )
        print(
            f"Best at pattern tasks: {best_pattern[0]} ({best_pattern[1]['pattern_matching']['solve_rate']:.1%})"
        )

        # Check if imagination-focused approach helps
        for name, results in all_results.items():
            if "imagination" in name.lower() or "invention" in name.lower():
                imagination_rate = results["imagination"]["solve_rate"]
                pattern_rate = results["pattern_matching"]["solve_rate"]

                if imagination_rate > pattern_rate:
                    print(f"\n✓ {name} performs better on imagination tasks!")
                    print(
                        f"  This validates our hypothesis about imagination being different"
                    )
                else:
                    print(f"\n✗ {name} doesn't show imagination advantage yet")
                    print(f"  More work needed on the imagination mechanism")


def test_evaluation_suite():
    """Test the evaluation suite with mock solvers."""

    class RandomSolver:
        """Baseline: random guessing."""

        def solve(self, examples, test_input):
            # Just return random grid of same shape as first output
            output_shape = examples[0][1].shape
            return np.random.randint(0, 10, output_shape)

    class PatternMatchingSolver:
        """Simple pattern matching (always returns first training output)."""

        def solve(self, examples, test_input):
            # Naive: just return first training output
            return examples[0][1]

    # Create evaluator
    evaluator = ARCImaginationEvaluator()

    # Test on a few tasks
    print("Testing evaluation suite on real ARC tasks...")
    print("=" * 60)

    # Compare approaches
    solvers = {
        "Random": RandomSolver(),
        "Pattern Matching": PatternMatchingSolver(),
    }

    evaluator.compare_approaches(solvers)

    print("\n" + "=" * 60)
    print("EVALUATION SUITE READY")
    print("=" * 60)
    print("\nThis evaluation uses ONLY real ARC tasks, avoiding bias")
    print("Tasks are categorized based on our analysis of imagination requirements")
    print("Ready to test V7, V8, and new imagination-based approaches!")


if __name__ == "__main__":
    test_evaluation_suite()
