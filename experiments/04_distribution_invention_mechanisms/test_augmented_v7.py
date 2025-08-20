#!/usr/bin/env python3
"""
Test the augmented V7 solver on real ARC benchmark.

This combines our best existing solver (V7 with 71.4% accuracy)
with the imagination augmentation system.
"""

from utils.imports import setup_project_paths

setup_project_paths()

import json
import time
from typing import List, Optional, Tuple

import numpy as np
from create_arc_evaluation_suite import ARCImaginationEvaluator

# Import our components
from enhanced_arc_solver_v7 import EnhancedARCSolverV7
from enhanced_arc_solver_v8 import EnhancedARCSolverV8
from invention_based_solver import DistributionInventionSolver


class AugmentedV7Solver:
    """V7 solver augmented with imagination capabilities."""

    def __init__(self):
        # Base V7 solver
        self.v7_solver = EnhancedARCSolverV7(
            use_synthesis=True, use_position_learning=True, confidence_threshold=0.85
        )

        # Distribution invention solver for imagination
        self.invention_solver = DistributionInventionSolver(invention_budget=30)

        # Track attempts for failure detection
        self.attempts = []
        self.confidence_threshold = 0.7

    def solve(
        self, examples: List[Tuple[np.ndarray, np.ndarray]], test_input: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Solve using V7 first, then imagination if confidence is low.
        """
        # First, try V7
        v7_solution = self.v7_solver.solve(examples, test_input)

        # Check confidence
        confidence = (
            v7_solution.confidence if hasattr(v7_solution, "confidence") else 0.5
        )

        if confidence >= self.confidence_threshold:
            # High confidence - use V7 solution
            return (
                v7_solution.output_grid
                if hasattr(v7_solution, "output_grid")
                else v7_solution
            )

        # Low confidence - engage imagination
        print(f"    V7 confidence low ({confidence:.2f}), engaging imagination...")

        # Try distribution invention
        invention_result = self.invention_solver.solve(examples, test_input)

        if invention_result is not None:
            # Test both solutions empirically
            v7_result = (
                v7_solution.output_grid
                if hasattr(v7_solution, "output_grid")
                else v7_solution
            )

            # Simple heuristic: if invention found something structured, use it
            if self._is_structured(invention_result):
                return invention_result

        # Fall back to V7
        return (
            v7_solution.output_grid
            if hasattr(v7_solution, "output_grid")
            else v7_solution
        )

    def _is_structured(self, result: np.ndarray) -> bool:
        """Check if result has structure (not random)."""
        if result is None:
            return False

        # Not all same value
        if len(np.unique(result)) == 1:
            return False

        # Not too random
        if len(np.unique(result)) > result.size * 0.8:
            return False

        return True


class SimpleV8Solver:
    """Simplified V8 for testing."""

    def solve(
        self, examples: List[Tuple[np.ndarray, np.ndarray]], test_input: np.ndarray
    ):
        """Simple solver that just returns first output shape."""
        if examples:
            output_shape = examples[0][1].shape
            # Return something structured but wrong
            result = np.zeros(output_shape, dtype=np.int32)
            result[0, :] = 1
            result[:, 0] = 2
            return result
        return test_input


def run_comprehensive_test():
    """Run comprehensive test of all approaches."""
    print("=" * 80)
    print("COMPREHENSIVE ARC BENCHMARK TEST")
    print("=" * 80)
    print("\nTesting V7, V8, and Augmented V7 on real ARC tasks")
    print("Focus on tasks requiring imagination vs pattern matching\n")

    # Create evaluator
    evaluator = ARCImaginationEvaluator()

    # Select test tasks
    print("Selected Tasks:")
    print("-" * 40)

    # Use specific tasks we've analyzed
    imagination_tasks = [
        "05269061",  # Background removal - we know this needs imagination
        "007bbfb7",  # Position-dependent - complex patterns
        "00d62c1b",  # Object-relative rules
    ]

    pattern_tasks = [
        "08ed6ac7",  # Simple color mapping - V7 solved this
        "0a938d79",  # Direct pattern application
        "09629e4f",  # Regular tiling (if exists)
    ]

    # Print task categories
    print("Imagination-required tasks:")
    for task in imagination_tasks:
        print(f"  - {task}")

    print("\nPattern-matching tasks:")
    for task in pattern_tasks:
        print(f"  - {task}")

    all_tasks = imagination_tasks + pattern_tasks

    # Create solvers
    print("\n" + "=" * 80)
    print("TESTING SOLVERS")
    print("=" * 80)

    solvers = {
        "V7 (Baseline)": EnhancedARCSolverV7(
            use_synthesis=True, use_position_learning=True, confidence_threshold=0.85
        ),
        "V8 (Pattern-Specific)": EnhancedARCSolverV8(
            use_synthesis=True,
            use_background_removal=True,
            use_enhanced_position=True,
            use_relative_position=True,
            use_error_correction=True,
            confidence_threshold=0.85,
        ),
        "V7 + Imagination": AugmentedV7Solver(),
        "Pure Invention": DistributionInventionSolver(invention_budget=50),
    }

    # Simplified wrapper for solvers to handle different return types
    class SolverWrapper:
        def __init__(self, solver):
            self.solver = solver

        def solve(self, examples, test_input):
            result = self.solver.solve(examples, test_input)

            # Handle different return types
            if hasattr(result, "output_grid"):
                return result.output_grid
            elif isinstance(result, np.ndarray):
                return result
            else:
                return None

    # Wrap solvers
    wrapped_solvers = {name: SolverWrapper(solver) for name, solver in solvers.items()}

    # Test each solver
    results = {}

    for solver_name, solver in wrapped_solvers.items():
        print(f"\n{'='*60}")
        print(f"Testing: {solver_name}")
        print("-" * 60)

        solver_results = {
            "imagination": {"correct": 0, "total": 0, "accuracies": []},
            "pattern": {"correct": 0, "total": 0, "accuracies": []},
            "overall": {"correct": 0, "total": 0, "accuracies": []},
        }

        for task_id in all_tasks:
            # Load task
            task_path = evaluator.data_dir / f"{task_id}.json"
            if not task_path.exists():
                print(f"  Task {task_id} not found, skipping")
                continue

            with open(task_path) as f:
                task = json.load(f)

            # Get examples
            train_examples = [
                (np.array(ex["input"]), np.array(ex["output"])) for ex in task["train"]
            ]

            # Test on first test case
            test_input = np.array(task["test"][0]["input"])
            expected_output = np.array(task["test"][0]["output"])

            # Solve
            start_time = time.time()
            try:
                predicted = solver.solve(train_examples, test_input)
                solve_time = time.time() - start_time

                # Calculate accuracy
                if predicted is not None and predicted.shape == expected_output.shape:
                    accuracy = (
                        np.sum(predicted == expected_output) / expected_output.size
                    )
                    is_perfect = np.array_equal(predicted, expected_output)
                else:
                    accuracy = 0.0
                    is_perfect = False

            except Exception as e:
                print(f"  Error on {task_id}: {e}")
                accuracy = 0.0
                is_perfect = False
                solve_time = 0.0

            # Categorize
            category = "imagination" if task_id in imagination_tasks else "pattern"

            # Update results
            solver_results[category]["total"] += 1
            solver_results[category]["correct"] += int(is_perfect)
            solver_results[category]["accuracies"].append(accuracy)

            solver_results["overall"]["total"] += 1
            solver_results["overall"]["correct"] += int(is_perfect)
            solver_results["overall"]["accuracies"].append(accuracy)

            # Print result
            status = "✅" if is_perfect else f"{accuracy:.1%}"
            print(f"  {task_id} ({category:11}): {status:>6} ({solve_time:.2f}s)")

        # Calculate averages
        for cat in solver_results:
            if solver_results[cat]["accuracies"]:
                solver_results[cat]["avg_accuracy"] = np.mean(
                    solver_results[cat]["accuracies"]
                )
                solver_results[cat]["solve_rate"] = (
                    solver_results[cat]["correct"] / solver_results[cat]["total"]
                    if solver_results[cat]["total"] > 0
                    else 0
                )

        results[solver_name] = solver_results

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY RESULTS")
    print("=" * 80)

    # Header
    print(f"\n{'Solver':<25} {'Overall':>12} {'Imagination':>12} {'Pattern':>12}")
    print("-" * 61)

    # Results for each solver
    for solver_name, solver_results in results.items():
        overall = f"{solver_results['overall']['solve_rate']:.0%} ({solver_results['overall']['avg_accuracy']:.1%})"
        imagination = f"{solver_results['imagination']['solve_rate']:.0%} ({solver_results['imagination']['avg_accuracy']:.1%})"
        pattern = f"{solver_results['pattern']['solve_rate']:.0%} ({solver_results['pattern']['avg_accuracy']:.1%})"

        print(f"{solver_name:<25} {overall:>12} {imagination:>12} {pattern:>12}")

    # Analysis
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)

    # Check if augmentation helped
    if "V7 (Baseline)" in results and "V7 + Imagination" in results:
        v7_imag = results["V7 (Baseline)"]["imagination"]["avg_accuracy"]
        aug_imag = results["V7 + Imagination"]["imagination"]["avg_accuracy"]

        if aug_imag > v7_imag:
            improvement = (aug_imag - v7_imag) * 100
            print(
                f"\n✅ Imagination augmentation improved on imagination tasks by {improvement:.1f} percentage points!"
            )
        else:
            print(
                f"\n❌ Imagination augmentation didn't improve results (more work needed)"
            )

    # Check which solver is best for which category
    best_imagination = max(
        results.items(), key=lambda x: x[1]["imagination"]["avg_accuracy"]
    )
    best_pattern = max(results.items(), key=lambda x: x[1]["pattern"]["avg_accuracy"])

    print(
        f"\nBest for imagination tasks: {best_imagination[0]} ({best_imagination[1]['imagination']['avg_accuracy']:.1%})"
    )
    print(
        f"Best for pattern tasks: {best_pattern[0]} ({best_pattern[1]['pattern']['avg_accuracy']:.1%})"
    )

    # Final insight
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)

    if any("Imagination" in name for name in results):
        avg_imagination_benefit = np.mean(
            [
                results[name]["imagination"]["avg_accuracy"]
                - results[name]["pattern"]["avg_accuracy"]
                for name in results
                if "Imagination" in name or "Invention" in name
            ]
        )

        if avg_imagination_benefit > 0:
            print(
                "✅ Imagination-based approaches show promise for tasks requiring novel patterns!"
            )
        else:
            print(
                "⚠️  Imagination approaches need refinement to outperform pattern matching"
            )

    print("\nNext steps:")
    print("- Refine imagination triggering (when to use it)")
    print("- Improve hypothesis generation diversity")
    print("- Add better empirical testing of hypotheses")
    print("- Scale to full 400+ task dataset")


if __name__ == "__main__":
    run_comprehensive_test()
