#!/usr/bin/env python3
"""Test fixed solvers on ARC training set."""

from utils.imports import setup_project_paths

setup_project_paths()

import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
from enhanced_arc_solver_v7 import EnhancedARCSolverV7
from enhanced_arc_solver_v7_fixed import EnhancedARCSolverV7Fixed
from tqdm import tqdm


def load_training_tasks(limit: int = 20) -> List[Dict]:
    """Load ARC training tasks."""
    data_dir = Path(
        "experiments/04_distribution_invention_mechanisms/data/arc_agi_official/ARC-AGI/data/training"
    )

    if not data_dir.exists():
        # Try alternative path
        data_dir = Path("data/arc_agi_official/ARC-AGI/data/training")
        if not data_dir.exists():
            print(f"❌ Training data directory not found")
            return []

    tasks = []
    task_files = sorted(data_dir.glob("*.json"))[:limit]

    for task_file in task_files:
        with open(task_file, "r") as f:
            task = json.load(f)
            task["id"] = task_file.stem
            tasks.append(task)

    return tasks


def evaluate_solver(
    solver, solver_name: str, tasks: List[Dict], verbose: bool = False
) -> Dict:
    """Evaluate a solver on tasks."""

    print(f"\n{'='*60}")
    print(f"Testing {solver_name}")
    print(f"{'='*60}")

    results = {
        "solver_name": solver_name,
        "correct": 0,
        "total": 0,
        "methods_used": {},
        "confidences": [],
        "time_taken": 0.0,
        "size_change_tasks": {"correct": 0, "total": 0},
        "no_size_change_tasks": {"correct": 0, "total": 0},
    }

    for task in tqdm(tasks, desc=f"Evaluating {solver_name}"):
        # Extract examples
        examples = [
            (np.array(ex["input"]), np.array(ex["output"])) for ex in task["train"]
        ]

        # Test on each test case
        for test_case in task["test"]:
            test_input = np.array(test_case["input"])
            expected_output = np.array(test_case["output"])

            # Detect if size changes
            has_size_change = test_input.shape != expected_output.shape

            # Time the solve
            start_time = time.time()

            try:
                # Solve
                if hasattr(solver, "solve"):
                    result = solver.solve(examples, test_input, verbose=False)

                    # Extract output based on solver type
                    if hasattr(result, "output_grid"):
                        predicted = result.output_grid
                        method = (
                            result.method_used
                            if hasattr(result, "method_used")
                            else result.method
                        )
                        confidence = result.confidence
                    else:
                        predicted = result
                        method = "unknown"
                        confidence = 0.5
                else:
                    predicted = solver(examples, test_input)
                    method = "unknown"
                    confidence = 0.5

                # Check correctness
                is_correct = np.array_equal(predicted, expected_output)

                if is_correct:
                    results["correct"] += 1
                    if has_size_change:
                        results["size_change_tasks"]["correct"] += 1
                    else:
                        results["no_size_change_tasks"]["correct"] += 1

                # Track method
                results["methods_used"][method] = (
                    results["methods_used"].get(method, 0) + 1
                )
                results["confidences"].append(confidence)

                # Track size change performance
                if has_size_change:
                    results["size_change_tasks"]["total"] += 1
                else:
                    results["no_size_change_tasks"]["total"] += 1

                if verbose and is_correct:
                    print(f"  ✅ {task['id']}: {method} (conf: {confidence:.2f})")

            except Exception as e:
                if verbose:
                    print(f"  ❌ Error on {task['id']}: {e}")
                method = "error"
                results["methods_used"][method] = (
                    results["methods_used"].get(method, 0) + 1
                )
                results["confidences"].append(0.0)

                if has_size_change:
                    results["size_change_tasks"]["total"] += 1
                else:
                    results["no_size_change_tasks"]["total"] += 1

            results["time_taken"] += time.time() - start_time
            results["total"] += 1

    # Calculate accuracy
    results["accuracy"] = (
        results["correct"] / results["total"] if results["total"] > 0 else 0.0
    )
    results["avg_confidence"] = (
        np.mean(results["confidences"]) if results["confidences"] else 0.0
    )
    results["avg_time"] = (
        results["time_taken"] / results["total"] if results["total"] > 0 else 0.0
    )

    # Calculate size-specific accuracies
    if results["size_change_tasks"]["total"] > 0:
        results["size_change_accuracy"] = (
            results["size_change_tasks"]["correct"]
            / results["size_change_tasks"]["total"]
        )
    else:
        results["size_change_accuracy"] = 0.0

    if results["no_size_change_tasks"]["total"] > 0:
        results["no_size_change_accuracy"] = (
            results["no_size_change_tasks"]["correct"]
            / results["no_size_change_tasks"]["total"]
        )
    else:
        results["no_size_change_accuracy"] = 0.0

    return results


def print_results_comparison(all_results: Dict[str, Dict]):
    """Print comparison of results."""

    print("\n" + "=" * 80)
    print("RESULTS COMPARISON")
    print("=" * 80)

    # Header
    print(
        f"{'Solver':<25} {'Accuracy':>10} {'Size Change':>12} {'No Size':>10} {'Avg Conf':>10} {'Time':>8}"
    )
    print("-" * 80)

    # Sort by accuracy
    sorted_results = sorted(
        all_results.items(), key=lambda x: x[1]["accuracy"], reverse=True
    )

    for solver_name, res in sorted_results:
        print(
            f"{solver_name:<25} "
            f"{res['accuracy']*100:>9.1f}% "
            f"{res['size_change_accuracy']*100:>11.1f}% "
            f"{res['no_size_change_accuracy']*100:>9.1f}% "
            f"{res['avg_confidence']*100:>9.1f}% "
            f"{res['avg_time']:>7.3f}s"
        )

    print("=" * 80)

    # Method breakdown for each solver
    for solver_name, res in sorted_results:
        if res["methods_used"]:
            print(f"\n{solver_name} Method Usage:")
            total_methods = sum(res["methods_used"].values())
            for method, count in sorted(
                res["methods_used"].items(), key=lambda x: x[1], reverse=True
            ):
                percentage = (count / total_methods * 100) if total_methods > 0 else 0
                print(f"  {method:<20} {count:>5} ({percentage:>5.1f}%)")


def main():
    """Test fixed solvers on training set."""

    # Load training tasks
    print("Loading ARC training tasks...")
    tasks = load_training_tasks(limit=20)  # Start with 20 tasks

    if not tasks:
        print("❌ No tasks loaded")
        return

    print(f"Loaded {len(tasks)} tasks")

    # Count test cases
    total_test_cases = sum(len(task["test"]) for task in tasks)
    print(f"Total test cases: {total_test_cases}")

    # Initialize solvers
    print("\nInitializing solvers...")

    solvers = {
        "V7 Original": EnhancedARCSolverV7(
            use_synthesis=False,  # Disable synthesis for speed
            use_position_learning=True,
            confidence_threshold=0.85,
        ),
        "V7 Fixed": EnhancedARCSolverV7Fixed(
            use_synthesis=False,  # Disable synthesis for speed
            use_position_learning=True,
            confidence_threshold=0.85,
        ),
    }

    # Test each solver
    all_results = {}

    for solver_name, solver in solvers.items():
        results = evaluate_solver(solver, solver_name, tasks, verbose=False)
        all_results[solver_name] = results

        # Print summary
        print(f"\n{solver_name} Summary:")
        print(f"  Correct: {results['correct']}/{results['total']}")
        print(f"  Accuracy: {results['accuracy']*100:.1f}%")
        print(
            f"  Size change tasks: {results['size_change_accuracy']*100:.1f}% "
            f"({results['size_change_tasks']['correct']}/{results['size_change_tasks']['total']})"
        )
        print(
            f"  No size change: {results['no_size_change_accuracy']*100:.1f}% "
            f"({results['no_size_change_tasks']['correct']}/{results['no_size_change_tasks']['total']})"
        )

    # Print comparison
    print_results_comparison(all_results)

    # Check improvement
    if "V7 Original" in all_results and "V7 Fixed" in all_results:
        orig_acc = all_results["V7 Original"]["accuracy"]
        fixed_acc = all_results["V7 Fixed"]["accuracy"]

        if fixed_acc > orig_acc:
            improvement = (fixed_acc - orig_acc) * 100
            print(
                f"\n🎉 SUCCESS! Fixed V7 improved by {improvement:.1f} percentage points"
            )
            print(
                f"   Especially on size-change tasks: "
                f"{all_results['V7 Fixed']['size_change_accuracy']*100:.1f}% vs "
                f"{all_results['V7 Original']['size_change_accuracy']*100:.1f}%"
            )
        elif fixed_acc == orig_acc:
            print(f"\n🤔 No overall improvement, but check specific task types")
        else:
            print(f"\n⚠️ Fixed version performed worse - needs investigation")


if __name__ == "__main__":
    main()
