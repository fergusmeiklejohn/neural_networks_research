#!/usr/bin/env python3
"""Test V8 improvements on specific problem tasks."""

from utils.imports import setup_project_paths

setup_project_paths()

import json
from pathlib import Path

import numpy as np
from enhanced_arc_solver_v7 import EnhancedARCSolverV7
from enhanced_arc_solver_v8 import EnhancedARCSolverV8

# Get data path
DATA_DIR = (
    Path(__file__).parent.parent.parent
    / "data"
    / "arc_agi_official"
    / "ARC-AGI"
    / "data"
    / "training"
)


def load_arc_task(task_path: Path):
    """Load an ARC task from JSON file."""
    with open(task_path) as f:
        return json.load(f)


def test_v8_improvements():
    """Test V8 on tasks it should improve."""
    print("=" * 80)
    print("TESTING V8 IMPROVEMENTS")
    print("=" * 80)

    # Initialize solvers
    solver_v7 = EnhancedARCSolverV7(
        use_synthesis=True,
        use_position_learning=True,
        confidence_threshold=0.85,
    )

    solver_v8 = EnhancedARCSolverV8(
        use_synthesis=True,
        use_background_removal=True,
        use_enhanced_position=True,
        use_relative_position=True,
        use_error_correction=True,
        confidence_threshold=0.85,
    )

    # Focus on tasks V8 should improve
    target_tasks = [
        ("05269061.json", "Background removal pattern"),
        ("007bbfb7.json", "Enhanced position learning"),
        ("00d62c1b.json", "Object-relative positioning"),
        ("05f2a901.json", "High accuracy - error correction"),
    ]

    results_comparison = []

    for task_name, description in target_tasks:
        task_path = DATA_DIR / task_name
        if not task_path.exists():
            continue

        print(f"\n{'='*60}")
        print(f"Task: {task_name.replace('.json', '')} - {description}")
        print("=" * 60)

        task = load_arc_task(task_path)

        # Extract training examples
        train_examples = [
            (np.array(ex["input"]), np.array(ex["output"])) for ex in task["train"]
        ]

        # Test on first test example
        test_example = task["test"][0]
        test_input = np.array(test_example["input"])
        expected_output = np.array(test_example["output"])

        # Test V7
        print("\nV7 Solution:")
        solution_v7 = solver_v7.solve(train_examples, test_input)

        # Test V8
        print("\nV8 Solution:")
        solution_v8 = solver_v8.solve(train_examples, test_input)

        # Calculate accuracies
        v7_acc = 0.0
        v8_acc = 0.0

        if solution_v7.output_grid.shape == expected_output.shape:
            v7_acc = (
                np.sum(solution_v7.output_grid == expected_output)
                / expected_output.size
            )

        if solution_v8.output_grid.shape == expected_output.shape:
            v8_acc = (
                np.sum(solution_v8.output_grid == expected_output)
                / expected_output.size
            )

            # Check if perfect
            if np.array_equal(solution_v8.output_grid, expected_output):
                print("  🎉 PERFECT SOLUTION!")

        # Display results
        print(f"\nResults:")
        print(f"  V7: {v7_acc:.1%} ({solution_v7.method_used})")
        print(f"  V8: {v8_acc:.1%} ({solution_v8.method_used})")

        improvement = v8_acc - v7_acc
        if improvement > 0:
            print(f"  ✅ V8 improved by {improvement*100:.1f} percentage points!")
        elif improvement == 0:
            print(f"  ➖ No change")
        else:
            print(f"  ❌ V8 decreased by {-improvement*100:.1f} percentage points")

        results_comparison.append(
            {
                "task": task_name.replace(".json", ""),
                "description": description,
                "v7_accuracy": v7_acc,
                "v7_method": solution_v7.method_used,
                "v8_accuracy": v8_acc,
                "v8_method": solution_v8.method_used,
                "improvement": improvement,
                "perfect": v8_acc == 1.0,
            }
        )

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY OF V8 IMPROVEMENTS")
    print("=" * 80)

    print("\nTask-by-Task Comparison:")
    print("-" * 60)

    for result in results_comparison:
        status = "✅ SOLVED" if result["perfect"] else f"{result['v8_accuracy']:.1%}"
        print(f"\n{result['task']} ({result['description']}):")
        print(f"  V7: {result['v7_accuracy']:.1%} - {result['v7_method']}")
        print(f"  V8: {status} - {result['v8_method']}")
        if result["improvement"] > 0:
            print(f"  Improvement: +{result['improvement']*100:.1f}%")

    # Overall stats
    v7_avg = sum(r["v7_accuracy"] for r in results_comparison) / len(results_comparison)
    v8_avg = sum(r["v8_accuracy"] for r in results_comparison) / len(results_comparison)

    v7_solved = sum(1 for r in results_comparison if r["v7_accuracy"] == 1.0)
    v8_solved = sum(1 for r in results_comparison if r["v8_accuracy"] == 1.0)

    print("\n" + "-" * 60)
    print("Overall Performance:")
    print(f"  V7 Average: {v7_avg:.1%}")
    print(f"  V8 Average: {v8_avg:.1%}")
    print(f"  Improvement: +{(v8_avg - v7_avg)*100:.1f} percentage points")

    print(f"\n  V7 Tasks Solved: {v7_solved}/{len(results_comparison)}")
    print(f"  V8 Tasks Solved: {v8_solved}/{len(results_comparison)}")

    if v8_solved > v7_solved:
        print(f"\n🎉 V8 solved {v8_solved - v7_solved} additional task(s)!")

    # Test on all 8 tasks
    print("\n" + "=" * 80)
    print("TESTING ON ALL 8 TASKS")
    print("=" * 80)

    all_tasks = [
        "007bbfb7.json",  # Complex tiling
        "00d62c1b.json",  # Color pattern
        "05f2a901.json",  # Conditional logic
        "0a938d79.json",  # Simple pattern
        "0d3d703e.json",  # Complex spatial
        "08ed6ac7.json",  # Pattern task (V7 solved)
        "0ca9ddb6.json",  # Complex pattern
        "05269061.json",  # Background removal
    ]

    v8_results = []

    for task_name in all_tasks:
        task_path = DATA_DIR / task_name
        if not task_path.exists():
            continue

        task = load_arc_task(task_path)
        train_examples = [
            (np.array(ex["input"]), np.array(ex["output"])) for ex in task["train"]
        ]

        test_example = task["test"][0]
        test_input = np.array(test_example["input"])
        expected_output = np.array(test_example["output"])

        solution = solver_v8.solve(train_examples, test_input)

        accuracy = 0.0
        if solution.output_grid.shape == expected_output.shape:
            accuracy = (
                np.sum(solution.output_grid == expected_output) / expected_output.size
            )

        is_perfect = np.array_equal(solution.output_grid, expected_output)

        v8_results.append(
            {
                "task": task_name.replace(".json", ""),
                "accuracy": accuracy,
                "perfect": is_perfect,
                "method": solution.method_used,
            }
        )

        status = "✅" if is_perfect else f"{accuracy:.1%}"
        print(f"  {task_name.replace('.json', '')}: {status} ({solution.method_used})")

    # Final summary
    total_solved = sum(1 for r in v8_results if r["perfect"])
    avg_accuracy = sum(r["accuracy"] for r in v8_results) / len(v8_results)

    print(f"\nV8 Final Results:")
    print(f"  Tasks Solved: {total_solved}/{len(v8_results)}")
    print(f"  Average Accuracy: {avg_accuracy:.1%}")

    print("\n" + "=" * 80)
    print("V8 TESTING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    test_v8_improvements()
