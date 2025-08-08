#!/usr/bin/env python3
"""Test the enhanced ARC solver V2 with all improvements.

This script tests the enhanced system on a subset of ARC tasks to verify
the improvements work correctly before full evaluation.
"""

from utils.imports import setup_project_paths

setup_project_paths()

import json
import time
from pathlib import Path
from typing import Dict

import numpy as np
from enhanced_arc_solver_v2 import EnhancedARCSolverV2

# Get data path
DATA_DIR = (
    Path(__file__).parent.parent.parent
    / "data"
    / "arc_agi_official"
    / "ARC-AGI"
    / "data"
    / "training"
)
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def load_arc_task(task_path: Path) -> Dict:
    """Load an ARC task from JSON file."""
    with open(task_path) as f:
        return json.load(f)


def test_on_sample_tasks():
    """Test enhanced solver on a small sample of tasks."""
    print("=" * 70)
    print("TESTING ENHANCED ARC SOLVER V2")
    print("=" * 70)

    # Initialize enhanced solver
    solver = EnhancedARCSolverV2(use_synthesis=True, adaptive_thresholds=True)

    # Test on a few specific tasks first
    print("\n" + "-" * 70)
    print("QUICK VALIDATION TESTS")
    print("-" * 70)

    # Test 1: Simple color arithmetic
    print("\nTest 1: Color Arithmetic (add 2 to all colors)")
    train_examples = [
        (np.array([[1, 2], [3, 4]]), np.array([[3, 4], [5, 6]])),
        (np.array([[2, 1], [4, 3]]), np.array([[4, 3], [6, 5]])),
    ]
    test_input = np.array([[1, 1], [2, 2]])

    solution = solver.solve(train_examples, test_input)
    expected = np.array([[3, 3], [4, 4]])

    print(f"Method: {solution.method_used}")
    print(f"Confidence: {solution.confidence:.2f}")
    print(f"Output:\n{solution.output_grid}")
    print(f"Expected:\n{expected}")
    print(f"Correct: {np.array_equal(solution.output_grid, expected)}")

    if solution.perception_analysis:
        print(f"Detected: {solution.perception_analysis['transformation_type']}")

    # Test 2: Conditional (if square, fill)
    print("\nTest 2: Conditional Logic (if square, fill with 5)")
    train_examples = [
        (
            np.array([[1, 1, 0], [1, 1, 0], [2, 2, 2]]),
            np.array([[5, 5, 0], [5, 5, 0], [2, 2, 2]]),
        ),
        (
            np.array([[3, 3, 0], [3, 3, 0], [0, 0, 0]]),
            np.array([[5, 5, 0], [5, 5, 0], [0, 0, 0]]),
        ),
    ]
    test_input = np.array([[4, 4, 0], [4, 4, 0], [1, 0, 0]])

    solution = solver.solve(train_examples, test_input)
    expected = np.array([[5, 5, 0], [5, 5, 0], [1, 0, 0]])

    print(f"Method: {solution.method_used}")
    print(f"Confidence: {solution.confidence:.2f}")
    print(f"Output:\n{solution.output_grid}")
    print(f"Expected:\n{expected}")
    print(f"Correct: {np.array_equal(solution.output_grid, expected)}")

    # Test 3: Spatial pattern (diagonal)
    print("\nTest 3: Spatial Pattern (diagonal)")
    train_examples = [
        (
            np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
            np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        ),
    ]
    test_input = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

    solution = solver.solve(train_examples, test_input)
    print(f"Method: {solution.method_used}")
    print(f"Confidence: {solution.confidence:.2f}")
    print(f"Output:\n{solution.output_grid}")

    # Now test on real ARC tasks
    print("\n" + "=" * 70)
    print("TESTING ON REAL ARC TASKS")
    print("=" * 70)

    # Check if ARC data exists
    if not DATA_DIR.exists():
        print(f"\nERROR: ARC data directory not found at {DATA_DIR}")
        print("Please run download_real_arc.py first to get the data.")
        return

    # Get sample of tasks
    task_files = sorted(DATA_DIR.glob("*.json"))[:5]  # Test on 5 tasks

    if not task_files:
        print(f"\nERROR: No task files found in {DATA_DIR}")
        return

    print(f"\nTesting on {len(task_files)} real ARC tasks...")

    results = []
    for i, task_path in enumerate(task_files, 1):
        print(f"\n" + "-" * 70)
        print(f"Task {i}/{len(task_files)}: {task_path.stem}")
        print("-" * 70)

        task = load_arc_task(task_path)

        # Extract training examples
        train_examples = [
            (np.array(ex["input"]), np.array(ex["output"])) for ex in task["train"]
        ]

        # Test on first test example
        test_example = task["test"][0]
        test_input = np.array(test_example["input"])

        # Get solution
        start_time = time.time()
        solution = solver.solve(train_examples, test_input)
        elapsed = time.time() - start_time

        print(f"Method used: {solution.method_used}")
        print(f"Confidence: {solution.confidence:.2f}")
        print(f"Time taken: {elapsed:.2f}s")

        if solution.perception_analysis:
            trans_type = solution.perception_analysis["transformation_type"]
            print(f"Detected pattern type: {trans_type}")

            # Show detected patterns
            all_patterns = (
                solution.perception_analysis.get("arithmetic_patterns", [])
                + solution.perception_analysis.get("conditional_patterns", [])
                + solution.perception_analysis.get("spatial_patterns", [])
                + solution.perception_analysis.get("structural_patterns", [])
            )

            if all_patterns:
                print("Detected patterns:")
                for pattern in all_patterns[:3]:  # Show top 3
                    print(f"  - {pattern.name} (conf: {pattern.confidence:.2f})")

        # Check if we have ground truth
        if "output" in test_example:
            expected_output = np.array(test_example["output"])
            is_correct = np.array_equal(solution.output_grid, expected_output)
            print(f"Correct: {'✓' if is_correct else '✗'}")

            results.append(
                {
                    "task": task_path.stem,
                    "correct": is_correct,
                    "method": solution.method_used,
                    "confidence": solution.confidence,
                    "time": elapsed,
                }
            )
        else:
            print("No ground truth available for test example")

    # Summary
    if results:
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)

        correct = sum(1 for r in results if r["correct"])
        total = len(results)
        accuracy = correct / total if total > 0 else 0

        print(f"\nAccuracy: {correct}/{total} ({accuracy:.1%})")

        # Method breakdown
        method_counts = {}
        for r in results:
            method = r["method"]
            if method not in method_counts:
                method_counts[method] = {"count": 0, "correct": 0}
            method_counts[method]["count"] += 1
            if r["correct"]:
                method_counts[method]["correct"] += 1

        print("\nMethod Usage:")
        for method, stats in method_counts.items():
            success_rate = (
                stats["correct"] / stats["count"] if stats["count"] > 0 else 0
            )
            print(
                f"  {method}: {stats['count']} uses, {stats['correct']} correct ({success_rate:.0%})"
            )

        avg_time = sum(r["time"] for r in results) / len(results)
        print(f"\nAverage time per task: {avg_time:.2f}s")

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
    print("\nNext step: Run full evaluation with evaluate_enhanced_system.py")


if __name__ == "__main__":
    test_on_sample_tasks()
