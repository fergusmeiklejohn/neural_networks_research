#!/usr/bin/env python3
"""Debug solver output to understand why we're getting 0% accuracy."""

from utils.imports import setup_project_paths

setup_project_paths()

import json
from pathlib import Path

import numpy as np
from enhanced_arc_solver_v7 import EnhancedARCSolverV7
from hybrid_v7_structured_imagination import HybridV7StructuredImagination


def test_single_task():
    """Test on a single task to debug output."""

    # Load a single task
    data_dir = Path(
        "experiments/04_distribution_invention_mechanisms/data/arc_agi_official/ARC-AGI/data/evaluation"
    )

    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        return

    # Get first task
    task_files = sorted(data_dir.glob("*.json"))
    if not task_files:
        print("❌ No task files found")
        return

    task_file = task_files[0]
    print(f"Testing on task: {task_file.stem}")

    with open(task_file, "r") as f:
        task = json.load(f)

    # Extract examples
    examples = [(np.array(ex["input"]), np.array(ex["output"])) for ex in task["train"]]

    if not task["test"]:
        print("❌ No test cases")
        return

    test_input = np.array(task["test"][0]["input"])
    expected_output = np.array(task["test"][0]["output"])

    print(f"Input shape: {test_input.shape}")
    print(f"Expected output shape: {expected_output.shape}")
    print(f"Number of training examples: {len(examples)}")

    # Test V7 solver
    print("\n" + "=" * 60)
    print("Testing V7 Solver")
    print("=" * 60)

    v7_solver = EnhancedARCSolverV7(
        use_synthesis=True, use_position_learning=True, confidence_threshold=0.85
    )

    result = v7_solver.solve(examples, test_input)

    print(f"Result type: {type(result)}")
    print(f"Result attributes: {dir(result)}")

    if hasattr(result, "output_grid"):
        print(f"Output grid shape: {result.output_grid.shape}")
        print(f"Confidence: {result.confidence}")
        print(f"Method used: {result.method_used}")

        # Check if correct
        is_correct = np.array_equal(result.output_grid, expected_output)
        print(f"Correct: {is_correct}")

        if not is_correct:
            print("\nPredicted output:")
            print(result.output_grid)
            print("\nExpected output:")
            print(expected_output)
    else:
        print(f"Unexpected result format: {result}")

    # Test Hybrid solver
    print("\n" + "=" * 60)
    print("Testing Hybrid V7+Imagination Solver")
    print("=" * 60)

    hybrid_solver = HybridV7StructuredImagination(
        v7_confidence_threshold=0.7,
        imagination_trigger_threshold=0.6,
        max_hypotheses=30,
    )

    result2 = hybrid_solver.solve(examples, test_input, verbose=True)

    print(f"Result type: {type(result2)}")
    print(f"Result attributes: {dir(result2)}")

    if hasattr(result2, "output_grid"):
        print(f"Output grid shape: {result2.output_grid.shape}")
        print(f"Confidence: {result2.confidence}")
        print(f"Method used: {result2.method}")

        # Check if correct
        is_correct = np.array_equal(result2.output_grid, expected_output)
        print(f"Correct: {is_correct}")

        if not is_correct:
            print("\nPredicted output:")
            print(
                result2.output_grid[:10, :10]
                if result2.output_grid.shape[0] > 10
                else result2.output_grid
            )
            print("\nExpected output:")
            print(
                expected_output[:10, :10]
                if expected_output.shape[0] > 10
                else expected_output
            )
    else:
        print(f"Unexpected result format: {result2}")


if __name__ == "__main__":
    test_single_task()
