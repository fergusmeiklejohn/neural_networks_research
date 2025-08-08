#!/usr/bin/env python3
"""Test V3 solver specifically on task 007bbfb7."""

from utils.imports import setup_project_paths

setup_project_paths()

import json
from pathlib import Path

import numpy as np
from enhanced_arc_solver_v3 import EnhancedARCSolverV3

# Get data path
DATA_DIR = (
    Path(__file__).parent.parent.parent
    / "data"
    / "arc_agi_official"
    / "ARC-AGI"
    / "data"
    / "training"
)


def test_on_007bbfb7():
    """Test V3 solver on task 007bbfb7."""
    print("Testing V3 Solver on Task 007bbfb7")
    print("=" * 60)

    # Load the task
    task_path = DATA_DIR / "007bbfb7.json"
    with open(task_path) as f:
        task = json.load(f)

    # Initialize V3 solver
    solver = EnhancedARCSolverV3(
        use_synthesis=True, force_synthesis_on_size_change=True, validate_patterns=True
    )

    # Get training examples
    train_examples = [
        (np.array(ex["input"]), np.array(ex["output"])) for ex in task["train"]
    ]

    # Get test input and expected output
    test_input = np.array(task["test"][0]["input"])
    expected_output = np.array(task["test"][0]["output"])

    print(f"Test input shape: {test_input.shape}")
    print(f"Expected output shape: {expected_output.shape}")
    print(f"Test input:\n{test_input}")

    # Solve
    print("\n" + "-" * 60)
    print("Running V3 Solver...")
    solution = solver.solve(train_examples, test_input)

    print(f"\nMethod used: {solution.method_used}")
    print(f"Confidence: {solution.confidence:.3f}")
    if solution.actual_confidence > 0:
        print(f"Validated confidence: {solution.actual_confidence:.3f}")

    print(f"\nSolution output shape: {solution.output_grid.shape}")

    # Check correctness
    if solution.output_grid.shape != expected_output.shape:
        print(
            f"\n❌ WRONG SHAPE! Expected {expected_output.shape}, got {solution.output_grid.shape}"
        )
    else:
        is_correct = np.array_equal(solution.output_grid, expected_output)
        if is_correct:
            print("\n✅ CORRECT! Task 007bbfb7 solved!")
        else:
            accuracy = (
                np.sum(solution.output_grid == expected_output) / expected_output.size
            )
            print(f"\n❌ Not correct. Pixel accuracy: {accuracy:.1%}")

            # Show first few differences
            diff_mask = solution.output_grid != expected_output
            diff_positions = np.argwhere(diff_mask)
            print(f"\nFirst 5 differences:")
            for i, (r, c) in enumerate(diff_positions[:5]):
                print(
                    f"  [{r}, {c}]: expected {expected_output[r, c]}, "
                    f"got {solution.output_grid[r, c]}"
                )

    print("\nExpected output:")
    print(expected_output)
    print("\nSolver output:")
    print(solution.output_grid)

    # Debug: check what patterns were detected
    if solution.perception_analysis:
        print("\n" + "-" * 60)
        print("Perception Analysis:")
        print(
            f"Transformation type: {solution.perception_analysis.get('transformation_type')}"
        )

        for pattern_type in [
            "arithmetic_patterns",
            "spatial_patterns",
            "conditional_patterns",
            "structural_patterns",
        ]:
            patterns = solution.perception_analysis.get(pattern_type, [])
            if patterns:
                print(f"\n{pattern_type}:")
                for p in patterns[:2]:
                    print(f"  - {p.name} (conf: {p.confidence:.2f})")


if __name__ == "__main__":
    test_on_007bbfb7()
