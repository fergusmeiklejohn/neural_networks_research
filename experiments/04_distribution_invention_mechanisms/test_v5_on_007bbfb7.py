#!/usr/bin/env python3
"""Test V5 solver with position-dependent modifications on task 007bbfb7."""

from utils.imports import setup_project_paths

setup_project_paths()

import json
from pathlib import Path

import numpy as np
from enhanced_arc_solver_v3 import EnhancedARCSolverV3
from enhanced_arc_solver_v4 import EnhancedARCSolverV4
from enhanced_arc_solver_v5 import EnhancedARCSolverV5

# Get data path
DATA_DIR = (
    Path(__file__).parent.parent.parent
    / "data"
    / "arc_agi_official"
    / "ARC-AGI"
    / "data"
    / "training"
)


def test_v5_on_007bbfb7():
    """Test V5 solver on task 007bbfb7 and compare with V3 and V4."""
    print("Testing V5 Solver (Position-Dependent) on Task 007bbfb7")
    print("=" * 80)

    # Load the task
    task_path = DATA_DIR / "007bbfb7.json"
    with open(task_path) as f:
        task = json.load(f)

    # Initialize solvers
    solver_v3 = EnhancedARCSolverV3(
        use_synthesis=False,  # Disable to speed up comparison
        force_synthesis_on_size_change=False,
        validate_patterns=True,
    )

    solver_v4 = EnhancedARCSolverV4(
        use_synthesis=False,
        learn_modifications=True,
    )

    solver_v5 = EnhancedARCSolverV5(
        use_synthesis=False,
        use_position_learning=True,
    )

    # Get training examples
    train_examples = [
        (np.array(ex["input"]), np.array(ex["output"])) for ex in task["train"]
    ]

    # Get test input and expected output
    test_input = np.array(task["test"][0]["input"])
    expected_output = np.array(task["test"][0]["output"])

    print(f"Number of training examples: {len(train_examples)}")
    print(f"Test input shape: {test_input.shape}")
    print(f"Expected output shape: {expected_output.shape}")

    # Test V3 (hardcoded modifications)
    print("\n" + "-" * 80)
    print("V3 Solver (Hardcoded Modifications):")
    solution_v3 = solver_v3.solve(train_examples, test_input)

    print(f"  Method: {solution_v3.method_used}")
    print(f"  Confidence: {solution_v3.confidence:.3f}")

    v3_correct = False
    if solution_v3.output_grid.shape != expected_output.shape:
        print(f"  ❌ Wrong shape!")
    else:
        v3_accuracy = (
            np.sum(solution_v3.output_grid == expected_output) / expected_output.size
        )
        v3_correct = np.array_equal(solution_v3.output_grid, expected_output)
        print(f"  Accuracy: {v3_accuracy:.1%} {'✅' if v3_correct else '❌'}")

    # Test V4 (simple learned modifications)
    print("\n" + "-" * 80)
    print("V4 Solver (Simple Learned Modifications):")
    solution_v4 = solver_v4.solve(train_examples, test_input)

    print(f"  Method: {solution_v4.method_used}")
    print(f"  Confidence: {solution_v4.confidence:.3f}")

    v4_correct = False
    if solution_v4.output_grid.shape != expected_output.shape:
        print(f"  ❌ Wrong shape!")
    else:
        v4_accuracy = (
            np.sum(solution_v4.output_grid == expected_output) / expected_output.size
        )
        v4_correct = np.array_equal(solution_v4.output_grid, expected_output)
        print(f"  Accuracy: {v4_accuracy:.1%} {'✅' if v4_correct else '❌'}")

    # Test V5 (position-dependent modifications)
    print("\n" + "-" * 80)
    print("V5 Solver (Position-Dependent Modifications):")
    solution_v5 = solver_v5.solve(train_examples, test_input)

    print(f"  Method: {solution_v5.method_used}")
    print(f"  Confidence: {solution_v5.confidence:.3f}")

    v5_correct = False
    if solution_v5.output_grid.shape != expected_output.shape:
        print(
            f"  ❌ Wrong shape! Expected {expected_output.shape}, got {solution_v5.output_grid.shape}"
        )
    else:
        v5_accuracy = (
            np.sum(solution_v5.output_grid == expected_output) / expected_output.size
        )
        v5_correct = np.array_equal(solution_v5.output_grid, expected_output)
        print(f"  Accuracy: {v5_accuracy:.1%} {'✅' if v5_correct else '❌'}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(
        f"\nV3 (Hardcoded):          {'✅ Correct' if v3_correct else f'❌ {v3_accuracy:.1%}' if 'v3_accuracy' in locals() else '❌ Wrong shape'}"
    )
    print(
        f"V4 (Simple Learning):    {'✅ Correct' if v4_correct else f'❌ {v4_accuracy:.1%}' if 'v4_accuracy' in locals() else '❌ Wrong shape'}"
    )
    print(
        f"V5 (Position Learning):  {'✅ Correct' if v5_correct else f'❌ {v5_accuracy:.1%}' if 'v5_accuracy' in locals() else '❌ Wrong shape'}"
    )

    if v5_correct and not (v3_correct or v4_correct):
        print("\n🎉 V5 BREAKTHROUGH! Position-dependent learning solved the task!")
    elif v5_correct:
        print("\n✅ V5 successfully learned the position-dependent pattern!")
    elif "v5_accuracy" in locals() and v5_accuracy > max(
        v3_accuracy if "v3_accuracy" in locals() else 0,
        v4_accuracy if "v4_accuracy" in locals() else 0,
    ):
        print(f"\n📈 V5 shows improvement! Better accuracy than V3 and V4.")
    else:
        print("\n🔧 V5 needs more work, but the approach is sound.")

    # Show V5 output for analysis
    if not v5_correct and solution_v5.output_grid.shape == expected_output.shape:
        print("\n" + "-" * 80)
        print("V5 Output Analysis:")
        diff_mask = solution_v5.output_grid != expected_output
        diff_positions = np.argwhere(diff_mask)

        if len(diff_positions) > 0:
            print(f"Number of incorrect pixels: {len(diff_positions)}")
            print("First 10 differences:")
            for i, (r, c) in enumerate(diff_positions[:10]):
                print(
                    f"  [{r}, {c}]: expected {expected_output[r, c]}, got {solution_v5.output_grid[r, c]}"
                )

        print("\nExpected output:")
        print(expected_output)
        print("\nV5 output:")
        print(solution_v5.output_grid)


if __name__ == "__main__":
    test_v5_on_007bbfb7()
