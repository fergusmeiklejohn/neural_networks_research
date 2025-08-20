#!/usr/bin/env python3
"""Test V4 solver with learnable modifications on task 007bbfb7."""

from utils.imports import setup_project_paths

setup_project_paths()

import json
from pathlib import Path

import numpy as np
from enhanced_arc_solver_v3 import EnhancedARCSolverV3
from enhanced_arc_solver_v4 import EnhancedARCSolverV4

# Get data path
DATA_DIR = (
    Path(__file__).parent.parent.parent
    / "data"
    / "arc_agi_official"
    / "ARC-AGI"
    / "data"
    / "training"
)


def test_v4_on_007bbfb7():
    """Test V4 solver on task 007bbfb7 and compare with V3."""
    print("Testing V4 Solver (Learnable) vs V3 (Hardcoded) on Task 007bbfb7")
    print("=" * 80)

    # Load the task
    task_path = DATA_DIR / "007bbfb7.json"
    with open(task_path) as f:
        task = json.load(f)

    # Initialize solvers
    solver_v3 = EnhancedARCSolverV3(
        use_synthesis=True, force_synthesis_on_size_change=True, validate_patterns=True
    )

    solver_v4 = EnhancedARCSolverV4(use_synthesis=True, learn_modifications=True)

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

    # Analyze first training example
    print("\nFirst training example:")
    print(f"  Input shape: {train_examples[0][0].shape}")
    print(f"  Output shape: {train_examples[0][1].shape}")
    print(
        f"  Scale factor: {train_examples[0][1].shape[0] / train_examples[0][0].shape[0]}"
    )

    # Test V3 (hardcoded modifications)
    print("\n" + "-" * 80)
    print("V3 Solver (Hardcoded Modifications):")
    solution_v3 = solver_v3.solve(train_examples, test_input)

    print(f"  Method: {solution_v3.method_used}")
    print(f"  Confidence: {solution_v3.confidence:.3f}")
    if solution_v3.actual_confidence > 0:
        print(f"  Validated confidence: {solution_v3.actual_confidence:.3f}")

    # Check V3 correctness
    if solution_v3.output_grid.shape != expected_output.shape:
        print(
            f"  ❌ WRONG SHAPE! Expected {expected_output.shape}, got {solution_v3.output_grid.shape}"
        )
        v3_correct = False
    else:
        v3_correct = np.array_equal(solution_v3.output_grid, expected_output)
        if v3_correct:
            print("  ✅ CORRECT!")
        else:
            accuracy = (
                np.sum(solution_v3.output_grid == expected_output)
                / expected_output.size
            )
            print(f"  ❌ Not correct. Pixel accuracy: {accuracy:.1%}")

    # Test V4 (learned modifications)
    print("\n" + "-" * 80)
    print("V4 Solver (Learned Modifications):")
    solution_v4 = solver_v4.solve(train_examples, test_input)

    print(f"  Method: {solution_v4.method_used}")
    print(f"  Confidence: {solution_v4.confidence:.3f}")

    # Check V4 correctness
    if solution_v4.output_grid.shape != expected_output.shape:
        print(
            f"  ❌ WRONG SHAPE! Expected {expected_output.shape}, got {solution_v4.output_grid.shape}"
        )
        v4_correct = False
    else:
        v4_correct = np.array_equal(solution_v4.output_grid, expected_output)
        if v4_correct:
            print("  ✅ CORRECT!")
        else:
            accuracy = (
                np.sum(solution_v4.output_grid == expected_output)
                / expected_output.size
            )
            print(f"  ❌ Not correct. Pixel accuracy: {accuracy:.1%}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"\nV3 (Hardcoded): {'✅ Correct' if v3_correct else '❌ Incorrect'}")
    print(f"V4 (Learned):   {'✅ Correct' if v4_correct else '❌ Incorrect'}")

    if v4_correct and not v3_correct:
        print("\n🎉 V4 IMPROVEMENT! Learned modifications work better than hardcoded!")
    elif v3_correct and not v4_correct:
        print("\n⚠️ V4 needs more work - hardcoded still performs better")
    elif v3_correct and v4_correct:
        print("\n✅ Both work! V4 successfully learned the pattern!")
    else:
        print("\n❌ Neither solver succeeded - more work needed")

    # Show what V4 learned
    if solution_v4.method_used == "geometric_learned":
        print("\n" + "-" * 80)
        print("What V4 Learned:")
        print("V4 analyzed the training examples and learned pattern modifications")
        print("without any hardcoded rules - a more generalizable approach!")

    # Compare outputs if different
    if not np.array_equal(solution_v3.output_grid, solution_v4.output_grid):
        print("\n" + "-" * 80)
        print("Output Differences:")

        if solution_v3.output_grid.shape == solution_v4.output_grid.shape:
            diff_mask = solution_v3.output_grid != solution_v4.output_grid
            diff_count = np.sum(diff_mask)
            print(f"  Number of different pixels: {diff_count}")

            if diff_count > 0 and diff_count < 20:
                diff_positions = np.argwhere(diff_mask)
                print("  First few differences:")
                for i, (r, c) in enumerate(diff_positions[:5]):
                    print(
                        f"    [{r}, {c}]: V3={solution_v3.output_grid[r, c]}, V4={solution_v4.output_grid[r, c]}"
                    )


if __name__ == "__main__":
    test_v4_on_007bbfb7()
