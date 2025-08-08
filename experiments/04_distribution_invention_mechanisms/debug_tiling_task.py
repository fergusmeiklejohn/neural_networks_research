#!/usr/bin/env python3
"""Debug the tiling task 007bbfb7 to understand why it's failing."""

from utils.imports import setup_project_paths

setup_project_paths()

import json
from pathlib import Path

import numpy as np
from arc_dsl_enhanced import EnhancedDSLLibrary
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


def debug_tiling_task():
    """Debug task 007bbfb7 specifically."""
    print("Debugging Tiling Task 007bbfb7")
    print("=" * 60)

    # Load the task
    task_path = DATA_DIR / "007bbfb7.json"
    with open(task_path) as f:
        task = json.load(f)

    # Get first training example
    train_ex = task["train"][0]
    input_grid = np.array(train_ex["input"])
    output_grid = np.array(train_ex["output"])

    print(f"Input shape: {input_grid.shape}")
    print(f"Output shape: {output_grid.shape}")
    print(f"Scale factor: {output_grid.shape[0] / input_grid.shape[0]}")

    print(f"\nInput:\n{input_grid}")
    print(f"\nExpected output:\n{output_grid}")

    # Try simple tiling
    print("\n" + "-" * 60)
    print("Testing simple TilePattern primitive:")
    library = EnhancedDSLLibrary()
    tile_primitive = library.get_primitive("tile_pattern", scale=3)
    tiled = tile_primitive.execute(input_grid)

    print(f"Tiled output shape: {tiled.shape}")
    print(f"Tiled output:\n{tiled}")

    # Check accuracy
    matches = np.sum(tiled == output_grid)
    total = output_grid.size
    accuracy = matches / total
    print(f"\nPixel accuracy: {matches}/{total} ({accuracy:.1%})")

    # Show differences
    if not np.array_equal(tiled, output_grid):
        print("\nDifferences (expected vs got):")
        diff_mask = tiled != output_grid
        diff_positions = np.argwhere(diff_mask)

        # Show first 10 differences
        for i, (r, c) in enumerate(diff_positions[:10]):
            print(
                f"  [{r:2d}, {c:2d}]: expected {output_grid[r, c]}, "
                f"got {tiled[r, c]}"
            )

        if len(diff_positions) > 10:
            print(f"  ... and {len(diff_positions) - 10} more differences")

    # Now test with the V3 solver
    print("\n" + "-" * 60)
    print("Testing V3 Solver:")

    solver = EnhancedARCSolverV3(
        use_synthesis=True, force_synthesis_on_size_change=True, validate_patterns=True
    )

    train_examples = [
        (np.array(ex["input"]), np.array(ex["output"])) for ex in task["train"]
    ]
    test_input = np.array(task["test"][0]["input"])

    solution = solver.solve(train_examples, test_input)

    print(f"Method used: {solution.method_used}")
    print(f"Confidence: {solution.confidence:.3f}")
    if solution.actual_confidence > 0:
        print(f"Validated confidence: {solution.actual_confidence:.3f}")

    # Check if it's correct
    expected_test = np.array(task["test"][0]["output"])
    is_correct = np.array_equal(solution.output_grid, expected_test)
    print(f"Correct: {'✓' if is_correct else '✗'}")

    if not is_correct:
        matches = np.sum(solution.output_grid == expected_test)
        total = expected_test.size
        print(f"Pixel accuracy: {matches}/{total} ({matches/total:.1%})")

    # Analyze what pattern was detected
    if solution.perception_analysis:
        print("\nDetected patterns:")
        for key in [
            "arithmetic_patterns",
            "spatial_patterns",
            "conditional_patterns",
            "structural_patterns",
        ]:
            patterns = solution.perception_analysis.get(key, [])
            if patterns:
                print(f"  {key}:")
                for p in patterns[:2]:
                    print(f"    - {p.name} (conf: {p.confidence:.2f})")


if __name__ == "__main__":
    debug_tiling_task()
