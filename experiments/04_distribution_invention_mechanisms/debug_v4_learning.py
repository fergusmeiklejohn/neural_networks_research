#!/usr/bin/env python3
"""Debug V4 learning on task 007bbfb7 to understand the issue."""

from utils.imports import setup_project_paths

setup_project_paths()

import json
from pathlib import Path

import numpy as np
from arc_dsl_enhanced import EnhancedDSLLibrary
from learnable_pattern_modifier import LearnablePatternModifier

# Get data path
DATA_DIR = (
    Path(__file__).parent.parent.parent
    / "data"
    / "arc_agi_official"
    / "ARC-AGI"
    / "data"
    / "training"
)


def debug_v4_learning():
    """Debug V4 learning process on 007bbfb7."""
    print("Debugging V4 Learning on Task 007bbfb7")
    print("=" * 80)

    # Load the task
    task_path = DATA_DIR / "007bbfb7.json"
    with open(task_path) as f:
        task = json.load(f)

    # Get training examples
    train_examples = [
        (np.array(ex["input"]), np.array(ex["output"])) for ex in task["train"]
    ]

    # Get test data
    test_input = np.array(task["test"][0]["input"])
    expected_output = np.array(task["test"][0]["output"])

    print(f"Test input:\n{test_input}")
    print(f"\nExpected output shape: {expected_output.shape}")

    # Apply simple tiling
    library = EnhancedDSLLibrary()
    tile_primitive = library.get_primitive("tile_pattern", scale=3)
    base_tiled = tile_primitive.execute(test_input)

    print(f"\nBase tiled output shape: {base_tiled.shape}")
    print(f"Base tiled output:\n{base_tiled}")

    # Learn modifications from training examples
    modifier = LearnablePatternModifier()

    # Collect learning examples
    learning_examples = []
    for input_grid, expected in train_examples:
        base = tile_primitive.execute(input_grid)
        if base.shape == expected.shape:
            learning_examples.append((input_grid, base, expected))

            # Show first example
            if len(learning_examples) == 1:
                print(f"\n--- Training Example 1 ---")
                print(f"Input shape: {input_grid.shape}")
                print(f"Base tiled shape: {base.shape}")
                print(f"Expected shape: {expected.shape}")

                # Show differences
                diff_mask = base != expected
                diff_count = np.sum(diff_mask)
                print(f"Differences: {diff_count} pixels")

                if diff_count > 0:
                    diff_positions = np.argwhere(diff_mask)
                    print(f"First 10 differences:")
                    for i, (r, c) in enumerate(diff_positions[:10]):
                        print(
                            f"  [{r}, {c}]: base={base[r, c]}, expected={expected[r, c]}"
                        )

    print(f"\n--- Learning from {len(learning_examples)} examples ---")

    # Learn rules
    rules = modifier.learn_from_examples(learning_examples, "tiling")

    print(f"Learned {len(rules)} rules:")
    for rule in rules:
        print(
            f"  - {rule.name} (conf: {rule.confidence:.2f}, examples: {rule.examples_matched})"
        )

    # Apply learned modifications to test
    print(f"\n--- Applying to Test Input ---")
    modified_output = modifier.apply_modifications(base_tiled, rules)

    print(f"Modified output shape: {modified_output.shape}")

    # Check accuracy
    is_correct = np.array_equal(modified_output, expected_output)
    if is_correct:
        print("✅ CORRECT!")
    else:
        accuracy = np.sum(modified_output == expected_output) / expected_output.size
        print(f"❌ Not correct. Pixel accuracy: {accuracy:.1%}")

        # Show differences
        diff_mask = modified_output != expected_output
        diff_positions = np.argwhere(diff_mask)
        print(f"\nDifferences: {len(diff_positions)} pixels")

        # Analyze pattern of differences
        print("\nFirst 20 differences:")
        for i, (r, c) in enumerate(diff_positions[:20]):
            print(
                f"  [{r:2d}, {c:2d}]: expected={expected_output[r, c]}, got={modified_output[r, c]}"
            )

        # Check if the differences follow a pattern
        print("\n--- Analyzing Difference Pattern ---")

        # Group by column
        col_diffs = {}
        for r, c in diff_positions:
            if c not in col_diffs:
                col_diffs[c] = []
            col_diffs[c].append(r)

        print(f"Columns with differences: {sorted(col_diffs.keys())}")

        # Are differences only in first 3 columns?
        if all(c < 3 for c in col_diffs.keys()):
            print("All differences are in first 3 columns")

            # What values are expected vs got?
            for c in range(3):
                if c in col_diffs:
                    print(f"\nColumn {c}:")
                    for r in col_diffs[c][:5]:
                        print(
                            f"  Row {r}: expected={expected_output[r, c]}, got={modified_output[r, c]}"
                        )

    # Show both outputs for comparison
    print("\n--- Expected Output ---")
    print(expected_output)

    print("\n--- Modified Output ---")
    print(modified_output)


if __name__ == "__main__":
    debug_v4_learning()
