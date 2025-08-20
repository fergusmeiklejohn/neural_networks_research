#!/usr/bin/env python3
"""Analyze the repeating pattern in task 05269061."""

from utils.imports import setup_project_paths

setup_project_paths()

import json
from pathlib import Path

import numpy as np

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


def find_repeating_pattern():
    """Find the repeating pattern in task 05269061."""
    task_path = DATA_DIR / "05269061.json"
    task = load_arc_task(task_path)

    print("Finding Repeating Pattern in Task 05269061")
    print("=" * 60)

    # Analyze each training example
    for i, example in enumerate(task["train"]):
        print(f"\nExample {i}:")
        inp = np.array(example["input"])
        out = np.array(example["output"])

        # Get non-zero values
        non_zero = inp[inp != 0]
        print(f"  Non-zero values: {non_zero}")

        # Get unique values (excluding 0)
        unique_values = sorted(set(inp.flatten()) - {0})
        print(f"  Unique non-zero colors: {unique_values}")

        # Check what pattern is in the output
        output_flat = out.flatten()

        # Find the shortest repeating pattern
        for pattern_len in range(1, len(unique_values) + 2):
            pattern = output_flat[:pattern_len]
            # Check if this pattern repeats
            is_repeating = True
            for j in range(pattern_len, len(output_flat), pattern_len):
                segment = output_flat[j : j + pattern_len]
                if len(segment) == pattern_len and not np.array_equal(segment, pattern):
                    is_repeating = False
                    break
                elif len(segment) < pattern_len and not np.array_equal(
                    segment, pattern[: len(segment)]
                ):
                    is_repeating = False
                    break

            if is_repeating:
                print(f"  ✓ Found repeating pattern of length {pattern_len}: {pattern}")
                print(f"    Pattern uses colors: {sorted(set(pattern))}")
                # Check if it matches unique values
                if set(pattern) == set(unique_values):
                    print(f"    Uses all unique colors!")
                break

    # Now test
    print("\n" + "=" * 60)
    print("Test Example:")
    test = task["test"][0]
    test_input = np.array(test["input"])
    test_output = np.array(test["output"])

    # Get unique non-zero colors
    unique_test = sorted(set(test_input.flatten()) - {0})
    print(f"  Unique non-zero colors in input: {unique_test}")

    # Check test output pattern
    test_flat = test_output.flatten()
    for pattern_len in range(1, 10):
        pattern = test_flat[:pattern_len]
        is_repeating = True
        for j in range(pattern_len, len(test_flat), pattern_len):
            segment = test_flat[j : j + pattern_len]
            if len(segment) == pattern_len and not np.array_equal(segment, pattern):
                is_repeating = False
                break
            elif len(segment) < pattern_len and not np.array_equal(
                segment, pattern[: len(segment)]
            ):
                is_repeating = False
                break

        if is_repeating:
            print(f"  Expected pattern of length {pattern_len}: {pattern}")
            print(f"    Uses colors: {sorted(set(pattern))}")
            break

    # Try to construct the pattern from unique values
    print("\n  Constructing pattern from unique values:")
    # The pattern seems to be a specific permutation of unique values
    # Let's try different permutations
    from itertools import permutations

    for perm in permutations(unique_test):
        test_pattern = np.array(perm)
        repeated = np.tile(test_pattern, (test_output.size // len(test_pattern)) + 1)[
            : test_output.size
        ]
        if np.array_equal(repeated, test_flat):
            print(f"  ✅ Found it! Pattern is permutation: {perm}")
            predicted = repeated.reshape(test_output.shape)
            print("\n  Predicted output:")
            print(predicted)
            print("\n  Matches expected? ", np.array_equal(predicted, test_output))
            return perm

    print("  Could not find exact permutation...")
    return None


if __name__ == "__main__":
    pattern = find_repeating_pattern()
