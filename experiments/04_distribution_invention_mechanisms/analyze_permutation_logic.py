#!/usr/bin/env python3
"""Analyze the permutation logic to understand the pattern."""

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


def analyze_permutation_logic():
    """Analyze how the permutation works across examples."""
    task_path = DATA_DIR / "05269061.json"
    task = load_arc_task(task_path)

    print("Analyzing Permutation Logic")
    print("=" * 60)

    # Analyze each training example
    for i, example in enumerate(task["train"]):
        print(f"\nTraining Example {i}:")
        inp = np.array(example["input"])
        out = np.array(example["output"])

        # Get unique values in the order they appear in input
        unique_in_order = []
        seen = set()
        for val in inp.flatten():
            if val != 0 and val not in seen:
                unique_in_order.append(val)
                seen.add(val)

        print(f"  Unique values (appearance order): {unique_in_order}")
        print(f"  Unique values (sorted): {sorted(unique_in_order)}")

        # Get the pattern from output
        output_pattern = out.flatten()[: len(unique_in_order)]
        print(f"  Output pattern: {list(output_pattern)}")

        # Find the mapping
        # If sorted values are [a, b, c] and output is [b, a, c]
        # Then indices are [1, 0, 2]
        sorted_vals = sorted(unique_in_order)
        if set(output_pattern) == set(sorted_vals):
            indices = [sorted_vals.index(v) for v in output_pattern]
            print(f"  Permutation indices (from sorted): {indices}")

            # Also check from appearance order
            if set(output_pattern) == set(unique_in_order):
                indices_appear = [unique_in_order.index(v) for v in output_pattern]
                print(f"  Permutation indices (from appearance): {indices_appear}")

    # Now the test
    print("\n" + "=" * 60)
    print("Test Example:")
    test = task["test"][0]
    test_input = np.array(test["input"])
    test_output = np.array(test["output"])

    # Get unique values in appearance order
    unique_test_order = []
    seen = set()
    for val in test_input.flatten():
        if val != 0 and val not in seen:
            unique_test_order.append(val)
            seen.add(val)

    print(f"  Unique values (appearance order): {unique_test_order}")
    print(f"  Unique values (sorted): {sorted(unique_test_order)}")

    expected_pattern = test_output.flatten()[: len(unique_test_order)]
    print(f"  Expected output pattern: {list(expected_pattern)}")

    # Try to find what permutation would work
    print("\n  Finding correct permutation:")
    sorted_test = sorted(unique_test_order)

    # Method 1: From sorted order
    if set(expected_pattern) == set(sorted_test):
        indices_sorted = [sorted_test.index(v) for v in expected_pattern]
        print(f"    From sorted → output indices: {indices_sorted}")
        # Apply these indices
        result = [sorted_test[i] for i in [1, 0, 2]]  # Using training pattern
        print(f"    Applying [1, 0, 2] to sorted: {result}")

    # Method 2: From appearance order
    if set(expected_pattern) == set(unique_test_order):
        indices_appear = [unique_test_order.index(v) for v in expected_pattern]
        print(f"    From appearance → output indices: {indices_appear}")
        # Apply these indices
        result = [unique_test_order[i] for i in [2, 1, 0]]  # Reversed?
        print(f"    Applying [2, 1, 0] to appearance: {result}")


if __name__ == "__main__":
    analyze_permutation_logic()
