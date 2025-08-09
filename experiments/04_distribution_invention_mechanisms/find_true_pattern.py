#!/usr/bin/env python3
"""Find the true pattern in task 05269061."""

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


def find_true_pattern():
    """Find the actual transformation pattern."""
    task_path = DATA_DIR / "05269061.json"
    task = load_arc_task(task_path)

    print("Finding True Pattern")
    print("=" * 60)

    # Maybe the pattern is based on the VALUES themselves, not their order
    # Let's check if there's a consistent rule based on value magnitude

    for i, example in enumerate(task["train"]):
        print(f"\nExample {i}:")
        inp = np.array(example["input"])
        out = np.array(example["output"])

        # Get unique values
        unique_vals = sorted(set(inp.flatten()) - {0})
        output_pattern = out.flatten()[: len(unique_vals)]

        print(f"  Sorted unique: {unique_vals}")
        print(f"  Output pattern: {list(output_pattern)}")

        # Check different orderings
        print("  Trying different orderings:")

        # Middle, max, min?
        if len(unique_vals) == 3:
            low, mid, high = unique_vals

            # Try different arrangements
            arrangements = {
                "low, mid, high": [low, mid, high],
                "low, high, mid": [low, high, mid],
                "mid, low, high": [mid, low, high],
                "mid, high, low": [mid, high, low],
                "high, low, mid": [high, low, mid],
                "high, mid, low": [high, mid, low],
            }

            for name, arr in arrangements.items():
                if arr == list(output_pattern):
                    print(f"    ✓ Pattern is: {name}")
                    break

    # Test
    print("\n" + "=" * 60)
    print("Test:")
    test = task["test"][0]
    test_input = np.array(test["input"])
    test_output = np.array(test["output"])

    unique_test = sorted(set(test_input.flatten()) - {0})
    expected = test_output.flatten()[: len(unique_test)]

    print(f"  Sorted unique: {unique_test}")
    print(f"  Expected: {list(expected)}")

    if len(unique_test) == 3:
        low, mid, high = unique_test
        print(f"  low={low}, mid={mid}, high={high}")
        print(f"  Expected pattern: {list(expected)}")

        # Check which arrangement
        if list(expected) == [mid, low, high]:
            print("  ✓ Pattern is: mid, low, high")
        else:
            print("  Pattern doesn't match mid, low, high")

    # Let's also check if it's based on position in the grid
    print("\n" + "=" * 60)
    print("Checking position-based pattern:")

    for i, example in enumerate(task["train"]):
        print(f"\nExample {i}:")
        inp = np.array(example["input"])

        # Find first occurrence of each unique value
        unique_vals = []
        seen = set()
        positions = {}

        for row in range(inp.shape[0]):
            for col in range(inp.shape[1]):
                val = inp[row, col]
                if val != 0 and val not in seen:
                    unique_vals.append(val)
                    positions[val] = (row, col)
                    seen.add(val)

        print(f"  Order of first appearance: {unique_vals}")
        for val in unique_vals:
            print(f"    {val} at position {positions[val]}")


if __name__ == "__main__":
    find_true_pattern()
