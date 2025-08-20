#!/usr/bin/env python3
"""Final check - maybe the pattern is in the input structure itself."""

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


def final_pattern_check():
    """Check if the pattern is encoded in the input structure."""
    task_path = DATA_DIR / "05269061.json"
    task = load_arc_task(task_path)

    print("Final Pattern Check - Input Structure Analysis")
    print("=" * 60)

    for i, example in enumerate(task["train"]):
        print(f"\nExample {i}:")
        inp = np.array(example["input"])
        out = np.array(example["output"])

        # Show the input grid
        print("  Input grid:")
        for row in inp:
            print(f"    {row}")

        # Get unique values and their positions
        unique_vals = sorted(set(inp.flatten()) - {0})
        output_pattern = list(out.flatten()[: len(unique_vals)])

        print(f"\n  Unique values: {unique_vals}")
        print(f"  Output pattern: {output_pattern}")

        # Count occurrences
        counts = {}
        for val in unique_vals:
            counts[val] = np.sum(inp == val)
        print(f"  Counts: {counts}")

        # Check first appearance order
        first_appear = []
        seen = set()
        for val in inp.flatten():
            if val != 0 and val not in seen:
                first_appear.append(val)
                seen.add(val)
        print(f"  First appearance order: {first_appear}")

        # Check if output matches first appearance
        if first_appear == output_pattern:
            print("  ✓ Output matches first appearance order!")

        # Check diagonal, row-by-row, column-by-column
        print("\n  Reading patterns:")

        # Row by row
        row_order = []
        seen = set()
        for row in inp:
            for val in row:
                if val != 0 and val not in seen:
                    row_order.append(val)
                    seen.add(val)
        print(f"    Row-by-row first seen: {row_order}")
        if row_order == output_pattern:
            print("      ✓ Matches output!")

        # Column by column
        col_order = []
        seen = set()
        for col in range(inp.shape[1]):
            for row in range(inp.shape[0]):
                val = inp[row, col]
                if val != 0 and val not in seen:
                    col_order.append(val)
                    seen.add(val)
        print(f"    Column-by-column first seen: {col_order}")
        if col_order == output_pattern:
            print("      ✓ Matches output!")

    # Test
    print("\n" + "=" * 60)
    print("Test Example:")
    test = task["test"][0]
    test_input = np.array(test["input"])
    test_output = np.array(test["output"])

    print("  Input grid:")
    for row in test_input:
        print(f"    {row}")

    unique_test = sorted(set(test_input.flatten()) - {0})
    expected = list(test_output.flatten()[: len(unique_test)])

    print(f"\n  Unique values: {unique_test}")
    print(f"  Expected output: {expected}")

    # Try row-by-row
    row_order = []
    seen = set()
    for row in test_input:
        for val in row:
            if val != 0 and val not in seen:
                row_order.append(val)
                seen.add(val)
    print(f"  Row-by-row first seen: {row_order}")
    if row_order == expected:
        print("    ✓ This would work!")

    # Try column-by-column
    col_order = []
    seen = set()
    for col in range(test_input.shape[1]):
        for row in range(test_input.shape[0]):
            val = test_input[row, col]
            if val != 0 and val not in seen:
                col_order.append(val)
                seen.add(val)
    print(f"  Column-by-column first seen: {col_order}")
    if col_order == expected:
        print("    ✓ This would work!")


if __name__ == "__main__":
    final_pattern_check()
