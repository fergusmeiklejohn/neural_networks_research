#!/usr/bin/env python3
"""Test task 05269061 to understand background removal pattern."""

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


def analyze_background_removal():
    """Analyze task 05269061 for background removal pattern."""
    task_path = DATA_DIR / "05269061.json"
    task = load_arc_task(task_path)

    print("Task 05269061: Background Removal Analysis")
    print("=" * 60)

    # Analyze training examples
    for i, example in enumerate(task["train"]):
        inp = np.array(example["input"])
        out = np.array(example["output"])

        print(f"\nExample {i}:")
        print(f"  Input shape: {inp.shape}")
        print(f"  Output shape: {out.shape}")
        print(f"  Input colors: {sorted(set(inp.flatten()))}")
        print(f"  Output colors: {sorted(set(out.flatten()))}")

        # Check if it's background removal
        # Background is typically 0
        non_zero_input = inp[inp != 0]
        non_zero_output = out.flatten()

        print(f"  Non-zero input cells: {len(non_zero_input)}")
        print(f"  Output total cells: {len(non_zero_output)}")

        # Check if output is just non-zero values
        if len(non_zero_output) == len(non_zero_input):
            print("  ✓ Output size matches non-zero input count!")

        # Show the actual transformation
        print("\n  Input grid:")
        print(inp)
        print("\n  Output grid:")
        print(out)

        # Analyze the mapping
        print("\n  Analysis:")
        if out.shape[0] * out.shape[1] == len(non_zero_input):
            print("    Output is compressed to only non-zero values!")
            # Check order preservation
            flat_non_zero = inp[inp != 0]
            flat_output = out.flatten()
            if np.array_equal(flat_non_zero, flat_output):
                print("    ✓ Order is preserved (reading left-to-right, top-to-bottom)")
            else:
                print("    Order might be different")

    # Test example
    print("\n" + "=" * 60)
    print("Test Example:")
    test = task["test"][0]
    test_input = np.array(test["input"])
    test_output = np.array(test["output"])

    print(f"  Input shape: {test_input.shape}")
    print(f"  Expected output shape: {test_output.shape}")
    print(f"  Input colors: {sorted(set(test_input.flatten()))}")
    print(f"  Expected output colors: {sorted(set(test_output.flatten()))}")

    non_zero_count = np.sum(test_input != 0)
    print(f"  Non-zero input cells: {non_zero_count}")
    print(f"  Output cells: {test_output.size}")

    if test_output.size == non_zero_count:
        print("  ✓ Confirms background removal pattern!")

    # Show what the output should be
    print("\n  Pattern Analysis:")
    non_zero_values = test_input[test_input != 0]
    print(f"  Non-zero values in order: {non_zero_values}")
    print(f"  Need to fill {test_output.size} cells with {len(non_zero_values)} values")

    # It seems to be repeating the pattern!
    if test_output.size > len(non_zero_values):
        # Create repeating pattern
        repetitions = test_output.size // len(non_zero_values) + 1
        repeated = np.tile(non_zero_values, repetitions)[: test_output.size]
        predicted = repeated.reshape(test_output.shape)

        print(f"\n  Trying repeating pattern (tile {len(non_zero_values)} values):")
        print("  Predicted output:")
        print(predicted)
        print("\n  Expected output:")
        print(test_output)

        if np.array_equal(predicted, test_output):
            print(
                "\n  ✅ Perfect match! Pattern: Extract non-zeros and tile to fill grid"
            )
        else:
            # Try different order or pattern
            print("\n  Testing if it's a specific subset being repeated...")
            # Check what's actually being repeated
            for start in range(len(non_zero_values)):
                for length in range(1, len(non_zero_values) - start + 1):
                    pattern = non_zero_values[start : start + length]
                    reps = test_output.size // len(pattern) + 1
                    test_pattern = np.tile(pattern, reps)[: test_output.size]
                    if np.array_equal(test_pattern, test_output.flatten()):
                        print(
                            f"  ✓ Found pattern! Repeating values {start}:{start+length}: {pattern}"
                        )
                        break


if __name__ == "__main__":
    analyze_background_removal()
