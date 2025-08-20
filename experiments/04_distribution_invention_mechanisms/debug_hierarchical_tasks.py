"""Deep dive into the two failed tasks to understand their hierarchical nature."""

import json
from pathlib import Path

import numpy as np


def analyze_task_68b16354():
    """Analyze task 68b16354 in detail."""
    data_dir = Path("data/arc_agi_official/ARC-AGI/data/training")
    with open(data_dir / "68b16354.json", "r") as f:
        task = json.load(f)

    print("=" * 60)
    print("Task 68b16354 - Detailed Analysis")
    print("=" * 60)

    for i, example in enumerate(task["train"][:3]):
        inp = np.array(example["input"])
        out = np.array(example["output"])

        print(f"\nExample {i+1}:")
        print(f"Input shape: {inp.shape}")
        print("Input:")
        print(inp)
        print(f"\nOutput shape: {out.shape}")
        print("Output:")
        print(out)

        # Analyze the pattern
        print("\nPattern analysis:")

        # This task seems to involve finding a specific color and creating output based on it
        unique_inp = np.unique(inp)
        unique_out = np.unique(out)
        print(f"Input colors: {unique_inp}")
        print(f"Output colors: {unique_out}")

        # Check if output is a fixed size
        print(f"Output size: {out.shape}")

        # Look for the pattern
        # It seems like the output is always 1x1 and contains a specific color
        if out.shape == (1, 1):
            print(f"Output is single pixel with value: {out[0, 0]}")

            # Find where this color appears in input
            positions = np.argwhere(inp == out[0, 0])
            if len(positions) > 0:
                print(f"This color appears at positions: {positions.tolist()}")

                # Check if it's the color that appears in a specific pattern
                # Could be: most frequent, least frequent, forms a shape, etc.
                color_counts = {
                    color: np.sum(inp == color) for color in unique_inp if color != 0
                }
                print(f"Color counts: {color_counts}")

                # Check which color appears exactly once or forms a specific pattern
                single_colors = [c for c, count in color_counts.items() if count == 1]
                if single_colors:
                    print(f"Colors appearing once: {single_colors}")
                    if out[0, 0] in single_colors:
                        print("✓ Output is the color that appears exactly once!")


def analyze_task_25ff71a9():
    """Analyze task 25ff71a9 in detail - the gravity task."""
    data_dir = Path("data/arc_agi_official/ARC-AGI/data/training")
    with open(data_dir / "25ff71a9.json", "r") as f:
        task = json.load(f)

    print("\n" + "=" * 60)
    print("Task 25ff71a9 - Detailed Analysis")
    print("=" * 60)

    for i, example in enumerate(task["train"]):
        inp = np.array(example["input"])
        out = np.array(example["output"])

        print(f"\nExample {i+1}:")
        print("Input:")
        print(inp)
        print("Output:")
        print(out)

        # This is NOT simple gravity - let's understand the actual pattern
        print("\nTransformation analysis:")

        # Track how each row changes
        for row_idx in range(3):
            inp_row = inp[row_idx]
            out_row = out[row_idx]
            if not np.array_equal(inp_row, out_row):
                print(f"Row {row_idx}: {inp_row} -> {out_row}")

        # It looks like rows are being shifted/moved
        # Check if it's a cyclic shift
        for shift in range(1, 3):
            shifted = np.roll(inp, shift, axis=0)
            if np.array_equal(shifted, out):
                print(f"✓ Output is input shifted down by {shift} rows (cyclic)")
                break

        # Check if pattern moves to specific position
        inp_nonzero_rows = [i for i in range(3) if np.any(inp[i] != 0)]
        out_nonzero_rows = [i for i in range(3) if np.any(out[i] != 0)]
        print(f"Input non-zero rows: {inp_nonzero_rows}")
        print(f"Output non-zero rows: {out_nonzero_rows}")

        # Check if the pattern is "move non-zero row to specific position"
        if len(inp_nonzero_rows) == 1 and len(out_nonzero_rows) == 1:
            moved_from = inp_nonzero_rows[0]
            moved_to = out_nonzero_rows[0]
            print(f"Pattern moved from row {moved_from} to row {moved_to}")

            # Check if it always moves to middle (row 1)
            if moved_to == 1:
                print("✓ Pattern: Move non-zero row to middle position!")


def analyze_hierarchical_structure():
    """Understand what makes these tasks hierarchical."""
    print("\n" + "=" * 60)
    print("HIERARCHICAL PATTERN INSIGHTS")
    print("=" * 60)

    print(
        """
Task 68b16354: Extract Unique Color
- Level 1: Identify all colors in grid
- Level 2: Count occurrences of each color
- Level 3: Find color with specific property (appears once)
- Level 4: Output that color as 1x1 grid

This is hierarchical because it requires:
1. Global analysis (count all colors)
2. Property detection (uniqueness)
3. Extraction based on global property

Task 25ff71a9: Move Pattern to Middle
- Level 1: Identify non-zero row
- Level 2: Determine target position (middle)
- Level 3: Move entire row as unit

This is hierarchical because it requires:
1. Pattern detection (which row has content)
2. Position reasoning (where should it go)
3. Transformation (move as complete unit)
    """
    )


if __name__ == "__main__":
    analyze_task_68b16354()
    analyze_task_25ff71a9()
    analyze_hierarchical_structure()
