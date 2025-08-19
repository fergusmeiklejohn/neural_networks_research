#!/usr/bin/env python3
"""Analyze why example 3 doesn't show cross pattern."""

from utils.imports import setup_project_paths

setup_project_paths()

import json
from pathlib import Path

import numpy as np


def analyze_example3():
    """Analyze the third example in detail."""

    # Load task
    data_dir = Path("data/arc_agi_official/ARC-AGI/data/training")
    task_file = data_dir / "ae3edfdc.json"

    with open(task_file, "r") as f:
        task = json.load(f)

    # Get third example
    inp = np.array(task["train"][2]["input"])
    out = np.array(task["train"][2]["output"])

    print("Analyzing Example 3")
    print("=" * 60)

    print(f"Input shape: {inp.shape}")
    print(f"Output shape: {out.shape}")

    # Find differences
    diff_mask = inp != out
    changed_pixels = np.argwhere(diff_mask)

    print(f"\nChanged pixels: {len(changed_pixels)}")

    if len(changed_pixels) > 0:
        print("\nChanges:")
        for pos in changed_pixels[:20]:  # Show first 20
            i, j = pos
            print(f"  ({i:2},{j:2}): {inp[i,j]} -> {out[i,j]}")

    # Look for cross-like patterns in changes
    print("\n" + "=" * 60)
    print("Looking for cross patterns in output:")

    h, w = out.shape
    crosses_found = []

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            # Check if this position has cross-like changes
            center_preserved = (inp[i, j] == out[i, j]) and (inp[i, j] != 0)

            if center_preserved:
                # Check if arms changed
                north_changed = (i > 0) and (inp[i - 1, j] != out[i - 1, j])
                south_changed = (i < h - 1) and (inp[i + 1, j] != out[i + 1, j])
                west_changed = (j > 0) and (inp[i, j - 1] != out[i, j - 1])
                east_changed = (j < w - 1) and (inp[i, j + 1] != out[i, j + 1])

                if north_changed or south_changed or west_changed or east_changed:
                    print(f"\nPotential cross center at ({i},{j}), color {inp[i,j]}:")
                    print(
                        f"  North: {inp[i-1,j]} -> {out[i-1,j]} (changed: {north_changed})"
                    )
                    print(
                        f"  South: {inp[i+1,j]} -> {out[i+1,j]} (changed: {south_changed})"
                    )
                    print(
                        f"  West: {inp[i,j-1]} -> {out[i,j-1]} (changed: {west_changed})"
                    )
                    print(
                        f"  East: {inp[i,j+1]} -> {out[i,j+1]} (changed: {east_changed})"
                    )

                    # Count how many arms changed
                    arms_changed = sum(
                        [north_changed, south_changed, west_changed, east_changed]
                    )

                    if arms_changed >= 3:  # At least 3 arms for a cross
                        crosses_found.append((i, j, arms_changed))
                        print(
                            f"  ✓ This looks like a cross! ({arms_changed} arms changed)"
                        )

    print(f"\n" + "=" * 60)
    print(f"Summary: Found {len(crosses_found)} cross-like patterns")

    # Analyze what makes this different from examples 1 and 2
    print("\n" + "=" * 60)
    print("Why might the detector miss this?")

    # Check the strict cross detection criteria
    for i, j, arms in crosses_found:
        print(f"\nChecking cross at ({i},{j}):")

        # The current detector requires ALL 4 arms to change
        all_arms_changed = (
            (i > 0 and inp[i - 1, j] != out[i - 1, j])
            and (i < h - 1 and inp[i + 1, j] != out[i + 1, j])
            and (j > 0 and inp[i, j - 1] != out[i, j - 1])
            and (j < w - 1 and inp[i, j + 1] != out[i, j + 1])
        )

        if all_arms_changed:
            print("  ✓ All 4 arms changed - should be detected")
        else:
            print(f"  ✗ Only {arms} arms changed - current detector requires all 4")

    # Check for partial crosses (less strict criteria)
    print("\n" + "=" * 60)
    print("Testing with relaxed criteria (3+ arms):")

    relaxed_crosses = []
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            arms_changed = 0
            if i > 0 and inp[i - 1, j] != out[i - 1, j]:
                arms_changed += 1
            if i < h - 1 and inp[i + 1, j] != out[i + 1, j]:
                arms_changed += 1
            if j > 0 and inp[i, j - 1] != out[i, j - 1]:
                arms_changed += 1
            if j < w - 1 and inp[i, j + 1] != out[i, j + 1]:
                arms_changed += 1

            if arms_changed >= 3 and inp[i, j] != 0:
                relaxed_crosses.append((i, j, arms_changed))

    print(f"With relaxed criteria: {len(relaxed_crosses)} crosses found")
    for i, j, arms in relaxed_crosses[:5]:
        print(f"  ({i},{j}): {arms} arms, center color {inp[i,j]}")


if __name__ == "__main__":
    analyze_example3()
