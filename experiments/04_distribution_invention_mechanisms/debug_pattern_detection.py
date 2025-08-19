#!/usr/bin/env python3
"""Debug why the ae3edfdc cross pattern isn't being detected.

This script adds detailed logging to understand where pattern detection fails.
"""

from utils.imports import setup_project_paths

setup_project_paths()

import json
from pathlib import Path

import numpy as np
from automated_primitive_discovery import PrimitiveDiscoverer
from compositional_dsl import ExecutionContext
from cross_pattern_primitive import FormCrossPattern


def analyze_cross_pattern_manually(inp, out):
    """Manually analyze what makes the cross pattern."""
    print("\n=== Manual Cross Pattern Analysis ===")
    print(f"Input shape: {inp.shape}, Output shape: {out.shape}")

    # Find differences
    diff_mask = inp != out
    diff_positions = np.argwhere(diff_mask)

    print(f"Number of changed pixels: {len(diff_positions)}")

    if len(diff_positions) > 0:
        print("\nChanged positions:")
        for pos in diff_positions[:10]:  # Show first 10
            i, j = pos
            print(f"  ({i},{j}): {inp[i,j]} -> {out[i,j]}")

    # Analyze patterns in changes
    print("\n=== Pattern Analysis ===")

    # Look for cross shapes
    h, w = inp.shape
    cross_patterns_found = []

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            # Check if this could be a cross center
            if inp[i, j] != out[i, j]:
                continue

            # Check if surrounded by changes in cross pattern
            if (
                i > 0
                and inp[i - 1, j] != out[i - 1, j]
                and i < h - 1
                and inp[i + 1, j] != out[i + 1, j]
                and j > 0
                and inp[i, j - 1] != out[i, j - 1]
                and j < w - 1
                and inp[i, j + 1] != out[i, j + 1]
            ):
                cross_patterns_found.append((i, j))
                print(f"Potential cross center at ({i},{j}), color {inp[i,j]}")
                print(f"  North: {inp[i-1,j]} -> {out[i-1,j]}")
                print(f"  South: {inp[i+1,j]} -> {out[i+1,j]}")
                print(f"  West: {inp[i,j-1]} -> {out[i,j-1]}")
                print(f"  East: {inp[i,j+1]} -> {out[i,j+1]}")

    print(f"\nFound {len(cross_patterns_found)} potential cross patterns")

    # Analyze what triggers cross formation
    print("\n=== Trigger Analysis ===")

    # Look at input patterns that lead to crosses
    for i, j in cross_patterns_found[:3]:  # Analyze first 3
        print(f"\nCross at ({i},{j}):")
        center_color = inp[i, j]
        print(f"  Center color: {center_color}")

        # Check horizontal line
        horizontal_colors = []
        for jj in range(w):
            if jj != j:
                horizontal_colors.append(inp[i, jj])
        print(f"  Horizontal line colors: {horizontal_colors}")

        # Check vertical line
        vertical_colors = []
        for ii in range(h):
            if ii != i:
                vertical_colors.append(inp[ii, j])
        print(f"  Vertical line colors: {vertical_colors}")

        # What color fills the cross?
        if i > 0:
            cross_color = out[i - 1, j]
            print(f"  Cross fill color: {cross_color}")


def test_pattern_detection_with_logging():
    """Test the PrimitiveDiscoverer with detailed logging."""

    # Load task
    data_dir = Path("data/arc_agi_official/ARC-AGI/data/training")
    task_file = data_dir / "ae3edfdc.json"

    with open(task_file, "r") as f:
        task = json.load(f)

    # Get examples
    train_examples = [
        (np.array(ex["input"]), np.array(ex["output"])) for ex in task["train"]
    ]

    print("=" * 60)
    print("Testing Pattern Detection on ae3edfdc")
    print("=" * 60)

    # First, verify the manual primitive works
    print("\n1. Testing manual FormCrossPattern primitive:")
    primitive = FormCrossPattern()

    for i, (inp, expected) in enumerate(train_examples[:1]):  # Test first example
        context = ExecutionContext(input_grid=inp.copy(), current_grid=inp.copy())
        result_context = primitive.execute(context)
        result = result_context.current_grid
        accuracy = np.mean(result == expected)
        print(f"   Example {i+1} accuracy: {accuracy:.1%}")

        # Manually analyze the pattern
        analyze_cross_pattern_manually(inp, expected)

    print("\n" + "=" * 60)
    print("2. Testing automated discovery:")

    # Create discoverer with verbose mode
    discoverer = PrimitiveDiscoverer(verbose=True)

    # Monkey-patch to add more logging
    original_extract = discoverer._extract_patterns

    def logged_extract(examples):
        print("\n=== Pattern Extraction Debug ===")
        patterns = []

        for idx, (inp, out) in enumerate(examples):
            print(f"\nExample {idx+1}:")
            print(f"  Input shape: {inp.shape}")
            print(f"  Output shape: {out.shape}")
            print(f"  Input colors: {np.unique(inp)}")
            print(f"  Output colors: {np.unique(out)}")

            # Try original extraction with logging
            example_patterns = original_extract([(inp, out)])
            print(f"  Patterns found: {len(example_patterns)}")

            for p in example_patterns:
                print(f"    - Type: {p['type']}")
                if "data" in p and p["data"]:
                    print(
                        f"      Data keys: {list(p['data'].keys()) if isinstance(p['data'], dict) else 'non-dict'}"
                    )

            patterns.extend(example_patterns)

        return patterns

    discoverer._extract_patterns = logged_extract

    # Also patch spatial analysis
    original_spatial = discoverer._analyze_spatial_pattern

    def logged_spatial(inp, out):
        print("\n    === Spatial Pattern Analysis ===")
        print(f"    Checking for spatial patterns...")

        # Check for crosses specifically
        h, w = inp.shape
        crosses_found = False

        for i in range(1, h - 1):
            for j in range(1, w - 1):
                # Simple cross detection: center stays, arms change
                if (
                    inp[i, j] in [1, 2]  # Center colors from FormCrossPattern
                    and out[i, j] == inp[i, j]
                ):  # Center preserved
                    # Check if arms changed
                    arms_changed = (
                        (i > 0 and inp[i - 1, j] != out[i - 1, j])
                        or (i < h - 1 and inp[i + 1, j] != out[i + 1, j])
                        or (j > 0 and inp[i, j - 1] != out[i, j - 1])
                        or (j < w - 1 and inp[i, j + 1] != out[i, j + 1])
                    )

                    if arms_changed:
                        crosses_found = True
                        print(f"    Potential cross at ({i},{j})")

        if crosses_found:
            print("    ✓ Cross patterns detected!")
        else:
            print("    ✗ No cross patterns found")

        # Call original
        result = original_spatial(inp, out)
        if result:
            print(f"    Original spatial returned: {result}")
        return result

    discoverer._analyze_spatial_pattern = logged_spatial

    # Try discovery
    print("\nAttempting automatic discovery...")
    discovered = discoverer.discover_primitive("ae3edfdc", train_examples)

    if discovered:
        print(f"\n✅ Successfully discovered primitive!")
        print(f"Code:\n{discovered}")
    else:
        print(f"\n❌ Failed to discover primitive")
        print("\nDiagnosis: The pattern extraction or matching is failing")
        print("Need to fix the _analyze_spatial_pattern method")


if __name__ == "__main__":
    test_pattern_detection_with_logging()
