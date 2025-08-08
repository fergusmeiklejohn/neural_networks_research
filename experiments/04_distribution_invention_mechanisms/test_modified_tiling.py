#!/usr/bin/env python3
"""Test the modified tiling pattern on task 007bbfb7."""

from utils.imports import setup_project_paths

setup_project_paths()

import numpy as np
from arc_dsl_enhanced import EnhancedDSLLibrary

# Task 007bbfb7 first training example
input_grid = np.array([[0, 7, 7], [7, 7, 7], [0, 7, 7]])

expected_output = np.array(
    [
        [0, 0, 0, 0, 7, 7, 0, 7, 7],
        [0, 0, 0, 7, 7, 7, 7, 7, 7],
        [0, 0, 0, 0, 7, 7, 0, 7, 7],
        [0, 0, 0, 0, 7, 7, 0, 7, 7],
        [0, 0, 0, 7, 7, 7, 7, 7, 7],
        [0, 0, 0, 0, 7, 7, 0, 7, 7],
        [0, 0, 0, 0, 7, 7, 0, 7, 7],
        [0, 0, 0, 7, 7, 7, 7, 7, 7],
        [0, 0, 0, 0, 7, 7, 0, 7, 7],
    ]
)

print("Testing Modified Tiling Pattern")
print("=" * 60)

library = EnhancedDSLLibrary()

# Test simple tiling
print("\n1. Simple Tiling:")
simple_tile = library.get_primitive("tile_pattern", scale=3)
simple_output = simple_tile.execute(input_grid)
print(f"Output shape: {simple_output.shape}")
simple_accuracy = np.sum(simple_output == expected_output) / expected_output.size
print(f"Accuracy: {simple_accuracy:.1%}")

# Test modified tiling
print("\n2. Modified Tiling:")
modified_tile = library.get_primitive(
    "modified_tile_pattern", scale=3, modify_first=True
)
modified_output = modified_tile.execute(input_grid)
print(f"Output shape: {modified_output.shape}")
modified_accuracy = np.sum(modified_output == expected_output) / expected_output.size
print(f"Accuracy: {modified_accuracy:.1%}")

print("\nExpected output:")
print(expected_output)
print("\nModified tiling output:")
print(modified_output)

if np.array_equal(modified_output, expected_output):
    print("\n✅ PERFECT MATCH!")
else:
    print(f"\n❌ Not exact match ({modified_accuracy:.1%} accuracy)")

    # Show differences
    diff_mask = modified_output != expected_output
    if np.any(diff_mask):
        diff_positions = np.argwhere(diff_mask)
        print(f"\nDifferences: {len(diff_positions)} pixels")
        for i, (r, c) in enumerate(diff_positions[:5]):
            print(
                f"  [{r}, {c}]: expected {expected_output[r, c]}, got {modified_output[r, c]}"
            )
        if len(diff_positions) > 5:
            print(f"  ... and {len(diff_positions) - 5} more")
