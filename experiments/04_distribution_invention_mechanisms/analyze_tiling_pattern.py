#!/usr/bin/env python3
"""Analyze the tiling pattern in task 00576224."""

import numpy as np

# The input is:
input_grid = np.array([[3, 2], [7, 8]])

# The expected output is:
expected = np.array(
    [
        [3, 2, 3, 2, 3, 2],
        [7, 8, 7, 8, 7, 8],
        [2, 3, 2, 3, 2, 3],
        [8, 7, 8, 7, 8, 7],
        [3, 2, 3, 2, 3, 2],
        [7, 8, 7, 8, 7, 8],
    ]
)

print("Input (2x2):")
print(input_grid)
print("\nExpected output (6x6):")
print(expected)

# Analyze the pattern
print("\n" + "=" * 60)
print("PATTERN ANALYSIS")
print("=" * 60)

# Check if it's simple 3x3 tiling
simple_tile = np.tile(input_grid, (3, 3))
print("\nSimple 3x3 tiling:")
print(simple_tile)
print(f"Matches expected: {np.array_equal(simple_tile, expected)}")

# Check for alternating pattern
print("\nLet's look at the 2x2 blocks in the output:")
for i in range(0, 6, 2):
    for j in range(0, 6, 2):
        block = expected[i : i + 2, j : j + 2]
        print(f"Block at ({i},{j}): {block.flatten()}")

# Analyze the transformation
print("\n" + "=" * 60)
print("TRANSFORMATION ANALYSIS")
print("=" * 60)

# The pattern appears to be:
# Row 0-1: Original | Flip-H | Original
# Row 2-3: Flip-V  | Flip-HV| Flip-V
# Row 4-5: Original | Flip-H | Original

# Let's verify this
original = input_grid
flip_h = np.fliplr(input_grid)  # Horizontal flip
flip_v = np.flipud(input_grid)  # Vertical flip
flip_hv = np.fliplr(np.flipud(input_grid))  # Both flips

print("Original:")
print(original)
print("\nFlip horizontal:")
print(flip_h)
print("\nFlip vertical:")
print(flip_v)
print("\nFlip both:")
print(flip_hv)

# Construct the pattern
constructed = np.zeros((6, 6), dtype=int)

# Top row of tiles
constructed[0:2, 0:2] = original
constructed[0:2, 2:4] = flip_h
constructed[0:2, 4:6] = original

# Middle row of tiles
constructed[2:4, 0:2] = flip_v
constructed[2:4, 2:4] = flip_hv
constructed[2:4, 4:6] = flip_v

# Bottom row of tiles
constructed[4:6, 0:2] = original
constructed[4:6, 2:4] = flip_h
constructed[4:6, 4:6] = original

print("\nConstructed pattern:")
print(constructed)
print(f"\nMatches expected: {np.array_equal(constructed, expected)}")

if np.array_equal(constructed, expected):
    print("\n✅ PATTERN IDENTIFIED!")
    print("The transformation is a 3x3 tiling with specific flips:")
    print("  [Original] [Flip-H]  [Original]")
    print("  [Flip-V]   [Flip-HV] [Flip-V]")
    print("  [Original] [Flip-H]  [Original]")
