#!/usr/bin/env python3
"""Analyze the tiling pattern in task 00576224 - v2."""

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

print("Looking at each row of the expected output:")
for i, row in enumerate(expected):
    print(f"Row {i}: {row}")

print("\n" + "=" * 60)
print("ANALYZING ROW PATTERNS")
print("=" * 60)

# Row 0: [3 2 3 2 3 2] - repeats [3, 2]
# Row 1: [7 8 7 8 7 8] - repeats [7, 8]
# Row 2: [2 3 2 3 2 3] - repeats [2, 3]
# Row 3: [8 7 8 7 8 7] - repeats [8, 7]
# Row 4: [3 2 3 2 3 2] - repeats [3, 2]
# Row 5: [7 8 7 8 7 8] - repeats [7, 8]

print("\nPattern observation:")
print("- Rows 0,1 repeat the original input rows 3 times")
print("- Rows 2,3 repeat the horizontally flipped input rows 3 times")
print("- Rows 4,5 repeat the original input rows 3 times")

# Let's verify this
flip_h = np.fliplr(input_grid)
print(
    f"\nOriginal input row 0: {input_grid[0]} -> repeated: {np.tile(input_grid[0], 3)}"
)
print(f"Original input row 1: {input_grid[1]} -> repeated: {np.tile(input_grid[1], 3)}")
print(f"Flipped input row 0: {flip_h[0]} -> repeated: {np.tile(flip_h[0], 3)}")
print(f"Flipped input row 1: {flip_h[1]} -> repeated: {np.tile(flip_h[1], 3)}")

# Construct based on this pattern
constructed = np.zeros((6, 6), dtype=int)
constructed[0] = np.tile(input_grid[0], 3)  # [3 2] repeated
constructed[1] = np.tile(input_grid[1], 3)  # [7 8] repeated
constructed[2] = np.tile(flip_h[0], 3)  # [2 3] repeated
constructed[3] = np.tile(flip_h[1], 3)  # [8 7] repeated
constructed[4] = np.tile(input_grid[0], 3)  # [3 2] repeated
constructed[5] = np.tile(input_grid[1], 3)  # [7 8] repeated

print("\nConstructed output:")
print(constructed)

print(f"\nMatches expected: {np.array_equal(constructed, expected)}")

if np.array_equal(constructed, expected):
    print("\n✅ PATTERN IDENTIFIED!")
    print("The transformation is:")
    print("1. Scale 3x horizontally (repeat each row element 3 times)")
    print("2. Stack vertically: Original -> Flipped -> Original")
    print("   where 'Flipped' means horizontally flipped version")

# Alternative view: as tile + modification
print("\n" + "=" * 60)
print("ALTERNATIVE VIEW: TILE WITH MODIFICATIONS")
print("=" * 60)

# Simple 3x3 tile
simple_tile = np.tile(input_grid, (3, 3))
print("Simple 3x3 tile:")
print(simple_tile)

print("\nDifference from expected:")
diff = simple_tile != expected
print(diff.astype(int))

print("\nModifications needed:")
print("Rows 2-3 need to be horizontally flipped")
