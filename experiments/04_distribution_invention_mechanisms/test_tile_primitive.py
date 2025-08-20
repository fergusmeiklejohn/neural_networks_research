#!/usr/bin/env python3
"""Test the TilePattern primitive to see why it's not working."""

from utils.imports import setup_project_paths

setup_project_paths()

import numpy as np
from arc_dsl_enhanced import EnhancedDSLLibrary, TilePattern

# Test input
input_grid = np.array([[3, 2], [7, 8]])

print("Input (2x2):")
print(input_grid)

# Expected output (6x6)
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

print("\nExpected (6x6):")
print(expected)

# Test the TilePattern primitive
print("\n" + "=" * 60)
print("Testing TilePattern Primitive")
print("=" * 60)

# Create DSL library
dsl = EnhancedDSLLibrary()

# Get TilePattern with scale=3
tile_primitive = dsl.get_primitive("tile_pattern", scale=3)
print(f"Primitive: {tile_primitive}")

# Execute
try:
    output = tile_primitive.execute(input_grid)
    print(f"\nOutput shape: {output.shape}")
    print("Output:")
    print(output)

    print(f"\nMatches expected: {np.array_equal(output, expected)}")

    # Check what we get
    simple_tile = np.tile(input_grid, (3, 3))
    print(f"Same as np.tile? {np.array_equal(output, simple_tile)}")

except Exception as e:
    print(f"Error executing primitive: {e}")
    import traceback

    traceback.print_exc()

# Test direct TilePattern class
print("\n" + "=" * 60)
print("Testing TilePattern Class Directly")
print("=" * 60)

tile = TilePattern(scale=3)
try:
    output2 = tile.execute(input_grid)
    print(f"Output shape: {output2.shape}")
    print("Output:")
    print(output2)
except Exception as e:
    print(f"Error: {e}")
    import traceback

    traceback.print_exc()
