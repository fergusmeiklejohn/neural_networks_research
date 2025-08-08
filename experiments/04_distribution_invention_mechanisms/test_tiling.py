#!/usr/bin/env python3
"""Test if tiling primitive solves ARC task."""

import json
from pathlib import Path

import numpy as np
from arc_dsl_enhanced import TilePattern

# Load task 007bbfb7
task_path = Path(
    "/Users/fergusmeiklejohn/dev/neural_networks_research/data/arc_agi_official/ARC-AGI/data/training/007bbfb7.json"
)
with open(task_path) as f:
    task = json.load(f)

# Test tiling on first example
inp = np.array(task["train"][0]["input"])
expected = np.array(task["train"][0]["output"])

print("Testing TilePattern on task 007bbfb7")
print("=" * 50)
print(f"Input shape: {inp.shape}")
print(f"Expected output shape: {expected.shape}")

# Try tiling with scale=3
tiler = TilePattern(scale=3)
output = tiler.execute(inp)

print(f"Tiled output shape: {output.shape}")
print(f"\nDoes it match? {np.array_equal(output, expected)}")

if not np.array_equal(output, expected):
    # Show differences
    diff_count = (output != expected).sum()
    print(f"Pixels different: {diff_count}/{expected.size}")
    print("\nInput:")
    print(inp)
    print("\nExpected:")
    print(expected)
    print("\nGot (tiling):")
    print(output)
