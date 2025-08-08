#!/usr/bin/env python3
"""Analyze what a specific ARC task actually requires."""

import json
from pathlib import Path

import numpy as np
from scipy.ndimage import zoom

# Load the first task that's failing
task_path = Path(
    "/Users/fergusmeiklejohn/dev/neural_networks_research/data/arc_agi_official/ARC-AGI/data/training/007bbfb7.json"
)
with open(task_path) as f:
    task = json.load(f)

print("Task 007bbfb7 - Actual transformation:")
print("=" * 50)

for i, ex in enumerate(task["train"][:2]):  # Show first 2 examples
    inp = np.array(ex["input"])
    out = np.array(ex["output"])
    print(f"\nExample {i+1}:")
    print(f"Input shape: {inp.shape} -> Output shape: {out.shape}")
    print(f"Size change: {out.shape[0]/inp.shape[0]:.1f}x scaling")

    # Check if it's just scaling
    scaled = zoom(inp, 3, order=0)  # 3x scaling with nearest neighbor
    if np.array_equal(scaled, out):
        print("Transformation: Simple 3x scaling")
    else:
        print("Transformation: Complex (not simple scaling)")
        # Check differences
        if scaled.shape == out.shape:
            diff = (scaled != out).sum()
            print(f"Pixels different from simple scaling: {diff}/{out.size}")

            # Show where they differ
            print("\nInput:")
            print(inp)
            print("\nOutput:")
            print(out)
            print("\nSimple 3x scaling would give:")
            print(scaled)
