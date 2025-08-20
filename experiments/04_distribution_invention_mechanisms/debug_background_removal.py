#!/usr/bin/env python3
"""Debug the background removal solver."""

from utils.imports import setup_project_paths

setup_project_paths()

import json
from pathlib import Path

import numpy as np
from enhanced_arc_solver_v8 import BackgroundRemovalSolver

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


def debug_background_removal():
    """Debug the background removal pattern solver."""
    print("Debugging Background Removal Solver")
    print("=" * 60)

    task_path = DATA_DIR / "05269061.json"
    task = load_arc_task(task_path)

    solver = BackgroundRemovalSolver()

    # Get examples
    train_examples = [
        (np.array(ex["input"]), np.array(ex["output"])) for ex in task["train"]
    ]

    test_input = np.array(task["test"][0]["input"])
    expected_output = np.array(task["test"][0]["output"])

    # Test detection
    print("Testing pattern detection:")
    is_pattern = solver.detect_pattern(train_examples)
    print(f"  Pattern detected: {is_pattern}")

    if not is_pattern:
        print("\n  Checking why pattern not detected:")
        for i, (inp, out) in enumerate(train_examples):
            has_zero = 0 in out
            print(f"    Example {i}: output has zeros = {has_zero}")

            unique_in = set(inp.flatten()) - {0}
            unique_out = set(out.flatten())
            print(f"      Input colors (non-zero): {sorted(unique_in)}")
            print(f"      Output colors: {sorted(unique_out)}")
            print(f"      Match: {unique_in == unique_out}")

    # Test permutation finding
    print("\nTesting permutation finding:")
    perm = solver.find_permutation(train_examples)
    print(f"  Found permutation: {perm}")

    # Manual check
    print("\nManual analysis of first example:")
    inp, out = train_examples[0]
    print(f"  Input non-zero values: {inp[inp != 0]}")
    print(f"  Output first few values: {out.flatten()[:10]}")

    # The pattern is actually: take unique colors and repeat them
    unique_colors = sorted(set(inp.flatten()) - {0})
    print(f"  Unique colors: {unique_colors}")

    # Check different orderings
    from itertools import permutations

    for perm_try in permutations(unique_colors):
        pattern = list(perm_try)
        repeated = np.tile(pattern, (out.size // len(pattern)) + 1)[: out.size]
        if np.array_equal(repeated, out.flatten()):
            print(f"  ✓ Found correct permutation: {pattern}")
            break

    # Test solving
    print("\nTesting solve function:")
    result = solver.solve(train_examples, test_input)

    if result is not None:
        print(f"  Result shape: {result.shape}")
        print(f"  Expected shape: {expected_output.shape}")

        if result.shape == expected_output.shape:
            accuracy = np.sum(result == expected_output) / expected_output.size
            print(f"  Accuracy: {accuracy:.1%}")

            if accuracy == 1.0:
                print("  ✅ Perfect solution!")
            else:
                print("\n  First few values:")
                print(f"    Predicted: {result.flatten()[:10]}")
                print(f"    Expected: {expected_output.flatten()[:10]}")
    else:
        print("  Result is None!")


if __name__ == "__main__":
    debug_background_removal()
