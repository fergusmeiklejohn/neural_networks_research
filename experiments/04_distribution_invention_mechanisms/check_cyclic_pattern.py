#!/usr/bin/env python3
"""Check if there's a cyclic or rotation pattern."""

from utils.imports import setup_project_paths

setup_project_paths()

import json
from pathlib import Path

import numpy as np

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


def check_cyclic_pattern():
    """Check if there's a cyclic shift or rotation pattern."""
    task_path = DATA_DIR / "05269061.json"
    task = load_arc_task(task_path)

    print("Checking Cyclic/Rotation Pattern")
    print("=" * 60)

    all_patterns = []

    for i, example in enumerate(task["train"]):
        print(f"\nExample {i}:")
        inp = np.array(example["input"])
        out = np.array(example["output"])

        # Get unique values in sorted order
        sorted_unique = sorted(set(inp.flatten()) - {0})
        output_pattern = list(out.flatten()[: len(sorted_unique)])

        print(f"  Sorted unique: {sorted_unique}")
        print(f"  Output pattern: {output_pattern}")

        # Check if it's a rotation of sorted
        for shift in range(len(sorted_unique)):
            rotated = sorted_unique[shift:] + sorted_unique[:shift]
            if rotated == output_pattern:
                print(f"  ✓ Right rotation by {shift}")
                all_patterns.append(("right_rotation", shift))
                break

            # Also check reverse then rotate
            reversed_sorted = sorted_unique[::-1]
            rotated_rev = reversed_sorted[shift:] + reversed_sorted[:shift]
            if rotated_rev == output_pattern:
                print(f"  ✓ Reverse then right rotation by {shift}")
                all_patterns.append(("reverse_rotation", shift))
                break

        # Maybe it's based on indices
        # Get the indices that would produce this output
        indices = []
        for val in output_pattern:
            indices.append(sorted_unique.index(val))
        print(f"  Indices from sorted: {indices}")
        all_patterns.append(("indices", indices))

    # Look for pattern in the patterns
    print("\n" + "=" * 60)
    print("Pattern Summary:")
    for i, (ptype, pval) in enumerate(all_patterns):
        print(f"  Example {i}: {ptype} = {pval}")

    # Maybe it cycles through different permutations?
    print("\nChecking if permutation cycles:")
    perms = [p[1] for p in all_patterns if p[0] == "indices"]
    print(f"  Permutations: {perms}")

    # Test what the next one would be
    if len(perms) >= 2:
        # Check if there's a pattern
        if perms[1] == perms[2]:  # Same permutation repeated
            print(f"  Pattern stabilizes at: {perms[1]}")
            next_perm = perms[1]
        else:
            # Alternating?
            print("  Pattern alternates")
            # For test (4th example), use the first pattern?
            next_perm = perms[0]

        print(f"\n  Predicted for test: {next_perm}")

        # Apply to test
        test = task["test"][0]
        test_input = np.array(test["input"])
        test_output = np.array(test["output"])

        unique_test = sorted(set(test_input.flatten()) - {0})
        if next_perm == [0, 2, 1]:
            predicted = [unique_test[i] for i in [0, 2, 1]]
        elif next_perm == [1, 2, 0]:
            predicted = [unique_test[i] for i in [1, 2, 0]]
        else:
            predicted = [unique_test[i] for i in next_perm]

        expected = list(test_output.flatten()[: len(unique_test)])

        print(f"  Test unique: {unique_test}")
        print(f"  Predicted: {predicted}")
        print(f"  Expected: {expected}")
        print(f"  Match: {predicted == expected}")

        # What permutation is actually needed?
        actual_indices = [unique_test.index(v) for v in expected]
        print(f"  Actual needed indices: {actual_indices}")


if __name__ == "__main__":
    check_cyclic_pattern()
