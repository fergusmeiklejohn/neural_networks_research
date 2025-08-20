#!/usr/bin/env python3
"""Fixed background removal solver with proper permutation learning."""

from utils.imports import setup_project_paths

setup_project_paths()

from itertools import permutations
from typing import List, Optional, Tuple

import numpy as np


class FixedBackgroundRemovalSolver:
    """Fixed solver for background removal patterns."""

    def detect_pattern(self, examples: List[Tuple[np.ndarray, np.ndarray]]) -> bool:
        """Check if this is a background removal pattern."""
        for inp, out in examples:
            # Check if output has no zeros (background removed)
            if 0 in out:
                return False

            # Check if unique non-zero values match
            unique_input = set(inp.flatten()) - {0}
            unique_output = set(out.flatten())

            if unique_input != unique_output:
                return False

        return True

    def learn_permutation_mapping(
        self, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[dict]:
        """Learn how to map input values to output permutation order."""
        # Collect all permutation patterns from examples
        patterns = []

        for inp, out in examples:
            # Get unique values sorted by their first appearance in input
            unique_values = []
            seen = set()
            for val in inp.flatten():
                if val != 0 and val not in seen:
                    unique_values.append(val)
                    seen.add(val)

            # Find the permutation used in output
            output_flat = out.flatten()

            for perm in permutations(unique_values):
                test_pattern = list(perm)
                repeated = np.tile(test_pattern, (out.size // len(test_pattern)) + 1)[
                    : out.size
                ]

                if np.array_equal(repeated, output_flat):
                    # Found the permutation! Store the mapping
                    # Map from sorted order to permutation indices
                    sorted_values = sorted(unique_values)
                    perm_indices = [sorted_values.index(v) for v in perm]
                    patterns.append(perm_indices)
                    break

        if patterns:
            # Use the first pattern (they should all be the same)
            return {"permutation_indices": patterns[0]}

        return None

    def solve(
        self, examples: List[Tuple[np.ndarray, np.ndarray]], test_input: np.ndarray
    ) -> Optional[np.ndarray]:
        """Solve background removal pattern with fixed permutation learning."""
        if not self.detect_pattern(examples):
            return None

        # Learn the permutation mapping from examples
        mapping = self.learn_permutation_mapping(examples)

        if mapping is None:
            return None

        # Get unique values from test input (sorted)
        unique_test = sorted(set(test_input.flatten()) - {0})

        if not unique_test:
            return None

        # Apply the learned permutation
        perm_indices = mapping["permutation_indices"]

        # Make sure we have the right number of values
        if len(perm_indices) != len(unique_test):
            # Try to adapt the pattern
            # This might happen if test has different number of unique values
            # For now, just use the values in order
            output_pattern = unique_test
        else:
            # Apply the permutation
            output_pattern = [unique_test[i] for i in perm_indices]

        # Get expected output shape from first example
        output_shape = examples[0][1].shape

        # Create output by repeating pattern
        output_size = output_shape[0] * output_shape[1]
        repeated = np.tile(output_pattern, (output_size // len(output_pattern)) + 1)[
            :output_size
        ]

        return repeated.reshape(output_shape)


def test_fixed_solver():
    """Test the fixed background removal solver."""
    import json
    from pathlib import Path

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

    print("Testing Fixed Background Removal Solver")
    print("=" * 60)

    task_path = DATA_DIR / "05269061.json"
    task = load_arc_task(task_path)

    solver = FixedBackgroundRemovalSolver()

    # Get examples
    train_examples = [
        (np.array(ex["input"]), np.array(ex["output"])) for ex in task["train"]
    ]

    test_input = np.array(task["test"][0]["input"])
    expected_output = np.array(task["test"][0]["output"])

    # Test detection
    print("Pattern detected:", solver.detect_pattern(train_examples))

    # Test permutation learning
    mapping = solver.learn_permutation_mapping(train_examples)
    print(f"Learned mapping: {mapping}")

    # Test solving
    result = solver.solve(train_examples, test_input)

    if result is not None:
        print(f"\nResult shape: {result.shape}")
        print(f"Expected shape: {expected_output.shape}")

        if result.shape == expected_output.shape:
            accuracy = np.sum(result == expected_output) / expected_output.size
            print(f"Accuracy: {accuracy:.1%}")

            if accuracy == 1.0:
                print("✅ PERFECT SOLUTION!")
            else:
                print("\nFirst few values:")
                print(f"  Predicted: {result.flatten()[:10]}")
                print(f"  Expected: {expected_output.flatten()[:10]}")

                # Debug the difference
                print("\nDebug info:")
                unique_test = sorted(set(test_input.flatten()) - {0})
                print(f"  Test unique values: {unique_test}")
                print(
                    f"  Expected pattern: {expected_output.flatten()[:len(unique_test)]}"
                )
    else:
        print("Result is None!")


if __name__ == "__main__":
    test_fixed_solver()
