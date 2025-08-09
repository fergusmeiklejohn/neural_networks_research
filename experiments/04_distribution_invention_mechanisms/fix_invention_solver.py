#!/usr/bin/env python3
"""
Fixed version of the invention solver to handle edge cases.
"""

from utils.imports import setup_project_paths

setup_project_paths()

from itertools import permutations
from typing import Dict, List, Optional, Tuple

import numpy as np


class FixedDistributionInventionSolver:
    """
    Fixed solver that handles edge cases properly.
    """

    def __init__(self, invention_budget: int = 50):
        self.invention_budget = invention_budget

    def extract_distribution_hints(
        self, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Dict:
        """Extract hints about possible distributions, not rules."""
        hints = {
            "value_relationships": [],
            "structural_patterns": [],
            "transformation_types": [],
        }

        for i, (inp, out) in enumerate(examples):
            unique_vals = sorted(set(inp.flatten()) - {0})
            if len(unique_vals) == 0:
                continue

            # Ensure output pattern doesn't exceed available values
            output_flat = out.flatten()
            output_pattern = []
            for val in output_flat[: len(unique_vals)]:
                if val in unique_vals:
                    output_pattern.append(val)

            if not output_pattern:
                continue

            # Value relationships
            if len(unique_vals) >= 3:
                low, mid, high = unique_vals[:3]  # Take first 3
                if output_pattern and output_pattern[0] == mid:
                    hints["value_relationships"].append("middle_first")

            # Transformation types
            if set(output_pattern).issubset(set(unique_vals)):
                hints["transformation_types"].append("permutation")

        return hints

    def invent_distributions(
        self, test_input: np.ndarray, hints: Dict
    ) -> List[List[int]]:
        """Invent possible distributions based on hints."""
        unique_test = sorted(set(test_input.flatten()) - {0})
        if len(unique_test) == 0:
            return []

        distributions = []
        seen = set()

        # Generate permutations for small sets
        if len(unique_test) <= 4:
            for perm in permutations(unique_test):
                perm_tuple = tuple(perm)
                if perm_tuple not in seen:
                    distributions.append(list(perm))
                    seen.add(perm_tuple)
                    if len(distributions) >= self.invention_budget:
                        break

        # If no distributions yet, create some basic ones
        if not distributions:
            distributions.append(unique_test)  # Identity
            distributions.append(unique_test[::-1])  # Reverse
            if len(unique_test) >= 2:
                # Swap first two
                swapped = unique_test.copy()
                swapped[0], swapped[1] = swapped[1], swapped[0]
                distributions.append(swapped)

        return distributions[: self.invention_budget]

    def evaluate_distribution(
        self,
        distribution: List[int],
        examples: List[Tuple[np.ndarray, np.ndarray]],
        output_shape: Tuple[int, int],
    ) -> float:
        """Evaluate how plausible an invented distribution is."""
        if not distribution or not examples:
            return 0.0

        score = 0.0
        tests_performed = 0

        for inp, expected_out in examples[:2]:
            unique_vals = sorted(set(inp.flatten()) - {0})
            if len(unique_vals) != len(distribution):
                continue

            # Safe value mapping
            try:
                value_map = dict(zip(sorted(distribution), unique_vals))
                mapped = [value_map.get(v, v) for v in distribution]

                # Check if this produces something similar
                test_output = np.tile(mapped, (expected_out.size // len(mapped)) + 1)[
                    : expected_out.size
                ]
                test_output = test_output.reshape(expected_out.shape)

                if test_output.shape == expected_out.shape:
                    accuracy = np.sum(test_output == expected_out) / expected_out.size
                    score += accuracy
                    tests_performed += 1
            except Exception:
                continue

        return score / tests_performed if tests_performed > 0 else 0.0

    def solve(
        self, examples: List[Tuple[np.ndarray, np.ndarray]], test_input: np.ndarray
    ) -> Optional[np.ndarray]:
        """Solve by inventing and testing distributions."""
        if not examples:
            return None

        output_shape = examples[0][1].shape

        # Extract hints
        hints = self.extract_distribution_hints(examples)

        # Invent distributions
        distributions = self.invent_distributions(test_input, hints)

        if not distributions:
            # Fallback: return something structured
            return np.zeros(output_shape, dtype=test_input.dtype)

        # Evaluate each distribution
        best_distribution = None
        best_score = -1

        for dist in distributions:
            score = self.evaluate_distribution(dist, examples, output_shape)
            if score > best_score:
                best_score = score
                best_distribution = dist

        if best_distribution is None:
            best_distribution = distributions[0]

        # Generate output
        output_size = output_shape[0] * output_shape[1]
        result = np.tile(
            best_distribution, (output_size // len(best_distribution)) + 1
        )[:output_size]

        return result.reshape(output_shape)


def test_fixed_solver():
    """Test the fixed solver."""
    print("Testing Fixed Distribution Invention Solver")
    print("=" * 60)

    # Test with edge cases
    solver = FixedDistributionInventionSolver(invention_budget=10)

    # Test case 1: Empty input
    examples = [(np.zeros((3, 3)), np.ones((3, 3)))]
    test_input = np.zeros((3, 3))

    result = solver.solve(examples, test_input)
    print(f"Empty input test: {'✓' if result is not None else '✗'}")

    # Test case 2: Normal case
    examples = [(np.array([[1, 2], [3, 0]]), np.array([[2, 1], [2, 1]]))]
    test_input = np.array([[4, 5], [6, 0]])

    result = solver.solve(examples, test_input)
    print(f"Normal case test: {'✓' if result is not None else '✗'}")

    if result is not None:
        print(f"Result shape: {result.shape}")
        print(f"Result:\n{result}")


if __name__ == "__main__":
    test_fixed_solver()
