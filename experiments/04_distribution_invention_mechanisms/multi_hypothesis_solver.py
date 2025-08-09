#!/usr/bin/env python3
"""
Multi-Hypothesis Imaginative Solver.

This solver embodies true "distribution invention" by:
1. Generating multiple plausible hypotheses
2. Testing them against validation criteria
3. Selecting the best through empirical testing

This is how humans solve these puzzles - imagining possibilities
and mentally testing them until one fits.
"""

from utils.imports import setup_project_paths

setup_project_paths()

from itertools import permutations
from typing import List, Optional, Tuple

import numpy as np


class MultiHypothesisSolver:
    """A solver that tests multiple imagined possibilities."""

    def __init__(self, max_hypotheses: int = 20):
        self.max_hypotheses = max_hypotheses

    def generate_all_hypotheses(
        self, test_input: np.ndarray, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> List[List[int]]:
        """Generate a diverse set of hypotheses."""
        unique_test = sorted(set(test_input.flatten()) - {0})
        if len(unique_test) == 0:
            return []

        hypotheses = []
        seen_hypotheses = set()

        # 1. Learn from examples - what permutations were used?
        for inp, out in examples:
            unique_vals = sorted(set(inp.flatten()) - {0})
            if len(unique_vals) != len(unique_test):
                continue

            output_pattern = out.flatten()[: len(unique_vals)]
            if set(output_pattern) == set(unique_vals):
                # Map this permutation to test values
                indices = [unique_vals.index(v) for v in output_pattern]
                try:
                    test_hypothesis = [unique_test[i] for i in indices]
                    h_tuple = tuple(test_hypothesis)
                    if h_tuple not in seen_hypotheses:
                        hypotheses.append(test_hypothesis)
                        seen_hypotheses.add(h_tuple)
                except IndexError:
                    pass

        # 2. Try all possible permutations for small sets
        if len(unique_test) <= 4:
            for perm in permutations(unique_test):
                h_tuple = tuple(perm)
                if h_tuple not in seen_hypotheses:
                    hypotheses.append(list(perm))
                    seen_hypotheses.add(h_tuple)
                    if len(hypotheses) >= self.max_hypotheses:
                        break

        # 3. Add reading order hypotheses
        reading_orders = [
            self._get_row_wise_order(test_input),
            self._get_column_wise_order(test_input),
            self._get_spiral_order(test_input),
            self._get_diagonal_order(test_input),
        ]

        for order in reading_orders:
            if set(order) == set(unique_test) and len(order) == len(unique_test):
                h_tuple = tuple(order)
                if h_tuple not in seen_hypotheses:
                    hypotheses.append(order)
                    seen_hypotheses.add(h_tuple)

        return hypotheses[: self.max_hypotheses]

    def _get_row_wise_order(self, grid: np.ndarray) -> List[int]:
        """Get order reading row by row."""
        order = []
        seen = set()
        for row in grid:
            for val in row:
                if val != 0 and val not in seen:
                    order.append(val)
                    seen.add(val)
        return order

    def _get_column_wise_order(self, grid: np.ndarray) -> List[int]:
        """Get order reading column by column."""
        order = []
        seen = set()
        for col in range(grid.shape[1]):
            for row in range(grid.shape[0]):
                val = grid[row, col]
                if val != 0 and val not in seen:
                    order.append(val)
                    seen.add(val)
        return order

    def _get_spiral_order(self, grid: np.ndarray) -> List[int]:
        """Get order reading in spiral pattern."""
        order = []
        seen = set()

        # Simple spiral: outside to inside
        h, w = grid.shape
        top, bottom, left, right = 0, h - 1, 0, w - 1

        while top <= bottom and left <= right:
            # Top row
            for col in range(left, right + 1):
                val = grid[top, col]
                if val != 0 and val not in seen:
                    order.append(val)
                    seen.add(val)
            top += 1

            # Right column
            for row in range(top, bottom + 1):
                val = grid[row, right]
                if val != 0 and val not in seen:
                    order.append(val)
                    seen.add(val)
            right -= 1

            # Bottom row
            if top <= bottom:
                for col in range(right, left - 1, -1):
                    val = grid[bottom, col]
                    if val != 0 and val not in seen:
                        order.append(val)
                        seen.add(val)
                bottom -= 1

            # Left column
            if left <= right:
                for row in range(bottom, top - 1, -1):
                    val = grid[row, left]
                    if val != 0 and val not in seen:
                        order.append(val)
                        seen.add(val)
                left += 1

        return order

    def _get_diagonal_order(self, grid: np.ndarray) -> List[int]:
        """Get order reading diagonally."""
        order = []
        seen = set()

        # Top-left to bottom-right
        for i in range(min(grid.shape)):
            val = grid[i, i]
            if val != 0 and val not in seen:
                order.append(val)
                seen.add(val)

        # Add remaining values in any order
        for row in range(grid.shape[0]):
            for col in range(grid.shape[1]):
                val = grid[row, col]
                if val != 0 and val not in seen:
                    order.append(val)
                    seen.add(val)

        return order

    def test_hypothesis(
        self,
        hypothesis: List[int],
        examples: List[Tuple[np.ndarray, np.ndarray]],
        output_shape: Tuple[int, int],
    ) -> float:
        """Test how well a hypothesis fits the examples."""
        score = 0.0
        total_tests = 0

        # Create the full output grid using this hypothesis
        output_size = output_shape[0] * output_shape[1]
        test_output = np.tile(hypothesis, (output_size // len(hypothesis)) + 1)[
            :output_size
        ]
        test_output = test_output.reshape(output_shape)

        # Check if this pattern would work on training examples
        for inp, expected_out in examples:
            unique_vals = sorted(set(inp.flatten()) - {0})
            if len(unique_vals) != len(hypothesis):
                continue

            # Map hypothesis to these values
            value_map = {
                test_val: train_val
                for test_val, train_val in zip(sorted(set(hypothesis)), unique_vals)
            }
            mapped_hypothesis = [value_map.get(v, v) for v in hypothesis]

            # Create output for this example
            predicted = np.tile(
                mapped_hypothesis, (expected_out.size // len(mapped_hypothesis)) + 1
            )[: expected_out.size]
            predicted = predicted.reshape(expected_out.shape)

            # Score similarity
            if predicted.shape == expected_out.shape:
                accuracy = np.sum(predicted == expected_out) / expected_out.size
                score += accuracy
                total_tests += 1

        return score / total_tests if total_tests > 0 else 0.0

    def solve(
        self, examples: List[Tuple[np.ndarray, np.ndarray]], test_input: np.ndarray
    ) -> Optional[np.ndarray]:
        """Solve by testing multiple hypotheses."""
        # Get expected output shape
        if not examples:
            return None
        output_shape = examples[0][1].shape

        # Generate all hypotheses
        hypotheses = self.generate_all_hypotheses(test_input, examples)

        if not hypotheses:
            return None

        # Test each hypothesis
        best_hypothesis = None
        best_score = -1

        for hypothesis in hypotheses:
            score = self.test_hypothesis(hypothesis, examples, output_shape)
            if score > best_score:
                best_score = score
                best_hypothesis = hypothesis

        if best_hypothesis is None:
            # Fallback to first hypothesis
            best_hypothesis = hypotheses[0]

        # Create output using best hypothesis
        output_size = output_shape[0] * output_shape[1]
        result = np.tile(best_hypothesis, (output_size // len(best_hypothesis)) + 1)[
            :output_size
        ]

        return result.reshape(output_shape)


def test_multi_hypothesis():
    """Test the multi-hypothesis solver."""
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

    print("Testing Multi-Hypothesis Solver")
    print("=" * 60)

    task_path = DATA_DIR / "05269061.json"
    task = load_arc_task(task_path)

    solver = MultiHypothesisSolver(max_hypotheses=30)

    # Get examples
    train_examples = [
        (np.array(ex["input"]), np.array(ex["output"])) for ex in task["train"]
    ]

    test_input = np.array(task["test"][0]["input"])
    expected_output = np.array(task["test"][0]["output"])

    # Generate hypotheses
    hypotheses = solver.generate_all_hypotheses(test_input, train_examples)
    print(f"Generated {len(hypotheses)} hypotheses")

    # Check if correct answer is in hypotheses
    unique_test = sorted(set(test_input.flatten()) - {0})
    correct_pattern = list(expected_output.flatten()[: len(unique_test)])

    if correct_pattern in hypotheses:
        idx = hypotheses.index(correct_pattern)
        print(f"✓ Correct pattern {correct_pattern} is hypothesis #{idx+1}")
    else:
        print(f"✗ Correct pattern {correct_pattern} not in hypotheses")

    # Test scoring
    print("\nTesting hypothesis scoring:")
    for i, h in enumerate(hypotheses[:5]):
        score = solver.test_hypothesis(h, train_examples, expected_output.shape)
        is_correct = "✓" if h == correct_pattern else " "
        print(f"  {is_correct} {i+1}. {h} (score: {score:.3f})")

    # Solve
    result = solver.solve(train_examples, test_input)

    if result is not None:
        print(f"\nResult shape: {result.shape}")
        print(f"Expected shape: {expected_output.shape}")

        if result.shape == expected_output.shape:
            accuracy = np.sum(result == expected_output) / expected_output.size
            print(f"Accuracy: {accuracy:.1%}")

            if accuracy == 1.0:
                print("✅ PERFECT SOLUTION through multi-hypothesis imagination!")
                print("\nThis demonstrates 'thinking outside the distribution':")
                print("- We didn't assume a single rule")
                print("- We imagined multiple possibilities")
                print("- We tested them empirically")
                print("- We found the one that works!")
            else:
                print("\nFirst few values:")
                print(f"  Predicted: {result.flatten()[:10]}")
                print(f"  Expected: {expected_output.flatten()[:10]}")


if __name__ == "__main__":
    test_multi_hypothesis()
