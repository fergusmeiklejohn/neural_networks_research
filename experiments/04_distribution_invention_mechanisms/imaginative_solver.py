#!/usr/bin/env python3
"""
Imaginative Solver - Explores possibilities guided by patterns.

Instead of looking for THE rule, this solver:
1. Learns multiple patterns from examples
2. Generates plausible hypotheses
3. Tests them against constraints
4. Selects the best fit

This embodies "thinking outside the distribution" by not assuming
a single deterministic mapping exists.
"""

from utils.imports import setup_project_paths

setup_project_paths()

from itertools import permutations
from typing import Dict, List, Optional, Tuple

import numpy as np


class ImaginativeSolver:
    """A solver that imagines and tests possibilities."""

    def __init__(self):
        self.pattern_library = []
        self.hypothesis_space = []

    def observe_patterns(self, examples: List[Tuple[np.ndarray, np.ndarray]]) -> Dict:
        """Observe and catalog patterns without assuming they're universal."""
        patterns = {
            "permutations_seen": [],
            "transformations": [],
            "value_orderings": [],
            "structural_hints": [],
        }

        for inp, out in examples:
            # Get unique values
            unique_vals = sorted(set(inp.flatten()) - {0})
            if len(unique_vals) == 0:
                continue

            output_pattern = out.flatten()[: len(unique_vals)]

            # Record the permutation
            if set(output_pattern) == set(unique_vals):
                indices = [unique_vals.index(v) for v in output_pattern]
                patterns["permutations_seen"].append(indices)

            # Record value ordering strategies
            patterns["value_orderings"].append(
                {
                    "sorted": unique_vals,
                    "output": list(output_pattern),
                    "first_appearance": self._get_first_appearance_order(inp),
                    "row_wise": self._get_row_wise_order(inp),
                    "column_wise": self._get_column_wise_order(inp),
                }
            )

            # Look for structural hints
            if len(unique_vals) == 3:
                low, mid, high = unique_vals
                if list(output_pattern) == [low, high, mid]:
                    patterns["structural_hints"].append("low_high_mid")
                elif list(output_pattern) == [mid, high, low]:
                    patterns["structural_hints"].append("mid_high_low")
                elif list(output_pattern) == [mid, low, high]:
                    patterns["structural_hints"].append("mid_low_high")

        return patterns

    def _get_first_appearance_order(self, grid: np.ndarray) -> List[int]:
        """Get order of first appearance of unique values."""
        order = []
        seen = set()
        for val in grid.flatten():
            if val != 0 and val not in seen:
                order.append(val)
                seen.add(val)
        return order

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

    def imagine_possibilities(
        self, test_input: np.ndarray, patterns: Dict, max_hypotheses: int = 10
    ) -> List[np.ndarray]:
        """Generate plausible outputs by imagining based on patterns."""
        unique_test = sorted(set(test_input.flatten()) - {0})
        if len(unique_test) == 0:
            return []

        hypotheses = []

        # Strategy 1: Try observed permutations
        for perm_indices in patterns["permutations_seen"]:
            if len(perm_indices) == len(unique_test):
                try:
                    hypothesis = [unique_test[i] for i in perm_indices]
                    hypotheses.append(hypothesis)
                except IndexError:
                    continue

        # Strategy 2: Try structural patterns
        if len(unique_test) == 3:
            low, mid, high = unique_test
            structural_maps = {
                "low_high_mid": [low, high, mid],
                "mid_high_low": [mid, high, low],
                "mid_low_high": [mid, low, high],
                "low_mid_high": [low, mid, high],
                "high_mid_low": [high, mid, low],
                "high_low_mid": [high, low, mid],
            }
            for pattern in patterns.get("structural_hints", []):
                if pattern in structural_maps:
                    hypotheses.append(structural_maps[pattern])

        # Strategy 3: Try reading orders from test input
        reading_orders = [
            self._get_first_appearance_order(test_input),
            self._get_row_wise_order(test_input),
            self._get_column_wise_order(test_input),
        ]
        for order in reading_orders:
            if set(order) == set(unique_test) and len(order) == len(unique_test):
                hypotheses.append(order)

        # Strategy 4: Generate variations of seen patterns
        # This is where we "think outside" - create new combinations
        if len(unique_test) <= 4:  # Only for small sets
            # Add some novel permutations inspired by patterns
            for perm in list(permutations(unique_test))[:max_hypotheses]:
                if list(perm) not in hypotheses:
                    hypotheses.append(list(perm))

        # Remove duplicates while preserving order
        seen = set()
        unique_hypotheses = []
        for h in hypotheses:
            h_tuple = tuple(h)
            if h_tuple not in seen:
                seen.add(h_tuple)
                unique_hypotheses.append(h)

        return unique_hypotheses[:max_hypotheses]

    def evaluate_hypothesis(
        self, hypothesis: List[int], test_input: np.ndarray, patterns: Dict
    ) -> float:
        """Score how plausible a hypothesis is based on patterns."""
        score = 0.0

        # Check if it matches any seen permutation exactly
        unique_test = sorted(set(test_input.flatten()) - {0})
        if len(hypothesis) == len(unique_test):
            indices = [unique_test.index(v) for v in hypothesis]
            if indices in patterns["permutations_seen"]:
                score += 1.0

        # Check if it matches a structural hint
        if len(hypothesis) == 3 and len(unique_test) == 3:
            low, mid, high = unique_test
            structure = None
            if hypothesis == [low, high, mid]:
                structure = "low_high_mid"
            elif hypothesis == [mid, high, low]:
                structure = "mid_high_low"
            elif hypothesis == [mid, low, high]:
                structure = "mid_low_high"

            if structure and structure in patterns.get("structural_hints", []):
                score += 0.5

        # Check if it matches a reading order
        if hypothesis == self._get_row_wise_order(test_input):
            score += 0.3
        elif hypothesis == self._get_column_wise_order(test_input):
            score += 0.3

        return score

    def solve(
        self, examples: List[Tuple[np.ndarray, np.ndarray]], test_input: np.ndarray
    ) -> Optional[np.ndarray]:
        """Solve by imagining and testing possibilities."""
        # Observe patterns from examples
        patterns = self.observe_patterns(examples)

        # Get expected output shape
        if examples:
            output_shape = examples[0][1].shape
        else:
            return None

        # Generate hypotheses
        hypotheses = self.imagine_possibilities(test_input, patterns)

        if not hypotheses:
            return None

        # Score and rank hypotheses
        scored_hypotheses = [
            (h, self.evaluate_hypothesis(h, test_input, patterns)) for h in hypotheses
        ]
        scored_hypotheses.sort(key=lambda x: x[1], reverse=True)

        # Use the best hypothesis
        best_hypothesis = scored_hypotheses[0][0]

        # Create output by tiling the pattern
        output_size = output_shape[0] * output_shape[1]
        repeated = np.tile(best_hypothesis, (output_size // len(best_hypothesis)) + 1)[
            :output_size
        ]

        return repeated.reshape(output_shape)


def test_imaginative_solver():
    """Test the imaginative solver on task 05269061."""
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

    print("Testing Imaginative Solver")
    print("=" * 60)

    task_path = DATA_DIR / "05269061.json"
    task = load_arc_task(task_path)

    solver = ImaginativeSolver()

    # Get examples
    train_examples = [
        (np.array(ex["input"]), np.array(ex["output"])) for ex in task["train"]
    ]

    test_input = np.array(task["test"][0]["input"])
    expected_output = np.array(task["test"][0]["output"])

    # Observe patterns
    patterns = solver.observe_patterns(train_examples)
    print("Observed patterns:")
    print(f"  Permutations seen: {patterns['permutations_seen']}")
    print(f"  Structural hints: {set(patterns['structural_hints'])}")

    # Generate hypotheses
    hypotheses = solver.imagine_possibilities(test_input, patterns)
    print(f"\nGenerated {len(hypotheses)} hypotheses:")
    for i, h in enumerate(hypotheses[:5]):
        score = solver.evaluate_hypothesis(h, test_input, patterns)
        print(f"  {i+1}. {h} (score: {score:.2f})")

    # Solve
    result = solver.solve(train_examples, test_input)

    if result is not None:
        print(f"\nResult shape: {result.shape}")
        print(f"Expected shape: {expected_output.shape}")

        if result.shape == expected_output.shape:
            accuracy = np.sum(result == expected_output) / expected_output.size
            print(f"Accuracy: {accuracy:.1%}")

            if accuracy == 1.0:
                print("✅ PERFECT SOLUTION through imagination!")
            else:
                print("\nFirst few values:")
                print(f"  Predicted: {result.flatten()[:10]}")
                print(f"  Expected: {expected_output.flatten()[:10]}")

                # Which hypothesis would have worked?
                unique_test = sorted(set(test_input.flatten()) - {0})
                correct_pattern = list(expected_output.flatten()[: len(unique_test)])
                print(f"\nCorrect pattern was: {correct_pattern}")

                if correct_pattern in hypotheses:
                    idx = hypotheses.index(correct_pattern)
                    print(f"  This was hypothesis #{idx+1}")
                else:
                    print("  This pattern wasn't in our hypotheses")
                    print("  We need to expand our imagination!")


if __name__ == "__main__":
    test_imaginative_solver()
