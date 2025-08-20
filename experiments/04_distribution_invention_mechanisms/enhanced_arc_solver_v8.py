#!/usr/bin/env python3
"""
Enhanced ARC Solver V8: Adding targeted pattern fixes.

Key improvements over V7:
1. Background removal pattern for tasks like 05269061
2. Enhanced column/row-specific position learning for 007bbfb7
3. Object-relative position rules for 00d62c1b
4. Local error correction for >85% accurate solutions
"""

from utils.imports import setup_project_paths

setup_project_paths()

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from enhanced_arc_solver_v7 import EnhancedARCSolverV7


@dataclass
class ARCSolutionV8:
    """Solution from V8 solver."""

    output_grid: np.ndarray
    confidence: float
    method_used: str
    program: Optional[Any] = None
    error_corrections: Optional[Dict] = None


class BackgroundRemovalSolver:
    """Solver for background removal patterns (like task 05269061)."""

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

    def find_permutation(
        self, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[List[int]]:
        """Find the permutation order for unique values."""
        from itertools import permutations

        for inp, out in examples:
            # Get unique non-zero values
            unique_values = sorted(set(inp.flatten()) - {0})

            if len(unique_values) == 0:
                continue

            # Check different permutations
            output_flat = out.flatten()

            for perm in permutations(unique_values):
                test_pattern = np.array(perm)
                repeated = np.tile(test_pattern, (out.size // len(test_pattern)) + 1)[
                    : out.size
                ]

                if np.array_equal(repeated, output_flat):
                    return list(perm)

        return None

    def solve(
        self, examples: List[Tuple[np.ndarray, np.ndarray]], test_input: np.ndarray
    ) -> Optional[np.ndarray]:
        """Solve background removal pattern."""
        if not self.detect_pattern(examples):
            return None

        # Get unique values from test input
        unique_test = sorted(set(test_input.flatten()) - {0})

        if not unique_test:
            return None

        # Find how to order the unique values by checking examples
        best_pattern = None

        # Try to find pattern from examples
        from itertools import permutations

        for inp, out in examples[:1]:  # Check first example
            unique_in = sorted(set(inp.flatten()) - {0})

            # Try different permutations of unique values
            for perm in permutations(unique_in):
                pattern = list(perm)
                repeated = np.tile(pattern, (out.size // len(pattern)) + 1)[: out.size]

                if np.array_equal(repeated, out.flatten()):
                    # Found the pattern! Now map to test unique values
                    if len(unique_test) == len(unique_in):
                        # Direct mapping based on sorted order
                        best_pattern = list(perm)
                        # Replace values with test values
                        value_map = {
                            v: t for v, t in zip(sorted(unique_in), sorted(unique_test))
                        }
                        best_pattern = [value_map.get(v, v) for v in best_pattern]
                    break

            if best_pattern:
                break

        # If no pattern found, try simple ordering
        if best_pattern is None:
            # Try different orderings based on value magnitude
            for perm in permutations(unique_test):
                best_pattern = list(perm)
                break  # Just use first permutation for now

        if best_pattern is None:
            return None

        # Get expected output shape (same as first example output)
        if examples:
            output_shape = examples[0][1].shape
        else:
            output_shape = test_input.shape

        # Create output by repeating pattern
        output_size = output_shape[0] * output_shape[1]
        repeated = np.tile(best_pattern, (output_size // len(best_pattern)) + 1)[
            :output_size
        ]

        return repeated.reshape(output_shape)


class EnhancedPositionLearner:
    """Enhanced position-dependent learning with column/row rules."""

    def learn_column_row_rules(
        self, base_pattern: np.ndarray, output: np.ndarray, scale: Tuple[int, int]
    ) -> Dict:
        """Learn column and row-specific transformation rules."""
        rules = {"column_rules": {}, "row_rules": {}, "position_rules": {}}

        scale_y, scale_x = int(scale[0]), int(scale[1])
        h, w = base_pattern.shape

        # Analyze each tile position
        for tile_row in range(scale_y):
            for tile_col in range(scale_x):
                # Extract this tile from output
                tile = output[
                    tile_row * h : (tile_row + 1) * h, tile_col * w : (tile_col + 1) * w
                ]

                # Compare with base pattern
                if not np.array_equal(tile, base_pattern):
                    # Find differences
                    diff_mask = tile != base_pattern

                    # Record column-specific rule
                    if tile_col not in rules["column_rules"]:
                        rules["column_rules"][tile_col] = []
                    rules["column_rules"][tile_col].append(
                        {
                            "row": tile_row,
                            "differences": diff_mask,
                            "values": tile[diff_mask],
                        }
                    )

                    # Record row-specific rule
                    if tile_row not in rules["row_rules"]:
                        rules["row_rules"][tile_row] = []
                    rules["row_rules"][tile_row].append(
                        {
                            "col": tile_col,
                            "differences": diff_mask,
                            "values": tile[diff_mask],
                        }
                    )

                    # Record position-specific rule
                    rules["position_rules"][(tile_row, tile_col)] = {
                        "differences": diff_mask,
                        "values": tile[diff_mask],
                    }

        return rules

    def apply_column_row_rules(
        self, base_pattern: np.ndarray, scale: Tuple[int, int], rules: Dict
    ) -> np.ndarray:
        """Apply learned column/row rules to generate output."""
        scale_y, scale_x = int(scale[0]), int(scale[1])
        h, w = base_pattern.shape
        output = np.zeros((h * scale_y, w * scale_x), dtype=base_pattern.dtype)

        # Fill with base pattern first
        for i in range(scale_y):
            for j in range(scale_x):
                output[i * h : (i + 1) * h, j * w : (j + 1) * w] = base_pattern

        # Apply position-specific rules
        for (tile_row, tile_col), rule in rules.get("position_rules", {}).items():
            tile = output[
                tile_row * h : (tile_row + 1) * h, tile_col * w : (tile_col + 1) * w
            ]
            tile[rule["differences"]] = rule["values"]

        return output


class ObjectRelativePositionSolver:
    """Solver for object-relative position patterns."""

    def find_relative_rules(
        self, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Dict:
        """Find rules for adding colors relative to existing objects."""
        rules = {}

        for inp, out in examples:
            # Find new colors in output
            input_colors = set(inp.flatten())
            output_colors = set(out.flatten())
            new_colors = output_colors - input_colors

            if not new_colors:
                continue

            for new_color in new_colors:
                # Find positions of new color
                new_positions = np.argwhere(out == new_color)

                # For each position, find nearest non-zero input
                for pos in new_positions:
                    y, x = pos

                    # Find nearest non-zero in input
                    min_dist = float("inf")
                    nearest_color = None
                    relative_pos = None

                    for iy in range(inp.shape[0]):
                        for ix in range(inp.shape[1]):
                            if inp[iy, ix] != 0:
                                dist = abs(y - iy) + abs(x - ix)
                                if dist < min_dist:
                                    min_dist = dist
                                    nearest_color = inp[iy, ix]
                                    relative_pos = (y - iy, x - ix)

                    # Record rule
                    if nearest_color is not None:
                        key = (nearest_color, new_color)
                        if key not in rules:
                            rules[key] = []
                        rules[key].append(relative_pos)

        # Consolidate rules
        consolidated = {}
        for key, positions in rules.items():
            # Find most common relative position
            from collections import Counter

            pos_counts = Counter(positions)
            if pos_counts:
                most_common = pos_counts.most_common(1)[0][0]
                consolidated[key] = most_common

        return consolidated

    def apply_relative_rules(self, input_grid: np.ndarray, rules: Dict) -> np.ndarray:
        """Apply object-relative position rules."""
        output = input_grid.copy()

        for (source_color, new_color), (dy, dx) in rules.items():
            # Find all positions of source color
            positions = np.argwhere(input_grid == source_color)

            for y, x in positions:
                new_y, new_x = y + dy, x + dx

                # Check bounds
                if 0 <= new_y < output.shape[0] and 0 <= new_x < output.shape[1]:
                    # Only add if currently background
                    if output[new_y, new_x] == 0:
                        output[new_y, new_x] = new_color

        return output


class LocalErrorCorrector:
    """Correct small errors in high-accuracy solutions."""

    def should_correct(self, accuracy: float) -> bool:
        """Check if solution is good enough for local correction."""
        return accuracy >= 0.85

    def find_error_pattern(
        self, predicted: np.ndarray, expected: np.ndarray
    ) -> Optional[Dict]:
        """Find patterns in prediction errors."""
        if predicted.shape != expected.shape:
            return None

        errors = predicted != expected
        error_positions = np.argwhere(errors)

        if len(error_positions) == 0:
            return None

        # Analyze error patterns
        pattern = {
            "positions": error_positions,
            "predicted_values": predicted[errors],
            "expected_values": expected[errors],
            "total_errors": len(error_positions),
            "error_rate": len(error_positions) / expected.size,
        }

        # Check if errors are systematic
        unique_pred = set(predicted[errors])
        unique_exp = set(expected[errors])

        if len(unique_pred) == 1 and len(unique_exp) == 1:
            # Simple substitution error
            pattern["type"] = "substitution"
            pattern["from_value"] = list(unique_pred)[0]
            pattern["to_value"] = list(unique_exp)[0]
        else:
            pattern["type"] = "complex"

        return pattern

    def correct_errors(self, predicted: np.ndarray, error_pattern: Dict) -> np.ndarray:
        """Apply corrections based on error pattern."""
        corrected = predicted.copy()

        if error_pattern["type"] == "substitution":
            # Simple substitution
            mask = predicted == error_pattern["from_value"]
            corrected[mask] = error_pattern["to_value"]
        else:
            # Position-specific corrections
            for i, pos in enumerate(error_pattern["positions"]):
                y, x = pos
                corrected[y, x] = error_pattern["expected_values"][i]

        return corrected


class EnhancedARCSolverV8(EnhancedARCSolverV7):
    """V8 solver with targeted pattern fixes."""

    def __init__(self, **kwargs):
        # Extract V8-specific parameters
        self.use_background_removal = kwargs.pop("use_background_removal", True)
        self.use_enhanced_position = kwargs.pop("use_enhanced_position", True)
        self.use_relative_position = kwargs.pop("use_relative_position", True)
        self.use_error_correction = kwargs.pop("use_error_correction", True)

        # Pass remaining kwargs to parent
        super().__init__(**kwargs)

        # New components
        self.background_solver = BackgroundRemovalSolver()
        self.position_learner = EnhancedPositionLearner()
        self.relative_solver = ObjectRelativePositionSolver()
        self.error_corrector = LocalErrorCorrector()

    def solve(
        self, examples: List[Tuple[np.ndarray, np.ndarray]], test_input: np.ndarray
    ) -> ARCSolutionV8:
        """Solve with V8 enhancements."""

        # Try background removal first (high confidence pattern)
        if self.use_background_removal:
            bg_result = self.background_solver.solve(examples, test_input)
            if bg_result is not None:
                print("  Background removal pattern detected!")
                return ARCSolutionV8(
                    output_grid=bg_result,
                    confidence=0.95,
                    method_used="background_removal",
                )

        # Check for size changes and enhanced position learning
        if examples:
            in_shape = examples[0][0].shape
            out_shape = examples[0][1].shape

            if in_shape != out_shape and self.use_enhanced_position:
                # Size change - try enhanced position learning
                scale = (out_shape[0] / in_shape[0], out_shape[1] / in_shape[1])

                if scale[0] == int(scale[0]) and scale[1] == int(scale[1]):
                    print(f"  Size change: tiling (scale: {scale})")

                    # Learn column/row rules
                    rules = self.position_learner.learn_column_row_rules(
                        examples[0][0], examples[0][1], scale
                    )

                    if rules["position_rules"]:
                        # Apply learned rules
                        result = self.position_learner.apply_column_row_rules(
                            test_input, scale, rules
                        )

                        # Validate on training examples
                        confidence = self._validate_solution(
                            examples,
                            lambda x: self.position_learner.apply_column_row_rules(
                                x, scale, rules
                            ),
                        )

                        if confidence > 0.7:
                            print(
                                f"  Enhanced position learning: {confidence:.2f} confidence"
                            )
                            return ARCSolutionV8(
                                output_grid=result,
                                confidence=confidence,
                                method_used="enhanced_position_tiling",
                            )

        # Try object-relative position rules
        if self.use_relative_position:
            relative_rules = self.relative_solver.find_relative_rules(examples)
            if relative_rules:
                result = self.relative_solver.apply_relative_rules(
                    test_input, relative_rules
                )

                # Validate
                confidence = self._validate_solution(
                    examples,
                    lambda x: self.relative_solver.apply_relative_rules(
                        x, relative_rules
                    ),
                )

                if confidence > 0.7:
                    print(f"  Object-relative rules: {confidence:.2f} confidence")
                    return ARCSolutionV8(
                        output_grid=result,
                        confidence=confidence,
                        method_used="object_relative",
                    )

        # Fall back to V7 solver
        v7_solution = super().solve(examples, test_input)

        # Convert V7 solution to V8
        v8_solution = ARCSolutionV8(
            output_grid=v7_solution.output_grid,
            confidence=v7_solution.confidence,
            method_used=v7_solution.method_used,
            program=v7_solution.program if hasattr(v7_solution, "program") else None,
        )

        # Try error correction if accuracy is high enough
        if self.use_error_correction and examples and v8_solution.confidence > 0.85:
            # Check accuracy on training examples
            for inp, expected in examples[:1]:  # Just check first for speed
                if np.array_equal(inp, test_input):
                    continue

                # Apply same method to training example
                train_result = super().solve(examples, inp)

                if train_result.output_grid.shape == expected.shape:
                    accuracy = (
                        np.sum(train_result.output_grid == expected) / expected.size
                    )

                    if self.error_corrector.should_correct(accuracy):
                        error_pattern = self.error_corrector.find_error_pattern(
                            train_result.output_grid, expected
                        )

                        if error_pattern and error_pattern["type"] == "substitution":
                            # Apply same correction to test
                            corrected = self.error_corrector.correct_errors(
                                v8_solution.output_grid, error_pattern
                            )

                            print(
                                f"  Applied error correction: {error_pattern['type']}"
                            )
                            v8_solution.output_grid = corrected
                            v8_solution.confidence = min(
                                0.95, v8_solution.confidence + 0.1
                            )
                            v8_solution.method_used += "_corrected"
                            v8_solution.error_corrections = error_pattern

        return v8_solution

    def _validate_solution(self, examples, solve_func) -> float:
        """Validate a solution function on training examples."""
        if not examples:
            return 0.0

        total_accuracy = 0.0
        valid_count = 0

        for inp, expected in examples[:3]:  # Check first 3 for speed
            try:
                predicted = solve_func(inp)
                if predicted is not None and predicted.shape == expected.shape:
                    accuracy = np.sum(predicted == expected) / expected.size
                    total_accuracy += accuracy
                    valid_count += 1
            except:
                continue

        return total_accuracy / valid_count if valid_count > 0 else 0.0
