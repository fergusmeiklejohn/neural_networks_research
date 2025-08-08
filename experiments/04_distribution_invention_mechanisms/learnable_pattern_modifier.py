#!/usr/bin/env python3
"""Learnable Pattern Modifier for ARC-AGI.

This module learns how patterns are modified in ARC tasks by analyzing
differences between simple transformations and expected outputs.
"""

from utils.imports import setup_project_paths

setup_project_paths()

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

import numpy as np


@dataclass
class ModificationRule:
    """A learned modification rule."""

    name: str
    condition: Callable[[np.ndarray, int, int], bool]  # (grid, row, col) -> bool
    action: Callable[[np.ndarray, int, int], Any]  # (grid, row, col) -> value
    confidence: float
    examples_matched: int


class LearnablePatternModifier:
    """Learns and applies pattern modifications from examples."""

    def __init__(self):
        self.learned_rules = []

    def learn_modifications(
        self,
        base_pattern: np.ndarray,
        expected_output: np.ndarray,
        pattern_type: str = "tiling",
    ) -> List[ModificationRule]:
        """Learn modification rules by comparing base pattern with expected output.

        Args:
            base_pattern: The simple/base transformation result
            expected_output: The expected output with modifications
            pattern_type: Type of pattern (e.g., 'tiling', 'scaling')

        Returns:
            List of learned modification rules
        """
        if base_pattern.shape != expected_output.shape:
            return []

        rules = []

        # Analyze differences
        diff_mask = base_pattern != expected_output
        diff_positions = np.argwhere(diff_mask)

        if len(diff_positions) == 0:
            # No modifications needed
            return []

        # For tiling patterns, analyze modifications by tile position
        if pattern_type == "tiling":
            rules.extend(
                self._learn_tiling_modifications(
                    base_pattern, expected_output, diff_positions
                )
            )

        # Learn position-based rules
        rules.extend(
            self._learn_position_rules(base_pattern, expected_output, diff_positions)
        )

        # Learn region-based rules
        rules.extend(
            self._learn_region_rules(base_pattern, expected_output, diff_positions)
        )

        return rules

    def _learn_tiling_modifications(
        self,
        base_pattern: np.ndarray,
        expected_output: np.ndarray,
        diff_positions: np.ndarray,
    ) -> List[ModificationRule]:
        """Learn modifications specific to tiling patterns."""
        rules = []
        h, w = base_pattern.shape

        # Detect tile size (assume square tiles for now)
        # Look for repeating patterns in the base
        tile_size = self._detect_tile_size(base_pattern)

        if tile_size is None:
            # If we can't detect tile size from base, try to infer from dimensions
            # For 3x3 -> 9x9, tile size is 3
            if h == w and h % 3 == 0:
                tile_size = (3, 3)
            else:
                return rules

        tile_h, tile_w = tile_size

        # Group differences by tile position
        tile_modifications = {}
        for r, c in diff_positions:
            tile_row = r // tile_h
            tile_col = c // tile_w
            local_r = r % tile_h
            local_c = c % tile_w

            key = (tile_row, tile_col)
            if key not in tile_modifications:
                tile_modifications[key] = []
            tile_modifications[key].append((local_r, local_c, expected_output[r, c]))

        # Analyze patterns in modifications
        # Check if left column tiles are modified
        num_tile_rows = h // tile_h
        w // tile_w

        left_col_modified = any(
            (tr, 0) in tile_modifications for tr in range(num_tile_rows)
        )

        if left_col_modified:
            # Check if it's zeroing the first part of each row in left tiles
            # For 007bbfb7, it zeros the first 3 columns (the entire first tile width)
            all_zeros = True
            zero_cols = set()

            for tr in range(num_tile_rows):
                if (tr, 0) in tile_modifications:
                    for local_r, local_c, val in tile_modifications[(tr, 0)]:
                        actual_col = local_c  # Column within the tile
                        zero_cols.add(actual_col)
                        if val != 0:
                            all_zeros = False

            if all_zeros and zero_cols:
                # Determine which columns to zero
                max_zero_col = max(zero_cols) + 1  # +1 because 0-indexed

                # Create rule for zeroing first N columns in left tiles
                def make_condition(tile_w, max_col):
                    def condition(grid, r, c):
                        # Check if we're in the leftmost tile column
                        # and within the first max_col columns
                        return c < max_col

                    return condition

                def action(grid, r, c):
                    return 0

                rule = ModificationRule(
                    name=f"zero_first_{max_zero_col}_cols",
                    condition=make_condition(tile_w, max_zero_col),
                    action=action,
                    confidence=0.9,
                    examples_matched=1,
                )
                rules.append(rule)

        # Check for row-specific modifications
        row_patterns = {}
        for (tr, tc), mods in tile_modifications.items():
            if tr not in row_patterns:
                row_patterns[tr] = []
            row_patterns[tr].extend(mods)

        # Add more sophisticated pattern detection here
        # For now, we'll focus on the simple patterns

        return rules

    def _learn_position_rules(
        self,
        base_pattern: np.ndarray,
        expected_output: np.ndarray,
        diff_positions: np.ndarray,
    ) -> List[ModificationRule]:
        """Learn rules based on absolute positions."""
        rules = []
        h, w = base_pattern.shape

        # Check for column-based modifications
        col_mods = {}
        for r, c in diff_positions:
            if c not in col_mods:
                col_mods[c] = []
            col_mods[c].append((r, expected_output[r, c]))

        # If certain columns are consistently modified
        for col, mods in col_mods.items():
            if len(mods) >= h * 0.8:  # At least 80% of column modified
                # Check if all modifications are the same
                values = [val for _, val in mods]
                if len(set(values)) == 1:  # All same value
                    target_val = values[0]

                    def make_condition(target_col):
                        return lambda grid, r, c: c == target_col

                    def make_action(val):
                        return lambda grid, r, c: val

                    rule = ModificationRule(
                        name=f"column_{col}_to_{target_val}",
                        condition=make_condition(col),
                        action=make_action(target_val),
                        confidence=0.8,
                        examples_matched=1,
                    )
                    rules.append(rule)

        return rules

    def _learn_region_rules(
        self,
        base_pattern: np.ndarray,
        expected_output: np.ndarray,
        diff_positions: np.ndarray,
    ) -> List[ModificationRule]:
        """Learn rules based on regions (e.g., corners, edges)."""
        rules = []
        h, w = base_pattern.shape

        # Check for border modifications
        border_positions = set()
        for r, c in diff_positions:
            if r == 0 or r == h - 1 or c == 0 or c == w - 1:
                border_positions.add((r, c))

        if len(border_positions) > (2 * h + 2 * w - 4) * 0.5:  # >50% of border
            # Border is being modified
            border_values = [expected_output[r, c] for r, c in border_positions]
            if len(set(border_values)) == 1:  # Uniform border modification
                target_val = border_values[0]

                def condition(grid, r, c):
                    h, w = grid.shape
                    return r == 0 or r == h - 1 or c == 0 or c == w - 1

                def action(grid, r, c):
                    return target_val

                rule = ModificationRule(
                    name=f"border_to_{target_val}",
                    condition=condition,
                    action=action,
                    confidence=0.7,
                    examples_matched=1,
                )
                rules.append(rule)

        return rules

    def _detect_tile_size(self, pattern: np.ndarray) -> Optional[Tuple[int, int]]:
        """Detect the size of tiles in a tiled pattern."""
        h, w = pattern.shape

        # Try common tile sizes
        for tile_size in [3, 4, 5, 2, 6]:
            if h % tile_size == 0 and w % tile_size == 0:
                # Check if pattern repeats
                tile_h = h // (h // tile_size)
                tile_w = w // (w // tile_size)

                # Extract first tile
                first_tile = pattern[:tile_h, :tile_w]

                # Check if it repeats
                repeats = True
                for i in range(0, h, tile_h):
                    for j in range(0, w, tile_w):
                        if i == 0 and j == 0:
                            continue
                        tile = pattern[i : i + tile_h, j : j + tile_w]
                        if not np.array_equal(tile, first_tile):
                            # Allow for some modifications
                            if np.sum(tile != first_tile) > tile.size * 0.5:
                                repeats = False
                                break
                    if not repeats:
                        break

                if repeats:
                    return (tile_h, tile_w)

        return None

    def apply_modifications(
        self, base_pattern: np.ndarray, rules: List[ModificationRule]
    ) -> np.ndarray:
        """Apply learned modification rules to a base pattern.

        Args:
            base_pattern: The base/simple transformation result
            rules: List of modification rules to apply

        Returns:
            Modified pattern
        """
        result = base_pattern.copy()

        # Sort rules by confidence
        sorted_rules = sorted(rules, key=lambda r: r.confidence, reverse=True)

        # Apply rules
        for rule in sorted_rules:
            h, w = result.shape
            for r in range(h):
                for c in range(w):
                    if rule.condition(result, r, c):
                        result[r, c] = rule.action(result, r, c)

        return result

    def learn_from_examples(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
        pattern_type: str = "tiling",
    ) -> List[ModificationRule]:
        """Learn modification rules from multiple examples.

        Args:
            examples: List of (input, base_output, expected_output) tuples
            pattern_type: Type of pattern

        Returns:
            Consolidated list of learned rules
        """
        all_rules = {}

        for input_grid, base_output, expected_output in examples:
            rules = self.learn_modifications(base_output, expected_output, pattern_type)

            # Consolidate rules
            for rule in rules:
                if rule.name not in all_rules:
                    all_rules[rule.name] = rule
                else:
                    # Update confidence based on multiple examples
                    existing = all_rules[rule.name]
                    existing.examples_matched += 1
                    existing.confidence = min(
                        0.95, existing.confidence + 0.1
                    )  # Increase confidence

        return list(all_rules.values())


def test_learnable_modifier():
    """Test the learnable pattern modifier."""
    print("Testing Learnable Pattern Modifier")
    print("=" * 60)

    modifier = LearnablePatternModifier()

    # Test case 1: Simple tiling with left column zeroing
    print("\nTest 1: Learning left column zeroing")

    # Create a simple pattern
    base = np.array(
        [
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2],
        ]
    )

    # Expected has left columns zeroed
    expected = np.array(
        [
            [0, 0, 0, 1, 1, 1],
            [0, 0, 0, 1, 1, 1],
            [0, 0, 0, 2, 2, 2],
            [0, 0, 0, 2, 2, 2],
        ]
    )

    # Learn modifications
    rules = modifier.learn_modifications(base, expected, "tiling")

    print(f"Learned {len(rules)} rules:")
    for rule in rules:
        print(f"  - {rule.name} (confidence: {rule.confidence:.2f})")

    # Apply learned rules to a new pattern
    new_base = np.array(
        [
            [3, 3, 3, 3, 3, 3],
            [3, 3, 3, 3, 3, 3],
            [4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4],
        ]
    )

    modified = modifier.apply_modifications(new_base, rules)
    print(f"\nNew base pattern:\n{new_base}")
    print(f"\nModified pattern:\n{modified}")

    # Test case 2: Border modification
    print("\n" + "-" * 60)
    print("Test 2: Learning border modifications")

    base2 = np.array(
        [
            [1, 1, 1, 1],
            [1, 2, 2, 1],
            [1, 2, 2, 1],
            [1, 1, 1, 1],
        ]
    )

    expected2 = np.array(
        [
            [5, 5, 5, 5],
            [5, 2, 2, 5],
            [5, 2, 2, 5],
            [5, 5, 5, 5],
        ]
    )

    rules2 = modifier.learn_modifications(base2, expected2, "pattern")
    print(f"Learned {len(rules2)} rules:")
    for rule in rules2:
        print(f"  - {rule.name} (confidence: {rule.confidence:.2f})")

    print("\n" + "=" * 60)
    print("Learnable modifier tests complete!")


if __name__ == "__main__":
    test_learnable_modifier()
