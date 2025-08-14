#!/usr/bin/env python3
"""Enhanced tiling primitives that handle position-dependent modifications."""

from typing import List, Optional, Tuple

import numpy as np


class SmartTilePattern:
    """Smart tiling that can learn modification patterns from examples."""

    def __init__(self):
        self.name = "smart_tile"

    def learn_and_apply(
        self,
        train_examples: List[Tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Learn tiling pattern from examples and apply to test input."""

        if not train_examples:
            return None

        # Detect scale factor
        scale_h = train_examples[0][1].shape[0] / train_examples[0][0].shape[0]
        scale_w = train_examples[0][1].shape[1] / train_examples[0][0].shape[1]

        if scale_h != int(scale_h) or scale_w != int(scale_w):
            return None  # Not a clean scaling

        scale_h, scale_w = int(scale_h), int(scale_w)

        # For each training example, learn the modification pattern
        modification_patterns = []

        for input_grid, output_grid in train_examples:
            # Create base tiling
            base_tile = np.tile(input_grid, (scale_h, scale_w))

            if base_tile.shape != output_grid.shape:
                continue

            # Analyze modifications per tile
            tile_h, tile_w = input_grid.shape
            modifications = {}

            for tile_row in range(scale_h):
                for tile_col in range(scale_w):
                    # Extract tile from output
                    r_start = tile_row * tile_h
                    r_end = r_start + tile_h
                    c_start = tile_col * tile_w
                    c_end = c_start + tile_w

                    output_tile = output_grid[r_start:r_end, c_start:c_end]

                    # Determine transformation
                    if np.array_equal(output_tile, input_grid):
                        modifications[(tile_row, tile_col)] = "identity"
                    elif np.array_equal(output_tile, np.fliplr(input_grid)):
                        modifications[(tile_row, tile_col)] = "flip_h"
                    elif np.array_equal(output_tile, np.flipud(input_grid)):
                        modifications[(tile_row, tile_col)] = "flip_v"
                    elif np.array_equal(output_tile, np.fliplr(np.flipud(input_grid))):
                        modifications[(tile_row, tile_col)] = "flip_both"
                    elif np.array_equal(output_tile, np.rot90(input_grid)):
                        modifications[(tile_row, tile_col)] = "rotate_90"
                    elif np.array_equal(output_tile, np.rot90(input_grid, 2)):
                        modifications[(tile_row, tile_col)] = "rotate_180"
                    elif np.array_equal(output_tile, np.rot90(input_grid, 3)):
                        modifications[(tile_row, tile_col)] = "rotate_270"
                    else:
                        modifications[(tile_row, tile_col)] = "unknown"

            modification_patterns.append(modifications)

        # Check if patterns are consistent across examples
        if not modification_patterns:
            return None

        consistent_pattern = modification_patterns[0]
        for pattern in modification_patterns[1:]:
            if pattern != consistent_pattern:
                # Try to find common pattern
                for key in consistent_pattern:
                    if key in pattern and pattern[key] != consistent_pattern[key]:
                        consistent_pattern[key] = "varies"

        # Apply to test input
        output_h = test_input.shape[0] * scale_h
        output_w = test_input.shape[1] * scale_w
        output = np.zeros((output_h, output_w), dtype=test_input.dtype)

        tile_h, tile_w = test_input.shape

        for tile_row in range(scale_h):
            for tile_col in range(scale_w):
                r_start = tile_row * tile_h
                r_end = r_start + tile_h
                c_start = tile_col * tile_w
                c_end = c_start + tile_w

                transformation = consistent_pattern.get(
                    (tile_row, tile_col), "identity"
                )

                if transformation == "identity":
                    output[r_start:r_end, c_start:c_end] = test_input
                elif transformation == "flip_h":
                    output[r_start:r_end, c_start:c_end] = np.fliplr(test_input)
                elif transformation == "flip_v":
                    output[r_start:r_end, c_start:c_end] = np.flipud(test_input)
                elif transformation == "flip_both":
                    output[r_start:r_end, c_start:c_end] = np.fliplr(
                        np.flipud(test_input)
                    )
                elif transformation == "rotate_90":
                    # Handle rotation with potential size mismatch
                    rotated = np.rot90(test_input)
                    if rotated.shape == (tile_h, tile_w):
                        output[r_start:r_end, c_start:c_end] = rotated
                    else:
                        output[r_start:r_end, c_start:c_end] = test_input
                elif transformation == "rotate_180":
                    output[r_start:r_end, c_start:c_end] = np.rot90(test_input, 2)
                elif transformation == "rotate_270":
                    rotated = np.rot90(test_input, 3)
                    if rotated.shape == (tile_h, tile_w):
                        output[r_start:r_end, c_start:c_end] = rotated
                    else:
                        output[r_start:r_end, c_start:c_end] = test_input
                else:
                    # Default to identity for unknown
                    output[r_start:r_end, c_start:c_end] = test_input

        return output


class AlternatingRowTile:
    """Specific pattern: Original-Flipped-Original for 3x3 tiling."""

    def __init__(self):
        self.name = "alternating_row_tile"

    def execute(self, input_grid: np.ndarray, scale: int = 3) -> np.ndarray:
        """Execute alternating row tile pattern."""

        if scale != 3:
            # For now, only handle 3x3 case
            return np.tile(input_grid, (scale, scale))

        # Create output
        h, w = input_grid.shape
        output = np.zeros((h * 3, w * 3), dtype=input_grid.dtype)

        # Pattern: Original-Flipped-Original
        flipped = np.fliplr(input_grid)

        # Top row of tiles (Original)
        output[0:h, :] = np.tile(input_grid, (1, 3))

        # Middle row of tiles (Flipped)
        output[h : 2 * h, :] = np.tile(flipped, (1, 3))

        # Bottom row of tiles (Original)
        output[2 * h : 3 * h, :] = np.tile(input_grid, (1, 3))

        return output


class CheckerboardTile:
    """Checkerboard pattern of transformations."""

    def __init__(self):
        self.name = "checkerboard_tile"

    def execute(self, input_grid: np.ndarray, scale: int = 3) -> np.ndarray:
        """Execute checkerboard tile pattern."""

        h, w = input_grid.shape
        output = np.zeros((h * scale, w * scale), dtype=input_grid.dtype)

        flipped_h = np.fliplr(input_grid)

        for i in range(scale):
            for j in range(scale):
                r_start = i * h
                r_end = r_start + h
                c_start = j * w
                c_end = c_start + w

                # Checkerboard pattern
                if (i + j) % 2 == 0:
                    output[r_start:r_end, c_start:c_end] = input_grid
                else:
                    output[r_start:r_end, c_start:c_end] = flipped_h

        return output


def test_enhanced_tiles():
    """Test the enhanced tiling primitives."""

    # Test input
    input_grid = np.array([[3, 2], [7, 8]])

    # Expected for task 00576224
    expected = np.array(
        [
            [3, 2, 3, 2, 3, 2],
            [7, 8, 7, 8, 7, 8],
            [2, 3, 2, 3, 2, 3],
            [8, 7, 8, 7, 8, 7],
            [3, 2, 3, 2, 3, 2],
            [7, 8, 7, 8, 7, 8],
        ]
    )

    print("Testing Enhanced Tiling Primitives")
    print("=" * 60)
    print("Input:")
    print(input_grid)
    print("\nExpected:")
    print(expected)

    # Test AlternatingRowTile
    print("\n" + "=" * 60)
    print("AlternatingRowTile")
    alt_tile = AlternatingRowTile()
    output = alt_tile.execute(input_grid)
    print("Output:")
    print(output)
    print(f"Matches expected: {np.array_equal(output, expected)}")

    # Test SmartTilePattern with learning
    print("\n" + "=" * 60)
    print("SmartTilePattern (with learning)")
    smart_tile = SmartTilePattern()

    # Create training example
    train_examples = [(input_grid, expected)]
    output2 = smart_tile.learn_and_apply(train_examples, input_grid)

    if output2 is not None:
        print("Output:")
        print(output2)
        print(f"Matches expected: {np.array_equal(output2, expected)}")
    else:
        print("Failed to learn pattern")

    # Test CheckerboardTile
    print("\n" + "=" * 60)
    print("CheckerboardTile")
    checker_tile = CheckerboardTile()
    output3 = checker_tile.execute(input_grid)
    print("Output:")
    print(output3)
    print(f"Matches expected: {np.array_equal(output3, expected)}")


if __name__ == "__main__":
    test_enhanced_tiles()
