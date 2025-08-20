#!/usr/bin/env python3
"""Implement a cross pattern primitive for ARC tasks.

Based on analysis of task ae3edfdc, this primitive creates cross patterns
around center pixels when certain conditions are met.
"""

from utils.imports import setup_project_paths

setup_project_paths()

from typing import Optional

import numpy as np
from compositional_dsl import ExecutionContext, Primitive


class FormCrossPattern(Primitive):
    """Form a cross pattern around a center pixel when conditions are met.

    The pattern appears to be:
    - Find a "center" color (1 or 2)
    - Check if there are two "marker" colors (3 or 7) in line with it
    - If so, form a cross pattern with the marker color around the center
    """

    def __init__(
        self, center_colors: Optional[list] = None, marker_colors: Optional[list] = None
    ):
        self.center_colors = center_colors or [1, 2]
        self.marker_colors = marker_colors or [3, 7]

    def execute(self, context: ExecutionContext) -> ExecutionContext:
        """Form cross patterns based on spatial relationships."""
        result = context.copy()
        grid = result.current_grid.copy()
        h, w = grid.shape

        # Find potential center pixels
        for i in range(h):
            for j in range(w):
                if grid[i, j] in self.center_colors:
                    center_color = grid[i, j]

                    # Check each marker color
                    for marker_color in self.marker_colors:
                        # Check horizontal line
                        horizontal_markers = []
                        for jj in range(w):
                            if jj != j and grid[i, jj] == marker_color:
                                horizontal_markers.append(jj)

                        # Check vertical line
                        vertical_markers = []
                        for ii in range(h):
                            if ii != i and grid[ii, j] == marker_color:
                                vertical_markers.append(ii)

                        # Form cross if we have markers on both sides
                        # OR in perpendicular directions
                        should_form_cross = False

                        # Check horizontal: markers on both sides
                        if len(horizontal_markers) >= 2:
                            left = [jj for jj in horizontal_markers if jj < j]
                            right = [jj for jj in horizontal_markers if jj > j]
                            if left and right:
                                should_form_cross = True

                        # Check vertical: markers on both sides
                        if len(vertical_markers) >= 2:
                            above = [ii for ii in vertical_markers if ii < i]
                            below = [ii for ii in vertical_markers if ii > i]
                            if above and below:
                                should_form_cross = True

                        # Check perpendicular: one horizontal, one vertical
                        if horizontal_markers and vertical_markers:
                            should_form_cross = True

                        if should_form_cross:
                            # Form the cross pattern
                            # Keep center
                            grid[i, j] = center_color

                            # Add cross arms (one pixel in each direction)
                            if i > 0:
                                grid[i - 1, j] = marker_color
                            if i < h - 1:
                                grid[i + 1, j] = marker_color
                            if j > 0:
                                grid[i, j - 1] = marker_color
                            if j < w - 1:
                                grid[i, j + 1] = marker_color

                            # Clear the original markers
                            for jj in horizontal_markers:
                                if abs(jj - j) > 1:  # Don't clear adjacent
                                    grid[i, jj] = 0
                            for ii in vertical_markers:
                                if abs(ii - i) > 1:  # Don't clear adjacent
                                    grid[ii, j] = 0

                            break  # Only form one cross per center

        result.current_grid = grid
        return result

    def __str__(self):
        return f"FormCrossPattern(centers={self.center_colors}, markers={self.marker_colors})"


def test_cross_pattern():
    """Test the cross pattern primitive on task ae3edfdc."""
    import json
    from pathlib import Path

    # Load task
    data_dir = Path("data/arc_agi_official/ARC-AGI/data/training")
    task_file = data_dir / f"ae3edfdc.json"

    with open(task_file, "r") as f:
        task = json.load(f)

    # Get examples
    train_examples = [
        (np.array(ex["input"]), np.array(ex["output"])) for ex in task["train"]
    ]

    print("Testing FormCrossPattern on ae3edfdc:")

    primitive = FormCrossPattern()

    for i, (inp, expected) in enumerate(train_examples):
        context = ExecutionContext(input_grid=inp.copy(), current_grid=inp.copy())

        result_context = primitive.execute(context)
        result = result_context.current_grid

        # Check accuracy
        accuracy = np.mean(result == expected)
        matches = np.sum(result == expected)
        total = result.size

        print(f"\nExample {i+1}:")
        print(f"  Accuracy: {accuracy:.1%} ({matches}/{total} pixels)")

        if accuracy == 1.0:
            print("  ✅ PERFECT MATCH!")
        elif accuracy > 0.95:
            print("  🎯 Very close!")

        # Show differences
        if accuracy < 1.0:
            diff_mask = result != expected
            diff_count = np.sum(diff_mask)
            print(f"  Differences: {diff_count} pixels")

            # Analyze differences
            for ii in range(min(5, result.shape[0])):  # Show first 5 rows
                for jj in range(min(15, result.shape[1])):
                    if diff_mask[ii, jj]:
                        print(
                            f"    ({ii},{jj}): {result[ii,jj]} should be {expected[ii,jj]}"
                        )


if __name__ == "__main__":
    test_cross_pattern()
