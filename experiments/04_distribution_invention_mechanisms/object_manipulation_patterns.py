#!/usr/bin/env python3
"""Object manipulation patterns for ARC tasks.

Implements patterns for:
- Object counting
- Object sorting (by size, position, color)
- Object duplication
- Object movement/rearrangement
"""

from utils.imports import setup_project_paths

setup_project_paths()


import numpy as np
from compositional_dsl import ExecutionContext, Primitive
from scipy import ndimage


class ObjectCounter(Primitive):
    """Count objects and encode the count in the output."""

    def __init__(self, encode_as_color: bool = True):
        self.encode_as_color = encode_as_color

    def execute(self, context: ExecutionContext) -> ExecutionContext:
        result = context.copy()
        grid = context.input_grid.copy()

        # Count connected components for each color
        counts = {}
        for color in np.unique(grid):
            if color != 0:
                mask = grid == color
                labeled, num = ndimage.label(mask)
                counts[color] = num

        # Create output based on counts
        if self.encode_as_color:
            # Simple encoding: fill with count value
            output = np.zeros_like(grid)
            for color, count in counts.items():
                # Put count in specific position (e.g., top-left)
                if count < 10:  # Only handle single digit counts
                    output[0, color - 1] = count
        else:
            # Alternative: create count visualization
            output = grid.copy()

        result.current_grid = output
        return result

    def __str__(self):
        return f"ObjectCounter(encode_as_color={self.encode_as_color})"


class SortBySize(Primitive):
    """Sort objects by size and arrange them."""

    def __init__(self, direction: str = "horizontal"):
        self.direction = direction  # horizontal or vertical

    def execute(self, context: ExecutionContext) -> ExecutionContext:
        result = context.copy()
        grid = context.input_grid.copy()
        h, w = grid.shape

        # Extract objects
        objects = []
        for color in np.unique(grid):
            if color != 0:
                mask = grid == color
                labeled, num = ndimage.label(mask)
                for i in range(1, num + 1):
                    obj_mask = labeled == i
                    size = np.sum(obj_mask)
                    # Get bounding box
                    positions = np.argwhere(obj_mask)
                    min_r, min_c = positions.min(axis=0)
                    max_r, max_c = positions.max(axis=0)

                    objects.append(
                        {
                            "color": color,
                            "size": size,
                            "mask": obj_mask,
                            "bbox": (min_r, min_c, max_r, max_c),
                            "positions": positions,
                        }
                    )

        # Sort by size
        objects.sort(key=lambda x: x["size"])

        # Create output grid with sorted arrangement
        output = np.zeros_like(grid)

        if self.direction == "horizontal":
            current_col = 0
            for obj in objects:
                min_r, min_c, max_r, max_c = obj["bbox"]
                height = max_r - min_r + 1
                width = max_c - min_c + 1

                if current_col + width <= w:
                    # Place object
                    for pos in obj["positions"]:
                        r, c = pos
                        new_r = r - min_r
                        new_c = current_col + (c - min_c)
                        if new_r < h and new_c < w:
                            output[new_r, new_c] = obj["color"]

                    current_col += width + 1
        else:  # vertical
            current_row = 0
            for obj in objects:
                min_r, min_c, max_r, max_c = obj["bbox"]
                height = max_r - min_r + 1
                width = max_c - min_c + 1

                if current_row + height <= h:
                    # Place object
                    for pos in obj["positions"]:
                        r, c = pos
                        new_r = current_row + (r - min_r)
                        new_c = c - min_c
                        if new_r < h and new_c < w:
                            output[new_r, new_c] = obj["color"]

                    current_row += height + 1

        result.current_grid = output
        return result

    def __str__(self):
        return f"SortBySize(direction='{self.direction}')"


class DuplicateObjects(Primitive):
    """Duplicate objects in a pattern."""

    def __init__(self, duplication_factor: int = 2, pattern: str = "horizontal"):
        self.duplication_factor = duplication_factor
        self.pattern = pattern  # horizontal, vertical, diagonal

    def execute(self, context: ExecutionContext) -> ExecutionContext:
        result = context.copy()
        grid = context.input_grid.copy()
        h, w = grid.shape

        # Extract objects
        objects = []
        for color in np.unique(grid):
            if color != 0:
                mask = grid == color
                labeled, num = ndimage.label(mask)
                for i in range(1, num + 1):
                    obj_mask = labeled == i
                    positions = np.argwhere(obj_mask)
                    min_r, min_c = positions.min(axis=0)
                    max_r, max_c = positions.max(axis=0)

                    objects.append(
                        {
                            "color": color,
                            "mask": obj_mask,
                            "positions": positions
                            - [min_r, min_c],  # Relative positions
                            "height": max_r - min_r + 1,
                            "width": max_c - min_c + 1,
                            "orig_pos": (min_r, min_c),
                        }
                    )

        # Create output with duplicates
        output = grid.copy()

        for obj in objects:
            orig_r, orig_c = obj["orig_pos"]

            for i in range(1, self.duplication_factor):
                if self.pattern == "horizontal":
                    offset_r = 0
                    offset_c = (obj["width"] + 1) * i
                elif self.pattern == "vertical":
                    offset_r = (obj["height"] + 1) * i
                    offset_c = 0
                elif self.pattern == "diagonal":
                    offset_r = (obj["height"] + 1) * i
                    offset_c = (obj["width"] + 1) * i
                else:
                    continue

                # Place duplicate
                for rel_pos in obj["positions"]:
                    r, c = rel_pos
                    new_r = orig_r + r + offset_r
                    new_c = orig_c + c + offset_c

                    if 0 <= new_r < h and 0 <= new_c < w:
                        output[new_r, new_c] = obj["color"]

        result.current_grid = output
        return result

    def __str__(self):
        return f"DuplicateObjects(factor={self.duplication_factor}, pattern='{self.pattern}')"


class MoveObjects(Primitive):
    """Move objects based on a rule."""

    def __init__(self, direction: str = "right", distance: int = 1):
        self.direction = direction  # up, down, left, right
        self.distance = distance

    def execute(self, context: ExecutionContext) -> ExecutionContext:
        result = context.copy()
        grid = context.input_grid.copy()
        h, w = grid.shape

        # Determine offset
        if self.direction == "up":
            offset = (-self.distance, 0)
        elif self.direction == "down":
            offset = (self.distance, 0)
        elif self.direction == "left":
            offset = (0, -self.distance)
        elif self.direction == "right":
            offset = (0, self.distance)
        else:
            offset = (0, 0)

        # Create output
        output = np.zeros_like(grid)

        for r in range(h):
            for c in range(w):
                if grid[r, c] != 0:
                    new_r = r + offset[0]
                    new_c = c + offset[1]

                    if 0 <= new_r < h and 0 <= new_c < w:
                        output[new_r, new_c] = grid[r, c]

        result.current_grid = output
        return result

    def __str__(self):
        return f"MoveObjects(direction='{self.direction}', distance={self.distance})"


class ExtractLargestObject(Primitive):
    """Extract only the largest object of each color."""

    def execute(self, context: ExecutionContext) -> ExecutionContext:
        result = context.copy()
        grid = context.input_grid.copy()

        output = np.zeros_like(grid)

        for color in np.unique(grid):
            if color != 0:
                mask = grid == color
                labeled, num = ndimage.label(mask)

                # Find largest component
                max_size = 0
                max_label = 0

                for i in range(1, num + 1):
                    size = np.sum(labeled == i)
                    if size > max_size:
                        max_size = size
                        max_label = i

                # Keep only largest
                if max_label > 0:
                    output[labeled == max_label] = color

        result.current_grid = output
        return result

    def __str__(self):
        return "ExtractLargestObject()"


def test_object_patterns():
    """Test object manipulation patterns."""

    print("Testing Object Manipulation Patterns")
    print("=" * 60)

    # Create test grid with objects
    test_grid = np.array(
        [
            [0, 1, 1, 0, 0, 0, 2, 0],
            [0, 1, 1, 0, 0, 2, 2, 2],
            [0, 0, 0, 0, 0, 0, 2, 0],
            [3, 3, 0, 0, 0, 0, 0, 0],
            [3, 3, 0, 0, 4, 0, 0, 0],
            [3, 3, 0, 4, 4, 4, 0, 0],
            [0, 0, 0, 0, 4, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )

    from compositional_dsl import ExecutionContext

    patterns = [
        ObjectCounter(encode_as_color=True),
        SortBySize(direction="horizontal"),
        DuplicateObjects(duplication_factor=2, pattern="horizontal"),
        MoveObjects(direction="right", distance=2),
        ExtractLargestObject(),
    ]

    for pattern in patterns:
        print(f"\nTesting {pattern}:")

        context = ExecutionContext(
            input_grid=test_grid.copy(), current_grid=test_grid.copy()
        )
        result_context = pattern.execute(context)
        result = result_context.current_grid

        print(f"  Input shape: {test_grid.shape}")
        print(f"  Output shape: {result.shape}")
        print(f"  Input colors: {np.unique(test_grid)}")
        print(f"  Output colors: {np.unique(result)}")

        # Check if transformation worked
        if not np.array_equal(result, test_grid):
            print("  ✓ Transformation applied")
        else:
            print("  ✗ No change")


if __name__ == "__main__":
    test_object_patterns()
