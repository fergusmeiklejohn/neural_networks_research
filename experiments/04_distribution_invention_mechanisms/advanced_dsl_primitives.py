#!/usr/bin/env python3
"""Advanced DSL primitives for ARC-AGI tasks.

Extends the compositional DSL with sophisticated operations needed for real ARC tasks:
- Flood fill and boundary detection
- Connected component analysis
- Shape recognition and manipulation
- Conditional operations based on spatial properties
"""

from utils.imports import setup_project_paths

setup_project_paths()

from typing import Optional, Tuple

import numpy as np
from compositional_dsl import ExecutionContext, Primitive
from scipy import ndimage


class FloodFill(Primitive):
    """Flood fill from a starting point with a given color."""

    def __init__(self, start_x: int, start_y: int, fill_color: int):
        self.start_x = start_x
        self.start_y = start_y
        self.fill_color = fill_color

    def execute(self, context: ExecutionContext) -> ExecutionContext:
        grid = context.current_grid.copy()
        h, w = grid.shape

        # Check bounds
        if not (0 <= self.start_y < h and 0 <= self.start_x < w):
            return context

        original_color = grid[self.start_y, self.start_x]
        if original_color == self.fill_color:
            return context

        # BFS flood fill
        queue = [(self.start_x, self.start_y)]
        visited = set()

        while queue:
            x, y = queue.pop(0)
            if (x, y) in visited:
                continue
            if not (0 <= y < h and 0 <= x < w):
                continue
            if grid[y, x] != original_color:
                continue

            visited.add((x, y))
            grid[y, x] = self.fill_color

            # Add neighbors
            queue.extend([(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)])

        context.current_grid = grid
        return context

    def __str__(self) -> str:
        return f"FloodFill({self.start_x}, {self.start_y}, color={self.fill_color})"


class FillInterior(Primitive):
    """Fill the interior of closed shapes formed by a boundary color."""

    def __init__(self, boundary_color: int, fill_color: int):
        self.boundary_color = boundary_color
        self.fill_color = fill_color

    def execute(self, context: ExecutionContext) -> ExecutionContext:
        grid = context.current_grid.copy()
        h, w = grid.shape

        # Find all regions enclosed by boundary_color
        boundary_mask = grid == self.boundary_color

        # Use connected component labeling on non-boundary pixels
        non_boundary = ~boundary_mask
        labeled, num_features = ndimage.label(non_boundary)

        # For each component, check if it touches the edge
        for label_id in range(1, num_features + 1):
            component_mask = labeled == label_id

            # Check if this component touches any edge
            touches_edge = (
                np.any(component_mask[0, :])
                or np.any(component_mask[-1, :])
                or np.any(component_mask[:, 0])
                or np.any(component_mask[:, -1])
            )

            # If it doesn't touch the edge, it's interior
            if not touches_edge:
                grid[component_mask] = self.fill_color

        context.current_grid = grid
        return context

    def __str__(self) -> str:
        return f"FillInterior(boundary={self.boundary_color}, fill={self.fill_color})"


class ExtractLargestObject(Primitive):
    """Extract the largest connected component of a given color."""

    def __init__(self, target_color: int):
        self.target_color = target_color

    def execute(self, context: ExecutionContext) -> ExecutionContext:
        grid = context.current_grid.copy()

        # Find connected components of target color
        mask = grid == self.target_color
        labeled, num_features = ndimage.label(mask)

        if num_features == 0:
            return context

        # Find largest component
        largest_size = 0
        largest_label = 0

        for label_id in range(1, num_features + 1):
            size = np.sum(labeled == label_id)
            if size > largest_size:
                largest_size = size
                largest_label = label_id

        # Keep only largest component
        new_grid = np.zeros_like(grid)
        new_grid[labeled == largest_label] = self.target_color

        context.current_grid = new_grid
        return context

    def __str__(self) -> str:
        return f"ExtractLargestObject(color={self.target_color})"


class ConnectPoints(Primitive):
    """Connect all points of a given color with lines."""

    def __init__(self, point_color: int, line_color: Optional[int] = None):
        self.point_color = point_color
        self.line_color = line_color if line_color is not None else point_color

    def execute(self, context: ExecutionContext) -> ExecutionContext:
        grid = context.current_grid.copy()

        # Find all points
        points = np.argwhere(grid == self.point_color)

        if len(points) < 2:
            return context

        # Connect consecutive points
        for i in range(len(points) - 1):
            y1, x1 = points[i]
            y2, x2 = points[i + 1]

            # Draw line using Bresenham's algorithm (simplified)
            steps = max(abs(x2 - x1), abs(y2 - y1))
            if steps > 0:
                for t in range(steps + 1):
                    x = round(x1 + t * (x2 - x1) / steps)
                    y = round(y1 + t * (y2 - y1) / steps)
                    if 0 <= y < grid.shape[0] and 0 <= x < grid.shape[1]:
                        grid[y, x] = self.line_color

        context.current_grid = grid
        return context

    def __str__(self) -> str:
        return f"ConnectPoints(point_color={self.point_color}, line_color={self.line_color})"


class DetectPattern(Primitive):
    """Detect and mark repeating patterns in the grid."""

    def __init__(self, pattern_size: Tuple[int, int], mark_color: int):
        self.pattern_size = pattern_size
        self.mark_color = mark_color

    def execute(self, context: ExecutionContext) -> ExecutionContext:
        grid = context.current_grid.copy()
        h, w = grid.shape
        ph, pw = self.pattern_size

        # Store detected patterns
        patterns = {}

        # Scan for patterns
        for y in range(h - ph + 1):
            for x in range(w - pw + 1):
                pattern = grid[y : y + ph, x : x + pw]
                pattern_tuple = tuple(pattern.flatten())

                if pattern_tuple not in patterns:
                    patterns[pattern_tuple] = []
                patterns[pattern_tuple].append((y, x))

        # Mark repeated patterns
        for pattern_tuple, locations in patterns.items():
            if len(locations) > 1:  # Pattern appears more than once
                for y, x in locations[1:]:  # Mark all but first occurrence
                    grid[y : y + ph, x : x + pw] = self.mark_color

        context.current_grid = grid
        return context

    def __str__(self) -> str:
        return f"DetectPattern(size={self.pattern_size}, mark={self.mark_color})"


class CropToContent(Primitive):
    """Crop the grid to remove empty borders (background color)."""

    def __init__(self, background_color: int = 0):
        self.background_color = background_color

    def execute(self, context: ExecutionContext) -> ExecutionContext:
        grid = context.current_grid

        # Find non-background pixels
        non_bg = grid != self.background_color
        if not np.any(non_bg):
            return context

        # Find bounding box
        rows = np.any(non_bg, axis=1)
        cols = np.any(non_bg, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        # Crop
        context.current_grid = grid[rmin : rmax + 1, cmin : cmax + 1]
        return context

    def __str__(self) -> str:
        return f"CropToContent(bg={self.background_color})"


class MirrorSymmetry(Primitive):
    """Create mirror symmetry along an axis."""

    def __init__(self, axis: str = "vertical", mode: str = "right_to_left"):
        self.axis = axis
        self.mode = mode

    def execute(self, context: ExecutionContext) -> ExecutionContext:
        grid = context.current_grid.copy()

        if self.axis == "vertical":
            if self.mode == "right_to_left":
                # Copy right half to left
                mid = grid.shape[1] // 2
                grid[:, :mid] = np.fliplr(grid[:, mid : mid * 2])
            else:  # left_to_right
                mid = grid.shape[1] // 2
                grid[:, mid:] = np.fliplr(grid[:, :mid])

        elif self.axis == "horizontal":
            if self.mode == "bottom_to_top":
                mid = grid.shape[0] // 2
                grid[:mid, :] = np.flipud(grid[mid : mid * 2, :])
            else:  # top_to_bottom
                mid = grid.shape[0] // 2
                grid[mid:, :] = np.flipud(grid[:mid, :])

        context.current_grid = grid
        return context

    def __str__(self) -> str:
        return f"MirrorSymmetry(axis={self.axis}, mode={self.mode})"


class CountObjects(Primitive):
    """Count connected components and store in metadata."""

    def __init__(self, target_color: Optional[int] = None):
        self.target_color = target_color

    def execute(self, context: ExecutionContext) -> ExecutionContext:
        grid = context.current_grid

        if self.target_color is not None:
            mask = grid == self.target_color
        else:
            mask = grid != 0  # All non-background

        labeled, num_features = ndimage.label(mask)

        # Store count in metadata
        if context.metadata is None:
            context.metadata = {}
        context.metadata["object_count"] = num_features
        context.metadata["object_labels"] = labeled

        return context

    def __str__(self) -> str:
        return f"CountObjects(color={self.target_color})"


class SelectBySize(Primitive):
    """Select objects based on their size."""

    def __init__(
        self,
        min_size: Optional[int] = None,
        max_size: Optional[int] = None,
        keep_color: int = 1,
    ):
        self.min_size = min_size
        self.max_size = max_size
        self.keep_color = keep_color

    def execute(self, context: ExecutionContext) -> ExecutionContext:
        grid = context.current_grid.copy()

        # Get object labels (assume CountObjects was called)
        if context.metadata and "object_labels" in context.metadata:
            labeled = context.metadata["object_labels"]
        else:
            # Compute labels
            mask = grid != 0
            labeled, num_features = ndimage.label(mask)

        new_grid = np.zeros_like(grid)

        # Check each object
        for label_id in range(1, labeled.max() + 1):
            mask = labeled == label_id
            size = np.sum(mask)

            keep = True
            if self.min_size is not None and size < self.min_size:
                keep = False
            if self.max_size is not None and size > self.max_size:
                keep = False

            if keep:
                new_grid[mask] = self.keep_color

        context.current_grid = new_grid
        return context

    def __str__(self) -> str:
        return f"SelectBySize(min={self.min_size}, max={self.max_size})"


class DrawLine(Primitive):
    """Draw a line between two points."""

    def __init__(self, x1: int, y1: int, x2: int, y2: int, color: int):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.color = color

    def execute(self, context: ExecutionContext) -> ExecutionContext:
        grid = context.current_grid.copy()

        # Bresenham's line algorithm
        dx = abs(self.x2 - self.x1)
        dy = abs(self.y2 - self.y1)
        sx = 1 if self.x1 < self.x2 else -1
        sy = 1 if self.y1 < self.y2 else -1
        err = dx - dy

        x, y = self.x1, self.y1

        while True:
            if 0 <= y < grid.shape[0] and 0 <= x < grid.shape[1]:
                grid[y, x] = self.color

            if x == self.x2 and y == self.y2:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

        context.current_grid = grid
        return context

    def __str__(self) -> str:
        return (
            f"DrawLine({self.x1},{self.y1} -> {self.x2},{self.y2}, color={self.color})"
        )


class ExtractSubgrid(Primitive):
    """Extract a rectangular subgrid."""

    def __init__(self, x1: int, y1: int, x2: int, y2: int):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def execute(self, context: ExecutionContext) -> ExecutionContext:
        grid = context.current_grid
        context.current_grid = grid[self.y1 : self.y2 + 1, self.x1 : self.x2 + 1].copy()
        return context

    def __str__(self) -> str:
        return f"ExtractSubgrid({self.x1},{self.y1} to {self.x2},{self.y2})"


class RepeatGrid(Primitive):
    """Repeat the grid in a tiling pattern."""

    def __init__(self, times_x: int, times_y: int):
        self.times_x = times_x
        self.times_y = times_y

    def execute(self, context: ExecutionContext) -> ExecutionContext:
        grid = context.current_grid
        h, w = grid.shape

        new_grid = np.zeros((h * self.times_y, w * self.times_x), dtype=grid.dtype)

        for i in range(self.times_y):
            for j in range(self.times_x):
                new_grid[i * h : (i + 1) * h, j * w : (j + 1) * w] = grid

        context.current_grid = new_grid
        return context

    def __str__(self) -> str:
        return f"RepeatGrid({self.times_x}x{self.times_y})"


# Conditional primitives
class IfColorPresent(Primitive):
    """Execute a primitive only if a color is present in the grid."""

    def __init__(
        self,
        color: int,
        then_primitive: Primitive,
        else_primitive: Optional[Primitive] = None,
    ):
        self.color = color
        self.then_primitive = then_primitive
        self.else_primitive = else_primitive

    def execute(self, context: ExecutionContext) -> ExecutionContext:
        if np.any(context.current_grid == self.color):
            return self.then_primitive.execute(context)
        elif self.else_primitive:
            return self.else_primitive.execute(context)
        return context

    def __str__(self) -> str:
        else_str = f" else {self.else_primitive}" if self.else_primitive else ""
        return f"IfColorPresent({self.color}) then {self.then_primitive}{else_str}"


class IfShapeIs(Primitive):
    """Execute a primitive based on grid shape."""

    def __init__(
        self,
        shape: Tuple[int, int],
        then_primitive: Primitive,
        else_primitive: Optional[Primitive] = None,
    ):
        self.shape = shape
        self.then_primitive = then_primitive
        self.else_primitive = else_primitive

    def execute(self, context: ExecutionContext) -> ExecutionContext:
        if context.current_grid.shape == self.shape:
            return self.then_primitive.execute(context)
        elif self.else_primitive:
            return self.else_primitive.execute(context)
        return context

    def __str__(self) -> str:
        else_str = f" else {self.else_primitive}" if self.else_primitive else ""
        return f"IfShapeIs({self.shape}) then {self.then_primitive}{else_str}"


def test_advanced_primitives():
    """Test the advanced DSL primitives."""
    print("Testing Advanced DSL Primitives\n")

    # Test 1: FillInterior
    print("Test 1: FillInterior")
    grid = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 3, 0, 0, 0],
            [0, 3, 0, 3, 0, 0],
            [0, 0, 3, 0, 3, 0],
            [0, 0, 0, 3, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ]
    )

    context = ExecutionContext(input_grid=grid.copy(), current_grid=grid.copy())
    fill_interior = FillInterior(boundary_color=3, fill_color=4)
    result = fill_interior.execute(context)

    print("Input:")
    print(grid)
    print("Output (should fill interior with 4):")
    print(result.current_grid)

    # Test 2: FloodFill
    print("\nTest 2: FloodFill")
    grid = np.array(
        [
            [1, 1, 0, 2, 2],
            [1, 0, 0, 0, 2],
            [0, 0, 3, 0, 0],
            [4, 0, 0, 0, 5],
            [4, 4, 0, 5, 5],
        ]
    )

    context = ExecutionContext(input_grid=grid.copy(), current_grid=grid.copy())
    flood = FloodFill(start_x=2, start_y=2, fill_color=7)
    result = flood.execute(context)

    print("Input:")
    print(grid)
    print("Output (flood fill from center):")
    print(result.current_grid)

    # Test 3: ConnectPoints
    print("\nTest 3: ConnectPoints")
    grid = np.zeros((7, 7), dtype=int)
    grid[1, 1] = 3
    grid[1, 5] = 3
    grid[5, 5] = 3
    grid[5, 1] = 3

    context = ExecutionContext(input_grid=grid.copy(), current_grid=grid.copy())
    connect = ConnectPoints(point_color=3, line_color=3)
    result = connect.execute(context)

    print("Input (4 corner points):")
    print(grid)
    print("Output (connected):")
    print(result.current_grid)

    # Test 4: MirrorSymmetry
    print("\nTest 4: MirrorSymmetry")
    grid = np.array([[1, 2, 0, 0], [3, 4, 0, 0], [5, 6, 0, 0], [7, 8, 0, 0]])

    context = ExecutionContext(input_grid=grid.copy(), current_grid=grid.copy())
    mirror = MirrorSymmetry(axis="vertical", mode="left_to_right")
    result = mirror.execute(context)

    print("Input:")
    print(grid)
    print("Output (mirrored left to right):")
    print(result.current_grid)


if __name__ == "__main__":
    test_advanced_primitives()
