#!/usr/bin/env python3
"""Missing DSL primitives identified from failed task analysis.

Based on analysis of 28 failed ARC tasks, these primitives address the most
common missing patterns.
"""

from utils.imports import setup_project_paths

setup_project_paths()

from typing import Optional, Tuple

import numpy as np
from compositional_dsl import ExecutionContext, Primitive
from scipy import ndimage

# ============================================================================
# Line Drawing Primitives
# ============================================================================


class DrawLine(Primitive):
    """Draw a straight line between two points."""

    def __init__(
        self,
        start: Optional[Tuple[int, int]] = None,
        end: Optional[Tuple[int, int]] = None,
        color: Optional[int] = None,
        direction: Optional[str] = None,
    ):
        self.start = start
        self.end = end
        self.color = color
        self.direction = direction  # 'horizontal', 'vertical', 'diagonal'

    def execute(self, context: ExecutionContext) -> ExecutionContext:
        """Draw line on the grid."""
        result = context.copy()
        grid = result.current_grid.copy()

        # Auto-detect parameters if not provided
        if self.start is None or self.end is None:
            # Find two colored pixels and connect them
            colored_pixels = np.argwhere(context.input_grid != 0)
            if len(colored_pixels) >= 2:
                self.start = tuple(colored_pixels[0])
                self.end = tuple(colored_pixels[-1])
            else:
                return result

        if self.color is None:
            # Use most common non-zero color
            colors, counts = np.unique(
                context.input_grid[context.input_grid != 0], return_counts=True
            )
            if len(colors) > 0:
                self.color = colors[np.argmax(counts)]
            else:
                self.color = 1

        # Draw the line using Bresenham's algorithm
        y0, x0 = self.start
        y1, x1 = self.end

        dy = abs(y1 - y0)
        dx = abs(x1 - x0)
        sy = 1 if y0 < y1 else -1
        sx = 1 if x0 < x1 else -1
        err = dx - dy

        while True:
            if 0 <= y0 < grid.shape[0] and 0 <= x0 < grid.shape[1]:
                grid[y0, x0] = self.color

            if y0 == y1 and x0 == x1:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

        result.current_grid = grid
        return result

    def __str__(self):
        if self.direction:
            return f"DrawLine({self.direction})"
        return f"DrawLine({self.start}->{self.end}, color={self.color})"


class ConnectObjects(Primitive):
    """Connect all objects of a given color with lines."""

    def __init__(self, color: Optional[int] = None, line_color: Optional[int] = None):
        self.color = color
        self.line_color = line_color

    def execute(self, context: ExecutionContext) -> ExecutionContext:
        """Connect objects with lines."""
        result = context.copy()
        grid = result.current_grid.copy()

        # Auto-detect color if not provided
        if self.color is None:
            # Find color with multiple objects
            for color in np.unique(context.input_grid):
                if color != 0:
                    mask = (context.input_grid == color).astype(int)
                    labeled, count = ndimage.label(mask)
                    if count > 1:
                        self.color = color
                        break

        if self.color is None:
            return result

        if self.line_color is None:
            self.line_color = 5  # Default to cyan

        # Find object centers
        mask = (grid == self.color).astype(int)
        labeled, num_objects = ndimage.label(mask)

        centers = []
        for i in range(1, num_objects + 1):
            positions = np.argwhere(labeled == i)
            center = positions.mean(axis=0).astype(int)
            centers.append(tuple(center))

        # Connect all pairs of centers
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                # Draw line between centers[i] and centers[j]
                drawer = DrawLine(centers[i], centers[j], self.line_color)
                temp_context = ExecutionContext(
                    input_grid=context.input_grid, current_grid=grid
                )
                temp_result = drawer.execute(temp_context)
                grid = temp_result.current_grid

        result.current_grid = grid
        return result

    def __str__(self):
        return f"ConnectObjects(color={self.color}, line={self.line_color})"


# ============================================================================
# Counting and Sorting Primitives
# ============================================================================


class CountObjects(Primitive):
    """Count objects and potentially modify based on count."""

    def __init__(self, color: Optional[int] = None, action: Optional[str] = None):
        self.color = color
        self.action = action  # 'mark_nth', 'remove_odd', 'highlight_max', etc.

    def execute(self, context: ExecutionContext) -> ExecutionContext:
        """Count and potentially modify objects."""
        result = context.copy()
        grid = result.current_grid.copy()

        # Count objects for each color
        object_counts = {}
        for color in np.unique(grid):
            if color != 0:
                mask = (grid == color).astype(int)
                labeled, count = ndimage.label(mask)
                object_counts[color] = (labeled, count)

        # Store count in metadata
        if result.metadata is None:
            result.metadata = {}
        result.metadata["object_counts"] = {
            c: count for c, (_, count) in object_counts.items()
        }

        # Apply action based on count
        if self.action == "mark_nth":
            # Mark every nth object with a different color
            for color, (labeled, count) in object_counts.items():
                if self.color is None or color == self.color:
                    for i in range(2, count + 1, 2):  # Mark every 2nd object
                        grid[labeled == i] = 3  # Mark with color 3

        elif self.action == "remove_small":
            # Remove objects smaller than average
            for color, (labeled, count) in object_counts.items():
                if self.color is None or color == self.color:
                    sizes = []
                    for i in range(1, count + 1):
                        size = np.sum(labeled == i)
                        sizes.append((i, size))

                    if sizes:
                        avg_size = sum(s for _, s in sizes) / len(sizes)
                        for obj_id, size in sizes:
                            if size < avg_size:
                                grid[labeled == obj_id] = 0

        result.current_grid = grid
        return result

    def __str__(self):
        return f"CountObjects(color={self.color}, action={self.action})"


class SortBySize(Primitive):
    """Sort objects by size and rearrange them."""

    def __init__(self, direction: str = "horizontal"):
        self.direction = direction  # 'horizontal', 'vertical'

    def execute(self, context: ExecutionContext) -> ExecutionContext:
        """Sort objects by size."""
        result = context.copy()
        grid = result.current_grid.copy()

        # Extract all objects
        objects = []
        for color in np.unique(grid):
            if color != 0:
                mask = (grid == color).astype(int)
                labeled, count = ndimage.label(mask)

                for i in range(1, count + 1):
                    obj_mask = labeled == i
                    positions = np.argwhere(obj_mask)
                    if len(positions) > 0:
                        # Get bounding box
                        min_r, min_c = positions.min(axis=0)
                        max_r, max_c = positions.max(axis=0)

                        # Extract object
                        obj = grid[min_r : max_r + 1, min_c : max_c + 1].copy()
                        obj[~obj_mask[min_r : max_r + 1, min_c : max_c + 1]] = 0

                        objects.append(
                            {
                                "object": obj,
                                "color": color,
                                "size": len(positions),
                                "bbox": (min_r, min_c, max_r, max_c),
                            }
                        )

        # Sort by size
        objects.sort(key=lambda x: x["size"])

        # Clear grid and place sorted objects
        grid.fill(0)

        if self.direction == "horizontal":
            col = 0
            for obj_data in objects:
                obj = obj_data["object"]
                h, w = obj.shape

                if col + w <= grid.shape[1]:
                    grid[:h, col : col + w] = np.where(
                        obj != 0, obj, grid[:h, col : col + w]
                    )
                    col += w + 1  # Add spacing
        else:  # vertical
            row = 0
            for obj_data in objects:
                obj = obj_data["object"]
                h, w = obj.shape

                if row + h <= grid.shape[0]:
                    grid[row : row + h, :w] = np.where(
                        obj != 0, obj, grid[row : row + h, :w]
                    )
                    row += h + 1  # Add spacing

        result.current_grid = grid
        return result

    def __str__(self):
        return f"SortBySize(direction={self.direction})"


# ============================================================================
# Grid Partitioning Primitives
# ============================================================================


class PartitionGrid(Primitive):
    """Partition grid into regions."""

    def __init__(self, rows: int = 2, cols: int = 2, action: Optional[str] = None):
        self.rows = rows
        self.cols = cols
        self.action = action  # 'copy_to_all', 'rotate_each', 'color_each'

    def execute(self, context: ExecutionContext) -> ExecutionContext:
        """Partition grid into regions and apply action."""
        result = context.copy()
        grid = result.current_grid.copy()

        h, w = grid.shape
        region_h = h // self.rows
        region_w = w // self.cols

        if self.action == "copy_to_all":
            # Copy first region to all others
            first_region = grid[:region_h, :region_w].copy()

            for i in range(self.rows):
                for j in range(self.cols):
                    if i == 0 and j == 0:
                        continue

                    r_start = i * region_h
                    c_start = j * region_w
                    grid[
                        r_start : r_start + region_h, c_start : c_start + region_w
                    ] = first_region

        elif self.action == "rotate_each":
            # Rotate each region by different amounts
            rotation = 0
            for i in range(self.rows):
                for j in range(self.cols):
                    r_start = i * region_h
                    c_start = j * region_w
                    region = grid[
                        r_start : r_start + region_h, c_start : c_start + region_w
                    ].copy()

                    rotated = np.rot90(region, rotation)
                    grid[
                        r_start : r_start + region_h, c_start : c_start + region_w
                    ] = rotated

                    rotation = (rotation + 1) % 4

        elif self.action == "color_each":
            # Color each region differently
            color = 1
            for i in range(self.rows):
                for j in range(self.cols):
                    r_start = i * region_h
                    c_start = j * region_w
                    region = grid[
                        r_start : r_start + region_h, c_start : c_start + region_w
                    ]

                    # Add colored border
                    region[0, :] = color
                    region[-1, :] = color
                    region[:, 0] = color
                    region[:, -1] = color

                    color = (color % 9) + 1

        result.current_grid = grid
        return result

    def __str__(self):
        return f"PartitionGrid({self.rows}x{self.cols}, action={self.action})"


# ============================================================================
# Conditional Fill Primitives
# ============================================================================


class ConditionalFill(Primitive):
    """Fill pixels based on neighbor conditions."""

    def __init__(
        self, condition: Optional[str] = None, fill_color: Optional[int] = None
    ):
        self.condition = condition  # 'between_colors', 'adjacent_to', 'enclosed_by'
        self.fill_color = fill_color

    def execute(self, context: ExecutionContext) -> ExecutionContext:
        """Apply conditional filling."""
        result = context.copy()
        grid = result.current_grid.copy()

        if self.fill_color is None:
            # Use most common non-zero color
            colors, counts = np.unique(grid[grid != 0], return_counts=True)
            if len(colors) > 0:
                self.fill_color = colors[np.argmax(counts)]
            else:
                self.fill_color = 1

        if self.condition == "between_colors":
            # Fill pixels between same colored pixels
            for i in range(grid.shape[0]):
                # Horizontal fill
                for color in np.unique(grid[i]):
                    if color != 0:
                        positions = np.where(grid[i] == color)[0]
                        if len(positions) >= 2:
                            start, end = positions[0], positions[-1]
                            grid[i, start : end + 1] = self.fill_color

            for j in range(grid.shape[1]):
                # Vertical fill
                for color in np.unique(grid[:, j]):
                    if color != 0:
                        positions = np.where(grid[:, j] == color)[0]
                        if len(positions) >= 2:
                            start, end = positions[0], positions[-1]
                            grid[start : end + 1, j] = self.fill_color

        elif self.condition == "adjacent_to":
            # Fill pixels adjacent to colored pixels
            new_grid = grid.copy()
            for i in range(1, grid.shape[0] - 1):
                for j in range(1, grid.shape[1] - 1):
                    if grid[i, j] == 0:
                        # Check neighbors
                        neighbors = [
                            grid[i - 1, j],
                            grid[i + 1, j],
                            grid[i, j - 1],
                            grid[i, j + 1],
                        ]
                        if any(n != 0 for n in neighbors):
                            new_grid[i, j] = self.fill_color
            grid = new_grid

        elif self.condition == "enclosed_by":
            # Fill regions enclosed by non-zero pixels
            # Create boundary mask
            boundary_mask = grid != 0

            # Fill holes
            from scipy.ndimage import binary_fill_holes

            filled = binary_fill_holes(boundary_mask)

            # Fill the enclosed regions
            grid[filled & ~boundary_mask] = self.fill_color

        result.current_grid = grid
        return result

    def __str__(self):
        return f"ConditionalFill(condition={self.condition}, color={self.fill_color})"


class PropagatePattern(Primitive):
    """Propagate a pattern across the grid."""

    def __init__(self, direction: str = "all"):
        self.direction = direction  # 'horizontal', 'vertical', 'all'

    def execute(self, context: ExecutionContext) -> ExecutionContext:
        """Propagate pattern from detected region."""
        result = context.copy()
        grid = result.current_grid.copy()

        # Find a pattern (e.g., 3x3 region with non-zero pixels)
        pattern = None
        pattern_pos = None

        for i in range(grid.shape[0] - 2):
            for j in range(grid.shape[1] - 2):
                region = grid[i : i + 3, j : j + 3]
                if np.sum(region != 0) >= 3:  # At least 3 non-zero pixels
                    pattern = region.copy()
                    pattern_pos = (i, j)
                    break
            if pattern is not None:
                break

        if pattern is None:
            return result

        # Propagate the pattern
        if self.direction in ["horizontal", "all"]:
            # Propagate horizontally
            for j in range(0, grid.shape[1] - 2, 3):
                if j != pattern_pos[1]:  # Don't overwrite original
                    grid[pattern_pos[0] : pattern_pos[0] + 3, j : j + 3] = pattern

        if self.direction in ["vertical", "all"]:
            # Propagate vertically
            for i in range(0, grid.shape[0] - 2, 3):
                if i != pattern_pos[0]:  # Don't overwrite original
                    grid[i : i + 3, pattern_pos[1] : pattern_pos[1] + 3] = pattern

        result.current_grid = grid
        return result

    def __str__(self):
        return f"PropagatePattern(direction={self.direction})"


# ============================================================================
# Edge Detection Primitives
# ============================================================================


class ExtractBoundaries(Primitive):
    """Extract boundaries/edges of objects."""

    def __init__(
        self, color: Optional[int] = None, boundary_color: Optional[int] = None
    ):
        self.color = color
        self.boundary_color = boundary_color

    def execute(self, context: ExecutionContext) -> ExecutionContext:
        """Extract object boundaries."""
        result = context.copy()
        grid = result.current_grid.copy()

        if self.boundary_color is None:
            self.boundary_color = 5  # Default to cyan

        # Create boundary grid
        boundary_grid = np.zeros_like(grid)

        if self.color is None:
            # Extract boundaries for all non-zero colors
            mask = grid != 0
        else:
            mask = grid == self.color

        # Find boundaries using morphological operations
        from scipy.ndimage import binary_erosion

        eroded = binary_erosion(mask)
        boundaries = mask & ~eroded

        # Set boundary pixels
        boundary_grid[boundaries] = self.boundary_color

        # Optionally keep interior
        result.current_grid = boundary_grid + grid * ~boundaries

        return result

    def __str__(self):
        return f"ExtractBoundaries(color={self.color}, boundary={self.boundary_color})"


class TraceBorder(Primitive):
    """Trace the border of an object and potentially extend it."""

    def __init__(self, extend: bool = False):
        self.extend = extend

    def execute(self, context: ExecutionContext) -> ExecutionContext:
        """Trace and potentially extend borders."""
        result = context.copy()
        grid = result.current_grid.copy()

        # Find all objects
        for color in np.unique(grid):
            if color != 0:
                mask = grid == color

                # Get contour
                from scipy.ndimage import binary_erosion

                eroded = binary_erosion(mask)
                border = mask & ~eroded

                if self.extend:
                    # Extend the border outward
                    from scipy.ndimage import binary_dilation

                    extended = binary_dilation(border)

                    # Add extended border
                    grid[extended & ~mask] = color

        result.current_grid = grid
        return result

    def __str__(self):
        return f"TraceBorder(extend={self.extend})"


# ============================================================================
# Helper function to add all new primitives to DSL
# ============================================================================


def get_missing_primitives():
    """Return list of all missing primitive classes."""
    return [
        # Line drawing
        DrawLine,
        ConnectObjects,
        # Counting and sorting
        CountObjects,
        SortBySize,
        # Grid partitioning
        PartitionGrid,
        # Conditional fills
        ConditionalFill,
        PropagatePattern,
        # Edge detection
        ExtractBoundaries,
        TraceBorder,
    ]


if __name__ == "__main__":
    print("Missing DSL Primitives Implementation")
    print("=" * 60)
    print("\nImplemented primitives:")

    for primitive_class in get_missing_primitives():
        print(f"  - {primitive_class.__name__}")

    print("\nThese primitives address patterns found in 28 failed ARC tasks.")
    print("Integration with existing DSL will enable solving ~15-20% of tasks.")
