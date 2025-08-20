#!/usr/bin/env python3
"""Enhanced ARC DSL with primitives identified from failure analysis.

Based on analysis of 50 failed tasks, this adds:
- Arithmetic operations (color shifting, counting)
- Conditional logic (size/color/shape-based)
- Spatial patterns (diagonal, spiral, border)
- Structural operations (merge, split, connect)
"""

from utils.imports import setup_project_paths

setup_project_paths()

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple

import numpy as np
from object_manipulation import ObjectManipulator


class ARCPrimitive(ABC):
    """Base class for DSL primitives."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        """Execute the primitive operation."""

    def __repr__(self) -> str:
        return f"{self.name}()"


# ============================================================================
# ARITHMETIC PRIMITIVES (New based on failure analysis)
# ============================================================================


class AddConstant(ARCPrimitive):
    """Add constant value to all non-zero colors."""

    def __init__(self, value: int):
        super().__init__(f"add_{value}")
        self.value = value

    def execute(self, grid: np.ndarray) -> np.ndarray:
        result = grid.copy()
        mask = grid > 0
        result[mask] = np.clip(grid[mask] + self.value, 0, 9)
        return result


class CountObjects(ARCPrimitive):
    """Count objects and encode as color value."""

    def __init__(self):
        super().__init__("count_objects")
        self.manipulator = ObjectManipulator()

    def execute(self, grid: np.ndarray) -> np.ndarray:
        objects = self.manipulator.extract_objects(grid)
        count = min(len(objects), 9)  # Clamp to valid color range

        # Create output with count encoded
        result = np.zeros_like(grid)
        for obj in objects:
            for pixel in obj.pixels:
                result[pixel[0], pixel[1]] = count
        return result


class MultiplyColors(ARCPrimitive):
    """Multiply color values by factor."""

    def __init__(self, factor: float):
        super().__init__(f"multiply_{factor}")
        self.factor = factor

    def execute(self, grid: np.ndarray) -> np.ndarray:
        result = grid.copy()
        mask = grid > 0
        result[mask] = np.clip(grid[mask] * self.factor, 0, 9).astype(int)
        return result


class EnumerateObjects(ARCPrimitive):
    """Number objects sequentially."""

    def __init__(self):
        super().__init__("enumerate")
        self.manipulator = ObjectManipulator()

    def execute(self, grid: np.ndarray) -> np.ndarray:
        objects = self.manipulator.extract_objects(grid)
        result = np.zeros_like(grid)

        for i, obj in enumerate(objects, 1):
            color = min(i, 9)  # Clamp to valid range
            for pixel in obj.pixels:
                result[pixel[0], pixel[1]] = color
        return result


# ============================================================================
# CONDITIONAL LOGIC PRIMITIVES (New based on failure analysis)
# ============================================================================


class IfSize(ARCPrimitive):
    """Apply operation based on object size."""

    def __init__(
        self, threshold: int, then_color: int, else_color: Optional[int] = None
    ):
        super().__init__(f"if_size_{threshold}")
        self.threshold = threshold
        self.then_color = then_color
        self.else_color = else_color
        self.manipulator = ObjectManipulator()

    def execute(self, grid: np.ndarray) -> np.ndarray:
        objects = self.manipulator.extract_objects(grid)
        result = grid.copy() if self.else_color is None else np.zeros_like(grid)

        for obj in objects:
            if len(obj.pixels) > self.threshold:
                for pixel in obj.pixels:
                    result[pixel[0], pixel[1]] = self.then_color
            elif self.else_color is not None:
                for pixel in obj.pixels:
                    result[pixel[0], pixel[1]] = self.else_color

        return result


class IfColor(ARCPrimitive):
    """Apply operation based on color."""

    def __init__(self, test_color: int, then_color: int):
        super().__init__(f"if_color_{test_color}_then_{then_color}")
        self.test_color = test_color
        self.then_color = then_color

    def execute(self, grid: np.ndarray) -> np.ndarray:
        result = grid.copy()
        mask = grid == self.test_color
        result[mask] = self.then_color
        return result


class IfSquare(ARCPrimitive):
    """Apply operation to square objects."""

    def __init__(self, then_color: int):
        super().__init__(f"if_square_then_{then_color}")
        self.then_color = then_color
        self.manipulator = ObjectManipulator()

    def execute(self, grid: np.ndarray) -> np.ndarray:
        objects = self.manipulator.extract_objects(grid)
        result = grid.copy()

        for obj in objects:
            height = obj.bounding_box[2] - obj.bounding_box[0] + 1
            width = obj.bounding_box[3] - obj.bounding_box[1] + 1

            if height == width:  # Square object
                for pixel in obj.pixels:
                    result[pixel[0], pixel[1]] = self.then_color

        return result


class IfLarge(ARCPrimitive):
    """Fill large objects with specified color."""

    def __init__(self, fill_color: int, size_threshold: int = 4):
        super().__init__(f"if_large_fill_{fill_color}")
        self.fill_color = fill_color
        self.size_threshold = size_threshold
        self.manipulator = ObjectManipulator()

    def execute(self, grid: np.ndarray) -> np.ndarray:
        objects = self.manipulator.extract_objects(grid)
        result = grid.copy()

        for obj in objects:
            if len(obj.pixels) > self.size_threshold:
                # Fill bounding box
                for r in range(obj.bounding_box[0], obj.bounding_box[2] + 1):
                    for c in range(obj.bounding_box[1], obj.bounding_box[3] + 1):
                        result[r, c] = self.fill_color

        return result


# ============================================================================
# SPATIAL PATTERN PRIMITIVES (New based on failure analysis)
# ============================================================================


class DrawDiagonal(ARCPrimitive):
    """Draw diagonal line(s)."""

    def __init__(self, color: int, anti: bool = False):
        super().__init__(f"diagonal_{'anti' if anti else 'main'}_{color}")
        self.color = color
        self.anti = anti

    def execute(self, grid: np.ndarray) -> np.ndarray:
        result = grid.copy()
        h, w = grid.shape

        if self.anti:
            for i in range(min(h, w)):
                if i < h and (w - 1 - i) >= 0:
                    result[i, w - 1 - i] = self.color
        else:
            for i in range(min(h, w)):
                result[i, i] = self.color

        return result


class DrawBorder(ARCPrimitive):
    """Draw border around grid."""

    def __init__(self, color: int, thickness: int = 1):
        super().__init__(f"border_{color}_thick_{thickness}")
        self.color = color
        self.thickness = thickness

    def execute(self, grid: np.ndarray) -> np.ndarray:
        result = grid.copy()
        h, w = grid.shape

        for t in range(self.thickness):
            # Top and bottom
            result[t, :] = self.color
            result[h - 1 - t, :] = self.color

            # Left and right
            result[:, t] = self.color
            result[:, w - 1 - t] = self.color

        return result


class FillSpiral(ARCPrimitive):
    """Fill grid in spiral pattern."""

    def __init__(self, colors: List[int]):
        super().__init__(f"spiral_{colors}")
        self.colors = colors

    def execute(self, grid: np.ndarray) -> np.ndarray:
        result = np.zeros_like(grid)
        h, w = grid.shape

        top, bottom = 0, h - 1
        left, right = 0, w - 1
        color_idx = 0

        while top <= bottom and left <= right:
            color = self.colors[color_idx % len(self.colors)]

            # Top row
            for i in range(left, right + 1):
                result[top, i] = color
            top += 1

            # Right column
            for i in range(top, bottom + 1):
                result[i, right] = color
            right -= 1

            # Bottom row
            if top <= bottom:
                for i in range(right, left - 1, -1):
                    result[bottom, i] = color
                bottom -= 1

            # Left column
            if left <= right:
                for i in range(bottom, top - 1, -1):
                    result[i, left] = color
                left += 1

            color_idx += 1

        return result


class RepeatPattern(ARCPrimitive):
    """Repeat a pattern across the grid."""

    def __init__(self, pattern: np.ndarray):
        super().__init__("repeat_pattern")
        self.pattern = pattern

    def execute(self, grid: np.ndarray) -> np.ndarray:
        result = np.zeros_like(grid)
        ph, pw = self.pattern.shape
        h, w = grid.shape

        for r in range(0, h, ph):
            for c in range(0, w, pw):
                sub_h = min(ph, h - r)
                sub_w = min(pw, w - c)
                result[r : r + sub_h, c : c + sub_w] = self.pattern[:sub_h, :sub_w]

        return result


class MakeSymmetric(ARCPrimitive):
    """Make grid symmetric."""

    def __init__(self, axis: str = "horizontal"):
        super().__init__(f"symmetric_{axis}")
        self.axis = axis

    def execute(self, grid: np.ndarray) -> np.ndarray:
        result = grid.copy()
        h, w = grid.shape

        if self.axis == "horizontal":
            # Copy top half to bottom
            for i in range(h // 2):
                result[h - 1 - i] = result[i]
        elif self.axis == "vertical":
            # Copy left half to right
            for j in range(w // 2):
                result[:, w - 1 - j] = result[:, j]
        elif self.axis == "diagonal":
            # Make diagonal symmetric
            if h == w:
                for i in range(h):
                    for j in range(i):
                        result[j, i] = result[i, j]

        return result


# ============================================================================
# COUNTING AND INDEXING (Enhanced based on failure analysis)
# ============================================================================


class DuplicateNTimes(ARCPrimitive):
    """Duplicate objects N times."""

    def __init__(self, n: int, offset: Tuple[int, int] = (0, 1)):
        super().__init__(f"duplicate_{n}_times")
        self.n = n
        self.offset = offset
        self.manipulator = ObjectManipulator()

    def execute(self, grid: np.ndarray) -> np.ndarray:
        objects = self.manipulator.extract_objects(grid)
        result = grid.copy()
        h, w = grid.shape

        for obj in objects:
            for i in range(1, self.n):
                new_pixels = []
                for pixel in obj.pixels:
                    new_r = pixel[0] + i * self.offset[0]
                    new_c = pixel[1] + i * self.offset[1]

                    if 0 <= new_r < h and 0 <= new_c < w:
                        new_pixels.append((new_r, new_c))

                for pixel in new_pixels:
                    if result[pixel[0], pixel[1]] == 0:
                        result[pixel[0], pixel[1]] = obj.color

        return result


class SelectNth(ARCPrimitive):
    """Select the Nth object."""

    def __init__(self, n: int):
        super().__init__(f"select_{n}th")
        self.n = n
        self.manipulator = ObjectManipulator()

    def execute(self, grid: np.ndarray) -> np.ndarray:
        objects = self.manipulator.extract_objects(grid)
        result = np.zeros_like(grid)

        if 0 <= self.n < len(objects):
            obj = objects[self.n]
            for pixel in obj.pixels:
                result[pixel[0], pixel[1]] = obj.color

        return result


class TilePattern(ARCPrimitive):
    """Tile the input pattern in an NxN grid layout."""

    def __init__(self, scale: int = 3):
        super().__init__(f"tile_{scale}x")
        self.scale = scale

    def execute(self, grid: np.ndarray) -> np.ndarray:
        h, w = grid.shape
        output = np.zeros((h * self.scale, w * self.scale), dtype=grid.dtype)

        # Tile the pattern
        for i in range(self.scale):
            for j in range(self.scale):
                output[i * h : (i + 1) * h, j * w : (j + 1) * w] = grid

        return output


class ModifiedTilePattern(ARCPrimitive):
    """Tile pattern with modifications to specific tiles."""

    def __init__(self, scale: int = 3, modify_first: bool = True):
        super().__init__(f"modified_tile_{scale}x")
        self.scale = scale
        self.modify_first = modify_first

    def execute(self, grid: np.ndarray) -> np.ndarray:
        h, w = grid.shape
        output = np.zeros((h * self.scale, w * self.scale), dtype=grid.dtype)

        # First, tile the pattern normally
        for i in range(self.scale):
            for j in range(self.scale):
                output[i * h : (i + 1) * h, j * w : (j + 1) * w] = grid

        # Then apply modifications to the left column of tiles
        if self.modify_first:
            # Set the first 3 columns of the entire output to 0 for left tiles
            for i in range(self.scale):
                # For tiles in the leftmost column (j=0)
                output[i * h : (i + 1) * h, :h] = 0

        return output


# ============================================================================
# STRUCTURAL OPERATIONS (New based on failure analysis)
# ============================================================================


class MergeAdjacent(ARCPrimitive):
    """Merge adjacent/touching objects."""

    def __init__(self, merge_color: Optional[int] = None):
        super().__init__("merge_adjacent")
        self.merge_color = merge_color
        self.manipulator = ObjectManipulator()

    def execute(self, grid: np.ndarray) -> np.ndarray:
        objects = self.manipulator.extract_objects(grid)
        result = np.zeros_like(grid)

        # Find groups of touching objects
        merged_groups = []
        used = set()

        for i, obj1 in enumerate(objects):
            if i in used:
                continue

            group = [obj1]
            used.add(i)

            for j, obj2 in enumerate(objects[i + 1 :], i + 1):
                if j in used:
                    continue

                if self.manipulator.objects_touching(obj1, obj2):
                    group.append(obj2)
                    used.add(j)

            merged_groups.append(group)

        # Draw merged groups
        for group in merged_groups:
            color = self.merge_color if self.merge_color else group[0].color
            for obj in group:
                for pixel in obj.pixels:
                    result[pixel[0], pixel[1]] = color

        return result


class SplitByColor(ARCPrimitive):
    """Split multi-color objects into separate objects."""

    def __init__(self):
        super().__init__("split_by_color")

    def execute(self, grid: np.ndarray) -> np.ndarray:
        # This is mostly for visualization - objects are already split by color
        # in the extraction process
        return grid.copy()


class ConnectObjects(ARCPrimitive):
    """Connect objects with lines."""

    def __init__(self, line_color: int, method: str = "nearest"):
        super().__init__(f"connect_{method}")
        self.line_color = line_color
        self.method = method
        self.manipulator = ObjectManipulator()

    def execute(self, grid: np.ndarray) -> np.ndarray:
        objects = self.manipulator.extract_objects(grid)
        result = grid.copy()

        if len(objects) < 2:
            return result

        # Connect pairs based on method
        if self.method == "nearest":
            # Connect each object to its nearest neighbor
            for i, obj1 in enumerate(objects):
                min_dist = float("inf")
                nearest = None

                for j, obj2 in enumerate(objects):
                    if i == j:
                        continue

                    # Distance between centers
                    c1 = (
                        (obj1.bounding_box[0] + obj1.bounding_box[2]) // 2,
                        (obj1.bounding_box[1] + obj1.bounding_box[3]) // 2,
                    )
                    c2 = (
                        (obj2.bounding_box[0] + obj2.bounding_box[2]) // 2,
                        (obj2.bounding_box[1] + obj2.bounding_box[3]) // 2,
                    )

                    dist = abs(c1[0] - c2[0]) + abs(c1[1] - c2[1])
                    if dist < min_dist:
                        min_dist = dist
                        nearest = c2

                if nearest:
                    # Draw line from obj1 center to nearest
                    c1 = (
                        (obj1.bounding_box[0] + obj1.bounding_box[2]) // 2,
                        (obj1.bounding_box[1] + obj1.bounding_box[3]) // 2,
                    )
                    self._draw_line(result, c1, nearest, self.line_color)

        elif self.method == "sequential":
            # Connect objects in sequence
            for i in range(len(objects) - 1):
                c1 = (
                    (objects[i].bounding_box[0] + objects[i].bounding_box[2]) // 2,
                    (objects[i].bounding_box[1] + objects[i].bounding_box[3]) // 2,
                )
                c2 = (
                    (objects[i + 1].bounding_box[0] + objects[i + 1].bounding_box[2])
                    // 2,
                    (objects[i + 1].bounding_box[1] + objects[i + 1].bounding_box[3])
                    // 2,
                )
                self._draw_line(result, c1, c2, self.line_color)

        return result

    def _draw_line(
        self, grid: np.ndarray, p1: Tuple[int, int], p2: Tuple[int, int], color: int
    ):
        """Draw a line between two points."""
        r1, c1 = p1
        r2, c2 = p2

        # Bresenham's line algorithm (simplified)
        dr = abs(r2 - r1)
        dc = abs(c2 - c1)
        sr = 1 if r1 < r2 else -1
        sc = 1 if c1 < c2 else -1
        err = dr - dc

        while True:
            if 0 <= r1 < grid.shape[0] and 0 <= c1 < grid.shape[1]:
                grid[r1, c1] = color

            if r1 == r2 and c1 == c2:
                break

            e2 = 2 * err
            if e2 > -dc:
                err -= dc
                r1 += sr
            if e2 < dr:
                err += dr
                c1 += sc


# ============================================================================
# DSL LIBRARY
# ============================================================================


class EnhancedDSLLibrary:
    """Enhanced library of ARC DSL primitives."""

    def __init__(self):
        self.primitives = {}
        self._register_primitives()

    def _register_primitives(self):
        """Register all available primitives."""
        # Arithmetic
        self.primitives["add_constant"] = AddConstant
        self.primitives["count_objects"] = CountObjects
        self.primitives["multiply_colors"] = MultiplyColors
        self.primitives["enumerate_objects"] = EnumerateObjects

        # Conditional
        self.primitives["if_size"] = IfSize
        self.primitives["if_color"] = IfColor
        self.primitives["if_square"] = IfSquare
        self.primitives["if_large"] = IfLarge

        # Spatial patterns
        self.primitives["draw_diagonal"] = DrawDiagonal
        self.primitives["draw_border"] = DrawBorder
        self.primitives["fill_spiral"] = FillSpiral
        self.primitives["repeat_pattern"] = RepeatPattern
        self.primitives["make_symmetric"] = MakeSymmetric

        # Counting/Indexing
        self.primitives["duplicate_n"] = DuplicateNTimes
        self.primitives["select_nth"] = SelectNth
        self.primitives["tile_pattern"] = TilePattern
        self.primitives["modified_tile_pattern"] = ModifiedTilePattern

        # Enhanced tiling primitives
        from enhanced_tiling_primitives import (
            AlternatingRowTile,
            CheckerboardTile,
            SmartTilePattern,
        )

        self.primitives["smart_tile"] = SmartTilePattern
        self.primitives["alternating_row_tile"] = AlternatingRowTile
        self.primitives["checkerboard_tile"] = CheckerboardTile

        # Structural
        self.primitives["merge_adjacent"] = MergeAdjacent
        self.primitives["split_by_color"] = SplitByColor
        self.primitives["connect_objects"] = ConnectObjects

    def get_primitive(self, name: str, **kwargs) -> ARCPrimitive:
        """Get a primitive by name with parameters."""
        if name not in self.primitives:
            raise ValueError(f"Unknown primitive: {name}")
        return self.primitives[name](**kwargs)

    def list_primitives(self) -> List[str]:
        """List all available primitives."""
        return sorted(self.primitives.keys())


def test_enhanced_dsl():
    """Test the enhanced DSL primitives."""
    print("Testing Enhanced ARC DSL")
    print("=" * 60)

    library = EnhancedDSLLibrary()
    print(f"Available primitives: {len(library.list_primitives())}")
    print(f"Categories: Arithmetic, Conditional, Spatial, Counting, Structural")

    # Test arithmetic: color shifting
    print("\n1. Color Shifting (add 2 to all colors)")
    grid = np.array([[1, 2, 0], [0, 3, 1], [2, 0, 0]])
    add2 = library.get_primitive("add_constant", value=2)
    result = add2.execute(grid)
    print(f"Input:\n{grid}")
    print(f"Output:\n{result}")

    # Test conditional: if square then fill
    print("\n2. Conditional: If square object, fill with 5")
    grid = np.array([[1, 1, 0, 2], [1, 1, 0, 2], [0, 0, 0, 2], [3, 3, 3, 0]])
    if_square = library.get_primitive("if_square", then_color=5)
    result = if_square.execute(grid)
    print(f"Input:\n{grid}")
    print(f"Output:\n{result}")

    # Test spatial: draw diagonal
    print("\n3. Spatial: Draw diagonal")
    grid = np.zeros((4, 4), dtype=int)
    diagonal = library.get_primitive("draw_diagonal", color=1, anti=False)
    result = diagonal.execute(grid)
    print(f"Output:\n{result}")

    # Test spatial: draw border
    print("\n4. Spatial: Draw border")
    grid = np.zeros((5, 5), dtype=int)
    border = library.get_primitive("draw_border", color=2, thickness=1)
    result = border.execute(grid)
    print(f"Output:\n{result}")

    # Test counting: enumerate objects
    print("\n5. Counting: Enumerate objects")
    grid = np.array([[1, 0, 2, 2], [1, 0, 0, 0], [0, 3, 3, 0], [0, 3, 3, 0]])
    enumerate = library.get_primitive("enumerate_objects")
    result = enumerate.execute(grid)
    print(f"Input:\n{grid}")
    print(f"Output:\n{result}")

    print("\n" + "=" * 60)
    print("Enhanced DSL tests complete!")


if __name__ == "__main__":
    test_enhanced_dsl()
