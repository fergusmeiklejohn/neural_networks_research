#!/usr/bin/env python3
"""Compositional DSL for ARC-AGI Program Synthesis.

This module implements a compositional domain-specific language with:
- Atomic operations (move, rotate, color, etc.)
- Compositional operators (sequence, conditional, loop)
- Spatial relations (inside, adjacent, aligned)
- Object manipulation primitives
"""

from utils.imports import setup_project_paths

setup_project_paths()

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage


# Base classes
@dataclass
class ExecutionContext:
    """Context passed through program execution."""

    input_grid: np.ndarray
    current_grid: np.ndarray
    objects: Optional[List[np.ndarray]] = None
    metadata: Dict[str, Any] = None

    def copy(self):
        """Create a deep copy of the context."""
        return ExecutionContext(
            input_grid=self.input_grid.copy(),
            current_grid=self.current_grid.copy(),
            objects=[obj.copy() for obj in self.objects] if self.objects else None,
            metadata=self.metadata.copy() if self.metadata else None,
        )


class Primitive(ABC):
    """Base class for all DSL primitives."""

    @abstractmethod
    def execute(self, context: ExecutionContext) -> ExecutionContext:
        """Execute the primitive on the given context."""

    @abstractmethod
    def __str__(self) -> str:
        """String representation for program readability."""


# Atomic Operations
class Move(Primitive):
    """Move all non-zero pixels by a given offset."""

    def __init__(self, dx: int, dy: int):
        self.dx = dx
        self.dy = dy

    def execute(self, context: ExecutionContext) -> ExecutionContext:
        grid = context.current_grid.copy()
        new_grid = np.zeros_like(grid)

        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if grid[i, j] != 0:
                    new_i = i + self.dy
                    new_j = j + self.dx
                    if 0 <= new_i < grid.shape[0] and 0 <= new_j < grid.shape[1]:
                        new_grid[new_i, new_j] = grid[i, j]

        context.current_grid = new_grid
        return context

    def __str__(self) -> str:
        return f"Move({self.dx}, {self.dy})"


class Rotate(Primitive):
    """Rotate the grid by 90 degree increments."""

    def __init__(self, angle: int):
        self.angle = angle  # Should be 0, 90, 180, or 270

    def execute(self, context: ExecutionContext) -> ExecutionContext:
        rotations = self.angle // 90
        context.current_grid = np.rot90(context.current_grid, rotations)
        return context

    def __str__(self) -> str:
        return f"Rotate({self.angle})"


class FlipH(Primitive):
    """Flip the grid horizontally."""

    def execute(self, context: ExecutionContext) -> ExecutionContext:
        context.current_grid = np.fliplr(context.current_grid)
        return context

    def __str__(self) -> str:
        return "FlipH()"


class FlipV(Primitive):
    """Flip the grid vertically."""

    def execute(self, context: ExecutionContext) -> ExecutionContext:
        context.current_grid = np.flipud(context.current_grid)
        return context

    def __str__(self) -> str:
        return "FlipV()"


class SetColor(Primitive):
    """Set all pixels of one color to another color."""

    def __init__(self, from_color: int, to_color: int):
        self.from_color = from_color
        self.to_color = to_color

    def execute(self, context: ExecutionContext) -> ExecutionContext:
        mask = context.current_grid == self.from_color
        context.current_grid[mask] = self.to_color
        return context

    def __str__(self) -> str:
        return f"SetColor({self.from_color} -> {self.to_color})"


class FillRectangle(Primitive):
    """Fill a rectangle with a given color."""

    def __init__(self, x1: int, y1: int, x2: int, y2: int, color: int):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.color = color

    def execute(self, context: ExecutionContext) -> ExecutionContext:
        context.current_grid[self.y1 : self.y2 + 1, self.x1 : self.x2 + 1] = self.color
        return context

    def __str__(self) -> str:
        return f"FillRectangle({self.x1}, {self.y1}, {self.x2}, {self.y2}, color={self.color})"


class ExtractObjects(Primitive):
    """Extract connected components as objects."""

    def execute(self, context: ExecutionContext) -> ExecutionContext:
        from scipy import ndimage

        # Find connected components (non-zero regions)
        labeled, num_features = ndimage.label(context.current_grid != 0)

        objects = []
        for i in range(1, num_features + 1):
            mask = labeled == i
            # Get bounding box
            coords = np.argwhere(mask)
            if len(coords) > 0:
                y_min, x_min = coords.min(axis=0)
                y_max, x_max = coords.max(axis=0)
                # Extract object
                obj = context.current_grid[y_min : y_max + 1, x_min : x_max + 1].copy()
                obj[~mask[y_min : y_max + 1, x_min : x_max + 1]] = 0
                objects.append(obj)

        context.objects = objects
        return context

    def __str__(self) -> str:
        return "ExtractObjects()"


# Compositional Operators
class Sequence(Primitive):
    """Execute a sequence of primitives in order."""

    def __init__(self, primitives: List[Primitive]):
        self.primitives = primitives

    def execute(self, context: ExecutionContext) -> ExecutionContext:
        for primitive in self.primitives:
            context = primitive.execute(context)
        return context

    def __str__(self) -> str:
        steps = "; ".join(str(p) for p in self.primitives)
        return f"Sequence([{steps}])"


class Conditional(Primitive):
    """Execute a primitive if a condition is met."""

    def __init__(
        self,
        condition: Callable[[ExecutionContext], bool],
        then_branch: Primitive,
        else_branch: Optional[Primitive] = None,
    ):
        self.condition = condition
        self.then_branch = then_branch
        self.else_branch = else_branch

    def execute(self, context: ExecutionContext) -> ExecutionContext:
        if self.condition(context):
            return self.then_branch.execute(context)
        elif self.else_branch:
            return self.else_branch.execute(context)
        return context

    def __str__(self) -> str:
        else_str = f" else {self.else_branch}" if self.else_branch else ""
        return f"If(condition) then {self.then_branch}{else_str}"


class Loop(Primitive):
    """Execute a primitive multiple times."""

    def __init__(self, primitive: Primitive, times: int):
        self.primitive = primitive
        self.times = times

    def execute(self, context: ExecutionContext) -> ExecutionContext:
        for _ in range(self.times):
            context = self.primitive.execute(context)
        return context

    def __str__(self) -> str:
        return f"Loop({self.primitive}, times={self.times})"


class ForEachObject(Primitive):
    """Apply a primitive to each extracted object."""

    def __init__(self, primitive: Primitive):
        self.primitive = primitive

    def execute(self, context: ExecutionContext) -> ExecutionContext:
        if not context.objects:
            # Extract objects if not already done
            context = ExtractObjects().execute(context)

        if context.objects:
            processed_objects = []
            for obj in context.objects:
                # Create sub-context for object
                obj_context = ExecutionContext(
                    input_grid=obj.copy(),
                    current_grid=obj.copy(),
                    objects=None,
                    metadata=context.metadata,
                )
                # Apply primitive
                result = self.primitive.execute(obj_context)
                processed_objects.append(result.current_grid)

            context.objects = processed_objects

        return context

    def __str__(self) -> str:
        return f"ForEachObject({self.primitive})"


# Spatial Relations
class IsInside(Primitive):
    """Check if one region is inside another."""

    def __init__(self, color1: int, color2: int):
        self.color1 = color1
        self.color2 = color2

    def check(self, context: ExecutionContext) -> bool:
        mask1 = context.current_grid == self.color1
        mask2 = context.current_grid == self.color2

        if not mask1.any() or not mask2.any():
            return False

        # Get bounding boxes
        coords1 = np.argwhere(mask1)
        y1_min, x1_min = coords1.min(axis=0)
        y1_max, x1_max = coords1.max(axis=0)

        coords2 = np.argwhere(mask2)
        y2_min, x2_min = coords2.min(axis=0)
        y2_max, x2_max = coords2.max(axis=0)

        # Check if box1 is inside box2
        return (
            y1_min >= y2_min
            and y1_max <= y2_max
            and x1_min >= x2_min
            and x1_max <= x2_max
        )

    def execute(self, context: ExecutionContext) -> ExecutionContext:
        # This is used as a condition, not an operation
        return context

    def __str__(self) -> str:
        return f"IsInside({self.color1}, {self.color2})"


class IsAdjacent(Primitive):
    """Check if two regions are adjacent."""

    def __init__(self, color1: int, color2: int):
        self.color1 = color1
        self.color2 = color2

    def check(self, context: ExecutionContext) -> bool:
        mask1 = context.current_grid == self.color1
        mask2 = context.current_grid == self.color2

        if not mask1.any() or not mask2.any():
            return False

        # Dilate mask1 and check overlap with mask2
        dilated = ndimage.binary_dilation(mask1)
        return (dilated & mask2).any()

    def execute(self, context: ExecutionContext) -> ExecutionContext:
        return context

    def __str__(self) -> str:
        return f"IsAdjacent({self.color1}, {self.color2})"


# Pattern-specific operations
class TilePattern(Primitive):
    """Tile a pattern to fill a larger grid."""

    def __init__(self, scale_x: int, scale_y: int):
        self.scale_x = scale_x
        self.scale_y = scale_y

    def execute(self, context: ExecutionContext) -> ExecutionContext:
        h, w = context.current_grid.shape
        new_h, new_w = h * self.scale_y, w * self.scale_x
        new_grid = np.zeros((new_h, new_w), dtype=context.current_grid.dtype)

        for i in range(self.scale_y):
            for j in range(self.scale_x):
                new_grid[
                    i * h : (i + 1) * h, j * w : (j + 1) * w
                ] = context.current_grid

        context.current_grid = new_grid
        return context

    def __str__(self) -> str:
        return f"TilePattern({self.scale_x}, {self.scale_y})"


class DrawBorder(Primitive):
    """Draw a border around the grid."""

    def __init__(self, color: int, thickness: int = 1):
        self.color = color
        self.thickness = thickness

    def execute(self, context: ExecutionContext) -> ExecutionContext:
        h, w = context.current_grid.shape
        t = self.thickness

        # Top and bottom
        context.current_grid[:t, :] = self.color
        context.current_grid[-t:, :] = self.color

        # Left and right
        context.current_grid[:, :t] = self.color
        context.current_grid[:, -t:] = self.color

        return context

    def __str__(self) -> str:
        return f"DrawBorder(color={self.color}, thickness={self.thickness})"


class FillEnclosed(Primitive):
    """Fill enclosed regions with a color."""

    def __init__(self, boundary_color: int, fill_color: int):
        self.boundary_color = boundary_color
        self.fill_color = fill_color

    def execute(self, context: ExecutionContext) -> ExecutionContext:
        from scipy import ndimage

        # Create binary mask of boundaries
        boundary = context.current_grid == self.boundary_color

        # Fill holes (enclosed regions)
        filled = ndimage.binary_fill_holes(boundary)

        # Apply fill color to newly filled regions
        new_fill = filled & ~boundary
        context.current_grid[new_fill] = self.fill_color

        return context

    def __str__(self) -> str:
        return f"FillEnclosed(boundary={self.boundary_color}, fill={self.fill_color})"


# DSL Library Manager
class CompositionalDSL:
    """Manager for the compositional DSL primitives."""

    def __init__(self):
        self.primitives = {
            # Atomic operations
            "move": Move,
            "rotate": Rotate,
            "flip_h": FlipH,
            "flip_v": FlipV,
            "set_color": SetColor,
            "fill_rectangle": FillRectangle,
            "extract_objects": ExtractObjects,
            # Compositional operators
            "sequence": Sequence,
            "conditional": Conditional,
            "loop": Loop,
            "for_each_object": ForEachObject,
            # Spatial operations
            "is_inside": IsInside,
            "is_adjacent": IsAdjacent,
            # Pattern operations
            "tile_pattern": TilePattern,
            "draw_border": DrawBorder,
            "fill_enclosed": FillEnclosed,
        }

    def get_primitive(self, name: str, **kwargs) -> Primitive:
        """Get a primitive by name with parameters."""
        if name not in self.primitives:
            raise ValueError(f"Unknown primitive: {name}")
        return self.primitives[name](**kwargs)

    def get_all_primitives(self) -> Dict[str, type]:
        """Get all available primitives."""
        return self.primitives.copy()

    def create_program(self, operations: List[Tuple[str, Dict]]) -> Primitive:
        """Create a program from a list of (primitive_name, kwargs) tuples."""
        primitives = []
        for name, kwargs in operations:
            primitives.append(self.get_primitive(name, **kwargs))

        if len(primitives) == 1:
            return primitives[0]
        return Sequence(primitives)

    def execute_program(self, program: Primitive, input_grid: np.ndarray) -> np.ndarray:
        """Execute a program on an input grid."""
        context = ExecutionContext(
            input_grid=input_grid.copy(), current_grid=input_grid.copy()
        )
        result = program.execute(context)
        return result.current_grid


# Helper functions for common patterns
def create_color_mapping_program(color_map: Dict[int, int]) -> Primitive:
    """Create a program that applies a color mapping."""
    operations = []
    for from_color, to_color in color_map.items():
        operations.append(SetColor(from_color, to_color))
    return Sequence(operations) if len(operations) > 1 else operations[0]


def create_symmetry_program(axis: str = "vertical") -> Primitive:
    """Create a program that makes the grid symmetric."""
    if axis == "vertical":
        return Sequence([FlipH(), FlipH()])  # Placeholder - needs proper implementation
    else:
        return Sequence([FlipV(), FlipV()])


def create_object_transformation_program(transform: Primitive) -> Primitive:
    """Create a program that transforms all objects."""
    return Sequence([ExtractObjects(), ForEachObject(transform)])


if __name__ == "__main__":
    # Test the DSL
    dsl = CompositionalDSL()

    # Create a simple test grid
    test_grid = np.array([[1, 0, 2], [0, 3, 0], [2, 0, 1]])

    print("Original grid:")
    print(test_grid)

    # Test sequence of operations
    program = dsl.create_program(
        [
            ("set_color", {"from_color": 1, "to_color": 5}),
            ("rotate", {"angle": 90}),
            ("flip_h", {}),
        ]
    )

    result = dsl.execute_program(program, test_grid)

    print("\nProgram:", program)
    print("\nResult:")
    print(result)

    # Test conditional
    condition = lambda ctx: np.sum(ctx.current_grid == 3) > 0
    cond_program = Conditional(condition, SetColor(3, 7))

    test_context = ExecutionContext(
        input_grid=test_grid.copy(), current_grid=test_grid.copy()
    )
    result_context = cond_program.execute(test_context)

    print("\nConditional result:")
    print(result_context.current_grid)
