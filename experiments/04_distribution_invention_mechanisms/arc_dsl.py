#!/usr/bin/env python3
"""ARC Domain-Specific Language (DSL) for grid transformations.

This module provides a library of composable primitives for expressing
transformations on ARC grids, designed for program synthesis.
"""

from utils.imports import setup_project_paths

setup_project_paths()

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from object_manipulation import GridObject, ObjectManipulator


class ARCPrimitive(ABC):
    """Base class for DSL primitives."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        """Execute the primitive operation."""

    def __repr__(self) -> str:
        return f"{self.name}()"


# Object extraction primitives


class ExtractObjects(ARCPrimitive):
    """Extract all objects from a grid."""

    def __init__(self):
        super().__init__("objects")
        self.manipulator = ObjectManipulator()

    def execute(self, grid: np.ndarray) -> List[GridObject]:
        return self.manipulator.extract_objects(grid)


class ExtractLargest(ARCPrimitive):
    """Extract the largest object."""

    def __init__(self):
        super().__init__("largest")
        self.manipulator = ObjectManipulator()

    def execute(self, grid: np.ndarray) -> Optional[GridObject]:
        return self.manipulator.extract_largest(grid)


class ExtractSmallest(ARCPrimitive):
    """Extract the smallest object."""

    def __init__(self):
        super().__init__("smallest")
        self.manipulator = ObjectManipulator()

    def execute(self, grid: np.ndarray) -> Optional[GridObject]:
        return self.manipulator.extract_smallest(grid)


class ExtractByColor(ARCPrimitive):
    """Extract objects of a specific color."""

    def __init__(self, color: int):
        super().__init__(f"color_{color}")
        self.color = color
        self.manipulator = ObjectManipulator()

    def execute(self, grid: np.ndarray) -> List[GridObject]:
        return self.manipulator.extract_by_color(grid, self.color)


# Spatial transformation primitives


class Move(ARCPrimitive):
    """Move/translate the grid or objects."""

    def __init__(self, delta_row: int, delta_col: int):
        super().__init__(f"move_{delta_row}_{delta_col}")
        self.delta_row = delta_row
        self.delta_col = delta_col

    def execute(self, grid: np.ndarray) -> np.ndarray:
        result = np.zeros_like(grid)
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                new_r = r + self.delta_row
                new_c = c + self.delta_col
                if 0 <= new_r < grid.shape[0] and 0 <= new_c < grid.shape[1]:
                    result[new_r, new_c] = grid[r, c]
        return result


class Rotate(ARCPrimitive):
    """Rotate the grid."""

    def __init__(self, degrees: int):
        super().__init__(f"rotate_{degrees}")
        self.degrees = degrees

    def execute(self, grid: np.ndarray) -> np.ndarray:
        k = self.degrees // 90
        return np.rot90(grid, k=-k)


class Mirror(ARCPrimitive):
    """Mirror/flip the grid."""

    def __init__(self, axis: str = "horizontal"):
        super().__init__(f"mirror_{axis}")
        self.axis = axis

    def execute(self, grid: np.ndarray) -> np.ndarray:
        if self.axis == "horizontal":
            return np.flip(grid, axis=1)
        elif self.axis == "vertical":
            return np.flip(grid, axis=0)
        else:
            return grid


class Scale(ARCPrimitive):
    """Scale the grid by an integer factor."""

    def __init__(self, factor: int):
        super().__init__(f"scale_{factor}")
        self.factor = factor

    def execute(self, grid: np.ndarray) -> np.ndarray:
        return np.kron(grid, np.ones((self.factor, self.factor), dtype=grid.dtype))


# Filtering primitives


class FilterSize(ARCPrimitive):
    """Filter objects by size."""

    def __init__(self, min_size: int = 0, max_size: int = 1000):
        super().__init__(f"filter_size_{min_size}_{max_size}")
        self.min_size = min_size
        self.max_size = max_size
        self.manipulator = ObjectManipulator()

    def execute(self, grid: np.ndarray) -> np.ndarray:
        objects = self.manipulator.extract_objects(grid)
        result = np.zeros_like(grid)

        for obj in objects:
            if self.min_size <= obj.size <= self.max_size:
                result = self.manipulator.place_object(result, obj, obj.position)

        return result


class FilterColor(ARCPrimitive):
    """Keep only specific colors."""

    def __init__(self, colors: List[int]):
        super().__init__(f"filter_color_{colors}")
        self.colors = set(colors)

    def execute(self, grid: np.ndarray) -> np.ndarray:
        result = grid.copy()
        mask = np.isin(grid, list(self.colors))
        result[~mask] = 0
        return result


# Composition primitives


class Fill(ARCPrimitive):
    """Fill regions with a color."""

    def __init__(self, color: int, fill_type: str = "background"):
        super().__init__(f"fill_{color}_{fill_type}")
        self.color = color
        self.fill_type = fill_type
        self.manipulator = ObjectManipulator()

    def execute(self, grid: np.ndarray) -> np.ndarray:
        return self.manipulator.fill_with_color(grid, self.color, self.fill_type)


class Paint(ARCPrimitive):
    """Add objects to the grid."""

    def __init__(self):
        super().__init__("paint")
        self.manipulator = ObjectManipulator()

    def execute(self, grid: np.ndarray, objects: List[GridObject]) -> np.ndarray:
        result = grid.copy()
        for obj in objects:
            result = self.manipulator.place_object(result, obj, obj.position)
        return result


class Remove(ARCPrimitive):
    """Remove objects from the grid."""

    def __init__(self, condition: Optional[Callable] = None):
        super().__init__("remove")
        self.condition = condition
        self.manipulator = ObjectManipulator()

    def execute(self, grid: np.ndarray) -> np.ndarray:
        if self.condition is None:
            # Remove all objects (return empty grid)
            return np.zeros_like(grid)

        objects = self.manipulator.extract_objects(grid)
        result = np.zeros_like(grid)

        for obj in objects:
            if not self.condition(obj):
                result = self.manipulator.place_object(result, obj, obj.position)

        return result


class Replace(ARCPrimitive):
    """Replace colors or patterns."""

    def __init__(self, old_color: int, new_color: int):
        super().__init__(f"replace_{old_color}_{new_color}")
        self.old_color = old_color
        self.new_color = new_color

    def execute(self, grid: np.ndarray) -> np.ndarray:
        result = grid.copy()
        result[grid == self.old_color] = self.new_color
        return result


# Logic primitives


class IfThen(ARCPrimitive):
    """Conditional transformation."""

    def __init__(
        self,
        condition: Callable,
        then_op: ARCPrimitive,
        else_op: Optional[ARCPrimitive] = None,
    ):
        super().__init__("if_then")
        self.condition = condition
        self.then_op = then_op
        self.else_op = else_op

    def execute(self, grid: np.ndarray) -> np.ndarray:
        if self.condition(grid):
            return self.then_op.execute(grid)
        elif self.else_op:
            return self.else_op.execute(grid)
        return grid


class Repeat(ARCPrimitive):
    """Repeat a transformation multiple times."""

    def __init__(self, operation: ARCPrimitive, times: int):
        super().__init__(f"repeat_{times}")
        self.operation = operation
        self.times = times

    def execute(self, grid: np.ndarray) -> np.ndarray:
        result = grid
        for _ in range(self.times):
            result = self.operation.execute(result)
        return result


class Compose(ARCPrimitive):
    """Compose multiple transformations."""

    def __init__(self, operations: List[ARCPrimitive]):
        super().__init__("compose")
        self.operations = operations

    def execute(self, grid: np.ndarray) -> np.ndarray:
        result = grid
        for op in self.operations:
            result = op.execute(result)
        return result


# Advanced primitives


class CountObjects(ARCPrimitive):
    """Count objects in the grid."""

    def __init__(self):
        super().__init__("count")
        self.manipulator = ObjectManipulator()

    def execute(self, grid: np.ndarray) -> int:
        objects = self.manipulator.extract_objects(grid)
        return len(objects)


class DuplicateObjects(ARCPrimitive):
    """Duplicate objects in various patterns."""

    def __init__(self, pattern: str = "double"):
        super().__init__(f"duplicate_{pattern}")
        self.pattern = pattern
        self.manipulator = ObjectManipulator()

    def execute(self, grid: np.ndarray) -> np.ndarray:
        return self.manipulator.duplicate_objects(grid, self.pattern)


class CompletePattern(ARCPrimitive):
    """Complete a partial pattern."""

    def __init__(self):
        super().__init__("complete")
        self.manipulator = ObjectManipulator()

    def execute(self, grid: np.ndarray) -> np.ndarray:
        return self.manipulator.complete_pattern(grid)


class RearrangeObjects(ARCPrimitive):
    """Rearrange objects according to a pattern."""

    def __init__(self, arrangement: str = "sort_by_size"):
        super().__init__(f"rearrange_{arrangement}")
        self.arrangement = arrangement
        self.manipulator = ObjectManipulator()

    def execute(self, grid: np.ndarray) -> np.ndarray:
        return self.manipulator.rearrange_objects(grid, self.arrangement)


class Crop(ARCPrimitive):
    """Crop the grid to a specific region."""

    def __init__(self, row_start: int, row_end: int, col_start: int, col_end: int):
        super().__init__(f"crop_{row_start}_{row_end}_{col_start}_{col_end}")
        self.row_start = row_start
        self.row_end = row_end
        self.col_start = col_start
        self.col_end = col_end

    def execute(self, grid: np.ndarray) -> np.ndarray:
        return grid[self.row_start : self.row_end, self.col_start : self.col_end]


class Pad(ARCPrimitive):
    """Pad the grid with a color."""

    def __init__(self, padding: int, color: int = 0):
        super().__init__(f"pad_{padding}_{color}")
        self.padding = padding
        self.color = color

    def execute(self, grid: np.ndarray) -> np.ndarray:
        return np.pad(grid, self.padding, constant_values=self.color)


@dataclass
class Program:
    """Represents a sequence of DSL operations."""

    operations: List[ARCPrimitive]

    def execute(self, grid: np.ndarray) -> np.ndarray:
        """Execute the program on a grid."""
        result = grid
        for op in self.operations:
            result = op.execute(result)
        return result

    def to_code(self) -> str:
        """Generate human-readable code representation."""
        lines = ["# ARC DSL Program"]
        for i, op in enumerate(self.operations):
            lines.append(f"step_{i}: {op}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"Program({len(self.operations)} operations)"


class DSLLibrary:
    """Library of composable DSL primitives."""

    def __init__(self):
        self.primitives = self._build_primitive_library()

    def _build_primitive_library(self) -> Dict[str, type]:
        """Build the library of available primitives."""
        return {
            # Object operations
            "objects": ExtractObjects,
            "largest": ExtractLargest,
            "smallest": ExtractSmallest,
            "color": ExtractByColor,
            # Spatial operations
            "move": Move,
            "rotate": Rotate,
            "mirror": Mirror,
            "scale": Scale,
            # Filtering
            "filter_size": FilterSize,
            "filter_color": FilterColor,
            # Composition
            "fill": Fill,
            "paint": Paint,
            "remove": Remove,
            "replace": Replace,
            # Logic
            "if_then": IfThen,
            "repeat": Repeat,
            "compose": Compose,
            # Advanced
            "count": CountObjects,
            "duplicate": DuplicateObjects,
            "complete": CompletePattern,
            "rearrange": RearrangeObjects,
            "crop": Crop,
            "pad": Pad,
        }

    def get_primitive(self, name: str, **kwargs) -> ARCPrimitive:
        """Get an instance of a primitive by name."""
        if name not in self.primitives:
            raise ValueError(f"Unknown primitive: {name}")

        primitive_class = self.primitives[name]

        # Handle different primitive constructors
        if name == "objects" or name == "largest" or name == "smallest":
            return primitive_class()
        elif name == "color":
            return primitive_class(kwargs.get("color", 1))
        elif name == "move":
            return primitive_class(
                kwargs.get("delta_row", 0), kwargs.get("delta_col", 0)
            )
        elif name == "rotate":
            return primitive_class(kwargs.get("degrees", 90))
        elif name == "mirror":
            return primitive_class(kwargs.get("axis", "horizontal"))
        elif name == "scale":
            return primitive_class(kwargs.get("factor", 2))
        elif name == "filter_size":
            return primitive_class(
                kwargs.get("min_size", 0), kwargs.get("max_size", 1000)
            )
        elif name == "filter_color":
            return primitive_class(kwargs.get("colors", [1]))
        elif name == "fill":
            return primitive_class(
                kwargs.get("color", 1), kwargs.get("fill_type", "background")
            )
        elif name == "replace":
            return primitive_class(
                kwargs.get("old_color", 0), kwargs.get("new_color", 1)
            )
        elif name == "duplicate":
            return primitive_class(kwargs.get("pattern", "double"))
        elif name == "rearrange":
            return primitive_class(kwargs.get("arrangement", "sort_by_size"))
        elif name == "crop":
            return primitive_class(
                kwargs.get("row_start", 0),
                kwargs.get("row_end", 10),
                kwargs.get("col_start", 0),
                kwargs.get("col_end", 10),
            )
        elif name == "pad":
            return primitive_class(kwargs.get("padding", 1), kwargs.get("color", 0))
        else:
            return primitive_class()

    def create_program(self, operation_specs: List[Tuple[str, Dict]]) -> Program:
        """Create a program from operation specifications.

        Args:
            operation_specs: List of (operation_name, kwargs) tuples

        Returns:
            Program instance
        """
        operations = []
        for op_name, kwargs in operation_specs:
            primitive = self.get_primitive(op_name, **kwargs)
            operations.append(primitive)

        return Program(operations)

    def get_all_primitive_names(self) -> List[str]:
        """Get names of all available primitives."""
        return list(self.primitives.keys())


def test_dsl():
    """Test the ARC DSL."""
    print("Testing ARC DSL")
    print("=" * 50)

    # Create test grid
    test_grid = np.array(
        [
            [0, 1, 1, 0, 0],
            [0, 1, 1, 0, 0],
            [2, 2, 0, 3, 3],
            [2, 2, 0, 3, 3],
            [0, 0, 0, 0, 0],
        ]
    )

    print("Original grid:")
    print(test_grid)
    print()

    # Test DSL library
    library = DSLLibrary()

    # Test simple transformations
    print("Testing mirror (horizontal):")
    mirror_op = library.get_primitive("mirror", axis="horizontal")
    result = mirror_op.execute(test_grid)
    print(result)
    print()

    print("Testing scale (2x):")
    scale_op = library.get_primitive("scale", factor=2)
    result = scale_op.execute(test_grid)
    print(result)
    print()

    print("Testing filter by size (keep >= 4):")
    filter_op = library.get_primitive("filter_size", min_size=4)
    result = filter_op.execute(test_grid)
    print(result)
    print()

    # Test program composition
    print("Testing program composition (mirror then scale):")
    program = library.create_program(
        [("mirror", {"axis": "horizontal"}), ("scale", {"factor": 2})]
    )
    result = program.execute(test_grid)
    print(result)
    print()

    print("Program representation:")
    print(program.to_code())
    print()

    # Test more complex program
    print("Testing complex program:")
    complex_program = library.create_program(
        [
            ("filter_size", {"min_size": 4}),  # Keep only large objects
            ("duplicate", {"pattern": "double"}),  # Duplicate them
            ("fill", {"color": 5, "fill_type": "background"}),  # Fill background
        ]
    )
    result = complex_program.execute(test_grid)
    print(result)


if __name__ == "__main__":
    test_dsl()
