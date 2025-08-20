#!/usr/bin/env python3
"""Object Manipulation Library for ARC-AGI Tasks.

This module provides comprehensive object extraction, transformation, and placement
capabilities for solving ARC tasks that involve object-based reasoning.
"""

from utils.imports import setup_project_paths

setup_project_paths()

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy import ndimage


@dataclass
class GridObject:
    """Represents an extractable object from a grid."""

    color: int
    pixels: List[Tuple[int, int]]  # List of (row, col) positions
    bounding_box: Tuple[int, int, int, int]  # min_row, min_col, max_row, max_col
    mask: np.ndarray  # Binary mask of object shape
    position: Tuple[int, int]  # Top-left corner position

    @property
    def size(self) -> int:
        """Number of pixels in the object."""
        return len(self.pixels)

    @property
    def height(self) -> int:
        """Height of the object's bounding box."""
        return self.bounding_box[2] - self.bounding_box[0] + 1

    @property
    def width(self) -> int:
        """Width of the object's bounding box."""
        return self.bounding_box[3] - self.bounding_box[1] + 1

    @property
    def center(self) -> Tuple[float, float]:
        """Center of the object."""
        rows = [p[0] for p in self.pixels]
        cols = [p[1] for p in self.pixels]
        return (np.mean(rows), np.mean(cols))

    def copy(self) -> "GridObject":
        """Create a deep copy of the object."""
        return GridObject(
            color=self.color,
            pixels=self.pixels.copy(),
            bounding_box=self.bounding_box,
            mask=self.mask.copy(),
            position=self.position,
        )


class ObjectManipulator:
    """Core object manipulation operations for ARC tasks."""

    def extract_objects(
        self, grid: np.ndarray, background: int = 0
    ) -> List[GridObject]:
        """Extract all objects from a grid.

        Args:
            grid: Input grid
            background: Background color to ignore (default 0)

        Returns:
            List of GridObject instances
        """
        objects = []

        # Get unique non-background colors
        unique_colors = np.unique(grid)
        unique_colors = unique_colors[unique_colors != background]

        for color in unique_colors:
            # Create binary mask for this color
            mask = (grid == color).astype(int)

            # Find connected components
            labeled, num_features = ndimage.label(mask)

            for i in range(1, num_features + 1):
                component_mask = labeled == i
                pixels = list(zip(*np.where(component_mask)))

                if pixels:
                    obj = self._create_object(color, pixels, component_mask)
                    objects.append(obj)

        return objects

    def extract_by_color(self, grid: np.ndarray, color: int) -> List[GridObject]:
        """Extract all objects of a specific color."""
        mask = (grid == color).astype(int)
        labeled, num_features = ndimage.label(mask)

        objects = []
        for i in range(1, num_features + 1):
            component_mask = labeled == i
            pixels = list(zip(*np.where(component_mask)))
            if pixels:
                obj = self._create_object(color, pixels, component_mask)
                objects.append(obj)

        return objects

    def extract_largest(
        self, grid: np.ndarray, background: int = 0
    ) -> Optional[GridObject]:
        """Extract the largest object from the grid."""
        objects = self.extract_objects(grid, background)
        if not objects:
            return None
        return max(objects, key=lambda o: o.size)

    def extract_smallest(
        self, grid: np.ndarray, background: int = 0
    ) -> Optional[GridObject]:
        """Extract the smallest object from the grid."""
        objects = self.extract_objects(grid, background)
        if not objects:
            return None
        return min(objects, key=lambda o: o.size)

    def rotate_object(self, obj: GridObject, degrees: int) -> GridObject:
        """Rotate an object by specified degrees (90, 180, 270).

        Args:
            obj: Object to rotate
            degrees: Rotation angle (must be multiple of 90)

        Returns:
            Rotated GridObject
        """
        if degrees not in [90, 180, 270]:
            raise ValueError("Rotation must be 90, 180, or 270 degrees")

        # Number of 90-degree rotations
        k = degrees // 90

        # Rotate the mask
        rotated_mask = np.rot90(obj.mask, k=-k)

        # Find new pixel positions
        pixels = list(zip(*np.where(rotated_mask)))

        return self._create_object(obj.color, pixels, rotated_mask)

    def scale_object(self, obj: GridObject, factor: int) -> GridObject:
        """Scale an object by an integer factor.

        Args:
            obj: Object to scale
            factor: Scaling factor (2 = double size, etc.)

        Returns:
            Scaled GridObject
        """
        if factor <= 0:
            raise ValueError("Scale factor must be positive")

        # Scale the mask
        scaled_mask = np.kron(obj.mask, np.ones((factor, factor), dtype=obj.mask.dtype))

        # Find new pixel positions
        pixels = list(zip(*np.where(scaled_mask)))

        return self._create_object(obj.color, pixels, scaled_mask)

    def mirror_object(self, obj: GridObject, axis: str = "horizontal") -> GridObject:
        """Mirror an object along specified axis.

        Args:
            obj: Object to mirror
            axis: 'horizontal' or 'vertical'

        Returns:
            Mirrored GridObject
        """
        if axis == "horizontal":
            mirrored_mask = np.flip(obj.mask, axis=1)
        elif axis == "vertical":
            mirrored_mask = np.flip(obj.mask, axis=0)
        else:
            raise ValueError("Axis must be 'horizontal' or 'vertical'")

        pixels = list(zip(*np.where(mirrored_mask)))
        return self._create_object(obj.color, pixels, mirrored_mask)

    def move_object(
        self, obj: GridObject, delta_row: int, delta_col: int
    ) -> GridObject:
        """Move an object by specified offset.

        Args:
            obj: Object to move
            delta_row: Row offset
            delta_col: Column offset

        Returns:
            Moved GridObject
        """
        new_pixels = [(r + delta_row, c + delta_col) for r, c in obj.pixels]
        new_bb = (
            obj.bounding_box[0] + delta_row,
            obj.bounding_box[1] + delta_col,
            obj.bounding_box[2] + delta_row,
            obj.bounding_box[3] + delta_col,
        )

        return GridObject(
            color=obj.color,
            pixels=new_pixels,
            bounding_box=new_bb,
            mask=obj.mask.copy(),
            position=(obj.position[0] + delta_row, obj.position[1] + delta_col),
        )

    def place_object(
        self,
        grid: np.ndarray,
        obj: GridObject,
        position: Tuple[int, int],
        overwrite: bool = True,
    ) -> np.ndarray:
        """Place an object on a grid at specified position.

        Args:
            grid: Target grid
            obj: Object to place
            position: (row, col) position for top-left corner
            overwrite: Whether to overwrite existing pixels

        Returns:
            Modified grid
        """
        result = grid.copy()

        for r in range(obj.mask.shape[0]):
            for c in range(obj.mask.shape[1]):
                if obj.mask[r, c]:
                    target_r = position[0] + r
                    target_c = position[1] + c

                    # Check bounds
                    if 0 <= target_r < grid.shape[0] and 0 <= target_c < grid.shape[1]:
                        if overwrite or result[target_r, target_c] == 0:
                            result[target_r, target_c] = obj.color

        return result

    def arrange_objects(
        self,
        objects: List[GridObject],
        pattern: str,
        grid_shape: Tuple[int, int] = None,
    ) -> np.ndarray:
        """Arrange multiple objects in a pattern.

        Args:
            objects: List of objects to arrange
            pattern: 'row', 'column', 'grid', 'diagonal'
            grid_shape: Shape of output grid (auto-sized if None)

        Returns:
            Grid with arranged objects
        """
        if not objects:
            return np.zeros((1, 1), dtype=np.int32)

        if pattern == "row":
            return self._arrange_row(objects, grid_shape)
        elif pattern == "column":
            return self._arrange_column(objects, grid_shape)
        elif pattern == "grid":
            return self._arrange_grid(objects, grid_shape)
        elif pattern == "diagonal":
            return self._arrange_diagonal(objects, grid_shape)
        else:
            raise ValueError(f"Unknown pattern: {pattern}")

    def tile_object(
        self, obj: GridObject, rows: int, cols: int, spacing: int = 0
    ) -> np.ndarray:
        """Tile an object in a grid pattern.

        Args:
            obj: Object to tile
            rows: Number of rows
            cols: Number of columns
            spacing: Spacing between tiles

        Returns:
            Grid with tiled objects
        """
        tile_h = obj.height + spacing
        tile_w = obj.width + spacing

        grid_h = rows * tile_h - spacing
        grid_w = cols * tile_w - spacing

        grid = np.zeros((grid_h, grid_w), dtype=np.int32)

        for r in range(rows):
            for c in range(cols):
                pos = (r * tile_h, c * tile_w)
                grid = self.place_object(grid, obj, pos)

        return grid

    def duplicate_objects(
        self, grid: np.ndarray, duplication_type: str = "double"
    ) -> np.ndarray:
        """Duplicate objects in the grid.

        Args:
            grid: Input grid
            duplication_type: 'double', 'mirror', 'tile'

        Returns:
            Grid with duplicated objects
        """
        objects = self.extract_objects(grid)

        if not objects:
            return grid

        if duplication_type == "double":
            # Place each object twice, offset
            result = grid.copy()
            for obj in objects:
                # Try to place to the right
                new_pos = (obj.position[0], obj.position[1] + obj.width + 1)
                if new_pos[1] + obj.width <= grid.shape[1]:
                    result = self.place_object(result, obj, new_pos, overwrite=False)
                else:
                    # Try below if no room to the right
                    new_pos = (obj.position[0] + obj.height + 1, obj.position[1])
                    if new_pos[0] + obj.height <= grid.shape[0]:
                        result = self.place_object(
                            result, obj, new_pos, overwrite=False
                        )
            return result

        elif duplication_type == "mirror":
            # Mirror objects horizontally
            result = grid.copy()
            for obj in objects:
                mirrored = self.mirror_object(obj, "horizontal")
                # Place mirrored version to the right
                new_pos = (obj.position[0], grid.shape[1] - obj.position[1] - obj.width)
                result = self.place_object(result, mirrored, new_pos, overwrite=False)
            return result

        elif duplication_type == "tile":
            # Tile the largest object
            largest = self.extract_largest(grid)
            if largest:
                return self.tile_object(largest, 2, 2)
            return grid

        return grid

    def remove_small_objects(self, grid: np.ndarray, threshold: int = 3) -> np.ndarray:
        """Remove objects smaller than threshold.

        Args:
            grid: Input grid
            threshold: Minimum object size to keep

        Returns:
            Grid with small objects removed
        """
        objects = self.extract_objects(grid)
        result = np.zeros_like(grid)

        for obj in objects:
            if obj.size >= threshold:
                result = self.place_object(result, obj, obj.position)

        return result

    def rearrange_objects(
        self, grid: np.ndarray, arrangement: str = "sort_by_size"
    ) -> np.ndarray:
        """Rearrange objects in the grid.

        Args:
            grid: Input grid
            arrangement: 'sort_by_size', 'sort_by_color', 'reverse', 'center'

        Returns:
            Grid with rearranged objects
        """
        objects = self.extract_objects(grid)

        if not objects:
            return grid

        if arrangement == "sort_by_size":
            objects.sort(key=lambda o: o.size)
            return self.arrange_objects(objects, "row", grid.shape)

        elif arrangement == "sort_by_color":
            objects.sort(key=lambda o: o.color)
            return self.arrange_objects(objects, "row", grid.shape)

        elif arrangement == "reverse":
            objects.reverse()
            return self.arrange_objects(objects, "row", grid.shape)

        elif arrangement == "center":
            # Move all objects to center
            result = np.zeros_like(grid)
            center_r = grid.shape[0] // 2
            center_c = grid.shape[1] // 2

            for obj in objects:
                # Calculate offset to center the object
                obj_center = obj.center
                delta_r = int(center_r - obj_center[0])
                delta_c = int(center_c - obj_center[1])
                moved = self.move_object(obj, delta_r, delta_c)
                result = self.place_object(
                    result, moved, moved.position, overwrite=False
                )

            return result

        return grid

    def complete_pattern(self, grid: np.ndarray) -> np.ndarray:
        """Attempt to complete a pattern in the grid.

        Args:
            grid: Input grid with partial pattern

        Returns:
            Grid with completed pattern
        """
        # Simple pattern completion: if we see a repeating pattern, continue it
        objects = self.extract_objects(grid)

        if len(objects) >= 2:
            # Check if objects are in a line
            positions = [obj.position for obj in objects]

            # Check horizontal pattern
            if all(p[0] == positions[0][0] for p in positions):
                # Objects are in same row
                if len(positions) >= 2:
                    spacing = positions[1][1] - positions[0][1]
                    # Add one more object following the pattern
                    last_obj = objects[-1]
                    new_pos = (last_obj.position[0], last_obj.position[1] + spacing)
                    if new_pos[1] + last_obj.width <= grid.shape[1]:
                        result = grid.copy()
                        return self.place_object(result, last_obj, new_pos)

            # Check vertical pattern
            if all(p[1] == positions[0][1] for p in positions):
                # Objects are in same column
                if len(positions) >= 2:
                    spacing = positions[1][0] - positions[0][0]
                    # Add one more object following the pattern
                    last_obj = objects[-1]
                    new_pos = (last_obj.position[0] + spacing, last_obj.position[1])
                    if new_pos[0] + last_obj.height <= grid.shape[0]:
                        result = grid.copy()
                        return self.place_object(result, last_obj, new_pos)

        return grid

    def fill_with_color(
        self, grid: np.ndarray, target_color: int, fill_type: str = "background"
    ) -> np.ndarray:
        """Fill regions with a specific color.

        Args:
            grid: Input grid
            target_color: Color to fill with
            fill_type: 'background', 'objects', 'border'

        Returns:
            Grid with filled regions
        """
        result = grid.copy()

        if fill_type == "background":
            # Fill empty spaces
            result[result == 0] = target_color

        elif fill_type == "objects":
            # Fill all non-background pixels
            result[result != 0] = target_color

        elif fill_type == "border":
            # Fill border pixels
            result[0, :] = target_color
            result[-1, :] = target_color
            result[:, 0] = target_color
            result[:, -1] = target_color

        return result

    # Helper methods

    def _create_object(
        self, color: int, pixels: List[Tuple[int, int]], mask: np.ndarray = None
    ) -> GridObject:
        """Create a GridObject from pixels."""
        if not pixels:
            raise ValueError("Cannot create object with no pixels")

        rows = [p[0] for p in pixels]
        cols = [p[1] for p in pixels]

        min_row, max_row = min(rows), max(rows)
        min_col, max_col = min(cols), max(cols)

        # Create mask if not provided
        if mask is None:
            height = max_row - min_row + 1
            width = max_col - min_col + 1
            mask = np.zeros((height, width), dtype=bool)
            for r, c in pixels:
                mask[r - min_row, c - min_col] = True
        else:
            # Extract the bounding box from the full mask
            if (
                mask.shape[0] > max_row - min_row + 1
                or mask.shape[1] > max_col - min_col + 1
            ):
                mask = mask[min_row : max_row + 1, min_col : max_col + 1]

        return GridObject(
            color=color,
            pixels=pixels,
            bounding_box=(min_row, min_col, max_row, max_col),
            mask=mask,
            position=(min_row, min_col),
        )

    def _arrange_row(
        self, objects: List[GridObject], grid_shape: Optional[Tuple[int, int]]
    ) -> np.ndarray:
        """Arrange objects in a row."""
        if not objects:
            return np.zeros((1, 1), dtype=np.int32)

        # Calculate required size
        max_height = max(obj.height for obj in objects)
        total_width = sum(obj.width for obj in objects) + len(objects) - 1

        if grid_shape:
            grid = np.zeros(grid_shape, dtype=np.int32)
        else:
            grid = np.zeros((max_height, total_width), dtype=np.int32)

        col_offset = 0
        for obj in objects:
            grid = self.place_object(grid, obj, (0, col_offset))
            col_offset += obj.width + 1

        return grid

    def _arrange_column(
        self, objects: List[GridObject], grid_shape: Optional[Tuple[int, int]]
    ) -> np.ndarray:
        """Arrange objects in a column."""
        if not objects:
            return np.zeros((1, 1), dtype=np.int32)

        # Calculate required size
        max_width = max(obj.width for obj in objects)
        total_height = sum(obj.height for obj in objects) + len(objects) - 1

        if grid_shape:
            grid = np.zeros(grid_shape, dtype=np.int32)
        else:
            grid = np.zeros((total_height, max_width), dtype=np.int32)

        row_offset = 0
        for obj in objects:
            grid = self.place_object(grid, obj, (row_offset, 0))
            row_offset += obj.height + 1

        return grid

    def _arrange_grid(
        self, objects: List[GridObject], grid_shape: Optional[Tuple[int, int]]
    ) -> np.ndarray:
        """Arrange objects in a grid pattern."""
        if not objects:
            return np.zeros((1, 1), dtype=np.int32)

        # Calculate grid dimensions
        n = len(objects)
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))

        # Calculate cell size
        max_height = max(obj.height for obj in objects)
        max_width = max(obj.width for obj in objects)

        if grid_shape:
            grid = np.zeros(grid_shape, dtype=np.int32)
        else:
            grid_h = rows * (max_height + 1) - 1
            grid_w = cols * (max_width + 1) - 1
            grid = np.zeros((grid_h, grid_w), dtype=np.int32)

        for i, obj in enumerate(objects):
            row = i // cols
            col = i % cols
            pos = (row * (max_height + 1), col * (max_width + 1))
            grid = self.place_object(grid, obj, pos)

        return grid

    def _arrange_diagonal(
        self, objects: List[GridObject], grid_shape: Optional[Tuple[int, int]]
    ) -> np.ndarray:
        """Arrange objects diagonally."""
        if not objects:
            return np.zeros((1, 1), dtype=np.int32)

        # Calculate required size
        max_size = max(max(obj.height, obj.width) for obj in objects)
        n = len(objects)
        grid_size = n * (max_size + 1)

        if grid_shape:
            grid = np.zeros(grid_shape, dtype=np.int32)
        else:
            grid = np.zeros((grid_size, grid_size), dtype=np.int32)

        offset = 0
        for obj in objects:
            grid = self.place_object(grid, obj, (offset, offset))
            offset += max_size + 1

        return grid

    # Relationship checking methods

    def objects_touching(self, obj1: GridObject, obj2: GridObject) -> bool:
        """Check if two objects are touching (adjacent pixels)."""
        for r1, c1 in obj1.pixels:
            for r2, c2 in obj2.pixels:
                if abs(r1 - r2) + abs(c1 - c2) == 1:
                    return True
        return False

    def object_inside(self, inner: GridObject, outer: GridObject) -> bool:
        """Check if one object is inside another."""
        inner_bb = inner.bounding_box
        outer_bb = outer.bounding_box

        return (
            outer_bb[0] <= inner_bb[0]
            and outer_bb[1] <= inner_bb[1]
            and outer_bb[2] >= inner_bb[2]
            and outer_bb[3] >= inner_bb[3]
        )

    def align_objects(
        self, objects: List[GridObject], alignment: str = "horizontal"
    ) -> List[GridObject]:
        """Align objects along an axis.

        Args:
            objects: Objects to align
            alignment: 'horizontal', 'vertical', 'center'

        Returns:
            List of aligned objects
        """
        if not objects:
            return []

        aligned = []

        if alignment == "horizontal":
            # Align all objects to same row
            target_row = objects[0].position[0]
            for obj in objects:
                delta_row = target_row - obj.position[0]
                aligned.append(self.move_object(obj, delta_row, 0))

        elif alignment == "vertical":
            # Align all objects to same column
            target_col = objects[0].position[1]
            for obj in objects:
                delta_col = target_col - obj.position[1]
                aligned.append(self.move_object(obj, 0, delta_col))

        elif alignment == "center":
            # Align centers
            target_center = objects[0].center
            for obj in objects:
                obj_center = obj.center
                delta_row = int(target_center[0] - obj_center[0])
                delta_col = int(target_center[1] - obj_center[1])
                aligned.append(self.move_object(obj, delta_row, delta_col))

        return aligned


def test_object_manipulation():
    """Test the object manipulation library."""
    print("Testing Object Manipulation Library")
    print("=" * 50)

    manipulator = ObjectManipulator()

    # Create test grid with objects
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

    # Test object extraction
    objects = manipulator.extract_objects(test_grid)
    print(f"Extracted {len(objects)} objects")
    for i, obj in enumerate(objects):
        print(
            f"  Object {i}: color={obj.color}, size={obj.size}, position={obj.position}"
        )

    # Test object duplication
    print("\nTesting duplication:")
    duplicated = manipulator.duplicate_objects(test_grid, "double")
    print(duplicated)

    # Test object removal
    print("\nTesting small object removal (threshold=5):")
    filtered = manipulator.remove_small_objects(test_grid, threshold=5)
    print(filtered)

    # Test object rearrangement
    print("\nTesting rearrangement (sort by size):")
    rearranged = manipulator.rearrange_objects(test_grid, "sort_by_size")
    print(rearranged)

    # Test object transformation
    if objects:
        print("\nTesting object rotation:")
        first_obj = objects[0]
        rotated = manipulator.rotate_object(first_obj, 90)
        grid = np.zeros((5, 5), dtype=np.int32)
        result = manipulator.place_object(grid, rotated, (0, 0))
        print(result)

        print("\nTesting object scaling (2x):")
        scaled = manipulator.scale_object(first_obj, 2)
        grid = np.zeros((8, 8), dtype=np.int32)
        result = manipulator.place_object(grid, scaled, (0, 0))
        print(result)

    # Test pattern completion
    pattern_grid = np.array(
        [
            [1, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
    )
    print("\nTesting pattern completion:")
    print("Input pattern:")
    print(pattern_grid)
    completed = manipulator.complete_pattern(pattern_grid)
    print("Completed pattern:")
    print(completed)


if __name__ == "__main__":
    test_object_manipulation()
