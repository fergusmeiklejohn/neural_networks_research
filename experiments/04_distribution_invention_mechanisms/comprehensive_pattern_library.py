#!/usr/bin/env python3
"""Comprehensive pattern library for ARC tasks.

Based on analysis showing we need:
1. Non-size-change transformations (75% of unsolved tasks)
2. Object manipulation (80% of unsolved tasks have multiple objects)
3. Color transformations (55% of unsolved tasks)
4. Symmetry operations (20% of unsolved tasks)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class TransformResult:
    """Result of applying a transformation."""

    output: np.ndarray
    confidence: float
    method: str
    metadata: Dict = None


class PatternPrimitive(ABC):
    """Base class for all pattern primitives."""

    @abstractmethod
    def can_apply(
        self, examples: List[Tuple[np.ndarray, np.ndarray]], test_input: np.ndarray
    ) -> bool:
        """Check if this primitive can be applied."""

    @abstractmethod
    def apply(
        self, examples: List[Tuple[np.ndarray, np.ndarray]], test_input: np.ndarray
    ) -> TransformResult:
        """Apply the transformation."""

    def validate_on_examples(
        self, examples: List[Tuple[np.ndarray, np.ndarray]], transform_func
    ) -> float:
        """Validate transformation on training examples."""
        if not examples:
            return 0.0

        correct = 0
        for inp, expected in examples:
            try:
                result = transform_func(inp)
                if np.array_equal(result, expected):
                    correct += 1
            except:
                pass

        return correct / len(examples)


# ============================================================================
# GEOMETRIC TRANSFORMATIONS (no size change)
# ============================================================================


class RotationReflection(PatternPrimitive):
    """Handles rotation and reflection transformations."""

    def can_apply(self, examples, test_input):
        """Check if rotation/reflection pattern exists."""
        if not examples:
            return False

        # Check if output has same shape as input
        for inp, out in examples:
            if inp.shape != out.shape:
                return False

        # Try to detect transformation
        return self._detect_transformation(examples) is not None

    def _detect_transformation(self, examples):
        """Detect which transformation is being applied."""
        transformations = {
            "rotate_90": lambda x: np.rot90(x, 1),
            "rotate_180": lambda x: np.rot90(x, 2),
            "rotate_270": lambda x: np.rot90(x, 3),
            "flip_horizontal": lambda x: np.flip(x, axis=0),
            "flip_vertical": lambda x: np.flip(x, axis=1),
            "transpose": lambda x: x.T if x.shape[0] == x.shape[1] else x,
            "flip_diagonal": lambda x: np.flip(x.T, axis=0)
            if x.shape[0] == x.shape[1]
            else x,
        }

        for name, transform in transformations.items():
            valid = True
            for inp, expected in examples:
                try:
                    result = transform(inp)
                    if not np.array_equal(result, expected):
                        valid = False
                        break
                except:
                    valid = False
                    break

            if valid:
                return (name, transform)

        return None

    def apply(self, examples, test_input):
        """Apply detected transformation."""
        result = self._detect_transformation(examples)
        if not result:
            return TransformResult(test_input, 0.0, "rotation_failed")

        name, transform = result
        output = transform(test_input)
        confidence = self.validate_on_examples(examples, transform)

        return TransformResult(output, confidence, f"rotation_{name}")


# ============================================================================
# COLOR TRANSFORMATIONS
# ============================================================================


class ColorMapper(PatternPrimitive):
    """Maps colors based on learned rules."""

    def can_apply(self, examples, test_input):
        """Check if color mapping pattern exists."""
        if not examples:
            return False

        # Check same shape
        for inp, out in examples:
            if inp.shape != out.shape:
                return False

        # Check if color mapping is consistent
        return self._learn_color_mapping(examples) is not None

    def _learn_color_mapping(self, examples):
        """Learn color mapping from examples."""
        # Collect all color mappings
        color_map = {}

        for inp, out in examples:
            for i in range(inp.shape[0]):
                for j in range(inp.shape[1]):
                    src_color = int(inp[i, j])
                    dst_color = int(out[i, j])

                    if src_color in color_map:
                        if color_map[src_color] != dst_color:
                            # Inconsistent mapping
                            return None
                    else:
                        color_map[src_color] = dst_color

        return color_map if color_map else None

    def apply(self, examples, test_input):
        """Apply color mapping."""
        color_map = self._learn_color_mapping(examples)
        if not color_map:
            return TransformResult(test_input, 0.0, "color_mapping_failed")

        output = np.zeros_like(test_input)
        for i in range(test_input.shape[0]):
            for j in range(test_input.shape[1]):
                src_color = int(test_input[i, j])
                output[i, j] = color_map.get(src_color, src_color)

        # Validate
        confidence = self.validate_on_examples(
            examples, lambda x: self._apply_color_map(x, color_map)
        )

        return TransformResult(output, confidence, "color_mapping")

    def _apply_color_map(self, grid, color_map):
        """Helper to apply color map."""
        output = np.zeros_like(grid)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                src_color = int(grid[i, j])
                output[i, j] = color_map.get(src_color, src_color)
        return output


# ============================================================================
# OBJECT DETECTION AND MANIPULATION
# ============================================================================


class ObjectExtractor(PatternPrimitive):
    """Extracts and manipulates discrete objects."""

    def can_apply(self, examples, test_input):
        """Check if object manipulation is needed."""
        # This is complex - for now, check if there are multiple colors
        unique_colors = len(np.unique(test_input))
        return unique_colors > 2

    def _find_objects(self, grid, background=0):
        """Find connected components (objects) in grid."""
        from scipy import ndimage

        objects = []
        visited = np.zeros_like(grid, dtype=bool)

        for color in np.unique(grid):
            if color == background:
                continue

            # Find all pixels of this color
            mask = grid == color
            labeled, num_features = ndimage.label(mask)

            for obj_id in range(1, num_features + 1):
                obj_mask = labeled == obj_id
                coords = np.argwhere(obj_mask)

                if len(coords) > 0:
                    objects.append(
                        {
                            "color": color,
                            "coords": coords,
                            "bbox": (coords.min(axis=0), coords.max(axis=0)),
                            "size": len(coords),
                            "mask": obj_mask,
                        }
                    )

        return objects

    def apply(self, examples, test_input):
        """Extract and potentially transform objects."""
        try:
            objects = self._find_objects(test_input)

            # For now, just return object count visualization
            # This would be extended with specific object transformations
            output = np.zeros_like(test_input)
            for i, obj in enumerate(objects):
                for coord in obj["coords"]:
                    output[coord[0], coord[1]] = (i + 1) % 10

            return TransformResult(
                output, 0.3, f"object_extraction_{len(objects)}_objects"
            )
        except:
            return TransformResult(test_input, 0.0, "object_extraction_failed")


# ============================================================================
# SYMMETRY OPERATIONS
# ============================================================================


class SymmetryApplier(PatternPrimitive):
    """Applies symmetry operations."""

    def can_apply(self, examples, test_input):
        """Check if symmetry operations are needed."""
        # Check if outputs show symmetry patterns
        for inp, out in examples:
            if self._has_symmetry(out):
                return True
        return False

    def _has_symmetry(self, grid):
        """Check if grid has symmetry."""
        h_sym = np.array_equal(grid, np.flip(grid, axis=0))
        v_sym = np.array_equal(grid, np.flip(grid, axis=1))

        if grid.shape[0] == grid.shape[1]:
            d_sym = np.array_equal(grid, grid.T)
            ad_sym = np.array_equal(grid, np.flip(grid.T))
        else:
            d_sym = ad_sym = False

        return h_sym or v_sym or d_sym or ad_sym

    def _detect_symmetry_operation(self, examples):
        """Detect which symmetry operation to apply."""
        operations = {
            "mirror_horizontal": lambda x: self._mirror_horizontal(x),
            "mirror_vertical": lambda x: self._mirror_vertical(x),
            "mirror_diagonal": lambda x: self._mirror_diagonal(x),
            "make_symmetric_h": lambda x: self._make_symmetric_h(x),
            "make_symmetric_v": lambda x: self._make_symmetric_v(x),
        }

        for name, op in operations.items():
            if all(np.array_equal(op(inp), out) for inp, out in examples):
                return (name, op)

        return None

    def _mirror_horizontal(self, grid):
        """Mirror grid horizontally."""
        h, w = grid.shape
        if w % 2 == 0:
            left = grid[:, : w // 2]
            return np.hstack([left, np.flip(left, axis=1)])
        return grid

    def _mirror_vertical(self, grid):
        """Mirror grid vertically."""
        h, w = grid.shape
        if h % 2 == 0:
            top = grid[: h // 2, :]
            return np.vstack([top, np.flip(top, axis=0)])
        return grid

    def _mirror_diagonal(self, grid):
        """Mirror grid diagonally."""
        if grid.shape[0] == grid.shape[1]:
            return (grid + grid.T) // 2  # Average of grid and transpose
        return grid

    def _make_symmetric_h(self, grid):
        """Make grid horizontally symmetric."""
        return np.hstack([grid, np.flip(grid, axis=1)])

    def _make_symmetric_v(self, grid):
        """Make grid vertically symmetric."""
        return np.vstack([grid, np.flip(grid, axis=0)])

    def apply(self, examples, test_input):
        """Apply symmetry operation."""
        result = self._detect_symmetry_operation(examples)
        if not result:
            # Try to make it symmetric as fallback
            output = self._make_symmetric_h(test_input)
            return TransformResult(output, 0.2, "symmetry_fallback")

        name, op = result
        output = op(test_input)
        confidence = self.validate_on_examples(examples, op)

        return TransformResult(output, confidence, f"symmetry_{name}")


# ============================================================================
# PATTERN COMPLETION
# ============================================================================


class PatternCompleter(PatternPrimitive):
    """Completes partial patterns."""

    def can_apply(self, examples, test_input):
        """Check if pattern completion is needed."""
        # Look for partial patterns (zeros or specific background)
        zero_ratio = np.mean(test_input == 0)
        return 0.3 < zero_ratio < 0.8  # Partially filled

    def _find_pattern(self, grid):
        """Find repeating pattern in grid."""
        h, w = grid.shape

        # Look for smallest repeating unit
        for ph in range(1, h // 2 + 1):
            for pw in range(1, w // 2 + 1):
                pattern = grid[:ph, :pw]

                # Check if this pattern tiles the whole grid
                if self._check_tiling(grid, pattern):
                    return pattern

        return None

    def _check_tiling(self, grid, pattern):
        """Check if pattern tiles the grid."""
        ph, pw = pattern.shape
        h, w = grid.shape

        for i in range(0, h, ph):
            for j in range(0, w, pw):
                # Get the tile region
                tile_h = min(ph, h - i)
                tile_w = min(pw, w - j)

                tile = grid[i : i + tile_h, j : j + tile_w]
                expected = pattern[:tile_h, :tile_w]

                # Allow for partial matches (incomplete patterns)
                if not np.array_equal(tile, expected):
                    # Check if tile is just zeros (to be filled)
                    if not np.all(tile == 0):
                        return False

        return True

    def apply(self, examples, test_input):
        """Complete the pattern."""
        # Find existing pattern
        pattern = self._find_pattern(test_input)

        if pattern is None:
            # Try to learn from examples
            if examples:
                inp, out = examples[0]
                pattern = self._find_pattern(out)

        if pattern is None:
            return TransformResult(test_input, 0.0, "pattern_completion_failed")

        # Complete the pattern
        output = self._complete_with_pattern(test_input, pattern)

        return TransformResult(output, 0.5, "pattern_completion")

    def _complete_with_pattern(self, grid, pattern):
        """Complete grid with pattern."""
        output = grid.copy()
        ph, pw = pattern.shape
        h, w = grid.shape

        for i in range(0, h, ph):
            for j in range(0, w, pw):
                tile_h = min(ph, h - i)
                tile_w = min(pw, w - j)

                # Only fill zeros
                mask = output[i : i + tile_h, j : j + tile_w] == 0
                output[i : i + tile_h, j : j + tile_w][mask] = pattern[
                    :tile_h, :tile_w
                ][mask]

        return output


# ============================================================================
# COUNTING AND LOGIC
# ============================================================================


class CountingPrimitive(PatternPrimitive):
    """Handles counting-based transformations."""

    def can_apply(self, examples, test_input):
        """Check if counting logic applies."""
        # Check if output size relates to object count
        for inp, out in examples:
            objects = self._count_objects(inp)
            if objects > 1:
                return True
        return False

    def _count_objects(self, grid, background=0):
        """Count distinct objects."""
        from scipy import ndimage

        non_bg = grid != background
        labeled, num_features = ndimage.label(non_bg)
        return num_features

    def apply(self, examples, test_input):
        """Apply counting-based transformation."""
        count = self._count_objects(test_input)

        # Simple example: create output with count visualization
        # This would be extended based on specific patterns
        output = np.full_like(test_input, count % 10)

        return TransformResult(output, 0.2, f"counting_{count}_objects")


class ConditionalTransform(PatternPrimitive):
    """Applies conditional (if-then-else) transformations."""

    def can_apply(self, examples, test_input):
        """Check if conditional logic applies."""
        # This is complex - would need pattern analysis
        return False  # Placeholder

    def apply(self, examples, test_input):
        """Apply conditional transformation."""
        # This would implement specific conditional rules
        # learned from examples
        return TransformResult(test_input, 0.0, "conditional_not_implemented")


# ============================================================================
# MAIN PATTERN LIBRARY
# ============================================================================


class ComprehensivePatternLibrary:
    """Main library containing all pattern primitives."""

    def __init__(self):
        """Initialize with all primitives."""
        self.primitives = [
            # Geometric (highest priority for non-size-change tasks)
            RotationReflection(),
            # Color transformations
            ColorMapper(),
            # Object-based
            ObjectExtractor(),
            # Symmetry
            SymmetryApplier(),
            # Pattern completion
            PatternCompleter(),
            # Counting/Logic
            CountingPrimitive(),
            ConditionalTransform(),
        ]

        # Import tiling primitives from our enhanced library
        try:
            from enhanced_tiling_primitives import (
                AlternatingRowTile,
                CheckerboardTile,
                SmartTilePattern,
            )

            self.tiling_primitives = [
                SmartTilePattern(),
                AlternatingRowTile(),
                CheckerboardTile(),
            ]
        except ImportError:
            self.tiling_primitives = []

    def find_applicable_primitives(self, examples, test_input):
        """Find all primitives that can be applied."""
        applicable = []

        # Check if size change - use tiling
        if examples and examples[0][0].shape != examples[0][1].shape:
            applicable.extend(self.tiling_primitives)
        else:
            # No size change - try other primitives
            for primitive in self.primitives:
                if primitive.can_apply(examples, test_input):
                    applicable.append(primitive)

        return applicable

    def apply_best_primitive(self, examples, test_input):
        """Apply the best matching primitive."""
        applicable = self.find_applicable_primitives(examples, test_input)

        if not applicable:
            return TransformResult(test_input, 0.0, "no_primitive_applicable")

        # Try each and pick best
        best_result = None
        best_confidence = 0.0

        for primitive in applicable:
            try:
                result = primitive.apply(examples, test_input)
                if result.confidence > best_confidence:
                    best_result = result
                    best_confidence = result.confidence
            except Exception:
                continue

        return (
            best_result
            if best_result
            else TransformResult(test_input, 0.0, "all_primitives_failed")
        )


if __name__ == "__main__":
    print("Comprehensive Pattern Library")
    print("=" * 60)

    library = ComprehensivePatternLibrary()

    print(f"Loaded {len(library.primitives)} general primitives:")
    for p in library.primitives:
        print(f"  - {p.__class__.__name__}")

    print(f"\nLoaded {len(library.tiling_primitives)} tiling primitives:")
    for p in library.tiling_primitives:
        print(f"  - {p.__class__.__name__}")

    print("\nLibrary ready for integration into solver!")
