#!/usr/bin/env python3
"""Enhanced Neural Perception Module with advanced pattern detection.

This module provides sophisticated pattern detection capabilities for ARC tasks,
including object detection, counting, spatial relationships, and transformation sequences.
"""

from utils.imports import setup_project_paths

setup_project_paths()

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage


@dataclass
class Object:
    """Represents a detected object in the grid."""

    color: int
    pixels: List[Tuple[int, int]]
    bounding_box: Tuple[int, int, int, int]  # min_row, min_col, max_row, max_col
    size: int
    center: Tuple[float, float]
    shape_type: str  # rectangle, line, L-shape, etc.


@dataclass
class SpatialRelationship:
    """Represents a spatial relationship between objects."""

    obj1_id: int
    obj2_id: int
    relationship: str  # inside, outside, adjacent, above, below, left, right
    distance: float


@dataclass
class PatternSequence:
    """Represents a detected pattern sequence."""

    pattern_type: str  # arithmetic, geometric, alternating, etc.
    sequence: List[int]
    rule: str  # Description of the pattern rule


class EnhancedNeuralPerception:
    """Enhanced perception module with advanced pattern detection."""

    def __init__(self):
        self.detected_objects = []
        self.spatial_relationships = []
        self.pattern_sequences = []

    def analyze_grid(self, grid: np.ndarray) -> Dict:
        """Comprehensive grid analysis."""
        analysis = {
            "objects": self.detect_objects(grid),
            "counting": self.count_elements(grid),
            "spatial": self.detect_spatial_relationships(grid),
            "patterns": self.detect_pattern_sequences(grid),
            "symmetry": self.detect_symmetries(grid),
            "topology": self.analyze_topology(grid),
            "transformations": self.detect_potential_transformations(grid),
        }
        return analysis

    def detect_objects(self, grid: np.ndarray) -> List[Object]:
        """Detect and characterize objects in the grid."""
        objects = []

        # Get unique non-zero colors
        unique_colors = np.unique(grid)
        unique_colors = unique_colors[unique_colors != 0]

        for color in unique_colors:
            # Create binary mask for this color
            mask = (grid == color).astype(int)

            # Find connected components
            labeled, num_features = ndimage.label(mask)

            for i in range(1, num_features + 1):
                component_mask = labeled == i
                pixels = list(zip(*np.where(component_mask)))

                if pixels:
                    obj = self._create_object(color, pixels, grid.shape)
                    objects.append(obj)

        self.detected_objects = objects
        return objects

    def _create_object(
        self, color: int, pixels: List[Tuple[int, int]], grid_shape: Tuple[int, int]
    ) -> Object:
        """Create an Object from pixels."""
        rows = [p[0] for p in pixels]
        cols = [p[1] for p in pixels]

        min_row, max_row = min(rows), max(rows)
        min_col, max_col = min(cols), max(cols)

        center = (np.mean(rows), np.mean(cols))

        # Classify shape
        shape_type = self._classify_shape(pixels, min_row, max_row, min_col, max_col)

        return Object(
            color=color,
            pixels=pixels,
            bounding_box=(min_row, min_col, max_row, max_col),
            size=len(pixels),
            center=center,
            shape_type=shape_type,
        )

    def _classify_shape(
        self,
        pixels: List[Tuple[int, int]],
        min_row: int,
        max_row: int,
        min_col: int,
        max_col: int,
    ) -> str:
        """Classify the shape of an object."""
        height = max_row - min_row + 1
        width = max_col - min_col + 1
        area = len(pixels)
        expected_rect_area = height * width

        # Check if it's a filled rectangle
        if area == expected_rect_area:
            if height == 1:
                return "horizontal_line"
            elif width == 1:
                return "vertical_line"
            elif height == width:
                return "square"
            else:
                return "rectangle"

        # Check for L-shape
        if area > 2 and area < expected_rect_area * 0.75:
            # Simple L-shape detection
            return "L_shape"

        # Check for diagonal
        if self._is_diagonal(pixels):
            return "diagonal"

        return "irregular"

    def _is_diagonal(self, pixels: List[Tuple[int, int]]) -> bool:
        """Check if pixels form a diagonal line."""
        if len(pixels) < 2:
            return False

        # Sort pixels
        sorted_pixels = sorted(pixels)

        # Check if consecutive pixels differ by (1,1) or (1,-1)
        for i in range(1, len(sorted_pixels)):
            dr = sorted_pixels[i][0] - sorted_pixels[i - 1][0]
            dc = sorted_pixels[i][1] - sorted_pixels[i - 1][1]

            if not (abs(dr) == 1 and abs(dc) == 1):
                return False

        return True

    def count_elements(self, grid: np.ndarray) -> Dict:
        """Count various elements in the grid."""
        counts = {
            "total_objects": len(self.detected_objects),
            "by_color": {},
            "by_size": {},
            "by_shape": {},
        }

        # Count by color
        for obj in self.detected_objects:
            color = str(obj.color)
            counts["by_color"][color] = counts["by_color"].get(color, 0) + 1

            # Count by size buckets
            size_bucket = self._get_size_bucket(obj.size)
            counts["by_size"][size_bucket] = counts["by_size"].get(size_bucket, 0) + 1

            # Count by shape
            counts["by_shape"][obj.shape_type] = (
                counts["by_shape"].get(obj.shape_type, 0) + 1
            )

        return counts

    def _get_size_bucket(self, size: int) -> str:
        """Categorize object size."""
        if size == 1:
            return "single"
        elif size <= 4:
            return "small"
        elif size <= 9:
            return "medium"
        else:
            return "large"

    def detect_spatial_relationships(
        self, grid: np.ndarray
    ) -> List[SpatialRelationship]:
        """Detect spatial relationships between objects."""
        relationships = []

        for i, obj1 in enumerate(self.detected_objects):
            for j, obj2 in enumerate(self.detected_objects):
                if i >= j:
                    continue

                rel = self._analyze_relationship(obj1, obj2)
                if rel:
                    relationships.append(
                        SpatialRelationship(
                            obj1_id=i,
                            obj2_id=j,
                            relationship=rel,
                            distance=self._calculate_distance(obj1.center, obj2.center),
                        )
                    )

        self.spatial_relationships = relationships
        return relationships

    def _analyze_relationship(self, obj1: Object, obj2: Object) -> Optional[str]:
        """Analyze the relationship between two objects."""
        bb1 = obj1.bounding_box
        bb2 = obj2.bounding_box

        # Check for containment
        if self._contains(bb1, bb2):
            return "contains"
        elif self._contains(bb2, bb1):
            return "inside"

        # Check for adjacency
        if self._are_adjacent(obj1.pixels, obj2.pixels):
            return "adjacent"

        # Check relative positions
        if bb1[1] > bb2[3]:  # obj1 is to the right of obj2
            return "right_of"
        elif bb1[3] < bb2[1]:  # obj1 is to the left of obj2
            return "left_of"
        elif bb1[0] > bb2[2]:  # obj1 is below obj2
            return "below"
        elif bb1[2] < bb2[0]:  # obj1 is above obj2
            return "above"

        return "overlapping" if self._overlaps(bb1, bb2) else None

    def _contains(self, bb1: Tuple, bb2: Tuple) -> bool:
        """Check if bb1 contains bb2."""
        return (
            bb1[0] <= bb2[0]
            and bb1[1] <= bb2[1]
            and bb1[2] >= bb2[2]
            and bb1[3] >= bb2[3]
        )

    def _overlaps(self, bb1: Tuple, bb2: Tuple) -> bool:
        """Check if two bounding boxes overlap."""
        return not (
            bb1[2] < bb2[0] or bb2[2] < bb1[0] or bb1[3] < bb2[1] or bb2[3] < bb1[1]
        )

    def _are_adjacent(self, pixels1: List[Tuple], pixels2: List[Tuple]) -> bool:
        """Check if two objects are adjacent (touching)."""
        for p1 in pixels1:
            for p2 in pixels2:
                if abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) == 1:
                    return True
        return False

    def _calculate_distance(self, center1: Tuple, center2: Tuple) -> float:
        """Calculate Euclidean distance between centers."""
        return np.sqrt((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2)

    def detect_pattern_sequences(self, grid: np.ndarray) -> List[PatternSequence]:
        """Detect pattern sequences in the grid."""
        sequences = []

        # Check rows for patterns
        for i, row in enumerate(grid):
            seq = self._detect_sequence_pattern(row)
            if seq:
                sequences.append(
                    PatternSequence(
                        pattern_type=f"row_{i}", sequence=list(row), rule=seq
                    )
                )

        # Check columns for patterns
        for j in range(grid.shape[1]):
            col = grid[:, j]
            seq = self._detect_sequence_pattern(col)
            if seq:
                sequences.append(
                    PatternSequence(
                        pattern_type=f"col_{j}", sequence=list(col), rule=seq
                    )
                )

        # Check diagonals
        if grid.shape[0] == grid.shape[1]:
            main_diag = np.diagonal(grid)
            seq = self._detect_sequence_pattern(main_diag)
            if seq:
                sequences.append(
                    PatternSequence(
                        pattern_type="main_diagonal", sequence=list(main_diag), rule=seq
                    )
                )

        self.pattern_sequences = sequences
        return sequences

    def _detect_sequence_pattern(self, sequence: np.ndarray) -> Optional[str]:
        """Detect pattern in a sequence."""
        if len(sequence) < 2:
            return None

        # Check for alternating pattern
        if self._is_alternating(sequence):
            return "alternating"

        # Check for arithmetic progression
        if self._is_arithmetic_progression(sequence):
            diff = sequence[1] - sequence[0]
            return f"arithmetic_diff_{diff}"

        # Check for repetition
        if self._has_repetition(sequence):
            return "repeating"

        return None

    def _is_alternating(self, seq: np.ndarray) -> bool:
        """Check if sequence alternates between values."""
        if len(seq) < 3:
            return False

        for i in range(2, len(seq)):
            if seq[i] != seq[i - 2]:
                return False

        return seq[0] != seq[1]

    def _is_arithmetic_progression(self, seq: np.ndarray) -> bool:
        """Check if sequence is an arithmetic progression."""
        if len(seq) < 3:
            return False

        diff = seq[1] - seq[0]
        for i in range(2, len(seq)):
            if seq[i] - seq[i - 1] != diff:
                return False

        return True

    def _has_repetition(self, seq: np.ndarray) -> bool:
        """Check if sequence has a repeating pattern."""
        # Try different pattern lengths
        for pattern_len in range(1, len(seq) // 2 + 1):
            if self._check_repetition_pattern(seq, pattern_len):
                return True
        return False

    def _check_repetition_pattern(self, seq: np.ndarray, pattern_len: int) -> bool:
        """Check if sequence repeats with given pattern length."""
        pattern = seq[:pattern_len]

        for i in range(pattern_len, len(seq), pattern_len):
            segment = seq[i : i + pattern_len]
            if len(segment) == pattern_len and not np.array_equal(segment, pattern):
                return False

        return True

    def detect_symmetries(self, grid: np.ndarray) -> Dict:
        """Detect various types of symmetry."""
        return {
            "horizontal": np.array_equal(grid, np.flip(grid, axis=1)),
            "vertical": np.array_equal(grid, np.flip(grid, axis=0)),
            "diagonal": np.array_equal(grid, grid.T)
            if grid.shape[0] == grid.shape[1]
            else False,
            "rotational_90": np.array_equal(grid, np.rot90(grid, 2))
            if grid.shape[0] == grid.shape[1]
            else False,
            "partial_horizontal": self._has_partial_symmetry(grid, axis=1),
            "partial_vertical": self._has_partial_symmetry(grid, axis=0),
        }

    def _has_partial_symmetry(self, grid: np.ndarray, axis: int) -> float:
        """Check for partial symmetry and return the percentage."""
        flipped = np.flip(grid, axis=axis)
        matches = np.sum(grid == flipped)
        total = grid.size
        return matches / total

    def analyze_topology(self, grid: np.ndarray) -> Dict:
        """Analyze topological properties."""
        topology = {
            "holes": self._count_holes(grid),
            "connected_regions": len(self.detected_objects),
            "boundary_objects": self._count_boundary_objects(grid),
            "interior_objects": len(self.detected_objects)
            - self._count_boundary_objects(grid),
        }
        return topology

    def _count_holes(self, grid: np.ndarray) -> int:
        """Count holes (enclosed empty regions) in the grid."""
        # Create inverse mask (0 becomes 1, non-zero becomes 0)
        inverse_mask = (grid == 0).astype(int)

        # Pad with 1s to ensure boundary is connected
        padded = np.pad(inverse_mask, 1, constant_values=1)

        # Find connected components in the inverse
        labeled, num_features = ndimage.label(padded)

        # The first component is the exterior, others are holes
        # Subtract 1 for the exterior and 1 for padding
        return max(0, num_features - 1)

    def _count_boundary_objects(self, grid: np.ndarray) -> int:
        """Count objects touching the boundary."""
        count = 0
        for obj in self.detected_objects:
            for r, c in obj.pixels:
                if r == 0 or r == grid.shape[0] - 1 or c == 0 or c == grid.shape[1] - 1:
                    count += 1
                    break
        return count

    def detect_potential_transformations(self, grid: np.ndarray) -> List[str]:
        """Detect potential transformations that could be applied."""
        transformations = []

        # Check if grid is suitable for various transformations
        if grid.shape[0] == grid.shape[1]:
            transformations.append("rotation")
            transformations.append("transpose")

        transformations.append("flip_horizontal")
        transformations.append("flip_vertical")

        # Check for scaling possibilities
        if all(s <= 15 for s in grid.shape):
            transformations.append("scale_2x")

        # Check for color mapping
        unique_colors = len(np.unique(grid))
        if 2 <= unique_colors <= 5:
            transformations.append("color_mapping")

        # Check for object-based transformations
        if self.detected_objects:
            transformations.append("object_manipulation")
            if len(self.detected_objects) > 1:
                transformations.append("object_rearrangement")

        return transformations

    def compare_grids(self, grid1: np.ndarray, grid2: np.ndarray) -> Dict:
        """Compare two grids to identify transformation."""
        comparison = {
            "same_shape": grid1.shape == grid2.shape,
            "size_change": None,
            "color_changes": {},
            "object_changes": {},
            "likely_transformation": None,
        }

        if grid1.shape != grid2.shape:
            comparison["size_change"] = {
                "from": grid1.shape,
                "to": grid2.shape,
                "ratio": (
                    grid2.shape[0] / grid1.shape[0],
                    grid2.shape[1] / grid1.shape[1],
                ),
            }

            # Check for scaling
            if (
                grid2.shape[0] == grid1.shape[0] * 2
                and grid2.shape[1] == grid1.shape[1] * 2
            ):
                comparison["likely_transformation"] = "2x_scaling"
        else:
            # Same shape - check for other transformations
            if np.array_equal(grid2, np.flip(grid1, axis=1)):
                comparison["likely_transformation"] = "horizontal_flip"
            elif np.array_equal(grid2, np.flip(grid1, axis=0)):
                comparison["likely_transformation"] = "vertical_flip"
            elif grid1.shape[0] == grid1.shape[1] and np.array_equal(
                grid2, np.rot90(grid1)
            ):
                comparison["likely_transformation"] = "rotation_90"
            elif grid1.shape[0] == grid1.shape[1] and np.array_equal(grid2, grid1.T):
                comparison["likely_transformation"] = "transpose"
            else:
                # Check for color mapping
                unique1 = set(grid1.flatten())
                unique2 = set(grid2.flatten())

                if len(unique1) == len(unique2):
                    # Possible color mapping
                    comparison["likely_transformation"] = "color_mapping"

                    # Try to find the mapping
                    color_map = {}
                    for i in range(grid1.shape[0]):
                        for j in range(grid1.shape[1]):
                            c1 = grid1[i, j]
                            c2 = grid2[i, j]
                            if c1 in color_map and color_map[c1] != c2:
                                comparison["likely_transformation"] = "complex_pattern"
                                break
                            color_map[c1] = c2

                    if comparison["likely_transformation"] == "color_mapping":
                        comparison["color_changes"] = color_map

        return comparison


def test_enhanced_perception():
    """Test the enhanced perception module."""
    print("Testing Enhanced Neural Perception Module")
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

    perception = EnhancedNeuralPerception()
    analysis = perception.analyze_grid(test_grid)

    print(f"Test grid:\n{test_grid}\n")

    print(f"Detected {len(analysis['objects'])} objects:")
    for obj in analysis["objects"]:
        print(f"  Color {obj.color}: {obj.shape_type}, size={obj.size}")

    print(f"\nCounting analysis:")
    print(f"  Total objects: {analysis['counting']['total_objects']}")
    print(f"  By color: {analysis['counting']['by_color']}")
    print(f"  By shape: {analysis['counting']['by_shape']}")

    print(f"\nSpatial relationships: {len(analysis['spatial'])} detected")
    for rel in analysis["spatial"]:
        print(f"  Object {rel.obj1_id} is {rel.relationship} object {rel.obj2_id}")

    print(f"\nSymmetry analysis:")
    for sym_type, value in analysis["symmetry"].items():
        if isinstance(value, bool) and value:
            print(f"  Has {sym_type} symmetry")
        elif isinstance(value, float) and value > 0.5:
            print(f"  Partial {sym_type} symmetry: {value:.1%}")

    print(f"\nTopology:")
    print(f"  Holes: {analysis['topology']['holes']}")
    print(f"  Connected regions: {analysis['topology']['connected_regions']}")

    # Test grid comparison
    print("\n" + "=" * 50)
    print("Testing Grid Comparison")

    flipped_grid = np.flip(test_grid, axis=1)
    comparison = perception.compare_grids(test_grid, flipped_grid)

    print(f"Comparing original with horizontally flipped:")
    print(f"  Likely transformation: {comparison['likely_transformation']}")


if __name__ == "__main__":
    test_enhanced_perception()
