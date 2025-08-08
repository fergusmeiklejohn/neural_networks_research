#!/usr/bin/env python3
"""Enhanced perception module for ARC-AGI with pattern detection capabilities.

Based on failure analysis, this module adds detection for:
- Arithmetic patterns (counting, color shifting)
- Conditional logic (size-based, color-based, shape-based)
- Spatial patterns (diagonal, spiral, border, repeating)
- Structural changes (merging, splitting, rearrangement)
"""

from utils.imports import setup_project_paths

setup_project_paths()

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage


@dataclass
class Pattern:
    """Represents a detected pattern in the grid."""

    type: str  # arithmetic, conditional, spatial, structural
    name: str  # specific pattern name
    confidence: float
    details: Dict[str, Any]


class EnhancedPerceptionV2:
    """Advanced perception for ARC-AGI tasks with pattern detection."""

    def __init__(self):
        self.patterns_detected = []

    def analyze(
        self, train_examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Dict[str, Any]:
        """Comprehensive analysis of training examples."""
        analysis = {
            "arithmetic_patterns": [],
            "conditional_patterns": [],
            "spatial_patterns": [],
            "structural_patterns": [],
            "transformation_type": None,
            "confidence": 0.0,
        }

        # Analyze each example
        for input_grid, output_grid in train_examples:
            # Arithmetic detection
            arithmetic = self.detect_arithmetic_patterns(input_grid, output_grid)
            if arithmetic:
                analysis["arithmetic_patterns"].extend(arithmetic)

            # Conditional detection
            conditional = self.detect_conditional_patterns(input_grid, output_grid)
            if conditional:
                analysis["conditional_patterns"].extend(conditional)

            # Spatial detection
            spatial = self.detect_spatial_patterns(input_grid, output_grid)
            if spatial:
                analysis["spatial_patterns"].extend(spatial)

            # Structural detection
            structural = self.detect_structural_patterns(input_grid, output_grid)
            if structural:
                analysis["structural_patterns"].extend(structural)

        # Determine primary transformation type
        analysis["transformation_type"] = self.determine_transformation_type(analysis)
        analysis["confidence"] = self.calculate_confidence(analysis)

        return analysis

    def detect_arithmetic_patterns(
        self, input_grid: np.ndarray, output_grid: np.ndarray
    ) -> List[Pattern]:
        """Detect arithmetic and counting patterns."""
        patterns = []

        # Color value shifting
        shift = self.detect_color_shift(input_grid, output_grid)
        if shift is not None:
            patterns.append(
                Pattern(
                    type="arithmetic",
                    name="color_shift",
                    confidence=0.9,
                    details={"shift_value": shift},
                )
            )

        # Object counting
        count_encoding = self.detect_count_encoding(input_grid, output_grid)
        if count_encoding:
            patterns.append(
                Pattern(
                    type="arithmetic",
                    name="count_encoding",
                    confidence=0.85,
                    details=count_encoding,
                )
            )

        # Incremental patterns
        incremental = self.detect_incremental_pattern(input_grid, output_grid)
        if incremental:
            patterns.append(
                Pattern(
                    type="arithmetic",
                    name="incremental",
                    confidence=0.8,
                    details=incremental,
                )
            )

        return patterns

    def detect_color_shift(
        self, input_grid: np.ndarray, output_grid: np.ndarray
    ) -> Optional[int]:
        """Detect if colors are shifted by constant value."""
        if input_grid.shape != output_grid.shape:
            return None

        input_colors = set(input_grid.flatten()) - {0}
        output_colors = set(output_grid.flatten()) - {0}

        if len(input_colors) != len(output_colors):
            return None

        # Check if there's a consistent shift
        shifts = []
        for in_c in input_colors:
            mask = input_grid == in_c
            out_values = output_grid[mask]
            if len(set(out_values)) == 1:
                shifts.append(out_values[0] - in_c)

        if shifts and all(s == shifts[0] for s in shifts):
            return int(shifts[0])

        return None

    def detect_count_encoding(
        self, input_grid: np.ndarray, output_grid: np.ndarray
    ) -> Optional[Dict]:
        """Detect if output encodes counts of input objects."""
        input_objects = self.extract_objects(input_grid)

        # Check if output colors correspond to counts
        for color in set(output_grid.flatten()) - {0}:
            count = np.sum(output_grid == color)
            # Check if color value matches some count
            if color == len(input_objects) or color == count:
                return {
                    "type": "object_count"
                    if color == len(input_objects)
                    else "pixel_count",
                    "encoding_value": int(color),
                }

        return None

    def detect_incremental_pattern(
        self, input_grid: np.ndarray, output_grid: np.ndarray
    ) -> Optional[Dict]:
        """Detect incremental/sequential patterns."""
        # Check for sequential numbering
        unique_output = sorted(set(output_grid.flatten()) - {0})
        if len(unique_output) > 2:
            # Check if values form sequence
            diffs = [
                unique_output[i + 1] - unique_output[i]
                for i in range(len(unique_output) - 1)
            ]
            if all(d == 1 for d in diffs):
                return {"sequence": unique_output, "increment": 1}

        return None

    def detect_conditional_patterns(
        self, input_grid: np.ndarray, output_grid: np.ndarray
    ) -> List[Pattern]:
        """Detect conditional logic patterns."""
        patterns = []

        objects = self.extract_objects(input_grid)

        # Size-based conditions
        size_conditions = self.detect_size_conditions(objects, input_grid, output_grid)
        if size_conditions:
            patterns.append(
                Pattern(
                    type="conditional",
                    name="size_based",
                    confidence=0.85,
                    details=size_conditions,
                )
            )

        # Color-based conditions
        color_conditions = self.detect_color_conditions(input_grid, output_grid)
        if color_conditions:
            patterns.append(
                Pattern(
                    type="conditional",
                    name="color_based",
                    confidence=0.9,
                    details=color_conditions,
                )
            )

        # Shape-based conditions
        shape_conditions = self.detect_shape_conditions(
            objects, input_grid, output_grid
        )
        if shape_conditions:
            patterns.append(
                Pattern(
                    type="conditional",
                    name="shape_based",
                    confidence=0.8,
                    details=shape_conditions,
                )
            )

        return patterns

    def detect_size_conditions(
        self, objects: List[Dict], input_grid: np.ndarray, output_grid: np.ndarray
    ) -> Optional[Dict]:
        """Detect size-based conditional transformations."""
        if not objects:
            return None

        # Group objects by size
        size_groups = defaultdict(list)
        for obj in objects:
            size_groups[obj["size"]].append(obj)

        # Check if large objects are treated differently
        sizes = sorted(size_groups.keys())
        if len(sizes) > 1:
            large_threshold = sizes[-1]
            large_objs = size_groups[large_threshold]

            # Check transformation of large objects
            for obj in large_objs:
                bbox = obj["bbox"]
                if bbox[2] < output_grid.shape[0] and bbox[3] < output_grid.shape[1]:
                    input_region = input_grid[
                        bbox[0] : bbox[2] + 1, bbox[1] : bbox[3] + 1
                    ]
                    output_region = output_grid[
                        bbox[0] : bbox[2] + 1, bbox[1] : bbox[3] + 1
                    ]

                    if not np.array_equal(input_region, output_region):
                        # Detect what happened
                        if len(set(output_region.flatten())) == 1:
                            return {
                                "condition": f"size > {large_threshold-1}",
                                "action": "fill",
                                "fill_color": int(output_region[0, 0]),
                            }

        return None

    def detect_color_conditions(
        self, input_grid: np.ndarray, output_grid: np.ndarray
    ) -> Optional[Dict]:
        """Detect color-based conditional transformations."""
        if input_grid.shape != output_grid.shape:
            return None

        conditions = {}
        for color in set(input_grid.flatten()) - {0}:
            mask = input_grid == color
            out_values = output_grid[mask]

            if len(out_values) > 0:
                unique_out = set(out_values)
                if len(unique_out) == 1 and list(unique_out)[0] != color:
                    conditions[int(color)] = int(list(unique_out)[0])

        if conditions:
            return {"color_mappings": conditions}

        return None

    def detect_shape_conditions(
        self, objects: List[Dict], input_grid: np.ndarray, output_grid: np.ndarray
    ) -> Optional[Dict]:
        """Detect shape-based conditional transformations."""
        if not objects:
            return None

        square_transforms = []
        rect_transforms = []

        for obj in objects:
            bbox = obj["bbox"]
            height = bbox[2] - bbox[0] + 1
            width = bbox[3] - bbox[1] + 1

            is_square = height == width

            # Check transformation
            if bbox[2] < input_grid.shape[0] and bbox[3] < input_grid.shape[1]:
                if bbox[2] < output_grid.shape[0] and bbox[3] < output_grid.shape[1]:
                    input_region = input_grid[
                        bbox[0] : bbox[2] + 1, bbox[1] : bbox[3] + 1
                    ]
                    output_region = output_grid[
                        bbox[0] : bbox[2] + 1, bbox[1] : bbox[3] + 1
                    ]

                    if not np.array_equal(input_region, output_region):
                        transform = self.describe_transform(input_region, output_region)
                        if is_square:
                            square_transforms.append(transform)
                        else:
                            rect_transforms.append(transform)

        if square_transforms:
            return {
                "square_objects": square_transforms[0] if square_transforms else None,
                "rectangle_objects": rect_transforms[0] if rect_transforms else None,
            }

        return None

    def detect_spatial_patterns(
        self, input_grid: np.ndarray, output_grid: np.ndarray
    ) -> List[Pattern]:
        """Detect spatial transformation patterns."""
        patterns = []

        # Diagonal patterns
        if self.has_diagonal_pattern(output_grid):
            patterns.append(
                Pattern(
                    type="spatial",
                    name="diagonal",
                    confidence=0.9,
                    details={
                        "type": "main" if self.is_main_diagonal(output_grid) else "anti"
                    },
                )
            )

        # Spiral patterns
        if self.has_spiral_pattern(output_grid):
            patterns.append(
                Pattern(
                    type="spatial",
                    name="spiral",
                    confidence=0.75,
                    details={"clockwise": True},  # TODO: detect direction
                )
            )

        # Border patterns
        if self.has_border_pattern(output_grid):
            patterns.append(
                Pattern(
                    type="spatial",
                    name="border",
                    confidence=0.85,
                    details=self.analyze_border(output_grid),
                )
            )

        # Repeating patterns
        repeat_info = self.detect_repeating_pattern(output_grid)
        if repeat_info:
            patterns.append(
                Pattern(
                    type="spatial",
                    name="repeating",
                    confidence=0.8,
                    details=repeat_info,
                )
            )

        # Symmetry
        symmetry = self.detect_symmetry(output_grid)
        if symmetry:
            patterns.append(
                Pattern(
                    type="spatial",
                    name="symmetric",
                    confidence=0.9,
                    details={"symmetry_type": symmetry},
                )
            )

        return patterns

    def has_diagonal_pattern(self, grid: np.ndarray) -> bool:
        """Check for diagonal patterns."""
        h, w = grid.shape

        # Check main diagonal
        if h == w and h > 2:
            diagonal = [grid[i, i] for i in range(min(h, w))]
            if len(set(diagonal)) <= 2 and any(d != 0 for d in diagonal):
                return True

        # Check anti-diagonal
        if h > 2 and w > 2:
            anti_diagonal = [grid[i, w - 1 - i] for i in range(min(h, w))]
            if len(set(anti_diagonal)) <= 2 and any(d != 0 for d in anti_diagonal):
                return True

        return False

    def is_main_diagonal(self, grid: np.ndarray) -> bool:
        """Check if main diagonal has pattern."""
        h, w = grid.shape
        if h != w:
            return False
        diagonal = [grid[i, i] for i in range(min(h, w))]
        return len(set(diagonal)) <= 2 and any(d != 0 for d in diagonal)

    def has_spiral_pattern(self, grid: np.ndarray) -> bool:
        """Detect spiral patterns in grid."""
        h, w = grid.shape
        if h < 3 or w < 3:
            return False

        # Extract spiral path
        spiral = self.extract_spiral(grid)

        # Check for pattern in spiral
        if len(spiral) > 4:
            # Check for sequential values
            unique_vals = list(dict.fromkeys(spiral))  # Preserve order
            if len(unique_vals) > 1 and len(unique_vals) < len(spiral) / 2:
                return True

        return False

    def extract_spiral(self, grid: np.ndarray) -> List[int]:
        """Extract values in spiral order."""
        h, w = grid.shape
        spiral = []

        top, bottom = 0, h - 1
        left, right = 0, w - 1

        while top <= bottom and left <= right:
            # Top row
            for i in range(left, right + 1):
                spiral.append(int(grid[top, i]))
            top += 1

            # Right column
            for i in range(top, bottom + 1):
                spiral.append(int(grid[i, right]))
            right -= 1

            # Bottom row
            if top <= bottom:
                for i in range(right, left - 1, -1):
                    spiral.append(int(grid[bottom, i]))
                bottom -= 1

            # Left column
            if left <= right:
                for i in range(bottom, top - 1, -1):
                    spiral.append(int(grid[i, left]))
                left += 1

        return spiral

    def has_border_pattern(self, grid: np.ndarray) -> bool:
        """Check for border patterns."""
        h, w = grid.shape
        if h < 3 or w < 3:
            return False

        # Extract border
        border = set()
        border.update(grid[0, :])  # Top
        border.update(grid[-1, :])  # Bottom
        border.update(grid[:, 0])  # Left
        border.update(grid[:, -1])  # Right

        # Extract interior
        interior = set(grid[1:-1, 1:-1].flatten())

        # Border should be distinct from interior
        if border and interior:
            # Check if border has consistent non-zero values
            border_nonzero = border - {0}
            if border_nonzero and not border_nonzero.intersection(interior):
                return True

        return False

    def analyze_border(self, grid: np.ndarray) -> Dict:
        """Analyze border pattern details."""
        h, w = grid.shape

        # Get border values
        top = list(grid[0, :])
        bottom = list(grid[-1, :])
        left = list(grid[:, 0])
        right = list(grid[:, -1])

        # Check thickness
        thickness = 1
        if h > 2 and w > 2:
            # Check if 2-pixel border
            inner_border = set()
            if h > 3 and w > 3:
                inner_border.update(grid[1, 1:-1])
                inner_border.update(grid[-2, 1:-1])
                inner_border.update(grid[1:-1, 1])
                inner_border.update(grid[1:-1, -2])

                if inner_border == set(top):
                    thickness = 2

        return {
            "thickness": thickness,
            "color": int(
                max(
                    set(top + bottom + left + right),
                    key=lambda x: (top + bottom + left + right).count(x),
                )
            ),
        }

    def detect_repeating_pattern(self, grid: np.ndarray) -> Optional[Dict]:
        """Detect repeating patterns in grid."""
        h, w = grid.shape

        # Check 2x2 repeating
        if h >= 4 and w >= 4:
            pattern = grid[:2, :2]
            is_repeating = True

            for i in range(0, h - 1, 2):
                for j in range(0, w - 1, 2):
                    sub_h = min(2, h - i)
                    sub_w = min(2, w - j)
                    if not np.array_equal(
                        grid[i : i + sub_h, j : j + sub_w], pattern[:sub_h, :sub_w]
                    ):
                        is_repeating = False
                        break
                if not is_repeating:
                    break

            if is_repeating:
                return {"pattern_size": (2, 2), "pattern": pattern.tolist()}

        # Check row repeating
        if h >= 2:
            for i in range(h - 1):
                if np.array_equal(grid[i], grid[i + 1]):
                    return {"pattern_size": (1, w), "direction": "horizontal"}

        # Check column repeating
        if w >= 2:
            for j in range(w - 1):
                if np.array_equal(grid[:, j], grid[:, j + 1]):
                    return {"pattern_size": (h, 1), "direction": "vertical"}

        return None

    def detect_symmetry(self, grid: np.ndarray) -> Optional[str]:
        """Detect symmetry in grid."""
        # Horizontal symmetry
        if np.array_equal(grid, np.flipud(grid)):
            return "horizontal"

        # Vertical symmetry
        if np.array_equal(grid, np.fliplr(grid)):
            return "vertical"

        # Rotational symmetry (180 degrees)
        if np.array_equal(grid, np.rot90(grid, 2)):
            return "rotational_180"

        # Rotational symmetry (90 degrees)
        if grid.shape[0] == grid.shape[1]:
            if np.array_equal(grid, np.rot90(grid)):
                return "rotational_90"

        return None

    def detect_structural_patterns(
        self, input_grid: np.ndarray, output_grid: np.ndarray
    ) -> List[Pattern]:
        """Detect structural changes between input and output."""
        patterns = []

        input_objects = self.extract_objects(input_grid)
        output_objects = self.extract_objects(output_grid)

        # Object count changes
        if len(output_objects) > len(input_objects):
            patterns.append(
                Pattern(
                    type="structural",
                    name="objects_split",
                    confidence=0.8,
                    details={
                        "input_count": len(input_objects),
                        "output_count": len(output_objects),
                    },
                )
            )
        elif len(output_objects) < len(input_objects):
            patterns.append(
                Pattern(
                    type="structural",
                    name="objects_merged",
                    confidence=0.8,
                    details={
                        "input_count": len(input_objects),
                        "output_count": len(output_objects),
                    },
                )
            )

        # Object rearrangement
        if len(input_objects) == len(output_objects) and len(input_objects) > 1:
            rearranged = self.detect_rearrangement(input_objects, output_objects)
            if rearranged:
                patterns.append(
                    Pattern(
                        type="structural",
                        name="objects_rearranged",
                        confidence=0.85,
                        details=rearranged,
                    )
                )

        return patterns

    def detect_rearrangement(
        self, input_objects: List[Dict], output_objects: List[Dict]
    ) -> Optional[Dict]:
        """Detect if objects were rearranged."""
        # Compare positions
        input_positions = [(obj["bbox"][0], obj["bbox"][1]) for obj in input_objects]
        output_positions = [(obj["bbox"][0], obj["bbox"][1]) for obj in output_objects]

        if set(input_positions) != set(output_positions):
            # Detect pattern of rearrangement
            # TODO: Detect specific patterns (sorted, reversed, etc.)
            return {"type": "position_change"}

        return None

    def extract_objects(self, grid: np.ndarray) -> List[Dict]:
        """Extract objects from grid."""
        objects = []
        colors = set(grid.flatten()) - {0}

        for color in colors:
            mask = grid == color
            labeled, num = ndimage.label(mask)

            for i in range(1, num + 1):
                component = labeled == i
                pixels = np.argwhere(component)

                if len(pixels) > 0:
                    min_r, min_c = pixels.min(axis=0)
                    max_r, max_c = pixels.max(axis=0)

                    objects.append(
                        {
                            "color": int(color),
                            "pixels": pixels.tolist(),
                            "bbox": (min_r, min_c, max_r, max_c),
                            "size": len(pixels),
                        }
                    )

        return objects

    def describe_transform(
        self, input_region: np.ndarray, output_region: np.ndarray
    ) -> str:
        """Describe transformation between regions."""
        # Check for fill
        if len(set(output_region.flatten())) == 1:
            return f"fill_with_{output_region[0, 0]}"

        # Check for rotation
        for k in [1, 2, 3]:
            if np.array_equal(output_region, np.rot90(input_region, k)):
                return f"rotate_{k*90}"

        # Check for flip
        if np.array_equal(output_region, np.flipud(input_region)):
            return "flip_horizontal"
        if np.array_equal(output_region, np.fliplr(input_region)):
            return "flip_vertical"

        return "complex_transform"

    def determine_transformation_type(self, analysis: Dict) -> str:
        """Determine primary transformation type from analysis."""
        scores = {
            "arithmetic": len(analysis["arithmetic_patterns"]),
            "conditional": len(analysis["conditional_patterns"]),
            "spatial": len(analysis["spatial_patterns"]),
            "structural": len(analysis["structural_patterns"]),
        }

        if max(scores.values()) == 0:
            return "unknown"

        return max(scores, key=scores.get)

    def calculate_confidence(self, analysis: Dict) -> float:
        """Calculate overall confidence in analysis."""
        all_patterns = (
            analysis["arithmetic_patterns"]
            + analysis["conditional_patterns"]
            + analysis["spatial_patterns"]
            + analysis["structural_patterns"]
        )

        if not all_patterns:
            return 0.0

        # Average confidence of detected patterns
        confidences = [p.confidence for p in all_patterns]
        return sum(confidences) / len(confidences)


def test_enhanced_perception():
    """Test the enhanced perception module."""
    print("Testing Enhanced Perception V2")
    print("=" * 60)

    perception = EnhancedPerceptionV2()

    # Test 1: Color shift detection
    print("\nTest 1: Color Shift")
    input1 = np.array([[1, 2, 3], [2, 3, 1], [3, 1, 2]])
    output1 = np.array([[3, 4, 5], [4, 5, 3], [5, 3, 4]])

    analysis = perception.analyze([(input1, output1)])
    print(f"Detected patterns: {analysis['transformation_type']}")
    if analysis["arithmetic_patterns"]:
        print(f"Arithmetic: {analysis['arithmetic_patterns'][0].details}")

    # Test 2: Diagonal pattern
    print("\nTest 2: Diagonal Pattern")
    output2 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    analysis = perception.analyze([(np.zeros((3, 3)), output2)])
    print(f"Detected patterns: {analysis['transformation_type']}")
    if analysis["spatial_patterns"]:
        for p in analysis["spatial_patterns"]:
            print(f"Spatial: {p.name} - {p.details}")

    # Test 3: Border pattern
    print("\nTest 3: Border Pattern")
    output3 = np.array([[1, 1, 1, 1], [1, 0, 0, 1], [1, 0, 0, 1], [1, 1, 1, 1]])

    analysis = perception.analyze([(np.zeros((4, 4)), output3)])
    print(f"Detected patterns: {analysis['transformation_type']}")
    if analysis["spatial_patterns"]:
        for p in analysis["spatial_patterns"]:
            print(f"Spatial: {p.name} - {p.details}")

    print("\n" + "=" * 60)
    print("Enhanced perception tests complete!")


if __name__ == "__main__":
    test_enhanced_perception()
