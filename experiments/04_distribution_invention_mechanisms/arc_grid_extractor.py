#!/usr/bin/env python3
"""ARC Grid Transformation Rule Extractor.

Applies our explicit extraction approach to ARC-AGI tasks, demonstrating that
distribution invention works for visual reasoning tasks.

Key insight: ARC tasks are about extracting transformation rules from examples,
just like extracting "X means jump" or "gravity = 25".
"""

from utils.imports import setup_project_paths

setup_project_paths()

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class GridTransformation:
    """Represents an extracted transformation rule."""

    rule_type: str  # "color_change", "rotation", "translation", "pattern_fill", etc.
    parameters: Dict[str, Any]  # Rule-specific parameters
    condition: Optional[str] = None  # When to apply the rule
    scope: str = "global"  # global, local, object-specific


@dataclass
class ARCRule:
    """Complete rule set for an ARC task."""

    transformations: List[GridTransformation]
    object_detection: Optional[str] = None  # How to identify objects
    composition_order: Optional[List[int]] = None  # Order to apply transformations


class ARCGridExtractor:
    """Extracts explicit transformation rules from ARC examples.

    Instead of learning pixel patterns, we extract discrete rules like:
    - "Rotate all red objects 90 degrees"
    - "Fill holes with the most common color"
    - "Mirror across vertical axis"
    """

    def __init__(self):
        # Common ARC transformation patterns
        self.transformation_types = {
            "color_mapping": self._detect_color_mapping,
            "rotation": self._detect_rotation,
            "translation": self._detect_translation,
            "scaling": self._detect_scaling,
            "symmetry": self._detect_symmetry,
            "pattern_fill": self._detect_pattern_fill,
            "object_manipulation": self._detect_object_manipulation,
            "counting": self._detect_counting,
            "sorting": self._detect_sorting,
        }

        # Color palette (ARC uses 0-9)
        self.colors = {
            0: "black",
            1: "blue",
            2: "red",
            3: "green",
            4: "yellow",
            5: "gray",
            6: "magenta",
            7: "orange",
            8: "light_blue",
            9: "brown",
        }

    def extract_rules(self, examples: List[Tuple[np.ndarray, np.ndarray]]) -> ARCRule:
        """Extract transformation rules from input-output examples.

        Args:
            examples: List of (input_grid, output_grid) pairs

        Returns:
            ARCRule containing extracted transformations
        """
        transformations = []

        # Analyze each transformation type
        for trans_type, detector in self.transformation_types.items():
            rule = detector(examples)
            if rule:
                transformations.append(rule)

        # Detect object-based rules
        object_detection = self._detect_object_structure(examples)

        # Determine composition order if multiple transformations
        composition_order = self._infer_composition_order(transformations)

        return ARCRule(
            transformations=transformations,
            object_detection=object_detection,
            composition_order=composition_order,
        )

    def _detect_color_mapping(
        self, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[GridTransformation]:
        """Detect color substitution rules."""
        color_maps = []

        for input_grid, output_grid in examples:
            if input_grid.shape != output_grid.shape:
                continue

            # Build color mapping
            mapping = {}
            for i in range(input_grid.shape[0]):
                for j in range(input_grid.shape[1]):
                    in_color = int(input_grid[i, j])
                    out_color = int(output_grid[i, j])

                    if in_color not in mapping:
                        mapping[in_color] = out_color
                    elif mapping[in_color] != out_color:
                        # Inconsistent mapping
                        mapping = None
                        break

                if mapping is None:
                    break

            if mapping and mapping != {c: c for c in mapping}:
                color_maps.append(mapping)

        # Check if consistent across examples
        if color_maps and all(m == color_maps[0] for m in color_maps):
            return GridTransformation(
                rule_type="color_mapping",
                parameters={"mapping": color_maps[0]},
                scope="global",
            )

        return None

    def _detect_rotation(
        self, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[GridTransformation]:
        """Detect rotation transformations."""
        for input_grid, output_grid in examples:
            # Check 90-degree rotations
            for k in [1, 2, 3]:  # 90, 180, 270 degrees
                rotated = np.rot90(input_grid, k)
                if np.array_equal(rotated, output_grid):
                    return GridTransformation(
                        rule_type="rotation",
                        parameters={"degrees": k * 90},
                        scope="global",
                    )
        return None

    def _detect_translation(
        self, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[GridTransformation]:
        """Detect translation/shift transformations."""
        for input_grid, output_grid in examples:
            if input_grid.shape != output_grid.shape:
                continue

            # Try different shifts
            for dx in range(-5, 6):
                for dy in range(-5, 6):
                    if dx == 0 and dy == 0:
                        continue

                    shifted = np.roll(input_grid, (dx, dy), axis=(0, 1))
                    if np.array_equal(shifted, output_grid):
                        return GridTransformation(
                            rule_type="translation",
                            parameters={"dx": dx, "dy": dy},
                            scope="global",
                        )
        return None

    def _detect_scaling(
        self, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[GridTransformation]:
        """Detect scaling transformations."""
        for input_grid, output_grid in examples:
            h_in, w_in = input_grid.shape
            h_out, w_out = output_grid.shape

            # Check for integer scaling
            if h_out % h_in == 0 and w_out % w_in == 0:
                scale_h = h_out // h_in
                scale_w = w_out // w_in

                if scale_h == scale_w:
                    # Verify it's actually scaling
                    scaled = np.repeat(
                        np.repeat(input_grid, scale_h, axis=0), scale_w, axis=1
                    )
                    if np.array_equal(scaled, output_grid):
                        return GridTransformation(
                            rule_type="scaling",
                            parameters={"factor": scale_h},
                            scope="global",
                        )
        return None

    def _detect_symmetry(
        self, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[GridTransformation]:
        """Detect symmetry operations (flip, mirror)."""
        for input_grid, output_grid in examples:
            # Check horizontal flip
            if np.array_equal(np.flip(input_grid, axis=1), output_grid):
                return GridTransformation(
                    rule_type="symmetry",
                    parameters={"operation": "horizontal_flip"},
                    scope="global",
                )

            # Check vertical flip
            if np.array_equal(np.flip(input_grid, axis=0), output_grid):
                return GridTransformation(
                    rule_type="symmetry",
                    parameters={"operation": "vertical_flip"},
                    scope="global",
                )

            # Check if output is symmetric completion of input
            if self._is_symmetric_completion(input_grid, output_grid):
                return GridTransformation(
                    rule_type="symmetry",
                    parameters={"operation": "complete_symmetry"},
                    scope="global",
                )

        return None

    def _detect_pattern_fill(
        self, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[GridTransformation]:
        """Detect pattern filling rules."""
        for input_grid, output_grid in examples:
            # Check if filling holes (0s) with patterns
            if 0 in input_grid:
                # Find what fills the zeros
                fill_mask = input_grid == 0
                filled_values = output_grid[fill_mask]

                if len(set(filled_values)) == 1:
                    # Single color fill
                    return GridTransformation(
                        rule_type="pattern_fill",
                        parameters={"pattern": "solid", "color": int(filled_values[0])},
                        condition="where_zero",
                    )
                elif self._is_checkerboard(filled_values):
                    return GridTransformation(
                        rule_type="pattern_fill",
                        parameters={"pattern": "checkerboard"},
                        condition="where_zero",
                    )

        return None

    def _detect_object_manipulation(
        self, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[GridTransformation]:
        """Detect object-based transformations."""
        # This would detect operations on connected components
        # For now, simplified version

        for input_grid, output_grid in examples:
            # Check if objects are moved, rotated, or modified
            input_objects = self._find_objects(input_grid)
            output_objects = self._find_objects(output_grid)

            if len(input_objects) == len(output_objects):
                # Check if objects are transformed consistently
                transformation = self._compare_objects(input_objects, output_objects)
                if transformation:
                    return GridTransformation(
                        rule_type="object_manipulation",
                        parameters=transformation,
                        scope="per_object",
                    )

        return None

    def _detect_counting(
        self, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[GridTransformation]:
        """Detect counting-based rules."""
        # Simplified: check if output encodes count of something
        for input_grid, output_grid in examples:
            # Count unique colors in input
            unique_colors = len(np.unique(input_grid))

            # Check if output represents this count somehow
            if output_grid.size == 1 and output_grid[0, 0] == unique_colors:
                return GridTransformation(
                    rule_type="counting",
                    parameters={"count_what": "unique_colors"},
                    scope="global",
                )

        return None

    def _detect_sorting(
        self, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[GridTransformation]:
        """Detect sorting operations."""
        # Check if output is sorted version of input
        for input_grid, output_grid in examples:
            input_flat = input_grid.flatten()
            output_flat = output_grid.flatten()

            if np.array_equal(np.sort(input_flat), output_flat):
                return GridTransformation(
                    rule_type="sorting",
                    parameters={"order": "ascending"},
                    scope="global",
                )

        return None

    def _detect_object_structure(
        self, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[str]:
        """Detect how objects are defined in the task."""
        # Simplified version - would use connected components analysis
        return "connected_components"

    def _infer_composition_order(
        self, transformations: List[GridTransformation]
    ) -> Optional[List[int]]:
        """Infer the order to apply transformations."""
        if len(transformations) <= 1:
            return None

        # Default order based on typical ARC patterns
        priority = {
            "object_manipulation": 1,
            "color_mapping": 2,
            "rotation": 3,
            "translation": 4,
            "scaling": 5,
            "symmetry": 6,
            "pattern_fill": 7,
        }

        indices = list(range(len(transformations)))
        indices.sort(key=lambda i: priority.get(transformations[i].rule_type, 99))
        return indices

    def _find_objects(self, grid: np.ndarray) -> List[np.ndarray]:
        """Find connected components in grid."""
        # Simplified - would use scipy.ndimage.label
        objects = []
        # ... implementation ...
        return objects

    def _compare_objects(
        self, objects1: List[np.ndarray], objects2: List[np.ndarray]
    ) -> Optional[Dict]:
        """Compare two sets of objects to find transformation."""
        # Simplified comparison
        return None

    def _is_symmetric_completion(
        self, input_grid: np.ndarray, output_grid: np.ndarray
    ) -> bool:
        """Check if output completes input symmetrically."""
        # Check various symmetry completions
        return False

    def _is_checkerboard(self, values: np.ndarray) -> bool:
        """Check if values form a checkerboard pattern."""
        # Simplified check
        return len(set(values)) == 2

    def apply_rules(self, input_grid: np.ndarray, rules: ARCRule) -> np.ndarray:
        """Apply extracted rules to a new input grid.

        Args:
            input_grid: Grid to transform
            rules: Extracted transformation rules

        Returns:
            Transformed grid
        """
        result = input_grid.copy()

        # Apply transformations in order
        if rules.composition_order:
            ordered_trans = [rules.transformations[i] for i in rules.composition_order]
        else:
            ordered_trans = rules.transformations

        for transformation in ordered_trans:
            result = self._apply_single_transformation(result, transformation)

        return result

    def _apply_single_transformation(
        self, grid: np.ndarray, trans: GridTransformation
    ) -> np.ndarray:
        """Apply a single transformation to a grid."""
        if trans.rule_type == "color_mapping":
            mapping = trans.parameters["mapping"]
            result = grid.copy()
            for old_color, new_color in mapping.items():
                result[grid == old_color] = new_color
            return result

        elif trans.rule_type == "rotation":
            k = trans.parameters["degrees"] // 90
            return np.rot90(grid, k)

        elif trans.rule_type == "translation":
            dx, dy = trans.parameters["dx"], trans.parameters["dy"]
            return np.roll(grid, (dx, dy), axis=(0, 1))

        elif trans.rule_type == "scaling":
            factor = trans.parameters["factor"]
            return np.repeat(np.repeat(grid, factor, axis=0), factor, axis=1)

        elif trans.rule_type == "symmetry":
            op = trans.parameters["operation"]
            if op == "horizontal_flip":
                return np.flip(grid, axis=1)
            elif op == "vertical_flip":
                return np.flip(grid, axis=0)

        elif trans.rule_type == "pattern_fill" and trans.condition == "where_zero":
            result = grid.copy()
            if trans.parameters["pattern"] == "solid":
                result[grid == 0] = trans.parameters["color"]
            return result

        # Default: return unchanged
        return grid


def test_arc_extraction():
    """Test ARC rule extraction with simple examples."""
    extractor = ARCGridExtractor()

    print("=" * 60)
    print("ARC GRID TRANSFORMATION EXTRACTION TESTS")
    print("=" * 60)

    # Test 1: Simple color mapping
    print("\nTest 1: Color Mapping (red → blue)")
    input1 = np.array([[2, 0, 2], [0, 2, 0], [2, 0, 2]])  # 2=red, 0=black
    output1 = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])  # 1=blue

    rules = extractor.extract_rules([(input1, output1)])
    print(f"Extracted rules: {len(rules.transformations)} transformation(s)")
    for trans in rules.transformations:
        print(f"  - {trans.rule_type}: {trans.parameters}")

    # Test 2: Rotation
    print("\nTest 2: 90-degree Rotation")
    input2 = np.array([[1, 2], [3, 4]])
    output2 = np.array([[2, 4], [1, 3]])  # 90 degrees clockwise

    rules = extractor.extract_rules([(input2, output2)])
    print(f"Extracted rules: {len(rules.transformations)} transformation(s)")
    for trans in rules.transformations:
        print(f"  - {trans.rule_type}: {trans.parameters}")

    # Test 3: Scaling
    print("\nTest 3: 2x Scaling")
    input3 = np.array([[1, 2], [3, 4]])
    output3 = np.array([[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]])

    rules = extractor.extract_rules([(input3, output3)])
    print(f"Extracted rules: {len(rules.transformations)} transformation(s)")
    for trans in rules.transformations:
        print(f"  - {trans.rule_type}: {trans.parameters}")

    # Test 4: Pattern fill
    print("\nTest 4: Fill zeros with color")
    input4 = np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]])
    output4 = np.array([[1, 3, 1], [3, 3, 3], [1, 3, 1]])  # Fill 0s with 3 (green)

    rules = extractor.extract_rules([(input4, output4)])
    print(f"Extracted rules: {len(rules.transformations)} transformation(s)")
    for trans in rules.transformations:
        print(f"  - {trans.rule_type}: {trans.parameters}")

    # Test 5: Apply extracted rules to new input
    print("\nTest 5: Apply Extracted Rules")
    new_input = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]])
    print(f"New input:\n{new_input}")

    # Use color mapping rules from Test 1
    rules = extractor.extract_rules([(input1, output1)])
    transformed = extractor.apply_rules(new_input, rules)
    print(f"After applying rules:\n{transformed}")

    print("\n" + "=" * 60)
    print("KEY INSIGHT:")
    print("ARC tasks are just another form of rule extraction -")
    print("like 'X means jump' or 'gravity = 25', but for grids!")
    print("=" * 60)


if __name__ == "__main__":
    test_arc_extraction()
