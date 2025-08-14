#!/usr/bin/env python3
"""Quick pattern detection for efficient primitive selection.

This module provides fast pattern fingerprinting to identify task characteristics
without full primitive application, enabling efficient primitive prioritization.
"""

from utils.imports import setup_project_paths

setup_project_paths()

import warnings
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from scipy import ndimage

warnings.filterwarnings("ignore")


@dataclass
class PatternFingerprint:
    """Fingerprint of a task's characteristics."""

    # Size properties
    has_size_change: bool
    size_change_type: str  # 'none', 'uniform_scale', 'non_uniform_scale', 'complex'
    scale_factors: Tuple[float, float]

    # Object properties
    object_count_input: int
    object_count_output: int
    object_count_change: str  # 'same', 'increase', 'decrease'
    distinct_colors_input: int
    distinct_colors_output: int

    # Transformation hints
    has_color_change: bool
    has_symmetry: bool
    has_repetition: bool
    has_rotation_hint: bool
    has_movement_hint: bool

    # Pattern complexity
    complexity_score: float  # 0-1, higher means more complex

    # Recommended primitives (in order)
    recommended_primitives: List[str]


class PatternFingerprinter:
    """Fast pattern detection for primitive selection."""

    def __init__(self):
        self.fingerprint_cache = {}

    def fingerprint(
        self, examples: List[Tuple[np.ndarray, np.ndarray]], test_input: np.ndarray
    ) -> PatternFingerprint:
        """Generate fingerprint for task."""

        if not examples:
            return self._default_fingerprint()

        # Analyze first example for quick assessment
        inp, out = examples[0]

        # Size analysis
        has_size_change = inp.shape != out.shape
        size_change_type, scale_factors = self._analyze_size_change(
            inp.shape, out.shape
        )

        # Object analysis
        obj_input = self._count_objects_fast(inp)
        obj_output = self._count_objects_fast(out)
        obj_change = self._categorize_count_change(obj_input, obj_output)

        # Color analysis
        colors_input = len(np.unique(inp))
        colors_output = len(np.unique(out))
        has_color_change = not np.array_equal(np.unique(inp), np.unique(out))

        # Pattern detection
        has_symmetry = self._detect_symmetry(out)
        has_repetition = self._detect_repetition(out)
        has_rotation_hint = self._detect_rotation_hint(inp, out)
        has_movement_hint = self._detect_movement_hint(inp, out)

        # Calculate complexity
        complexity = self._calculate_complexity(
            has_size_change,
            obj_input,
            colors_input,
            has_color_change,
            has_movement_hint,
        )

        # Recommend primitives
        recommended = self._recommend_primitives(
            has_size_change,
            size_change_type,
            obj_input,
            obj_change,
            has_color_change,
            has_symmetry,
            has_repetition,
            has_rotation_hint,
            has_movement_hint,
        )

        return PatternFingerprint(
            has_size_change=has_size_change,
            size_change_type=size_change_type,
            scale_factors=scale_factors,
            object_count_input=obj_input,
            object_count_output=obj_output,
            object_count_change=obj_change,
            distinct_colors_input=colors_input,
            distinct_colors_output=colors_output,
            has_color_change=has_color_change,
            has_symmetry=has_symmetry,
            has_repetition=has_repetition,
            has_rotation_hint=has_rotation_hint,
            has_movement_hint=has_movement_hint,
            complexity_score=complexity,
            recommended_primitives=recommended,
        )

    def _default_fingerprint(self) -> PatternFingerprint:
        """Return default fingerprint when no examples."""
        return PatternFingerprint(
            has_size_change=False,
            size_change_type="none",
            scale_factors=(1.0, 1.0),
            object_count_input=0,
            object_count_output=0,
            object_count_change="same",
            distinct_colors_input=0,
            distinct_colors_output=0,
            has_color_change=False,
            has_symmetry=False,
            has_repetition=False,
            has_rotation_hint=False,
            has_movement_hint=False,
            complexity_score=0.5,
            recommended_primitives=[
                "smart_tiling",
                "color_mapping",
                "object_manipulation",
            ],
        )

    def _analyze_size_change(
        self, inp_shape: Tuple, out_shape: Tuple
    ) -> Tuple[str, Tuple[float, float]]:
        """Analyze type of size change."""
        if inp_shape == out_shape:
            return "none", (1.0, 1.0)

        h_scale = out_shape[0] / inp_shape[0]
        w_scale = out_shape[1] / inp_shape[1]

        if h_scale == w_scale:
            if h_scale == int(h_scale):
                return "uniform_scale", (h_scale, w_scale)
            else:
                return "uniform_scale", (h_scale, w_scale)
        else:
            if h_scale == int(h_scale) and w_scale == int(w_scale):
                return "non_uniform_scale", (h_scale, w_scale)
            else:
                return "complex", (h_scale, w_scale)

    def _count_objects_fast(self, grid: np.ndarray) -> int:
        """Fast object counting using connected components."""
        # Count non-background connected components
        non_bg = grid != 0
        labeled, num_features = ndimage.label(non_bg)
        return num_features

    def _categorize_count_change(self, count_in: int, count_out: int) -> str:
        """Categorize object count change."""
        if count_in == count_out:
            return "same"
        elif count_out > count_in:
            return "increase"
        else:
            return "decrease"

    def _detect_symmetry(self, grid: np.ndarray) -> bool:
        """Quick symmetry detection."""
        # Check horizontal symmetry
        if np.array_equal(grid, np.flip(grid, axis=0)):
            return True
        # Check vertical symmetry
        if np.array_equal(grid, np.flip(grid, axis=1)):
            return True
        # Check diagonal symmetry (if square)
        if grid.shape[0] == grid.shape[1]:
            if np.array_equal(grid, grid.T):
                return True
        return False

    def _detect_repetition(self, grid: np.ndarray) -> bool:
        """Quick repetition detection."""
        h, w = grid.shape

        # Check 2x2 repetition
        if h >= 4 and w >= 4 and h % 2 == 0 and w % 2 == 0:
            h2, w2 = h // 2, w // 2
            quadrants = [grid[:h2, :w2], grid[:h2, w2:], grid[h2:, :w2], grid[h2:, w2:]]
            if all(np.array_equal(quadrants[0], q) for q in quadrants[1:]):
                return True

        # Check row repetition
        if h > 1:
            first_row = grid[0]
            if all(np.array_equal(first_row, grid[i]) for i in range(1, min(h, 3))):
                return True

        return False

    def _detect_rotation_hint(self, inp: np.ndarray, out: np.ndarray) -> bool:
        """Detect if rotation might be involved."""
        if inp.shape != out.shape:
            return False

        # Check if output could be rotation of input
        for rot in range(1, 4):
            if np.array_equal(np.rot90(inp, rot), out):
                return True

        # Check if pattern suggests rotation (simplified)
        inp_diag_sum = np.trace(inp) if inp.shape[0] == inp.shape[1] else 0
        out_diag_sum = np.trace(out) if out.shape[0] == out.shape[1] else 0

        return inp_diag_sum != out_diag_sum and inp.sum() == out.sum()

    def _detect_movement_hint(self, inp: np.ndarray, out: np.ndarray) -> bool:
        """Detect if objects might have moved."""
        if inp.shape != out.shape:
            return False

        # Check if same values but different positions
        if not np.array_equal(np.sort(inp.flatten()), np.sort(out.flatten())):
            return False

        # Check if center of mass shifted
        inp_com = ndimage.center_of_mass(inp > 0)
        out_com = ndimage.center_of_mass(out > 0)

        distance = np.linalg.norm(np.array(inp_com) - np.array(out_com))
        return distance > 1.0

    def _calculate_complexity(
        self,
        has_size_change: bool,
        obj_count: int,
        color_count: int,
        has_color_change: bool,
        has_movement: bool,
    ) -> float:
        """Calculate task complexity score."""
        complexity = 0.0

        if has_size_change:
            complexity += 0.2
        if obj_count > 3:
            complexity += 0.3
        if color_count > 3:
            complexity += 0.2
        if has_color_change:
            complexity += 0.15
        if has_movement:
            complexity += 0.15

        return min(complexity, 1.0)

    def _recommend_primitives(
        self,
        has_size_change: bool,
        size_change_type: str,
        obj_count: int,
        obj_change: str,
        has_color_change: bool,
        has_symmetry: bool,
        has_repetition: bool,
        has_rotation: bool,
        has_movement: bool,
    ) -> List[str]:
        """Recommend primitives based on fingerprint."""
        recommendations = []

        # Priority 1: Size change
        if has_size_change:
            if size_change_type in ["uniform_scale", "non_uniform_scale"]:
                recommendations.append("smart_tiling")
            else:
                recommendations.append("complex_scaling")

        # Priority 2: Object manipulation
        if obj_count > 1 or obj_change != "same":
            recommendations.append("object_manipulation")

        # Priority 3: Color changes
        if has_color_change:
            recommendations.append("color_mapping")

        # Priority 4: Geometric transformations
        if has_rotation:
            recommendations.append("rotation_reflection")

        if has_movement:
            recommendations.append("object_movement")

        # Priority 5: Pattern-based
        if has_symmetry:
            recommendations.append("symmetry_operations")

        if has_repetition:
            recommendations.append("pattern_completion")

        # Always include fallbacks
        if "smart_tiling" not in recommendations:
            recommendations.append("smart_tiling")
        if "color_mapping" not in recommendations:
            recommendations.append("color_mapping")

        return recommendations[:5]  # Top 5 recommendations


def fingerprint_task(
    examples: List[Tuple[np.ndarray, np.ndarray]], test_input: np.ndarray
) -> PatternFingerprint:
    """Convenience function to fingerprint a task."""
    fingerprinter = PatternFingerprinter()
    return fingerprinter.fingerprint(examples, test_input)


if __name__ == "__main__":
    print("Pattern Fingerprinting System")
    print("=" * 60)

    # Test with sample data
    fingerprinter = PatternFingerprinter()

    # Test case 1: Size change
    inp1 = np.array([[1, 2], [3, 4]])
    out1 = np.array([[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]])

    fp1 = fingerprinter.fingerprint([(inp1, out1)], inp1)
    print("\nTest 1 - Size Change:")
    print(f"  Has size change: {fp1.has_size_change}")
    print(f"  Type: {fp1.size_change_type}")
    print(f"  Scale: {fp1.scale_factors}")
    print(f"  Recommended: {fp1.recommended_primitives}")

    # Test case 2: Object movement
    inp2 = np.array([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 2, 2], [0, 0, 2, 2]])
    out2 = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 0, 0], [2, 2, 0, 0]])

    fp2 = fingerprinter.fingerprint([(inp2, out2)], inp2)
    print("\nTest 2 - Object Movement:")
    print(f"  Object count: {fp2.object_count_input} -> {fp2.object_count_output}")
    print(f"  Has movement: {fp2.has_movement_hint}")
    print(f"  Complexity: {fp2.complexity_score:.2f}")
    print(f"  Recommended: {fp2.recommended_primitives}")
