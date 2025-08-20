#!/usr/bin/env python3
"""Symmetry pattern detection for ARC tasks.

Detects mirror, rotation, and flip transformations.
"""

from typing import Dict, Optional

import numpy as np


class SymmetryPatternDetector:
    """Detects symmetry-based transformations."""

    def detect_mirror_patterns(
        self, inp: np.ndarray, out: np.ndarray
    ) -> Optional[Dict]:
        """Detect mirror/reflection patterns."""
        h, w = inp.shape

        # Check horizontal mirror (flip left-right)
        if out.shape == inp.shape:
            flipped_h = np.fliplr(inp)
            if np.array_equal(out, flipped_h):
                return {"type": "horizontal_mirror", "axis": "vertical"}

            # Check vertical mirror (flip top-bottom)
            flipped_v = np.flipud(inp)
            if np.array_equal(out, flipped_v):
                return {"type": "vertical_mirror", "axis": "horizontal"}

            # Check diagonal mirror (transpose)
            if h == w:
                transposed = inp.T
                if np.array_equal(out, transposed):
                    return {"type": "diagonal_mirror", "axis": "main_diagonal"}

                # Anti-diagonal
                anti_transposed = np.fliplr(inp.T)
                if np.array_equal(out, anti_transposed):
                    return {"type": "diagonal_mirror", "axis": "anti_diagonal"}

        # Check partial mirrors (e.g., only non-zero values mirrored)
        partial_mirror = self._detect_partial_mirror(inp, out)
        if partial_mirror:
            return partial_mirror

        return None

    def detect_rotation_patterns(
        self, inp: np.ndarray, out: np.ndarray
    ) -> Optional[Dict]:
        """Detect rotation patterns."""
        if inp.shape != out.shape:
            return None

        # Check 90-degree rotations
        rot90 = np.rot90(inp, 1)
        if np.array_equal(out, rot90):
            return {"type": "rotation", "angle": 90}

        rot180 = np.rot90(inp, 2)
        if np.array_equal(out, rot180):
            return {"type": "rotation", "angle": 180}

        rot270 = np.rot90(inp, 3)
        if np.array_equal(out, rot270):
            return {"type": "rotation", "angle": 270}

        # Check for partial rotations (only specific colors)
        partial_rotation = self._detect_partial_rotation(inp, out)
        if partial_rotation:
            return partial_rotation

        return None

    def detect_symmetry_fill(self, inp: np.ndarray, out: np.ndarray) -> Optional[Dict]:
        """Detect patterns that complete symmetry."""
        if inp.shape != out.shape:
            return None

        h, w = inp.shape

        # Check if output completes horizontal symmetry
        if self._check_horizontal_symmetry_completion(inp, out):
            return {"type": "symmetry_completion", "axis": "vertical"}

        # Check if output completes vertical symmetry
        if self._check_vertical_symmetry_completion(inp, out):
            return {"type": "symmetry_completion", "axis": "horizontal"}

        # Check if output completes diagonal symmetry
        if h == w and self._check_diagonal_symmetry_completion(inp, out):
            return {"type": "symmetry_completion", "axis": "diagonal"}

        return None

    def _detect_partial_mirror(
        self, inp: np.ndarray, out: np.ndarray
    ) -> Optional[Dict]:
        """Detect partial mirror where only some elements are mirrored."""
        if inp.shape != out.shape:
            return None

        h, w = inp.shape

        # Check if non-zero values are mirrored horizontally
        non_zero_mask = inp != 0
        if np.any(non_zero_mask):
            # Create mirrored version of non-zero values
            mirrored = np.zeros_like(inp)
            for i in range(h):
                for j in range(w):
                    if inp[i, j] != 0:
                        mirrored[i, w - 1 - j] = inp[i, j]

            # Check if output matches this pattern
            if np.sum(np.abs(out - mirrored)) / np.sum(non_zero_mask) < 0.2:
                return {
                    "type": "partial_mirror",
                    "axis": "vertical",
                    "elements": "non_zero",
                }

        return None

    def _detect_partial_rotation(
        self, inp: np.ndarray, out: np.ndarray
    ) -> Optional[Dict]:
        """Detect partial rotation where only some elements are rotated."""
        if inp.shape != out.shape:
            return None

        # Check if specific colors are rotated
        unique_colors = np.unique(inp)
        for color in unique_colors:
            if color == 0:
                continue

            color_mask = inp == color
            if np.any(color_mask):
                # Create rotated version of this color
                np.zeros_like(inp)
                for angle in [90, 180, 270]:
                    rot_inp = np.rot90(inp, angle // 90)
                    rot_mask = rot_inp == color

                    # Check if this color's pattern matches rotation
                    if np.sum(out[rot_mask] == color) > np.sum(color_mask) * 0.8:
                        return {
                            "type": "partial_rotation",
                            "angle": angle,
                            "color": int(color),
                        }

        return None

    def _check_horizontal_symmetry_completion(
        self, inp: np.ndarray, out: np.ndarray
    ) -> bool:
        """Check if output completes horizontal symmetry."""
        h, w = inp.shape

        # Check if right half mirrors left half
        for i in range(h):
            for j in range(w // 2):
                if inp[i, j] != 0 and out[i, w - 1 - j] == inp[i, j]:
                    # Found mirroring
                    continue
                elif inp[i, j] != 0 and out[i, w - 1 - j] != inp[i, j]:
                    return False

        # Check if pattern is consistently applied
        symmetry_count = 0
        total_count = 0
        for i in range(h):
            for j in range(w // 2):
                if inp[i, j] != 0:
                    total_count += 1
                    if out[i, w - 1 - j] == inp[i, j]:
                        symmetry_count += 1

        return total_count > 0 and symmetry_count / total_count > 0.8

    def _check_vertical_symmetry_completion(
        self, inp: np.ndarray, out: np.ndarray
    ) -> bool:
        """Check if output completes vertical symmetry."""
        h, w = inp.shape

        # Check if bottom half mirrors top half
        symmetry_count = 0
        total_count = 0
        for i in range(h // 2):
            for j in range(w):
                if inp[i, j] != 0:
                    total_count += 1
                    if out[h - 1 - i, j] == inp[i, j]:
                        symmetry_count += 1

        return total_count > 0 and symmetry_count / total_count > 0.8

    def _check_diagonal_symmetry_completion(
        self, inp: np.ndarray, out: np.ndarray
    ) -> bool:
        """Check if output completes diagonal symmetry."""
        h, w = inp.shape
        if h != w:
            return False

        # Check if elements are mirrored across diagonal
        symmetry_count = 0
        total_count = 0
        for i in range(h):
            for j in range(i):
                if inp[i, j] != 0:
                    total_count += 1
                    if out[j, i] == inp[i, j]:
                        symmetry_count += 1

        return total_count > 0 and symmetry_count / total_count > 0.8


def generate_symmetry_primitive(pattern_details: Dict, task_id: str) -> str:
    """Generate primitive code for symmetry patterns."""
    pattern_type = pattern_details.get("type", "")
    class_name = f"SymmetryPattern_{task_id.replace('-', '_')}"

    if pattern_type == "horizontal_mirror":
        code = f'''
class {class_name}(Primitive):
    """Auto-discovered horizontal mirror for {task_id}."""

    def execute(self, context: ExecutionContext) -> ExecutionContext:
        result = context.copy()
        result.current_grid = np.fliplr(context.current_grid)
        return result

    def __str__(self):
        return "{class_name}()"
'''

    elif pattern_type == "vertical_mirror":
        code = f'''
class {class_name}(Primitive):
    """Auto-discovered vertical mirror for {task_id}."""

    def execute(self, context: ExecutionContext) -> ExecutionContext:
        result = context.copy()
        result.current_grid = np.flipud(context.current_grid)
        return result

    def __str__(self):
        return "{class_name}()"
'''

    elif pattern_type == "diagonal_mirror":
        axis = pattern_details.get("axis", "main_diagonal")
        if axis == "main_diagonal":
            code = f'''
class {class_name}(Primitive):
    """Auto-discovered diagonal mirror for {task_id}."""

    def execute(self, context: ExecutionContext) -> ExecutionContext:
        result = context.copy()
        result.current_grid = context.current_grid.T
        return result

    def __str__(self):
        return "{class_name}()"
'''
        else:  # anti-diagonal
            code = f'''
class {class_name}(Primitive):
    """Auto-discovered anti-diagonal mirror for {task_id}."""

    def execute(self, context: ExecutionContext) -> ExecutionContext:
        result = context.copy()
        result.current_grid = np.fliplr(context.current_grid.T)
        return result

    def __str__(self):
        return "{class_name}()"
'''

    elif pattern_type == "rotation":
        angle = pattern_details.get("angle", 90)
        rotations = angle // 90
        code = f'''
class {class_name}(Primitive):
    """Auto-discovered {angle}-degree rotation for {task_id}."""

    def execute(self, context: ExecutionContext) -> ExecutionContext:
        result = context.copy()
        result.current_grid = np.rot90(context.current_grid, {rotations})
        return result

    def __str__(self):
        return "{class_name}()"
'''

    elif pattern_type == "symmetry_completion":
        axis = pattern_details.get("axis", "vertical")
        code = f'''
class {class_name}(Primitive):
    """Auto-discovered symmetry completion for {task_id}."""

    def execute(self, context: ExecutionContext) -> ExecutionContext:
        result = context.copy()
        grid = context.current_grid.copy()
        h, w = grid.shape

        # Complete symmetry along {axis} axis
'''
        if axis == "vertical":
            code += """        for i in range(h):
            for j in range(w // 2):
                if grid[i, j] != 0 and grid[i, w - 1 - j] == 0:
                    grid[i, w - 1 - j] = grid[i, j]
                elif grid[i, w - 1 - j] != 0 and grid[i, j] == 0:
                    grid[i, j] = grid[i, w - 1 - j]
"""
        elif axis == "horizontal":
            code += """        for i in range(h // 2):
            for j in range(w):
                if grid[i, j] != 0 and grid[h - 1 - i, j] == 0:
                    grid[h - 1 - i, j] = grid[i, j]
                elif grid[h - 1 - i, j] != 0 and grid[i, j] == 0:
                    grid[i, j] = grid[h - 1 - i, j]
"""
        else:  # diagonal
            code += """        for i in range(h):
            for j in range(i):
                if grid[i, j] != 0 and grid[j, i] == 0:
                    grid[j, i] = grid[i, j]
                elif grid[j, i] != 0 and grid[i, j] == 0:
                    grid[i, j] = grid[j, i]
"""

        code += f"""
        result.current_grid = grid
        return result

    def __str__(self):
        return "{class_name}()"
"""

    else:
        # Default case
        code = f'''
class {class_name}(Primitive):
    """Auto-discovered symmetry pattern for {task_id}."""

    def execute(self, context: ExecutionContext) -> ExecutionContext:
        return context

    def __str__(self):
        return "{class_name}()"
'''

    return code
