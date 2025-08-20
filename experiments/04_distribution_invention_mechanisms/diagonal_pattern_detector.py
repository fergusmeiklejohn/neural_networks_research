#!/usr/bin/env python3
"""Diagonal pattern detection and generation for ARC tasks.

Implements detection and generation for:
- Diagonal lines (45°, 135° angles)
- Anti-diagonal lines
- Diagonal fills and patterns
"""

from utils.imports import setup_project_paths

setup_project_paths()

from typing import Dict, List, Optional, Tuple

import numpy as np
from compositional_dsl import ExecutionContext, Primitive


class DiagonalPatternDetector:
    """Detects diagonal patterns in ARC task transformations."""

    def detect_diagonal_lines(self, inp: np.ndarray, out: np.ndarray) -> Optional[Dict]:
        """Detect diagonal line patterns in transformation."""
        h, w = inp.shape
        diagonals = []

        # Check main diagonals (top-left to bottom-right)
        for offset in range(-(h - 1), w):
            diagonal_coords = self._get_diagonal_coords(h, w, offset, main=True)
            if self._check_line_formed(inp, out, diagonal_coords):
                diagonals.append(
                    {
                        "type": "main_diagonal",
                        "offset": offset,
                        "coords": diagonal_coords,
                    }
                )

        # Check anti-diagonals (top-right to bottom-left)
        for offset in range(h + w - 1):
            diagonal_coords = self._get_diagonal_coords(h, w, offset, main=False)
            if self._check_line_formed(inp, out, diagonal_coords):
                diagonals.append(
                    {
                        "type": "anti_diagonal",
                        "offset": offset,
                        "coords": diagonal_coords,
                    }
                )

        return {"diagonals": diagonals} if diagonals else None

    def _get_diagonal_coords(
        self, h: int, w: int, offset: int, main: bool = True
    ) -> List[Tuple[int, int]]:
        """Get coordinates for a diagonal line."""
        coords = []

        if main:  # Main diagonal (top-left to bottom-right)
            if offset >= 0:
                # Start from top row
                start_i, start_j = 0, offset
            else:
                # Start from left column
                start_i, start_j = -offset, 0

            i, j = start_i, start_j
            while i < h and j < w:
                coords.append((i, j))
                i += 1
                j += 1
        else:  # Anti-diagonal (top-right to bottom-left)
            if offset < w:
                # Start from top row
                start_i, start_j = 0, offset
            else:
                # Start from right column
                start_i, start_j = offset - w + 1, w - 1

            i, j = start_i, start_j
            while i < h and j >= 0:
                coords.append((i, j))
                i += 1
                j -= 1

        return coords

    def _check_line_formed(
        self, inp: np.ndarray, out: np.ndarray, coords: List[Tuple[int, int]]
    ) -> bool:
        """Check if a line was formed along the given coordinates."""
        if len(coords) < 3:  # Too short to be meaningful
            return False

        # Check if pixels along diagonal changed consistently
        changes = 0
        colors = []

        for i, j in coords:
            if inp[i, j] != out[i, j]:
                changes += 1
                if out[i, j] != 0:
                    colors.append(out[i, j])

        # Line formed if >50% of diagonal changed to same non-zero color
        if changes > len(coords) * 0.5 and colors:
            # Check if all changed to same color
            unique_colors = set(colors)
            if len(unique_colors) == 1:
                return True

        return False

    def detect_diagonal_fill(self, inp: np.ndarray, out: np.ndarray) -> Optional[Dict]:
        """Detect diagonal fill patterns (e.g., upper/lower triangular fills)."""
        h, w = inp.shape

        # Check upper triangular fill (above main diagonal)
        upper_changed = 0
        upper_total = 0
        for i in range(h):
            for j in range(w):
                if j > i:  # Above main diagonal
                    upper_total += 1
                    if inp[i, j] != out[i, j]:
                        upper_changed += 1

        # Check lower triangular fill (below main diagonal)
        lower_changed = 0
        lower_total = 0
        for i in range(h):
            for j in range(w):
                if j < i:  # Below main diagonal
                    lower_total += 1
                    if inp[i, j] != out[i, j]:
                        lower_changed += 1

        fills = []
        if upper_total > 0 and upper_changed / upper_total > 0.8:
            fills.append(
                {"type": "upper_triangular", "coverage": upper_changed / upper_total}
            )
        if lower_total > 0 and lower_changed / lower_total > 0.8:
            fills.append(
                {"type": "lower_triangular", "coverage": lower_changed / lower_total}
            )

        return {"fills": fills} if fills else None

    def detect_diagonal_symmetry(
        self, inp: np.ndarray, out: np.ndarray
    ) -> Optional[Dict]:
        """Detect diagonal symmetry transformations."""
        h, w = inp.shape

        # Only check for square grids for simplicity
        if h != w:
            return None

        # Check if output is transpose of input
        if np.array_equal(out, inp.T):
            return {"type": "transpose", "axis": "main_diagonal"}

        # Check if output is anti-diagonal transpose
        anti_transpose = np.fliplr(inp.T)
        if np.array_equal(out, anti_transpose):
            return {"type": "transpose", "axis": "anti_diagonal"}

        return None


class DiagonalLinePattern(Primitive):
    """Auto-generated diagonal line pattern primitive."""

    def __init__(self, diagonal_type: str, offsets: List[int], color: int):
        self.diagonal_type = diagonal_type  # "main_diagonal" or "anti_diagonal"
        self.offsets = offsets
        self.color = color

    def execute(self, context: ExecutionContext) -> ExecutionContext:
        result = context.copy()
        grid = result.current_grid.copy()
        h, w = grid.shape

        for offset in self.offsets:
            coords = self._get_diagonal_coords(h, w, offset)
            for i, j in coords:
                grid[i, j] = self.color

        result.current_grid = grid
        return result

    def _get_diagonal_coords(
        self, h: int, w: int, offset: int
    ) -> List[Tuple[int, int]]:
        """Get coordinates for a diagonal line."""
        coords = []

        if self.diagonal_type == "main_diagonal":
            if offset >= 0:
                start_i, start_j = 0, offset
            else:
                start_i, start_j = -offset, 0

            i, j = start_i, start_j
            while i < h and j < w:
                coords.append((i, j))
                i += 1
                j += 1
        else:  # anti_diagonal
            if offset < w:
                start_i, start_j = 0, offset
            else:
                start_i, start_j = offset - w + 1, w - 1

            i, j = start_i, start_j
            while i < h and j >= 0:
                coords.append((i, j))
                i += 1
                j -= 1

        return coords

    def __str__(self):
        return f"DiagonalLinePattern({self.diagonal_type}, offsets={self.offsets}, color={self.color})"


class DiagonalFillPattern(Primitive):
    """Auto-generated diagonal fill pattern primitive."""

    def __init__(self, fill_type: str, color: int):
        self.fill_type = fill_type  # "upper_triangular" or "lower_triangular"
        self.color = color

    def execute(self, context: ExecutionContext) -> ExecutionContext:
        result = context.copy()
        grid = result.current_grid.copy()
        h, w = grid.shape

        for i in range(h):
            for j in range(w):
                if self.fill_type == "upper_triangular" and j > i:
                    grid[i, j] = self.color
                elif self.fill_type == "lower_triangular" and j < i:
                    grid[i, j] = self.color

        result.current_grid = grid
        return result

    def __str__(self):
        return f"DiagonalFillPattern({self.fill_type}, color={self.color})"


def generate_diagonal_primitive(diagonal_data: Dict, task_id: str) -> str:
    """Generate Python code for diagonal pattern primitive."""

    if "diagonals" in diagonal_data:
        # Diagonal lines pattern
        diagonals = diagonal_data["diagonals"]
        diagonal_type = diagonals[0]["type"]
        offsets = [d["offset"] for d in diagonals]

        # Find common color (simplified - use first non-zero)
        color = 1  # Default

        class_name = f"DiagonalPattern_{task_id.replace('-', '_')}"
        code = f'''
class {class_name}(Primitive):
    """Auto-discovered diagonal pattern for {task_id}."""

    def __init__(self):
        self.diagonal_type = "{diagonal_type}"
        self.offsets = {offsets}
        self.color = {color}

    def execute(self, context: ExecutionContext) -> ExecutionContext:
        result = context.copy()
        grid = result.current_grid.copy()
        h, w = grid.shape

        for offset in self.offsets:
            coords = self._get_diagonal_coords(h, w, offset)
            for i, j in coords:
                if 0 <= i < h and 0 <= j < w:
                    # Find color from existing non-zero pixels
                    if grid[i, j] != 0:
                        color = grid[i, j]
                    else:
                        color = self.color
                    grid[i, j] = color

        result.current_grid = grid
        return result

    def _get_diagonal_coords(self, h, w, offset):
        coords = []
        if self.diagonal_type == "main_diagonal":
            if offset >= 0:
                i, j = 0, offset
            else:
                i, j = -offset, 0
            while i < h and j < w:
                coords.append((i, j))
                i += 1
                j += 1
        else:  # anti_diagonal
            if offset < w:
                i, j = 0, offset
            else:
                i, j = offset - w + 1, w - 1
            while i < h and j >= 0:
                coords.append((i, j))
                i += 1
                j -= 1
        return coords

    def __str__(self):
        return "{class_name}()"
'''
        return code

    elif "fills" in diagonal_data:
        # Diagonal fill pattern
        fill_type = diagonal_data["fills"][0]["type"]

        class_name = f"DiagonalFill_{task_id.replace('-', '_')}"
        code = f'''
class {class_name}(Primitive):
    """Auto-discovered diagonal fill for {task_id}."""

    def __init__(self):
        self.fill_type = "{fill_type}"
        self.fill_color = 4  # Common fill color in ARC

    def execute(self, context: ExecutionContext) -> ExecutionContext:
        result = context.copy()
        grid = result.current_grid.copy()
        h, w = grid.shape

        for i in range(h):
            for j in range(w):
                if self.fill_type == "upper_triangular" and j > i:
                    if grid[i, j] == 0:  # Only fill empty cells
                        grid[i, j] = self.fill_color
                elif self.fill_type == "lower_triangular" and j < i:
                    if grid[i, j] == 0:
                        grid[i, j] = self.fill_color

        result.current_grid = grid
        return result

    def __str__(self):
        return "{class_name}()"
'''
        return code

    elif "type" in diagonal_data and "transpose" in diagonal_data["type"]:
        # Diagonal symmetry/transpose
        axis = diagonal_data["axis"]

        class_name = f"DiagonalTranspose_{task_id.replace('-', '_')}"
        code = f'''
class {class_name}(Primitive):
    """Auto-discovered diagonal transpose for {task_id}."""

    def __init__(self):
        self.axis = "{axis}"

    def execute(self, context: ExecutionContext) -> ExecutionContext:
        result = context.copy()
        grid = result.current_grid.copy()

        if self.axis == "main_diagonal":
            result.current_grid = grid.T
        else:  # anti_diagonal
            import numpy as np
            result.current_grid = np.fliplr(grid.T)

        return result

    def __str__(self):
        return "{class_name}()"
'''
        return code

    return None


# Test the diagonal pattern detector
if __name__ == "__main__":
    print("Diagonal Pattern Detector Test")
    print("=" * 60)

    # Create test case with diagonal line
    inp = np.zeros((5, 5), dtype=int)
    out = np.zeros((5, 5), dtype=int)

    # Add main diagonal
    for i in range(5):
        out[i, i] = 3

    # Add anti-diagonal
    for i in range(5):
        out[i, 4 - i] = 2

    print("Input:")
    print(inp)
    print("\nOutput:")
    print(out)

    detector = DiagonalPatternDetector()

    # Test diagonal line detection
    diagonals = detector.detect_diagonal_lines(inp, out)
    if diagonals:
        print(f"\nDetected diagonal lines: {diagonals}")

    # Test with triangular fill
    out2 = np.zeros((5, 5), dtype=int)
    for i in range(5):
        for j in range(5):
            if j > i:
                out2[i, j] = 4

    fills = detector.detect_diagonal_fill(inp, out2)
    if fills:
        print(f"\nDetected diagonal fills: {fills}")

    # Test transpose
    inp3 = np.array([[1, 2], [3, 4]])
    out3 = np.array([[1, 3], [2, 4]])

    symmetry = detector.detect_diagonal_symmetry(inp3, out3)
    if symmetry:
        print(f"\nDetected diagonal symmetry: {symmetry}")

    # Generate primitive code
    if diagonals:
        code = generate_diagonal_primitive(diagonals, "test-task")
        print(f"\nGenerated code preview:")
        print(code[:500] + "...")
