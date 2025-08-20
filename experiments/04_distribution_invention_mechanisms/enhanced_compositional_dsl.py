#!/usr/bin/env python3
"""Enhanced Compositional DSL combining all primitives.

This integrates:
1. Original compositional DSL primitives
2. Advanced DSL primitives (FillInterior, etc.)
3. Newly identified missing primitives
"""

from utils.imports import setup_project_paths

setup_project_paths()

from typing import List, Optional, Tuple

import numpy as np

# Import advanced primitives
from advanced_dsl_primitives import ConnectPoints
from advanced_dsl_primitives import (
    CountObjects as CountObjectsAdv,  # Rename to avoid conflict
)
from advanced_dsl_primitives import CropToContent, DetectPattern
from advanced_dsl_primitives import DrawLine as DrawLineAdv  # Rename to avoid conflict
from advanced_dsl_primitives import (
    ExtractLargestObject,
    ExtractSubgrid,
    FillInterior,
    FloodFill,
    IfColorPresent,
    IfShapeIs,
    MirrorSymmetry,
    RepeatGrid,
    SelectBySize,
)

# Import base DSL components
from compositional_dsl import (  # Base primitives; Compositional operators; Pattern operations; Spatial relations
    CompositionalDSL,
    Conditional,
    DrawBorder,
    ExecutionContext,
    ExtractObjects,
    FillRectangle,
    FlipH,
    FlipV,
    ForEachObject,
    IsAdjacent,
    IsInside,
    Loop,
    Move,
    Primitive,
    Rotate,
    Sequence,
    SetColor,
    TilePattern,
)

# Import missing primitives
from missing_dsl_primitives import (
    ConditionalFill,
    ConnectObjects,
    CountObjects,
    DrawLine,
    ExtractBoundaries,
    PartitionGrid,
    PropagatePattern,
    SortBySize,
    TraceBorder,
)


class EnhancedCompositionalDSL(CompositionalDSL):
    """Enhanced DSL with all primitive types."""

    def __init__(self):
        """Initialize with all primitive types."""
        super().__init__()

        # Base atomic primitives from parent class
        self.atomic_primitives = [
            Move,
            Rotate,
            FlipH,
            FlipV,
            SetColor,
            FillRectangle,
            ExtractObjects,
            ForEachObject,
            TilePattern,
            DrawBorder,
        ]

        # Add advanced primitives
        self.advanced_primitives = [
            FillInterior,
            FloodFill,
            ConnectPoints,
            MirrorSymmetry,
            ExtractLargestObject,
            CropToContent,
            SelectBySize,
            ExtractSubgrid,
            RepeatGrid,
            DetectPattern,
            IfColorPresent,
            IfShapeIs,
        ]

        # Add missing primitives
        self.missing_primitives = [
            DrawLine,
            ConnectObjects,
            CountObjects,
            SortBySize,
            PartitionGrid,
            ConditionalFill,
            PropagatePattern,
            ExtractBoundaries,
            TraceBorder,
        ]

        # Combine all primitives
        self.all_primitives = (
            self.atomic_primitives + self.advanced_primitives + self.missing_primitives
        )

        # Update primitive type counts for search
        self.primitive_categories = {
            "spatial": [Move, Rotate, FlipH, FlipV, CropToContent],
            "color": [
                SetColor,
                FillRectangle,
                FillInterior,
                FloodFill,
                ConditionalFill,
            ],
            "object": [
                ExtractObjects,
                ForEachObject,
                ExtractLargestObject,
                SelectBySize,
                SortBySize,
                CountObjects,
            ],
            "pattern": [
                TilePattern,
                DrawBorder,
                MirrorSymmetry,
                PropagatePattern,
                DetectPattern,
                RepeatGrid,
            ],
            "line": [DrawLine, ConnectObjects, ConnectPoints],
            "grid": [PartitionGrid, ExtractSubgrid],
            "edge": [ExtractBoundaries, TraceBorder],
            "conditional": [IfColorPresent, IfShapeIs],
        }

    def get_primitives_for_task(
        self, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> List[Primitive]:
        """Get relevant primitives based on task analysis."""
        relevant = []

        # Analyze examples
        inp, out = examples[0]

        # Size change analysis
        if inp.shape != out.shape:
            # Size change - add tiling, cropping, partitioning
            relevant.extend([TilePattern, CropToContent, PartitionGrid])
        else:
            # Same size - focus on transformations
            relevant.extend([Rotate, FlipH, FlipV, SetColor])

        # Color analysis
        in_colors = set(np.unique(inp))
        out_colors = set(np.unique(out))

        if out_colors - in_colors:
            # New colors appear - need drawing/filling
            relevant.extend([FillInterior, FloodFill, ConditionalFill, DrawLine])

        if len(in_colors) > 3:
            # Multiple colors - likely needs object manipulation
            relevant.extend([ExtractObjects, ForEachObject, SortBySize, CountObjects])

        # Check for patterns
        if self._has_repeating_pattern(inp) or self._has_repeating_pattern(out):
            relevant.extend([PropagatePattern, MirrorSymmetry])

        # Check for lines
        if self._has_lines(out):
            relevant.extend([DrawLine, ConnectObjects, ConnectPoints])

        # Check for boundaries
        if self._needs_boundary_detection(inp, out):
            relevant.extend([ExtractBoundaries, TraceBorder])

        return relevant

    def _has_repeating_pattern(self, grid: np.ndarray) -> bool:
        """Check if grid has repeating patterns."""
        h, w = grid.shape

        # Check for 3x3 patterns
        if h >= 6 and w >= 6:
            pattern = grid[:3, :3]
            for i in range(0, h - 2, 3):
                for j in range(0, w - 2, 3):
                    if not np.array_equal(grid[i : i + 3, j : j + 3], pattern):
                        return False
            return True

        return False

    def _has_lines(self, grid: np.ndarray) -> bool:
        """Check if grid has lines."""
        # Check for horizontal lines
        for row in grid:
            non_zero = row[row != 0]
            if len(non_zero) > 3 and len(np.unique(non_zero)) == 1:
                return True

        # Check for vertical lines
        for col in grid.T:
            non_zero = col[col != 0]
            if len(non_zero) > 3 and len(np.unique(non_zero)) == 1:
                return True

        return False

    def _needs_boundary_detection(self, inp: np.ndarray, out: np.ndarray) -> bool:
        """Check if task needs boundary detection."""
        # If output has thin structures not in input
        if inp.shape == out.shape:
            diff = (out != 0) & (inp == 0)
            if np.sum(diff) > 0:
                # Check if the new pixels form boundaries
                from scipy.ndimage import binary_erosion

                eroded = binary_erosion(out != 0)
                boundaries = (out != 0) & ~eroded

                # If most new pixels are boundaries
                if np.sum(boundaries & diff) > np.sum(diff) * 0.5:
                    return True

        return False

    def create_sketch_for_task(
        self, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[Primitive]:
        """Create a sketch based on task analysis."""
        inp, out = examples[0]

        # Check for specific patterns

        # Interior filling pattern
        if self._is_interior_fill(examples):
            # Detect boundary and fill colors
            boundary_color = self._detect_boundary_color(examples)
            fill_color = self._detect_fill_color(examples)
            return FillInterior(boundary_color, fill_color)

        # Line drawing pattern
        if self._is_line_drawing(examples):
            return Sequence([ExtractObjects(), ConnectObjects()])

        # Sorting pattern
        if self._is_sorting(examples):
            return Sequence([ExtractObjects(), SortBySize()])

        # Grid partitioning
        if self._is_partitioning(examples):
            rows, cols = self._detect_partition_size(examples)
            return PartitionGrid(rows, cols)

        # Conditional fill
        if self._is_conditional_fill(examples):
            return ConditionalFill("adjacent_to")

        # Fall back to base sketch creation
        return super().create_sketch_for_task(examples)

    def _is_interior_fill(self, examples):
        """Check if task is interior filling."""
        for inp, out in examples:
            if inp.shape == out.shape:
                # Check if there are enclosed regions that get filled
                diff = out != inp
                if np.sum(diff) > 0:
                    # Check if changes are inside boundaries
                    from scipy.ndimage import binary_fill_holes

                    for color in np.unique(inp):
                        if color != 0:
                            boundary = inp == color
                            filled = binary_fill_holes(boundary)
                            if np.sum(filled & ~boundary) > 0:
                                return True
        return False

    def _is_line_drawing(self, examples):
        """Check if task involves drawing lines."""
        for inp, out in examples:
            if inp.shape == out.shape:
                # Check for new lines in output
                diff = (out != 0) & (inp == 0)
                if np.sum(diff) > 0:
                    # Check if new pixels form lines
                    for row in out:
                        new_pixels = row[
                            diff[0] if len(diff.shape) == 1 else diff.any(axis=0)
                        ]
                        if len(new_pixels) > 3:
                            return True
        return False

    def _is_sorting(self, examples):
        """Check if task involves sorting objects."""
        for inp, out in examples:
            # Different object arrangements suggest sorting
            from scipy import ndimage

            in_objects = []
            out_objects = []

            for color in np.unique(inp):
                if color != 0:
                    mask = (inp == color).astype(int)
                    labeled, count = ndimage.label(mask)
                    for i in range(1, count + 1):
                        size = np.sum(labeled == i)
                        in_objects.append(size)

            for color in np.unique(out):
                if color != 0:
                    mask = (out == color).astype(int)
                    labeled, count = ndimage.label(mask)
                    for i in range(1, count + 1):
                        size = np.sum(labeled == i)
                        out_objects.append(size)

            # Check if objects are reordered
            if sorted(in_objects) == sorted(out_objects) and in_objects != out_objects:
                return True

        return False

    def _is_partitioning(self, examples):
        """Check if task involves grid partitioning."""
        for inp, out in examples:
            if out.shape[0] > inp.shape[0] or out.shape[1] > inp.shape[1]:
                # Output is larger - might be partitioned
                h_ratio = out.shape[0] / inp.shape[0]
                w_ratio = out.shape[1] / inp.shape[1]

                if h_ratio == int(h_ratio) and w_ratio == int(w_ratio):
                    return True

        return False

    def _is_conditional_fill(self, examples):
        """Check if task involves conditional filling."""
        for inp, out in examples:
            if inp.shape == out.shape:
                diff = out != inp
                if np.sum(diff) > 0:
                    # Check if changes depend on neighbors
                    changed_positions = np.argwhere(diff)
                    for pos in changed_positions[:10]:  # Check first 10
                        i, j = pos
                        if 0 < i < inp.shape[0] - 1 and 0 < j < inp.shape[1] - 1:
                            neighbors = [
                                inp[i - 1, j],
                                inp[i + 1, j],
                                inp[i, j - 1],
                                inp[i, j + 1],
                            ]
                            if any(n != 0 for n in neighbors):
                                return True

        return False

    def _detect_boundary_color(self, examples):
        """Detect boundary color for interior filling."""
        for inp, out in examples:
            # Find color that forms boundaries
            for color in np.unique(inp):
                if color != 0:
                    mask = inp == color
                    from scipy.ndimage import binary_fill_holes

                    filled = binary_fill_holes(mask)
                    if np.sum(filled & ~mask) > 0:
                        return int(color)
        return None

    def _detect_fill_color(self, examples):
        """Detect fill color for interior filling."""
        for inp, out in examples:
            # Find new color in output
            out_colors = set(np.unique(out))
            in_colors = set(np.unique(inp))
            new_colors = out_colors - in_colors
            if new_colors:
                return int(list(new_colors)[0])
        return None

    def _detect_partition_size(self, examples):
        """Detect partition size for grid partitioning."""
        inp, out = examples[0]

        h_ratio = out.shape[0] / inp.shape[0]
        w_ratio = out.shape[1] / inp.shape[1]

        return int(h_ratio), int(w_ratio)


def test_enhanced_dsl():
    """Test the enhanced DSL on a sample task."""
    print("Testing Enhanced Compositional DSL")
    print("=" * 60)

    # Create DSL
    dsl = EnhancedCompositionalDSL()

    print(f"Total primitives: {len(dsl.all_primitives)}")
    print(f"Categories: {list(dsl.primitive_categories.keys())}")

    # Test on a simple example
    input_grid = np.array(
        [
            [0, 3, 0, 0, 3, 0],
            [3, 0, 0, 0, 0, 3],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [3, 0, 0, 0, 0, 3],
            [0, 3, 0, 0, 3, 0],
        ]
    )

    # Test FillInterior
    print("\nTesting FillInterior:")
    fill_interior = FillInterior(boundary_color=3, fill_color=4)
    context = ExecutionContext(input_grid=input_grid, current_grid=input_grid.copy())
    result = fill_interior.execute(context)
    print("Output has interior filled:", 4 in result.current_grid)

    # Test DrawLine
    print("\nTesting DrawLine:")
    draw_line = DrawLine(start=(0, 1), end=(5, 4), color=5)
    context = ExecutionContext(input_grid=input_grid, current_grid=input_grid.copy())
    result = draw_line.execute(context)
    print("Output has line:", 5 in result.current_grid)

    # Test ConditionalFill
    print("\nTesting ConditionalFill:")
    cond_fill = ConditionalFill(condition="adjacent_to", fill_color=2)
    context = ExecutionContext(input_grid=input_grid, current_grid=input_grid.copy())
    result = cond_fill.execute(context)
    print("Output has conditional fill:", 2 in result.current_grid)

    print("\n✅ Enhanced DSL ready with", len(dsl.all_primitives), "primitive types!")


if __name__ == "__main__":
    test_enhanced_dsl()
