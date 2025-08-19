#!/usr/bin/env python3
from compositional_dsl import ExecutionContext, Primitive


class RegionFill_00d62c1b(Primitive):
    """Auto-discovered region filling for 00d62c1b."""

    def __init__(self):
        self.boundary_colors = [3]
        self.fill_color = 4  # Common fill color in ARC

    def execute(self, context: ExecutionContext) -> ExecutionContext:
        result = context.copy()
        grid = result.current_grid.copy()
        from scipy import ndimage

        # Fill regions enclosed by boundary colors
        for boundary_color in self.boundary_colors:
            if boundary_color != 0:
                # Create mask of boundary
                boundary_mask = grid == boundary_color

                # Fill holes in the boundary
                filled = ndimage.binary_fill_holes(boundary_mask)

                # Get interior (filled minus boundary)
                interior = filled & ~boundary_mask

                # Fill interior with fill color
                grid[interior] = self.fill_color

        result.current_grid = grid
        return result

    def __str__(self):
        return "RegionFill_00d62c1b()"
