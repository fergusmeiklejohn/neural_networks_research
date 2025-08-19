#!/usr/bin/env python3
from compositional_dsl import ExecutionContext, Primitive


class CrossPattern_ae3edfdc(Primitive):
    """Auto-discovered cross pattern for ae3edfdc."""

    def __init__(self):
        self.center_colors = [1, 2]
        self.marker_colors = [3, 7]

    def execute(self, context: ExecutionContext) -> ExecutionContext:
        result = context.copy()
        grid = result.current_grid.copy()
        h, w = grid.shape

        # Find positions to form crosses based on detected pattern
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                # Check if this position should have a cross
                if grid[i, j] in self.center_colors:
                    # Check for markers on same row/column
                    has_h_markers = False
                    has_v_markers = False
                    marker_color = None

                    # Check horizontal
                    for jj in range(w):
                        if jj != j and grid[i, jj] in self.marker_colors:
                            has_h_markers = True
                            marker_color = grid[i, jj]
                            break

                    # Check vertical
                    for ii in range(h):
                        if ii != i and grid[ii, j] in self.marker_colors:
                            has_v_markers = True
                            if marker_color is None:
                                marker_color = grid[ii, j]
                            break

                    # Form cross if markers found
                    if has_h_markers or has_v_markers:
                        if marker_color is not None:
                            # Form the cross
                            if i > 0:
                                grid[i - 1, j] = marker_color
                            if i < h - 1:
                                grid[i + 1, j] = marker_color
                            if j > 0:
                                grid[i, j - 1] = marker_color
                            if j < w - 1:
                                grid[i, j + 1] = marker_color

                            # Clear original markers
                            for jj in range(w):
                                if abs(jj - j) > 1 and grid[i, jj] == marker_color:
                                    grid[i, jj] = 0
                            for ii in range(h):
                                if abs(ii - i) > 1 and grid[ii, j] == marker_color:
                                    grid[ii, j] = 0

        result.current_grid = grid
        return result

    def __str__(self):
        return "CrossPattern_ae3edfdc()"
