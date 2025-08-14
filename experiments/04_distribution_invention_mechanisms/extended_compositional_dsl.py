#!/usr/bin/env python3
"""Extended Compositional DSL combining basic and advanced primitives."""

from utils.imports import setup_project_paths

setup_project_paths()

from advanced_dsl_primitives import (
    ConnectPoints,
    CountObjects,
    CropToContent,
    DetectPattern,
    DrawLine,
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
from compositional_dsl import CompositionalDSL


class ExtendedCompositionalDSL(CompositionalDSL):
    """Extended DSL with advanced primitives for ARC tasks."""

    def __init__(self):
        super().__init__()

        # Add advanced primitives
        self.primitives.update(
            {
                # Advanced filling
                "flood_fill": FloodFill,
                "fill_interior": FillInterior,
                # Object manipulation
                "extract_largest": ExtractLargestObject,
                "connect_points": ConnectPoints,
                "count_objects": CountObjects,
                "select_by_size": SelectBySize,
                # Pattern detection
                "detect_pattern": DetectPattern,
                # Grid manipulation
                "crop_to_content": CropToContent,
                "extract_subgrid": ExtractSubgrid,
                "repeat_grid": RepeatGrid,
                # Symmetry
                "mirror_symmetry": MirrorSymmetry,
                # Drawing
                "draw_line": DrawLine,
                # Conditional
                "if_color_present": IfColorPresent,
                "if_shape_is": IfShapeIs,
            }
        )

    def get_all_primitive_names(self):
        """Get list of all primitive names."""
        return list(self.primitives.keys())

    def suggest_primitives_for_task(self, examples):
        """Suggest relevant primitives based on task analysis."""
        suggestions = []

        for inp, out in examples:
            # Check for size changes
            if inp.shape != out.shape:
                suggestions.extend(
                    [
                        "tile_pattern",
                        "repeat_grid",
                        "crop_to_content",
                        "extract_subgrid",
                    ]
                )

            # Check for new colors in output
            in_colors = set(inp.flatten())
            out_colors = set(out.flatten())
            if out_colors - in_colors:
                suggestions.extend(["fill_interior", "flood_fill", "set_color"])

            # Check for connectivity patterns
            if 3 in inp.flatten() or 3 in out.flatten():
                suggestions.extend(
                    ["connect_points", "fill_interior", "extract_objects"]
                )

            # Check for symmetry
            import numpy as np

            if np.array_equal(
                out[:, : out.shape[1] // 2], np.fliplr(out[:, out.shape[1] // 2 :])
            ):
                suggestions.append("mirror_symmetry")

        return list(set(suggestions))  # Remove duplicates
