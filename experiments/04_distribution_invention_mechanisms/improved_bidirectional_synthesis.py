#!/usr/bin/env python3
"""Improved bidirectional synthesis with better sketch detection for ARC tasks."""

from utils.imports import setup_project_paths

setup_project_paths()

from typing import List, Optional, Tuple

import numpy as np
from bidirectional_synthesis import (
    BidirectionalSynthesizer,
    BottomUpEnumerator,
    ProgramNode,
    TopDownSynthesizer,
)
from extended_compositional_dsl import ExtendedCompositionalDSL


class ImprovedTopDownSynthesizer(TopDownSynthesizer):
    """Improved top-down synthesis with better pattern detection."""

    def __init__(self, dsl: ExtendedCompositionalDSL):
        super().__init__(dsl)
        self.dsl = dsl  # Use extended DSL
        self.sketch_patterns = [
            "color_transform",
            "spatial_transform",
            "object_manipulation",
            "pattern_tiling",
            "fill_interior",  # New sketch!
            "flood_fill",  # New sketch!
            "symmetry",  # New sketch!
        ]

    def synthesize_from_sketch(
        self, sketch: str, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[ProgramNode]:
        """Synthesize based on sketch type."""
        if sketch == "fill_interior":
            return self._synthesize_fill_interior(examples)
        elif sketch == "flood_fill":
            return self._synthesize_flood_fill(examples)
        elif sketch == "symmetry":
            return self._synthesize_symmetry(examples)
        else:
            # Use parent implementation for other sketches
            return super().synthesize_from_sketch(sketch, examples)

    def _synthesize_fill_interior(
        self, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[ProgramNode]:
        """Synthesize fill interior operations."""
        # Analyze examples for interior filling pattern
        for inp, out in examples:
            if inp.shape != out.shape:
                continue

            # Find colors that appear in output but not input (fill colors)
            in_colors = set(np.unique(inp))
            out_colors = set(np.unique(out))
            new_colors = out_colors - in_colors

            if not new_colors:
                continue

            # Check each potential boundary color
            for boundary_color in in_colors:
                if boundary_color == 0:  # Skip background
                    continue

                for fill_color in new_colors:
                    # Try FillInterior
                    from advanced_dsl_primitives import FillInterior

                    program = FillInterior(
                        boundary_color=int(boundary_color), fill_color=int(fill_color)
                    )

                    # Evaluate on all examples
                    enumerator = BottomUpEnumerator(self.dsl)
                    node = enumerator._evaluate_program(program, examples, 1)

                    if node.score >= 0.99:  # Perfect match
                        return node

        return None

    def _synthesize_flood_fill(
        self, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[ProgramNode]:
        """Synthesize flood fill operations."""
        # This would need analysis to find flood fill patterns
        # For now, skip implementation
        return None

    def _synthesize_symmetry(
        self, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[ProgramNode]:
        """Synthesize symmetry operations."""
        for inp, out in examples:
            # Check vertical symmetry
            if out.shape[1] % 2 == 0:
                left = out[:, : out.shape[1] // 2]
                right = out[:, out.shape[1] // 2 :]
                if np.array_equal(left, np.fliplr(right)):
                    from advanced_dsl_primitives import MirrorSymmetry

                    program = MirrorSymmetry(axis="vertical", mode="left_to_right")
                    enumerator = BottomUpEnumerator(self.dsl)
                    node = enumerator._evaluate_program(program, examples, 1)
                    if node.score > 0.9:
                        return node

        return None


class ImprovedBottomUpEnumerator(BottomUpEnumerator):
    """Improved bottom-up enumerator with extended primitive support."""

    def _generate_parameters(
        self, primitive_name: str, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> List[dict]:
        """Generate parameters for extended primitives."""
        params_list = []

        if primitive_name == "fill_interior":
            # Analyze for boundary and fill colors
            for inp, out in examples:
                in_colors = set(np.unique(inp))
                out_colors = set(np.unique(out))
                new_colors = out_colors - in_colors

                for boundary in in_colors:
                    if boundary != 0:
                        for fill in new_colors:
                            params_list.append(
                                {
                                    "boundary_color": int(boundary),
                                    "fill_color": int(fill),
                                }
                            )

            return params_list[:10]

        elif primitive_name == "flood_fill":
            # Would need to analyze starting points
            return []

        elif primitive_name == "extract_largest":
            # Extract largest object of each color
            colors = set()
            for inp, _ in examples:
                colors.update(np.unique(inp))
            colors.discard(0)  # Remove background

            for color in colors:
                params_list.append({"target_color": int(color)})

            return params_list

        elif primitive_name == "mirror_symmetry":
            # Try different symmetry modes
            params_list.extend(
                [
                    {"axis": "vertical", "mode": "left_to_right"},
                    {"axis": "vertical", "mode": "right_to_left"},
                    {"axis": "horizontal", "mode": "top_to_bottom"},
                    {"axis": "horizontal", "mode": "bottom_to_top"},
                ]
            )
            return params_list

        elif primitive_name == "crop_to_content":
            params_list.append({"background_color": 0})
            return params_list

        elif primitive_name == "repeat_grid":
            # Check if output is larger than input
            size_ratios = []
            for inp, out in examples:
                if (
                    out.shape[0] % inp.shape[0] == 0
                    and out.shape[1] % inp.shape[1] == 0
                ):
                    times_y = out.shape[0] // inp.shape[0]
                    times_x = out.shape[1] // inp.shape[1]
                    size_ratios.append((times_x, times_y))

            for times_x, times_y in set(size_ratios):
                params_list.append({"times_x": times_x, "times_y": times_y})

            return params_list

        else:
            # Use parent implementation
            return super()._generate_parameters(primitive_name, examples)


class ImprovedBidirectionalSynthesizer(BidirectionalSynthesizer):
    """Improved bidirectional synthesizer with extended DSL support."""

    def __init__(self, timeout: float = 30.0):
        dsl = ExtendedCompositionalDSL()
        super().__init__(dsl, timeout)
        self.bottom_up = ImprovedBottomUpEnumerator(dsl)
        self.top_down = ImprovedTopDownSynthesizer(dsl)


def test_improved_synthesis():
    """Test the improved synthesis on ARC tasks."""
    print("Testing Improved Synthesis\n")

    synthesizer = ImprovedBidirectionalSynthesizer(timeout=15.0)

    # Test on task 00d62c1b (fill interior)
    print("Task 00d62c1b (Fill Interior):")
    from test_extended_synthesis import load_arc_task

    train_examples, test_examples = load_arc_task("00d62c1b")

    program = synthesizer.synthesize(train_examples)
    if program:
        print(f"  Found: {program}")

        # Test accuracy
        test_result = synthesizer.dsl.execute_program(program, test_examples[0][0])
        match = np.array_equal(test_result, test_examples[0][1])
        print(f"  Test accuracy: {'100%' if match else '0%'}")
    else:
        print("  No program found")

    # Test on task 3c9b0459 (rotation)
    print("\nTask 3c9b0459 (Rotation):")
    train_examples, test_examples = load_arc_task("3c9b0459")

    program = synthesizer.synthesize(train_examples)
    if program:
        print(f"  Found: {program}")
        test_result = synthesizer.dsl.execute_program(program, test_examples[0][0])
        match = np.array_equal(test_result, test_examples[0][1])
        print(f"  Test accuracy: {'100%' if match else '0%'}")
    else:
        print("  No program found")


if __name__ == "__main__":
    test_improved_synthesis()
