"""Hierarchical Pattern Detector - Patterns of Patterns.

This module implements detection for:
1. Multi-level transformations (analyze → decide → transform)
2. Recursive patterns (patterns applied to their own output)
3. Compositional patterns (patterns combined in structured ways)
4. Scale-invariant patterns (same pattern at different scales)
"""

from typing import List, Optional, Tuple

import numpy as np
from scipy import ndimage


class HierarchicalPatternDetector:
    """Detect and generate code for hierarchical patterns."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def detect_hierarchical_patterns(
        self, task_id: str, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[str]:
        """Main entry point for hierarchical pattern detection."""

        # Try different hierarchical pattern types
        detectors = [
            ("row_reversal", self._detect_row_reversal),
            ("cyclic_shift", self._detect_cyclic_shift),
            ("global_property_extraction", self._detect_global_property_extraction),
            ("recursive_application", self._detect_recursive_pattern),
            ("multi_stage_transform", self._detect_multi_stage),
            ("pattern_repetition", self._detect_pattern_repetition),
            ("conditional_hierarchy", self._detect_conditional_hierarchy),
        ]

        for pattern_name, detector in detectors:
            if self.verbose:
                print(f"Trying {pattern_name}...")

            result = detector(task_id, examples)
            if result:
                if self.verbose:
                    print(f"✓ Found {pattern_name} pattern!")
                return result

        return None

    def _detect_row_reversal(
        self, task_id: str, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[str]:
        """Detect if output is input with reversed rows."""

        for inp, out in examples:
            if inp.shape != out.shape:
                return None

            # Check if output is vertically flipped input
            if not np.array_equal(out, np.flipud(inp)):
                return None

        # All examples match - generate code
        code = f'''
class RowReversal_{task_id}:
    """Reverse the order of rows (vertical flip)."""

    def execute(self, input_grid):
        return np.flipud(np.array(input_grid))
'''
        return code

    def _detect_cyclic_shift(
        self, task_id: str, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[str]:
        """Detect cyclic shifting patterns."""

        shift_amount = None
        shift_axis = None

        for inp, out in examples:
            if inp.shape != out.shape:
                return None

            # Try different shift amounts and axes
            found_shift = False
            for axis in [0, 1]:  # 0=vertical, 1=horizontal
                for shift in range(1, max(inp.shape)):
                    shifted = np.roll(inp, shift, axis=axis)
                    if np.array_equal(shifted, out):
                        if shift_amount is None:
                            shift_amount = shift
                            shift_axis = axis
                        elif shift_amount != shift or shift_axis != axis:
                            return None  # Inconsistent shifts
                        found_shift = True
                        break
                if found_shift:
                    break

            if not found_shift:
                return None

        if shift_amount is not None:
            axis_name = "rows" if shift_axis == 0 else "columns"
            code = f'''
class CyclicShift_{task_id}:
    """Cyclically shift {axis_name} by {shift_amount}."""

    def execute(self, input_grid):
        return np.roll(np.array(input_grid), {shift_amount}, axis={shift_axis})
'''
            return code

        return None

    def _detect_global_property_extraction(
        self, task_id: str, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[str]:
        """Detect patterns that extract based on global properties."""

        # Check if output is always 1x1 (extracting a single value)
        all_single_output = all(out.shape == (1, 1) for _, out in examples)

        if all_single_output:
            # Analyze what value is being extracted
            extraction_patterns = []

            for inp, out in examples:
                extracted_value = out[0, 0]

                # Count occurrences of each color
                unique, counts = np.unique(inp, return_counts=True)
                color_counts = dict(zip(unique, counts))

                # Remove background (0)
                if 0 in color_counts:
                    del color_counts[0]

                # Check different extraction rules
                if extracted_value in color_counts:
                    count = color_counts[extracted_value]

                    # Is it the rarest color?
                    if count == min(color_counts.values()):
                        extraction_patterns.append("rarest")
                    # Is it the most common?
                    elif count == max(color_counts.values()):
                        extraction_patterns.append("most_common")
                    # Does it appear exactly once?
                    elif count == 1:
                        extraction_patterns.append("unique")

            # Check consistency
            if extraction_patterns and all(
                p == extraction_patterns[0] for p in extraction_patterns
            ):
                pattern_type = extraction_patterns[0]

                code = f'''
class GlobalPropertyExtraction_{task_id}:
    """Extract color based on global property: {pattern_type}."""

    def execute(self, input_grid):
        grid = np.array(input_grid)
        unique, counts = np.unique(grid, return_counts=True)
        color_counts = dict(zip(unique, counts))

        # Remove background
        if 0 in color_counts:
            del color_counts[0]

        if not color_counts:
            return np.array([[0]])

        # Extract based on property
'''
                if pattern_type == "rarest":
                    code += """        min_count = min(color_counts.values())
        for color, count in color_counts.items():
            if count == min_count:
                return np.array([[color]])
"""
                elif pattern_type == "unique":
                    code += """        for color, count in color_counts.items():
            if count == 1:
                return np.array([[color]])
        return np.array([[0]])
"""
                elif pattern_type == "most_common":
                    code += """        max_count = max(color_counts.values())
        for color, count in color_counts.items():
            if count == max_count:
                return np.array([[color]])
"""

                return code

        return None

    def _detect_recursive_pattern(
        self, task_id: str, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[str]:
        """Detect patterns applied recursively."""

        # Check if output shows self-similar structure
        for inp, out in examples:
            if out.shape[0] == inp.shape[0] * 3 and out.shape[1] == inp.shape[1] * 3:
                # Check if it's a fractal-like expansion
                # Each non-zero cell becomes a copy of the whole pattern

                scale = 3
                is_recursive = True

                for y in range(inp.shape[0]):
                    for x in range(inp.shape[1]):
                        if inp[y, x] != 0:
                            # Check if this region contains a scaled version
                            region = out[
                                y * scale : (y + 1) * scale, x * scale : (x + 1) * scale
                            ]

                            # For now, simple check - region should have structure
                            if np.sum(region != 0) < 2:
                                is_recursive = False
                                break
                    if not is_recursive:
                        break

                if is_recursive:
                    code = f'''
class RecursiveExpansion_{task_id}:
    """Recursively expand pattern - each cell becomes the whole pattern."""

    def execute(self, input_grid):
        grid = np.array(input_grid)
        scale = 3
        h, w = grid.shape
        output = np.zeros((h * scale, w * scale), dtype=int)

        for y in range(h):
            for x in range(w):
                if grid[y, x] != 0:
                    # Place pattern in this region
                    output[y*scale:(y+1)*scale, x*scale:(x+1)*scale] = grid

        return output
'''
                    return code

        return None

    def _detect_multi_stage(
        self, task_id: str, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[str]:
        """Detect multi-stage transformations (analyze → transform → refine)."""

        # Example: First identify regions, then transform each differently
        # This is complex - simplified version for now

        for inp, out in examples:
            # Check if transformation involves multiple logical steps
            # For instance: segment → process each segment → combine

            # Detect if there are distinct regions being processed
            labeled_inp, num_inp = ndimage.label(inp != 0)
            labeled_out, num_out = ndimage.label(out != 0)

            if num_inp > 1 and num_out == num_inp:
                # Multiple regions preserved - might be processing each
                code = f'''
class MultiStageTransform_{task_id}:
    """Process each connected component independently."""

    def execute(self, input_grid):
        from scipy import ndimage
        grid = np.array(input_grid)
        output = np.zeros_like(grid)

        # Identify components
        labeled, num_features = ndimage.label(grid != 0)

        # Process each component
        for i in range(1, num_features + 1):
            component_mask = (labeled == i)
            # Apply transformation to each component
            # (Simplified - would need specific logic here)
            output[component_mask] = grid[component_mask]

        return output
'''
                return code

        return None

    def _detect_pattern_repetition(
        self, task_id: str, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[str]:
        """Detect patterns that repeat a sub-pattern in a structured way."""

        for inp, out in examples:
            # Check if output contains repeated structure
            h_out, w_out = out.shape
            h_inp, w_inp = inp.shape

            # Check for tiling with transformation
            if h_out >= h_inp * 2 and w_out >= w_inp * 2:
                # Check if input appears multiple times in output
                found_positions = []

                for y in range(h_out - h_inp + 1):
                    for x in range(w_out - w_inp + 1):
                        region = out[y : y + h_inp, x : x + w_inp]
                        if np.array_equal(region, inp):
                            found_positions.append((y, x))

                if len(found_positions) >= 2:
                    # Pattern repeats - check if it's a grid
                    code = f'''
class PatternRepetition_{task_id}:
    """Repeat pattern in a grid structure."""

    def execute(self, input_grid):
        grid = np.array(input_grid)
        # Create 2x2 repetition
        return np.tile(grid, (2, 2))
'''
                    return code

        return None

    def _detect_conditional_hierarchy(
        self, task_id: str, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[str]:
        """Detect patterns with conditional logic at multiple levels."""

        # Example: If global property X, then apply transformation Y to regions with property Z
        # This is complex and would need specific pattern matching

        # Simplified version - detect if transformation depends on both global and local properties
        for inp, out in examples:
            # Check if certain regions are transformed differently based on context
            # This would require sophisticated analysis
            pass

        return None


def test_hierarchical_detection():
    """Test hierarchical detection on the failed tasks."""
    import json
    from pathlib import Path

    detector = HierarchicalPatternDetector(verbose=True)

    # Test on the two tasks that failed
    test_tasks = ["68b16354", "25ff71a9"]

    print("=" * 60)
    print("TESTING HIERARCHICAL PATTERN DETECTION")
    print("=" * 60)

    data_dir = Path("data/arc_agi_official/ARC-AGI/data/training")

    for task_id in test_tasks:
        print(f"\nTask {task_id}:")
        print("-" * 40)

        with open(data_dir / f"{task_id}.json", "r") as f:
            task = json.load(f)

        examples = [
            (np.array(e["input"]), np.array(e["output"])) for e in task["train"]
        ]

        result = detector.detect_hierarchical_patterns(task_id, examples)

        if result:
            print("Generated code:")
            print(result)

            # Test the generated code
            namespace = {"np": np, "ndimage": ndimage}
            exec(result, namespace)

            # Find the class
            class_name = None
            for name in namespace:
                if name.startswith(
                    (
                        "Row",
                        "Cyclic",
                        "Global",
                        "Recursive",
                        "Multi",
                        "Pattern",
                        "Conditional",
                    )
                ):
                    class_name = name
                    break

            if class_name:
                primitive = namespace[class_name]()

                # Test accuracy
                correct = 0
                for inp, expected_out in examples:
                    predicted = primitive.execute(inp)
                    if np.array_equal(predicted, expected_out):
                        correct += 1

                accuracy = correct / len(examples)
                print(f"Accuracy: {accuracy:.1%}")
        else:
            print("No hierarchical pattern detected")


if __name__ == "__main__":
    test_hierarchical_detection()
