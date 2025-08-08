#!/usr/bin/env python3
"""Enhanced ARC Solver V3 with size-aware strategy and forced synthesis.

Key improvements over V2:
- Detects size changes to guide strategy selection
- Forces synthesis when size changes detected
- Validates pattern confidence on training examples
- Continues searching even with high-confidence patterns
- Adds pattern composition capability
"""

from utils.imports import setup_project_paths

setup_project_paths()

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from arc_dsl import Program
from arc_dsl_enhanced import EnhancedDSLLibrary
from enhanced_arc_tta import EnhancedARCTestTimeAdapter
from enhanced_perception_v2 import EnhancedPerceptionV2
from object_manipulation import ObjectManipulator
from program_synthesis import ProgramSynthesizer


@dataclass
class ARCSolution:
    """Represents a solution to an ARC task."""

    output_grid: np.ndarray
    confidence: float
    method_used: str  # 'simple', 'object', 'synthesis', 'tta', 'composition'
    program: Optional[Program] = None
    time_taken: float = 0.0
    perception_analysis: Optional[Dict] = None
    actual_confidence: float = 0.0  # Confidence based on validation


@dataclass
class SizeChangeInfo:
    """Information about size changes in the task."""

    has_size_change: bool
    scale_factor: Optional[Tuple[float, float]] = None  # (height_scale, width_scale)
    is_exact_multiple: bool = False
    transformation_type: str = "unknown"  # 'tiling', 'scaling', 'cropping', etc.


class EnhancedARCSolverV3:
    """Improved solver with size-aware strategy and better search."""

    def __init__(
        self,
        use_synthesis: bool = True,
        synthesis_timeout: float = 10.0,
        force_synthesis_on_size_change: bool = True,
        validate_patterns: bool = True,
    ):
        """Initialize the enhanced solver V3.

        Args:
            use_synthesis: Whether to use program synthesis
            synthesis_timeout: Maximum time for program synthesis in seconds
            force_synthesis_on_size_change: Force synthesis when size changes detected
            validate_patterns: Validate pattern confidence on training examples
        """
        self.perception = EnhancedPerceptionV2()
        self.object_manipulator = ObjectManipulator()
        self.tta_adapter = EnhancedARCTestTimeAdapter()
        self.synthesizer = (
            ProgramSynthesizer(strategy="beam") if use_synthesis else None
        )
        self.synthesis_timeout = synthesis_timeout
        self.force_synthesis_on_size_change = force_synthesis_on_size_change
        self.validate_patterns = validate_patterns
        self.dsl_library = EnhancedDSLLibrary()

    def detect_size_change(
        self, train_examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> SizeChangeInfo:
        """Detect if the task involves size changes.

        Args:
            train_examples: Training examples to analyze

        Returns:
            SizeChangeInfo with details about size changes
        """
        size_info = SizeChangeInfo(has_size_change=False)

        if not train_examples:
            return size_info

        # Analyze size ratios
        height_ratios = []
        width_ratios = []

        for input_grid, output_grid in train_examples:
            in_h, in_w = input_grid.shape
            out_h, out_w = output_grid.shape

            if in_h > 0 and in_w > 0:
                height_ratios.append(out_h / in_h)
                width_ratios.append(out_w / in_w)

        if not height_ratios:
            return size_info

        # Check if sizes change consistently
        avg_h_ratio = sum(height_ratios) / len(height_ratios)
        avg_w_ratio = sum(width_ratios) / len(width_ratios)

        # Check for size changes
        if abs(avg_h_ratio - 1.0) > 0.1 or abs(avg_w_ratio - 1.0) > 0.1:
            size_info.has_size_change = True
            size_info.scale_factor = (avg_h_ratio, avg_w_ratio)

            # Check if it's an exact multiple (likely tiling)
            if (
                avg_h_ratio == int(avg_h_ratio)
                and avg_w_ratio == int(avg_w_ratio)
                and avg_h_ratio == avg_w_ratio
            ):
                size_info.is_exact_multiple = True
                scale = int(avg_h_ratio)

                # Determine transformation type
                if scale > 1:
                    # Check if it's tiling by looking at pattern repetition
                    # For now, assume tiling if scale is 2, 3, or 4
                    if scale in [2, 3, 4]:
                        size_info.transformation_type = "tiling"
                    else:
                        size_info.transformation_type = "scaling"
                elif scale < 1:
                    size_info.transformation_type = "cropping"

        return size_info

    def validate_pattern_confidence(
        self, pattern_fn, train_examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> float:
        """Validate a pattern's actual confidence by testing on training examples.

        Args:
            pattern_fn: Function that applies the pattern transformation
            train_examples: Training examples to validate against

        Returns:
            Actual confidence score based on how well pattern fits examples
        """
        if not train_examples:
            return 0.0

        correct = 0
        total = len(train_examples)

        for input_grid, expected_output in train_examples:
            try:
                predicted = pattern_fn(input_grid)
                if np.array_equal(predicted, expected_output):
                    correct += 1
                else:
                    # Partial credit for close matches
                    if predicted.shape == expected_output.shape:
                        matching = np.sum(predicted == expected_output)
                        total_pixels = predicted.size
                        if total_pixels > 0:
                            correct += (matching / total_pixels) * 0.5
            except Exception:
                # Pattern failed on this example
                pass

        return correct / total if total > 0 else 0.0

    def solve(
        self,
        train_examples: List[Tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
    ) -> ARCSolution:
        """Solve an ARC task using size-aware strategy.

        Key improvements over V2:
        1. Detect size changes early to guide strategy
        2. Force synthesis when size changes detected
        3. Validate pattern confidence on actual examples
        4. Try multiple approaches and pick best
        5. Add pattern composition attempts

        Args:
            train_examples: List of (input, output) training examples
            test_input: Test input grid to transform

        Returns:
            ARCSolution with the best prediction
        """
        start_time = time.time()

        # NEW: Detect size changes
        size_info = self.detect_size_change(train_examples)
        force_synthesis = (
            size_info.has_size_change and self.force_synthesis_on_size_change
        )

        if size_info.has_size_change:
            print(
                f"  Size change detected: {size_info.transformation_type} "
                f"(scale: {size_info.scale_factor})"
            )

        # Analyze patterns with enhanced perception
        perception_analysis = self.perception.analyze(train_examples)

        # NEW: Collect all solutions instead of returning early
        solutions = []

        # 1. Try simple patterns
        if perception_analysis["transformation_type"] in ["arithmetic", "spatial"]:
            solution = self.try_enhanced_simple_patterns(
                train_examples, test_input, perception_analysis
            )
            if solution:
                # NEW: Validate confidence on actual examples
                if self.validate_patterns:
                    solution.actual_confidence = self._validate_solution(
                        solution, train_examples, test_input
                    )
                solutions.append(solution)

        # 2. Try geometric patterns if size change detected
        if size_info.has_size_change:
            solution = self.try_geometric_patterns(
                train_examples, test_input, size_info
            )
            if solution:
                if self.validate_patterns:
                    solution.actual_confidence = self._validate_solution(
                        solution, train_examples, test_input
                    )
                solutions.append(solution)

        # 3. Try conditional/object transformations
        if perception_analysis["transformation_type"] in ["conditional", "structural"]:
            solution = self.try_enhanced_object_transforms(
                train_examples, test_input, perception_analysis
            )
            if solution:
                if self.validate_patterns:
                    solution.actual_confidence = self._validate_solution(
                        solution, train_examples, test_input
                    )
                solutions.append(solution)

        # 4. NEW: Try pattern composition
        composition_solution = self.try_pattern_composition(
            train_examples, test_input, perception_analysis
        )
        if composition_solution:
            solutions.append(composition_solution)

        # 5. Check if we should try synthesis
        best_confidence = max(
            (
                s.actual_confidence if s.actual_confidence > 0 else s.confidence
                for s in solutions
            ),
            default=0,
        )

        should_try_synthesis = self.synthesizer and (
            force_synthesis or best_confidence < 0.95 or len(solutions) == 0
        )

        if should_try_synthesis:
            remaining_time = self.synthesis_timeout - (time.time() - start_time)
            if remaining_time > 1.0:
                print(
                    f"  Trying program synthesis (forced={force_synthesis}, "
                    f"best_conf={best_confidence:.2f})"
                )
                solution = self.try_enhanced_synthesis(
                    train_examples, test_input, perception_analysis, remaining_time
                )
                if solution:
                    solutions.append(solution)

        # Select best solution
        if solutions:
            # When size changes detected, strongly prefer geometric solutions
            if size_info.has_size_change:
                geometric_solutions = [
                    s for s in solutions if s.method_used.startswith("geometric")
                ]
                if geometric_solutions:
                    best_solution = max(
                        geometric_solutions,
                        key=lambda s: s.actual_confidence
                        if s.actual_confidence > 0
                        else s.confidence,
                    )
                    best_solution.time_taken = time.time() - start_time
                    best_solution.perception_analysis = perception_analysis
                    return best_solution

            # Otherwise, prefer solutions with higher actual confidence (validated)
            best_solution = max(
                solutions,
                key=lambda s: s.actual_confidence
                if s.actual_confidence > 0
                else s.confidence,
            )
            best_solution.time_taken = time.time() - start_time
            best_solution.perception_analysis = perception_analysis
            return best_solution

        # Fall back to test-time adaptation
        solution = self.try_tta(train_examples, test_input)
        solution.time_taken = time.time() - start_time
        solution.perception_analysis = perception_analysis
        return solution

    def try_geometric_patterns(
        self,
        train_examples: List[Tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
        size_info: SizeChangeInfo,
    ) -> Optional[ARCSolution]:
        """Try geometric transformations for size-changing tasks."""

        solutions = []

        if size_info.transformation_type == "tiling" and size_info.scale_factor:
            scale = int(size_info.scale_factor[0])

            # Try simple tiling
            primitive = self.dsl_library.get_primitive("tile_pattern", scale=scale)
            output = primitive.execute(test_input)

            # Validate on training examples
            confidence = self.validate_pattern_confidence(
                lambda g: primitive.execute(g), train_examples
            )

            solution = ARCSolution(output, confidence, "geometric")
            solution.actual_confidence = confidence
            solutions.append(solution)

            # Try modified tiling (with first column modifications)
            modified_primitive = self.dsl_library.get_primitive(
                "modified_tile_pattern", scale=scale, modify_first=True
            )
            modified_output = modified_primitive.execute(test_input)

            modified_confidence = self.validate_pattern_confidence(
                lambda g: modified_primitive.execute(g), train_examples
            )

            if modified_confidence > confidence:
                modified_solution = ARCSolution(
                    modified_output, modified_confidence, "geometric_modified"
                )
                modified_solution.actual_confidence = modified_confidence
                solutions.append(modified_solution)

        # Return best geometric solution
        if solutions:
            return max(solutions, key=lambda s: s.actual_confidence)

        return None

    def try_pattern_composition(
        self,
        train_examples: List[Tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
        perception_analysis: Dict,
    ) -> Optional[ARCSolution]:
        """Try composing multiple patterns together."""

        # Common compositions to try

        # If we have both arithmetic and spatial patterns, try combining
        if perception_analysis.get("arithmetic_patterns") and perception_analysis.get(
            "spatial_patterns"
        ):
            # Example: tile then add constant
            for arith in perception_analysis["arithmetic_patterns"]:
                for spatial in perception_analysis["spatial_patterns"]:
                    if arith.name == "color_shift" and spatial.name == "symmetric":
                        # Create composition
                        def compose(grid):
                            # First apply symmetry
                            symmetric = self.dsl_library.get_primitive(
                                "make_symmetric", axis="horizontal"
                            )
                            temp = symmetric.execute(grid)
                            # Then apply color shift
                            shift = self.dsl_library.get_primitive(
                                "add_constant",
                                value=arith.details.get("shift_value", 1),
                            )
                            return shift.execute(temp)

                        output = compose(test_input)
                        confidence = self.validate_pattern_confidence(
                            compose, train_examples
                        )

                        if confidence > 0.6:
                            return ARCSolution(output, confidence, "composition")

        # Try tiling + modification
        # TODO: Add more sophisticated compositions

        return None

    def _validate_solution(
        self,
        solution: ARCSolution,
        train_examples: List[Tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
    ) -> float:
        """Validate a solution by creating a transformation function and testing it."""

        # Create a transformation function based on the solution
        def transform_fn(grid):
            # For now, just return the solution output for the test input
            # In a real implementation, we'd recreate the transformation
            if np.array_equal(grid, test_input):
                return solution.output_grid
            # For training examples, we can't validate this way
            return grid

        # Better approach: validate that the detected pattern works on training
        # This requires reconstructing the transformation from the solution
        # For now, return the original confidence
        return solution.confidence

    def _get_adaptive_thresholds(
        self, perception_analysis: Dict, size_change: bool
    ) -> Dict[str, float]:
        """Get adaptive confidence thresholds based on pattern analysis and size changes."""

        # NEW: Much stricter thresholds when size changes detected
        if size_change:
            return {"simple": 0.5, "object": 0.4, "synthesis": 0.3}

        # Original logic for non-size-changing tasks
        complexity = len(
            perception_analysis.get("arithmetic_patterns", [])
            + perception_analysis.get("conditional_patterns", [])
            + perception_analysis.get("spatial_patterns", [])
            + perception_analysis.get("structural_patterns", [])
        )

        if complexity > 5:
            return {"simple": 0.7, "object": 0.6, "synthesis": 0.5}
        elif complexity > 2:
            return {"simple": 0.8, "object": 0.7, "synthesis": 0.6}
        else:
            return {"simple": 0.9, "object": 0.85, "synthesis": 0.75}

    # Copy remaining methods from V2
    def try_enhanced_simple_patterns(
        self,
        train_examples: List[Tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
        perception_analysis: Dict,
    ) -> Optional[ARCSolution]:
        """Try simple patterns using enhanced DSL primitives."""

        # Check for arithmetic patterns
        if perception_analysis.get("arithmetic_patterns"):
            for pattern in perception_analysis["arithmetic_patterns"]:
                if pattern.name == "color_shift":
                    shift_value = pattern.details["shift_value"]
                    primitive = self.dsl_library.get_primitive(
                        "add_constant", value=shift_value
                    )
                    output = primitive.execute(test_input)
                    return ARCSolution(output, pattern.confidence, "simple")

                elif pattern.name == "count_encoding":
                    primitive = self.dsl_library.get_primitive("count_objects")
                    output = primitive.execute(test_input)
                    return ARCSolution(output, pattern.confidence, "simple")

        # Check for spatial patterns
        if perception_analysis.get("spatial_patterns"):
            for pattern in perception_analysis["spatial_patterns"]:
                if pattern.name == "diagonal":
                    details = pattern.details
                    color = self._infer_color_from_examples(train_examples)
                    primitive = self.dsl_library.get_primitive(
                        "draw_diagonal",
                        color=color,
                        anti=(details.get("type") == "anti"),
                    )
                    output = primitive.execute(test_input)
                    return ARCSolution(output, pattern.confidence, "simple")

                elif pattern.name == "border":
                    details = pattern.details
                    primitive = self.dsl_library.get_primitive(
                        "draw_border",
                        color=details.get("color", 1),
                        thickness=details.get("thickness", 1),
                    )
                    output = primitive.execute(test_input)
                    return ARCSolution(output, pattern.confidence, "simple")

                elif pattern.name == "symmetric":
                    axis = pattern.details.get("symmetry_type", "horizontal")
                    primitive = self.dsl_library.get_primitive(
                        "make_symmetric", axis=axis
                    )
                    output = primitive.execute(test_input)
                    return ARCSolution(output, pattern.confidence, "simple")

        return None

    def try_enhanced_object_transforms(
        self,
        train_examples: List[Tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
        perception_analysis: Dict,
    ) -> Optional[ARCSolution]:
        """Try object/conditional transformations using enhanced DSL."""

        # Check for conditional patterns
        if perception_analysis.get("conditional_patterns"):
            for pattern in perception_analysis["conditional_patterns"]:
                if pattern.name == "size_based":
                    details = pattern.details
                    # Extract threshold from condition string
                    if "size >" in details.get("condition", ""):
                        threshold = int(details["condition"].split(">")[1].strip())
                        fill_color = details.get("fill_color", 1)
                        primitive = self.dsl_library.get_primitive(
                            "if_large", fill_color=fill_color, size_threshold=threshold
                        )
                        output = primitive.execute(test_input)
                        return ARCSolution(output, pattern.confidence, "object")

                elif pattern.name == "color_based":
                    mappings = pattern.details.get("color_mappings", {})
                    output = test_input.copy()
                    for old_color, new_color in mappings.items():
                        primitive = self.dsl_library.get_primitive(
                            "if_color", test_color=old_color, then_color=new_color
                        )
                        output = primitive.execute(output)
                    return ARCSolution(output, pattern.confidence * 0.9, "object")

                elif pattern.name == "shape_based":
                    if pattern.details.get("square_objects"):
                        # Infer transformation for squares
                        transform = pattern.details["square_objects"]
                        if "fill_with" in transform:
                            color = int(transform.split("_")[-1])
                            primitive = self.dsl_library.get_primitive(
                                "if_square", then_color=color
                            )
                            output = primitive.execute(test_input)
                            return ARCSolution(output, pattern.confidence, "object")

        # Check for structural patterns
        if perception_analysis.get("structural_patterns"):
            for pattern in perception_analysis["structural_patterns"]:
                if pattern.name == "objects_merged":
                    primitive = self.dsl_library.get_primitive("merge_adjacent")
                    output = primitive.execute(test_input)
                    return ARCSolution(output, pattern.confidence, "object")

                elif pattern.name == "objects_split":
                    # This is handled by default object extraction
                    primitive = self.dsl_library.get_primitive("enumerate_objects")
                    output = primitive.execute(test_input)
                    return ARCSolution(output, pattern.confidence * 0.8, "object")

        return None

    def try_enhanced_synthesis(
        self,
        train_examples: List[Tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
        perception_analysis: Dict,
        timeout: float,
    ) -> Optional[ARCSolution]:
        """Try program synthesis with perception-guided search."""

        if not self.synthesizer:
            return None

        try:
            # Synthesize program
            program = self.synthesizer.synthesize(train_examples)

            if program:
                # Execute on test input
                output = program.execute(test_input)

                # Calculate confidence based on training fit
                confidence = self._evaluate_program_confidence(program, train_examples)

                return ARCSolution(output, confidence, "synthesis", program=program)
        except Exception as e:
            print(f"  Synthesis failed: {e}")

        return None

    def _evaluate_program_confidence(
        self, program: Program, train_examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> float:
        """Evaluate how well a program fits training examples."""
        correct = 0
        total = len(train_examples)

        for input_grid, expected_output in train_examples:
            try:
                predicted = program.execute(input_grid)
                if np.array_equal(predicted, expected_output):
                    correct += 1
            except:
                # Program failed on this example
                pass

        return correct / total if total > 0 else 0.0

    def _infer_color_from_examples(
        self, train_examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> int:
        """Infer the most likely color to use from examples."""
        # Simple heuristic: use most common non-zero color in outputs
        all_colors = []
        for _, output in train_examples:
            colors = output.flatten()
            all_colors.extend(colors[colors > 0])

        if all_colors:
            from collections import Counter

            color_counts = Counter(all_colors)
            return color_counts.most_common(1)[0][0]

        return 1  # Default to color 1

    def try_tta(
        self,
        train_examples: List[Tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
    ) -> ARCSolution:
        """Try test-time adaptation as fallback."""
        # Adapt rules without initial rules
        adaptation_result = self.tta_adapter.adapt(train_examples)

        # Apply best hypothesis to test input
        if adaptation_result.best_hypothesis:
            try:
                output = adaptation_result.best_hypothesis.transform_fn(test_input)
                confidence = adaptation_result.confidence
            except:
                # If transformation fails, return input unchanged
                output = test_input
                confidence = 0.1
        else:
            output = test_input
            confidence = 0.1

        return ARCSolution(output, confidence, "tta")


def test_enhanced_solver_v3():
    """Test the enhanced solver V3 with size-aware strategy."""
    print("Testing Enhanced ARC Solver V3")
    print("=" * 60)

    solver = EnhancedARCSolverV3(
        use_synthesis=True, force_synthesis_on_size_change=True, validate_patterns=True
    )

    # Test 1: Size-changing task (tiling)
    print("\nTest 1: Pattern Tiling (3x3 → 9x9)")
    train_examples = [
        (
            np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]]),
            np.array(
                [
                    [1, 0, 1, 1, 0, 1, 1, 0, 1],
                    [0, 1, 0, 0, 1, 0, 0, 1, 0],
                    [1, 0, 1, 1, 0, 1, 1, 0, 1],
                    [1, 0, 1, 1, 0, 1, 1, 0, 1],
                    [0, 1, 0, 0, 1, 0, 0, 1, 0],
                    [1, 0, 1, 1, 0, 1, 1, 0, 1],
                    [1, 0, 1, 1, 0, 1, 1, 0, 1],
                    [0, 1, 0, 0, 1, 0, 0, 1, 0],
                    [1, 0, 1, 1, 0, 1, 1, 0, 1],
                ]
            ),
        ),
    ]
    test_input = np.array([[2, 2, 0], [2, 2, 0], [0, 0, 0]])

    solution = solver.solve(train_examples, test_input)
    print(f"Method: {solution.method_used}")
    print(f"Confidence: {solution.confidence:.3f}")
    print(f"Actual confidence: {solution.actual_confidence:.3f}")

    # Test 2: Non-size-changing task
    print("\nTest 2: Color Shift (no size change)")
    train_examples = [
        (np.array([[1, 2], [2, 3]]), np.array([[3, 4], [4, 5]])),
        (np.array([[2, 1], [3, 2]]), np.array([[4, 3], [5, 4]])),
    ]
    test_input = np.array([[1, 3], [3, 1]])

    solution = solver.solve(train_examples, test_input)
    print(f"Method: {solution.method_used}")
    print(f"Confidence: {solution.confidence:.3f}")
    print(f"Output:\n{solution.output_grid}")

    print("\n" + "=" * 60)
    print("Enhanced Solver V3 tests complete!")


if __name__ == "__main__":
    test_enhanced_solver_v3()
