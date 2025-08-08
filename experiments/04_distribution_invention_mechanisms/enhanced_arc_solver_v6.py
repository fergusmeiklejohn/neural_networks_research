#!/usr/bin/env python3
"""Enhanced ARC Solver V6 with improved method selection and broader position learning.

Key improvements over V5:
- Better method selection logic (no premature TTA fallback)
- Position-dependent learning applies to more task types
- Confidence-based method switching
- Program synthesis re-enabled with proper triggering
"""

from utils.imports import setup_project_paths

setup_project_paths()

import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from arc_dsl import Program
from arc_dsl_enhanced import EnhancedDSLLibrary
from enhanced_arc_solver_v3 import ARCSolution, SizeChangeInfo
from enhanced_arc_tta import EnhancedARCTestTimeAdapter
from enhanced_perception_v2 import EnhancedPerceptionV2
from learnable_pattern_modifier import LearnablePatternModifier
from object_manipulation import ObjectManipulator
from position_dependent_modifier import PositionDependentModifier, PositionRule
from program_synthesis import ProgramSynthesizer


class EnhancedARCSolverV6:
    """V6 solver with improved method selection and broader pattern learning."""

    def __init__(
        self,
        use_synthesis: bool = True,
        synthesis_timeout: float = 5.0,
        use_position_learning: bool = True,
        confidence_threshold: float = 0.85,  # Threshold to try next method
    ):
        """Initialize the enhanced solver V6.

        Args:
            use_synthesis: Whether to use program synthesis
            synthesis_timeout: Maximum time for program synthesis in seconds
            use_position_learning: Whether to use position-dependent learning
            confidence_threshold: Minimum confidence before trying next method
        """
        self.perception = EnhancedPerceptionV2()
        self.object_manipulator = ObjectManipulator()
        self.tta_adapter = EnhancedARCTestTimeAdapter()
        self.synthesizer = (
            ProgramSynthesizer(strategy="beam") if use_synthesis else None
        )
        self.synthesis_timeout = synthesis_timeout
        self.use_position_learning = use_position_learning
        self.confidence_threshold = confidence_threshold
        self.dsl_library = EnhancedDSLLibrary()
        self.position_modifier = PositionDependentModifier()
        self.simple_modifier = LearnablePatternModifier()

    def detect_size_change(
        self, train_examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> SizeChangeInfo:
        """Detect if the task involves size changes."""
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

                if scale > 1 and scale in [2, 3, 4]:
                    size_info.transformation_type = "tiling"
                elif scale > 1:
                    size_info.transformation_type = "scaling"
                elif scale < 1:
                    size_info.transformation_type = "cropping"

        return size_info

    def detect_spatial_patterns(
        self, train_examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> bool:
        """Detect if the task involves spatial/positional patterns."""

        if not train_examples:
            return False

        # Check for position-dependent changes
        for input_grid, output_grid in train_examples:
            if input_grid.shape != output_grid.shape:
                continue

            # Check if changes are position-dependent
            diff_mask = input_grid != output_grid
            if np.any(diff_mask):
                # Check for patterns in differences
                diff_positions = np.argwhere(diff_mask)

                # Check for border modifications
                h, w = input_grid.shape
                border_changes = sum(
                    1
                    for r, c in diff_positions
                    if r == 0 or r == h - 1 or c == 0 or c == w - 1
                )
                if border_changes > len(diff_positions) * 0.5:
                    return True  # Border pattern detected

                # Check for regional patterns
                # Divide into quadrants and check if changes cluster
                mid_h, mid_w = h // 2, w // 2
                quadrants = [0, 0, 0, 0]
                for r, c in diff_positions:
                    if r < mid_h and c < mid_w:
                        quadrants[0] += 1
                    elif r < mid_h and c >= mid_w:
                        quadrants[1] += 1
                    elif r >= mid_h and c < mid_w:
                        quadrants[2] += 1
                    else:
                        quadrants[3] += 1

                # If changes are concentrated in specific quadrants
                if max(quadrants) > sum(quadrants) * 0.6:
                    return True  # Regional pattern detected

        return False

    def solve(
        self,
        train_examples: List[Tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
    ) -> ARCSolution:
        """Solve an ARC task with improved method selection.

        Key improvements over V5:
        1. Always try simple patterns first
        2. Apply position learning more broadly
        3. Use confidence thresholds for method switching
        4. TTA only as last resort

        Args:
            train_examples: List of (input, output) training examples
            test_input: Test input grid to transform

        Returns:
            ARCSolution with the best prediction
        """
        start_time = time.time()

        # Detect task characteristics
        size_info = self.detect_size_change(train_examples)
        has_spatial_patterns = self.detect_spatial_patterns(train_examples)

        if size_info.has_size_change:
            print(
                f"  Size change: {size_info.transformation_type} (scale: {size_info.scale_factor})"
            )
        if has_spatial_patterns:
            print(f"  Spatial patterns detected")

        # Analyze patterns with enhanced perception
        perception_analysis = self.perception.analyze(train_examples)

        # Track best solution across all methods
        best_solution = None
        best_confidence = 0.0

        # 1. ALWAYS try simple patterns first (they work for 60%+ of tasks)
        solution = self.try_simple_patterns(
            train_examples, test_input, perception_analysis
        )
        if solution and solution.confidence > best_confidence:
            best_solution = solution
            best_confidence = solution.confidence
            print(f"  Simple patterns: {best_confidence:.2f} confidence")

        # 2. Try object/conditional transformations if confidence not high enough
        if best_confidence < self.confidence_threshold:
            solution = self.try_object_transforms(
                train_examples, test_input, perception_analysis
            )
            if solution and solution.confidence > best_confidence:
                best_solution = solution
                best_confidence = solution.confidence
                print(f"  Object transforms: {best_confidence:.2f} confidence")

        # 3. Try position-dependent learning for size changes OR spatial patterns
        if best_confidence < self.confidence_threshold and self.use_position_learning:
            if size_info.has_size_change or has_spatial_patterns:
                if (
                    size_info.has_size_change
                    and size_info.transformation_type == "tiling"
                ):
                    # Use tiling-specific position learning
                    solution = self.try_position_dependent_tiling(
                        train_examples, test_input, size_info
                    )
                else:
                    # Use general position learning
                    solution = self.try_general_position_learning(
                        train_examples, test_input
                    )

                if solution and solution.confidence > best_confidence:
                    best_solution = solution
                    best_confidence = solution.confidence
                    print(f"  Position learning: {best_confidence:.2f} confidence")

        # 4. Try simple learned modifications (V4 style)
        if best_confidence < self.confidence_threshold:
            solution = self.try_simple_learned_modifications(train_examples, test_input)
            if solution and solution.confidence > best_confidence:
                best_solution = solution
                best_confidence = solution.confidence
                print(f"  Learned modifications: {best_confidence:.2f} confidence")

        # 5. Try program synthesis if enabled and confidence still not high
        if best_confidence < self.confidence_threshold and self.synthesizer:
            remaining_time = self.synthesis_timeout - (time.time() - start_time)
            if remaining_time > 1.0:
                print(f"  Trying synthesis (current best: {best_confidence:.2f})")
                solution = self.try_synthesis(
                    train_examples, test_input, perception_analysis, remaining_time
                )
                if solution and solution.confidence > best_confidence:
                    best_solution = solution
                    best_confidence = solution.confidence
                    print(f"  Synthesis: {best_confidence:.2f} confidence")

        # 6. ONLY use TTA as absolute last resort
        if best_solution is None or best_confidence < 0.5:
            print(f"  Falling back to TTA (best so far: {best_confidence:.2f})")
            solution = self.try_tta(train_examples, test_input)
            if solution and (
                best_solution is None or solution.confidence > best_confidence
            ):
                best_solution = solution
                best_confidence = solution.confidence

        # Return best solution found
        if best_solution:
            best_solution.time_taken = time.time() - start_time
            best_solution.perception_analysis = perception_analysis
            return best_solution

        # Should never reach here, but just in case
        return ARCSolution(
            test_input,
            0.1,
            "failed",
            time_taken=time.time() - start_time,
            perception_analysis=perception_analysis,
        )

    def try_simple_patterns(
        self,
        train_examples: List[Tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
        perception_analysis: Dict,
    ) -> Optional[ARCSolution]:
        """Try simple patterns using enhanced DSL primitives."""

        best_solution = None
        best_confidence = 0.0

        # Check for arithmetic patterns
        if perception_analysis.get("arithmetic_patterns"):
            for pattern in perception_analysis["arithmetic_patterns"]:
                if pattern.name == "color_shift":
                    shift_value = pattern.details.get("shift_value", 0)
                    primitive = self.dsl_library.get_primitive(
                        "add_constant", value=shift_value
                    )
                    output = primitive.execute(test_input)

                    # Validate on training
                    confidence = self._validate_primitive(primitive, train_examples)

                    if confidence > best_confidence:
                        best_solution = ARCSolution(
                            output, confidence, "simple_arithmetic"
                        )
                        best_confidence = confidence

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

                    confidence = self._validate_primitive(primitive, train_examples)

                    if confidence > best_confidence:
                        best_solution = ARCSolution(
                            output, confidence, "simple_spatial"
                        )
                        best_confidence = confidence

                elif pattern.name == "symmetric":
                    axis = pattern.details.get("symmetry_type", "horizontal")
                    primitive = self.dsl_library.get_primitive(
                        "make_symmetric", axis=axis
                    )
                    output = primitive.execute(test_input)

                    confidence = self._validate_primitive(primitive, train_examples)

                    if confidence > best_confidence:
                        best_solution = ARCSolution(
                            output, confidence, "simple_symmetric"
                        )
                        best_confidence = confidence

        return best_solution

    def try_object_transforms(
        self,
        train_examples: List[Tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
        perception_analysis: Dict,
    ) -> Optional[ARCSolution]:
        """Try object/conditional transformations."""

        best_solution = None
        best_confidence = 0.0

        # Check for conditional patterns
        if perception_analysis.get("conditional_patterns"):
            for pattern in perception_analysis["conditional_patterns"]:
                if pattern.name == "size_based":
                    details = pattern.details
                    if "size >" in details.get("condition", ""):
                        try:
                            threshold = int(details["condition"].split(">")[1].strip())
                            fill_color = details.get("fill_color", 1)
                            primitive = self.dsl_library.get_primitive(
                                "if_large",
                                fill_color=fill_color,
                                size_threshold=threshold,
                            )
                            output = primitive.execute(test_input)

                            confidence = self._validate_primitive(
                                primitive, train_examples
                            )

                            if confidence > best_confidence:
                                best_solution = ARCSolution(
                                    output, confidence, "object_conditional"
                                )
                                best_confidence = confidence
                        except:
                            pass

        # Check for structural patterns
        if perception_analysis.get("structural_patterns"):
            for pattern in perception_analysis["structural_patterns"]:
                if pattern.name == "objects_merged":
                    primitive = self.dsl_library.get_primitive("merge_adjacent")
                    output = primitive.execute(test_input)

                    confidence = self._validate_primitive(primitive, train_examples)

                    if confidence > best_confidence:
                        best_solution = ARCSolution(
                            output, confidence, "object_structural"
                        )
                        best_confidence = confidence

        return best_solution

    def try_position_dependent_tiling(
        self,
        train_examples: List[Tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
        size_info: SizeChangeInfo,
    ) -> Optional[ARCSolution]:
        """Try tiling with position-dependent modifications."""

        if size_info.scale_factor:
            scale = int(size_info.scale_factor[0])
            tile_size = (test_input.shape[0], test_input.shape[1])

            # Apply simple tiling first
            primitive = self.dsl_library.get_primitive("tile_pattern", scale=scale)
            base_output = primitive.execute(test_input)

            # Learn position-dependent rules
            learning_examples = []
            for input_grid, expected_output in train_examples:
                base = primitive.execute(input_grid)
                if base.shape == expected_output.shape:
                    learning_examples.append((input_grid, base, expected_output))

            if learning_examples:
                position_rules = self.position_modifier.learn_from_multiple_examples(
                    learning_examples, tile_size=tile_size
                )

                if position_rules:
                    # Apply position rules
                    modified_output = self.position_modifier.apply_position_rules(
                        base_output, position_rules, tile_size=tile_size
                    )

                    # Validate
                    validation_score = self._validate_position_pattern(
                        train_examples, primitive.execute, position_rules, tile_size
                    )

                    return ARCSolution(
                        modified_output,
                        validation_score,
                        "position_tiling",
                    )

        return None

    def try_general_position_learning(
        self,
        train_examples: List[Tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
    ) -> Optional[ARCSolution]:
        """Try general position-dependent learning for non-tiling patterns."""

        # For same-size transformations with position-dependent changes
        if train_examples:
            # Check if input/output sizes match
            if train_examples[0][0].shape == train_examples[0][1].shape:
                # Learn position-dependent modifications
                learning_examples = []
                for input_grid, output_grid in train_examples:
                    # Use input as base for same-size transformations
                    learning_examples.append((input_grid, input_grid, output_grid))

                # Detect appropriate tile size (could be 1x1 for pixel-level)
                h, w = train_examples[0][0].shape
                # Try to find natural tile size
                tile_size = None
                for size in [1, 2, 3, 4, 5]:
                    if h % size == 0 and w % size == 0:
                        tile_size = (size, size)
                        break

                if tile_size and learning_examples:
                    position_rules = (
                        self.position_modifier.learn_from_multiple_examples(
                            learning_examples, tile_size=tile_size
                        )
                    )

                    if position_rules:
                        # Apply to test input
                        modified_output = self.position_modifier.apply_position_rules(
                            test_input, position_rules, tile_size=tile_size
                        )

                        # Validate
                        confidence = self._validate_general_position(
                            train_examples, position_rules, tile_size
                        )

                        return ARCSolution(
                            modified_output, confidence, "position_general"
                        )

        return None

    def try_simple_learned_modifications(
        self,
        train_examples: List[Tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
    ) -> Optional[ARCSolution]:
        """Try simple learned modifications (V4 style)."""

        # For transformations where patterns can be learned
        if train_examples and train_examples[0][0].shape == train_examples[0][1].shape:
            # Learn modifications
            learning_examples = []
            for input_grid, output_grid in train_examples:
                learning_examples.append((input_grid, input_grid, output_grid))

            rules = self.simple_modifier.learn_from_examples(
                learning_examples, "pattern"
            )

            if rules:
                # Apply to test
                modified = self.simple_modifier.apply_modifications(test_input, rules)

                # Validate
                confidence = self._validate_simple_learned(train_examples, rules)

                return ARCSolution(modified, confidence, "learned_simple")

        return None

    def try_synthesis(
        self,
        train_examples: List[Tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
        perception_analysis: Dict,
        timeout: float,
    ) -> Optional[ARCSolution]:
        """Try program synthesis."""

        if not self.synthesizer:
            return None

        try:
            program = self.synthesizer.synthesize(train_examples)
            if program:
                output = program.execute(test_input)
                confidence = self._evaluate_program_confidence(program, train_examples)
                return ARCSolution(output, confidence, "synthesis", program=program)
        except Exception as e:
            print(f"    Synthesis error: {e}")

        return None

    def try_tta(
        self,
        train_examples: List[Tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
    ) -> ARCSolution:
        """Try test-time adaptation as last resort."""
        adaptation_result = self.tta_adapter.adapt(train_examples)

        if adaptation_result.best_hypothesis:
            try:
                output = adaptation_result.best_hypothesis.transform_fn(test_input)
                confidence = adaptation_result.confidence
            except:
                output = test_input
                confidence = 0.1
        else:
            output = test_input
            confidence = 0.1

        return ARCSolution(output, confidence, "tta")

    def _validate_primitive(
        self, primitive, train_examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> float:
        """Validate a primitive on training examples."""
        if not train_examples:
            return 0.5

        correct = 0
        total = len(train_examples)

        for input_grid, expected_output in train_examples:
            try:
                predicted = primitive.execute(input_grid)
                if np.array_equal(predicted, expected_output):
                    correct += 1
                elif predicted.shape == expected_output.shape:
                    accuracy = (
                        np.sum(predicted == expected_output) / expected_output.size
                    )
                    correct += accuracy * 0.5
            except:
                pass

        return correct / total if total > 0 else 0.0

    def _validate_position_pattern(
        self,
        train_examples: List[Tuple[np.ndarray, np.ndarray]],
        base_transform: callable,
        position_rules: List[PositionRule],
        tile_size: Tuple[int, int],
    ) -> float:
        """Validate position-dependent pattern on training examples."""
        if not train_examples:
            return 0.0

        correct = 0
        total = len(train_examples)

        for input_grid, expected_output in train_examples:
            try:
                base_output = base_transform(input_grid)
                modified = self.position_modifier.apply_position_rules(
                    base_output,
                    position_rules,
                    tile_size=(input_grid.shape[0], input_grid.shape[1]),
                )

                if np.array_equal(modified, expected_output):
                    correct += 1
                elif modified.shape == expected_output.shape:
                    accuracy = (
                        np.sum(modified == expected_output) / expected_output.size
                    )
                    correct += accuracy * 0.8
            except:
                pass

        return correct / total if total > 0 else 0.0

    def _validate_general_position(
        self,
        train_examples: List[Tuple[np.ndarray, np.ndarray]],
        position_rules: List[PositionRule],
        tile_size: Tuple[int, int],
    ) -> float:
        """Validate general position rules."""
        if not train_examples:
            return 0.0

        correct = 0
        total = len(train_examples)

        for input_grid, expected_output in train_examples:
            try:
                modified = self.position_modifier.apply_position_rules(
                    input_grid, position_rules, tile_size=tile_size
                )

                if np.array_equal(modified, expected_output):
                    correct += 1
                elif modified.shape == expected_output.shape:
                    accuracy = (
                        np.sum(modified == expected_output) / expected_output.size
                    )
                    correct += accuracy * 0.7
            except:
                pass

        return correct / total if total > 0 else 0.0

    def _validate_simple_learned(
        self,
        train_examples: List[Tuple[np.ndarray, np.ndarray]],
        rules,
    ) -> float:
        """Validate simple learned rules."""
        if not train_examples:
            return 0.0

        correct = 0
        total = len(train_examples)

        for input_grid, expected_output in train_examples:
            try:
                modified = self.simple_modifier.apply_modifications(input_grid, rules)

                if np.array_equal(modified, expected_output):
                    correct += 1
                elif modified.shape == expected_output.shape:
                    accuracy = (
                        np.sum(modified == expected_output) / expected_output.size
                    )
                    correct += accuracy * 0.6
            except:
                pass

        return correct / total if total > 0 else 0.0

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
                pass

        return correct / total if total > 0 else 0.0

    def _infer_color_from_examples(
        self, train_examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> int:
        """Infer the most likely color to use from examples."""
        all_colors = []
        for _, output in train_examples:
            colors = output.flatten()
            all_colors.extend(colors[colors > 0])

        if all_colors:
            from collections import Counter

            color_counts = Counter(all_colors)
            return color_counts.most_common(1)[0][0]

        return 1


def test_enhanced_solver_v6():
    """Test the enhanced solver V6."""
    print("Testing Enhanced ARC Solver V6")
    print("=" * 60)

    solver = EnhancedARCSolverV6(
        use_synthesis=True,
        use_position_learning=True,
        confidence_threshold=0.85,
    )

    # Test: Simple pattern
    print("\nTest 1: Simple Color Shift")
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
    print("V6 test complete!")


if __name__ == "__main__":
    test_enhanced_solver_v6()
