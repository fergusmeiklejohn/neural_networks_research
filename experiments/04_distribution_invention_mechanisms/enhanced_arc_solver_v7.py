#!/usr/bin/env python3
"""Enhanced ARC Solver V7 with integrated program synthesis.

Key improvements over V6:
- Program synthesis properly integrated with perception hints
- Synthesis triggered strategically based on confidence
- Better synthesis timeout management
- Synthesis uses enhanced DSL primitives
"""

from utils.imports import setup_project_paths

setup_project_paths()

import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from arc_dsl_enhanced import EnhancedDSLLibrary
from enhanced_arc_solver_v3 import ARCSolution, SizeChangeInfo
from enhanced_arc_tta import EnhancedARCTestTimeAdapter
from enhanced_perception_v2 import EnhancedPerceptionV2
from enhanced_program_synthesis import EnhancedProgramSynthesizer, Program
from learnable_pattern_modifier import LearnablePatternModifier
from object_manipulation import ObjectManipulator
from position_dependent_modifier import PositionDependentModifier


class EnhancedARCSolverV7:
    """V7 solver with integrated program synthesis."""

    def __init__(
        self,
        use_synthesis: bool = True,
        synthesis_timeout: float = 8.0,
        synthesis_confidence_threshold: float = 0.75,  # When to trigger synthesis
        use_position_learning: bool = True,
        confidence_threshold: float = 0.85,  # When to try next method
    ):
        """Initialize the enhanced solver V7.

        Args:
            use_synthesis: Whether to use program synthesis
            synthesis_timeout: Maximum time for program synthesis in seconds
            synthesis_confidence_threshold: Min confidence before trying synthesis
            use_position_learning: Whether to use position-dependent learning
            confidence_threshold: Minimum confidence before trying next method
        """
        self.perception = EnhancedPerceptionV2()
        self.object_manipulator = ObjectManipulator()
        self.tta_adapter = EnhancedARCTestTimeAdapter()
        self.use_synthesis = use_synthesis
        self.synthesizer = (
            EnhancedProgramSynthesizer(
                beam_width=30,
                max_depth=4,
                use_perception_hints=True,
                timeout=synthesis_timeout,
            )
            if use_synthesis
            else None
        )
        self.synthesis_timeout = synthesis_timeout
        self.synthesis_confidence_threshold = synthesis_confidence_threshold
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

        for input_grid, output_grid in train_examples:
            if input_grid.shape != output_grid.shape:
                continue

            diff_mask = input_grid != output_grid
            if np.any(diff_mask):
                diff_positions = np.argwhere(diff_mask)

                # Check for border modifications
                h, w = input_grid.shape
                border_changes = sum(
                    1
                    for r, c in diff_positions
                    if r == 0 or r == h - 1 or c == 0 or c == w - 1
                )
                if border_changes > len(diff_positions) * 0.5:
                    return True

                # Check for regional patterns
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

                if max(quadrants) > sum(quadrants) * 0.6:
                    return True

        return False

    def solve(
        self,
        train_examples: List[Tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
    ) -> ARCSolution:
        """Solve an ARC task with integrated program synthesis.

        Key improvements over V6:
        1. Synthesis triggered based on confidence thresholds
        2. Synthesis uses perception hints to guide search
        3. Better timeout management
        4. Synthesis can be tried multiple times with different hints

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
        synthesis_attempted = False

        # 1. Try simple patterns first
        solution = self.try_simple_patterns(
            train_examples, test_input, perception_analysis
        )
        if solution and solution.confidence > best_confidence:
            best_solution = solution
            best_confidence = solution.confidence
            print(f"  Simple patterns: {best_confidence:.2f} confidence")

        # 2. Try early synthesis if confidence not high enough and synthesis enabled
        if (
            best_confidence < self.synthesis_confidence_threshold
            and self.use_synthesis
            and not synthesis_attempted
        ):
            remaining_time = (
                self.synthesis_timeout * 0.4
            )  # Use 40% of time for first attempt
            elapsed = time.time() - start_time

            if elapsed < self.synthesis_timeout * 0.5:  # Only if we have time
                print(f"  Early synthesis attempt (confidence: {best_confidence:.2f})")
                solution = self.try_synthesis_with_hints(
                    train_examples,
                    test_input,
                    perception_analysis,
                    remaining_time,
                    "early",
                )
                synthesis_attempted = True

                if solution and solution.confidence > best_confidence:
                    best_solution = solution
                    best_confidence = solution.confidence
                    print(
                        f"  Synthesis found solution: {best_confidence:.2f} confidence"
                    )

                    # If synthesis found perfect solution, return early
                    if best_confidence >= 0.99:
                        best_solution.time_taken = time.time() - start_time
                        best_solution.perception_analysis = perception_analysis
                        return best_solution

        # 3. Try object/conditional transformations
        if best_confidence < self.confidence_threshold:
            solution = self.try_object_transforms(
                train_examples, test_input, perception_analysis
            )
            if solution and solution.confidence > best_confidence:
                best_solution = solution
                best_confidence = solution.confidence
                print(f"  Object transforms: {best_confidence:.2f} confidence")

        # 4. Try position-dependent learning
        if best_confidence < self.confidence_threshold and self.use_position_learning:
            if size_info.has_size_change or has_spatial_patterns:
                if (
                    size_info.has_size_change
                    and size_info.transformation_type == "tiling"
                ):
                    solution = self.try_position_dependent_tiling(
                        train_examples, test_input, size_info
                    )
                else:
                    solution = self.try_general_position_learning(
                        train_examples, test_input
                    )

                if solution and solution.confidence > best_confidence:
                    best_solution = solution
                    best_confidence = solution.confidence
                    print(f"  Position learning: {best_confidence:.2f} confidence")

        # 5. Try simple learned modifications
        if best_confidence < self.confidence_threshold:
            solution = self.try_simple_learned_modifications(train_examples, test_input)
            if solution and solution.confidence > best_confidence:
                best_solution = solution
                best_confidence = solution.confidence
                print(f"  Learned modifications: {best_confidence:.2f} confidence")

        # 6. Try synthesis again with remaining time if still not confident
        if (
            best_confidence < self.confidence_threshold
            and self.use_synthesis
            and not synthesis_attempted
        ):
            elapsed = time.time() - start_time
            remaining_time = self.synthesis_timeout - elapsed

            if remaining_time > 2.0:  # Need at least 2 seconds
                print(f"  Final synthesis attempt (confidence: {best_confidence:.2f})")
                solution = self.try_synthesis_with_hints(
                    train_examples,
                    test_input,
                    perception_analysis,
                    remaining_time,
                    "final",
                )

                if solution and solution.confidence > best_confidence:
                    best_solution = solution
                    best_confidence = solution.confidence
                    print(f"  Synthesis improved to: {best_confidence:.2f}")

        # 7. Use TTA as absolute last resort
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

        # Should never reach here
        return ARCSolution(
            test_input,
            0.1,
            "failed",
            time_taken=time.time() - start_time,
            perception_analysis=perception_analysis,
        )

    def try_synthesis_with_hints(
        self,
        train_examples: List[Tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
        perception_analysis: Dict,
        timeout: float,
        phase: str = "early",
    ) -> Optional[ARCSolution]:
        """Try program synthesis with perception hints."""

        if not self.synthesizer:
            return None

        try:
            # Adjust synthesizer timeout
            self.synthesizer.timeout = timeout

            # Synthesize program
            program = self.synthesizer.synthesize(train_examples, perception_analysis)

            if program:
                # Execute on test input
                output = program.execute(test_input)

                # Calculate confidence
                confidence = self._evaluate_program_confidence(program, train_examples)

                # Create solution
                solution = ARCSolution(
                    output, confidence, f"synthesis_{phase}", program=program
                )

                return solution

        except Exception as e:
            print(f"    Synthesis error: {e}")

        return None

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

                    confidence = self._validate_primitive(primitive, train_examples)

                    if confidence > best_confidence:
                        best_solution = ARCSolution(
                            output, confidence, "simple_arithmetic"
                        )
                        best_confidence = confidence

        # Check for spatial patterns
        if perception_analysis.get("spatial_patterns"):
            for pattern in perception_analysis["spatial_patterns"]:
                if pattern.name == "symmetric":
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

            primitive = self.dsl_library.get_primitive("tile_pattern", scale=scale)
            base_output = primitive.execute(test_input)

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
                    modified_output = self.position_modifier.apply_position_rules(
                        base_output, position_rules, tile_size=tile_size
                    )

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
        """Try general position-dependent learning."""

        if train_examples and train_examples[0][0].shape == train_examples[0][1].shape:
            learning_examples = []
            for input_grid, output_grid in train_examples:
                learning_examples.append((input_grid, input_grid, output_grid))

            h, w = train_examples[0][0].shape
            tile_size = None
            for size in [1, 2, 3, 4, 5]:
                if h % size == 0 and w % size == 0:
                    tile_size = (size, size)
                    break

            if tile_size and learning_examples:
                position_rules = self.position_modifier.learn_from_multiple_examples(
                    learning_examples, tile_size=tile_size
                )

                if position_rules:
                    modified_output = self.position_modifier.apply_position_rules(
                        test_input, position_rules, tile_size=tile_size
                    )

                    confidence = self._validate_general_position(
                        train_examples, position_rules, tile_size
                    )

                    return ARCSolution(modified_output, confidence, "position_general")

        return None

    def try_simple_learned_modifications(
        self,
        train_examples: List[Tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
    ) -> Optional[ARCSolution]:
        """Try simple learned modifications."""

        if train_examples and train_examples[0][0].shape == train_examples[0][1].shape:
            learning_examples = []
            for input_grid, output_grid in train_examples:
                learning_examples.append((input_grid, input_grid, output_grid))

            rules = self.simple_modifier.learn_from_examples(
                learning_examples, "pattern"
            )

            if rules:
                modified = self.simple_modifier.apply_modifications(test_input, rules)
                confidence = self._validate_simple_learned(train_examples, rules)

                return ARCSolution(modified, confidence, "learned_simple")

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
                elif predicted.shape == expected_output.shape:
                    accuracy = (
                        np.sum(predicted == expected_output) / expected_output.size
                    )
                    correct += accuracy * 0.8
            except:
                pass

        return correct / total if total > 0 else 0.0

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
        position_rules,
        tile_size: Tuple[int, int],
    ) -> float:
        """Validate position-dependent pattern."""
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
        position_rules,
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


def test_enhanced_solver_v7():
    """Test the enhanced solver V7 with synthesis."""
    print("Testing Enhanced ARC Solver V7 with Program Synthesis")
    print("=" * 60)

    solver = EnhancedARCSolverV7(
        use_synthesis=True,
        synthesis_timeout=5.0,
        synthesis_confidence_threshold=0.75,
        use_position_learning=True,
        confidence_threshold=0.85,
    )

    # Test: Simple color shift (synthesis should solve this)
    print("\nTest: Color Shift (+2)")
    train_examples = [
        (np.array([[1, 2], [2, 3]]), np.array([[3, 4], [4, 5]])),
        (np.array([[0, 1], [1, 2]]), np.array([[2, 3], [3, 4]])),
    ]
    test_input = np.array([[2, 3], [3, 4]])

    solution = solver.solve(train_examples, test_input)
    print(f"Method: {solution.method_used}")
    print(f"Confidence: {solution.confidence:.3f}")
    print(f"Output:\n{solution.output_grid}")

    expected = np.array([[4, 5], [5, 6]])
    if np.array_equal(solution.output_grid, expected):
        print("✅ Correct!")
    else:
        print(f"❌ Expected:\n{expected}")

    print("\n" + "=" * 60)
    print("V7 test complete!")


if __name__ == "__main__":
    test_enhanced_solver_v7()
