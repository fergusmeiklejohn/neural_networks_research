#!/usr/bin/env python3
"""Enhanced ARC Solver V5 with position-dependent pattern modifications.

Key improvements over V4:
- Position-dependent rule learning (different tiles get different modifications)
- Tile-specific modification detection
- Better handling of complex patterns like 007bbfb7
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
from object_manipulation import ObjectManipulator
from position_dependent_modifier import PositionDependentModifier, PositionRule
from program_synthesis import ProgramSynthesizer


class EnhancedARCSolverV5:
    """V5 solver with position-dependent pattern modifications."""

    def __init__(
        self,
        use_synthesis: bool = True,
        synthesis_timeout: float = 10.0,
        use_position_learning: bool = True,
    ):
        """Initialize the enhanced solver V5.

        Args:
            use_synthesis: Whether to use program synthesis
            synthesis_timeout: Maximum time for program synthesis in seconds
            use_position_learning: Whether to use position-dependent learning
        """
        self.perception = EnhancedPerceptionV2()
        self.object_manipulator = ObjectManipulator()
        self.tta_adapter = EnhancedARCTestTimeAdapter()
        self.synthesizer = (
            ProgramSynthesizer(strategy="beam") if use_synthesis else None
        )
        self.synthesis_timeout = synthesis_timeout
        self.use_position_learning = use_position_learning
        self.dsl_library = EnhancedDSLLibrary()
        self.position_modifier = PositionDependentModifier()

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

                # Determine transformation type
                if scale > 1:
                    if scale in [2, 3, 4]:
                        size_info.transformation_type = "tiling"
                    else:
                        size_info.transformation_type = "scaling"
                elif scale < 1:
                    size_info.transformation_type = "cropping"

        return size_info

    def solve(
        self,
        train_examples: List[Tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
    ) -> ARCSolution:
        """Solve an ARC task using position-dependent pattern modifications.

        Key improvements over V4:
        1. Learn position-dependent modifications (different for each tile)
        2. Better handle complex patterns like 007bbfb7
        3. More accurate pattern application

        Args:
            train_examples: List of (input, output) training examples
            test_input: Test input grid to transform

        Returns:
            ARCSolution with the best prediction
        """
        start_time = time.time()

        # Detect size changes
        size_info = self.detect_size_change(train_examples)

        if size_info.has_size_change:
            print(
                f"  Size change detected: {size_info.transformation_type} "
                f"(scale: {size_info.scale_factor})"
            )

        # Analyze patterns with enhanced perception
        perception_analysis = self.perception.analyze(train_examples)

        # Collect all solutions
        solutions = []

        # 1. Try simple patterns
        if perception_analysis["transformation_type"] in ["arithmetic", "spatial"]:
            solution = self.try_simple_patterns(
                train_examples, test_input, perception_analysis
            )
            if solution:
                solutions.append(solution)

        # 2. Try geometric patterns with position-dependent learning
        if size_info.has_size_change and self.use_position_learning:
            solution = self.try_position_dependent_tiling(
                train_examples, test_input, size_info
            )
            if solution:
                solutions.append(solution)
        elif size_info.has_size_change:
            # Fall back to simple geometric if position learning disabled
            solution = self.try_simple_geometric(train_examples, test_input, size_info)
            if solution:
                solutions.append(solution)

        # 3. Try conditional/object transformations
        if perception_analysis["transformation_type"] in ["conditional", "structural"]:
            solution = self.try_object_transforms(
                train_examples, test_input, perception_analysis
            )
            if solution:
                solutions.append(solution)

        # 4. Try synthesis if needed
        best_confidence = max(
            (s.confidence for s in solutions),
            default=0,
        )

        should_try_synthesis = self.synthesizer and (
            size_info.has_size_change or best_confidence < 0.95 or len(solutions) == 0
        )

        if should_try_synthesis:
            remaining_time = self.synthesis_timeout - (time.time() - start_time)
            if remaining_time > 1.0:
                print(f"  Trying program synthesis (confidence={best_confidence:.2f})")
                solution = self.try_synthesis(
                    train_examples, test_input, perception_analysis, remaining_time
                )
                if solution:
                    solutions.append(solution)

        # Select best solution
        if solutions:
            # Strongly prefer position-dependent solutions for size-changing tasks
            if size_info.has_size_change:
                position_solutions = [
                    s for s in solutions if "position" in s.method_used
                ]
                if position_solutions:
                    best_solution = max(position_solutions, key=lambda s: s.confidence)
                    best_solution.time_taken = time.time() - start_time
                    best_solution.perception_analysis = perception_analysis
                    return best_solution

            # Otherwise, pick highest confidence
            best_solution = max(solutions, key=lambda s: s.confidence)
            best_solution.time_taken = time.time() - start_time
            best_solution.perception_analysis = perception_analysis
            return best_solution

        # Fall back to test-time adaptation
        solution = self.try_tta(train_examples, test_input)
        solution.time_taken = time.time() - start_time
        solution.perception_analysis = perception_analysis
        return solution

    def try_position_dependent_tiling(
        self,
        train_examples: List[Tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
        size_info: SizeChangeInfo,
    ) -> Optional[ARCSolution]:
        """Try tiling with position-dependent modifications."""

        if size_info.transformation_type == "tiling" and size_info.scale_factor:
            scale = int(size_info.scale_factor[0])
            tile_size = (test_input.shape[0], test_input.shape[1])  # Size of input

            # First apply simple tiling
            primitive = self.dsl_library.get_primitive("tile_pattern", scale=scale)
            base_output = primitive.execute(test_input)

            print(
                f"  Learning position-dependent modifications from {len(train_examples)} examples..."
            )

            # Learn position-dependent rules from training examples
            learning_examples = []
            for input_grid, expected_output in train_examples:
                base = primitive.execute(input_grid)
                if base.shape == expected_output.shape:
                    learning_examples.append((input_grid, base, expected_output))

            if learning_examples:
                # Learn position rules
                position_rules = self.position_modifier.learn_from_multiple_examples(
                    learning_examples, tile_size=tile_size
                )

                if position_rules:
                    print(f"  Learned {len(position_rules)} position-dependent rules:")

                    # Show unique patterns
                    patterns = {}
                    for rule in position_rules:
                        if rule.details and "pattern" in rule.details:
                            pattern = rule.details["pattern"]
                            if pattern not in patterns:
                                patterns[pattern] = []
                            if rule.details.get("rows"):
                                patterns[pattern].extend(rule.details["rows"])

                    for pattern, rows in patterns.items():
                        unique_rows = sorted(set(rows))
                        if unique_rows:
                            print(f"    - Rows {unique_rows}: {pattern}")

                    # Apply position rules to base output
                    modified_output = self.position_modifier.apply_position_rules(
                        base_output, position_rules, tile_size=tile_size
                    )

                    # Validate on training examples
                    validation_score = self._validate_position_pattern(
                        train_examples, primitive.execute, position_rules, tile_size
                    )

                    return ARCSolution(
                        modified_output,
                        validation_score,
                        "position_dependent_tiling",
                    )
                else:
                    # No position modifications needed, use simple tiling
                    return ARCSolution(base_output, 0.7, "simple_tiling")

        return None

    def try_simple_geometric(
        self,
        train_examples: List[Tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
        size_info: SizeChangeInfo,
    ) -> Optional[ARCSolution]:
        """Try simple geometric transformations without position learning."""

        if size_info.transformation_type == "tiling" and size_info.scale_factor:
            scale = int(size_info.scale_factor[0])
            primitive = self.dsl_library.get_primitive("tile_pattern", scale=scale)
            output = primitive.execute(test_input)
            return ARCSolution(output, 0.6, "geometric_simple")

        return None

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
                # Apply base transform
                base_output = base_transform(input_grid)
                # Apply position rules
                modified = self.position_modifier.apply_position_rules(
                    base_output,
                    position_rules,
                    tile_size=(input_grid.shape[0], input_grid.shape[1]),
                )

                if np.array_equal(modified, expected_output):
                    correct += 1
                elif modified.shape == expected_output.shape:
                    # Partial credit for close matches
                    matching = np.sum(modified == expected_output)
                    total_pixels = modified.size
                    if total_pixels > 0:
                        accuracy = matching / total_pixels
                        correct += accuracy * 0.8  # Weight partial matches
            except Exception as e:
                print(f"    Validation error: {e}")

        return correct / total if total > 0 else 0.0

    # Simplified versions of other methods
    def try_simple_patterns(
        self,
        train_examples: List[Tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
        perception_analysis: Dict,
    ) -> Optional[ARCSolution]:
        """Try simple patterns using enhanced DSL primitives."""

        if perception_analysis.get("arithmetic_patterns"):
            for pattern in perception_analysis["arithmetic_patterns"]:
                if pattern.name == "color_shift":
                    shift_value = pattern.details["shift_value"]
                    primitive = self.dsl_library.get_primitive(
                        "add_constant", value=shift_value
                    )
                    output = primitive.execute(test_input)
                    return ARCSolution(output, pattern.confidence, "simple")

        return None

    def try_object_transforms(
        self,
        train_examples: List[Tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
        perception_analysis: Dict,
    ) -> Optional[ARCSolution]:
        """Try object/conditional transformations."""

        if perception_analysis.get("conditional_patterns"):
            for pattern in perception_analysis["conditional_patterns"]:
                if pattern.name == "size_based":
                    details = pattern.details
                    if "size >" in details.get("condition", ""):
                        threshold = int(details["condition"].split(">")[1].strip())
                        fill_color = details.get("fill_color", 1)
                        primitive = self.dsl_library.get_primitive(
                            "if_large", fill_color=fill_color, size_threshold=threshold
                        )
                        output = primitive.execute(test_input)
                        return ARCSolution(output, pattern.confidence, "object")

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
                pass

        return correct / total if total > 0 else 0.0

    def try_tta(
        self,
        train_examples: List[Tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
    ) -> ARCSolution:
        """Try test-time adaptation as fallback."""
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


def test_enhanced_solver_v5():
    """Test the enhanced solver V5 with position-dependent modifications."""
    print("Testing Enhanced ARC Solver V5")
    print("=" * 60)

    solver = EnhancedARCSolverV5(
        use_synthesis=True,
        use_position_learning=True,
    )

    # Test: Pattern with position-dependent modifications
    print("\nTest: Position-Dependent Tiling")

    # Create training examples that show position-dependent pattern
    # Top rows: keep-zero-keep, Bottom row: keep-keep-zero
    train_examples = [
        (
            np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]]),
            np.array(
                [
                    [1, 0, 1, 0, 0, 0, 1, 0, 1],
                    [0, 1, 0, 0, 0, 0, 0, 1, 0],
                    [1, 0, 1, 0, 0, 0, 1, 0, 1],
                    [1, 0, 1, 0, 0, 0, 1, 0, 1],
                    [0, 1, 0, 0, 0, 0, 0, 1, 0],
                    [1, 0, 1, 0, 0, 0, 1, 0, 1],
                    [1, 0, 1, 1, 0, 1, 0, 0, 0],
                    [0, 1, 0, 0, 1, 0, 0, 0, 0],
                    [1, 0, 1, 1, 0, 1, 0, 0, 0],
                ]
            ),
        ),
    ]

    test_input = np.array([[2, 2, 0], [2, 2, 0], [0, 0, 2]])

    solution = solver.solve(train_examples, test_input)
    print(f"Method: {solution.method_used}")
    print(f"Confidence: {solution.confidence:.3f}")
    print(f"Output shape: {solution.output_grid.shape}")
    print(f"Output:\n{solution.output_grid}")

    print("\n" + "=" * 60)
    print("Enhanced Solver V5 test complete!")


if __name__ == "__main__":
    test_enhanced_solver_v5()
