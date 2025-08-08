#!/usr/bin/env python3
"""Enhanced ARC Solver V4 with learnable pattern modifications.

Key improvements over V3:
- Learns pattern modifications from training examples
- No hardcoded modifications - all learned
- Better generalization to test examples
- Improved pattern composition
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
from learnable_pattern_modifier import LearnablePatternModifier, ModificationRule
from object_manipulation import ObjectManipulator
from program_synthesis import ProgramSynthesizer


class EnhancedARCSolverV4:
    """V4 solver with learnable pattern modifications."""

    def __init__(
        self,
        use_synthesis: bool = True,
        synthesis_timeout: float = 10.0,
        learn_modifications: bool = True,
    ):
        """Initialize the enhanced solver V4.

        Args:
            use_synthesis: Whether to use program synthesis
            synthesis_timeout: Maximum time for program synthesis in seconds
            learn_modifications: Whether to learn pattern modifications
        """
        self.perception = EnhancedPerceptionV2()
        self.object_manipulator = ObjectManipulator()
        self.tta_adapter = EnhancedARCTestTimeAdapter()
        self.synthesizer = (
            ProgramSynthesizer(strategy="beam") if use_synthesis else None
        )
        self.synthesis_timeout = synthesis_timeout
        self.learn_modifications = learn_modifications
        self.dsl_library = EnhancedDSLLibrary()
        self.pattern_modifier = LearnablePatternModifier()

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

    def learn_pattern_modifications(
        self,
        train_examples: List[Tuple[np.ndarray, np.ndarray]],
        base_transform: callable,
        pattern_type: str = "tiling",
    ) -> List[ModificationRule]:
        """Learn how to modify base patterns from training examples.

        Args:
            train_examples: Training examples
            base_transform: Function that applies base transformation
            pattern_type: Type of pattern (e.g., 'tiling')

        Returns:
            List of learned modification rules
        """
        if not self.learn_modifications:
            return []

        # Collect examples of (input, base_output, expected_output)
        modification_examples = []

        for input_grid, expected_output in train_examples:
            try:
                base_output = base_transform(input_grid)
                if base_output.shape == expected_output.shape:
                    modification_examples.append(
                        (input_grid, base_output, expected_output)
                    )
            except Exception:
                continue

        if not modification_examples:
            return []

        # Learn rules from examples
        rules = self.pattern_modifier.learn_from_examples(
            modification_examples, pattern_type
        )

        return rules

    def solve(
        self,
        train_examples: List[Tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
    ) -> ARCSolution:
        """Solve an ARC task using learnable pattern modifications.

        Key improvements over V3:
        1. Learn modifications from training examples
        2. Apply learned modifications to test input
        3. No hardcoded patterns

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

        # 2. Try geometric patterns with learned modifications
        if size_info.has_size_change:
            solution = self.try_geometric_with_learning(
                train_examples, test_input, size_info
            )
            if solution:
                solutions.append(solution)

        # 3. Try conditional/object transformations
        if perception_analysis["transformation_type"] in ["conditional", "structural"]:
            solution = self.try_object_transforms(
                train_examples, test_input, perception_analysis
            )
            if solution:
                solutions.append(solution)

        # 4. Try pattern composition with learning
        composition_solution = self.try_learned_composition(
            train_examples, test_input, perception_analysis
        )
        if composition_solution:
            solutions.append(composition_solution)

        # 5. Try synthesis if needed
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
            # Prefer solutions with learned modifications for size-changing tasks
            if size_info.has_size_change:
                learned_solutions = [s for s in solutions if "learned" in s.method_used]
                if learned_solutions:
                    best_solution = max(learned_solutions, key=lambda s: s.confidence)
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

    def try_geometric_with_learning(
        self,
        train_examples: List[Tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
        size_info: SizeChangeInfo,
    ) -> Optional[ARCSolution]:
        """Try geometric transformations with learned modifications."""

        if size_info.transformation_type == "tiling" and size_info.scale_factor:
            scale = int(size_info.scale_factor[0])

            # First try simple tiling
            primitive = self.dsl_library.get_primitive("tile_pattern", scale=scale)
            base_transform = lambda g: primitive.execute(g)

            # Learn modifications from training examples
            print(
                f"  Learning pattern modifications from {len(train_examples)} examples..."
            )
            learned_rules = self.learn_pattern_modifications(
                train_examples, base_transform, "tiling"
            )

            if learned_rules:
                print(f"  Learned {len(learned_rules)} modification rules:")
                for rule in learned_rules[:3]:  # Show first 3
                    print(f"    - {rule.name} (conf: {rule.confidence:.2f})")

                # Apply base transform then learned modifications
                base_output = base_transform(test_input)
                modified_output = self.pattern_modifier.apply_modifications(
                    base_output, learned_rules
                )

                # Calculate confidence based on how well rules matched training
                avg_confidence = (
                    sum(r.confidence for r in learned_rules) / len(learned_rules)
                    if learned_rules
                    else 0.5
                )

                # Validate on training examples
                validation_score = self._validate_learned_pattern(
                    train_examples, base_transform, learned_rules
                )

                final_confidence = (avg_confidence + validation_score) / 2

                return ARCSolution(
                    modified_output,
                    final_confidence,
                    "geometric_learned",
                )
            else:
                # No modifications learned, use simple tiling
                output = base_transform(test_input)
                return ARCSolution(output, 0.6, "geometric_simple")

        return None

    def try_learned_composition(
        self,
        train_examples: List[Tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
        perception_analysis: Dict,
    ) -> Optional[ARCSolution]:
        """Try composing patterns with learned modifications."""

        # Don't attempt composition for size-changing tasks
        # These should be handled by geometric patterns
        if train_examples:
            in_shape = train_examples[0][0].shape
            out_shape = train_examples[0][1].shape
            if in_shape != out_shape:
                return None

        # If we have multiple pattern types, try learning their composition
        pattern_types = []
        if perception_analysis.get("arithmetic_patterns"):
            pattern_types.append("arithmetic")
        if perception_analysis.get("spatial_patterns"):
            pattern_types.append("spatial")
        if perception_analysis.get("conditional_patterns"):
            pattern_types.append("conditional")

        if len(pattern_types) >= 2:
            # For now, try a simple sequential composition
            # Future: learn the composition order from examples

            # Example: spatial + arithmetic
            if "spatial" in pattern_types and "arithmetic" in pattern_types:
                spatial_patterns = perception_analysis.get("spatial_patterns", [])
                arithmetic_patterns = perception_analysis.get("arithmetic_patterns", [])

                if spatial_patterns and arithmetic_patterns:
                    best_spatial = spatial_patterns[0]
                    best_arithmetic = arithmetic_patterns[0]

                    # Create composed transform
                    def composed_transform(grid):
                        # Apply spatial first
                        if best_spatial.name == "symmetric":
                            axis = best_spatial.details.get(
                                "symmetry_type", "horizontal"
                            )
                            spatial_prim = self.dsl_library.get_primitive(
                                "make_symmetric", axis=axis
                            )
                            temp = spatial_prim.execute(grid)
                        else:
                            temp = grid

                        # Then arithmetic
                        if best_arithmetic.name == "color_shift":
                            shift = best_arithmetic.details.get("shift_value", 1)
                            arith_prim = self.dsl_library.get_primitive(
                                "add_constant", value=shift
                            )
                            result = arith_prim.execute(temp)
                        else:
                            result = temp

                        return result

                    # Learn modifications for the composition
                    learned_rules = self.learn_pattern_modifications(
                        train_examples, composed_transform, "composition"
                    )

                    output = composed_transform(test_input)
                    if learned_rules:
                        output = self.pattern_modifier.apply_modifications(
                            output, learned_rules
                        )

                    return ARCSolution(output, 0.7, "composition_learned")

        return None

    def _validate_learned_pattern(
        self,
        train_examples: List[Tuple[np.ndarray, np.ndarray]],
        base_transform: callable,
        learned_rules: List[ModificationRule],
    ) -> float:
        """Validate learned pattern on training examples."""
        if not train_examples:
            return 0.0

        correct = 0
        total = len(train_examples)

        for input_grid, expected_output in train_examples:
            try:
                # Apply base transform
                base_output = base_transform(input_grid)
                # Apply learned modifications
                modified = self.pattern_modifier.apply_modifications(
                    base_output, learned_rules
                )

                if np.array_equal(modified, expected_output):
                    correct += 1
                elif modified.shape == expected_output.shape:
                    # Partial credit for close matches
                    matching = np.sum(modified == expected_output)
                    total_pixels = modified.size
                    if total_pixels > 0:
                        correct += (matching / total_pixels) * 0.5
            except Exception:
                pass

        return correct / total if total > 0 else 0.0

    # Copy simplified versions of other methods from V3
    def try_simple_patterns(
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

        return None

    def try_object_transforms(
        self,
        train_examples: List[Tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
        perception_analysis: Dict,
    ) -> Optional[ARCSolution]:
        """Try object/conditional transformations."""

        # Check for conditional patterns
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


def test_enhanced_solver_v4():
    """Test the enhanced solver V4 with learnable modifications."""
    print("Testing Enhanced ARC Solver V4")
    print("=" * 60)

    solver = EnhancedARCSolverV4(
        use_synthesis=True,
        learn_modifications=True,
    )

    # Test 1: Pattern with learnable modifications
    print("\nTest 1: Tiling with Learnable Modifications")

    # Create training examples that show a pattern
    train_examples = [
        (
            np.array([[1, 0], [0, 1]]),
            np.array(
                [
                    [0, 0, 1, 0],
                    [0, 0, 0, 1],
                    [1, 0, 1, 0],
                    [0, 1, 0, 1],
                ]
            ),
        ),
        (
            np.array([[2, 2], [2, 2]]),
            np.array(
                [
                    [0, 0, 2, 2],
                    [0, 0, 2, 2],
                    [2, 2, 2, 2],
                    [2, 2, 2, 2],
                ]
            ),
        ),
    ]

    test_input = np.array([[3, 0], [0, 3]])

    solution = solver.solve(train_examples, test_input)
    print(f"Method: {solution.method_used}")
    print(f"Confidence: {solution.confidence:.3f}")
    print(f"Output:\n{solution.output_grid}")

    print("\n" + "=" * 60)
    print("Enhanced Solver V4 tests complete!")


if __name__ == "__main__":
    test_enhanced_solver_v4()
