#!/usr/bin/env python3
"""Enhanced ARC Solver V2 with improved search strategy.

Key improvements:
- Always tries program synthesis when other methods fail
- Uses enhanced perception to guide search
- Adaptive confidence thresholds
- Better integration of new DSL primitives
"""

from utils.imports import setup_project_paths

setup_project_paths()

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

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
    method_used: str  # 'simple', 'object', 'synthesis', 'tta'
    program: Optional[Program] = None
    time_taken: float = 0.0
    perception_analysis: Optional[Dict] = None


class EnhancedARCSolverV2:
    """Improved solver with better search strategy."""

    def __init__(
        self,
        use_synthesis: bool = True,
        synthesis_timeout: float = 10.0,
        adaptive_thresholds: bool = True,
    ):
        """Initialize the enhanced solver.

        Args:
            use_synthesis: Whether to use program synthesis
            synthesis_timeout: Maximum time for program synthesis in seconds
            adaptive_thresholds: Whether to adapt confidence thresholds based on perception
        """
        self.perception = EnhancedPerceptionV2()
        self.object_manipulator = ObjectManipulator()
        self.tta_adapter = EnhancedARCTestTimeAdapter()
        self.synthesizer = (
            ProgramSynthesizer(strategy="beam") if use_synthesis else None
        )
        self.synthesis_timeout = synthesis_timeout
        self.adaptive_thresholds = adaptive_thresholds
        self.dsl_library = EnhancedDSLLibrary()

    def solve(
        self,
        train_examples: List[Tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
    ) -> ARCSolution:
        """Solve an ARC task using improved strategy.

        Key improvements:
        1. Use perception to analyze patterns first
        2. Adjust confidence thresholds based on pattern complexity
        3. Always try synthesis if other methods fail
        4. Use enhanced DSL primitives

        Args:
            train_examples: List of (input, output) training examples
            test_input: Test input grid to transform

        Returns:
            ARCSolution with the best prediction
        """
        start_time = time.time()

        # Analyze patterns with enhanced perception
        perception_analysis = self.perception.analyze(train_examples)

        # Determine confidence thresholds based on pattern complexity
        thresholds = self._get_adaptive_thresholds(perception_analysis)

        # 1. Try simple patterns if detected
        if perception_analysis["transformation_type"] in ["arithmetic", "spatial"]:
            solution = self.try_enhanced_simple_patterns(
                train_examples, test_input, perception_analysis
            )
            if solution and solution.confidence >= thresholds["simple"]:
                solution.time_taken = time.time() - start_time
                solution.perception_analysis = perception_analysis
                return solution

        # 2. Try conditional/object transformations
        if perception_analysis["transformation_type"] in ["conditional", "structural"]:
            solution = self.try_enhanced_object_transforms(
                train_examples, test_input, perception_analysis
            )
            if solution and solution.confidence >= thresholds["object"]:
                solution.time_taken = time.time() - start_time
                solution.perception_analysis = perception_analysis
                return solution

        # 3. ALWAYS try program synthesis if we haven't found a good solution
        # This was the key missing piece - synthesis was never triggered!
        if self.synthesizer:
            remaining_time = self.synthesis_timeout - (time.time() - start_time)
            if remaining_time > 1.0:  # Need at least 1 second
                solution = self.try_enhanced_synthesis(
                    train_examples, test_input, perception_analysis, remaining_time
                )
                if solution and solution.confidence >= thresholds["synthesis"]:
                    solution.time_taken = time.time() - start_time
                    solution.perception_analysis = perception_analysis
                    return solution

        # 4. Fall back to test-time adaptation
        solution = self.try_tta(train_examples, test_input)
        solution.time_taken = time.time() - start_time
        solution.perception_analysis = perception_analysis
        return solution

    def _get_adaptive_thresholds(self, perception_analysis: Dict) -> Dict[str, float]:
        """Get adaptive confidence thresholds based on pattern analysis."""
        if not self.adaptive_thresholds:
            return {"simple": 0.9, "object": 0.8, "synthesis": 0.7}

        # Lower thresholds for complex patterns
        complexity = len(
            perception_analysis["arithmetic_patterns"]
            + perception_analysis["conditional_patterns"]
            + perception_analysis["spatial_patterns"]
            + perception_analysis["structural_patterns"]
        )

        if complexity > 5:
            # Very complex - be more permissive
            return {"simple": 0.7, "object": 0.6, "synthesis": 0.5}
        elif complexity > 2:
            # Moderately complex
            return {"simple": 0.8, "object": 0.7, "synthesis": 0.6}
        else:
            # Simple patterns - require high confidence
            return {"simple": 0.9, "object": 0.85, "synthesis": 0.75}

    def try_enhanced_simple_patterns(
        self,
        train_examples: List[Tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
        perception_analysis: Dict,
    ) -> Optional[ARCSolution]:
        """Try simple patterns using enhanced DSL primitives."""

        # Check for arithmetic patterns
        if perception_analysis["arithmetic_patterns"]:
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
        if perception_analysis["spatial_patterns"]:
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
        if perception_analysis["conditional_patterns"]:
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
        if perception_analysis["structural_patterns"]:
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

        # Configure synthesizer with perception hints
        self._get_synthesis_hints(perception_analysis)

        try:
            # Synthesize program (hints would need custom implementation)
            # For now, just use standard synthesis with timeout
            # Note: timeout handling would need to be implemented in synthesizer
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

    def _get_synthesis_hints(self, perception_analysis: Dict) -> Dict[str, Any]:
        """Convert perception analysis to synthesis hints."""
        hints = {
            "preferred_primitives": [],
            "avoid_primitives": [],
            "max_depth": 5,
        }

        # Add primitives based on detected patterns
        if perception_analysis["arithmetic_patterns"]:
            hints["preferred_primitives"].extend(
                ["add_constant", "count_objects", "enumerate_objects"]
            )

        if perception_analysis["conditional_patterns"]:
            hints["preferred_primitives"].extend(
                ["if_size", "if_color", "if_square", "if_large"]
            )

        if perception_analysis["spatial_patterns"]:
            hints["preferred_primitives"].extend(
                ["draw_diagonal", "draw_border", "fill_spiral", "repeat_pattern"]
            )

        if perception_analysis["structural_patterns"]:
            hints["preferred_primitives"].extend(
                ["merge_adjacent", "connect_objects", "duplicate_n"]
            )

        # Adjust depth based on complexity
        pattern_count = sum(
            len(patterns)
            for patterns in [
                perception_analysis["arithmetic_patterns"],
                perception_analysis["conditional_patterns"],
                perception_analysis["spatial_patterns"],
                perception_analysis["structural_patterns"],
            ]
        )

        if pattern_count > 3:
            hints["max_depth"] = 7  # Allow deeper programs for complex patterns

        return hints

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


def test_enhanced_solver_v2():
    """Test the enhanced solver with better search strategy."""
    print("Testing Enhanced ARC Solver V2")
    print("=" * 60)

    solver = EnhancedARCSolverV2(use_synthesis=True, adaptive_thresholds=True)

    # Test 1: Arithmetic pattern (color shift)
    print("\nTest 1: Color Shift (+2)")
    train_examples = [
        (np.array([[1, 2], [2, 3]]), np.array([[3, 4], [4, 5]])),
        (np.array([[2, 1], [3, 2]]), np.array([[4, 3], [5, 4]])),
    ]
    test_input = np.array([[1, 3], [3, 1]])

    solution = solver.solve(train_examples, test_input)
    print(f"Method: {solution.method_used}")
    print(f"Confidence: {solution.confidence:.3f}")
    print(f"Output:\n{solution.output_grid}")

    # Test 2: Conditional pattern (if square then fill)
    print("\nTest 2: Conditional (If square, fill with 5)")
    train_examples = [
        (
            np.array([[1, 1, 0], [1, 1, 0], [2, 2, 2]]),
            np.array([[5, 5, 0], [5, 5, 0], [2, 2, 2]]),
        ),
    ]
    test_input = np.array([[3, 3, 0], [3, 3, 0], [0, 0, 0]])

    solution = solver.solve(train_examples, test_input)
    print(f"Method: {solution.method_used}")
    print(f"Confidence: {solution.confidence:.3f}")
    if solution.perception_analysis:
        print(
            f"Detected patterns: {solution.perception_analysis['transformation_type']}"
        )

    print("\n" + "=" * 60)
    print("Enhanced Solver V2 tests complete!")


if __name__ == "__main__":
    test_enhanced_solver_v2()
