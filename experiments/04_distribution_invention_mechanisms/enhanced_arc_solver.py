#!/usr/bin/env python3
"""Enhanced ARC Solver with program synthesis and object manipulation.

This module integrates all components to provide a complete solver for ARC-AGI tasks.
"""

from utils.imports import setup_project_paths

setup_project_paths()

import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from arc_dsl import Program
from enhanced_arc_tta import EnhancedARCTestTimeAdapter
from enhanced_neural_perception import EnhancedNeuralPerception
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


class EnhancedARCSolver:
    """Complete solver with all advanced capabilities."""

    def __init__(self, use_synthesis: bool = True, synthesis_timeout: float = 5.0):
        """Initialize the enhanced solver.

        Args:
            use_synthesis: Whether to use program synthesis (slower but more powerful)
            synthesis_timeout: Maximum time for program synthesis in seconds
        """
        self.perception = EnhancedNeuralPerception()
        self.object_manipulator = ObjectManipulator()
        self.tta_adapter = EnhancedARCTestTimeAdapter()
        self.synthesizer = (
            ProgramSynthesizer(strategy="beam") if use_synthesis else None
        )
        self.synthesis_timeout = synthesis_timeout

    def solve(
        self,
        train_examples: List[Tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
    ) -> ARCSolution:
        """Solve an ARC task using all available methods.

        Args:
            train_examples: List of (input, output) training examples
            test_input: Test input grid to transform

        Returns:
            ARCSolution with the best prediction
        """
        start_time = time.time()

        # 1. Try simple pattern matching (fast path)
        solution = self.try_simple_patterns(train_examples, test_input)
        if solution and solution.confidence >= 0.9:
            solution.time_taken = time.time() - start_time
            return solution

        # 2. Try object-based transformations
        solution = self.try_object_transforms(train_examples, test_input)
        if solution and solution.confidence >= 0.8:
            solution.time_taken = time.time() - start_time
            return solution

        # 3. Try program synthesis (if enabled and time allows)
        if self.synthesizer and (time.time() - start_time) < self.synthesis_timeout:
            solution = self.try_program_synthesis(train_examples, test_input)
            if solution and solution.confidence >= 0.7:
                solution.time_taken = time.time() - start_time
                return solution

        # 4. Fall back to test-time adaptation
        solution = self.try_tta(train_examples, test_input)
        solution.time_taken = time.time() - start_time
        return solution

    def try_simple_patterns(
        self,
        train_examples: List[Tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
    ) -> Optional[ARCSolution]:
        """Try simple geometric transformations."""

        # Check if all training examples have the same transformation
        transformations = []

        for input_grid, output_grid in train_examples:
            comparison = self.perception.compare_grids(input_grid, output_grid)
            if comparison.get("likely_transformation"):
                transformations.append(comparison["likely_transformation"])

        # If all examples have the same transformation, apply it
        if transformations and all(t == transformations[0] for t in transformations):
            trans_type = transformations[0]

            if trans_type == "horizontal_flip":
                output = np.flip(test_input, axis=1)
                return ARCSolution(output, 0.95, "simple")
            elif trans_type == "vertical_flip":
                output = np.flip(test_input, axis=0)
                return ARCSolution(output, 0.95, "simple")
            elif trans_type == "2x_scaling":
                output = np.kron(test_input, np.ones((2, 2), dtype=test_input.dtype))
                return ARCSolution(output, 0.95, "simple")
            elif trans_type == "rotation_90":
                output = np.rot90(test_input, -1)
                return ARCSolution(output, 0.95, "simple")
            elif trans_type == "transpose":
                output = test_input.T
                return ARCSolution(output, 0.95, "simple")
            elif trans_type == "color_mapping":
                # Extract color mapping from first example
                first_comparison = self.perception.compare_grids(
                    train_examples[0][0], train_examples[0][1]
                )
                if first_comparison.get("color_changes"):
                    color_map = first_comparison["color_changes"]
                    output = test_input.copy()
                    for old_c, new_c in color_map.items():
                        output[test_input == old_c] = new_c
                    return ARCSolution(output, 0.9, "simple")

        return None

    def try_object_transforms(
        self,
        train_examples: List[Tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
    ) -> Optional[ARCSolution]:
        """Try object-based transformations."""

        # Analyze object changes in training examples
        object_patterns = []

        for input_grid, output_grid in train_examples:
            input_objects = self.object_manipulator.extract_objects(input_grid)
            output_objects = self.object_manipulator.extract_objects(output_grid)

            pattern = {
                "input_count": len(input_objects),
                "output_count": len(output_objects),
                "size_changes": [],
                "position_changes": [],
            }

            # Check for size changes
            if input_objects and output_objects:
                input_sizes = sorted([obj.size for obj in input_objects])
                output_sizes = sorted([obj.size for obj in output_objects])
                if input_sizes != output_sizes:
                    pattern["size_changes"] = (input_sizes, output_sizes)

            object_patterns.append(pattern)

        # Check for consistent object manipulation patterns
        if all(p["output_count"] > p["input_count"] for p in object_patterns):
            # Objects are being duplicated
            output = self.object_manipulator.duplicate_objects(test_input, "double")
            return ARCSolution(output, 0.85, "object")

        elif all(p["output_count"] < p["input_count"] for p in object_patterns):
            # Objects are being removed
            output = self.object_manipulator.remove_small_objects(
                test_input, threshold=3
            )
            return ARCSolution(output, 0.85, "object")

        elif all(
            p["output_count"] == p["input_count"] and p["input_count"] > 0
            for p in object_patterns
        ):
            # Objects are being rearranged
            output = self.object_manipulator.rearrange_objects(
                test_input, "sort_by_size"
            )
            return ARCSolution(output, 0.8, "object")

        # Try pattern completion
        test_analysis = self.perception.analyze_grid(test_input)
        if test_analysis.get("patterns"):
            output = self.object_manipulator.complete_pattern(test_input)
            if not np.array_equal(output, test_input):
                return ARCSolution(output, 0.75, "object")

        return None

    def try_program_synthesis(
        self,
        train_examples: List[Tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
    ) -> Optional[ARCSolution]:
        """Try to synthesize a program that solves the task."""

        if not self.synthesizer:
            return None

        try:
            # Synthesize program from training examples
            program = self.synthesizer.synthesize(train_examples)

            if program:
                # Apply program to test input
                output = program.execute(test_input)

                # Verify program works on training examples
                verification_score = 0.0
                for inp, expected in train_examples:
                    predicted = program.execute(inp)
                    if np.array_equal(predicted, expected):
                        verification_score += 1.0

                confidence = (
                    verification_score / len(train_examples) if train_examples else 0.0
                )

                if confidence > 0.5:
                    return ARCSolution(output, confidence * 0.9, "synthesis", program)

        except Exception as e:
            print(f"Program synthesis failed: {e}")

        return None

    def try_tta(
        self,
        train_examples: List[Tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
    ) -> ARCSolution:
        """Use test-time adaptation as fallback."""

        try:
            # Adapt using TTA
            adaptation_result = self.tta_adapter.adapt(train_examples)

            if adaptation_result.best_hypothesis:
                # Apply best hypothesis
                output = adaptation_result.best_hypothesis.transform_fn(test_input)
                confidence = adaptation_result.confidence

                return ARCSolution(output, confidence, "tta")

        except Exception as e:
            print(f"TTA failed: {e}")

        # Ultimate fallback: return input unchanged
        return ARCSolution(test_input, 0.1, "tta")

    def solve_batch(
        self,
        tasks: List[Tuple[List[Tuple[np.ndarray, np.ndarray]], np.ndarray]],
        verbose: bool = False,
    ) -> List[ARCSolution]:
        """Solve multiple ARC tasks.

        Args:
            tasks: List of (train_examples, test_input) tuples
            verbose: Whether to print progress

        Returns:
            List of solutions
        """
        solutions = []

        for i, (train_examples, test_input) in enumerate(tasks):
            if verbose:
                print(f"Solving task {i+1}/{len(tasks)}...")

            solution = self.solve(train_examples, test_input)
            solutions.append(solution)

            if verbose:
                print(
                    f"  Method: {solution.method_used}, Confidence: {solution.confidence:.2f}"
                )

        return solutions


def test_enhanced_solver():
    """Test the enhanced ARC solver."""
    print("Testing Enhanced ARC Solver")
    print("=" * 50)

    solver = EnhancedARCSolver(use_synthesis=True)

    # Test 1: Simple horizontal flip
    print("\nTest 1: Horizontal Flip")
    print("-" * 30)

    train_examples = [
        (np.array([[1, 2], [3, 4]]), np.array([[2, 1], [4, 3]])),
        (np.array([[5, 6], [7, 8]]), np.array([[6, 5], [8, 7]])),
    ]
    test_input = np.array([[9, 8], [7, 6]])

    solution = solver.solve(train_examples, test_input)
    print(f"Method used: {solution.method_used}")
    print(f"Confidence: {solution.confidence:.2f}")
    print(f"Time taken: {solution.time_taken:.3f}s")
    print("Output:")
    print(solution.output_grid)

    # Test 2: Object duplication
    print("\nTest 2: Object Duplication")
    print("-" * 30)

    train_examples = [
        (
            np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]]),
            np.array([[1, 0, 1], [0, 0, 0], [0, 0, 0]]),
        ),
        (
            np.array([[2, 2, 0], [2, 2, 0], [0, 0, 0]]),
            np.array([[2, 2, 0], [2, 2, 0], [2, 2, 0]]),
        ),
    ]
    test_input = np.array([[3, 3, 0], [3, 3, 0], [0, 0, 0]])

    solution = solver.solve(train_examples, test_input)
    print(f"Method used: {solution.method_used}")
    print(f"Confidence: {solution.confidence:.2f}")
    print(f"Time taken: {solution.time_taken:.3f}s")
    print("Output:")
    print(solution.output_grid)

    # Test 3: Complex pattern requiring synthesis
    print("\nTest 3: Complex Pattern")
    print("-" * 30)

    train_examples = [
        (
            np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
            np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]]),  # 180 rotation
        )
    ]
    test_input = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])

    solution = solver.solve(train_examples, test_input)
    print(f"Method used: {solution.method_used}")
    print(f"Confidence: {solution.confidence:.2f}")
    print(f"Time taken: {solution.time_taken:.3f}s")
    if solution.program:
        print("Program found:")
        print(solution.program.to_code())
    print("Output:")
    print(solution.output_grid)

    # Test batch solving
    print("\nTest 4: Batch Solving")
    print("-" * 30)

    tasks = [
        (train_examples, test_input)
        for train_examples, test_input in [
            ([(np.array([[1, 2]]), np.array([[2, 1]]))], np.array([[3, 4]])),
            ([(np.array([[1]]), np.array([[2]]))], np.array([[5]])),
        ]
    ]

    solutions = solver.solve_batch(tasks, verbose=True)
    print(f"\nSolved {len(solutions)} tasks")
    print(f"Average confidence: {np.mean([s.confidence for s in solutions]):.2f}")


if __name__ == "__main__":
    test_enhanced_solver()
