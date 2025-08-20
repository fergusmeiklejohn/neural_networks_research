#!/usr/bin/env python3
"""Enhanced Program Synthesis for ARC-AGI with perception-guided search.

This module implements an improved program synthesis engine that uses
perception analysis to guide the search and the enhanced DSL library.
"""

from utils.imports import setup_project_paths

setup_project_paths()

import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from arc_dsl_enhanced import EnhancedDSLLibrary
from enhanced_perception_v2 import EnhancedPerceptionV2


@dataclass
class Program:
    """Represents a program as a sequence of operations."""

    operations: List

    def execute(self, input_grid: np.ndarray) -> np.ndarray:
        """Execute the program on an input grid."""
        result = input_grid.copy()
        for op in self.operations:
            result = op.execute(result)
        return result


@dataclass
class SearchNode:
    """Node in the program search tree."""

    program: Program
    score: float
    depth: int
    operations_tried: set  # Track what we've tried to avoid loops

    def __lt__(self, other):
        return self.score > other.score


@dataclass
class SynthesisResult:
    """Result from program synthesis."""

    best_program: Optional[Program]
    score: float
    candidates_explored: int
    success: bool
    time_taken: float


class EnhancedProgramSynthesizer:
    """Enhanced program synthesis with perception-guided search."""

    def __init__(
        self,
        beam_width: int = 30,
        max_depth: int = 4,
        use_perception_hints: bool = True,
        timeout: float = 10.0,
    ):
        """Initialize the enhanced synthesizer.

        Args:
            beam_width: Number of candidates to keep at each level
            max_depth: Maximum program length
            use_perception_hints: Whether to use perception to guide search
            timeout: Maximum time in seconds
        """
        self.beam_width = beam_width
        self.max_depth = max_depth
        self.use_perception_hints = use_perception_hints
        self.timeout = timeout
        self.dsl_library = EnhancedDSLLibrary()
        self.perception = EnhancedPerceptionV2()

    def synthesize(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]],
        perception_analysis: Optional[Dict] = None,
    ) -> Optional[Program]:
        """Synthesize a program from examples.

        Args:
            examples: List of (input, output) grid pairs
            perception_analysis: Optional pre-computed perception analysis

        Returns:
            Best program found or None
        """
        start_time = time.time()

        # Analyze examples if not provided
        if perception_analysis is None and self.use_perception_hints:
            perception_analysis = self.perception.analyze(examples)

        # Get initial candidates based on perception
        initial_candidates = self._get_initial_candidates(examples, perception_analysis)

        # Run beam search
        result = self._beam_search(examples, initial_candidates, start_time)

        if result.success:
            return result.best_program
        return None

    def _beam_search(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]],
        initial_candidates: List[Program],
        start_time: float,
    ) -> SynthesisResult:
        """Run beam search to find program."""

        # Initialize beam with initial candidates
        beam = []
        for prog in initial_candidates:
            score = self._score_program(prog, examples)
            beam.append(SearchNode(prog, score, 1, set([str(prog.operations[0])])))

        # Also start with empty program
        beam.append(SearchNode(Program([]), 0.0, 0, set()))

        best_program = None
        best_score = 0.0
        candidates_explored = 0

        for depth in range(self.max_depth):
            # Check timeout
            if time.time() - start_time > self.timeout:
                break

            new_beam = []

            for node in beam:
                # Check if current program solves all examples
                if node.program.operations:
                    score = self._score_program(node.program, examples)
                    candidates_explored += 1

                    if score > best_score:
                        best_score = score
                        best_program = node.program

                    # If perfect score, return immediately
                    if score >= 0.99:
                        return SynthesisResult(
                            best_program=node.program,
                            score=score,
                            candidates_explored=candidates_explored,
                            success=True,
                            time_taken=time.time() - start_time,
                        )

                # Generate children
                if node.depth < self.max_depth:
                    children = self._expand_node(node, examples)
                    new_beam.extend(children)

            # Keep top beam_width nodes
            new_beam.sort(key=lambda n: n.score, reverse=True)
            beam = new_beam[: self.beam_width]

            if not beam:
                break

        return SynthesisResult(
            best_program=best_program,
            score=best_score,
            candidates_explored=candidates_explored,
            success=best_score >= 0.99,
            time_taken=time.time() - start_time,
        )

    def _get_initial_candidates(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]],
        perception_analysis: Optional[Dict],
    ) -> List[Program]:
        """Get initial program candidates based on perception."""

        candidates = []

        if not perception_analysis or not self.use_perception_hints:
            return candidates

        # Based on detected patterns, create initial programs

        # Arithmetic patterns
        if perception_analysis.get("arithmetic_patterns"):
            for pattern in perception_analysis["arithmetic_patterns"][:2]:
                if pattern.name == "color_shift":
                    shift = pattern.details.get("shift_value", 1)
                    prim = self.dsl_library.get_primitive("add_constant", value=shift)
                    candidates.append(Program([prim]))
                elif pattern.name == "count_encoding":
                    prim = self.dsl_library.get_primitive("count_objects")
                    candidates.append(Program([prim]))

        # Spatial patterns
        if perception_analysis.get("spatial_patterns"):
            for pattern in perception_analysis["spatial_patterns"][:2]:
                if pattern.name == "diagonal":
                    prim = self.dsl_library.get_primitive("draw_diagonal", color=1)
                    candidates.append(Program([prim]))
                elif pattern.name == "border":
                    color = pattern.details.get("color", 1)
                    prim = self.dsl_library.get_primitive("draw_border", color=color)
                    candidates.append(Program([prim]))
                elif pattern.name == "symmetric":
                    axis = pattern.details.get("symmetry_type", "horizontal")
                    prim = self.dsl_library.get_primitive("make_symmetric", axis=axis)
                    candidates.append(Program([prim]))

        # Conditional patterns
        if perception_analysis.get("conditional_patterns"):
            for pattern in perception_analysis["conditional_patterns"][:2]:
                if pattern.name == "size_based":
                    prim = self.dsl_library.get_primitive("if_large", fill_color=1)
                    candidates.append(Program([prim]))
                elif pattern.name == "color_based":
                    # Try color replacement
                    mappings = pattern.details.get("color_mappings", {})
                    if mappings:
                        old_c, new_c = list(mappings.items())[0]
                        prim = self.dsl_library.get_primitive(
                            "if_color", test_color=old_c, then_color=new_c
                        )
                        candidates.append(Program([prim]))

        # Structural patterns
        if perception_analysis.get("structural_patterns"):
            for pattern in perception_analysis["structural_patterns"][:1]:
                if pattern.name == "objects_merged":
                    prim = self.dsl_library.get_primitive("merge_adjacent")
                    candidates.append(Program([prim]))

        # Size changes - add tiling
        if examples:
            in_shape = examples[0][0].shape
            out_shape = examples[0][1].shape
            if out_shape[0] > in_shape[0] and out_shape[1] > in_shape[1]:
                if out_shape[0] % in_shape[0] == 0:
                    scale = out_shape[0] // in_shape[0]
                    prim = self.dsl_library.get_primitive("tile_pattern", scale=scale)
                    candidates.append(Program([prim]))

        return candidates

    def _expand_node(
        self,
        node: SearchNode,
        examples: List[Tuple[np.ndarray, np.ndarray]],
    ) -> List[SearchNode]:
        """Expand a node by adding possible next operations."""

        children = []

        # Get candidate operations
        if node.program.operations:
            # If we have operations, get context-aware candidates
            candidates = self._get_next_operations(node.program, examples)
        else:
            # First operation - use broad set
            candidates = self._get_first_operations(examples)

        for op_name, kwargs in candidates:
            # Skip if we've tried this before in this path
            op_str = f"{op_name}_{kwargs}"
            if op_str in node.operations_tried:
                continue

            try:
                # Create new primitive
                primitive = self.dsl_library.get_primitive(op_name, **kwargs)

                # Create new program
                new_operations = node.program.operations + [primitive]
                new_program = Program(new_operations)

                # Score the new program
                score = self._score_program(new_program, examples)

                # Only keep if score improves or maintains
                if score >= node.score * 0.9:  # Allow small decreases
                    new_tried = node.operations_tried.copy()
                    new_tried.add(op_str)
                    child = SearchNode(new_program, score, node.depth + 1, new_tried)
                    children.append(child)

            except Exception:
                continue

        return children

    def _get_first_operations(
        self, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> List[Tuple[str, Dict]]:
        """Get candidate first operations."""

        candidates = []

        if not examples:
            return candidates

        input_grid, output_grid = examples[0]

        # Check for size changes
        if input_grid.shape != output_grid.shape:
            h_ratio = output_grid.shape[0] / input_grid.shape[0]
            w_ratio = output_grid.shape[1] / input_grid.shape[1]

            if h_ratio == w_ratio and h_ratio == int(h_ratio):
                scale = int(h_ratio)
                if scale > 1:
                    candidates.append(("tile_pattern", {"scale": scale}))
                    candidates.append(("modified_tile_pattern", {"scale": scale}))

        # Color operations
        unique_in = set(input_grid.flatten())
        unique_out = set(output_grid.flatten())

        if len(unique_out - unique_in) > 0:
            # New colors appear
            for shift in [1, 2, 3, -1, -2]:
                candidates.append(("add_constant", {"value": shift}))

        # Common transformations
        candidates.extend(
            [
                ("make_symmetric", {"axis": "horizontal"}),
                ("make_symmetric", {"axis": "vertical"}),
                ("draw_diagonal", {"color": 1, "anti": False}),
                ("draw_border", {"color": 1, "thickness": 1}),
                ("count_objects", {}),
                ("enumerate_objects", {}),
                ("if_large", {"fill_color": 1, "size_threshold": 4}),
                ("if_square", {"then_color": 1}),
                ("merge_adjacent", {}),
            ]
        )

        # Limit and randomize
        random.shuffle(candidates)
        return candidates[:15]

    def _get_next_operations(
        self,
        current_program: Program,
        examples: List[Tuple[np.ndarray, np.ndarray]],
    ) -> List[Tuple[str, Dict]]:
        """Get candidate next operations based on current program."""

        candidates = []

        # Execute current program on first example
        try:
            input_grid = examples[0][0]
            output_grid = examples[0][1]
            intermediate = current_program.execute(input_grid)

            # See what's still needed
            if not np.array_equal(intermediate, output_grid):
                # Check for remaining color differences
                diff_mask = intermediate != output_grid

                # If specific regions differ, suggest regional operations
                if np.sum(diff_mask) < intermediate.size * 0.3:
                    candidates.append(("draw_border", {"color": 1}))
                    candidates.append(("if_color", {"test_color": 0, "then_color": 1}))

                # If many differences, suggest global operations
                else:
                    candidates.append(("add_constant", {"value": 1}))
                    candidates.append(("make_symmetric", {"axis": "horizontal"}))
        except:
            pass

        # Add some random operations
        candidates.extend(
            [
                ("if_size", {"threshold": 3, "then_color": 1}),
                ("duplicate_n", {"n": 2}),
                ("select_nth", {"n": 0}),
            ]
        )

        random.shuffle(candidates)
        return candidates[:10]

    def _score_program(
        self, program: Program, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> float:
        """Score how well a program solves the examples."""

        if not program.operations:
            return 0.0

        total_score = 0.0

        for input_grid, expected_output in examples:
            try:
                predicted = program.execute(input_grid)

                # Check exact match
                if np.array_equal(predicted, expected_output):
                    total_score += 1.0
                else:
                    # Partial credit
                    if predicted.shape == expected_output.shape:
                        similarity = (
                            np.sum(predicted == expected_output) / expected_output.size
                        )
                        total_score += similarity * 0.8
                    else:
                        # Wrong shape - small penalty
                        total_score += 0.1

            except Exception:
                # Program failed
                total_score += 0.0

        return total_score / len(examples) if examples else 0.0


def test_enhanced_synthesis():
    """Test the enhanced program synthesis."""
    print("Testing Enhanced Program Synthesis")
    print("=" * 60)

    synthesizer = EnhancedProgramSynthesizer(
        beam_width=20,
        max_depth=3,
        use_perception_hints=True,
        timeout=5.0,
    )

    # Test 1: Simple color shift
    print("\nTest 1: Color Shift")
    examples = [
        (np.array([[1, 2], [2, 3]]), np.array([[3, 4], [4, 5]])),
        (np.array([[0, 1], [1, 2]]), np.array([[2, 3], [3, 4]])),
    ]

    program = synthesizer.synthesize(examples)
    if program:
        print(f"Found program with {len(program.operations)} operations")
        for op in program.operations:
            print(f"  - {op}")

        # Test on new input
        test_input = np.array([[2, 3], [3, 4]])
        result = program.execute(test_input)
        print(f"Test: {test_input.flatten()} -> {result.flatten()}")
    else:
        print("No program found")

    # Test 2: Symmetry
    print("\nTest 2: Make Symmetric")
    examples = [
        (
            np.array([[1, 2, 0], [0, 0, 0], [0, 0, 0]]),
            np.array([[1, 2, 0], [0, 0, 0], [1, 2, 0]]),
        ),
    ]

    program = synthesizer.synthesize(examples)
    if program:
        print(f"Found program with {len(program.operations)} operations")
        for op in program.operations:
            print(f"  - {op}")
    else:
        print("No program found")

    print("\n" + "=" * 60)
    print("Enhanced synthesis test complete!")


if __name__ == "__main__":
    test_enhanced_synthesis()
