#!/usr/bin/env python3
"""Bidirectional Program Synthesis for ARC-AGI.

Implements bottom-up enumeration, top-down synthesis, and bidirectional search
with aggressive pruning strategies.
"""

from utils.imports import setup_project_paths

setup_project_paths()

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from compositional_dsl import CompositionalDSL, ExecutionContext, Primitive, Sequence


@dataclass
class ProgramNode:
    """Node representing a partial or complete program."""

    program: Primitive
    size: int  # Program size (for complexity penalty)
    score: float  # How well it matches examples
    examples_satisfied: Set[int] = field(
        default_factory=set
    )  # Which examples it solves

    def __lt__(self, other):
        # For heap - prioritize by score/size ratio
        return (self.score / (self.size + 1)) > (other.score / (other.size + 1))


@dataclass
class SearchState:
    """State of the synthesis search."""

    nodes_explored: int = 0
    programs_evaluated: int = 0
    best_program: Optional[Primitive] = None
    best_score: float = 0.0
    start_time: float = 0.0
    timeout: float = 30.0


class BottomUpEnumerator:
    """Bottom-up enumeration with aggressive pruning."""

    def __init__(self, dsl: CompositionalDSL, max_size: int = 5):
        self.dsl = dsl
        self.max_size = max_size
        self.primitive_cache = {}  # Cache primitive evaluations
        self.equivalence_classes = defaultdict(list)  # Group equivalent programs

    def enumerate(
        self, examples: List[Tuple[np.ndarray, np.ndarray]], max_programs: int = 1000
    ) -> List[ProgramNode]:
        """Enumerate programs bottom-up with pruning."""
        programs_by_size = defaultdict(list)

        # Size 1: Atomic primitives
        for name, primitive_class in self.dsl.primitives.items():
            if name in ["sequence", "conditional", "loop", "for_each_object"]:
                continue  # Skip compositional operators for size 1

            # Generate parameter values
            param_sets = self._generate_parameters(name, examples)

            for params in param_sets[:10]:  # Limit parameter combinations
                try:
                    primitive = self.dsl.get_primitive(name, **params)
                    node = self._evaluate_program(primitive, examples, size=1)
                    if node.score > 0:
                        programs_by_size[1].append(node)
                except Exception:
                    continue

        # Sizes 2-max_size: Compositions
        for size in range(2, self.max_size + 1):
            # Generate compositions
            for size1 in range(1, size):
                size2 = size - size1

                for prog1 in programs_by_size[size1][:20]:  # Limit beam
                    for prog2 in programs_by_size[size2][:20]:
                        # Try sequence
                        seq = Sequence([prog1.program, prog2.program])
                        node = self._evaluate_program(seq, examples, size)

                        if node.score > prog1.score and node.score > prog2.score:
                            # Only keep if better than parts
                            programs_by_size[size].append(node)

                        # Try loop (if size2 == 1)
                        if size2 == 1:
                            for times in [2, 3, 4]:
                                from compositional_dsl import Loop

                                loop = Loop(prog1.program, times)
                                node = self._evaluate_program(loop, examples, size)
                                if node.score > prog1.score:
                                    programs_by_size[size].append(node)

            # Prune to top K at each size
            programs_by_size[size] = sorted(programs_by_size[size], reverse=True)[:50]

            if len(programs_by_size[size]) >= max_programs:
                break

        # Collect all programs
        all_programs = []
        for programs in programs_by_size.values():
            all_programs.extend(programs)

        return sorted(all_programs, reverse=True)[:max_programs]

    def _generate_parameters(
        self, primitive_name: str, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> List[Dict]:
        """Generate parameter values for a primitive based on examples."""
        params_list = []

        # Analyze examples to extract relevant values
        colors = set()
        sizes = []
        for inp, out in examples:
            colors.update(np.unique(inp))
            colors.update(np.unique(out))
            sizes.append(inp.shape)

        colors = list(colors)
        max_h = max(s[0] for s in sizes)
        max_w = max(s[1] for s in sizes)

        if primitive_name == "move":
            # Generate move offsets
            for dx in [-3, -2, -1, 0, 1, 2, 3]:
                for dy in [-3, -2, -1, 0, 1, 2, 3]:
                    if dx != 0 or dy != 0:
                        params_list.append({"dx": dx, "dy": dy})

        elif primitive_name == "rotate":
            params_list.extend([{"angle": 90}, {"angle": 180}, {"angle": 270}])

        elif primitive_name == "set_color":
            # Generate color mappings
            for c1 in colors:
                for c2 in colors:
                    if c1 != c2:
                        params_list.append({"from_color": int(c1), "to_color": int(c2)})

        elif primitive_name == "fill_rectangle":
            # Generate some rectangle positions
            for color in colors:
                if color != 0:  # Don't fill with background
                    # Small rectangles
                    params_list.append(
                        {"x1": 0, "y1": 0, "x2": 2, "y2": 2, "color": int(color)}
                    )
                    # Full grid
                    params_list.append(
                        {
                            "x1": 0,
                            "y1": 0,
                            "x2": max_w - 1,
                            "y2": max_h - 1,
                            "color": int(color),
                        }
                    )

        elif primitive_name == "tile_pattern":
            # Common tiling factors
            for scale in [2, 3]:
                params_list.append({"scale_x": scale, "scale_y": scale})

        elif primitive_name == "draw_border":
            for color in colors:
                if color != 0:
                    params_list.append({"color": int(color), "thickness": 1})

        elif primitive_name in ["flip_h", "flip_v", "extract_objects"]:
            # No parameters
            params_list.append({})

        # Limit number of parameter sets
        return params_list[:20]

    def _evaluate_program(
        self,
        program: Primitive,
        examples: List[Tuple[np.ndarray, np.ndarray]],
        size: int,
    ) -> ProgramNode:
        """Evaluate a program on examples."""
        score = 0.0
        satisfied = set()

        for i, (inp, expected) in enumerate(examples):
            try:
                context = ExecutionContext(
                    input_grid=inp.copy(), current_grid=inp.copy()
                )
                result = program.execute(context)
                output = result.current_grid

                # Check exact match
                if np.array_equal(output, expected):
                    score += 1.0
                    satisfied.add(i)
                else:
                    # Partial credit for similarity
                    if output.shape == expected.shape:
                        matching = np.sum(output == expected)
                        total = output.size
                        score += 0.5 * (matching / total)

            except Exception:
                # Program failed on this example
                continue

        return ProgramNode(
            program=program,
            size=size,
            score=score / len(examples),
            examples_satisfied=satisfied,
        )


class TopDownSynthesizer:
    """Top-down synthesis from sketches."""

    def __init__(self, dsl: CompositionalDSL):
        self.dsl = dsl
        self.sketch_patterns = self._initialize_sketches()

    def _initialize_sketches(self) -> List[str]:
        """Initialize common program sketches."""
        return [
            "color_transform",  # Change colors
            "spatial_transform",  # Move/rotate/flip
            "object_manipulation",  # Extract and transform objects
            "pattern_tiling",  # Tile or repeat patterns
            "conditional_transform",  # If-then transformations
        ]

    def synthesize_from_sketch(
        self, sketch: str, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[ProgramNode]:
        """Synthesize a program from a sketch."""
        if sketch == "color_transform":
            return self._synthesize_color_transform(examples)
        elif sketch == "spatial_transform":
            return self._synthesize_spatial_transform(examples)
        elif sketch == "object_manipulation":
            return self._synthesize_object_manipulation(examples)
        elif sketch == "pattern_tiling":
            return self._synthesize_pattern_tiling(examples)
        else:
            return None

    def _synthesize_color_transform(
        self, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[ProgramNode]:
        """Synthesize color transformation program."""
        # Analyze color mappings
        color_map = {}
        for inp, out in examples:
            if inp.shape != out.shape:
                return None

            # Find color correspondence
            for color_in in np.unique(inp):
                mask = inp == color_in
                out_colors = out[mask]
                if len(out_colors) > 0:
                    # Most common color in output at those positions
                    unique, counts = np.unique(out_colors, return_counts=True)
                    most_common = unique[np.argmax(counts)]
                    if color_in != most_common:
                        color_map[int(color_in)] = int(most_common)

        if not color_map:
            return None

        # Create program
        from compositional_dsl import SetColor

        operations = [SetColor(k, v) for k, v in color_map.items()]
        program = Sequence(operations) if len(operations) > 1 else operations[0]

        # Evaluate
        enumerator = BottomUpEnumerator(self.dsl)
        return enumerator._evaluate_program(program, examples, len(operations))

    def _synthesize_spatial_transform(
        self, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[ProgramNode]:
        """Synthesize spatial transformation program."""
        # Try rotations
        for angle in [90, 180, 270]:
            from compositional_dsl import Rotate

            program = Rotate(angle)
            enumerator = BottomUpEnumerator(self.dsl)
            node = enumerator._evaluate_program(program, examples, 1)
            if node.score > 0.9:
                return node

        # Try flips
        from compositional_dsl import FlipH, FlipV

        for program in [FlipH(), FlipV()]:
            enumerator = BottomUpEnumerator(self.dsl)
            node = enumerator._evaluate_program(program, examples, 1)
            if node.score > 0.9:
                return node

        return None

    def _synthesize_object_manipulation(
        self, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[ProgramNode]:
        """Synthesize object manipulation program."""
        # This would need more sophisticated object analysis
        # For now, try simple object extraction and transformation
        from compositional_dsl import ExtractObjects, ForEachObject, Rotate

        program = Sequence([ExtractObjects(), ForEachObject(Rotate(90))])
        enumerator = BottomUpEnumerator(self.dsl)
        return enumerator._evaluate_program(program, examples, 2)

    def _synthesize_pattern_tiling(
        self, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[ProgramNode]:
        """Synthesize pattern tiling program."""
        # Check if output is larger than input
        size_changes = []
        for inp, out in examples:
            if out.shape[0] % inp.shape[0] == 0 and out.shape[1] % inp.shape[1] == 0:
                scale_y = out.shape[0] // inp.shape[0]
                scale_x = out.shape[1] // inp.shape[1]
                size_changes.append((scale_x, scale_y))

        if size_changes and all(s == size_changes[0] for s in size_changes):
            scale_x, scale_y = size_changes[0]
            from compositional_dsl import TilePattern

            program = TilePattern(scale_x, scale_y)
            enumerator = BottomUpEnumerator(self.dsl)
            return enumerator._evaluate_program(program, examples, 1)

        return None


class BidirectionalSynthesizer:
    """Bidirectional synthesis combining bottom-up and top-down."""

    def __init__(self, dsl: CompositionalDSL, timeout: float = 30.0):
        self.dsl = dsl
        self.bottom_up = BottomUpEnumerator(dsl)
        self.top_down = TopDownSynthesizer(dsl)
        self.timeout = timeout

    def synthesize(
        self, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[Primitive]:
        """Synthesize a program using bidirectional search."""
        state = SearchState(start_time=time.time(), timeout=self.timeout)

        # Try top-down synthesis first (faster for common patterns)
        print("Trying top-down synthesis...")
        for sketch in self.top_down.sketch_patterns:
            if time.time() - state.start_time > self.timeout:
                break

            node = self.top_down.synthesize_from_sketch(sketch, examples)
            if node and node.score > state.best_score:
                state.best_program = node.program
                state.best_score = node.score
                print(
                    f"  Found program with score {node.score:.2f} using sketch '{sketch}'"
                )

                if node.score >= 0.99:  # Perfect match
                    return node.program

        # Try bottom-up enumeration
        print("Trying bottom-up enumeration...")
        remaining_time = self.timeout - (time.time() - state.start_time)
        if remaining_time > 0:
            programs = self.bottom_up.enumerate(examples, max_programs=100)

            for node in programs:
                if node.score > state.best_score:
                    state.best_program = node.program
                    state.best_score = node.score
                    print(f"  Found program with score {node.score:.2f}")

                    if node.score >= 0.99:
                        return node.program

        print(f"Search complete. Best score: {state.best_score:.2f}")
        return state.best_program


def test_synthesis():
    """Test the synthesis system on a simple example."""
    dsl = CompositionalDSL()
    synthesizer = BidirectionalSynthesizer(dsl, timeout=10.0)

    # Test 1: Simple color transformation
    print("\nTest 1: Color transformation")
    examples = [
        (np.array([[1, 2], [2, 1]]), np.array([[3, 4], [4, 3]])),
        (np.array([[1, 1], [2, 2]]), np.array([[3, 3], [4, 4]])),
    ]

    program = synthesizer.synthesize(examples)
    if program:
        print(f"Found program: {program}")
        # Test on first example
        result = dsl.execute_program(program, examples[0][0])
        print(f"Input:\n{examples[0][0]}")
        print(f"Expected:\n{examples[0][1]}")
        print(f"Got:\n{result}")
        print(f"Match: {np.array_equal(result, examples[0][1])}")

    # Test 2: Rotation
    print("\nTest 2: Rotation")
    examples = [
        (np.array([[1, 2], [3, 4]]), np.array([[2, 4], [1, 3]])),
        (np.array([[5, 6], [7, 8]]), np.array([[6, 8], [5, 7]])),
    ]

    program = synthesizer.synthesize(examples)
    if program:
        print(f"Found program: {program}")
        result = dsl.execute_program(program, examples[0][0])
        print(f"Match: {np.array_equal(result, examples[0][1])}")

    # Test 3: Tiling
    print("\nTest 3: Pattern tiling")
    examples = [
        (
            np.array([[1, 2], [3, 4]]),
            np.array([[1, 2, 1, 2], [3, 4, 3, 4], [1, 2, 1, 2], [3, 4, 3, 4]]),
        ),
    ]

    program = synthesizer.synthesize(examples)
    if program:
        print(f"Found program: {program}")
        result = dsl.execute_program(program, examples[0][0])
        print(f"Match: {np.array_equal(result, examples[0][1])}")


if __name__ == "__main__":
    test_synthesis()
