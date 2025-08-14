#!/usr/bin/env python3
"""Neural-Guided Program Synthesis for ARC-AGI.

Combines bidirectional synthesis with neural guidance for efficient program discovery.
"""

from utils.imports import setup_project_paths

setup_project_paths()

import heapq
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
from bidirectional_synthesis import BottomUpEnumerator
from compositional_dsl import CompositionalDSL, ExecutionContext, Primitive, Sequence
from neural_program_ranker import NeuralProgramRanker, ProgramTokenizer


@dataclass
class GuidedSearchNode:
    """Search node with neural guidance score."""

    program: Primitive
    neural_score: float  # From neural ranker
    empirical_score: float  # From actual execution
    combined_score: float  # Weighted combination
    depth: int

    def __lt__(self, other):
        return self.combined_score > other.combined_score


class NeuralGuidedSynthesizer:
    """Program synthesizer with neural guidance."""

    def __init__(
        self,
        dsl: CompositionalDSL,
        neural_ranker: Optional[NeuralProgramRanker] = None,
        beam_width: int = 50,
        max_depth: int = 5,
        neural_weight: float = 0.3,
        timeout: float = 30.0,
    ):
        """Initialize the neural-guided synthesizer.

        Args:
            dsl: The compositional DSL
            neural_ranker: Trained neural program ranker (optional)
            beam_width: Beam size for search
            max_depth: Maximum program depth
            neural_weight: Weight for neural score (0-1)
            timeout: Search timeout in seconds
        """
        self.dsl = dsl
        self.neural_ranker = neural_ranker
        self.beam_width = beam_width
        self.max_depth = max_depth
        self.neural_weight = neural_weight
        self.empirical_weight = 1.0 - neural_weight
        self.timeout = timeout
        self.tokenizer = ProgramTokenizer()
        self.enumerator = BottomUpEnumerator(dsl)

        # Use CPU for neural ranker if no GPU
        if self.neural_ranker:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.neural_ranker = self.neural_ranker.to(self.device)
            self.neural_ranker.eval()

    def synthesize(
        self, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[Primitive]:
        """Synthesize a program using neural-guided beam search.

        Args:
            examples: List of (input, output) grid pairs

        Returns:
            Best program found or None
        """
        start_time = time.time()
        best_program = None
        best_score = 0.0

        # Initialize beam with atomic primitives
        beam = self._initialize_beam(examples)

        # Beam search with neural guidance
        for depth in range(1, self.max_depth + 1):
            if time.time() - start_time > self.timeout:
                print(f"Timeout reached after {depth} iterations")
                break

            new_beam = []

            for node in beam:
                # Check if we found a perfect solution
                if node.empirical_score >= 0.99:
                    print(f"Found perfect solution: {node.program}")
                    return node.program

                # Track best so far
                if node.combined_score > best_score:
                    best_score = node.combined_score
                    best_program = node.program

                # Expand node
                if depth < self.max_depth:
                    children = self._expand_node(node, examples, depth)
                    new_beam.extend(children)

            # Select top K for next iteration
            if new_beam:
                new_beam = heapq.nlargest(self.beam_width, new_beam)
                beam = new_beam
                print(f"Depth {depth}: Best score = {beam[0].combined_score:.3f}")
            else:
                break

        return best_program

    def _initialize_beam(
        self, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> List[GuidedSearchNode]:
        """Initialize beam with scored atomic primitives."""
        initial_nodes = []

        # Get atomic primitives
        for name, primitive_class in self.dsl.primitives.items():
            if name in ["sequence", "conditional", "loop", "for_each_object"]:
                continue

            # Generate parameter sets
            param_sets = self.enumerator._generate_parameters(name, examples)

            for params in param_sets[:5]:  # Limit initial candidates
                try:
                    primitive = self.dsl.get_primitive(name, **params)

                    # Get empirical score
                    empirical = self._evaluate_empirical(primitive, examples)

                    # Get neural score if available
                    neural = (
                        self._evaluate_neural(primitive, examples)
                        if self.neural_ranker
                        else empirical
                    )

                    # Combine scores
                    combined = (
                        self.neural_weight * neural + self.empirical_weight * empirical
                    )

                    node = GuidedSearchNode(
                        program=primitive,
                        neural_score=neural,
                        empirical_score=empirical,
                        combined_score=combined,
                        depth=1,
                    )

                    if combined > 0.1:  # Threshold to reduce noise
                        initial_nodes.append(node)

                except Exception:
                    continue

        # Return top beam_width nodes
        return heapq.nlargest(self.beam_width, initial_nodes)

    def _expand_node(
        self,
        parent: GuidedSearchNode,
        examples: List[Tuple[np.ndarray, np.ndarray]],
        depth: int,
    ) -> List[GuidedSearchNode]:
        """Expand a node by adding operations."""
        children = []

        # Try composing with other primitives
        for name, primitive_class in self.dsl.primitives.items():
            if name in ["conditional"]:  # Skip complex operators for now
                continue

            if name == "sequence":
                # Don't nest sequences too deep
                if (
                    isinstance(parent.program, Sequence)
                    and len(parent.program.primitives) >= 3
                ):
                    continue

            # Generate parameters
            param_sets = self.enumerator._generate_parameters(name, examples)

            for params in param_sets[:3]:  # Limit expansions
                try:
                    if name == "sequence":
                        # Add to existing sequence or create new one
                        if isinstance(parent.program, Sequence):
                            continue  # Skip for now to avoid deep nesting
                        else:
                            # Try adding a new primitive to parent
                            new_primitive = self._get_primitive_with_params(
                                name, params
                            )
                            if new_primitive:
                                new_program = Sequence([parent.program, new_primitive])
                            else:
                                continue
                    elif name == "loop":
                        # Loop the parent program
                        for times in [2, 3]:
                            from compositional_dsl import Loop

                            new_program = Loop(parent.program, times)
                            child = self._create_child_node(
                                new_program, examples, depth
                            )
                            if (
                                child
                                and child.combined_score > parent.combined_score * 0.9
                            ):
                                children.append(child)
                        continue
                    elif name == "for_each_object":
                        # Apply parent to each object
                        from compositional_dsl import ForEachObject

                        new_program = ForEachObject(parent.program)
                    else:
                        # Create sequence with new primitive
                        new_primitive = self.dsl.get_primitive(name, **params)
                        new_program = Sequence([parent.program, new_primitive])

                    # Create and score child node
                    child = self._create_child_node(new_program, examples, depth)
                    if child and child.combined_score > parent.combined_score * 0.9:
                        children.append(child)

                except Exception:
                    continue

        return children

    def _get_primitive_with_params(
        self, name: str, params: dict
    ) -> Optional[Primitive]:
        """Helper to get a primitive with parameters."""
        try:
            # For now, generate a simple primitive
            # This would need more sophisticated parameter generation
            if name == "set_color" and "from_color" in params:
                return self.dsl.get_primitive(name, **params)
            elif name == "rotate" and "angle" in params:
                return self.dsl.get_primitive(name, **params)
            else:
                return self.dsl.get_primitive("flip_h")  # Default
        except Exception:
            return None

    def _create_child_node(
        self,
        program: Primitive,
        examples: List[Tuple[np.ndarray, np.ndarray]],
        depth: int,
    ) -> Optional[GuidedSearchNode]:
        """Create and score a child node."""
        try:
            # Get empirical score
            empirical = self._evaluate_empirical(program, examples)

            # Skip if too poor
            if empirical < 0.1:
                return None

            # Get neural score
            neural = (
                self._evaluate_neural(program, examples)
                if self.neural_ranker
                else empirical
            )

            # Combine scores
            combined = self.neural_weight * neural + self.empirical_weight * empirical

            return GuidedSearchNode(
                program=program,
                neural_score=neural,
                empirical_score=empirical,
                combined_score=combined,
                depth=depth,
            )

        except Exception:
            return None

    def _evaluate_empirical(
        self, program: Primitive, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> float:
        """Evaluate program empirically on examples."""
        total_score = 0.0

        for inp, expected in examples:
            try:
                context = ExecutionContext(
                    input_grid=inp.copy(), current_grid=inp.copy()
                )
                result = program.execute(context)
                output = result.current_grid

                # Exact match
                if np.array_equal(output, expected):
                    total_score += 1.0
                elif output.shape == expected.shape:
                    # Partial credit
                    matching = np.sum(output == expected)
                    total = output.size
                    total_score += 0.5 * (matching / total)

            except Exception:
                continue

        return total_score / len(examples)

    def _evaluate_neural(
        self, program: Primitive, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> float:
        """Evaluate program using neural ranker."""
        if not self.neural_ranker:
            return 0.5  # Default score

        try:
            # Tokenize program
            program_str = str(program)
            program_tokens = (
                self.tokenizer.tokenize(program_str).unsqueeze(0).to(self.device)
            )

            # Prepare grids
            input_grids = [torch.tensor(inp, device=self.device) for inp, _ in examples]
            output_grids = [
                torch.tensor(out, device=self.device) for _, out in examples
            ]

            # Get neural score
            with torch.no_grad():
                logit = self.neural_ranker(program_tokens, input_grids, output_grids)
                score = torch.sigmoid(logit).item()

            return score

        except Exception:
            return 0.5  # Default on error


def test_neural_guided_synthesis():
    """Test the neural-guided synthesis system."""
    print("Testing Neural-Guided Synthesis\n")

    # Create DSL and synthesizer
    dsl = CompositionalDSL()

    # Create neural ranker (untrained for now)
    neural_ranker = NeuralProgramRanker(
        vocab_size=50, hidden_dim=128, num_heads=4, num_layers=2
    )

    # Create synthesizer
    synthesizer = NeuralGuidedSynthesizer(
        dsl,
        neural_ranker=neural_ranker,
        beam_width=20,
        max_depth=3,
        neural_weight=0.2,  # Lower weight since untrained
    )

    # Test 1: Simple color transformation
    print("Test 1: Color transformation")
    examples = [
        (
            np.array([[1, 2, 1], [2, 1, 2], [1, 2, 1]]),
            np.array([[3, 4, 3], [4, 3, 4], [3, 4, 3]]),
        ),
        (
            np.array([[1, 1, 2], [1, 2, 2], [2, 1, 1]]),
            np.array([[3, 3, 4], [3, 4, 4], [4, 3, 3]]),
        ),
    ]

    program = synthesizer.synthesize(examples)
    if program:
        print(f"Found program: {program}")
        # Test on first example
        result = dsl.execute_program(program, examples[0][0])
        match = np.array_equal(result, examples[0][1])
        print(f"Correct: {match}\n")
    else:
        print("No program found\n")

    # Test 2: Spatial transformation
    print("Test 2: Rotation")
    examples = [
        (np.array([[1, 2], [3, 4]]), np.array([[2, 4], [1, 3]])),
        (np.array([[5, 6], [7, 8]]), np.array([[6, 8], [5, 7]])),
        (np.array([[1, 0], [0, 1]]), np.array([[0, 1], [1, 0]])),
    ]

    program = synthesizer.synthesize(examples)
    if program:
        print(f"Found program: {program}")
        result = dsl.execute_program(program, examples[0][0])
        match = np.array_equal(result, examples[0][1])
        print(f"Correct: {match}\n")
    else:
        print("No program found\n")

    # Test 3: Tiling
    print("Test 3: Tiling pattern")
    examples = [
        (
            np.array([[1, 2], [3, 4]]),
            np.array([[1, 2, 1, 2], [3, 4, 3, 4], [1, 2, 1, 2], [3, 4, 3, 4]]),
        ),
        (
            np.array([[5, 6], [7, 8]]),
            np.array([[5, 6, 5, 6], [7, 8, 7, 8], [5, 6, 5, 6], [7, 8, 7, 8]]),
        ),
    ]

    program = synthesizer.synthesize(examples)
    if program:
        print(f"Found program: {program}")
        result = dsl.execute_program(program, examples[0][0])
        match = np.array_equal(result, examples[0][1])
        print(f"Correct: {match}\n")
    else:
        print("No program found\n")


if __name__ == "__main__":
    test_neural_guided_synthesis()
