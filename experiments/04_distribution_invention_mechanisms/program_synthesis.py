#!/usr/bin/env python3
"""Program Synthesis Engine for ARC-AGI tasks.

This module implements search algorithms to synthesize programs from
input-output examples using the ARC DSL.
"""

from utils.imports import setup_project_paths

setup_project_paths()

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from arc_dsl import DSLLibrary, Program
from enhanced_neural_perception import EnhancedNeuralPerception


@dataclass
class SearchNode:
    """Node in the program search tree."""

    program: Program
    score: float
    depth: int

    def __lt__(self, other):
        # For heap - higher score is better
        return self.score > other.score


@dataclass
class SynthesisResult:
    """Result from program synthesis."""

    best_program: Optional[Program]
    score: float
    candidates_explored: int
    success: bool


class BeamSearch:
    """Beam search strategy for program synthesis."""

    def __init__(self, beam_width: int = 50, max_depth: int = 5):
        self.beam_width = beam_width
        self.max_depth = max_depth
        self.dsl_library = DSLLibrary()
        self.perception = EnhancedNeuralPerception()

    def search(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]],
        initial_candidates: List[Program] = None,
    ) -> SynthesisResult:
        """Search for a program that solves the examples.

        Args:
            examples: List of (input, output) grid pairs
            initial_candidates: Optional initial programs to start from

        Returns:
            SynthesisResult with best program found
        """
        # Initialize beam with empty program or initial candidates
        if initial_candidates:
            beam = [
                SearchNode(prog, self._score_program(prog, examples), 0)
                for prog in initial_candidates
            ]
        else:
            beam = [SearchNode(Program([]), 0.0, 0)]

        best_program = None
        best_score = 0.0
        candidates_explored = 0

        for depth in range(self.max_depth):
            new_beam = []

            for node in beam:
                # Check if current program solves all examples
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
                    )

                # Generate children by adding one more operation
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
        )

    def _expand_node(
        self, node: SearchNode, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> List[SearchNode]:
        """Expand a node by adding possible next operations."""
        children = []

        # Get candidate operations based on analysis
        candidate_ops = self._get_candidate_operations(node.program, examples)

        for op_name, kwargs in candidate_ops:
            try:
                # Create new primitive
                primitive = self.dsl_library.get_primitive(op_name, **kwargs)

                # Create new program with this operation added
                new_operations = node.program.operations + [primitive]
                new_program = Program(new_operations)

                # Score the new program
                score = self._score_program(new_program, examples)

                # Create new node
                child = SearchNode(new_program, score, node.depth + 1)
                children.append(child)

            except Exception:
                # Skip invalid operations
                continue

        return children

    def _get_candidate_operations(
        self, current_program: Program, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> List[Tuple[str, Dict]]:
        """Get candidate operations to try next."""
        candidates = []

        # Analyze first example to understand transformation
        if examples:
            input_grid, output_grid = examples[0]
            input_analysis = self.perception.analyze_grid(input_grid)
            output_analysis = self.perception.analyze_grid(output_grid)
            comparison = self.perception.compare_grids(input_grid, output_grid)

            # Based on comparison, suggest operations
            if comparison.get("likely_transformation"):
                trans = comparison["likely_transformation"]

                if trans == "horizontal_flip":
                    candidates.append(("mirror", {"axis": "horizontal"}))
                elif trans == "vertical_flip":
                    candidates.append(("mirror", {"axis": "vertical"}))
                elif trans == "2x_scaling":
                    candidates.append(("scale", {"factor": 2}))
                elif trans == "rotation_90":
                    candidates.append(("rotate", {"degrees": 90}))
                elif trans == "color_mapping" and comparison.get("color_changes"):
                    # Add replace operations for each color change
                    for old_c, new_c in comparison["color_changes"].items():
                        candidates.append(
                            ("replace", {"old_color": old_c, "new_color": new_c})
                        )

            # Check for object-based transformations
            input_obj_count = input_analysis["counting"]["total_objects"]
            output_obj_count = output_analysis["counting"]["total_objects"]

            if output_obj_count > input_obj_count:
                candidates.append(("duplicate", {"pattern": "double"}))
            elif output_obj_count < input_obj_count:
                candidates.append(("filter_size", {"min_size": 3}))
                candidates.append(("remove", {}))

            # Check for pattern completion
            if output_analysis.get("patterns") and not input_analysis.get("patterns"):
                candidates.append(("complete", {}))

            # Check for rearrangement
            if input_obj_count == output_obj_count and input_obj_count > 0:
                candidates.append(("rearrange", {"arrangement": "sort_by_size"}))
                candidates.append(("rearrange", {"arrangement": "center"}))

        # Always include some common operations
        candidates.extend(
            [
                ("mirror", {"axis": "horizontal"}),
                ("mirror", {"axis": "vertical"}),
                ("rotate", {"degrees": 90}),
                ("rotate", {"degrees": 180}),
                ("scale", {"factor": 2}),
                ("filter_size", {"min_size": 1, "max_size": 10}),
                ("duplicate", {"pattern": "double"}),
                ("complete", {}),
                ("fill", {"color": 1, "fill_type": "background"}),
            ]
        )

        # Limit number of candidates to avoid explosion
        random.shuffle(candidates)
        return candidates[:20]

    def _score_program(
        self, program: Program, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> float:
        """Score how well a program solves the examples."""
        if not program.operations:
            return 0.0

        total_score = 0.0

        for input_grid, expected_output in examples:
            try:
                # Execute program
                predicted = program.execute(input_grid)

                # Check exact match
                if np.array_equal(predicted, expected_output):
                    total_score += 1.0
                else:
                    # Partial credit based on similarity
                    similarity = self._grid_similarity(predicted, expected_output)
                    total_score += similarity

            except Exception:
                # Program failed on this example
                total_score += 0.0

        # Average score across examples
        return total_score / len(examples) if examples else 0.0

    def _grid_similarity(self, grid1: np.ndarray, grid2: np.ndarray) -> float:
        """Calculate similarity between two grids."""
        # Handle different sizes
        if grid1.shape != grid2.shape:
            # Resize penalty but still check content
            size_penalty = 0.5

            # Crop or pad to match sizes
            min_rows = min(grid1.shape[0], grid2.shape[0])
            min_cols = min(grid1.shape[1], grid2.shape[1])

            grid1_cropped = grid1[:min_rows, :min_cols]
            grid2_cropped = grid2[:min_rows, :min_cols]

            matches = np.sum(grid1_cropped == grid2_cropped)
            total = min_rows * min_cols

            return (matches / total) * size_penalty if total > 0 else 0.0

        # Same size - calculate pixel accuracy
        matches = np.sum(grid1 == grid2)
        total = grid1.size

        return matches / total if total > 0 else 0.0


class GeneticSearch:
    """Genetic programming approach to program synthesis."""

    def __init__(
        self,
        population_size: int = 50,
        generations: int = 30,
        mutation_rate: float = 0.3,
        crossover_rate: float = 0.7,
    ):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.dsl_library = DSLLibrary()
        self.perception = EnhancedNeuralPerception()

    def search(self, examples: List[Tuple[np.ndarray, np.ndarray]]) -> SynthesisResult:
        """Search using genetic programming."""
        # Initialize population
        population = self._initialize_population()

        best_program = None
        best_score = 0.0
        candidates_explored = 0

        for generation in range(self.generations):
            # Evaluate fitness
            fitness_scores = []
            for program in population:
                score = self._score_program(program, examples)
                fitness_scores.append(score)
                candidates_explored += 1

                if score > best_score:
                    best_score = score
                    best_program = program

                # Early termination if solution found
                if score >= 0.99:
                    return SynthesisResult(
                        best_program=program,
                        score=score,
                        candidates_explored=candidates_explored,
                        success=True,
                    )

            # Selection and reproduction
            new_population = []

            # Keep best individuals (elitism)
            elite_size = self.population_size // 10
            elite_indices = np.argsort(fitness_scores)[-elite_size:]
            for idx in elite_indices:
                new_population.append(population[idx])

            # Generate rest of population
            while len(new_population) < self.population_size:
                # Selection
                parent1 = self._tournament_select(population, fitness_scores)
                parent2 = self._tournament_select(population, fitness_scores)

                # Crossover
                if random.random() < self.crossover_rate:
                    child = self._crossover(parent1, parent2)
                else:
                    child = parent1 if random.random() < 0.5 else parent2

                # Mutation
                if random.random() < self.mutation_rate:
                    child = self._mutate(child)

                new_population.append(child)

            population = new_population

        return SynthesisResult(
            best_program=best_program,
            score=best_score,
            candidates_explored=candidates_explored,
            success=best_score >= 0.99,
        )

    def _initialize_population(self) -> List[Program]:
        """Initialize random population of programs."""
        population = []

        for _ in range(self.population_size):
            # Random program length
            length = random.randint(1, 5)
            operations = []

            for _ in range(length):
                # Random operation
                op_name = random.choice(self.dsl_library.get_all_primitive_names())
                kwargs = self._get_random_kwargs(op_name)

                try:
                    primitive = self.dsl_library.get_primitive(op_name, **kwargs)
                    operations.append(primitive)
                except Exception:
                    continue

            if operations:
                population.append(Program(operations))
            else:
                # Fallback to simple operation
                population.append(Program([self.dsl_library.get_primitive("mirror")]))

        return population

    def _get_random_kwargs(self, op_name: str) -> Dict:
        """Get random kwargs for an operation."""
        if op_name == "move":
            return {
                "delta_row": random.randint(-3, 3),
                "delta_col": random.randint(-3, 3),
            }
        elif op_name == "rotate":
            return {"degrees": random.choice([90, 180, 270])}
        elif op_name == "mirror":
            return {"axis": random.choice(["horizontal", "vertical"])}
        elif op_name == "scale":
            return {"factor": random.choice([2, 3])}
        elif op_name == "color":
            return {"color": random.randint(1, 9)}
        elif op_name == "filter_size":
            return {"min_size": random.randint(1, 5), "max_size": random.randint(5, 20)}
        elif op_name == "filter_color":
            return {
                "colors": [random.randint(1, 9) for _ in range(random.randint(1, 3))]
            }
        elif op_name == "fill":
            return {
                "color": random.randint(1, 9),
                "fill_type": random.choice(["background", "objects"]),
            }
        elif op_name == "replace":
            return {
                "old_color": random.randint(0, 9),
                "new_color": random.randint(0, 9),
            }
        elif op_name == "duplicate":
            return {"pattern": random.choice(["double", "mirror", "tile"])}
        elif op_name == "rearrange":
            return {
                "arrangement": random.choice(
                    ["sort_by_size", "sort_by_color", "center"]
                )
            }
        else:
            return {}

    def _tournament_select(
        self,
        population: List[Program],
        fitness_scores: List[float],
        tournament_size: int = 3,
    ) -> Program:
        """Select individual using tournament selection."""
        indices = random.sample(range(len(population)), tournament_size)
        best_idx = max(indices, key=lambda i: fitness_scores[i])
        return population[best_idx]

    def _crossover(self, parent1: Program, parent2: Program) -> Program:
        """Create child by crossing over two parent programs."""
        if not parent1.operations or not parent2.operations:
            return parent1 if parent1.operations else parent2

        # Single-point crossover
        point1 = random.randint(0, len(parent1.operations))
        point2 = random.randint(0, len(parent2.operations))

        child_ops = parent1.operations[:point1] + parent2.operations[point2:]

        # Limit length
        if len(child_ops) > 7:
            child_ops = child_ops[:7]

        return Program(child_ops) if child_ops else parent1

    def _mutate(self, program: Program) -> Program:
        """Mutate a program."""
        if not program.operations:
            return program

        operations = program.operations.copy()
        mutation_type = random.choice(["add", "remove", "replace"])

        if mutation_type == "add" and len(operations) < 7:
            # Add random operation
            op_name = random.choice(self.dsl_library.get_all_primitive_names())
            kwargs = self._get_random_kwargs(op_name)
            try:
                primitive = self.dsl_library.get_primitive(op_name, **kwargs)
                position = random.randint(0, len(operations))
                operations.insert(position, primitive)
            except Exception:
                pass

        elif mutation_type == "remove" and len(operations) > 1:
            # Remove random operation
            idx = random.randint(0, len(operations) - 1)
            operations.pop(idx)

        elif mutation_type == "replace" and operations:
            # Replace random operation
            idx = random.randint(0, len(operations) - 1)
            op_name = random.choice(self.dsl_library.get_all_primitive_names())
            kwargs = self._get_random_kwargs(op_name)
            try:
                primitive = self.dsl_library.get_primitive(op_name, **kwargs)
                operations[idx] = primitive
            except Exception:
                pass

        return Program(operations)

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

                if np.array_equal(predicted, expected_output):
                    total_score += 1.0
                else:
                    # Partial credit
                    similarity = self._grid_similarity(predicted, expected_output)
                    total_score += similarity

            except Exception:
                total_score += 0.0

        return total_score / len(examples) if examples else 0.0

    def _grid_similarity(self, grid1: np.ndarray, grid2: np.ndarray) -> float:
        """Calculate similarity between two grids."""
        if grid1.shape != grid2.shape:
            size_penalty = 0.5
            min_rows = min(grid1.shape[0], grid2.shape[0])
            min_cols = min(grid1.shape[1], grid2.shape[1])

            grid1_cropped = grid1[:min_rows, :min_cols]
            grid2_cropped = grid2[:min_rows, :min_cols]

            matches = np.sum(grid1_cropped == grid2_cropped)
            total = min_rows * min_cols

            return (matches / total) * size_penalty if total > 0 else 0.0

        matches = np.sum(grid1 == grid2)
        total = grid1.size

        return matches / total if total > 0 else 0.0


class ProgramSynthesizer:
    """Main program synthesizer combining different search strategies."""

    def __init__(self, strategy: str = "beam"):
        """Initialize synthesizer.

        Args:
            strategy: 'beam' or 'genetic'
        """
        self.strategy = strategy
        self.dsl_library = DSLLibrary()
        self.perception = EnhancedNeuralPerception()

        if strategy == "beam":
            self.searcher = BeamSearch()
        elif strategy == "genetic":
            self.searcher = GeneticSearch()
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def synthesize(
        self, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Optional[Program]:
        """Synthesize a program from examples.

        Args:
            examples: List of (input, output) grid pairs

        Returns:
            Program that solves the examples, or None if not found
        """
        # Analyze examples for patterns
        patterns = self.analyze_patterns(examples)

        # Generate initial candidates based on patterns
        initial_candidates = self.generate_candidates(patterns)

        # Search for best program
        if self.strategy == "beam":
            result = self.searcher.search(examples, initial_candidates)
        else:
            result = self.searcher.search(examples)

        if result.success:
            return result.best_program

        return result.best_program if result.score > 0.5 else None

    def analyze_patterns(self, examples: List[Tuple[np.ndarray, np.ndarray]]) -> Dict:
        """Analyze examples to identify patterns."""
        patterns = {
            "transformations": [],
            "object_changes": [],
            "size_changes": [],
            "color_changes": [],
        }

        for input_grid, output_grid in examples:
            # Use perception module for analysis
            input_analysis = self.perception.analyze_grid(input_grid)
            output_analysis = self.perception.analyze_grid(output_grid)
            comparison = self.perception.compare_grids(input_grid, output_grid)

            if comparison.get("likely_transformation"):
                patterns["transformations"].append(comparison["likely_transformation"])

            if comparison.get("size_change"):
                patterns["size_changes"].append(comparison["size_change"])

            if comparison.get("color_changes"):
                patterns["color_changes"].append(comparison["color_changes"])

            # Object changes
            input_count = input_analysis["counting"]["total_objects"]
            output_count = output_analysis["counting"]["total_objects"]
            if input_count != output_count:
                patterns["object_changes"].append((input_count, output_count))

        return patterns

    def generate_candidates(self, patterns: Dict) -> List[Program]:
        """Generate initial candidate programs based on patterns."""
        candidates = []

        # Based on detected transformations
        for trans in patterns.get("transformations", []):
            if trans == "horizontal_flip":
                candidates.append(
                    Program(
                        [self.dsl_library.get_primitive("mirror", axis="horizontal")]
                    )
                )
            elif trans == "vertical_flip":
                candidates.append(
                    Program([self.dsl_library.get_primitive("mirror", axis="vertical")])
                )
            elif trans == "2x_scaling":
                candidates.append(
                    Program([self.dsl_library.get_primitive("scale", factor=2)])
                )
            elif trans == "rotation_90":
                candidates.append(
                    Program([self.dsl_library.get_primitive("rotate", degrees=90)])
                )

        # Based on object changes
        for input_count, output_count in patterns.get("object_changes", []):
            if output_count > input_count:
                candidates.append(
                    Program([self.dsl_library.get_primitive("duplicate")])
                )
            elif output_count < input_count:
                candidates.append(
                    Program([self.dsl_library.get_primitive("filter_size", min_size=3)])
                )

        return candidates


def test_program_synthesis():
    """Test program synthesis."""
    print("Testing Program Synthesis")
    print("=" * 50)

    # Create test examples (horizontal flip)
    input1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    output1 = np.array([[3, 2, 1], [6, 5, 4], [9, 8, 7]])

    input2 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    output2 = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])

    examples = [(input1, output1), (input2, output2)]

    print("Test examples:")
    print("Example 1:")
    print("Input:\n", input1)
    print("Output:\n", output1)
    print("\nExample 2:")
    print("Input:\n", input2)
    print("Output:\n", output2)
    print()

    # Test beam search
    print("Testing Beam Search:")
    print("-" * 30)
    synthesizer = ProgramSynthesizer(strategy="beam")
    program = synthesizer.synthesize(examples)

    if program:
        print("Found program!")
        print(program.to_code())

        # Test on examples
        print("\nTesting on examples:")
        for i, (inp, expected) in enumerate(examples):
            predicted = program.execute(inp)
            match = np.array_equal(predicted, expected)
            print(f"Example {i+1}: {'✓' if match else '✗'}")
    else:
        print("No program found")

    print()

    # Test genetic search
    print("Testing Genetic Search:")
    print("-" * 30)
    synthesizer = ProgramSynthesizer(strategy="genetic")
    program = synthesizer.synthesize(examples)

    if program:
        print("Found program!")
        print(program.to_code())

        # Test on examples
        print("\nTesting on examples:")
        for i, (inp, expected) in enumerate(examples):
            predicted = program.execute(inp)
            match = np.array_equal(predicted, expected)
            print(f"Example {i+1}: {'✓' if match else '✗'}")
    else:
        print("No program found")


if __name__ == "__main__":
    test_program_synthesis()
