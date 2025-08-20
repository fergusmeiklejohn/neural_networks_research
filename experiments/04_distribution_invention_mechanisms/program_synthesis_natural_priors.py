"""Program Synthesis with Natural Priors.

This module generates human-like programs for transformations by:
1. Preferring simple over complex solutions (Occam's razor)
2. Using causal understanding to guide synthesis
3. Respecting invariants and symmetries
4. Composing operations in natural ways
"""

import itertools
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from causal_reasoning_module import (
    CausalReasoningModule,
    Invariant,
    TransformationPrinciple,
)
from pattern_grammar_learner import AtomicOperation, PatternGrammarLearner


class ComplexityPrior(Enum):
    """Natural priors for program complexity."""

    SIMPLE = 1  # Single operation
    COMPOSITIONAL = 2  # 2-3 operations
    COMPLEX = 3  # 4+ operations
    RECURSIVE = 4  # Self-referential


@dataclass
class ProgramNode:
    """Node in a program AST."""

    operation: str
    node_type: str  # 'atomic', 'composite', 'conditional', 'loop'
    children: List["ProgramNode"] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)

    def complexity(self) -> int:
        """Calculate program complexity."""
        if self.node_type == "atomic":
            return 1
        elif self.node_type == "composite":
            return sum(child.complexity() for child in self.children)
        elif self.node_type == "conditional":
            return 2 + sum(child.complexity() for child in self.children)
        elif self.node_type == "loop":
            return 3 + sum(child.complexity() for child in self.children)
        return 1

    def to_code(self, indent: int = 0) -> str:
        """Convert to readable pseudocode."""
        spaces = "  " * indent

        if self.node_type == "atomic":
            params = ", ".join(f"{k}={v}" for k, v in self.parameters.items())
            return f"{spaces}{self.operation}({params})"

        elif self.node_type == "composite":
            code = f"{spaces}compose:\n"
            for child in self.children:
                code += child.to_code(indent + 1) + "\n"
            return code.rstrip()

        elif self.node_type == "conditional":
            condition = self.parameters.get("condition", "unknown")
            code = f"{spaces}if {condition}:\n"
            if self.children:
                code += self.children[0].to_code(indent + 1)
            if len(self.children) > 1:
                code += f"\n{spaces}else:\n"
                code += self.children[1].to_code(indent + 1)
            return code

        elif self.node_type == "loop":
            iterations = self.parameters.get("iterations", "unknown")
            code = f"{spaces}repeat {iterations} times:\n"
            for child in self.children:
                code += child.to_code(indent + 1) + "\n"
            return code.rstrip()

        return f"{spaces}{self.operation}"


@dataclass
class SynthesizedProgram:
    """A synthesized program with metadata."""

    root: ProgramNode
    score: float
    complexity: int
    respects_invariants: bool
    follows_causality: bool
    description: str

    def execute(self, input_array: np.ndarray) -> np.ndarray:
        """Execute the program on input."""
        return self._execute_node(self.root, input_array)

    def _execute_node(self, node: ProgramNode, arr: np.ndarray) -> np.ndarray:
        """Execute a single node."""
        if node.node_type == "atomic":
            return self._execute_atomic(node.operation, arr, node.parameters)

        elif node.node_type == "composite":
            result = arr
            for child in node.children:
                result = self._execute_node(child, result)
            return result

        elif node.node_type == "conditional":
            if self._evaluate_condition(node.parameters.get("condition"), arr):
                return self._execute_node(node.children[0], arr)
            elif len(node.children) > 1:
                return self._execute_node(node.children[1], arr)
            return arr

        elif node.node_type == "loop":
            result = arr
            iterations = node.parameters.get("iterations", 1)
            for _ in range(iterations):
                for child in node.children:
                    result = self._execute_node(child, result)
            return result

        return arr

    def _execute_atomic(
        self, operation: str, arr: np.ndarray, params: Dict
    ) -> np.ndarray:
        """Execute atomic operation."""
        if operation == "rotate":
            degrees = params.get("degrees", 90)
            k = degrees // 90
            return np.rot90(arr, k)

        elif operation == "flip_vertical":
            return np.flipud(arr)

        elif operation == "flip_horizontal":
            return np.fliplr(arr)

        elif operation == "scale":
            factor = params.get("factor", 2)
            if factor > 1:
                return np.repeat(
                    np.repeat(arr, int(factor), axis=0), int(factor), axis=1
                )
            else:
                n = int(1 / factor)
                return arr[::n, ::n]

        elif operation == "translate":
            dy = params.get("dy", 0)
            dx = params.get("dx", 0)
            return np.roll(np.roll(arr, dy, axis=0), dx, axis=1)

        elif operation == "fill":
            color = params.get("color", 1)
            mask = params.get("mask", arr == 0)
            result = arr.copy()
            result[mask] = color
            return result

        elif operation == "color_map":
            mapping = params.get("mapping", {})
            result = arr.copy()
            for old_color, new_color in mapping.items():
                result[arr == old_color] = new_color
            return result

        return arr

    def _evaluate_condition(self, condition: str, arr: np.ndarray) -> bool:
        """Evaluate a condition."""
        if condition == "has_pattern":
            return np.any(arr != 0)
        elif condition == "is_square":
            return arr.shape[0] == arr.shape[1]
        elif condition == "has_symmetry":
            return np.array_equal(arr, np.flipud(arr)) or np.array_equal(
                arr, np.fliplr(arr)
            )
        return False


class ProgramSynthesizer:
    """Synthesizes programs with natural priors."""

    def __init__(
        self,
        grammar_learner: Optional[PatternGrammarLearner] = None,
        causal_module: Optional[CausalReasoningModule] = None,
        verbose: bool = True,
    ):
        self.grammar_learner = grammar_learner or PatternGrammarLearner(verbose=False)
        self.causal_module = causal_module or CausalReasoningModule(verbose=False)
        self.verbose = verbose

        # Natural priors
        self.simplicity_weight = 0.3
        self.causality_weight = 0.2
        self.invariant_weight = 0.2
        self.symmetry_weight = 0.15
        self.compositionality_weight = 0.15

    def synthesize(
        self, examples: List[Tuple[np.ndarray, np.ndarray]], max_programs: int = 10
    ) -> List[SynthesizedProgram]:
        """Synthesize programs from examples."""

        if self.verbose:
            print(f"Synthesizing programs from {len(examples)} examples...")

        # Analyze examples
        causal_analysis = self.causal_module.analyze_transformation(examples)
        atomic_ops = self.grammar_learner._extract_atomic_operations(examples)

        # Generate candidate programs
        candidates = self._generate_candidates(
            examples, atomic_ops, causal_analysis, max_programs * 3
        )

        # Score and rank programs
        scored_programs = []
        for program in candidates:
            score = self._score_program(program, examples, causal_analysis)
            program.score = score
            scored_programs.append(program)

        # Sort by score and return top programs
        scored_programs.sort(key=lambda p: -p.score)

        if self.verbose and scored_programs:
            print(f"\nTop synthesized program (score: {scored_programs[0].score:.2f}):")
            print(scored_programs[0].root.to_code())

        return scored_programs[:max_programs]

    def _generate_candidates(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]],
        atomic_ops: Set[AtomicOperation],
        causal_analysis: Dict,
        max_candidates: int,
    ) -> List[SynthesizedProgram]:
        """Generate candidate programs."""
        candidates = []

        # 1. Simple programs (single operations)
        for op in atomic_ops:
            node = self._atomic_to_node(op)
            program = SynthesizedProgram(
                root=node,
                score=0,
                complexity=1,
                respects_invariants=True,
                follows_causality=True,
                description=f"Simple {op.name}",
            )
            candidates.append(program)

        # 2. Compositional programs (2-3 operations)
        if len(atomic_ops) >= 2:
            for ops in itertools.combinations(atomic_ops, 2):
                # Sequential composition
                node = ProgramNode(
                    operation="sequence",
                    node_type="composite",
                    children=[self._atomic_to_node(op) for op in ops],
                )
                program = SynthesizedProgram(
                    root=node,
                    score=0,
                    complexity=2,
                    respects_invariants=self._check_invariants(ops, causal_analysis),
                    follows_causality=self._check_causality(ops, causal_analysis),
                    description=f"Compose {ops[0].name} then {ops[1].name}",
                )
                candidates.append(program)

        # 3. Conditional programs (based on causal analysis)
        if causal_analysis.get("invariants"):
            for op in atomic_ops:
                # Create conditional based on invariants
                condition = self._invariant_to_condition(
                    causal_analysis["invariants"][0]
                )
                if condition:
                    node = ProgramNode(
                        operation="conditional",
                        node_type="conditional",
                        children=[self._atomic_to_node(op)],
                        parameters={"condition": condition},
                    )
                    program = SynthesizedProgram(
                        root=node,
                        score=0,
                        complexity=2,
                        respects_invariants=True,
                        follows_causality=True,
                        description=f"If {condition} then {op.name}",
                    )
                    candidates.append(program)

        # 4. Loop-based programs (for repetitive patterns)
        for op in atomic_ops:
            if self._is_idempotent(op):
                continue  # Skip operations that don't change with repetition

            for iterations in [2, 3, 4]:
                node = ProgramNode(
                    operation="loop",
                    node_type="loop",
                    children=[self._atomic_to_node(op)],
                    parameters={"iterations": iterations},
                )
                program = SynthesizedProgram(
                    root=node,
                    score=0,
                    complexity=1 + iterations,
                    respects_invariants=True,
                    follows_causality=True,
                    description=f"Repeat {op.name} {iterations} times",
                )
                candidates.append(program)

        # 5. Programs guided by causal mechanisms
        if causal_analysis.get("mechanisms"):
            mech = causal_analysis["mechanisms"]
            if mech["transformation_type"] == "rotation":
                degrees = mech["parameters"].get("degrees", 90)
                node = ProgramNode(
                    operation="rotate",
                    node_type="atomic",
                    parameters={"degrees": degrees},
                )
                program = SynthesizedProgram(
                    root=node,
                    score=0,
                    complexity=1,
                    respects_invariants=True,
                    follows_causality=True,
                    description=f"Rotate {degrees} degrees (from causal analysis)",
                )
                candidates.append(program)

        return candidates[:max_candidates]

    def _atomic_to_node(self, op: AtomicOperation) -> ProgramNode:
        """Convert atomic operation to program node."""
        # Map operation types to program operations
        if op.operation_type == "spatial":
            if "rotate" in op.name:
                degrees = op.parameters.get("degrees", 90)
                return ProgramNode("rotate", "atomic", parameters={"degrees": degrees})
            elif "flip" in op.name:
                if "vertical" in op.name:
                    return ProgramNode("flip_vertical", "atomic")
                else:
                    return ProgramNode("flip_horizontal", "atomic")
            elif "scale" in op.name:
                factor = op.parameters.get("factor", 2)
                return ProgramNode("scale", "atomic", parameters={"factor": factor})
            elif "translate" in op.name:
                dy = op.parameters.get("dy", 0)
                dx = op.parameters.get("dx", 0)
                return ProgramNode(
                    "translate", "atomic", parameters={"dy": dy, "dx": dx}
                )

        elif op.operation_type == "color":
            if "fill" in op.name:
                return ProgramNode("fill", "atomic", parameters={"color": 1})
            elif "map" in op.name:
                return ProgramNode("color_map", "atomic", parameters={"mapping": {}})

        # Default
        return ProgramNode(op.name, "atomic", parameters=op.parameters)

    def _check_invariants(
        self, ops: Tuple[AtomicOperation, ...], causal_analysis: Dict
    ) -> bool:
        """Check if operations respect detected invariants."""
        invariants = causal_analysis.get("invariants", [])

        for inv in invariants:
            if inv.invariant_type == "shape" and inv.value:
                # Check if operations preserve shape
                for op in ops:
                    if "scale" in op.name or "crop" in op.name:
                        return False

            elif inv.invariant_type == "count" and inv.value:
                # Check if operations preserve element count
                for op in ops:
                    if "fill" in op.name or "delete" in op.name:
                        return False

        return True

    def _check_causality(
        self, ops: Tuple[AtomicOperation, ...], causal_analysis: Dict
    ) -> bool:
        """Check if operations follow causal relations."""
        causal_analysis.get("causal_relations", [])

        # Simple check: operations should respect causal ordering
        # In full implementation, would check detailed causal constraints
        return True

    def _invariant_to_condition(self, invariant: Invariant) -> Optional[str]:
        """Convert invariant to a condition."""
        if invariant.invariant_type == "shape":
            if invariant.name == "shape":
                return "is_square"
        elif invariant.invariant_type == "spatial":
            if "symmetry" in invariant.description.lower():
                return "has_symmetry"
        elif invariant.invariant_type == "count":
            if invariant.value:
                return "has_pattern"
        return None

    def _is_idempotent(self, op: AtomicOperation) -> bool:
        """Check if operation is idempotent (doesn't change with repetition)."""
        idempotent_ops = {"flip_vertical", "flip_horizontal", "color_map"}
        return any(idm in op.name for idm in idempotent_ops)

    def _score_program(
        self,
        program: SynthesizedProgram,
        examples: List[Tuple[np.ndarray, np.ndarray]],
        causal_analysis: Dict,
    ) -> float:
        """Score a program based on natural priors."""
        score = 0.0

        # 1. Correctness (most important)
        correct_count = 0
        for inp, expected_out in examples:
            try:
                predicted_out = program.execute(inp)
                if predicted_out.shape == expected_out.shape:
                    if np.array_equal(predicted_out, expected_out):
                        correct_count += 1
            except:
                pass

        correctness = correct_count / len(examples) if examples else 0
        score += correctness * 0.5  # 50% weight on correctness

        # 2. Simplicity prior (Occam's razor)
        complexity_penalty = 1.0 / (1.0 + program.complexity)
        score += complexity_penalty * self.simplicity_weight

        # 3. Causal consistency
        if program.follows_causality:
            score += self.causality_weight

        # 4. Invariant respect
        if program.respects_invariants:
            score += self.invariant_weight

        # 5. Symmetry preference (humans prefer symmetric solutions)
        if self._has_symmetry(program):
            score += self.symmetry_weight

        # 6. Compositionality (prefer natural compositions)
        if 2 <= program.complexity <= 3:
            score += self.compositionality_weight

        return score

    def _has_symmetry(self, program: SynthesizedProgram) -> bool:
        """Check if program has symmetric structure."""
        # Simple check: programs with balanced composition
        if program.root.node_type == "composite":
            return len(program.root.children) == 2
        return False

    def explain_synthesis(
        self, program: SynthesizedProgram, causal_analysis: Dict
    ) -> str:
        """Generate explanation for why this program was synthesized."""
        explanation = "PROGRAM SYNTHESIS EXPLANATION\n"
        explanation += "=" * 50 + "\n\n"

        explanation += f"Program structure:\n{program.root.to_code()}\n\n"

        explanation += f"Synthesis reasoning:\n"
        explanation += f"  • Complexity: {program.complexity} operations\n"
        explanation += f"  • Score: {program.score:.2f}\n"
        explanation += f"  • Respects invariants: {program.respects_invariants}\n"
        explanation += f"  • Follows causality: {program.follows_causality}\n\n"

        if causal_analysis.get("invariants"):
            explanation += "Preserved invariants:\n"
            for inv in causal_analysis["invariants"][:3]:
                explanation += f"  • {inv.description}\n"
            explanation += "\n"

        if causal_analysis.get("principle"):
            prin = causal_analysis["principle"]
            explanation += f"Guided by principle: {prin.name}\n"
            explanation += f"  {prin.description}\n\n"

        explanation += "Natural priors applied:\n"
        explanation += (
            f"  • Simplicity (Occam's razor): Prefer {program.complexity} operations\n"
        )
        explanation += f"  • Compositionality: Natural operation ordering\n"
        explanation += f"  • Symmetry: Balanced program structure\n"

        return explanation

    def generate_novel_program(
        self,
        principle: TransformationPrinciple,
        target_complexity: ComplexityPrior = ComplexityPrior.SIMPLE,
    ) -> SynthesizedProgram:
        """Generate a novel program based on a principle."""

        if principle.name == "rotation_principle":
            # Generate variations of rotation
            if target_complexity == ComplexityPrior.SIMPLE:
                node = ProgramNode("rotate", "atomic", parameters={"degrees": 180})
            elif target_complexity == ComplexityPrior.COMPOSITIONAL:
                node = ProgramNode(
                    "sequence",
                    "composite",
                    children=[
                        ProgramNode("rotate", "atomic", parameters={"degrees": 90}),
                        ProgramNode("flip_vertical", "atomic"),
                    ],
                )
            else:
                node = ProgramNode(
                    "loop",
                    "loop",
                    children=[
                        ProgramNode("rotate", "atomic", parameters={"degrees": 90})
                    ],
                    parameters={"iterations": 2},
                )

        elif principle.name == "scaling_principle":
            if target_complexity == ComplexityPrior.SIMPLE:
                node = ProgramNode("scale", "atomic", parameters={"factor": 3})
            else:
                node = ProgramNode(
                    "sequence",
                    "composite",
                    children=[
                        ProgramNode("scale", "atomic", parameters={"factor": 2}),
                        ProgramNode(
                            "translate", "atomic", parameters={"dy": 1, "dx": 1}
                        ),
                    ],
                )

        else:
            # Default: simple transformation
            node = ProgramNode("flip_horizontal", "atomic")

        return SynthesizedProgram(
            root=node,
            score=0.5,
            complexity=node.complexity(),
            respects_invariants=True,
            follows_causality=True,
            description=f"Novel program from {principle.name}",
        )
