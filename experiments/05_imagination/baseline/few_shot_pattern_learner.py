"""Few-Shot Pattern Learning System.

Learn to solve new patterns from just 3-4 examples, like humans do.
This implements hypothesis generation, testing, and abstraction.
"""

import itertools
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
from pattern_grammar_learner import AtomicOperation, PatternGrammarLearner


@dataclass
class Hypothesis:
    """Represents a hypothesis about a pattern."""

    name: str
    program: Callable
    confidence: float
    explanation: str
    operations: List[AtomicOperation]

    def test(self, inp: np.ndarray) -> np.ndarray:
        """Apply hypothesis to input."""
        return self.program(inp)


class ProgramGenerator:
    """Generates executable programs from atomic operations."""

    @staticmethod
    def compose_operations(operations: List[AtomicOperation]) -> Callable:
        """Compose multiple operations into a single program."""

        def program(inp: np.ndarray) -> np.ndarray:
            result = inp.copy()

            for op in operations:
                if op.operation_type == "spatial":
                    result = ProgramGenerator._apply_spatial(result, op)
                elif op.operation_type == "color":
                    result = ProgramGenerator._apply_color(result, op)
                elif op.operation_type == "object":
                    result = ProgramGenerator._apply_object(result, op)
                elif op.operation_type == "logical":
                    result = ProgramGenerator._apply_logical(result, op)
                elif op.operation_type == "arithmetic":
                    result = ProgramGenerator._apply_arithmetic(result, op)

            return result

        return program

    @staticmethod
    def _apply_spatial(inp: np.ndarray, op: AtomicOperation) -> np.ndarray:
        """Apply spatial operation."""
        if op.name.startswith("translate"):
            dy = op.parameters.get("dy", 0)
            dx = op.parameters.get("dx", 0)
            return np.roll(np.roll(inp, dy, axis=0), dx, axis=1)
        elif op.name.startswith("rotate"):
            degrees = op.parameters.get("degrees", 0)
            k = degrees // 90
            return np.rot90(inp, k)
        elif op.name == "flip_vertical":
            return np.flipud(inp)
        elif op.name == "flip_horizontal":
            return np.fliplr(inp)
        elif op.name.startswith("scale"):
            factor = op.parameters.get("factor", 1)
            if factor > 1:
                return np.repeat(
                    np.repeat(inp, int(factor), axis=0), int(factor), axis=1
                )
            else:
                # Downscale by taking every nth element
                n = int(1 / factor)
                return inp[::n, ::n]
        elif op.name == "crop":
            # Simple center crop
            h, w = inp.shape
            out_h = min(h, op.parameters.get("output_shape", (h, w))[0])
            out_w = min(w, op.parameters.get("output_shape", (h, w))[1])
            start_h = (h - out_h) // 2
            start_w = (w - out_w) // 2
            return inp[start_h : start_h + out_h, start_w : start_w + out_w]
        return inp

    @staticmethod
    def _apply_color(inp: np.ndarray, op: AtomicOperation) -> np.ndarray:
        """Apply color operation."""
        if op.name == "add_colors":
            # Add new color to empty spaces
            colors = op.parameters.get("colors", [])
            if colors and 0 in inp:
                result = inp.copy()
                empty_mask = result == 0
                if np.any(empty_mask):
                    # Add first new color to some empty spaces
                    result[empty_mask] = colors[0]
                return result
        elif op.name == "color_map":
            # Simple color swap
            result = inp.copy()
            unique_colors = np.unique(inp)
            if len(unique_colors) > 1:
                # Swap first two colors
                c1, c2 = unique_colors[0], unique_colors[1]
                mask1 = inp == c1
                mask2 = inp == c2
                result[mask1] = c2
                result[mask2] = c1
            return result
        return inp

    @staticmethod
    def _apply_object(inp: np.ndarray, op: AtomicOperation) -> np.ndarray:
        """Apply object operation."""
        # Simplified object operations
        return inp

    @staticmethod
    def _apply_logical(inp: np.ndarray, op: AtomicOperation) -> np.ndarray:
        """Apply logical operation."""
        if op.name == "conditional_fill":
            # Fill based on neighbor count
            from scipy import ndimage

            result = inp.copy()
            # Count neighbors for each position
            kernel = np.ones((3, 3))
            kernel[1, 1] = 0
            neighbor_count = ndimage.convolve(inp != 0, kernel, mode="constant")
            # Fill positions with many neighbors
            result[neighbor_count >= 3] = 1
            return result
        return inp

    @staticmethod
    def _apply_arithmetic(inp: np.ndarray, op: AtomicOperation) -> np.ndarray:
        """Apply arithmetic operation."""
        if op.name == "extract_property":
            # Extract most common non-zero value
            nonzero = inp[inp != 0]
            if len(nonzero) > 0:
                from collections import Counter

                most_common = Counter(nonzero).most_common(1)[0][0]
                return np.array([[most_common]])
        return inp


class FewShotPatternLearner:
    """Learn patterns from few examples using hypothesis generation and testing."""

    def __init__(self, grammar_learner: Optional[PatternGrammarLearner] = None):
        self.grammar_learner = grammar_learner or PatternGrammarLearner(verbose=False)
        self.program_generator = ProgramGenerator()

    def learn_pattern(
        self, examples: List[Tuple[np.ndarray, np.ndarray]], max_hypotheses: int = 20
    ) -> Optional[Hypothesis]:
        """Learn a pattern from few examples."""

        # Generate hypotheses
        hypotheses = self._generate_hypotheses(examples, max_hypotheses)

        # Test each hypothesis
        best_hypothesis = None
        best_score = 0

        for hypothesis in hypotheses:
            score = self._test_hypothesis(hypothesis, examples)
            if score > best_score:
                best_score = score
                best_hypothesis = hypothesis

        # Refine best hypothesis if needed
        if best_hypothesis and best_score < 1.0:
            best_hypothesis = self._refine_hypothesis(best_hypothesis, examples)

        return best_hypothesis if best_score > 0.5 else None

    def _generate_hypotheses(
        self, examples: List[Tuple[np.ndarray, np.ndarray]], max_hypotheses: int
    ) -> List[Hypothesis]:
        """Generate hypotheses about what pattern explains the examples."""
        hypotheses = []

        # Extract atomic operations from examples
        all_operations = set()
        for inp, out in examples:
            ops = self.grammar_learner._extract_atomic_operations([(inp, out)])
            all_operations.update(ops)

        # Single operation hypotheses
        for op in all_operations:
            program = self.program_generator.compose_operations([op])
            hypothesis = Hypothesis(
                name=f"single_{op.name}",
                program=program,
                confidence=0.5,
                explanation=f"Apply {op.name}",
                operations=[op],
            )
            hypotheses.append(hypothesis)

        # Two-operation compositions
        if len(all_operations) >= 2:
            for op1, op2 in itertools.combinations(all_operations, 2):
                # Sequential composition
                program = self.program_generator.compose_operations([op1, op2])
                hypothesis = Hypothesis(
                    name=f"seq_{op1.name}_{op2.name}",
                    program=program,
                    confidence=0.3,
                    explanation=f"First {op1.name}, then {op2.name}",
                    operations=[op1, op2],
                )
                hypotheses.append(hypothesis)

                if len(hypotheses) >= max_hypotheses:
                    break

        # Add some creative hypotheses based on pattern analysis
        creative_hypotheses = self._generate_creative_hypotheses(examples)
        hypotheses.extend(creative_hypotheses[: max_hypotheses - len(hypotheses)])

        return hypotheses[:max_hypotheses]

    def _generate_creative_hypotheses(
        self, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> List[Hypothesis]:
        """Generate creative hypotheses based on pattern analysis."""
        hypotheses = []

        # Check for extraction patterns
        inp, out = examples[0]
        if out.shape == (1, 1):
            # Might be extracting a property
            def extract_unique_color(x):
                from collections import Counter

                colors = x[x != 0]
                if len(colors) > 0:
                    counts = Counter(colors)
                    # Find color that appears exactly once
                    for color, count in counts.items():
                        if count == 1:
                            return np.array([[color]])
                return np.array([[0]])

            hypothesis = Hypothesis(
                name="extract_unique",
                program=extract_unique_color,
                confidence=0.4,
                explanation="Extract color that appears exactly once",
                operations=[],
            )
            hypotheses.append(hypothesis)

        # Check for pattern-based filling
        if inp.shape == out.shape:
            diff = np.sum(inp != out)
            if diff > 0 and diff < inp.size // 2:

                def pattern_fill(x):
                    result = x.copy()
                    # Fill corners or specific positions
                    if result.shape[0] > 2 and result.shape[1] > 2:
                        # Try filling corners
                        result[0, 0] = 1
                        result[0, -1] = 1
                        result[-1, 0] = 1
                        result[-1, -1] = 1
                    return result

                hypothesis = Hypothesis(
                    name="corner_fill",
                    program=pattern_fill,
                    confidence=0.3,
                    explanation="Fill corner positions",
                    operations=[],
                )
                hypotheses.append(hypothesis)

        return hypotheses

    def _test_hypothesis(
        self, hypothesis: Hypothesis, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> float:
        """Test how well a hypothesis explains the examples."""
        total_score = 0

        for inp, expected_out in examples:
            try:
                predicted_out = hypothesis.test(inp)

                # Score based on similarity
                if predicted_out.shape == expected_out.shape:
                    if np.array_equal(predicted_out, expected_out):
                        total_score += 1.0
                    else:
                        # Partial credit for partial match
                        matching = np.sum(predicted_out == expected_out)
                        total_score += matching / expected_out.size
                else:
                    # Shape mismatch - small penalty
                    total_score += 0.1
            except:
                # Hypothesis failed - no score
                pass

        return total_score / len(examples) if examples else 0

    def _refine_hypothesis(
        self, hypothesis: Hypothesis, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Hypothesis:
        """Refine a hypothesis to better fit the examples."""
        # For now, just return the hypothesis
        # In a full implementation, this would adjust parameters
        return hypothesis

    def explain_pattern(self, hypothesis: Hypothesis) -> str:
        """Generate human-readable explanation of the pattern."""
        if not hypothesis:
            return "No pattern found"

        explanation = f"Pattern: {hypothesis.explanation}\n"
        explanation += f"Confidence: {hypothesis.confidence:.1%}\n"

        if hypothesis.operations:
            explanation += "Operations:\n"
            for op in hypothesis.operations:
                explanation += f"  - {op.name} ({op.operation_type})\n"

        return explanation


def test_few_shot_learning():
    """Test few-shot pattern learning on ARC examples."""
    import json
    from pathlib import Path

    # Initialize learner
    grammar_learner = PatternGrammarLearner(verbose=False)
    learner = FewShotPatternLearner(grammar_learner)

    # Load a task
    data_dir = Path("data/arc_agi_official/ARC-AGI/data/training")
    task_id = "ed36ccf7"  # Rotation task

    with open(data_dir / f"{task_id}.json", "r") as f:
        task = json.load(f)

    # Use first 3 examples for learning
    train_examples = [
        (np.array(e["input"]), np.array(e["output"])) for e in task["train"][:3]
    ]

    print("=" * 60)
    print("FEW-SHOT PATTERN LEARNING TEST")
    print("=" * 60)
    print(f"Task: {task_id}")
    print(f"Learning from {len(train_examples)} examples")

    # Learn pattern
    hypothesis = learner.learn_pattern(train_examples)

    if hypothesis:
        print(f"\n✓ Pattern discovered!")
        print(learner.explain_pattern(hypothesis))

        # Test on remaining examples
        if len(task["train"]) > 3:
            test_example = task["train"][3]
            inp = np.array(test_example["input"])
            expected = np.array(test_example["output"])
            predicted = hypothesis.test(inp)

            if np.array_equal(predicted, expected):
                print(f"\n✓ Correctly predicted 4th example!")
            else:
                print(f"\n✗ Failed on 4th example")
                print(f"Expected shape: {expected.shape}, Got: {predicted.shape}")
    else:
        print("\n✗ No pattern found")

    # Try another task
    print("\n" + "=" * 60)
    task_id = "1cf80156"  # Extraction task
    print(f"Task: {task_id}")

    with open(data_dir / f"{task_id}.json", "r") as f:
        task = json.load(f)

    train_examples = [
        (np.array(e["input"]), np.array(e["output"])) for e in task["train"][:3]
    ]

    hypothesis = learner.learn_pattern(train_examples)
    if hypothesis:
        print(f"\n✓ Pattern discovered!")
        print(learner.explain_pattern(hypothesis))
    else:
        print("\n✗ No pattern found")


if __name__ == "__main__":
    test_few_shot_learning()
