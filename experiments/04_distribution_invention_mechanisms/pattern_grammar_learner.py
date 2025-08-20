"""Pattern Grammar Learner - Learning the language of transformations.

Instead of hard-coding patterns, we learn a grammar of atomic operations
and how they compose to form complex transformations.
"""

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
from scipy import ndimage


@dataclass
class AtomicOperation:
    """Represents a basic transformation operation."""

    name: str
    operation_type: str  # 'spatial', 'color', 'object', 'arithmetic', 'logical'
    parameters: Dict[str, Any]
    frequency: float = 0.0

    def __hash__(self):
        return hash((self.name, self.operation_type))

    def __eq__(self, other):
        return self.name == other.name and self.operation_type == other.operation_type


@dataclass
class CompositionRule:
    """Represents how atomic operations can be composed."""

    operations: Tuple[str, ...]
    composition_type: str  # 'sequential', 'parallel', 'conditional', 'recursive'
    frequency: float = 0.0
    constraints: Dict[str, Any] = None


class PatternGrammarLearner:
    """Learns a grammar of transformations from ARC tasks."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.atomic_operations: Set[AtomicOperation] = set()
        self.composition_rules: List[CompositionRule] = []
        self.operation_sequences: List[List[str]] = []

    def learn_from_tasks(self, task_ids: List[str], data_dir: Path) -> Dict:
        """Extract grammar from a set of ARC tasks."""
        if self.verbose:
            print(f"Learning pattern grammar from {len(task_ids)} tasks...")

        # Extract operations from each task
        for task_id in task_ids:
            try:
                with open(data_dir / f"{task_id}.json", "r") as f:
                    task = json.load(f)

                examples = [
                    (np.array(e["input"]), np.array(e["output"])) for e in task["train"]
                ]

                # Extract atomic operations
                operations = self._extract_atomic_operations(examples)
                self.atomic_operations.update(operations)

                # Track operation sequences
                sequence = [op.name for op in operations]
                if sequence:
                    self.operation_sequences.append(sequence)

            except Exception as e:
                if self.verbose:
                    print(f"Error processing {task_id}: {e}")

        # Learn composition rules from sequences
        self._learn_composition_rules()

        # Calculate operation frequencies
        self._calculate_frequencies()

        return self.get_grammar()

    def _extract_atomic_operations(
        self, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Set[AtomicOperation]:
        """Extract atomic operations from input-output examples."""
        operations = set()

        for inp, out in examples:
            # Spatial operations
            ops = self._detect_spatial_operations(inp, out)
            operations.update(ops)

            # Color operations
            ops = self._detect_color_operations(inp, out)
            operations.update(ops)

            # Object operations
            ops = self._detect_object_operations(inp, out)
            operations.update(ops)

            # Arithmetic operations
            ops = self._detect_arithmetic_operations(inp, out)
            operations.update(ops)

            # Logical operations
            ops = self._detect_logical_operations(inp, out)
            operations.update(ops)

        return operations

    def _detect_spatial_operations(
        self, inp: np.ndarray, out: np.ndarray
    ) -> Set[AtomicOperation]:
        """Detect spatial transformation operations."""
        operations = set()

        # Translation
        if inp.shape == out.shape:
            # Check for shifts
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    if dy == 0 and dx == 0:
                        continue
                    shifted = np.roll(np.roll(inp, dy, axis=0), dx, axis=1)
                    if np.array_equal(shifted, out):
                        operations.add(
                            AtomicOperation(
                                f"translate_{dy}_{dx}", "spatial", {"dy": dy, "dx": dx}
                            )
                        )

        # Rotation
        if inp.shape == out.shape:
            for k in [1, 2, 3]:
                if np.array_equal(np.rot90(inp, k), out):
                    operations.add(
                        AtomicOperation(
                            f"rotate_{k*90}", "spatial", {"degrees": k * 90}
                        )
                    )

        # Flipping
        if inp.shape == out.shape:
            if np.array_equal(np.flipud(inp), out):
                operations.add(AtomicOperation("flip_vertical", "spatial", {}))
            if np.array_equal(np.fliplr(inp), out):
                operations.add(AtomicOperation("flip_horizontal", "spatial", {}))

        # Scaling
        if inp.shape != out.shape:
            h_ratio = out.shape[0] / inp.shape[0] if inp.shape[0] > 0 else 0
            w_ratio = out.shape[1] / inp.shape[1] if inp.shape[1] > 0 else 0

            if h_ratio == w_ratio and h_ratio in [2, 3, 4, 0.5, 0.33]:
                operations.add(
                    AtomicOperation(f"scale_{h_ratio}", "spatial", {"factor": h_ratio})
                )

        # Cropping/Extraction
        if out.shape[0] < inp.shape[0] or out.shape[1] < inp.shape[1]:
            operations.add(
                AtomicOperation(
                    "crop",
                    "spatial",
                    {"input_shape": inp.shape, "output_shape": out.shape},
                )
            )

        return operations

    def _detect_color_operations(
        self, inp: np.ndarray, out: np.ndarray
    ) -> Set[AtomicOperation]:
        """Detect color transformation operations."""
        operations = set()

        inp_colors = set(np.unique(inp))
        out_colors = set(np.unique(out))

        # Color mapping
        if inp_colors != out_colors:
            # New colors added
            new_colors = out_colors - inp_colors
            if new_colors:
                operations.add(
                    AtomicOperation("add_colors", "color", {"colors": list(new_colors)})
                )

            # Colors removed
            removed_colors = inp_colors - out_colors
            if removed_colors:
                operations.add(
                    AtomicOperation(
                        "remove_colors", "color", {"colors": list(removed_colors)}
                    )
                )

            # Color substitution
            if len(inp_colors) == len(out_colors):
                operations.add(AtomicOperation("color_map", "color", {}))

        # Color fill
        if inp.shape == out.shape:
            diff_mask = inp != out
            if np.any(diff_mask):
                # Check if it's filling a specific pattern
                fill_ratio = np.sum(diff_mask) / inp.size
                if fill_ratio < 0.5:  # Sparse fill
                    operations.add(
                        AtomicOperation("sparse_fill", "color", {"ratio": fill_ratio})
                    )

        return operations

    def _detect_object_operations(
        self, inp: np.ndarray, out: np.ndarray
    ) -> Set[AtomicOperation]:
        """Detect object-level operations."""
        operations = set()

        # Object counting
        inp_objects = ndimage.label(inp != 0)[1]
        out_objects = ndimage.label(out != 0)[1]

        if inp_objects != out_objects:
            if out_objects > inp_objects:
                operations.add(
                    AtomicOperation(
                        "object_creation",
                        "object",
                        {"count_change": out_objects - inp_objects},
                    )
                )
            else:
                operations.add(
                    AtomicOperation(
                        "object_deletion",
                        "object",
                        {"count_change": inp_objects - out_objects},
                    )
                )

        # Object movement (simplified detection)
        if inp_objects == out_objects and inp_objects > 0:
            # Check if objects moved
            if not np.array_equal(inp, out):
                operations.add(AtomicOperation("object_movement", "object", {}))

        return operations

    def _detect_arithmetic_operations(
        self, inp: np.ndarray, out: np.ndarray
    ) -> Set[AtomicOperation]:
        """Detect arithmetic/counting operations."""
        operations = set()

        # Count-based operations
        inp_count = np.sum(inp != 0)
        out_count = np.sum(out != 0)

        if out_count == inp_count * 2:
            operations.add(AtomicOperation("double_count", "arithmetic", {}))
        elif out_count == inp_count // 2:
            operations.add(AtomicOperation("halve_count", "arithmetic", {}))

        # Size relationship
        if out.shape == (1, 1):
            # Might be extracting a count or property
            operations.add(AtomicOperation("extract_property", "arithmetic", {}))

        return operations

    def _detect_logical_operations(
        self, inp: np.ndarray, out: np.ndarray
    ) -> Set[AtomicOperation]:
        """Detect logical/conditional operations."""
        operations = set()

        # Conditional fill based on neighbors
        if inp.shape == out.shape:
            # Check if output depends on neighbor count
            changed_positions = np.argwhere(inp != out)
            if len(changed_positions) > 0:
                # Sample a few positions to check for patterns
                for pos in changed_positions[: min(5, len(changed_positions))]:
                    y, x = pos
                    # Count neighbors
                    neighbors = 0
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if dy == 0 and dx == 0:
                                continue
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < inp.shape[0] and 0 <= nx < inp.shape[1]:
                                if inp[ny, nx] != 0:
                                    neighbors += 1

                    # Check if change correlates with neighbor count
                    if neighbors >= 3:
                        operations.add(
                            AtomicOperation(
                                "conditional_fill",
                                "logical",
                                {"condition": "neighbors"},
                            )
                        )
                        break

        return operations

    def _learn_composition_rules(self):
        """Learn how atomic operations compose from observed sequences."""
        if len(self.operation_sequences) < 2:
            return

        # Find common subsequences
        bigrams = defaultdict(int)
        trigrams = defaultdict(int)

        for sequence in self.operation_sequences:
            # Bigrams
            for i in range(len(sequence) - 1):
                bigram = (sequence[i], sequence[i + 1])
                bigrams[bigram] += 1

            # Trigrams
            for i in range(len(sequence) - 2):
                trigram = (sequence[i], sequence[i + 1], sequence[i + 2])
                trigrams[trigram] += 1

        # Create composition rules for frequent patterns
        total_sequences = len(self.operation_sequences)

        for bigram, count in bigrams.items():
            if count >= 2:  # Appears in at least 2 tasks
                self.composition_rules.append(
                    CompositionRule(
                        operations=bigram,
                        composition_type="sequential",
                        frequency=count / total_sequences,
                    )
                )

        for trigram, count in trigrams.items():
            if count >= 2:
                self.composition_rules.append(
                    CompositionRule(
                        operations=trigram,
                        composition_type="sequential",
                        frequency=count / total_sequences,
                    )
                )

    def _calculate_frequencies(self):
        """Calculate frequency of each atomic operation."""
        operation_counts = Counter()

        for sequence in self.operation_sequences:
            for op_name in sequence:
                operation_counts[op_name] += 1

        total = sum(operation_counts.values())

        for op in self.atomic_operations:
            op.frequency = operation_counts.get(op.name, 0) / total if total > 0 else 0

    def get_grammar(self) -> Dict:
        """Return the learned grammar."""
        return {
            "atomic_operations": [
                {
                    "name": op.name,
                    "type": op.operation_type,
                    "frequency": float(op.frequency),
                    "parameters": {
                        k: (
                            v.tolist()
                            if isinstance(v, np.ndarray)
                            else int(v)
                            if isinstance(v, (np.integer, np.int64))
                            else float(v)
                            if isinstance(v, (np.floating, np.float64))
                            else v
                        )
                        for k, v in op.parameters.items()
                    },
                }
                for op in sorted(self.atomic_operations, key=lambda x: -x.frequency)
            ],
            "composition_rules": [
                {
                    "operations": rule.operations,
                    "type": rule.composition_type,
                    "frequency": float(rule.frequency),
                }
                for rule in sorted(self.composition_rules, key=lambda x: -x.frequency)
            ],
            "statistics": {
                "total_atomic_operations": len(self.atomic_operations),
                "total_composition_rules": len(self.composition_rules),
                "tasks_analyzed": len(self.operation_sequences),
            },
        }

    def generate_hypothesis(
        self, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> List[List[AtomicOperation]]:
        """Generate hypotheses for what operations could explain the examples."""
        hypotheses = []

        # Extract operations from these specific examples
        observed_ops = self._extract_atomic_operations(examples)

        # Generate hypotheses based on grammar
        # Single operation hypotheses
        for op in observed_ops:
            hypotheses.append([op])

        # Composition hypotheses based on learned rules
        op_names = {op.name for op in observed_ops}
        for rule in self.composition_rules:
            if all(op_name in op_names for op_name in rule.operations):
                # This composition rule could apply
                hypothesis = [op for op in observed_ops if op.name in rule.operations]
                if hypothesis:
                    hypotheses.append(hypothesis)

        return hypotheses


def test_grammar_learner():
    """Test the pattern grammar learner on a sample of ARC tasks."""

    learner = PatternGrammarLearner(verbose=True)
    data_dir = Path("data/arc_agi_official/ARC-AGI/data/training")

    # Sample some tasks to learn from
    sample_tasks = [
        "ae3edfdc",
        "00d62c1b",
        "0ca9ddb6",
        "ed36ccf7",
        "68b16354",
        "32597951",
        "045e512c",
        "05f2a901",
        "1cf80156",
        "25ff71a9",
        "3aa6fb7a",
        "a416b8f3",
    ]

    # Learn grammar
    grammar = learner.learn_from_tasks(sample_tasks, data_dir)

    # Print results
    print("\n" + "=" * 60)
    print("LEARNED PATTERN GRAMMAR")
    print("=" * 60)

    print(f"\nStatistics:")
    print(
        f"  Total atomic operations: {grammar['statistics']['total_atomic_operations']}"
    )
    print(
        f"  Total composition rules: {grammar['statistics']['total_composition_rules']}"
    )
    print(f"  Tasks analyzed: {grammar['statistics']['tasks_analyzed']}")

    print(f"\nTop Atomic Operations:")
    for op in grammar["atomic_operations"][:10]:
        print(f"  {op['name']} ({op['type']}): {op['frequency']:.2%}")

    print(f"\nTop Composition Rules:")
    for rule in grammar["composition_rules"][:5]:
        print(
            f"  {' → '.join(rule['operations'])} ({rule['type']}): {rule['frequency']:.2%}"
        )

    # Save grammar
    with open("learned_pattern_grammar.json", "w") as f:
        json.dump(grammar, f, indent=2)
    print(f"\nGrammar saved to learned_pattern_grammar.json")


if __name__ == "__main__":
    test_grammar_learner()
