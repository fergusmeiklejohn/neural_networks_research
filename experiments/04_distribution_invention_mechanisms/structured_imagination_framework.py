#!/usr/bin/env python3
"""
Structured Imagination Framework - A principled approach to hypothesis generation.

Key insights from failure analysis:
1. Current imagination has only 1.3% diversity - hypotheses are too similar
2. Average only 1.9 hypotheses generated - need more candidates
3. 91.6% of failures due to insufficient diversity
4. Imagination often triggered unnecessarily (when V7 already has good solution)

This framework addresses these issues through:
- Structured variation operators instead of random generation
- Constraint-aware hypothesis generation
- Progressive novelty curriculum
- Meta-learning from successful patterns
"""

from utils.imports import setup_project_paths

setup_project_paths()

import itertools
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


class VariationType(Enum):
    """Types of structured variations we can apply."""

    PERMUTATION = "permutation"  # Reorder elements
    REFLECTION = "reflection"  # Mirror horizontally/vertically
    ROTATION = "rotation"  # Rotate 90/180/270 degrees
    COLOR_SHIFT = "color_shift"  # Systematic color changes
    PATTERN_TILE = "pattern_tile"  # Repeat patterns
    BOUNDARY_MODIFY = "boundary_modify"  # Change edges/borders
    SYMMETRY_BREAK = "symmetry_break"  # Introduce controlled asymmetry
    SCALE_CHANGE = "scale_change"  # Resize with interpolation
    MASK_APPLY = "mask_apply"  # Apply selective masks
    ARITHMETIC_OP = "arithmetic_op"  # Add/subtract/multiply values


@dataclass
class Constraint:
    """Represents a constraint that must be satisfied."""

    name: str
    check_fn: Callable[[np.ndarray], bool]
    description: str

    def is_satisfied(self, grid: np.ndarray) -> bool:
        """Check if constraint is satisfied."""
        return self.check_fn(grid)


@dataclass
class Hypothesis:
    """A structured hypothesis with metadata."""

    grid: np.ndarray
    variations_applied: List[VariationType]
    novelty_score: float
    constraint_satisfaction: float
    parent_hypothesis: Optional["Hypothesis"] = None

    def __hash__(self):
        return hash(self.grid.tobytes())


class ConstraintAnalyzer:
    """Analyzes input/output pairs to identify constraints."""

    def analyze(
        self, examples: List[Tuple[np.ndarray, np.ndarray]], test_input: np.ndarray
    ) -> List[Constraint]:
        """Extract constraints from examples."""
        constraints = []

        # Size constraints
        output_shapes = [out.shape for _, out in examples]
        if len(set(output_shapes)) == 1:
            expected_shape = output_shapes[0]
            constraints.append(
                Constraint(
                    name="output_shape",
                    check_fn=lambda g: g.shape == expected_shape,
                    description=f"Output must be shape {expected_shape}",
                )
            )

        # Color constraints
        all_input_colors = set()
        all_output_colors = set()
        for inp, out in examples:
            all_input_colors.update(inp.flatten())
            all_output_colors.update(out.flatten())

        # Check if outputs only use input colors
        if all_output_colors.issubset(all_input_colors):
            test_colors = set(test_input.flatten())
            constraints.append(
                Constraint(
                    name="color_preservation",
                    check_fn=lambda g: set(g.flatten()).issubset(test_colors | {0}),
                    description="Output can only use colors from input",
                )
            )

        # Structural constraints
        # Check if all outputs have same number of non-zero elements
        non_zero_counts = [np.count_nonzero(out) for _, out in examples]
        if len(set(non_zero_counts)) == 1:
            expected_count = non_zero_counts[0]
            tolerance = int(expected_count * 0.1)  # 10% tolerance
            constraints.append(
                Constraint(
                    name="non_zero_count",
                    check_fn=lambda g: abs(np.count_nonzero(g) - expected_count)
                    <= tolerance,
                    description=f"Output should have ~{expected_count} non-zero elements",
                )
            )

        # Symmetry constraints
        for _, out in examples:
            if self._is_symmetric(out, "horizontal"):
                constraints.append(
                    Constraint(
                        name="horizontal_symmetry",
                        check_fn=lambda g: self._is_symmetric(g, "horizontal"),
                        description="Output must be horizontally symmetric",
                    )
                )
                break
            if self._is_symmetric(out, "vertical"):
                constraints.append(
                    Constraint(
                        name="vertical_symmetry",
                        check_fn=lambda g: self._is_symmetric(g, "vertical"),
                        description="Output must be vertically symmetric",
                    )
                )
                break

        return constraints

    def _is_symmetric(self, grid: np.ndarray, axis: str) -> bool:
        """Check if grid is symmetric along axis."""
        if axis == "horizontal":
            return np.array_equal(grid, np.fliplr(grid))
        elif axis == "vertical":
            return np.array_equal(grid, np.flipud(grid))
        return False


class VariationGenerator:
    """Generates systematic variations of hypotheses."""

    def __init__(self, constraints: List[Constraint]):
        self.constraints = constraints
        self.variation_operators = self._initialize_operators()

    def _initialize_operators(self) -> Dict[VariationType, Callable]:
        """Initialize variation operators."""
        return {
            VariationType.PERMUTATION: self._permute_colors,
            VariationType.REFLECTION: self._reflect,
            VariationType.ROTATION: self._rotate,
            VariationType.COLOR_SHIFT: self._shift_colors,
            VariationType.PATTERN_TILE: self._tile_pattern,
            VariationType.BOUNDARY_MODIFY: self._modify_boundary,
            VariationType.SYMMETRY_BREAK: self._break_symmetry,
            VariationType.SCALE_CHANGE: self._scale,
            VariationType.MASK_APPLY: self._apply_mask,
            VariationType.ARITHMETIC_OP: self._arithmetic_operation,
        }

    def generate_variations(
        self,
        base: np.ndarray,
        variation_types: List[VariationType],
        max_variations: int = 20,
    ) -> List[Hypothesis]:
        """Generate variations of a base hypothesis."""
        hypotheses = []

        # Start with base
        base_hypothesis = Hypothesis(
            grid=base.copy(),
            variations_applied=[],
            novelty_score=0.0,
            constraint_satisfaction=self._calculate_constraint_satisfaction(base),
        )
        hypotheses.append(base_hypothesis)

        # Apply single variations
        for var_type in variation_types:
            if len(hypotheses) >= max_variations:
                break

            operator = self.variation_operators[var_type]
            try:
                varied = operator(base)
                if varied is not None and not self._is_duplicate(varied, hypotheses):
                    hypothesis = Hypothesis(
                        grid=varied,
                        variations_applied=[var_type],
                        novelty_score=self._calculate_novelty(varied, base),
                        constraint_satisfaction=self._calculate_constraint_satisfaction(
                            varied
                        ),
                        parent_hypothesis=base_hypothesis,
                    )
                    hypotheses.append(hypothesis)
            except Exception:
                continue

        # Apply combinations of variations (up to 2)
        for var1, var2 in itertools.combinations(variation_types, 2):
            if len(hypotheses) >= max_variations:
                break

            try:
                intermediate = self.variation_operators[var1](base)
                if intermediate is not None:
                    final = self.variation_operators[var2](intermediate)
                    if final is not None and not self._is_duplicate(final, hypotheses):
                        hypothesis = Hypothesis(
                            grid=final,
                            variations_applied=[var1, var2],
                            novelty_score=self._calculate_novelty(final, base),
                            constraint_satisfaction=self._calculate_constraint_satisfaction(
                                final
                            ),
                            parent_hypothesis=base_hypothesis,
                        )
                        hypotheses.append(hypothesis)
            except Exception:
                continue

        return hypotheses

    def _calculate_constraint_satisfaction(self, grid: np.ndarray) -> float:
        """Calculate how well a grid satisfies constraints."""
        if not self.constraints:
            return 1.0

        satisfied = sum(1 for c in self.constraints if c.is_satisfied(grid))
        return satisfied / len(self.constraints)

    def _calculate_novelty(self, varied: np.ndarray, base: np.ndarray) -> float:
        """Calculate novelty score between varied and base."""
        if varied.shape != base.shape:
            return 1.0  # Maximum novelty for different shapes

        difference = np.mean(varied != base)
        return float(difference)

    def _is_duplicate(self, grid: np.ndarray, hypotheses: List[Hypothesis]) -> bool:
        """Check if grid is duplicate of existing hypothesis."""
        for h in hypotheses:
            if h.grid.shape == grid.shape and np.array_equal(h.grid, grid):
                return True
        return False

    # Variation operators
    def _permute_colors(self, grid: np.ndarray) -> Optional[np.ndarray]:
        """Permute colors in the grid."""
        unique_colors = sorted(set(grid.flatten()) - {0})
        if len(unique_colors) < 2:
            return None

        # Create a random permutation
        perm = np.random.permutation(unique_colors)
        mapping = {old: new for old, new in zip(unique_colors, perm)}
        mapping[0] = 0  # Keep background

        result = np.zeros_like(grid)
        for old, new in mapping.items():
            result[grid == old] = new

        return result

    def _reflect(self, grid: np.ndarray) -> np.ndarray:
        """Reflect grid horizontally or vertically."""
        if np.random.random() < 0.5:
            return np.fliplr(grid)
        else:
            return np.flipud(grid)

    def _rotate(self, grid: np.ndarray) -> np.ndarray:
        """Rotate grid by 90, 180, or 270 degrees."""
        k = np.random.choice([1, 2, 3])
        return np.rot90(grid, k)

    def _shift_colors(self, grid: np.ndarray) -> np.ndarray:
        """Shift all non-zero colors by a constant."""
        shift = np.random.randint(1, 4)
        result = grid.copy()
        result[result > 0] += shift
        result = np.clip(result, 0, 9)  # Keep in valid color range
        return result

    def _tile_pattern(self, grid: np.ndarray) -> Optional[np.ndarray]:
        """Tile the grid pattern."""
        h, w = grid.shape
        if h > 10 or w > 10:  # Don't tile large grids
            return None

        # Tile 2x2
        result = np.tile(grid, (2, 2))
        return result[: h * 2, : w * 2]  # Crop to reasonable size

    def _modify_boundary(self, grid: np.ndarray) -> np.ndarray:
        """Modify the boundary of the grid."""
        result = grid.copy()
        # Set border to a specific color
        color = np.random.choice([0, 1, 2, 3])
        result[0, :] = color
        result[-1, :] = color
        result[:, 0] = color
        result[:, -1] = color
        return result

    def _break_symmetry(self, grid: np.ndarray) -> np.ndarray:
        """Break symmetry by modifying one quadrant."""
        result = grid.copy()
        h, w = grid.shape

        # Modify top-left quadrant
        quadrant = result[: h // 2, : w // 2]
        if quadrant.size > 0:
            # Shift colors in quadrant
            quadrant[quadrant > 0] = (quadrant[quadrant > 0] + 1) % 10
            result[: h // 2, : w // 2] = quadrant

        return result

    def _scale(self, grid: np.ndarray) -> Optional[np.ndarray]:
        """Scale grid up or down."""
        h, w = grid.shape

        # Random scale factor
        scale = np.random.choice([0.5, 2.0])

        if scale == 0.5:
            # Downsample
            return grid[::2, ::2]
        else:
            # Upsample using nearest neighbor
            result = np.repeat(np.repeat(grid, 2, axis=0), 2, axis=1)
            return result

    def _apply_mask(self, grid: np.ndarray) -> np.ndarray:
        """Apply a random mask to the grid."""
        result = grid.copy()
        h, w = grid.shape

        # Create random rectangular mask
        y1, y2 = sorted(np.random.randint(0, h, 2))
        x1, x2 = sorted(np.random.randint(0, w, 2))

        # Zero out or fill masked region
        if np.random.random() < 0.5:
            result[y1:y2, x1:x2] = 0
        else:
            result[y1:y2, x1:x2] = np.random.randint(1, 4)

        return result

    def _arithmetic_operation(self, grid: np.ndarray) -> np.ndarray:
        """Apply arithmetic operation to grid values."""
        result = grid.copy()
        op = np.random.choice(["add", "multiply", "modulo"])

        if op == "add":
            result[result > 0] = (result[result > 0] + 2) % 10
        elif op == "multiply":
            result[result > 0] = (result[result > 0] * 2) % 10
        else:  # modulo
            result[result > 0] = result[result > 0] % 3 + 1

        return result


class ImaginationCurriculum:
    """Manages progressive novelty in imagination."""

    def __init__(self):
        self.novelty_level = 0.1  # Start conservative
        self.success_history = []
        self.variation_sequence = [
            # Level 1: Simple variations
            [VariationType.REFLECTION, VariationType.ROTATION],
            # Level 2: Color modifications
            [VariationType.COLOR_SHIFT, VariationType.PERMUTATION],
            # Level 3: Structural changes
            [VariationType.PATTERN_TILE, VariationType.SCALE_CHANGE],
            # Level 4: Complex modifications
            [VariationType.BOUNDARY_MODIFY, VariationType.MASK_APPLY],
            # Level 5: Combined operations
            [VariationType.SYMMETRY_BREAK, VariationType.ARITHMETIC_OP],
        ]
        self.current_level = 0

    def get_allowed_variations(self) -> List[VariationType]:
        """Get currently allowed variation types based on curriculum."""
        allowed = []
        for i in range(min(self.current_level + 1, len(self.variation_sequence))):
            allowed.extend(self.variation_sequence[i])
        return allowed

    def update_curriculum(self, success: bool, hypothesis: Hypothesis):
        """Update curriculum based on success/failure."""
        self.success_history.append(success)

        # Calculate recent success rate
        recent = (
            self.success_history[-10:]
            if len(self.success_history) >= 10
            else self.success_history
        )
        success_rate = sum(recent) / len(recent) if recent else 0

        # Adjust curriculum level
        if success_rate > 0.7 and self.current_level < len(self.variation_sequence) - 1:
            self.current_level += 1
            self.novelty_level = min(0.9, self.novelty_level + 0.1)
            print(f"  📈 Advancing curriculum to level {self.current_level + 1}")
        elif success_rate < 0.3 and self.current_level > 0:
            self.current_level -= 1
            self.novelty_level = max(0.1, self.novelty_level - 0.1)
            print(f"  📉 Reducing curriculum to level {self.current_level + 1}")

    def should_explore(self, confidence: float) -> bool:
        """Decide whether to explore based on confidence and curriculum."""
        # More likely to explore as novelty level increases
        threshold = 1.0 - self.novelty_level
        return confidence < threshold


class StructuredImaginationFramework:
    """Main framework combining all components."""

    def __init__(self):
        self.constraint_analyzer = ConstraintAnalyzer()
        self.curriculum = ImaginationCurriculum()
        self.meta_memory = defaultdict(list)  # Track successful patterns

    def imagine(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
        base_solution: Optional[np.ndarray] = None,
        confidence: float = 0.5,
    ) -> List[Hypothesis]:
        """Generate structured hypotheses for the test input."""

        # Check if we should even use imagination
        if not self.curriculum.should_explore(confidence):
            return []

        print(f"  🧠 Structured imagination engaged (confidence: {confidence:.2f})")

        # Analyze constraints
        constraints = self.constraint_analyzer.analyze(examples, test_input)
        print(f"  📏 Identified {len(constraints)} constraints")

        # Get allowed variations from curriculum
        allowed_variations = self.curriculum.get_allowed_variations()
        print(f"  🎯 Using variations: {[v.value for v in allowed_variations]}")

        # Initialize variation generator
        generator = VariationGenerator(constraints)

        # Generate base hypothesis if not provided
        if base_solution is None:
            base_solution = self._create_base_hypothesis(examples, test_input)

        # Generate variations
        hypotheses = generator.generate_variations(
            base_solution,
            allowed_variations,
            max_variations=30,  # More than before (was ~2)
        )

        # Filter by constraint satisfaction
        valid_hypotheses = [
            h
            for h in hypotheses
            if h.constraint_satisfaction >= 0.5  # At least half constraints
        ]

        # Sort by combination of constraint satisfaction and novelty
        valid_hypotheses.sort(
            key=lambda h: h.constraint_satisfaction - 0.3 * h.novelty_score,
            reverse=True,
        )

        print(f"  ✨ Generated {len(valid_hypotheses)} valid hypotheses")

        # Report diversity
        if len(valid_hypotheses) > 1:
            diversity = self._calculate_diversity(valid_hypotheses)
            print(f"  🌈 Hypothesis diversity: {diversity:.2%}")

        return valid_hypotheses

    def _create_base_hypothesis(
        self, examples: List[Tuple[np.ndarray, np.ndarray]], test_input: np.ndarray
    ) -> np.ndarray:
        """Create a base hypothesis to start from."""
        # Simple strategy: use first output as template
        if examples:
            return examples[0][1].copy()

        # Fallback: return input
        return test_input.copy()

    def _calculate_diversity(self, hypotheses: List[Hypothesis]) -> float:
        """Calculate diversity among hypotheses."""
        if len(hypotheses) < 2:
            return 0.0

        diversities = []
        for i, h1 in enumerate(hypotheses):
            for h2 in hypotheses[i + 1 :]:
                if h1.grid.shape == h2.grid.shape:
                    diversity = np.mean(h1.grid != h2.grid)
                    diversities.append(diversity)

        return np.mean(diversities) if diversities else 0.0

    def update_from_result(self, hypothesis: Hypothesis, success: bool):
        """Update framework based on result."""
        self.curriculum.update_curriculum(success, hypothesis)

        if success:
            # Remember successful variation patterns
            for variation in hypothesis.variations_applied:
                self.meta_memory[variation].append(
                    {
                        "constraint_satisfaction": hypothesis.constraint_satisfaction,
                        "novelty_score": hypothesis.novelty_score,
                    }
                )

    def get_statistics(self) -> Dict:
        """Get framework statistics."""
        stats = {
            "curriculum_level": self.curriculum.current_level + 1,
            "novelty_level": self.curriculum.novelty_level,
            "success_rate": (
                sum(self.curriculum.success_history[-20:])
                / min(20, len(self.curriculum.success_history))
                if self.curriculum.success_history
                else 0
            ),
            "successful_variations": {
                var.value: len(memories) for var, memories in self.meta_memory.items()
            },
        }
        return stats


def main():
    """Test the structured imagination framework."""
    print("=" * 60)
    print("STRUCTURED IMAGINATION FRAMEWORK TEST")
    print("=" * 60)

    # Create framework
    framework = StructuredImaginationFramework()

    # Create simple test case
    examples = [
        (np.array([[1, 2], [3, 4]]), np.array([[4, 3], [2, 1]])),  # Rotation
        (np.array([[5, 6], [7, 8]]), np.array([[8, 7], [6, 5]])),  # Rotation
    ]

    test_input = np.array([[2, 3], [4, 5]])

    # Generate hypotheses with different confidence levels
    for confidence in [0.9, 0.5, 0.2]:
        print(f"\n🔍 Testing with confidence: {confidence}")
        hypotheses = framework.imagine(examples, test_input, confidence=confidence)

        if hypotheses:
            print(f"\nTop 3 hypotheses:")
            for i, h in enumerate(hypotheses[:3]):
                print(f"  {i+1}. Variations: {[v.value for v in h.variations_applied]}")
                print(f"     Constraint satisfaction: {h.constraint_satisfaction:.2f}")
                print(f"     Novelty: {h.novelty_score:.2f}")
                print(f"     Grid:\n{h.grid}")
        else:
            print("  No hypotheses generated (confidence too high)")

    # Test curriculum progression
    print("\n" + "=" * 60)
    print("CURRICULUM PROGRESSION TEST")
    print("=" * 60)

    # Simulate successes to advance curriculum
    for i in range(15):
        success = i % 3 != 2  # 66% success rate
        if hypotheses:
            framework.update_from_result(hypotheses[0], success)

    stats = framework.get_statistics()
    print(f"\nFramework Statistics:")
    print(f"  Curriculum Level: {stats['curriculum_level']}/5")
    print(f"  Novelty Level: {stats['novelty_level']:.2f}")
    print(f"  Success Rate: {stats['success_rate']:.2%}")
    print(f"  Successful Variations: {stats['successful_variations']}")


if __name__ == "__main__":
    main()
