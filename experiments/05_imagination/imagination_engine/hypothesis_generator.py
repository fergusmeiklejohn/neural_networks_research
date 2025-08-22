"""Minimal Hypothesis Generator for discovering novel patterns.

This module implements explicit imagination mechanisms that can discover
patterns not present in training data. Unlike gradient-based learning,
it uses controlled randomness, constraint relaxation, and systematic
exploration to find genuinely novel solutions.

Key Innovation: Generate hypotheses with LOW probability under training
distribution but HIGH functional validity.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GenerationStrategy(Enum):
    """Different strategies for hypothesis generation."""

    RANDOM = "random"
    CONSTRAINT_RELAXATION = "constraint_relaxation"
    SYSTEMATIC = "systematic"
    COMPOSITIONAL = "compositional"


@dataclass
class Hypothesis:
    """Represents a hypothesis about a transformation."""

    transform_type: str
    parameters: Dict[str, Any]
    transform_fn: Callable
    confidence: float = 0.0
    evidence: List[bool] = None

    def __post_init__(self):
        if self.evidence is None:
            self.evidence = []

    def apply(self, input_grid: np.ndarray) -> np.ndarray:
        """Apply the hypothesis transformation to an input."""
        try:
            return self.transform_fn(input_grid, **self.parameters)
        except Exception as e:
            logger.debug(f"Hypothesis application failed: {e}")
            return np.zeros_like(input_grid)

    def update_confidence(self, success: bool):
        """Update confidence based on test result."""
        self.evidence.append(success)
        if len(self.evidence) > 0:
            self.confidence = sum(self.evidence) / len(self.evidence)


class MinimalHypothesisGenerator:
    """Generate novel hypotheses through multiple strategies."""

    def __init__(self, seed: Optional[int] = None):
        """Initialize the hypothesis generator.

        Args:
            seed: Random seed for reproducibility
        """
        self.rng = np.random.RandomState(seed)
        self.discovered_patterns = []
        self.generation_count = 0

    def generate_hypotheses(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]],
        n_hypotheses: int = 100,
        strategy: GenerationStrategy = GenerationStrategy.RANDOM,
    ) -> List[Hypothesis]:
        """Generate hypotheses using specified strategy.

        Args:
            examples: Training examples (input, output) pairs
            n_hypotheses: Number of hypotheses to generate
            strategy: Generation strategy to use

        Returns:
            List of generated hypotheses
        """
        logger.info(f"Generating {n_hypotheses} hypotheses using {strategy.value}")

        if strategy == GenerationStrategy.RANDOM:
            return self._generate_random_hypotheses(examples, n_hypotheses)
        elif strategy == GenerationStrategy.CONSTRAINT_RELAXATION:
            return self._generate_relaxed_hypotheses(examples, n_hypotheses)
        elif strategy == GenerationStrategy.SYSTEMATIC:
            return self._generate_systematic_hypotheses(examples, n_hypotheses)
        elif strategy == GenerationStrategy.COMPOSITIONAL:
            return self._generate_compositional_hypotheses(examples, n_hypotheses)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _generate_random_hypotheses(
        self, examples: List[Tuple[np.ndarray, np.ndarray]], n_hypotheses: int
    ) -> List[Hypothesis]:
        """Generate completely random transformation hypotheses."""
        hypotheses = []

        for i in range(n_hypotheses):
            self.generation_count += 1

            # Random choice of transformation type
            transform_type = self.rng.choice(
                [
                    "matrix_transform",
                    "row_shift",
                    "col_shift",
                    "diagonal_shift",
                    "spiral",
                    "wave",
                ]
            )

            if transform_type == "matrix_transform":
                # Random 3x3 transformation matrix
                matrix = self.rng.randn(3, 3) * 2
                hypothesis = Hypothesis(
                    transform_type="matrix",
                    parameters={"matrix": matrix},
                    transform_fn=self._apply_matrix_transform,
                )

            elif transform_type == "row_shift":
                # Shift each row by a function of its index
                shift_fn = self.rng.choice(["linear", "quadratic", "modulo"])
                hypothesis = Hypothesis(
                    transform_type="row_shift",
                    parameters={"shift_fn": shift_fn, "factor": self.rng.randint(1, 4)},
                    transform_fn=self._apply_row_shift,
                )

            elif transform_type == "col_shift":
                # Shift each column by a function of its index
                shift_fn = self.rng.choice(["linear", "quadratic", "modulo"])
                hypothesis = Hypothesis(
                    transform_type="col_shift",
                    parameters={"shift_fn": shift_fn, "factor": self.rng.randint(1, 4)},
                    transform_fn=self._apply_col_shift,
                )

            elif transform_type == "diagonal_shift":
                # Shift diagonally (SHEAR!)
                hypothesis = Hypothesis(
                    transform_type="diagonal_shift",
                    parameters={"direction": self.rng.choice(["right", "left"])},
                    transform_fn=self._apply_diagonal_shift,
                )

            elif transform_type == "spiral":
                # Spiral pattern from center
                hypothesis = Hypothesis(
                    transform_type="spiral",
                    parameters={"clockwise": self.rng.choice([True, False])},
                    transform_fn=self._apply_spiral_transform,
                )

            elif transform_type == "wave":
                # Wave-like transformation
                hypothesis = Hypothesis(
                    transform_type="wave",
                    parameters={
                        "amplitude": self.rng.randint(1, 3),
                        "frequency": self.rng.random() * 2,
                    },
                    transform_fn=self._apply_wave_transform,
                )

            hypotheses.append(hypothesis)

        return hypotheses

    def _generate_relaxed_hypotheses(
        self, examples: List[Tuple[np.ndarray, np.ndarray]], n_hypotheses: int
    ) -> List[Hypothesis]:
        """Generate hypotheses by relaxing constraints on known patterns."""
        hypotheses = []

        # Start with known transformations and relax constraints
        base_patterns = ["rotate", "flip", "scale", "translate"]

        for i in range(n_hypotheses):
            self.generation_count += 1

            base = self.rng.choice(base_patterns)

            if base == "rotate":
                # Relax rotation to non-90-degree angles and non-orthogonal
                angle = self.rng.random() * 360
                allow_stretch = self.rng.choice([True, False])
                hypothesis = Hypothesis(
                    transform_type="relaxed_rotate",
                    parameters={"angle": angle, "stretch": allow_stretch},
                    transform_fn=self._apply_relaxed_rotation,
                )

            elif base == "flip":
                # Relax flip to partial flips and shears
                axis = self.rng.choice(["horizontal", "vertical", "diagonal", "custom"])
                partial = self.rng.random()
                hypothesis = Hypothesis(
                    transform_type="relaxed_flip",
                    parameters={"axis": axis, "partial": partial},
                    transform_fn=self._apply_relaxed_flip,
                )

            elif base == "scale":
                # Non-uniform scaling, with position-dependent factors
                x_scale = self.rng.random() * 3
                y_scale = self.rng.random() * 3
                position_dependent = self.rng.choice([True, False])
                hypothesis = Hypothesis(
                    transform_type="relaxed_scale",
                    parameters={
                        "x_scale": x_scale,
                        "y_scale": y_scale,
                        "position_dependent": position_dependent,
                    },
                    transform_fn=self._apply_relaxed_scale,
                )

            elif base == "translate":
                # Position-dependent translation (creates shear!)
                x_shift_fn = lambda row, col: row  # Shift depends on row
                y_shift_fn = lambda row, col: 0
                hypothesis = Hypothesis(
                    transform_type="position_translate",
                    parameters={"x_shift_fn": x_shift_fn, "y_shift_fn": y_shift_fn},
                    transform_fn=self._apply_position_dependent_translate,
                )

            hypotheses.append(hypothesis)

        return hypotheses

    def _generate_systematic_hypotheses(
        self, examples: List[Tuple[np.ndarray, np.ndarray]], n_hypotheses: int
    ) -> List[Hypothesis]:
        """Systematically explore parameter space."""
        hypotheses = []

        # Grid search over shear parameters (most likely to work)
        shear_params = []
        for x_shear in np.linspace(-2, 2, 5):
            for y_shear in np.linspace(-2, 2, 5):
                shear_params.append((x_shear, y_shear))

        # Sample from parameter grid
        n_shear = min(n_hypotheses // 2, len(shear_params))
        selected_shears = self.rng.choice(len(shear_params), n_shear, replace=False)

        for idx in selected_shears:
            self.generation_count += 1
            x_shear, y_shear = shear_params[idx]
            hypothesis = Hypothesis(
                transform_type="systematic_shear",
                parameters={"x_shear": x_shear, "y_shear": y_shear},
                transform_fn=lambda g, xs=x_shear, ys=y_shear: self._apply_systematic_shear(g, xs, ys),
            )
            hypotheses.append(hypothesis)

        # Fill remaining with row/col shifts
        remaining = n_hypotheses - len(hypotheses)
        for i in range(remaining):
            self.generation_count += 1
            shift_amount = i % 5  # Systematic shift amounts
            shift_type = "row" if i < remaining // 2 else "col"

            hypothesis = Hypothesis(
                transform_type=f"systematic_{shift_type}_shift",
                parameters={"shift": shift_amount, "type": shift_type},
                transform_fn=self._apply_systematic_shift,
            )
            hypotheses.append(hypothesis)

        return hypotheses

    def _generate_compositional_hypotheses(
        self, examples: List[Tuple[np.ndarray, np.ndarray]], n_hypotheses: int
    ) -> List[Hypothesis]:
        """Generate hypotheses by composing atomic operations."""
        hypotheses = []

        atomic_ops = ["shift_row", "shift_col", "rotate_90", "flip_h", "flip_v", "scale_2x"]

        for i in range(n_hypotheses):
            self.generation_count += 1

            # Compose 2-3 operations
            n_ops = self.rng.choice([2, 3])
            ops = self.rng.choice(atomic_ops, n_ops)

            hypothesis = Hypothesis(
                transform_type="composition",
                parameters={"operations": ops.tolist()},
                transform_fn=self._apply_composition,
            )
            hypotheses.append(hypothesis)

        return hypotheses

    def test_hypothesis(
        self, hypothesis: Hypothesis, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> float:
        """Test hypothesis on examples and return accuracy."""
        if not examples:
            return 0.0

        correct = 0
        total = 0

        for input_grid, expected_output in examples:
            predicted = hypothesis.apply(input_grid)

            # Check if shapes match
            if predicted.shape != expected_output.shape:
                hypothesis.update_confidence(False)
                continue

            # Check if values match
            if np.array_equal(predicted, expected_output):
                correct += 1
                hypothesis.update_confidence(True)
            else:
                # Partial credit for partial matches
                matching = np.sum(predicted == expected_output)
                total_elements = predicted.size
                partial_score = matching / total_elements
                if partial_score > 0.8:  # 80% match threshold
                    correct += partial_score
                    hypothesis.update_confidence(True)
                else:
                    hypothesis.update_confidence(False)

            total += 1

        return correct / total if total > 0 else 0.0

    def discover_pattern(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]],
        max_attempts: int = 1000,
        strategies: Optional[List[GenerationStrategy]] = None,
    ) -> Optional[Hypothesis]:
        """Attempt to discover the pattern in examples.

        Args:
            examples: Training examples
            max_attempts: Maximum hypotheses to test
            strategies: Strategies to use (defaults to all)

        Returns:
            Best hypothesis if found, None otherwise
        """
        if strategies is None:
            strategies = list(GenerationStrategy)

        logger.info(f"Starting pattern discovery with {len(examples)} examples")

        best_hypothesis = None
        best_score = 0.0
        attempts = 0

        # Try each strategy
        for strategy in strategies:
            if attempts >= max_attempts:
                break

            # Generate batch of hypotheses
            batch_size = min(100, max_attempts - attempts)
            hypotheses = self.generate_hypotheses(examples, batch_size, strategy)

            # Test each hypothesis
            for hypothesis in hypotheses:
                score = self.test_hypothesis(hypothesis, examples)
                attempts += 1

                if score > best_score:
                    best_score = score
                    best_hypothesis = hypothesis
                    logger.info(
                        f"New best: {hypothesis.transform_type} "
                        f"(score: {score:.2f}, attempts: {attempts})"
                    )

                # Early stopping if perfect solution found
                if score >= 1.0:
                    logger.info(f"Perfect solution found after {attempts} attempts!")
                    self.discovered_patterns.append(hypothesis)
                    return hypothesis

                if attempts >= max_attempts:
                    break

        if best_hypothesis and best_score > 0.5:
            logger.info(
                f"Pattern discovered: {best_hypothesis.transform_type} "
                f"(score: {best_score:.2f})"
            )
            self.discovered_patterns.append(best_hypothesis)
            return best_hypothesis

        logger.warning(f"No pattern discovered after {attempts} attempts")
        return None

    # Transform functions
    def _apply_matrix_transform(self, grid: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """Apply matrix transformation to grid."""
        result = np.zeros_like(grid)
        h, w = grid.shape

        for i in range(h):
            for j in range(w):
                # Convert to homogeneous coordinates
                pos = np.array([i, j, 1])
                new_pos = matrix @ pos
                new_i, new_j = int(new_pos[0]), int(new_pos[1])

                # Check bounds
                if 0 <= new_i < h and 0 <= new_j < w:
                    result[new_i, new_j] = grid[i, j]

        return result

    def _apply_row_shift(
        self, grid: np.ndarray, shift_fn: str, factor: int
    ) -> np.ndarray:
        """Apply row-dependent shift."""
        result = np.zeros_like(grid)
        h, w = grid.shape

        for row in range(h):
            if shift_fn == "linear":
                shift = row * factor
            elif shift_fn == "quadratic":
                shift = row * row
            else:  # modulo
                shift = row

            for col in range(w):
                new_col = (col + shift) % w
                result[row, new_col] = grid[row, col]

        return result

    def _apply_col_shift(
        self, grid: np.ndarray, shift_fn: str, factor: int
    ) -> np.ndarray:
        """Apply column-dependent shift."""
        result = np.zeros_like(grid)
        h, w = grid.shape

        for col in range(w):
            if shift_fn == "linear":
                shift = col * factor
            elif shift_fn == "quadratic":
                shift = col * col
            else:  # modulo
                shift = col

            for row in range(h):
                new_row = (row + shift) % h
                result[new_row, col] = grid[row, col]

        return result

    def _apply_diagonal_shift(self, grid: np.ndarray, direction: str) -> np.ndarray:
        """Apply diagonal shift (SHEAR transformation!)."""
        result = np.zeros_like(grid)
        h, w = grid.shape

        for row in range(h):
            for col in range(w):
                if direction == "right":
                    # Shift right based on row (classic shear)
                    new_col = (col + row) % w
                    result[row, new_col] = grid[row, col]
                else:
                    # Shift left based on row
                    new_col = (col - row) % w
                    result[row, new_col] = grid[row, col]

        return result

    def _apply_spiral_transform(self, grid: np.ndarray, clockwise: bool) -> np.ndarray:
        """Apply spiral transformation from center."""
        result = np.zeros_like(grid)
        h, w = grid.shape
        center_r, center_c = h // 2, w // 2

        # Create spiral ordering
        visited = np.zeros_like(grid, dtype=bool)
        spiral_order = []

        r, c = center_r, center_c
        dr, dc = 0, 1 if clockwise else -1
        steps = 1

        while len(spiral_order) < h * w:
            for _ in range(2):
                for _ in range(steps):
                    if 0 <= r < h and 0 <= c < w and not visited[r, c]:
                        spiral_order.append((r, c))
                        visited[r, c] = True
                    r, c = r + dr, c + dc

                # Change direction
                if clockwise:
                    dr, dc = dc, -dr
                else:
                    dr, dc = -dc, dr

            steps += 1

        # Apply spiral mapping
        flat_grid = grid.flatten()
        for i, (r, c) in enumerate(spiral_order[: len(flat_grid)]):
            if i < len(flat_grid):
                result[r, c] = flat_grid[i]

        return result

    def _apply_wave_transform(
        self, grid: np.ndarray, amplitude: int, frequency: float
    ) -> np.ndarray:
        """Apply wave-like transformation."""
        result = np.zeros_like(grid)
        h, w = grid.shape

        for row in range(h):
            wave_offset = int(amplitude * np.sin(frequency * row))
            for col in range(w):
                new_col = (col + wave_offset) % w
                result[row, new_col] = grid[row, col]

        return result

    def _apply_relaxed_rotation(
        self, grid: np.ndarray, angle: float, stretch: bool
    ) -> np.ndarray:
        """Apply rotation with optional stretching."""
        result = np.zeros_like(grid)
        h, w = grid.shape
        center_r, center_c = h // 2, w // 2

        angle_rad = np.radians(angle)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

        # Add stretching if enabled
        stretch_factor = 1.5 if stretch else 1.0

        for r in range(h):
            for c in range(w):
                # Center coordinates
                dr, dc = r - center_r, c - center_c

                # Apply rotation (and stretch)
                new_dr = cos_a * dr - sin_a * dc * stretch_factor
                new_dc = sin_a * dr + cos_a * dc

                # Back to grid coordinates
                new_r = int(new_dr + center_r)
                new_c = int(new_dc + center_c)

                if 0 <= new_r < h and 0 <= new_c < w:
                    result[new_r, new_c] = grid[r, c]

        return result

    def _apply_relaxed_flip(
        self, grid: np.ndarray, axis: str, partial: float
    ) -> np.ndarray:
        """Apply partial or custom axis flip."""
        result = grid.copy()
        h, w = grid.shape

        if axis == "horizontal":
            flip_rows = int(h * partial)
            result[:flip_rows] = np.flipud(result[:flip_rows])
        elif axis == "vertical":
            flip_cols = int(w * partial)
            result[:, :flip_cols] = np.fliplr(result[:, :flip_cols])
        elif axis == "diagonal":
            # Flip along main diagonal
            for i in range(min(h, w)):
                for j in range(i):
                    if partial > 0.5 or (i + j) % 2 == 0:
                        result[i, j], result[j, i] = result[j, i], result[i, j]
        else:  # custom
            # Random axis flip
            for i in range(h):
                for j in range(w):
                    if self.rng.random() < partial:
                        new_i = h - 1 - i
                        new_j = w - 1 - j
                        if 0 <= new_i < h and 0 <= new_j < w:
                            result[i, j], result[new_i, new_j] = (
                                result[new_i, new_j],
                                result[i, j],
                            )

        return result

    def _apply_relaxed_scale(
        self,
        grid: np.ndarray,
        x_scale: float,
        y_scale: float,
        position_dependent: bool,
    ) -> np.ndarray:
        """Apply non-uniform scaling."""
        h, w = grid.shape
        new_h = int(h * y_scale)
        new_w = int(w * x_scale)
        result = np.zeros((new_h, new_w), dtype=grid.dtype)

        for r in range(min(h, new_h)):
            for c in range(min(w, new_w)):
                if position_dependent:
                    # Scale depends on position
                    local_x_scale = x_scale * (1 + c / w)
                    local_y_scale = y_scale * (1 + r / h)
                    src_r = int(r / local_y_scale)
                    src_c = int(c / local_x_scale)
                else:
                    src_r = int(r / y_scale)
                    src_c = int(c / x_scale)

                if 0 <= src_r < h and 0 <= src_c < w:
                    result[r, c] = grid[src_r, src_c]

        # Resize back to original shape
        if result.shape != grid.shape:
            # Simple resize by trimming or padding
            final_result = np.zeros_like(grid)
            copy_h = min(result.shape[0], grid.shape[0])
            copy_w = min(result.shape[1], grid.shape[1])
            final_result[:copy_h, :copy_w] = result[:copy_h, :copy_w]
            return final_result

        return result

    def _apply_position_dependent_translate(
        self, grid: np.ndarray, x_shift_fn: Callable, y_shift_fn: Callable
    ) -> np.ndarray:
        """Apply position-dependent translation (creates shear!)."""
        result = np.zeros_like(grid)
        h, w = grid.shape

        for row in range(h):
            for col in range(w):
                x_shift = x_shift_fn(row, col)
                y_shift = y_shift_fn(row, col)

                new_row = (row + y_shift) % h
                new_col = (col + x_shift) % w

                result[new_row, new_col] = grid[row, col]

        return result

    def _apply_systematic_shear(
        self, grid: np.ndarray, x_shear: float, y_shear: float
    ) -> np.ndarray:
        """Apply systematic shear transformation."""
        result = np.zeros_like(grid)
        h, w = grid.shape

        for row in range(h):
            for col in range(w):
                # Shear transformation
                new_row = row + int(y_shear * col)
                new_col = col + int(x_shear * row)

                # Wrap around
                new_row = new_row % h
                new_col = new_col % w

                result[new_row, new_col] = grid[row, col]

        return result

    def _apply_systematic_shift(
        self, grid: np.ndarray, shift: int, type: str
    ) -> np.ndarray:
        """Apply systematic row or column shift."""
        result = np.zeros_like(grid)
        h, w = grid.shape

        if type == "row":
            for row in range(h):
                for col in range(w):
                    new_col = (col + row * shift) % w
                    result[row, new_col] = grid[row, col]
        else:  # col
            for col in range(w):
                for row in range(h):
                    new_row = (row + col * shift) % h
                    result[new_row, col] = grid[row, col]

        return result

    def _apply_composition(
        self, grid: np.ndarray, operations: List[str]
    ) -> np.ndarray:
        """Apply composition of atomic operations."""
        result = grid.copy()

        for op in operations:
            if op == "shift_row":
                # Shift each row by 1
                temp = np.zeros_like(result)
                for r in range(result.shape[0]):
                    for c in range(result.shape[1]):
                        new_c = (c + 1) % result.shape[1]
                        temp[r, new_c] = result[r, c]
                result = temp

            elif op == "shift_col":
                # Shift each column by 1
                temp = np.zeros_like(result)
                for c in range(result.shape[1]):
                    for r in range(result.shape[0]):
                        new_r = (r + 1) % result.shape[0]
                        temp[new_r, c] = result[r, c]
                result = temp

            elif op == "rotate_90":
                result = np.rot90(result)

            elif op == "flip_h":
                result = np.flipud(result)

            elif op == "flip_v":
                result = np.fliplr(result)

            elif op == "scale_2x":
                # Simple 2x scale
                h, w = result.shape
                temp = np.zeros((h * 2, w * 2), dtype=result.dtype)
                for r in range(h):
                    for c in range(w):
                        temp[r * 2 : r * 2 + 2, c * 2 : c * 2 + 2] = result[r, c]
                # Resize back
                result = temp[::2, ::2]

        return result

    def refine_hypothesis(
        self, hypothesis: Hypothesis, feedback: Dict[str, Any]
    ) -> Hypothesis:
        """Refine hypothesis based on feedback.

        Args:
            hypothesis: Current hypothesis
            feedback: Information about what worked/didn't

        Returns:
            Refined hypothesis
        """
        # This is a placeholder for future refinement logic
        # Could use partial matches to guide parameter adjustment
        return hypothesis

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about hypothesis generation."""
        return {
            "total_generated": self.generation_count,
            "patterns_discovered": len(self.discovered_patterns),
            "discovery_rate": (
                len(self.discovered_patterns) / self.generation_count
                if self.generation_count > 0
                else 0
            ),
        }