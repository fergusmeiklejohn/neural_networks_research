#!/usr/bin/env python3
"""
Hybrid V7 + Structured Imagination Solver.

Combines the strengths of:
- V7's program synthesis (71.4% baseline accuracy)
- Structured imagination for true extrapolation

Key improvements over previous imagination attempts:
1. Much higher hypothesis diversity (83% vs 1.3%)
2. More hypotheses generated (20-30 vs 1.9)
3. Constraint-aware generation
4. Progressive curriculum
5. Better triggering logic
"""

from utils.imports import setup_project_paths

setup_project_paths()

import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from enhanced_arc_solver_v7 import EnhancedARCSolverV7
from structured_imagination_framework import Hypothesis, StructuredImaginationFramework


@dataclass
class SolverResult:
    """Result from solver with metadata."""

    output_grid: np.ndarray
    confidence: float
    method: str
    time_taken: float
    hypotheses_tested: int = 0


class HybridV7StructuredImagination:
    """Hybrid solver combining V7 with structured imagination."""

    def __init__(
        self,
        v7_confidence_threshold: float = 0.7,
        imagination_trigger_threshold: float = 0.6,
        max_hypotheses: int = 30,
    ):
        """
        Initialize hybrid solver.

        Args:
            v7_confidence_threshold: Min confidence to accept V7 solution
            imagination_trigger_threshold: Max confidence to trigger imagination
            max_hypotheses: Maximum hypotheses to test
        """
        self.v7_solver = EnhancedARCSolverV7(
            use_synthesis=True, use_position_learning=True, confidence_threshold=0.85
        )
        self.imagination_framework = StructuredImaginationFramework()
        self.v7_threshold = v7_confidence_threshold
        self.imagination_threshold = imagination_trigger_threshold
        self.max_hypotheses = max_hypotheses

        # Track performance for adaptive thresholds
        self.performance_history = []

    def solve(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
        verbose: bool = False,
    ) -> SolverResult:
        """
        Solve using hybrid approach.

        Strategy:
        1. Try V7 first (fast, works well for pattern matching)
        2. If confidence is high enough, return V7 solution
        3. If confidence is moderate, use V7 as base for imagination
        4. If confidence is low, use pure imagination
        """
        start_time = time.time()

        # Step 1: Try V7 solver
        if verbose:
            print("🔍 Attempting V7 solution...")

        v7_result = self.v7_solver.solve(examples, test_input)

        # Extract result and confidence
        if hasattr(v7_result, "output_grid"):
            v7_output = v7_result.output_grid
            v7_confidence = (
                v7_result.confidence if hasattr(v7_result, "confidence") else 0.5
            )
        else:
            v7_output = v7_result
            v7_confidence = self._estimate_confidence(v7_output, examples)

        if verbose:
            print(f"  V7 confidence: {v7_confidence:.2f}")

        # Step 2: Check if V7 solution is good enough
        if v7_confidence >= self.v7_threshold:
            if verbose:
                print(f"  ✅ High confidence - using V7 solution")
            return SolverResult(
                output_grid=v7_output,
                confidence=v7_confidence,
                method="v7_high_confidence",
                time_taken=time.time() - start_time,
            )

        # Step 3: Decide whether to use imagination
        if v7_confidence > self.imagination_threshold:
            # Moderate confidence - use V7 as base for imagination
            if verbose:
                print(f"  🤔 Moderate confidence - refining with imagination")
            base_solution = v7_output
        else:
            # Low confidence - use pure imagination
            if verbose:
                print(f"  ❓ Low confidence - using pure imagination")
            base_solution = None

        # Step 4: Apply structured imagination
        hypotheses = self.imagination_framework.imagine(
            examples=examples,
            test_input=test_input,
            base_solution=base_solution,
            confidence=v7_confidence,
        )

        if not hypotheses:
            # No imagination hypotheses - fall back to V7
            if verbose:
                print(f"  ⚠️  No imagination hypotheses - using V7 fallback")
            return SolverResult(
                output_grid=v7_output,
                confidence=v7_confidence,
                method="v7_fallback",
                time_taken=time.time() - start_time,
            )

        # Step 5: Test hypotheses
        best_hypothesis = self._select_best_hypothesis(
            hypotheses[: self.max_hypotheses], examples, test_input, verbose=verbose
        )

        if best_hypothesis:
            # Calculate final confidence
            final_confidence = self._calculate_final_confidence(
                best_hypothesis, v7_confidence
            )

            if verbose:
                print(
                    f"  🎯 Selected hypothesis with confidence: {final_confidence:.2f}"
                )

            # Update learning
            self.imagination_framework.update_from_result(
                best_hypothesis, success=final_confidence > 0.5
            )

            return SolverResult(
                output_grid=best_hypothesis.grid,
                confidence=final_confidence,
                method="structured_imagination",
                time_taken=time.time() - start_time,
                hypotheses_tested=len(hypotheses),
            )

        # No good hypothesis found - fall back to V7
        if verbose:
            print(f"  ⚠️  No good hypothesis found - using V7 fallback")

        return SolverResult(
            output_grid=v7_output,
            confidence=v7_confidence,
            method="v7_fallback_after_imagination",
            time_taken=time.time() - start_time,
            hypotheses_tested=len(hypotheses),
        )

    def _estimate_confidence(
        self, output: np.ndarray, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> float:
        """Estimate confidence when not provided."""
        if output is None:
            return 0.0

        # Check if output matches any training output exactly
        for _, train_out in examples:
            if output.shape == train_out.shape and np.array_equal(output, train_out):
                return 0.9  # High confidence if exact match

        # Check if output has reasonable properties
        if output.size == 0:
            return 0.1

        # Default moderate confidence
        return 0.5

    def _select_best_hypothesis(
        self,
        hypotheses: List[Hypothesis],
        examples: List[Tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
        verbose: bool = False,
    ) -> Optional[Hypothesis]:
        """Select best hypothesis through empirical testing."""
        if not hypotheses:
            return None

        best_hypothesis = None
        best_score = -1

        for i, hypothesis in enumerate(hypotheses):
            # Score based on multiple factors
            score = self._score_hypothesis(hypothesis, examples, test_input)

            if verbose and i < 5:  # Show first 5
                variations = [v.value for v in hypothesis.variations_applied]
                print(f"    Hypothesis {i+1}: {variations} - Score: {score:.3f}")

            if score > best_score:
                best_score = score
                best_hypothesis = hypothesis

        return best_hypothesis if best_score > 0.3 else None

    def _score_hypothesis(
        self,
        hypothesis: Hypothesis,
        examples: List[Tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
    ) -> float:
        """Score a hypothesis based on multiple criteria."""
        score = 0.0

        # Factor 1: Constraint satisfaction (40%)
        score += 0.4 * hypothesis.constraint_satisfaction

        # Factor 2: Pattern consistency (30%)
        pattern_score = self._check_pattern_consistency(hypothesis.grid, examples)
        score += 0.3 * pattern_score

        # Factor 3: Reasonable novelty (20%)
        # Not too similar (boring) but not too different (random)
        if 0.2 < hypothesis.novelty_score < 0.8:
            novelty_bonus = 1.0 - abs(hypothesis.novelty_score - 0.5) * 2
            score += 0.2 * novelty_bonus

        # Factor 4: Structural similarity (10%)
        struct_score = self._check_structural_similarity(
            hypothesis.grid, test_input, examples
        )
        score += 0.1 * struct_score

        return score

    def _check_pattern_consistency(
        self, output: np.ndarray, examples: List[Tuple[np.ndarray, np.ndarray]]
    ) -> float:
        """Check if output follows patterns from examples."""
        if not examples:
            return 0.5

        scores = []

        # Check color distribution similarity
        output_colors = set(output.flatten())
        for _, ex_out in examples:
            ex_colors = set(ex_out.flatten())
            if output_colors and ex_colors:
                overlap = len(output_colors & ex_colors) / len(
                    output_colors | ex_colors
                )
                scores.append(overlap)

        # Check size ratio similarity
        for ex_in, ex_out in examples:
            if ex_in.size > 0:
                ex_ratio = ex_out.size / ex_in.size
                if output.size > 0:
                    out_ratio = output.size / max(examples[0][0].size, 1)
                    ratio_diff = abs(ex_ratio - out_ratio)
                    scores.append(max(0, 1 - ratio_diff))

        return np.mean(scores) if scores else 0.5

    def _check_structural_similarity(
        self,
        output: np.ndarray,
        test_input: np.ndarray,
        examples: List[Tuple[np.ndarray, np.ndarray]],
    ) -> float:
        """Check structural similarity to examples."""
        # Simple check: non-zero element ratio
        if output.size == 0:
            return 0.0

        output_density = np.count_nonzero(output) / output.size

        # Compare to example densities
        example_densities = []
        for _, ex_out in examples:
            if ex_out.size > 0:
                density = np.count_nonzero(ex_out) / ex_out.size
                example_densities.append(density)

        if example_densities:
            avg_density = np.mean(example_densities)
            diff = abs(output_density - avg_density)
            return max(0, 1 - diff * 2)

        return 0.5

    def _calculate_final_confidence(
        self, hypothesis: Hypothesis, v7_confidence: float
    ) -> float:
        """Calculate final confidence for selected hypothesis."""
        # Weighted combination
        imagination_confidence = (
            hypothesis.constraint_satisfaction * 0.6
            + (1 - hypothesis.novelty_score) * 0.4
        )

        # If V7 had some confidence, blend it in
        if v7_confidence > 0.3:
            return 0.3 * v7_confidence + 0.7 * imagination_confidence

        return imagination_confidence

    def update_thresholds(self, task_success: bool):
        """Adaptively update thresholds based on performance."""
        self.performance_history.append(task_success)

        if len(self.performance_history) >= 10:
            recent_success_rate = sum(self.performance_history[-10:]) / 10

            if recent_success_rate < 0.4:
                # Too aggressive - be more conservative
                self.v7_threshold = max(0.6, self.v7_threshold - 0.05)
                self.imagination_threshold = max(0.4, self.imagination_threshold - 0.05)
            elif recent_success_rate > 0.7:
                # Doing well - can be more adventurous
                self.v7_threshold = min(0.8, self.v7_threshold + 0.05)
                self.imagination_threshold = min(0.7, self.imagination_threshold + 0.05)


def test_hybrid_solver():
    """Test the hybrid solver on sample tasks."""
    print("=" * 60)
    print("HYBRID V7 + STRUCTURED IMAGINATION TEST")
    print("=" * 60)

    # Create solver
    solver = HybridV7StructuredImagination()

    # Test Case 1: Simple pattern (V7 should handle)
    print("\n📋 Test 1: Simple Pattern (Color Inversion)")
    examples = [
        (np.array([[1, 2], [3, 4]]), np.array([[4, 3], [2, 1]])),
        (np.array([[5, 6], [7, 8]]), np.array([[8, 7], [6, 5]])),
    ]
    test_input = np.array([[2, 3], [4, 5]])

    result = solver.solve(examples, test_input, verbose=True)
    print(f"\nResult:")
    print(f"  Method: {result.method}")
    print(f"  Confidence: {result.confidence:.2f}")
    print(f"  Time: {result.time_taken:.3f}s")
    print(f"  Output:\n{result.output_grid}")

    # Test Case 2: Complex pattern (needs imagination)
    print("\n" + "=" * 60)
    print("📋 Test 2: Complex Pattern (Needs Imagination)")

    # Pattern that's hard to detect
    examples = [
        (
            np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]]),
            np.array([[2, 0, 2], [0, 3, 0], [2, 0, 2]]),
        ),  # Add 1, but 1->3
        (
            np.array([[2, 0, 2], [0, 2, 0], [2, 0, 2]]),
            np.array([[3, 0, 3], [0, 4, 0], [3, 0, 3]]),
        ),  # Similar pattern
    ]
    test_input = np.array([[3, 0, 3], [0, 3, 0], [3, 0, 3]])

    result = solver.solve(examples, test_input, verbose=True)
    print(f"\nResult:")
    print(f"  Method: {result.method}")
    print(f"  Confidence: {result.confidence:.2f}")
    print(f"  Hypotheses tested: {result.hypotheses_tested}")
    print(f"  Time: {result.time_taken:.3f}s")
    print(f"  Output:\n{result.output_grid}")

    # Get framework statistics
    stats = solver.imagination_framework.get_statistics()
    print(f"\n📊 Imagination Framework Stats:")
    print(f"  Curriculum Level: {stats['curriculum_level']}/5")
    print(f"  Novelty Level: {stats['novelty_level']:.2f}")
    print(f"  Success Rate: {stats['success_rate']:.1%}")


if __name__ == "__main__":
    test_hybrid_solver()
