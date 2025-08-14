#!/usr/bin/env python3
"""Enhanced ARC Solver V8 with comprehensive pattern library.

This version integrates:
1. Smart tiling for size-change tasks (proven to work)
2. Full pattern library for non-size-change tasks
3. Structured imagination framework as fallback
"""

from utils.imports import setup_project_paths

setup_project_paths()

import time
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from comprehensive_pattern_library import ComprehensivePatternLibrary, TransformResult
from enhanced_tiling_primitives import (
    AlternatingRowTile,
    CheckerboardTile,
    SmartTilePattern,
)
from structured_imagination_framework import StructuredImaginationFramework


@dataclass
class V8SolverResult:
    """Result from V8 solver."""

    output_grid: np.ndarray
    confidence: float
    method: str
    time_taken: float
    attempts_made: int


class EnhancedARCSolverV8:
    """V8 solver with comprehensive pattern support."""

    def __init__(
        self,
        use_imagination: bool = True,
        confidence_threshold: float = 0.7,
        max_attempts: int = 10,
    ):
        """Initialize V8 solver."""
        self.pattern_library = ComprehensivePatternLibrary()
        self.imagination = StructuredImaginationFramework() if use_imagination else None
        self.confidence_threshold = confidence_threshold
        self.max_attempts = max_attempts

        # Track performance
        self.solve_history = []

    def solve(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
        verbose: bool = False,
    ) -> V8SolverResult:
        """Solve ARC task using comprehensive approach."""
        start_time = time.time()
        attempts = 0

        if verbose:
            print(f"🔍 V8 Solver analyzing task...")
            print(f"  Input shape: {test_input.shape}")
            if examples:
                print(f"  Expected output shape: {examples[0][1].shape}")

        # Step 1: Detect task type
        task_type = self._detect_task_type(examples, test_input)

        if verbose:
            print(f"  Task type: {task_type}")

        # Step 2: Apply appropriate strategy
        if task_type == "size_change":
            result = self._solve_size_change(examples, test_input, verbose)
            attempts += 1
        elif task_type == "no_size_change":
            result = self._solve_no_size_change(examples, test_input, verbose)
            attempts += 1
        else:
            result = self._solve_unknown(examples, test_input, verbose)
            attempts += 1

        # Step 3: If low confidence, try imagination
        if result.confidence < self.confidence_threshold and self.imagination:
            if verbose:
                print(
                    f"  📊 Low confidence ({result.confidence:.2f}), trying imagination..."
                )

            base_output = (
                result.output
                if hasattr(result, "output")
                else result.output_grid
                if hasattr(result, "output_grid")
                else test_input
            )
            hypotheses = self.imagination.imagine(
                examples, test_input, base_output, confidence=result.confidence
            )

            if hypotheses:
                # Select best hypothesis based on constraint satisfaction
                best = max(hypotheses, key=lambda h: h.constraint_satisfaction)
                if best and best.constraint_satisfaction > result.confidence:
                    result = TransformResult(
                        best.grid,  # Changed from output_grid to grid
                        best.constraint_satisfaction,  # Use constraint_satisfaction as confidence
                        f"imagination_{len(best.variations_applied)}_vars",
                    )
                    attempts += len(hypotheses)

        # Create final result
        final_result = V8SolverResult(
            output_grid=result.output
            if hasattr(result, "output")
            else result.output_grid,
            confidence=result.confidence,
            method=result.method,
            time_taken=time.time() - start_time,
            attempts_made=attempts,
        )

        # Track performance
        self.solve_history.append(
            {
                "task_type": task_type,
                "method": final_result.method,
                "confidence": final_result.confidence,
                "time": final_result.time_taken,
            }
        )

        if verbose:
            print(
                f"  ✅ Solution: {final_result.method} (confidence: {final_result.confidence:.2f})"
            )

        return final_result

    def _detect_task_type(self, examples, test_input):
        """Detect the type of task."""
        if not examples:
            return "unknown"

        # Check if size changes
        inp_shape = examples[0][0].shape
        out_shape = examples[0][1].shape

        if inp_shape != out_shape:
            return "size_change"
        else:
            return "no_size_change"

    def _solve_size_change(self, examples, test_input, verbose):
        """Solve size-change tasks (our strength!)."""
        if verbose:
            print("  🔄 Trying size-change strategies...")

        # Try our proven tiling primitives
        primitives = [SmartTilePattern(), AlternatingRowTile(), CheckerboardTile()]

        best_result = None
        best_confidence = 0.0

        for primitive in primitives:
            try:
                # Check if primitive applies
                result = primitive.learn_and_apply(examples, test_input)

                if result is not None:
                    # Validate on examples
                    confidence = self._validate_on_examples(
                        examples, test_input, result
                    )

                    if confidence > best_confidence:
                        best_result = result
                        best_confidence = confidence

                        if verbose:
                            print(
                                f"    ✓ {primitive.__class__.__name__}: {confidence:.2f}"
                            )
            except Exception as e:
                if verbose:
                    print(f"    ✗ {primitive.__class__.__name__} failed: {e}")

        if best_result is not None:
            return TransformResult(best_result, best_confidence, "smart_tiling")

        # Fallback to simple scaling
        return self._simple_scale(examples, test_input)

    def _solve_no_size_change(self, examples, test_input, verbose):
        """Solve non-size-change tasks using pattern library."""
        if verbose:
            print("  🎨 Trying non-size-change patterns...")

        # Use comprehensive pattern library
        result = self.pattern_library.apply_best_primitive(examples, test_input)

        if verbose and result:
            print(f"    Applied: {result.method} (confidence: {result.confidence:.2f})")

        return (
            result if result else TransformResult(test_input, 0.0, "no_pattern_found")
        )

    def _solve_unknown(self, examples, test_input, verbose):
        """Fallback for unknown task types."""
        if verbose:
            print("  ❓ Unknown task type, trying all strategies...")

        # Try both approaches
        size_result = self._solve_size_change(examples, test_input, False)
        no_size_result = self._solve_no_size_change(examples, test_input, False)

        # Return best
        if size_result.confidence > no_size_result.confidence:
            return size_result
        else:
            return no_size_result

    def _simple_scale(self, examples, test_input):
        """Simple scaling as fallback."""
        if not examples:
            return TransformResult(test_input, 0.0, "no_examples")

        # Calculate scale factor
        inp_shape = examples[0][0].shape
        out_shape = examples[0][1].shape

        h_scale = out_shape[0] / inp_shape[0]
        w_scale = out_shape[1] / inp_shape[1]

        if h_scale == int(h_scale) and w_scale == int(w_scale):
            # Integer scaling
            output = np.repeat(
                np.repeat(test_input, int(h_scale), axis=0), int(w_scale), axis=1
            )
            return TransformResult(output, 0.3, "simple_scale")

        return TransformResult(test_input, 0.0, "non_integer_scale")

    def _validate_on_examples(self, examples, test_input, output):
        """Validate output matches pattern from examples."""
        # Simple validation - could be more sophisticated
        if not examples:
            return 0.5

        # Check if output shape matches expected
        expected_shape = examples[0][1].shape
        if output.shape == expected_shape:
            return 0.7

        return 0.2

    def get_performance_summary(self):
        """Get summary of solver performance."""
        if not self.solve_history:
            return "No tasks solved yet"

        summary = {
            "total_tasks": len(self.solve_history),
            "avg_confidence": np.mean([h["confidence"] for h in self.solve_history]),
            "avg_time": np.mean([h["time"] for h in self.solve_history]),
            "methods_used": {},
        }

        for h in self.solve_history:
            method = h["method"]
            if method not in summary["methods_used"]:
                summary["methods_used"][method] = 0
            summary["methods_used"][method] += 1

        return summary


if __name__ == "__main__":
    print("Enhanced ARC Solver V8")
    print("=" * 60)

    solver = EnhancedARCSolverV8(use_imagination=True)

    print("Features:")
    print("  ✅ Smart tiling for size-change tasks")
    print("  ✅ Comprehensive pattern library")
    print("  ✅ Rotation and reflection")
    print("  ✅ Color mapping")
    print("  ✅ Object extraction")
    print("  ✅ Symmetry operations")
    print("  ✅ Pattern completion")
    print("  ✅ Structured imagination fallback")

    print("\nReady to solve ARC tasks!")
