#!/usr/bin/env python3
"""Enhanced ARC Solver V9 with intelligent object manipulation and pattern fingerprinting.

Major improvements:
1. Smart object manipulation that learns from examples
2. Pattern fingerprinting for efficient primitive selection
3. Parallel primitive testing for speed
4. Better integration of all components
"""

from utils.imports import setup_project_paths

setup_project_paths()

import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

warnings.filterwarnings("ignore")

# Import our components
from comprehensive_pattern_library import ComprehensivePatternLibrary, TransformResult
from enhanced_object_manipulation import SmartObjectManipulator
from enhanced_tiling_primitives import (
    AlternatingRowTile,
    CheckerboardTile,
    SmartTilePattern,
)
from pattern_fingerprinting import PatternFingerprint, PatternFingerprinter
from structured_imagination_framework import StructuredImaginationFramework


@dataclass
class V9SolverResult:
    """Result from V9 solver."""

    output_grid: np.ndarray
    confidence: float
    method: str
    fingerprint: PatternFingerprint
    time_taken: float
    attempts_made: int
    parallel_results: Dict[str, float]  # Method -> confidence scores


class EnhancedARCSolverV9:
    """V9 solver with comprehensive improvements."""

    def __init__(
        self,
        use_imagination: bool = True,
        use_parallel: bool = True,
        confidence_threshold: float = 0.7,
        max_parallel_workers: int = 4,
    ):
        """Initialize V9 solver."""
        # Core components
        self.pattern_library = ComprehensivePatternLibrary()
        self.object_manipulator = SmartObjectManipulator(verbose=False)
        self.fingerprinter = PatternFingerprinter()
        self.imagination = StructuredImaginationFramework() if use_imagination else None

        # Tiling primitives
        self.tiling_primitives = {
            "smart_tiling": SmartTilePattern(),
            "alternating_tile": AlternatingRowTile(),
            "checkerboard": CheckerboardTile(),
        }

        # Configuration
        self.use_parallel = use_parallel
        self.confidence_threshold = confidence_threshold
        self.max_workers = max_parallel_workers

        # Performance tracking
        self.solve_history = []

    def solve(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
        verbose: bool = False,
    ) -> V9SolverResult:
        """Main solving method with intelligent primitive selection."""
        start_time = time.time()
        attempts = 0

        # Step 1: Fingerprint the task
        fingerprint = self.fingerprinter.fingerprint(examples, test_input)

        if verbose:
            print(f"🔍 V9 Solver analyzing task...")
            print(f"  Pattern type: {fingerprint.size_change_type}")
            print(
                f"  Objects: {fingerprint.object_count_input} -> {fingerprint.object_count_output}"
            )
            print(f"  Complexity: {fingerprint.complexity_score:.2f}")
            print(f"  Recommended: {fingerprint.recommended_primitives[:3]}")

        # Step 2: Apply primitives based on fingerprint
        if self.use_parallel and len(fingerprint.recommended_primitives) > 1:
            result, parallel_results = self._solve_parallel(
                examples, test_input, fingerprint, verbose
            )
            attempts = len(parallel_results)
        else:
            result, parallel_results = self._solve_sequential(
                examples, test_input, fingerprint, verbose
            )
            attempts = 1

        # Step 3: Fallback to imagination if needed
        if result.confidence < self.confidence_threshold and self.imagination:
            if verbose:
                print(
                    f"  📊 Low confidence ({result.confidence:.2f}), trying imagination..."
                )

            base_output = result.output if hasattr(result, "output") else test_input
            hypotheses = self.imagination.imagine(
                examples, test_input, base_output, confidence=result.confidence
            )

            if hypotheses:
                best = max(hypotheses, key=lambda h: h.constraint_satisfaction)
                if best.constraint_satisfaction > result.confidence:
                    result = TransformResult(
                        best.grid,
                        best.constraint_satisfaction,
                        f"imagination_{len(best.variations_applied)}",
                    )
                    attempts += len(hypotheses)

        # Create final result
        final_result = V9SolverResult(
            output_grid=result.output if hasattr(result, "output") else test_input,
            confidence=result.confidence,
            method=result.method,
            fingerprint=fingerprint,
            time_taken=time.time() - start_time,
            attempts_made=attempts,
            parallel_results=parallel_results,
        )

        # Track performance
        self.solve_history.append(
            {
                "fingerprint": fingerprint,
                "method": final_result.method,
                "confidence": final_result.confidence,
                "time": final_result.time_taken,
            }
        )

        if verbose:
            print(
                f"  ✅ Solution: {final_result.method} (confidence: {final_result.confidence:.2f})"
            )
            print(
                f"  ⏱️ Time: {final_result.time_taken:.3f}s, Attempts: {final_result.attempts_made}"
            )

        return final_result

    def _solve_parallel(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
        fingerprint: PatternFingerprint,
        verbose: bool,
    ) -> Tuple[TransformResult, Dict]:
        """Solve using parallel primitive testing."""
        parallel_results = {}
        futures = {}

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit tasks for recommended primitives
            for primitive_name in fingerprint.recommended_primitives[
                : self.max_workers
            ]:
                future = executor.submit(
                    self._apply_primitive, primitive_name, examples, test_input
                )
                futures[future] = primitive_name

            # Collect results
            best_result = None
            best_confidence = 0.0

            for future in as_completed(futures):
                primitive_name = futures[future]
                try:
                    result = future.result(timeout=2.0)
                    parallel_results[primitive_name] = result.confidence

                    if result.confidence > best_confidence:
                        best_result = result
                        best_confidence = result.confidence

                    if verbose:
                        print(f"    {primitive_name}: {result.confidence:.2f}")

                except Exception as e:
                    parallel_results[primitive_name] = 0.0
                    if verbose:
                        print(f"    {primitive_name}: failed ({str(e)[:50]})")

        if best_result is None:
            best_result = TransformResult(test_input, 0.0, "all_failed")

        return best_result, parallel_results

    def _solve_sequential(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
        fingerprint: PatternFingerprint,
        verbose: bool,
    ) -> Tuple[TransformResult, Dict]:
        """Solve using sequential primitive testing."""
        parallel_results = {}

        for primitive_name in fingerprint.recommended_primitives[:3]:
            try:
                result = self._apply_primitive(primitive_name, examples, test_input)
                parallel_results[primitive_name] = result.confidence

                if verbose:
                    print(f"    {primitive_name}: {result.confidence:.2f}")

                # Early stopping if high confidence
                if result.confidence >= self.confidence_threshold:
                    return result, parallel_results

            except Exception:
                parallel_results[primitive_name] = 0.0
                if verbose:
                    print(f"    {primitive_name}: failed")

        # Return best result
        if parallel_results:
            best_name = max(parallel_results, key=parallel_results.get)
            if parallel_results[best_name] > 0:
                return (
                    self._apply_primitive(best_name, examples, test_input),
                    parallel_results,
                )

        return TransformResult(test_input, 0.0, "all_failed"), parallel_results

    def _apply_primitive(
        self,
        primitive_name: str,
        examples: List[Tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
    ) -> TransformResult:
        """Apply a specific primitive by name."""

        # Smart tiling primitives
        if primitive_name == "smart_tiling":
            result = self.tiling_primitives["smart_tiling"].learn_and_apply(
                examples, test_input
            )
            if result is not None:
                confidence = self._validate_on_examples(examples, test_input, result)
                return TransformResult(result, confidence, "smart_tiling")
            return TransformResult(test_input, 0.0, "smart_tiling_failed")

        # Object manipulation
        elif primitive_name == "object_manipulation":
            try:
                result = self.object_manipulator.solve(examples, test_input)
                confidence = self._validate_on_examples(examples, test_input, result)
                return TransformResult(result, confidence, "object_manipulation")
            except:
                return TransformResult(test_input, 0.0, "object_manipulation_failed")

        # Pattern library primitives
        else:
            # Map primitive name to actual primitive
            primitive_map = {
                "color_mapping": "ColorMapper",
                "rotation_reflection": "RotationReflection",
                "symmetry_operations": "SymmetryApplier",
                "pattern_completion": "PatternCompleter",
                "object_movement": "ObjectExtractor",  # For now
            }

            if primitive_name in primitive_map:
                for primitive in self.pattern_library.primitives:
                    if primitive.__class__.__name__ == primitive_map[primitive_name]:
                        if primitive.can_apply(examples, test_input):
                            return primitive.apply(examples, test_input)

            # Fallback to pattern library's best guess
            return self.pattern_library.apply_best_primitive(examples, test_input)

    def _validate_on_examples(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
        output: np.ndarray,
    ) -> float:
        """Validate output against training examples."""
        if not examples:
            return 0.5

        # Check shape consistency
        expected_shape = examples[0][1].shape
        if output.shape == expected_shape:
            base_score = 0.6
        else:
            base_score = 0.2

        # Check color consistency
        expected_colors = set()
        for _, out in examples:
            expected_colors.update(np.unique(out))

        output_colors = set(np.unique(output))
        color_overlap = len(output_colors & expected_colors) / max(
            len(expected_colors), 1
        )

        return base_score + 0.4 * color_overlap

    def get_performance_summary(self) -> Dict:
        """Get performance summary."""
        if not self.solve_history:
            return {"message": "No tasks solved yet"}

        summary = {
            "total_tasks": len(self.solve_history),
            "avg_confidence": np.mean([h["confidence"] for h in self.solve_history]),
            "avg_time": np.mean([h["time"] for h in self.solve_history]),
            "avg_complexity": np.mean(
                [h["fingerprint"].complexity_score for h in self.solve_history]
            ),
            "methods_used": {},
            "top_patterns": [],
        }

        # Count methods
        for h in self.solve_history:
            method = h["method"]
            if method not in summary["methods_used"]:
                summary["methods_used"][method] = 0
            summary["methods_used"][method] += 1

        # Top patterns
        pattern_counts = {}
        for h in self.solve_history:
            for prim in h["fingerprint"].recommended_primitives[:2]:
                pattern_counts[prim] = pattern_counts.get(prim, 0) + 1

        summary["top_patterns"] = sorted(
            pattern_counts.items(), key=lambda x: x[1], reverse=True
        )[:5]

        return summary


if __name__ == "__main__":
    print("Enhanced ARC Solver V9")
    print("=" * 60)

    solver = EnhancedARCSolverV9(use_parallel=True)

    print("\n🚀 Features:")
    print("  ✅ Pattern fingerprinting for task analysis")
    print("  ✅ Smart object manipulation with learning")
    print("  ✅ Parallel primitive testing")
    print("  ✅ Intelligent primitive prioritization")
    print("  ✅ All V8 features plus improvements")

    print("\n📊 Components:")
    print(f"  - Tiling primitives: {len(solver.tiling_primitives)}")
    print(f"  - Pattern library: {len(solver.pattern_library.primitives)} primitives")
    print(f"  - Object manipulator: Enabled")
    print(f"  - Parallel workers: {solver.max_workers}")

    print("\n🎯 Ready to solve ARC tasks with improved accuracy!")
