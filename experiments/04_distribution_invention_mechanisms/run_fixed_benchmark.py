#!/usr/bin/env python3
"""Run benchmark with fixed V7 solver."""

from utils.imports import setup_project_paths

setup_project_paths()

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from enhanced_arc_solver_v7 import EnhancedARCSolverV7
from enhanced_arc_solver_v7_fixed import EnhancedARCSolverV7Fixed
from tqdm import tqdm


class HybridV7FixedImagination:
    """Hybrid solver using fixed V7 with structured imagination."""

    def __init__(
        self,
        v7_confidence_threshold: float = 0.7,
        imagination_trigger_threshold: float = 0.6,
        max_hypotheses: int = 30,
    ):
        """Initialize hybrid solver with fixed V7."""
        # Use FIXED V7 as base
        self.v7_solver = EnhancedARCSolverV7Fixed(
            use_synthesis=True, use_position_learning=True, confidence_threshold=0.85
        )

        # Import imagination framework
        from structured_imagination_framework import StructuredImaginationFramework

        self.imagination_framework = StructuredImaginationFramework()

        self.v7_threshold = v7_confidence_threshold
        self.imagination_threshold = imagination_trigger_threshold
        self.max_hypotheses = max_hypotheses
        self.performance_history = []

    def solve(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
        verbose: bool = False,
    ):
        """Solve using hybrid approach with fixed V7."""
        start_time = time.time()

        # Try fixed V7 first
        if verbose:
            print("🔍 Attempting Fixed V7 solution...")

        v7_result = self.v7_solver.solve(examples, test_input)
        v7_output = v7_result.output_grid
        v7_confidence = v7_result.confidence

        if verbose:
            print(
                f"  V7 confidence: {v7_confidence:.2f} (method: {v7_result.method_used})"
            )

        # High confidence - use V7
        if v7_confidence >= self.v7_threshold:
            if verbose:
                print(f"  ✅ High confidence - using V7 solution")

            from hybrid_v7_structured_imagination import SolverResult

            return SolverResult(
                output_grid=v7_output,
                confidence=v7_confidence,
                method="v7_fixed",
                time_taken=time.time() - start_time,
                hypotheses_tested=0,
            )

        # Low confidence - use imagination
        if v7_confidence < self.imagination_threshold:
            if verbose:
                if v7_confidence < 0.3:
                    print(f"  ❓ Very low confidence - using pure imagination")
                else:
                    print(f"  🤔 Low confidence - using V7 as base for imagination")

            # Generate hypotheses
            base_hypothesis = v7_output if v7_confidence > 0.3 else None
            hypotheses = self.imagination_framework.generate_hypotheses(
                examples, test_input, base_hypothesis, self.max_hypotheses
            )

            if hypotheses and verbose:
                print(f"  ✨ Generated {len(hypotheses)} valid hypotheses")

            # Score and select best
            if hypotheses:
                best_hypothesis = self.imagination_framework.select_best_hypothesis(
                    hypotheses, examples, test_input
                )

                if best_hypothesis:
                    if verbose:
                        print(
                            f"  🎯 Selected hypothesis with confidence: {best_hypothesis.confidence:.2f}"
                        )

                    from hybrid_v7_structured_imagination import SolverResult

                    return SolverResult(
                        output_grid=best_hypothesis.output_grid,
                        confidence=best_hypothesis.confidence,
                        method="structured_imagination",
                        time_taken=time.time() - start_time,
                        hypotheses_tested=len(hypotheses),
                    )

        # Default: return V7 result
        if verbose:
            print(f"  📊 Moderate confidence - using V7 solution")

        from hybrid_v7_structured_imagination import SolverResult

        return SolverResult(
            output_grid=v7_output,
            confidence=v7_confidence,
            method="v7_fallback",
            time_taken=time.time() - start_time,
            hypotheses_tested=0,
        )


def load_arc_tasks(data_dir: Path, limit: int = None) -> List[Dict]:
    """Load ARC tasks."""
    tasks = []
    task_files = sorted(data_dir.glob("*.json"))

    if limit:
        task_files = task_files[:limit]

    for task_file in task_files:
        with open(task_file, "r") as f:
            task = json.load(f)
            task["id"] = task_file.stem
            tasks.append(task)

    return tasks


def evaluate_task(solver, task: Dict, verbose: bool = False) -> Dict:
    """Evaluate a single task."""
    task_id = task["id"]

    examples = [(np.array(ex["input"]), np.array(ex["output"])) for ex in task["train"]]
    test_cases = task["test"]

    results = {
        "task_id": task_id,
        "correct": 0,
        "total": len(test_cases),
        "accuracy": 0.0,
        "time_taken": 0.0,
        "method_used": [],
        "confidence": [],
    }

    for test_case in test_cases:
        test_input = np.array(test_case["input"])
        expected_output = np.array(test_case["output"])

        start_time = time.time()

        try:
            result = solver.solve(examples, test_input, verbose=verbose)

            if hasattr(result, "output_grid"):
                predicted = result.output_grid
                method = result.method if hasattr(result, "method") else "unknown"
                confidence = result.confidence if hasattr(result, "confidence") else 0.5
            else:
                predicted = result
                method = "unknown"
                confidence = 0.5

            if np.array_equal(predicted, expected_output):
                results["correct"] += 1

            results["method_used"].append(method)
            results["confidence"].append(confidence)

        except Exception as e:
            if verbose:
                print(f"  ❌ Error: {e}")
            results["method_used"].append("error")
            results["confidence"].append(0.0)

        results["time_taken"] += time.time() - start_time

    results["accuracy"] = (
        results["correct"] / results["total"] if results["total"] > 0 else 0.0
    )

    return results


def run_benchmark(
    solver, solver_name: str, tasks: List[Dict], verbose: bool = False
) -> Dict:
    """Run benchmark on tasks."""
    print(f"\n{'='*60}")
    print(f"Testing {solver_name}")
    print(f"{'='*60}")

    all_results = []
    total_correct = 0
    total_cases = 0

    for task in tqdm(tasks, desc=f"Evaluating {solver_name}"):
        result = evaluate_task(solver, task, verbose=False)
        all_results.append(result)
        total_correct += result["correct"]
        total_cases += result["total"]

        if verbose and result["accuracy"] == 1.0:
            print(f"✅ Solved {result['task_id']}")

    overall_accuracy = total_correct / total_cases if total_cases > 0 else 0.0
    solved_tasks = sum(1 for r in all_results if r["accuracy"] == 1.0)

    all_methods = []
    for r in all_results:
        all_methods.extend(r["method_used"])

    method_counts = {}
    for method in all_methods:
        method_counts[method] = method_counts.get(method, 0) + 1

    all_confidences = []
    for r in all_results:
        all_confidences.extend(r["confidence"])

    avg_confidence = np.mean(all_confidences) if all_confidences else 0.0
    total_time = sum(r["time_taken"] for r in all_results)

    return {
        "solver_name": solver_name,
        "overall_accuracy": overall_accuracy,
        "solved_tasks": solved_tasks,
        "total_tasks": len(tasks),
        "solve_rate": solved_tasks / len(tasks) if tasks else 0.0,
        "total_correct": total_correct,
        "total_cases": total_cases,
        "avg_confidence": avg_confidence,
        "total_time": total_time,
        "avg_time_per_task": total_time / len(tasks) if tasks else 0.0,
        "method_counts": method_counts,
        "detailed_results": all_results,
    }


def main():
    """Run fixed benchmark."""

    # Setup paths
    data_dir = Path(
        "experiments/04_distribution_invention_mechanisms/data/arc_agi_official/ARC-AGI/data/evaluation"
    )

    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        return

    # Load tasks
    print("Loading ARC evaluation tasks...")
    tasks = load_arc_tasks(data_dir, limit=10)  # Start with 10 tasks
    print(f"Loaded {len(tasks)} tasks")

    # Initialize solvers
    print("\nInitializing solvers...")

    solvers = {
        "V7 Original": EnhancedARCSolverV7(
            use_synthesis=False,  # Disable for speed
            use_position_learning=True,
            confidence_threshold=0.85,
        ),
        "V7 Fixed": EnhancedARCSolverV7Fixed(
            use_synthesis=False,  # Disable for speed
            use_position_learning=True,
            confidence_threshold=0.85,
        ),
        "Hybrid Fixed+Imagination": HybridV7FixedImagination(
            v7_confidence_threshold=0.7,
            imagination_trigger_threshold=0.6,
            max_hypotheses=30,
        ),
    }

    # Run benchmarks
    all_results = {}

    for solver_name, solver in solvers.items():
        results = run_benchmark(solver, solver_name, tasks, verbose=False)
        all_results[solver_name] = results

        print(f"\n{solver_name} Summary:")
        print(f"  Overall Accuracy: {results['overall_accuracy']*100:.1f}%")
        print(f"  Tasks Solved: {results['solved_tasks']}/{results['total_tasks']}")
        print(f"  Average Time: {results['avg_time_per_task']:.3f}s per task")
        print(f"  Average Confidence: {results['avg_confidence']*100:.1f}%")

    # Print comparison
    print("\n" + "=" * 80)
    print("BENCHMARK COMPARISON")
    print("=" * 80)

    print(
        f"{'Solver':<30} {'Accuracy':>10} {'Solved':>10} {'Avg Time':>12} {'Confidence':>12}"
    )
    print("-" * 80)

    sorted_results = sorted(
        all_results.items(), key=lambda x: x[1]["overall_accuracy"], reverse=True
    )

    for solver_name, res in sorted_results:
        print(
            f"{solver_name:<30} "
            f"{res['overall_accuracy']*100:>9.1f}% "
            f"{res['solved_tasks']:>4}/{res['total_tasks']:<5} "
            f"{res['avg_time_per_task']:>11.3f}s "
            f"{res['avg_confidence']*100:>11.1f}%"
        )

    print("=" * 80)

    # Check improvement
    if "V7 Original" in all_results and "V7 Fixed" in all_results:
        orig_acc = all_results["V7 Original"]["overall_accuracy"]
        fixed_acc = all_results["V7 Fixed"]["overall_accuracy"]

        if fixed_acc > orig_acc:
            print(
                f"\n🎉 Fixed V7 improved by {(fixed_acc - orig_acc)*100:.1f} percentage points!"
            )

    if "Hybrid Fixed+Imagination" in all_results:
        hybrid_acc = all_results["Hybrid Fixed+Imagination"]["overall_accuracy"]
        if "V7 Fixed" in all_results:
            v7_acc = all_results["V7 Fixed"]["overall_accuracy"]
            if hybrid_acc > v7_acc:
                print(
                    f"🌟 Hybrid improved over Fixed V7 by {(hybrid_acc - v7_acc)*100:.1f} percentage points!"
                )


if __name__ == "__main__":
    main()
