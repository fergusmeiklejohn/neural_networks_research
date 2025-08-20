#!/usr/bin/env python3
"""
Full benchmark test of Hybrid V7 + Structured Imagination.

Tests on complete ARC evaluation set and compares with all baselines:
- V7 baseline: 66.6% accuracy
- Old imagination: 45.5% accuracy
- Target: >70% accuracy
"""

from utils.imports import setup_project_paths

setup_project_paths()

import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
from enhanced_arc_solver_v7 import EnhancedARCSolverV7
from hybrid_v7_structured_imagination import HybridV7StructuredImagination
from imaginative_solver import ImaginativeSolver  # Old imagination approach
from tqdm import tqdm


def load_arc_tasks(data_dir: Path, limit: int = None) -> List[Dict]:
    """Load ARC tasks from evaluation directory."""
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
    """Evaluate a single task with a solver."""
    task_id = task["id"]

    # Extract examples and test cases
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
            # Get solver result
            if hasattr(solver, "solve"):
                result = solver.solve(examples, test_input, verbose=verbose)

                # Extract output based on solver type
                if hasattr(result, "output_grid"):
                    predicted = result.output_grid
                    method = result.method if hasattr(result, "method") else "unknown"
                    confidence = (
                        result.confidence if hasattr(result, "confidence") else 0.5
                    )
                else:
                    predicted = result
                    method = "unknown"
                    confidence = 0.5
            else:
                # For old solvers without structured output
                predicted = solver(examples, test_input)
                method = "unknown"
                confidence = 0.5

            # Check correctness
            if np.array_equal(predicted, expected_output):
                results["correct"] += 1

            results["method_used"].append(method)
            results["confidence"].append(confidence)

        except Exception as e:
            if verbose:
                print(f"  ❌ Error on task {task_id}: {e}")
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
    """Run benchmark on all tasks."""
    print(f"\n{'='*60}")
    print(f"Testing {solver_name}")
    print(f"{'='*60}")

    all_results = []
    total_correct = 0
    total_cases = 0

    for task in tqdm(tasks, desc=f"Evaluating {solver_name}"):
        result = evaluate_task(
            solver, task, verbose=False
        )  # Set to False for cleaner output
        all_results.append(result)
        total_correct += result["correct"]
        total_cases += result["total"]

        if verbose and result["accuracy"] == 1.0:
            print(f"✅ Solved {result['task_id']}")

    # Calculate overall stats
    overall_accuracy = total_correct / total_cases if total_cases > 0 else 0.0
    solved_tasks = sum(1 for r in all_results if r["accuracy"] == 1.0)

    # Method usage stats
    all_methods = []
    for r in all_results:
        all_methods.extend(r["method_used"])

    method_counts = {}
    for method in all_methods:
        method_counts[method] = method_counts.get(method, 0) + 1

    # Confidence stats
    all_confidences = []
    for r in all_results:
        all_confidences.extend(r["confidence"])

    avg_confidence = np.mean(all_confidences) if all_confidences else 0.0

    # Total time
    total_time = sum(r["time_taken"] for r in all_results)

    benchmark_results = {
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

    return benchmark_results


def print_comparison(results: Dict[str, Dict]):
    """Print comparison table of all results."""
    print("\n" + "=" * 80)
    print("BENCHMARK COMPARISON")
    print("=" * 80)

    # Header
    print(
        f"{'Solver':<30} {'Accuracy':>10} {'Solved':>10} {'Avg Time':>12} {'Confidence':>12}"
    )
    print("-" * 80)

    # Sort by accuracy
    sorted_results = sorted(
        results.items(), key=lambda x: x[1]["overall_accuracy"], reverse=True
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

    # Method usage for hybrid solver
    if "Hybrid V7+Imagination" in results:
        hybrid_res = results["Hybrid V7+Imagination"]
        if "method_counts" in hybrid_res:
            print("\nHybrid Solver Method Usage:")
            print("-" * 40)
            total_methods = sum(hybrid_res["method_counts"].values())
            for method, count in sorted(
                hybrid_res["method_counts"].items(), key=lambda x: x[1], reverse=True
            ):
                percentage = (count / total_methods * 100) if total_methods > 0 else 0
                print(f"  {method:<20} {count:>5} ({percentage:>5.1f}%)")


def main():
    """Run full benchmark test."""
    # Setup paths
    data_dir = Path(
        "experiments/04_distribution_invention_mechanisms/data/arc_agi_official/ARC-AGI/data/evaluation"
    )

    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        print("Please ensure ARC-AGI dataset is downloaded.")
        return

    # Load tasks (use limit for testing, None for full benchmark)
    print("Loading ARC evaluation tasks...")
    tasks = load_arc_tasks(data_dir, limit=50)  # Start with 50 for testing
    print(f"Loaded {len(tasks)} tasks")

    # Initialize solvers
    print("\nInitializing solvers...")

    solvers = {
        "V7 Baseline": EnhancedARCSolverV7(
            use_synthesis=True, use_position_learning=True, confidence_threshold=0.85
        ),
        "Hybrid V7+Imagination": HybridV7StructuredImagination(
            v7_confidence_threshold=0.7,
            imagination_trigger_threshold=0.6,
            max_hypotheses=30,
        ),
    }

    # Optionally add old imagination solver if available
    try:
        solvers["Old Imagination"] = ImaginativeSolver(
            base_solver=EnhancedARCSolverV7(), max_hypotheses=10
        )
    except:
        print("⚠️ Old imagination solver not available for comparison")

    # Run benchmarks
    all_results = {}

    for solver_name, solver in solvers.items():
        results = run_benchmark(solver, solver_name, tasks, verbose=False)
        all_results[solver_name] = results

        # Print summary for this solver
        print(f"\n{solver_name} Summary:")
        print(f"  Overall Accuracy: {results['overall_accuracy']*100:.1f}%")
        print(f"  Tasks Solved: {results['solved_tasks']}/{results['total_tasks']}")
        print(f"  Average Time: {results['avg_time_per_task']:.3f}s per task")
        print(f"  Average Confidence: {results['avg_confidence']*100:.1f}%")

    # Print comparison
    print_comparison(all_results)

    # Save detailed results
    output_dir = Path("experiments/04_distribution_invention_mechanisms/outputs")
    output_dir.mkdir(exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"benchmark_results_{timestamp}.json"

    # Convert numpy values for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        return obj

    # Save results
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2, default=convert_for_json)

    print(f"\n✅ Results saved to: {output_file}")

    # Check if target met
    if "Hybrid V7+Imagination" in all_results:
        hybrid_acc = all_results["Hybrid V7+Imagination"]["overall_accuracy"] * 100
        if hybrid_acc > 70:
            print(
                f"\n🎉 SUCCESS! Hybrid solver achieved {hybrid_acc:.1f}% accuracy (target: >70%)"
            )
        else:
            print(
                f"\n⚠️ Hybrid solver achieved {hybrid_acc:.1f}% accuracy (target: >70%)"
            )
            print("Need to analyze failures and improve variation strategies.")


if __name__ == "__main__":
    main()
