#!/usr/bin/env python3
"""Run full evaluation on complete ARC evaluation set."""

from utils.imports import setup_project_paths

setup_project_paths()

import datetime
import json
import time
from collections import defaultdict
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
        self.v7_solver = EnhancedARCSolverV7Fixed(
            use_synthesis=True, use_position_learning=True, confidence_threshold=0.85
        )

        from structured_imagination_framework import StructuredImaginationFramework

        self.imagination_framework = StructuredImaginationFramework()

        self.v7_threshold = v7_confidence_threshold
        self.imagination_threshold = imagination_trigger_threshold
        self.max_hypotheses = max_hypotheses

    def solve(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]],
        test_input: np.ndarray,
        verbose: bool = False,
    ):
        """Solve using hybrid approach with fixed V7."""
        start_time = time.time()

        # Try fixed V7 first
        v7_result = self.v7_solver.solve(examples, test_input)
        v7_output = v7_result.output_grid
        v7_confidence = v7_result.confidence

        # High confidence - use V7
        if v7_confidence >= self.v7_threshold:
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
            base_hypothesis = v7_output if v7_confidence > 0.3 else None
            hypotheses = self.imagination_framework.generate_hypotheses(
                examples, test_input, base_hypothesis, self.max_hypotheses
            )

            if hypotheses:
                best_hypothesis = self.imagination_framework.select_best_hypothesis(
                    hypotheses, examples, test_input
                )

                if best_hypothesis:
                    from hybrid_v7_structured_imagination import SolverResult

                    return SolverResult(
                        output_grid=best_hypothesis.output_grid,
                        confidence=best_hypothesis.confidence,
                        method="structured_imagination",
                        time_taken=time.time() - start_time,
                        hypotheses_tested=len(hypotheses),
                    )

        # Default: return V7 result
        from hybrid_v7_structured_imagination import SolverResult

        return SolverResult(
            output_grid=v7_output,
            confidence=v7_confidence,
            method="v7_fallback",
            time_taken=time.time() - start_time,
            hypotheses_tested=0,
        )


def load_all_evaluation_tasks(data_dir: Path) -> List[Dict]:
    """Load all ARC evaluation tasks."""
    tasks = []
    task_files = sorted(data_dir.glob("*.json"))

    print(f"Loading {len(task_files)} evaluation tasks...")

    for task_file in tqdm(task_files, desc="Loading tasks"):
        with open(task_file, "r") as f:
            task = json.load(f)
            task["id"] = task_file.stem

            # Analyze task characteristics
            if task["train"] and task["test"]:
                input_shape = np.array(task["train"][0]["input"]).shape
                output_shape = np.array(task["train"][0]["output"]).shape
                np.array(task["test"][0]["input"]).shape
                np.array(task["test"][0]["output"]).shape

                task["has_size_change"] = input_shape != output_shape
                task["is_tiling"] = (
                    output_shape[0] > input_shape[0]
                    and output_shape[1] > input_shape[1]
                    and output_shape[0] % input_shape[0] == 0
                    and output_shape[1] % input_shape[1] == 0
                )
                task["input_size"] = input_shape
                task["output_size"] = output_shape

                tasks.append(task)

    return tasks


def evaluate_solver_on_task(solver, task: Dict, timeout: float = 30.0) -> Dict:
    """Evaluate solver on a single task with timeout."""
    task_id = task["id"]

    examples = [(np.array(ex["input"]), np.array(ex["output"])) for ex in task["train"]]

    results = {
        "task_id": task_id,
        "has_size_change": task.get("has_size_change", False),
        "is_tiling": task.get("is_tiling", False),
        "correct": False,
        "method": None,
        "confidence": 0.0,
        "time_taken": 0.0,
        "error": None,
    }

    if not task["test"]:
        results["error"] = "No test cases"
        return results

    test_input = np.array(task["test"][0]["input"])
    expected_output = np.array(task["test"][0]["output"])

    start_time = time.time()

    try:
        # Set a reasonable timeout
        result = solver.solve(examples, test_input, verbose=False)

        if hasattr(result, "output_grid"):
            predicted = result.output_grid
            results["method"] = (
                result.method if hasattr(result, "method") else result.method_used
            )
            results["confidence"] = result.confidence
        else:
            predicted = result
            results["method"] = "unknown"
            results["confidence"] = 0.5

        results["correct"] = np.array_equal(predicted, expected_output)

    except Exception as e:
        results["error"] = str(e)
        results["method"] = "error"

    results["time_taken"] = time.time() - start_time

    # Timeout check
    if results["time_taken"] > timeout:
        results["error"] = "Timeout"
        results["method"] = "timeout"

    return results


def run_full_evaluation(solver, solver_name: str, tasks: List[Dict]) -> Dict:
    """Run evaluation on all tasks."""
    print(f"\n{'='*60}")
    print(f"Evaluating {solver_name} on {len(tasks)} tasks")
    print(f"{'='*60}")

    all_results = []

    # Statistics
    stats = {
        "total_tasks": len(tasks),
        "tasks_solved": 0,
        "tasks_with_size_change": 0,
        "size_change_solved": 0,
        "tiling_tasks": 0,
        "tiling_solved": 0,
        "methods_used": defaultdict(int),
        "errors": defaultdict(int),
        "total_time": 0.0,
    }

    # Progress bar
    for task in tqdm(tasks, desc=f"Evaluating {solver_name}"):
        result = evaluate_solver_on_task(solver, task)
        all_results.append(result)

        # Update statistics
        if result["correct"]:
            stats["tasks_solved"] += 1

            if result["has_size_change"]:
                stats["size_change_solved"] += 1

            if result["is_tiling"]:
                stats["tiling_solved"] += 1

        if result["has_size_change"]:
            stats["tasks_with_size_change"] += 1

        if result["is_tiling"]:
            stats["tiling_tasks"] += 1

        if result["method"]:
            stats["methods_used"][result["method"]] += 1

        if result["error"]:
            stats["errors"][result["error"]] += 1

        stats["total_time"] += result["time_taken"]

    # Calculate percentages
    stats["overall_accuracy"] = stats["tasks_solved"] / stats["total_tasks"] * 100
    stats["avg_time_per_task"] = stats["total_time"] / stats["total_tasks"]

    if stats["tasks_with_size_change"] > 0:
        stats["size_change_accuracy"] = (
            stats["size_change_solved"] / stats["tasks_with_size_change"] * 100
        )
    else:
        stats["size_change_accuracy"] = 0.0

    if stats["tiling_tasks"] > 0:
        stats["tiling_accuracy"] = stats["tiling_solved"] / stats["tiling_tasks"] * 100
    else:
        stats["tiling_accuracy"] = 0.0

    return {"solver_name": solver_name, "stats": stats, "results": all_results}


def print_evaluation_summary(evaluations: List[Dict]):
    """Print comprehensive evaluation summary."""
    print("\n" + "=" * 80)
    print("FULL EVALUATION SUMMARY")
    print("=" * 80)

    # Overall comparison
    print("\n📊 Overall Performance:")
    print("-" * 60)
    print(f"{'Solver':<25} {'Accuracy':>10} {'Solved':>10} {'Avg Time':>12}")
    print("-" * 60)

    for eval_data in evaluations:
        stats = eval_data["stats"]
        print(
            f"{eval_data['solver_name']:<25} "
            f"{stats['overall_accuracy']:>9.1f}% "
            f"{stats['tasks_solved']:>4}/{stats['total_tasks']:<5} "
            f"{stats['avg_time_per_task']:>11.3f}s"
        )

    # Task type analysis
    print("\n📈 Performance by Task Type:")
    print("-" * 60)
    print(f"{'Solver':<25} {'Size Change':>15} {'Tiling':>15}")
    print("-" * 60)

    for eval_data in evaluations:
        stats = eval_data["stats"]
        size_str = f"{stats['size_change_accuracy']:.1f}% ({stats['size_change_solved']}/{stats['tasks_with_size_change']})"
        tile_str = f"{stats['tiling_accuracy']:.1f}% ({stats['tiling_solved']}/{stats['tiling_tasks']})"
        print(f"{eval_data['solver_name']:<25} {size_str:>15} {tile_str:>15}")

    # Method usage for best performer
    best_eval = max(evaluations, key=lambda x: x["stats"]["overall_accuracy"])
    print(f"\n🎯 Methods Used by {best_eval['solver_name']}:")
    print("-" * 60)

    total_methods = sum(best_eval["stats"]["methods_used"].values())
    for method, count in sorted(
        best_eval["stats"]["methods_used"].items(), key=lambda x: x[1], reverse=True
    )[:10]:
        percentage = count / total_methods * 100
        print(f"  {method:<20} {count:>5} ({percentage:>5.1f}%)")

    # Error analysis
    if any(eval_data["stats"]["errors"] for eval_data in evaluations):
        print("\n⚠️ Errors Encountered:")
        print("-" * 60)
        for eval_data in evaluations:
            if eval_data["stats"]["errors"]:
                print(f"{eval_data['solver_name']}:")
                for error, count in eval_data["stats"]["errors"].items():
                    print(f"  {error}: {count}")

    # Successful tasks
    print("\n✅ Tasks Solved by Each Solver:")
    print("-" * 60)

    for eval_data in evaluations:
        solved_tasks = [r["task_id"] for r in eval_data["results"] if r["correct"]]
        if solved_tasks:
            print(f"{eval_data['solver_name']}: {len(solved_tasks)} tasks")
            print(f"  First 5: {', '.join(solved_tasks[:5])}")


def save_results(evaluations: List[Dict], output_dir: Path):
    """Save evaluation results to file."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save detailed results as JSON
    output_file = output_dir / f"full_evaluation_{timestamp}.json"

    # Convert numpy values for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, defaultdict):
            return dict(obj)
        return obj

    with open(output_file, "w") as f:
        json.dump(evaluations, f, indent=2, default=convert_for_json)

    print(f"\n💾 Results saved to: {output_file}")

    # Save summary as markdown
    summary_file = output_dir / f"evaluation_summary_{timestamp}.md"

    with open(summary_file, "w") as f:
        f.write(f"# ARC Evaluation Results\n")
        f.write(f"*{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")

        f.write("## Overall Results\n\n")
        f.write("| Solver | Accuracy | Tasks Solved | Avg Time |\n")
        f.write("|--------|----------|--------------|----------|\n")

        for eval_data in evaluations:
            stats = eval_data["stats"]
            f.write(
                f"| {eval_data['solver_name']} | "
                f"{stats['overall_accuracy']:.1f}% | "
                f"{stats['tasks_solved']}/{stats['total_tasks']} | "
                f"{stats['avg_time_per_task']:.3f}s |\n"
            )

        f.write("\n## Task Type Performance\n\n")
        f.write("| Solver | Size Change Tasks | Tiling Tasks |\n")
        f.write("|--------|-------------------|---------------|\n")

        for eval_data in evaluations:
            stats = eval_data["stats"]
            f.write(
                f"| {eval_data['solver_name']} | "
                f"{stats['size_change_accuracy']:.1f}% ({stats['size_change_solved']}/{stats['tasks_with_size_change']}) | "
                f"{stats['tiling_accuracy']:.1f}% ({stats['tiling_solved']}/{stats['tiling_tasks']}) |\n"
            )

    print(f"📄 Summary saved to: {summary_file}")


def main():
    """Run full evaluation."""

    # Setup paths
    data_dir = Path(
        "experiments/04_distribution_invention_mechanisms/data/arc_agi_official/ARC-AGI/data/evaluation"
    )
    output_dir = Path("experiments/04_distribution_invention_mechanisms/outputs")
    output_dir.mkdir(exist_ok=True)

    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        return

    # Load all tasks
    tasks = load_all_evaluation_tasks(data_dir)
    print(f"\n📚 Loaded {len(tasks)} evaluation tasks")

    # Analyze task distribution
    size_change_tasks = sum(1 for t in tasks if t.get("has_size_change", False))
    tiling_tasks = sum(1 for t in tasks if t.get("is_tiling", False))
    print(
        f"  - Tasks with size change: {size_change_tasks} ({size_change_tasks/len(tasks)*100:.1f}%)"
    )
    print(f"  - Tiling tasks: {tiling_tasks} ({tiling_tasks/len(tasks)*100:.1f}%)")

    # Initialize solvers
    print("\n🔧 Initializing solvers...")

    solvers = [
        (
            "V7 Original",
            EnhancedARCSolverV7(
                use_synthesis=False,  # Disable for speed
                use_position_learning=True,
                confidence_threshold=0.85,
            ),
        ),
        (
            "V7 Fixed",
            EnhancedARCSolverV7Fixed(
                use_synthesis=False,  # Disable for speed
                use_position_learning=True,
                confidence_threshold=0.85,
            ),
        ),
        (
            "Hybrid Fixed+Imagination",
            HybridV7FixedImagination(
                v7_confidence_threshold=0.7,
                imagination_trigger_threshold=0.6,
                max_hypotheses=30,
            ),
        ),
    ]

    # Run evaluations
    evaluations = []

    for solver_name, solver in solvers:
        eval_result = run_full_evaluation(solver, solver_name, tasks)
        evaluations.append(eval_result)

        # Print immediate summary
        stats = eval_result["stats"]
        print(f"\n{solver_name} Results:")
        print(
            f"  ✅ Solved: {stats['tasks_solved']}/{stats['total_tasks']} ({stats['overall_accuracy']:.1f}%)"
        )
        print(f"  ⏱️ Total time: {stats['total_time']:.1f}s")
        print(f"  📊 Avg time per task: {stats['avg_time_per_task']:.3f}s")

    # Print comprehensive summary
    print_evaluation_summary(evaluations)

    # Save results
    save_results(evaluations, output_dir)

    # Final comparison
    print("\n" + "=" * 80)
    print("🏆 FINAL RESULTS")
    print("=" * 80)

    if len(evaluations) >= 2:
        orig_acc = evaluations[0]["stats"]["overall_accuracy"]
        fixed_acc = evaluations[1]["stats"]["overall_accuracy"]

        if fixed_acc > orig_acc:
            improvement = fixed_acc - orig_acc
            print(f"✨ Fixed V7 improved by {improvement:.1f} percentage points!")
            print(f"   Relative improvement: {(fixed_acc/orig_acc - 1)*100:.1f}%")

    if len(evaluations) >= 3:
        hybrid_acc = evaluations[2]["stats"]["overall_accuracy"]
        if hybrid_acc > fixed_acc:
            improvement = hybrid_acc - fixed_acc
            print(
                f"🌟 Hybrid solver improved by additional {improvement:.1f} percentage points!"
            )


if __name__ == "__main__":
    main()
