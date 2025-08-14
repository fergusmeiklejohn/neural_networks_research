#!/usr/bin/env python3
"""Comprehensive test of V9 solver on ARC evaluation tasks."""

from utils.imports import setup_project_paths

setup_project_paths()

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from enhanced_arc_solver_v8_comprehensive import EnhancedARCSolverV8
from enhanced_arc_solver_v9 import EnhancedARCSolverV9
from tqdm import tqdm


def load_task(task_file: Path):
    """Load a single task."""
    with open(task_file, "r") as f:
        task = json.load(f)
    return task


def evaluate_solver(
    solver, task_files, solver_name="Solver", verbose=False, limit=None
):
    """Evaluate a solver on multiple tasks."""
    if limit:
        task_files = task_files[:limit]

    results = {
        "solver_name": solver_name,
        "total": len(task_files),
        "solved": 0,
        "by_pattern": defaultdict(lambda: {"total": 0, "solved": 0}),
        "by_method": defaultdict(lambda: {"total": 0, "solved": 0}),
        "by_complexity": defaultdict(lambda: {"total": 0, "solved": 0}),
        "task_results": [],
        "total_time": 0,
    }

    for task_file in tqdm(task_files, desc=f"Testing {solver_name}"):
        task = load_task(task_file)
        task_id = task_file.stem

        if not (task["train"] and task["test"]):
            continue

        # Prepare data
        examples = [
            (np.array(ex["input"]), np.array(ex["output"])) for ex in task["train"]
        ]
        test_input = np.array(task["test"][0]["input"])
        expected = np.array(task["test"][0]["output"])

        # Solve
        start = time.time()
        try:
            result = solver.solve(examples, test_input, verbose=False)
            solve_time = time.time() - start

            # Check correctness
            is_correct = np.array_equal(result.output_grid, expected)

            # Update statistics
            results["total_time"] += solve_time
            if is_correct:
                results["solved"] += 1

            # Track by pattern (V9 only)
            if hasattr(result, "fingerprint"):
                pattern_type = result.fingerprint.size_change_type
                complexity = (
                    int(result.fingerprint.complexity_score * 10) / 10
                )  # Round to 0.1

                results["by_pattern"][pattern_type]["total"] += 1
                if is_correct:
                    results["by_pattern"][pattern_type]["solved"] += 1

                results["by_complexity"][complexity]["total"] += 1
                if is_correct:
                    results["by_complexity"][complexity]["solved"] += 1

            # Track by method
            method = result.method
            results["by_method"][method]["total"] += 1
            if is_correct:
                results["by_method"][method]["solved"] += 1

            # Store individual result
            task_result = {
                "id": task_id,
                "correct": is_correct,
                "method": method,
                "confidence": result.confidence,
                "time": solve_time,
            }

            if hasattr(result, "fingerprint"):
                task_result["pattern"] = result.fingerprint.size_change_type
                task_result["complexity"] = result.fingerprint.complexity_score
                task_result["objects"] = result.fingerprint.object_count_input

            results["task_results"].append(task_result)

        except Exception as e:
            if verbose:
                print(f"Error on {task_id}: {e}")
            results["task_results"].append(
                {
                    "id": task_id,
                    "correct": False,
                    "method": "error",
                    "error": str(e)[:100],
                }
            )

    return results


def print_results(results):
    """Print formatted results."""
    print(f"\n{'='*60}")
    print(f"{results['solver_name']} RESULTS")
    print(f"{'='*60}")

    accuracy = results["solved"] / results["total"] * 100
    avg_time = results["total_time"] / results["total"]

    print(f"\n📊 Overall Performance:")
    print(f"  Accuracy: {results['solved']}/{results['total']} = {accuracy:.1f}%")
    print(f"  Avg time: {avg_time:.3f}s per task")
    print(f"  Total time: {results['total_time']:.1f}s")

    # By pattern type (V9 only)
    if results["by_pattern"]:
        print(f"\n📈 By Pattern Type:")
        for pattern, stats in sorted(results["by_pattern"].items()):
            if stats["total"] > 0:
                acc = stats["solved"] / stats["total"] * 100
                print(
                    f"  {pattern:20s}: {stats['solved']:3d}/{stats['total']:3d} = {acc:5.1f}%"
                )

    # By complexity (V9 only)
    if results["by_complexity"]:
        print(f"\n📊 By Complexity:")
        for complexity, stats in sorted(results["by_complexity"].items()):
            if stats["total"] > 0:
                acc = stats["solved"] / stats["total"] * 100
                print(
                    f"  {complexity:3.1f}: {stats['solved']:3d}/{stats['total']:3d} = {acc:5.1f}%"
                )

    # By method
    print(f"\n🎯 By Method:")
    method_stats = sorted(
        results["by_method"].items(), key=lambda x: x[1]["total"], reverse=True
    )[:10]
    for method, stats in method_stats:
        if stats["total"] > 0:
            acc = stats["solved"] / stats["total"] * 100
            print(
                f"  {method:30s}: {stats['solved']:3d}/{stats['total']:3d} = {acc:5.1f}%"
            )

    # Solved tasks
    solved_tasks = [t for t in results["task_results"] if t["correct"]]
    if solved_tasks:
        print(f"\n✅ Sample Solved Tasks (first 10):")
        for task in solved_tasks[:10]:
            conf = task.get("confidence", 0)
            print(f"  - {task['id']}: {task['method']} (conf: {conf:.2f})")


def compare_solvers(v8_results, v9_results):
    """Compare V8 and V9 performance."""
    print(f"\n{'='*60}")
    print("COMPARISON: V8 vs V9")
    print(f"{'='*60}")

    v8_acc = v8_results["solved"] / v8_results["total"] * 100
    v9_acc = v9_results["solved"] / v9_results["total"] * 100
    improvement = v9_acc - v8_acc

    print(f"\n📊 Accuracy Comparison:")
    print(f"  V8: {v8_results['solved']}/{v8_results['total']} = {v8_acc:.1f}%")
    print(f"  V9: {v9_results['solved']}/{v9_results['total']} = {v9_acc:.1f}%")
    print(f"  Improvement: {improvement:+.1f}%")

    v8_time = v8_results["total_time"] / v8_results["total"]
    v9_time = v9_results["total_time"] / v9_results["total"]
    speedup = v8_time / v9_time if v9_time > 0 else 1.0

    print(f"\n⏱️ Speed Comparison:")
    print(f"  V8: {v8_time:.3f}s per task")
    print(f"  V9: {v9_time:.3f}s per task")
    print(f"  Speedup: {speedup:.1f}x")

    # Find tasks solved by V9 but not V8
    v8_solved_ids = {t["id"] for t in v8_results["task_results"] if t["correct"]}
    v9_solved_ids = {t["id"] for t in v9_results["task_results"] if t["correct"]}

    v9_only = v9_solved_ids - v8_solved_ids
    v8_only = v8_solved_ids - v9_solved_ids
    both = v8_solved_ids & v9_solved_ids

    print(f"\n🔍 Task Analysis:")
    print(f"  Solved by both: {len(both)}")
    print(f"  V9 only: {len(v9_only)}")
    print(f"  V8 only: {len(v8_only)}")

    if v9_only:
        print(f"\n✨ New tasks solved by V9 (first 5):")
        v9_task_map = {t["id"]: t for t in v9_results["task_results"]}
        for task_id in list(v9_only)[:5]:
            task = v9_task_map[task_id]
            print(f"  - {task_id}: {task['method']}")


def main():
    """Run comprehensive evaluation."""
    data_dir = Path("data/arc_agi_official/ARC-AGI/data/evaluation")

    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        return

    # Load tasks
    task_files = sorted(data_dir.glob("*.json"))

    # User can specify how many tasks to test
    num_tasks = min(100, len(task_files))  # Test on 100 tasks

    print(f"Testing on {num_tasks} ARC evaluation tasks")
    print("=" * 60)

    # Test V9
    print("\n🚀 Testing V9 Solver...")
    v9_solver = EnhancedARCSolverV9(use_parallel=True, use_imagination=True)
    v9_results = evaluate_solver(v9_solver, task_files, "V9", limit=num_tasks)
    print_results(v9_results)

    # Test V8 for comparison
    print("\n📦 Testing V8 Solver (baseline)...")
    v8_solver = EnhancedARCSolverV8(use_imagination=True)
    v8_results = evaluate_solver(v8_solver, task_files, "V8", limit=num_tasks)
    print_results(v8_results)

    # Compare
    compare_solvers(v8_results, v9_results)

    # Save results
    output_file = Path("v9_evaluation_results.json")
    with open(output_file, "w") as f:
        json.dump(
            {"v9": v9_results, "v8": v8_results, "num_tasks": num_tasks},
            f,
            indent=2,
            default=str,
        )

    print(f"\n💾 Results saved to {output_file}")


if __name__ == "__main__":
    main()
