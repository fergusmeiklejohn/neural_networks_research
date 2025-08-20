#!/usr/bin/env python3
"""Comprehensive evaluation of V5 solver on multiple ARC tasks.

Tests V3, V4, and V5 solvers on a diverse set of ARC tasks to measure
improvement from position-dependent learning.
"""

from utils.imports import setup_project_paths

setup_project_paths()

import json
import time
from pathlib import Path
from typing import Dict

import numpy as np
from enhanced_arc_solver_v3 import EnhancedARCSolverV3
from enhanced_arc_solver_v4 import EnhancedARCSolverV4
from enhanced_arc_solver_v5 import EnhancedARCSolverV5

# Get data path
DATA_DIR = (
    Path(__file__).parent.parent.parent
    / "data"
    / "arc_agi_official"
    / "ARC-AGI"
    / "data"
    / "training"
)


def load_arc_task(task_path: Path) -> Dict:
    """Load an ARC task from JSON file."""
    with open(task_path) as f:
        return json.load(f)


def evaluate_solver_on_task(
    solver, task: Dict, task_name: str, solver_name: str
) -> Dict:
    """Evaluate a solver on a single task."""

    # Extract training examples
    train_examples = [
        (np.array(ex["input"]), np.array(ex["output"])) for ex in task["train"]
    ]

    # Test on first test example
    test_example = task["test"][0]
    test_input = np.array(test_example["input"])

    # Check for size changes
    size_change = False
    if train_examples:
        in_shape = train_examples[0][0].shape
        out_shape = train_examples[0][1].shape
        if in_shape != out_shape:
            size_change = True

    # Solve
    start_time = time.time()
    solution = solver.solve(train_examples, test_input)
    solve_time = time.time() - start_time

    # Check correctness if ground truth available
    correct = False
    accuracy = 0.0
    if "output" in test_example:
        expected_output = np.array(test_example["output"])

        if solution.output_grid.shape == expected_output.shape:
            correct = np.array_equal(solution.output_grid, expected_output)
            accuracy = (
                np.sum(solution.output_grid == expected_output) / expected_output.size
            )
        else:
            # Wrong shape
            accuracy = 0.0

    return {
        "task": task_name,
        "solver": solver_name,
        "correct": correct,
        "accuracy": accuracy,
        "method": solution.method_used,
        "confidence": solution.confidence,
        "time": solve_time,
        "size_change": size_change,
    }


def comprehensive_evaluation():
    """Run comprehensive evaluation of V3, V4, and V5 solvers."""
    print("=" * 80)
    print("COMPREHENSIVE EVALUATION: V3 vs V4 vs V5")
    print("=" * 80)

    # Initialize solvers (no synthesis for speed)
    solver_v3 = EnhancedARCSolverV3(
        use_synthesis=False,
        force_synthesis_on_size_change=False,
        validate_patterns=True,
    )

    solver_v4 = EnhancedARCSolverV4(
        use_synthesis=False,
        learn_modifications=True,
    )

    solver_v5 = EnhancedARCSolverV5(
        use_synthesis=False,
        use_position_learning=True,
    )

    solvers = [
        (solver_v3, "V3"),
        (solver_v4, "V4"),
        (solver_v5, "V5"),
    ]

    # Select diverse tasks for testing
    # Include tasks we know plus random selection
    specific_tasks = [
        "007bbfb7.json",  # Known tiling task - V5 should excel
        "00d62c1b.json",  # Color pattern task
        "0520fde7.json",  # Spatial transformation
        "05269061.json",  # Object manipulation
        "05f2a901.json",  # Conditional logic
        "0692e18c.json",  # Size change task
        "08ed6ac7.json",  # Pattern repetition
        "0a938d79.json",  # Geometric transformation
        "0b148d64.json",  # Rule-based transformation
        "0ca9ddb6.json",  # Complex pattern
    ]

    # Additional random tasks
    all_task_files = sorted(DATA_DIR.glob("*.json"))
    random_tasks = [
        f.name
        for f in all_task_files[10:25]  # Tasks 10-24
        if f.name not in specific_tasks
    ]

    task_names = specific_tasks + random_tasks[:5]  # Total of 15 tasks

    # Check data exists
    if not DATA_DIR.exists():
        print(f"ERROR: ARC data directory not found at {DATA_DIR}")
        return

    print(f"\nEvaluating on {len(task_names)} diverse ARC tasks...")
    print("Tasks include: tiling, spatial, conditional, and object manipulation\n")

    # Results storage
    all_results = []

    # Evaluate each task
    for i, task_name in enumerate(task_names, 1):
        task_path = DATA_DIR / task_name
        if not task_path.exists():
            continue

        print(f"Task {i}/{len(task_names)}: {task_name.replace('.json', '')}")

        task = load_arc_task(task_path)

        # Check task properties
        train_ex = task["train"][0]
        in_shape = np.array(train_ex["input"]).shape
        out_shape = np.array(train_ex["output"]).shape

        if in_shape != out_shape:
            print(f"  Size change: {in_shape} → {out_shape}")

        # Test each solver
        for solver, solver_name in solvers:
            result = evaluate_solver_on_task(
                solver, task, task_name.replace(".json", ""), solver_name
            )
            all_results.append(result)

            # Show inline results
            status = "✓" if result["correct"] else f"{result['accuracy']:.0%}"
            print(f"  {solver_name}: {status} ({result['method']})")

        print()

    # Analyze results
    print("=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    # Overall accuracy
    for solver_name in ["V3", "V4", "V5"]:
        solver_results = [r for r in all_results if r["solver"] == solver_name]
        correct_count = sum(1 for r in solver_results if r["correct"])
        total = len(solver_results)
        avg_accuracy = (
            sum(r["accuracy"] for r in solver_results) / total if total > 0 else 0
        )

        print(f"\n{solver_name} Performance:")
        print(
            f"  Tasks solved: {correct_count}/{total} ({correct_count/total*100:.1f}%)"
        )
        print(f"  Average accuracy: {avg_accuracy:.1%}")

        # Method breakdown
        method_counts = {}
        for r in solver_results:
            method = r["method"]
            if method not in method_counts:
                method_counts[method] = {"count": 0, "correct": 0, "accuracy_sum": 0}
            method_counts[method]["count"] += 1
            if r["correct"]:
                method_counts[method]["correct"] += 1
            method_counts[method]["accuracy_sum"] += r["accuracy"]

        print("  Methods used:")
        for method, stats in sorted(method_counts.items()):
            avg_acc = (
                stats["accuracy_sum"] / stats["count"] if stats["count"] > 0 else 0
            )
            print(
                f"    {method}: {stats['count']} uses, {stats['correct']} correct, {avg_acc:.0%} avg accuracy"
            )

    # Size-changing tasks analysis
    print("\n" + "-" * 80)
    print("SIZE-CHANGING TASKS ANALYSIS")
    print("-" * 80)

    size_change_results = [r for r in all_results if r["size_change"]]
    if size_change_results:
        for solver_name in ["V3", "V4", "V5"]:
            solver_size_results = [
                r for r in size_change_results if r["solver"] == solver_name
            ]
            if solver_size_results:
                correct = sum(1 for r in solver_size_results if r["correct"])
                total = len(solver_size_results)
                avg_acc = sum(r["accuracy"] for r in solver_size_results) / total
                print(
                    f"{solver_name} on size-changing: {correct}/{total} correct, {avg_acc:.1%} avg accuracy"
                )

    # Direct comparison
    print("\n" + "-" * 80)
    print("DIRECT COMPARISON")
    print("-" * 80)

    # Find tasks where V5 improved
    task_improvements = {}
    for task_name in set(r["task"] for r in all_results):
        v3_result = next(
            (r for r in all_results if r["task"] == task_name and r["solver"] == "V3"),
            None,
        )
        v4_result = next(
            (r for r in all_results if r["task"] == task_name and r["solver"] == "V4"),
            None,
        )
        v5_result = next(
            (r for r in all_results if r["task"] == task_name and r["solver"] == "V5"),
            None,
        )

        if v3_result and v4_result and v5_result:
            v3_acc = v3_result["accuracy"]
            v4_acc = v4_result["accuracy"]
            v5_acc = v5_result["accuracy"]

            if v5_acc > max(v3_acc, v4_acc):
                improvement = v5_acc - max(v3_acc, v4_acc)
                task_improvements[task_name] = {
                    "v3": v3_acc,
                    "v4": v4_acc,
                    "v5": v5_acc,
                    "improvement": improvement,
                }

    if task_improvements:
        print(f"\nV5 improved on {len(task_improvements)} tasks:")
        for task, stats in sorted(
            task_improvements.items(), key=lambda x: x[1]["improvement"], reverse=True
        )[:5]:
            print(
                f"  {task}: V3={stats['v3']:.0%}, V4={stats['v4']:.0%}, V5={stats['v5']:.0%} (+{stats['improvement']:.0%})"
            )

    # Tasks V5 solved that others didn't
    v5_unique_solves = []
    for task_name in set(r["task"] for r in all_results):
        v3_correct = any(
            r["task"] == task_name and r["solver"] == "V3" and r["correct"]
            for r in all_results
        )
        v4_correct = any(
            r["task"] == task_name and r["solver"] == "V4" and r["correct"]
            for r in all_results
        )
        v5_correct = any(
            r["task"] == task_name and r["solver"] == "V5" and r["correct"]
            for r in all_results
        )

        if v5_correct and not (v3_correct or v4_correct):
            v5_unique_solves.append(task_name)

    if v5_unique_solves:
        print(
            f"\n🎉 V5 uniquely solved {len(v5_unique_solves)} tasks: {', '.join(v5_unique_solves)}"
        )

    # Performance summary
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)

    v3_correct = sum(1 for r in all_results if r["solver"] == "V3" and r["correct"])
    v4_correct = sum(1 for r in all_results if r["solver"] == "V4" and r["correct"])
    v5_correct = sum(1 for r in all_results if r["solver"] == "V5" and r["correct"])

    total_tasks = len(task_names)

    improvement_v5_v3 = v5_correct - v3_correct
    improvement_v5_v4 = v5_correct - v4_correct

    if improvement_v5_v3 > 0 or improvement_v5_v4 > 0:
        print(f"✅ V5 shows improvement!")
        if improvement_v5_v3 > 0:
            print(
                f"  +{improvement_v5_v3} tasks solved vs V3 ({improvement_v5_v3/total_tasks*100:.1f}% improvement)"
            )
        if improvement_v5_v4 > 0:
            print(
                f"  +{improvement_v5_v4} tasks solved vs V4 ({improvement_v5_v4/total_tasks*100:.1f}% improvement)"
            )
    else:
        print("Position-dependent learning shows promise but needs refinement")

    # Average accuracy improvement
    v3_avg = (
        sum(r["accuracy"] for r in all_results if r["solver"] == "V3") / total_tasks
    )
    v4_avg = (
        sum(r["accuracy"] for r in all_results if r["solver"] == "V4") / total_tasks
    )
    v5_avg = (
        sum(r["accuracy"] for r in all_results if r["solver"] == "V5") / total_tasks
    )

    print(f"\nAverage accuracy across all tasks:")
    print(f"  V3: {v3_avg:.1%}")
    print(f"  V4: {v4_avg:.1%}")
    print(f"  V5: {v5_avg:.1%}")

    if v5_avg > max(v3_avg, v4_avg):
        print(
            f"\n📈 V5 achieves {(v5_avg - max(v3_avg, v4_avg))*100:.1f} percentage points higher average accuracy!"
        )


if __name__ == "__main__":
    comprehensive_evaluation()
