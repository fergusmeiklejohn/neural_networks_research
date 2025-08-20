#!/usr/bin/env python3
"""Test V3 improvements on real ARC tasks.

This script compares V2 and V3 solvers on the same tasks to measure improvement.
"""

from utils.imports import setup_project_paths

setup_project_paths()

import json
import time
from pathlib import Path
from typing import Dict

import numpy as np
from enhanced_arc_solver_v2 import EnhancedARCSolverV2
from enhanced_arc_solver_v3 import EnhancedARCSolverV3

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


def compare_solvers_on_tasks():
    """Compare V2 and V3 solvers on real ARC tasks."""
    print("=" * 80)
    print("COMPARING V2 vs V3 SOLVERS ON REAL ARC TASKS")
    print("=" * 80)

    # Initialize solvers
    solver_v2 = EnhancedARCSolverV2(use_synthesis=True, adaptive_thresholds=True)
    solver_v3 = EnhancedARCSolverV3(
        use_synthesis=True, force_synthesis_on_size_change=True, validate_patterns=True
    )

    # Check if ARC data exists
    if not DATA_DIR.exists():
        print(f"\nERROR: ARC data directory not found at {DATA_DIR}")
        return

    # Focus on tasks we know have interesting patterns
    # Including task 007bbfb7 which we know uses tiling
    specific_tasks = [
        "007bbfb7.json",  # Known tiling task
        "00d62c1b.json",  # Another task
        "0520fde7.json",
        "05269061.json",
        "05f2a901.json",
    ]

    task_files = []
    for task_name in specific_tasks:
        task_path = DATA_DIR / task_name
        if task_path.exists():
            task_files.append(task_path)

    # Add a few more random tasks
    all_tasks = sorted(DATA_DIR.glob("*.json"))
    for task in all_tasks[:10]:
        if task not in task_files:
            task_files.append(task)

    task_files = task_files[:10]  # Test on 10 tasks total

    if not task_files:
        print(f"\nERROR: No task files found in {DATA_DIR}")
        return

    print(f"\nTesting on {len(task_files)} ARC tasks...")
    print("Will show detailed results for tasks with differences.\n")

    v2_results = []
    v3_results = []

    for i, task_path in enumerate(task_files, 1):
        print(f"\nTask {i}/{len(task_files)}: {task_path.stem}")
        print("-" * 40)

        task = load_arc_task(task_path)

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
                print(f"  Size change: {in_shape} → {out_shape}")

        # Test V2
        print("\n  V2 Solver:")
        start_time = time.time()
        solution_v2 = solver_v2.solve(train_examples, test_input)
        v2_time = time.time() - start_time
        print(f"    Method: {solution_v2.method_used}")
        print(f"    Confidence: {solution_v2.confidence:.3f}")
        print(f"    Time: {v2_time:.2f}s")

        # Test V3
        print("\n  V3 Solver:")
        start_time = time.time()
        solution_v3 = solver_v3.solve(train_examples, test_input)
        v3_time = time.time() - start_time
        print(f"    Method: {solution_v3.method_used}")
        print(f"    Confidence: {solution_v3.confidence:.3f}")
        if solution_v3.actual_confidence > 0:
            print(f"    Validated confidence: {solution_v3.actual_confidence:.3f}")
        print(f"    Time: {v3_time:.2f}s")

        # Check if we have ground truth
        if "output" in test_example:
            expected_output = np.array(test_example["output"])
            v2_correct = np.array_equal(solution_v2.output_grid, expected_output)
            v3_correct = np.array_equal(solution_v3.output_grid, expected_output)

            print(
                f"\n  Correct: V2={'✓' if v2_correct else '✗'}, "
                f"V3={'✓' if v3_correct else '✗'}"
            )

            # Show improvement
            if not v2_correct and v3_correct:
                print("  🎉 V3 IMPROVED! Solved a task V2 couldn't!")
            elif v2_correct and not v3_correct:
                print("  ⚠️  V3 regression - V2 was correct but V3 wasn't")
            elif (
                v2_correct
                and v3_correct
                and solution_v3.method_used != solution_v2.method_used
            ):
                print(f"  📝 Both correct but different methods")

            v2_results.append(
                {
                    "task": task_path.stem,
                    "correct": v2_correct,
                    "method": solution_v2.method_used,
                    "confidence": solution_v2.confidence,
                    "time": v2_time,
                    "size_change": size_change,
                }
            )

            v3_results.append(
                {
                    "task": task_path.stem,
                    "correct": v3_correct,
                    "method": solution_v3.method_used,
                    "confidence": solution_v3.confidence,
                    "actual_confidence": solution_v3.actual_confidence,
                    "time": v3_time,
                    "size_change": size_change,
                }
            )

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    v2_correct = sum(1 for r in v2_results if r["correct"])
    v3_correct = sum(1 for r in v3_results if r["correct"])
    total = len(v2_results)

    print(f"\nV2 Accuracy: {v2_correct}/{total} ({v2_correct/total*100:.1f}%)")
    print(f"V3 Accuracy: {v3_correct}/{total} ({v3_correct/total*100:.1f}%)")

    improvement = v3_correct - v2_correct
    if improvement > 0:
        print(
            f"\n✅ IMPROVEMENT: +{improvement} tasks ({improvement/total*100:.1f}% increase)"
        )
    elif improvement < 0:
        print(f"\n❌ REGRESSION: {improvement} tasks")
    else:
        print(f"\n➖ NO CHANGE in accuracy")

    # Method breakdown for V3
    print("\nV3 Method Usage:")
    method_counts = {}
    for r in v3_results:
        method = r["method"]
        if method not in method_counts:
            method_counts[method] = {"count": 0, "correct": 0}
        method_counts[method]["count"] += 1
        if r["correct"]:
            method_counts[method]["correct"] += 1

    for method, stats in method_counts.items():
        success_rate = stats["correct"] / stats["count"] if stats["count"] > 0 else 0
        print(
            f"  {method}: {stats['count']} uses, "
            f"{stats['correct']} correct ({success_rate:.0%})"
        )

    # Size-change analysis
    size_change_tasks = [r for r in v3_results if r["size_change"]]
    if size_change_tasks:
        size_correct = sum(1 for r in size_change_tasks if r["correct"])
        print(
            f"\nSize-changing tasks: {size_correct}/{len(size_change_tasks)} correct "
            f"({size_correct/len(size_change_tasks)*100:.0f}%)"
        )

        # Compare with V2 on size-changing tasks
        v2_size_tasks = [r for r in v2_results if r["size_change"]]
        v2_size_correct = sum(1 for r in v2_size_tasks if r["correct"])
        if v2_size_tasks:
            print(
                f"  V2 on size-changing: {v2_size_correct}/{len(v2_size_tasks)} "
                f"({v2_size_correct/len(v2_size_tasks)*100:.0f}%)"
            )

    # Performance
    avg_v2_time = sum(r["time"] for r in v2_results) / len(v2_results)
    avg_v3_time = sum(r["time"] for r in v3_results) / len(v3_results)
    print(f"\nAverage time per task:")
    print(f"  V2: {avg_v2_time:.2f}s")
    print(f"  V3: {avg_v3_time:.2f}s")

    # Tasks where V3 improved
    improvements = []
    regressions = []
    for v2, v3 in zip(v2_results, v3_results):
        if not v2["correct"] and v3["correct"]:
            improvements.append(v3["task"])
        elif v2["correct"] and not v3["correct"]:
            regressions.append(v3["task"])

    if improvements:
        print(f"\n🎉 Tasks V3 solved that V2 couldn't: {', '.join(improvements)}")
    if regressions:
        print(f"\n⚠️  Tasks V2 solved that V3 couldn't: {', '.join(regressions)}")

    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    compare_solvers_on_tasks()
