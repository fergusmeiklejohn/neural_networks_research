#!/usr/bin/env python3
"""Test V7 with program synthesis on ARC tasks."""

from utils.imports import setup_project_paths

setup_project_paths()

import json
import time
from pathlib import Path
from typing import Dict

import numpy as np
from enhanced_arc_solver_v6 import EnhancedARCSolverV6
from enhanced_arc_solver_v7 import EnhancedARCSolverV7

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


def test_v7_synthesis():
    """Test V7 with synthesis on selected ARC tasks."""
    print("=" * 80)
    print("TESTING V7 WITH PROGRAM SYNTHESIS")
    print("=" * 80)

    # Initialize solvers
    solver_v6 = EnhancedARCSolverV6(
        use_synthesis=False,  # V6 without synthesis
        use_position_learning=True,
        confidence_threshold=0.85,
    )

    solver_v7 = EnhancedARCSolverV7(
        use_synthesis=True,  # V7 with synthesis enabled
        synthesis_timeout=8.0,
        synthesis_confidence_threshold=0.75,
        use_position_learning=True,
        confidence_threshold=0.85,
    )

    # Test on tasks that might benefit from synthesis
    test_tasks = [
        "007bbfb7.json",  # Complex tiling
        "00d62c1b.json",  # Color pattern
        "05f2a901.json",  # Conditional logic
        "0a938d79.json",  # Simple pattern
        "0d3d703e.json",  # Complex spatial
        "08ed6ac7.json",  # Pattern task
        "0ca9ddb6.json",  # Complex pattern
        "05269061.json",  # Object manipulation
    ]

    if not DATA_DIR.exists():
        print(f"ERROR: ARC data directory not found at {DATA_DIR}")
        return

    print(f"\nTesting on {len(test_tasks)} tasks...")
    print("Focus: Tasks that might benefit from program synthesis\n")

    v6_results = []
    v7_results = []

    for i, task_name in enumerate(test_tasks, 1):
        task_path = DATA_DIR / task_name
        if not task_path.exists():
            continue

        print(f"Task {i}/{len(test_tasks)}: {task_name.replace('.json', '')}")

        task = load_arc_task(task_path)

        # Extract training examples
        train_examples = [
            (np.array(ex["input"]), np.array(ex["output"])) for ex in task["train"]
        ]

        # Test on first test example
        test_example = task["test"][0]
        test_input = np.array(test_example["input"])

        # Check for size changes
        if train_examples:
            in_shape = train_examples[0][0].shape
            out_shape = train_examples[0][1].shape
            if in_shape != out_shape:
                print(f"  Size change: {in_shape} → {out_shape}")

        # Test V6 (no synthesis)
        print("  V6 (no synthesis):", end=" ")
        start = time.time()
        solution_v6 = solver_v6.solve(train_examples, test_input)
        v6_time = time.time() - start

        # Test V7 (with synthesis)
        print("\n  V7 (with synthesis):", end=" ")
        start = time.time()
        solution_v7 = solver_v7.solve(train_examples, test_input)
        v7_time = time.time() - start

        # Evaluate correctness
        if "output" in test_example:
            expected_output = np.array(test_example["output"])

            v6_acc = 0.0
            v7_acc = 0.0

            if solution_v6.output_grid.shape == expected_output.shape:
                v6_acc = (
                    np.sum(solution_v6.output_grid == expected_output)
                    / expected_output.size
                )
                v6_correct = np.array_equal(solution_v6.output_grid, expected_output)
            else:
                v6_correct = False

            if solution_v7.output_grid.shape == expected_output.shape:
                v7_acc = (
                    np.sum(solution_v7.output_grid == expected_output)
                    / expected_output.size
                )
                v7_correct = np.array_equal(solution_v7.output_grid, expected_output)
            else:
                v7_correct = False

            v6_results.append(
                {
                    "task": task_name.replace(".json", ""),
                    "accuracy": v6_acc,
                    "correct": v6_correct,
                    "method": solution_v6.method_used,
                    "time": v6_time,
                }
            )

            v7_results.append(
                {
                    "task": task_name.replace(".json", ""),
                    "accuracy": v7_acc,
                    "correct": v7_correct,
                    "method": solution_v7.method_used,
                    "time": v7_time,
                }
            )

            # Show results
            print(f"\n  Results:")
            print(
                f"    V6: {v6_acc:.1%} ({solution_v6.method_used}) {'✓' if v6_correct else '✗'}"
            )
            print(
                f"    V7: {v7_acc:.1%} ({solution_v7.method_used}) {'✓' if v7_correct else '✗'}"
            )

            if v7_acc > v6_acc:
                print(
                    f"    ✅ V7 improved by {(v7_acc - v6_acc)*100:.1f} percentage points!"
                )

            # Highlight synthesis usage
            if "synthesis" in solution_v7.method_used:
                print(f"    🔧 V7 used program synthesis!")
                if hasattr(solution_v7, "program") and solution_v7.program:
                    print(
                        f"    Program: {len(solution_v7.program.operations)} operations"
                    )

        print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Method usage comparison
    print("\nMethod Usage:")
    print("-" * 40)

    v6_methods = {}
    v7_methods = {}

    for r in v6_results:
        method = r["method"]
        if method not in v6_methods:
            v6_methods[method] = {"count": 0, "correct": 0}
        v6_methods[method]["count"] += 1
        if r["correct"]:
            v6_methods[method]["correct"] += 1

    for r in v7_results:
        method = r["method"]
        if method not in v7_methods:
            v7_methods[method] = {"count": 0, "correct": 0}
        v7_methods[method]["count"] += 1
        if r["correct"]:
            v7_methods[method]["correct"] += 1

    print("V6 methods (no synthesis):")
    for method, stats in v6_methods.items():
        print(f"  {method}: {stats['count']} uses, {stats['correct']} correct")

    print("\nV7 methods (with synthesis):")
    for method, stats in v7_methods.items():
        print(f"  {method}: {stats['count']} uses, {stats['correct']} correct")

    # Synthesis usage
    synthesis_count = sum(1 for r in v7_results if "synthesis" in r["method"])
    synthesis_correct = sum(
        1 for r in v7_results if "synthesis" in r["method"] and r["correct"]
    )

    if synthesis_count > 0:
        print(f"\n🔧 Synthesis Usage:")
        print(f"  Used in {synthesis_count}/{len(v7_results)} tasks")
        print(f"  Correct: {synthesis_correct}/{synthesis_count}")

    # Accuracy comparison
    print("\nAccuracy Comparison:")
    print("-" * 40)

    v6_correct_count = sum(1 for r in v6_results if r["correct"])
    v7_correct_count = sum(1 for r in v7_results if r["correct"])

    v6_avg = (
        sum(r["accuracy"] for r in v6_results) / len(v6_results) if v6_results else 0
    )
    v7_avg = (
        sum(r["accuracy"] for r in v7_results) / len(v7_results) if v7_results else 0
    )

    print(f"V6 (no synthesis):")
    print(f"  Tasks solved: {v6_correct_count}/{len(v6_results)}")
    print(f"  Average accuracy: {v6_avg:.1%}")

    print(f"\nV7 (with synthesis):")
    print(f"  Tasks solved: {v7_correct_count}/{len(v7_results)}")
    print(f"  Average accuracy: {v7_avg:.1%}")

    if v7_avg > v6_avg:
        print(f"\n🎉 V7 shows {(v7_avg - v6_avg)*100:.1f} percentage point improvement!")

    if v7_correct_count > v6_correct_count:
        print(f"✅ V7 solved {v7_correct_count - v6_correct_count} more tasks!")

    # Task-by-task improvements
    improvements = []
    for v6r, v7r in zip(v6_results, v7_results):
        if v7r["accuracy"] > v6r["accuracy"]:
            improvements.append(
                {
                    "task": v6r["task"],
                    "improvement": v7r["accuracy"] - v6r["accuracy"],
                    "v6_method": v6r["method"],
                    "v7_method": v7r["method"],
                }
            )

    if improvements:
        print(f"\nTasks with improvement ({len(improvements)}):")
        for imp in sorted(improvements, key=lambda x: x["improvement"], reverse=True)[
            :5
        ]:
            synthesis_used = "synthesis" in imp["v7_method"]
            mark = "🔧" if synthesis_used else "📈"
            print(
                f"  {mark} {imp['task']}: +{imp['improvement']*100:.1f}% ({imp['v6_method']} → {imp['v7_method']})"
            )

    # Performance analysis
    avg_v6_time = sum(r["time"] for r in v6_results) / len(v6_results)
    avg_v7_time = sum(r["time"] for r in v7_results) / len(v7_results)

    print(f"\nPerformance:")
    print(f"  V6 avg time: {avg_v6_time:.2f}s")
    print(f"  V7 avg time: {avg_v7_time:.2f}s (includes synthesis)")

    print("\n" + "=" * 80)
    print("V7 SYNTHESIS TESTING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    test_v7_synthesis()
