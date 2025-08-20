#!/usr/bin/env python3
"""Test V6 improvements against V5 on multiple ARC tasks."""

from utils.imports import setup_project_paths

setup_project_paths()

import json
import time
from pathlib import Path
from typing import Dict

import numpy as np
from enhanced_arc_solver_v5 import EnhancedARCSolverV5
from enhanced_arc_solver_v6 import EnhancedARCSolverV6

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


def test_v6_improvements():
    """Test V6 improvements on a selection of ARC tasks."""
    print("=" * 80)
    print("TESTING V6 IMPROVEMENTS")
    print("=" * 80)

    # Initialize solvers
    solver_v5 = EnhancedARCSolverV5(
        use_synthesis=False,  # Disable for speed
        use_position_learning=True,
    )

    solver_v6 = EnhancedARCSolverV6(
        use_synthesis=False,  # Disable for speed
        use_position_learning=True,
        confidence_threshold=0.85,
    )

    # Test on tasks where V5 had issues
    test_tasks = [
        "007bbfb7.json",  # Position-dependent tiling
        "00d62c1b.json",  # V5 fell back to TTA unnecessarily
        "05f2a901.json",  # Another unnecessary TTA
        "0a938d79.json",  # Simple pattern task
        "0d3d703e.json",  # V5 showed improvement
        "08ed6ac7.json",  # V5 completely failed with TTA
        "0ca9ddb6.json",  # Complex pattern
    ]

    if not DATA_DIR.exists():
        print(f"ERROR: ARC data directory not found at {DATA_DIR}")
        return

    print(f"\nTesting on {len(test_tasks)} selected tasks...")
    print("Focus: Tasks where V5 fell back to TTA unnecessarily\n")

    v5_results = []
    v6_results = []

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

        # Test V5
        print("  V5:", end=" ")
        start = time.time()
        solution_v5 = solver_v5.solve(train_examples, test_input)
        v5_time = time.time() - start

        # Test V6
        print("\n  V6:", end=" ")
        start = time.time()
        solution_v6 = solver_v6.solve(train_examples, test_input)
        v6_time = time.time() - start

        # Evaluate correctness
        if "output" in test_example:
            expected_output = np.array(test_example["output"])

            v5_acc = 0.0
            v6_acc = 0.0

            if solution_v5.output_grid.shape == expected_output.shape:
                v5_acc = (
                    np.sum(solution_v5.output_grid == expected_output)
                    / expected_output.size
                )

            if solution_v6.output_grid.shape == expected_output.shape:
                v6_acc = (
                    np.sum(solution_v6.output_grid == expected_output)
                    / expected_output.size
                )

            v5_results.append(
                {
                    "task": task_name.replace(".json", ""),
                    "accuracy": v5_acc,
                    "method": solution_v5.method_used,
                    "time": v5_time,
                }
            )

            v6_results.append(
                {
                    "task": task_name.replace(".json", ""),
                    "accuracy": v6_acc,
                    "method": solution_v6.method_used,
                    "time": v6_time,
                }
            )

            # Show results
            print(f"\n  Results:")
            print(f"    V5: {v5_acc:.1%} ({solution_v5.method_used})")
            print(f"    V6: {v6_acc:.1%} ({solution_v6.method_used})")

            if v6_acc > v5_acc:
                print(
                    f"    ✅ V6 improved by {(v6_acc - v5_acc)*100:.1f} percentage points!"
                )
            elif v6_acc < v5_acc:
                print(
                    f"    ⚠️  V6 decreased by {(v5_acc - v6_acc)*100:.1f} percentage points"
                )

            # Highlight method changes
            if solution_v5.method_used == "tta" and solution_v6.method_used != "tta":
                print(f"    📝 V6 avoided TTA fallback (used {solution_v6.method_used})")

        print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Method usage comparison
    print("\nMethod Usage:")
    print("-" * 40)

    v5_methods = {}
    v6_methods = {}

    for r in v5_results:
        method = r["method"]
        if method not in v5_methods:
            v5_methods[method] = {"count": 0, "total_acc": 0}
        v5_methods[method]["count"] += 1
        v5_methods[method]["total_acc"] += r["accuracy"]

    for r in v6_results:
        method = r["method"]
        if method not in v6_methods:
            v6_methods[method] = {"count": 0, "total_acc": 0}
        v6_methods[method]["count"] += 1
        v6_methods[method]["total_acc"] += r["accuracy"]

    print("V5 methods:")
    for method, stats in v5_methods.items():
        avg_acc = stats["total_acc"] / stats["count"] if stats["count"] > 0 else 0
        print(f"  {method}: {stats['count']} uses, {avg_acc:.1%} avg accuracy")

    print("\nV6 methods:")
    for method, stats in v6_methods.items():
        avg_acc = stats["total_acc"] / stats["count"] if stats["count"] > 0 else 0
        print(f"  {method}: {stats['count']} uses, {avg_acc:.1%} avg accuracy")

    # TTA reduction
    v5_tta_count = v5_methods.get("tta", {}).get("count", 0)
    v6_tta_count = v6_methods.get("tta", {}).get("count", 0)

    if v5_tta_count > v6_tta_count:
        print(
            f"\n✅ V6 reduced TTA usage: {v5_tta_count} → {v6_tta_count} (-{v5_tta_count - v6_tta_count})"
        )

    # Accuracy comparison
    print("\nAccuracy Comparison:")
    print("-" * 40)

    v5_avg = (
        sum(r["accuracy"] for r in v5_results) / len(v5_results) if v5_results else 0
    )
    v6_avg = (
        sum(r["accuracy"] for r in v6_results) / len(v6_results) if v6_results else 0
    )

    print(f"V5 average: {v5_avg:.1%}")
    print(f"V6 average: {v6_avg:.1%}")

    if v6_avg > v5_avg:
        print(f"\n🎉 V6 shows {(v6_avg - v5_avg)*100:.1f} percentage point improvement!")

    # Task-by-task improvements
    improvements = []
    for v5r, v6r in zip(v5_results, v6_results):
        if v6r["accuracy"] > v5r["accuracy"]:
            improvements.append(
                {
                    "task": v5r["task"],
                    "improvement": v6r["accuracy"] - v5r["accuracy"],
                    "v5_method": v5r["method"],
                    "v6_method": v6r["method"],
                }
            )

    if improvements:
        print(f"\nTasks with improvement ({len(improvements)}):")
        for imp in sorted(improvements, key=lambda x: x["improvement"], reverse=True):
            print(
                f"  {imp['task']}: +{imp['improvement']*100:.1f}% ({imp['v5_method']} → {imp['v6_method']})"
            )

    print("\n" + "=" * 80)
    print("V6 TESTING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    test_v6_improvements()
