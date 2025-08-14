#!/usr/bin/env python3
"""Test V8 solver on evaluation tasks."""

from utils.imports import setup_project_paths

setup_project_paths()

import json
from pathlib import Path

import numpy as np
from enhanced_arc_solver_v8_comprehensive import EnhancedARCSolverV8
from tqdm import tqdm


def test_on_sample():
    """Test V8 on a sample of tasks."""

    data_dir = Path("data/arc_agi_official/ARC-AGI/data/evaluation")

    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        return

    # Load sample tasks
    task_files = sorted(data_dir.glob("*.json"))[:50]  # Test on 50 tasks

    print(f"Testing V8 Solver on {len(task_files)} tasks")
    print("=" * 60)

    solver = EnhancedARCSolverV8(use_imagination=True)

    results = {
        "total": len(task_files),
        "solved": 0,
        "by_type": {"size_change": 0, "no_size_change": 0},
        "by_method": {},
        "task_details": [],
    }

    for task_file in tqdm(task_files, desc="Testing"):
        with open(task_file, "r") as f:
            task = json.load(f)

        task_id = task_file.stem

        if task["train"] and task["test"]:
            examples = [
                (np.array(ex["input"]), np.array(ex["output"])) for ex in task["train"]
            ]
            test_input = np.array(task["test"][0]["input"])
            expected = np.array(task["test"][0]["output"])

            # Detect task type
            has_size_change = examples[0][0].shape != examples[0][1].shape
            task_type = "size_change" if has_size_change else "no_size_change"

            # Solve
            result = solver.solve(examples, test_input, verbose=False)

            # Check if correct
            is_correct = np.array_equal(result.output_grid, expected)

            if is_correct:
                results["solved"] += 1
                results["by_type"][task_type] += 1

            # Track method
            method = result.method
            if method not in results["by_method"]:
                results["by_method"][method] = {"total": 0, "correct": 0}
            results["by_method"][method]["total"] += 1
            if is_correct:
                results["by_method"][method]["correct"] += 1

            # Store details
            results["task_details"].append(
                {
                    "id": task_id,
                    "type": task_type,
                    "method": method,
                    "confidence": result.confidence,
                    "correct": is_correct,
                    "time": result.time_taken,
                }
            )

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    accuracy = results["solved"] / results["total"] * 100
    print(f"\n📊 Overall: {results['solved']}/{results['total']} = {accuracy:.1f}%")

    # By task type
    print("\n📈 By Task Type:")
    for task_type, count in results["by_type"].items():
        type_total = sum(1 for d in results["task_details"] if d["type"] == task_type)
        if type_total > 0:
            type_acc = count / type_total * 100
            print(f"  {task_type}: {count}/{type_total} = {type_acc:.1f}%")

    # By method
    print("\n🎯 By Method Used:")
    for method, stats in sorted(
        results["by_method"].items(), key=lambda x: x[1]["total"], reverse=True
    ):
        if stats["total"] > 0:
            method_acc = stats["correct"] / stats["total"] * 100
            print(
                f"  {method}: {stats['correct']}/{stats['total']} = {method_acc:.1f}%"
            )

    # Solved tasks
    solved_tasks = [d for d in results["task_details"] if d["correct"]]
    if solved_tasks:
        print("\n✅ Solved Tasks:")
        for task in solved_tasks[:10]:  # Show first 10
            print(
                f"  - {task['id']}: {task['method']} (conf: {task['confidence']:.2f})"
            )

    # Performance summary
    summary = solver.get_performance_summary()
    print("\n⚡ Performance Summary:")
    print(f"  Average confidence: {summary['avg_confidence']:.2f}")
    print(f"  Average time: {summary['avg_time']:.3f}s")

    return results


if __name__ == "__main__":
    test_on_sample()
