#!/usr/bin/env python3
"""Test the fixed V7 solver on specific tasks."""

from utils.imports import setup_project_paths

setup_project_paths()

import json
from pathlib import Path

import numpy as np
from enhanced_arc_solver_v7 import EnhancedARCSolverV7
from enhanced_arc_solver_v7_fixed import EnhancedARCSolverV7Fixed


def test_on_task_00576224():
    """Test on the specific tiling task we analyzed."""

    print("=" * 60)
    print("Testing on Task 00576224 (Alternating Row Tiling)")
    print("=" * 60)

    # Load the task
    data_dir = Path(
        "experiments/04_distribution_invention_mechanisms/data/arc_agi_official/ARC-AGI/data/evaluation"
    )
    task_file = data_dir / "00576224.json"

    if not task_file.exists():
        print(f"❌ Task file not found: {task_file}")
        return

    with open(task_file, "r") as f:
        task = json.load(f)

    # Extract examples
    examples = [(np.array(ex["input"]), np.array(ex["output"])) for ex in task["train"]]

    test_input = np.array(task["test"][0]["input"])
    expected_output = np.array(task["test"][0]["output"])

    print(f"Input shape: {test_input.shape}")
    print(f"Expected output shape: {expected_output.shape}")
    print(f"Training examples: {len(examples)}")

    # Test original V7
    print("\n" + "-" * 40)
    print("Original V7 Solver:")
    print("-" * 40)

    v7_original = EnhancedARCSolverV7(
        use_synthesis=True, use_position_learning=True, confidence_threshold=0.85
    )

    result = v7_original.solve(examples, test_input)

    print(f"Output shape: {result.output_grid.shape}")
    print(f"Method: {result.method_used}")
    print(f"Confidence: {result.confidence:.2f}")

    is_correct = np.array_equal(result.output_grid, expected_output)
    print(f"✅ CORRECT!" if is_correct else "❌ INCORRECT")

    if not is_correct and result.output_grid.shape == expected_output.shape:
        diff = np.sum(result.output_grid != expected_output)
        total = result.output_grid.size
        print(f"Accuracy: {(total - diff) / total * 100:.1f}% pixels correct")

    # Test fixed V7
    print("\n" + "-" * 40)
    print("Fixed V7 Solver:")
    print("-" * 40)

    v7_fixed = EnhancedARCSolverV7Fixed(
        use_synthesis=True, use_position_learning=True, confidence_threshold=0.85
    )

    result2 = v7_fixed.solve(examples, test_input)

    print(f"Output shape: {result2.output_grid.shape}")
    print(f"Method: {result2.method_used}")
    print(f"Confidence: {result2.confidence:.2f}")

    is_correct2 = np.array_equal(result2.output_grid, expected_output)
    print(f"✅ CORRECT!" if is_correct2 else "❌ INCORRECT")

    if not is_correct2 and result2.output_grid.shape == expected_output.shape:
        diff = np.sum(result2.output_grid != expected_output)
        total = result2.output_grid.size
        print(f"Accuracy: {(total - diff) / total * 100:.1f}% pixels correct")

    # Show actual vs expected if wrong
    if not is_correct2:
        print("\nFirst 3 rows of output:")
        print("Predicted:")
        print(result2.output_grid[:3])
        print("Expected:")
        print(expected_output[:3])


def test_multiple_tasks():
    """Test on multiple evaluation tasks."""

    print("\n" + "=" * 60)
    print("Testing on Multiple Tasks")
    print("=" * 60)

    data_dir = Path(
        "experiments/04_distribution_invention_mechanisms/data/arc_agi_official/ARC-AGI/data/evaluation"
    )

    # Test on first 5 tasks
    task_files = sorted(data_dir.glob("*.json"))[:5]

    results_original = []
    results_fixed = []

    v7_original = EnhancedARCSolverV7(use_synthesis=True, use_position_learning=True)
    v7_fixed = EnhancedARCSolverV7Fixed(use_synthesis=True, use_position_learning=True)

    for task_file in task_files:
        with open(task_file, "r") as f:
            task = json.load(f)

        examples = [
            (np.array(ex["input"]), np.array(ex["output"])) for ex in task["train"]
        ]

        if not task["test"]:
            continue

        test_input = np.array(task["test"][0]["input"])
        expected_output = np.array(task["test"][0]["output"])

        # Test original
        result1 = v7_original.solve(examples, test_input)
        correct1 = np.array_equal(result1.output_grid, expected_output)
        results_original.append(correct1)

        # Test fixed
        result2 = v7_fixed.solve(examples, test_input)
        correct2 = np.array_equal(result2.output_grid, expected_output)
        results_fixed.append(correct2)

        print(f"\nTask {task_file.stem}:")
        print(
            f"  Original V7: {'✅' if correct1 else '❌'} (method: {result1.method_used}, conf: {result1.confidence:.2f})"
        )
        print(
            f"  Fixed V7:    {'✅' if correct2 else '❌'} (method: {result2.method_used}, conf: {result2.confidence:.2f})"
        )

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    original_accuracy = (
        sum(results_original) / len(results_original) if results_original else 0
    )
    fixed_accuracy = sum(results_fixed) / len(results_fixed) if results_fixed else 0

    print(
        f"Original V7: {sum(results_original)}/{len(results_original)} correct ({original_accuracy*100:.1f}%)"
    )
    print(
        f"Fixed V7:    {sum(results_fixed)}/{len(results_fixed)} correct ({fixed_accuracy*100:.1f}%)"
    )

    if fixed_accuracy > original_accuracy:
        print(
            f"\n🎉 Fixed version improved by {(fixed_accuracy - original_accuracy)*100:.1f}%!"
        )
    elif fixed_accuracy == original_accuracy:
        print(f"\n🤔 No improvement yet, but specific patterns might be better")
    else:
        print(f"\n⚠️ Fixed version performed worse - needs debugging")


if __name__ == "__main__":
    # Test specific task
    test_on_task_00576224()

    # Test multiple tasks
    test_multiple_tasks()
