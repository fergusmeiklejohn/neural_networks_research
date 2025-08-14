#!/usr/bin/env python3
"""Debug why solvers are failing on training set."""

from utils.imports import setup_project_paths

setup_project_paths()

import json
import traceback
from pathlib import Path

import numpy as np
from enhanced_arc_solver_v7 import EnhancedARCSolverV7
from enhanced_arc_solver_v7_fixed import EnhancedARCSolverV7Fixed


def test_single_training_task():
    """Test on a single training task with detailed debugging."""

    # Try to find training data
    data_dirs = [
        Path(
            "experiments/04_distribution_invention_mechanisms/data/arc_agi_official/ARC-AGI/data/training"
        ),
        Path("data/arc_agi_official/ARC-AGI/data/training"),
        Path(
            "experiments/04_distribution_invention_mechanisms/data/arc/sample_tasks.json"
        ),
    ]

    task_file = None
    for data_dir in data_dirs:
        if data_dir.exists():
            if data_dir.is_file():
                task_file = data_dir
                break
            else:
                task_files = list(data_dir.glob("*.json"))
                if task_files:
                    task_file = task_files[0]
                    break

    if not task_file:
        print("❌ No task files found in any location")
        print("Searched:")
        for d in data_dirs:
            print(f"  - {d}")
        return

    print(f"Loading task from: {task_file}")

    try:
        with open(task_file, "r") as f:
            data = json.load(f)

        # Handle different formats
        if isinstance(data, list):
            # It's a list of tasks
            task = data[0]
        elif "train" in data and "test" in data:
            # It's a single task
            task = data
        else:
            print(
                f"❌ Unknown task format: {data.keys() if isinstance(data, dict) else type(data)}"
            )
            return

        print(
            f"Task has {len(task['train'])} training examples and {len(task['test'])} test cases"
        )

        # Extract examples
        examples = [
            (np.array(ex["input"]), np.array(ex["output"])) for ex in task["train"]
        ]

        test_input = np.array(task["test"][0]["input"])
        expected_output = np.array(task["test"][0]["output"])

        print(f"Input shape: {test_input.shape}")
        print(f"Expected output shape: {expected_output.shape}")

        # Test V7 Original
        print("\n" + "=" * 60)
        print("Testing V7 Original")
        print("=" * 60)

        try:
            v7_original = EnhancedARCSolverV7(
                use_synthesis=False, use_position_learning=True
            )

            result = v7_original.solve(examples, test_input)

            print(f"✅ Solver succeeded!")
            print(f"  Output shape: {result.output_grid.shape}")
            print(f"  Method: {result.method_used}")
            print(f"  Confidence: {result.confidence:.2f}")

            is_correct = np.array_equal(result.output_grid, expected_output)
            print(f"  Correct: {'YES' if is_correct else 'NO'}")

        except Exception as e:
            print(f"❌ Solver failed with error:")
            print(f"  {type(e).__name__}: {e}")
            traceback.print_exc()

        # Test V7 Fixed
        print("\n" + "=" * 60)
        print("Testing V7 Fixed")
        print("=" * 60)

        try:
            v7_fixed = EnhancedARCSolverV7Fixed(
                use_synthesis=False, use_position_learning=True
            )

            result = v7_fixed.solve(examples, test_input)

            print(f"✅ Solver succeeded!")
            print(f"  Output shape: {result.output_grid.shape}")
            print(f"  Method: {result.method_used}")
            print(f"  Confidence: {result.confidence:.2f}")

            is_correct = np.array_equal(result.output_grid, expected_output)
            print(f"  Correct: {'YES' if is_correct else 'NO'}")

        except Exception as e:
            print(f"❌ Solver failed with error:")
            print(f"  {type(e).__name__}: {e}")
            traceback.print_exc()

    except Exception as e:
        print(f"❌ Failed to load task:")
        print(f"  {type(e).__name__}: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    test_single_training_task()
