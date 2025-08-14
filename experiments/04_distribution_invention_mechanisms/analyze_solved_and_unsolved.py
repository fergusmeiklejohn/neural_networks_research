#!/usr/bin/env python3
"""Analyze solved and unsolved tasks to identify needed patterns."""

from utils.imports import setup_project_paths

setup_project_paths()

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_evaluation_results():
    """Load the full evaluation results."""
    results_file = Path("outputs/full_evaluation_20250814_083530.json")

    with open(results_file, "r") as f:
        data = json.load(f)

    # Find the Hybrid solver results
    for solver_data in data:
        if solver_data["solver_name"] == "Hybrid Fixed+Imagination":
            return solver_data["results"]

    return None


def analyze_task(task_id: str, data_dir: Path):
    """Analyze a specific task to understand its pattern."""
    task_file = data_dir / f"{task_id}.json"

    if not task_file.exists():
        return None

    with open(task_file, "r") as f:
        task = json.load(f)

    analysis = {
        "id": task_id,
        "train_count": len(task["train"]),
        "test_count": len(task["test"]),
    }

    # Analyze first training example
    if task["train"]:
        input_grid = np.array(task["train"][0]["input"])
        output_grid = np.array(task["train"][0]["output"])

        analysis["input_shape"] = input_grid.shape
        analysis["output_shape"] = output_grid.shape
        analysis["size_change"] = input_grid.shape != output_grid.shape

        # Check for scaling
        if analysis["size_change"]:
            h_scale = output_grid.shape[0] / input_grid.shape[0]
            w_scale = output_grid.shape[1] / input_grid.shape[1]
            analysis["h_scale"] = h_scale
            analysis["w_scale"] = w_scale
            analysis["is_uniform_scale"] = h_scale == w_scale
            analysis["is_integer_scale"] = h_scale == int(h_scale) and w_scale == int(
                w_scale
            )

        # Count unique values
        analysis["input_colors"] = len(np.unique(input_grid))
        analysis["output_colors"] = len(np.unique(output_grid))
        analysis["color_change"] = set(np.unique(output_grid)) != set(
            np.unique(input_grid)
        )

        # Check for patterns
        analysis["has_symmetry"] = check_symmetry(output_grid)
        analysis["has_repetition"] = check_repetition(output_grid)
        analysis["has_objects"] = analysis["input_colors"] > 2  # Simple heuristic

    return analysis


def check_symmetry(grid):
    """Check if grid has symmetry."""
    # Horizontal symmetry
    h_sym = np.array_equal(grid, np.flip(grid, axis=0))
    # Vertical symmetry
    v_sym = np.array_equal(grid, np.flip(grid, axis=1))
    # Diagonal symmetry
    if grid.shape[0] == grid.shape[1]:
        d_sym = np.array_equal(grid, grid.T)
    else:
        d_sym = False

    return h_sym or v_sym or d_sym


def check_repetition(grid):
    """Check if grid has repetitive patterns."""
    h, w = grid.shape

    # Check for 2x2 repetition
    if h % 2 == 0 and w % 2 == 0:
        h2, w2 = h // 2, w // 2
        quadrants = [grid[:h2, :w2], grid[:h2, w2:], grid[h2:, :w2], grid[h2:, w2:]]
        if all(np.array_equal(quadrants[0], q) for q in quadrants[1:]):
            return True

    # Check for row repetition
    if h > 1:
        first_row = grid[0]
        if all(np.array_equal(first_row, grid[i]) for i in range(1, h)):
            return True

    return False


def main():
    """Analyze solved and unsolved tasks."""

    data_dir = Path("data/arc_agi_official/ARC-AGI/data/evaluation")

    # Load results
    results = load_evaluation_results()
    if not results:
        print("❌ Could not load evaluation results")
        return

    # Separate solved and unsolved
    solved_tasks = []
    unsolved_tasks = []

    for result in results:
        if result["correct"]:
            solved_tasks.append(result["task_id"])
        else:
            unsolved_tasks.append(result["task_id"])

    print(f"📊 Analysis of {len(results)} tasks")
    print(f"  ✅ Solved: {len(solved_tasks)}")
    print(f"  ❌ Unsolved: {len(unsolved_tasks)}")
    print()

    # Analyze solved tasks
    print("=" * 60)
    print("SOLVED TASKS")
    print("=" * 60)

    solved_patterns = defaultdict(int)

    for task_id in solved_tasks:
        analysis = analyze_task(task_id, data_dir)
        if analysis:
            print(f"\n{task_id}:")
            print(
                f"  Input: {analysis['input_shape']} → Output: {analysis['output_shape']}"
            )

            if analysis["size_change"]:
                print(f"  Scale: {analysis['h_scale']:.1f}x{analysis['w_scale']:.1f}")
                if analysis["is_integer_scale"]:
                    solved_patterns["integer_scaling"] += 1
                if analysis["is_uniform_scale"]:
                    solved_patterns["uniform_scaling"] += 1

            if analysis["has_symmetry"]:
                solved_patterns["symmetry"] += 1
            if analysis["has_repetition"]:
                solved_patterns["repetition"] += 1
            if analysis["color_change"]:
                solved_patterns["color_change"] += 1

    print("\n📈 Patterns in solved tasks:")
    for pattern, count in sorted(solved_patterns.items()):
        print(
            f"  {pattern}: {count}/{len(solved_tasks)} ({count/len(solved_tasks)*100:.0f}%)"
        )

    # Sample analysis of unsolved tasks
    print("\n" + "=" * 60)
    print("SAMPLE OF UNSOLVED TASKS (first 20)")
    print("=" * 60)

    unsolved_patterns = defaultdict(int)

    for task_id in unsolved_tasks[:20]:
        analysis = analyze_task(task_id, data_dir)
        if analysis:
            if analysis["size_change"]:
                if analysis["is_integer_scale"]:
                    unsolved_patterns["integer_scaling"] += 1
                else:
                    unsolved_patterns["non_integer_scaling"] += 1
            else:
                unsolved_patterns["no_size_change"] += 1

            if analysis["has_symmetry"]:
                unsolved_patterns["symmetry"] += 1
            if analysis["has_repetition"]:
                unsolved_patterns["repetition"] += 1
            if analysis["color_change"]:
                unsolved_patterns["color_change"] += 1
            if analysis["has_objects"]:
                unsolved_patterns["multiple_objects"] += 1

    print("\n📊 Patterns in unsolved sample:")
    for pattern, count in sorted(
        unsolved_patterns.items(), key=lambda x: x[1], reverse=True
    ):
        print(f"  {pattern}: {count}/20 ({count/20*100:.0f}%)")

    # Identify missing capabilities
    print("\n" + "=" * 60)
    print("MISSING CAPABILITIES")
    print("=" * 60)

    print("\n🔴 Critical gaps based on unsolved patterns:")
    print("  1. Non-size-change transformations (color mapping, rotation, etc.)")
    print("  2. Object detection and manipulation")
    print("  3. Complex symmetry operations")
    print("  4. Non-integer scaling patterns")
    print("  5. Pattern completion and extrapolation")

    print("\n🟡 Recommended primitives to implement:")
    print("  1. ColorMapper - map colors based on rules")
    print("  2. ObjectExtractor - identify and manipulate discrete objects")
    print("  3. SymmetryApplier - apply various symmetry operations")
    print("  4. PatternCompleter - complete partial patterns")
    print("  5. RotationReflection - geometric transformations")
    print("  6. ConditionalTransform - if-then-else logic")
    print("  7. CountingPrimitive - count objects and apply rules")


if __name__ == "__main__":
    main()
