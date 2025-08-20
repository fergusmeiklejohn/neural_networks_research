#!/usr/bin/env python3
"""Analyze baseline ARC evaluation results to identify failure patterns."""

from utils.imports import setup_project_paths

setup_project_paths()

import json
from pathlib import Path
from typing import Dict, List

import numpy as np


def load_results() -> Dict:
    """Load the baseline evaluation results."""
    results_file = Path(__file__).parent / "outputs" / "real_arc_baseline_results.json"
    with open(results_file) as f:
        return json.load(f)


def load_task_data(task_id: str) -> Dict:
    """Load the actual task data."""
    base_path = Path(__file__).parent.parent.parent
    task_path = (
        base_path
        / "data"
        / "arc_agi_official"
        / "ARC-AGI"
        / "data"
        / "training"
        / f"{task_id}.json"
    )

    if not task_path.exists():
        # Try evaluation set
        task_path = (
            base_path
            / "data"
            / "arc_agi_official"
            / "ARC-AGI"
            / "data"
            / "evaluation"
            / f"{task_id}.json"
        )

    if task_path.exists():
        with open(task_path) as f:
            return json.load(f)
    return None


def analyze_transformation(task_data: Dict) -> str:
    """Try to identify the type of transformation in a task."""
    if not task_data:
        return "unknown"

    train_examples = task_data.get("train", [])
    if not train_examples:
        return "no_examples"

    # Check for simple patterns
    for ex in train_examples:
        input_grid = np.array(ex["input"])
        output_grid = np.array(ex["output"])

        # Check for scaling
        if (
            output_grid.shape[0] == input_grid.shape[0] * 2
            and output_grid.shape[1] == input_grid.shape[1] * 2
        ):
            return "2x_scaling"

        # Check for flip/mirror
        if np.array_equal(output_grid, np.flip(input_grid, axis=1)):
            return "horizontal_flip"
        if np.array_equal(output_grid, np.flip(input_grid, axis=0)):
            return "vertical_flip"

        # Check for rotation
        if np.array_equal(output_grid, np.rot90(input_grid)):
            return "rotation_90"

        # Check for color mapping (simple)
        if input_grid.shape == output_grid.shape:
            unique_in = set(input_grid.flatten())
            unique_out = set(output_grid.flatten())
            if len(unique_in) == len(unique_out) and unique_in != unique_out:
                return "color_mapping"

        # Check for size change
        if input_grid.shape != output_grid.shape:
            return "size_change"

    return "complex_pattern"


def analyze_grid_properties(grid: np.ndarray) -> Dict:
    """Analyze properties of a grid."""
    return {
        "shape": grid.shape,
        "unique_colors": len(np.unique(grid)),
        "has_symmetry": check_symmetry(grid),
        "has_patterns": check_patterns(grid),
        "object_count": count_objects(grid),
    }


def check_symmetry(grid: np.ndarray) -> Dict:
    """Check for various types of symmetry."""
    return {
        "horizontal": np.array_equal(grid, np.flip(grid, axis=1)),
        "vertical": np.array_equal(grid, np.flip(grid, axis=0)),
        "diagonal": np.array_equal(grid, grid.T),
    }


def check_patterns(grid: np.ndarray) -> List[str]:
    """Check for common patterns."""
    patterns = []

    # Check for repeating rows
    if grid.shape[0] > 1:
        for i in range(grid.shape[0] - 1):
            if np.array_equal(grid[i], grid[i + 1]):
                patterns.append("repeating_rows")
                break

    # Check for repeating columns
    if grid.shape[1] > 1:
        for i in range(grid.shape[1] - 1):
            if np.array_equal(grid[:, i], grid[:, i + 1]):
                patterns.append("repeating_columns")
                break

    # Check for diagonal patterns
    if grid.shape[0] == grid.shape[1]:
        diagonal = np.diagonal(grid)
        if len(np.unique(diagonal)) == 1:
            patterns.append("uniform_diagonal")

    return patterns


def count_objects(grid: np.ndarray) -> int:
    """Count connected components (objects) in the grid."""
    # Simple connected component counting for non-zero values
    visited = np.zeros_like(grid, dtype=bool)
    count = 0

    def dfs(i, j, color):
        if i < 0 or i >= grid.shape[0] or j < 0 or j >= grid.shape[1]:
            return
        if visited[i, j] or grid[i, j] != color or grid[i, j] == 0:
            return
        visited[i, j] = True
        dfs(i + 1, j, color)
        dfs(i - 1, j, color)
        dfs(i, j + 1, color)
        dfs(i, j - 1, color)

    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if not visited[i, j] and grid[i, j] != 0:
                dfs(i, j, grid[i, j])
                count += 1

    return count


def main():
    """Analyze baseline results."""
    print("=" * 70)
    print("ARC BASELINE ANALYSIS")
    print("=" * 70)

    # Load results
    results = load_results()
    task_results = results["task_results"]

    # Categorize successes and failures
    successes = []
    failures = []

    for task in task_results:
        task_id = task["task_id"]
        if task.get("explicit_only_accuracy", 0) > 0:
            successes.append(task_id)
        else:
            failures.append(task_id)

    print(f"\nSuccessful tasks: {len(successes)}/{len(task_results)}")
    print(f"Failed tasks: {len(failures)}/{len(task_results)}")

    # Analyze successful tasks
    print("\n" + "=" * 70)
    print("SUCCESSFUL TASKS ANALYSIS")
    print("=" * 70)

    for task_id in successes:
        print(f"\nTask: {task_id}")
        task_data = load_task_data(task_id)
        if task_data:
            transformation = analyze_transformation(task_data)
            print(f"  Transformation type: {transformation}")

            # Analyze first example
            if task_data["train"]:
                ex = task_data["train"][0]
                input_grid = np.array(ex["input"])
                output_grid = np.array(ex["output"])

                print(f"  Input shape: {input_grid.shape}")
                print(f"  Output shape: {output_grid.shape}")

                input_props = analyze_grid_properties(input_grid)
                output_props = analyze_grid_properties(output_grid)

                print(f"  Input colors: {input_props['unique_colors']}")
                print(f"  Output colors: {output_props['unique_colors']}")

    # Sample analysis of failed tasks
    print("\n" + "=" * 70)
    print("FAILED TASKS ANALYSIS (Sample)")
    print("=" * 70)

    transformation_types = {}

    # Analyze a sample of failed tasks
    for task_id in failures[:20]:  # Analyze first 20 failures
        task_data = load_task_data(task_id)
        if task_data:
            transformation = analyze_transformation(task_data)
            transformation_types[transformation] = (
                transformation_types.get(transformation, 0) + 1
            )

    print("\nTransformation types in failed tasks:")
    for trans_type, count in sorted(
        transformation_types.items(), key=lambda x: x[1], reverse=True
    ):
        print(f"  {trans_type}: {count}")

    # Identify patterns we need to handle
    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)

    print("\n1. Current capabilities:")
    print("   - Horizontal flip (mirror)")
    print("   - 2x scaling")

    print("\n2. Missing capabilities (based on failures):")
    print("   - Object detection and manipulation")
    print("   - Complex pattern recognition")
    print("   - Conditional transformations")
    print("   - Multi-step reasoning")
    print("   - Counting and arithmetic operations")

    print("\n3. Priority improvements needed:")
    print("   - Connected component analysis for object detection")
    print("   - Pattern sequence detection")
    print("   - Spatial relationship understanding")
    print("   - Rule composition and chaining")

    # Detailed analysis of a few failed tasks
    print("\n" + "=" * 70)
    print("DETAILED FAILURE EXAMPLES")
    print("=" * 70)

    for task_id in failures[:3]:  # Analyze first 3 failures in detail
        print(f"\nTask: {task_id}")
        task_data = load_task_data(task_id)
        if task_data and task_data["train"]:
            ex = task_data["train"][0]
            input_grid = np.array(ex["input"])
            output_grid = np.array(ex["output"])

            print(f"  Input: {input_grid.shape}, Output: {output_grid.shape}")

            # Try to understand the transformation
            if input_grid.shape == output_grid.shape:
                # Same size - likely object manipulation or color change
                diff_count = np.sum(input_grid != output_grid)
                total_cells = input_grid.size
                print(
                    f"  Changed cells: {diff_count}/{total_cells} ({diff_count/total_cells*100:.1f}%)"
                )

                # Check if it's selective modification
                if diff_count < total_cells * 0.5:
                    print("  Likely selective modification (objects or patterns)")
            else:
                # Different size - analyze the relationship
                size_ratio_h = output_grid.shape[0] / input_grid.shape[0]
                size_ratio_w = output_grid.shape[1] / input_grid.shape[1]
                print(f"  Size ratio: {size_ratio_h:.2f}x{size_ratio_w:.2f}")

                if size_ratio_h != size_ratio_w:
                    print("  Non-uniform scaling or cropping/padding")

            # Check object counts
            input_objects = count_objects(input_grid)
            output_objects = count_objects(output_grid)
            print(f"  Objects: {input_objects} -> {output_objects}")


if __name__ == "__main__":
    main()
