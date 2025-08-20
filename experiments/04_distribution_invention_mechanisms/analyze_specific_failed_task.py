#!/usr/bin/env python3
"""Analyze specific failed tasks to understand what DSL primitives are missing."""

from utils.imports import setup_project_paths

setup_project_paths()

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage


def load_arc_task(task_id: str):
    """Load an ARC task by ID."""
    # Try training directory first
    data_dir = Path("data/arc_agi_official/ARC-AGI/data/training")
    task_file = data_dir / f"{task_id}.json"

    if not task_file.exists():
        # Try evaluation directory
        data_dir = Path("data/arc_agi_official/ARC-AGI/data/evaluation")
        task_file = data_dir / f"{task_id}.json"

    if not task_file.exists():
        return None

    with open(task_file, "r") as f:
        return json.load(f)


def analyze_task(task_id: str):
    """Analyze a specific task to understand its transformation pattern."""
    task = load_arc_task(task_id)
    if not task:
        print(f"Task {task_id} not found")
        return

    print(f"\n{'='*60}")
    print(f"Task: {task_id}")
    print("=" * 60)

    # Analyze each example
    for i, example in enumerate(task["train"]):
        inp = np.array(example["input"])
        out = np.array(example["output"])

        print(f"\nExample {i+1}:")
        print(f"  Input shape: {inp.shape}")
        print(f"  Output shape: {out.shape}")
        print(f"  Input colors: {sorted(np.unique(inp))}")
        print(f"  Output colors: {sorted(np.unique(out))}")

        # Size change analysis
        if inp.shape != out.shape:
            h_ratio = out.shape[0] / inp.shape[0]
            w_ratio = out.shape[1] / inp.shape[1]
            print(f"  Size change: {h_ratio:.2f}x height, {w_ratio:.2f}x width")

            # Check for tiling
            if h_ratio == int(h_ratio) and w_ratio == int(w_ratio):
                print(f"  Possible tiling: {int(h_ratio)}x{int(w_ratio)}")
        else:
            print("  Same size transformation")

            # Check what changed
            diff = inp != out
            changed_pixels = np.sum(diff)
            total_pixels = inp.size
            print(
                f"  Changed pixels: {changed_pixels}/{total_pixels} ({changed_pixels/total_pixels*100:.1f}%)"
            )

            # Analyze the changes
            if changed_pixels > 0:
                # What colors changed?
                for color in np.unique(inp):
                    mask = inp == color
                    changed = np.sum(mask & diff)
                    if changed > 0:
                        new_colors = out[mask & diff]
                        unique_new = np.unique(new_colors)
                        print(f"    Color {color} -> {list(unique_new)}")

        # Object analysis
        print("  Object analysis:")
        for color in np.unique(inp):
            if color != 0:  # Non-background
                mask = (inp == color).astype(int)
                labeled, num_objects = ndimage.label(mask)
                if num_objects > 0:
                    print(f"    Color {color}: {num_objects} object(s)")

                    # Get object sizes
                    sizes = []
                    for obj_id in range(1, num_objects + 1):
                        size = np.sum(labeled == obj_id)
                        sizes.append(size)
                    if len(sizes) > 1:
                        print(f"      Sizes: {sizes}")

        # Pattern detection
        print("  Pattern analysis:")

        # Check for lines
        has_horizontal_line = False
        has_vertical_line = False

        for row in out:
            non_zero = row[row != 0]
            if len(non_zero) > 3 and len(np.unique(non_zero)) == 1:
                has_horizontal_line = True

        for col in out.T:
            non_zero = col[col != 0]
            if len(non_zero) > 3 and len(np.unique(non_zero)) == 1:
                has_vertical_line = True

        if has_horizontal_line:
            print("    Has horizontal line(s)")
        if has_vertical_line:
            print("    Has vertical line(s)")

        # Check for symmetry
        if np.array_equal(out, np.fliplr(out)):
            print("    Horizontally symmetric")
        if np.array_equal(out, np.flipud(out)):
            print("    Vertically symmetric")

        # Check for rotations
        if inp.shape == out.shape:
            if np.array_equal(inp, np.rot90(out, 1)):
                print("    Output is input rotated 90°")
            elif np.array_equal(inp, np.rot90(out, 2)):
                print("    Output is input rotated 180°")
            elif np.array_equal(inp, np.rot90(out, 3)):
                print("    Output is input rotated 270°")

    # Try to identify the transformation rule
    print("\n" + "=" * 60)
    print("TRANSFORMATION HYPOTHESIS:")

    # Collect all examples
    examples = [(np.array(ex["input"]), np.array(ex["output"])) for ex in task["train"]]

    # Check common patterns
    all_same_size = all(inp.shape == out.shape for inp, out in examples)
    all_different_size = all(inp.shape != out.shape for inp, out in examples)

    if all_same_size:
        print("- Same-size transformation")

        # Check if it's a color mapping
        color_map_consistent = True
        color_map = {}
        for inp, out in examples:
            for i in range(inp.shape[0]):
                for j in range(inp.shape[1]):
                    in_color = inp[i, j]
                    out_color = out[i, j]
                    if in_color in color_map:
                        if color_map[in_color] != out_color:
                            color_map_consistent = False
                            break
                    else:
                        color_map[in_color] = out_color

        if color_map_consistent and color_map:
            print(f"- Simple color mapping: {color_map}")

    elif all_different_size:
        print("- Size-changing transformation")

        # Check if it's tiling
        ratios = []
        for inp, out in examples:
            h_ratio = out.shape[0] / inp.shape[0]
            w_ratio = out.shape[1] / inp.shape[1]
            ratios.append((h_ratio, w_ratio))

        if len(set(ratios)) == 1:
            h, w = ratios[0]
            if h == int(h) and w == int(w):
                print(f"- Consistent tiling: {int(h)}x{int(w)}")

    # Look for object-based transformations
    object_counts_change = False
    for inp, out in examples:
        in_objects = 0
        out_objects = 0
        for color in np.unique(inp):
            if color != 0:
                mask = (inp == color).astype(int)
                _, count = ndimage.label(mask)
                in_objects += count

        for color in np.unique(out):
            if color != 0:
                mask = (out == color).astype(int)
                _, count = ndimage.label(mask)
                out_objects += count

        if in_objects != out_objects:
            object_counts_change = True

    if object_counts_change:
        print("- Object count changes (needs object manipulation)")

    print("\nSuggested DSL primitives needed:")
    suggestions = []

    if all_different_size:
        suggestions.append("- Tiling/Resizing operations")

    if has_horizontal_line or has_vertical_line:
        suggestions.append("- Line drawing primitives")

    if object_counts_change:
        suggestions.append("- Object extraction and manipulation")

    # Check if we need conditional operations
    needs_conditional = False
    for inp, out in examples:
        if inp.shape == out.shape:
            # Check if changes depend on neighbors
            for i in range(1, inp.shape[0] - 1):
                for j in range(1, inp.shape[1] - 1):
                    if inp[i, j] != out[i, j]:
                        neighbors = [
                            inp[i - 1, j],
                            inp[i + 1, j],
                            inp[i, j - 1],
                            inp[i, j + 1],
                        ]
                        if out[i, j] in neighbors:
                            needs_conditional = True
                            break

    if needs_conditional:
        suggestions.append("- Conditional operations based on neighbors")

    if not suggestions:
        suggestions.append("- Complex pattern that needs analysis")

    for suggestion in suggestions:
        print(suggestion)


def visualize_task(task_id: str):
    """Visualize a task's examples."""
    task = load_arc_task(task_id)
    if not task:
        print(f"Task {task_id} not found")
        return

    n_examples = len(task["train"])
    fig, axes = plt.subplots(n_examples, 2, figsize=(8, 4 * n_examples))

    if n_examples == 1:
        axes = axes.reshape(1, -1)

    for i, example in enumerate(task["train"]):
        inp = np.array(example["input"])
        out = np.array(example["output"])

        # Input
        axes[i, 0].imshow(inp, cmap="tab10", vmin=0, vmax=9)
        axes[i, 0].set_title(f"Input {i+1}")
        axes[i, 0].grid(True, alpha=0.3)

        # Output
        axes[i, 1].imshow(out, cmap="tab10", vmin=0, vmax=9)
        axes[i, 1].set_title(f"Output {i+1}")
        axes[i, 1].grid(True, alpha=0.3)

    plt.suptitle(f"Task {task_id}")
    plt.tight_layout()
    plt.savefig(f"task_{task_id}_visualization.png", dpi=150)
    print(f"Saved visualization to task_{task_id}_visualization.png")


def main():
    """Analyze specific failed tasks."""
    # Load the failed task IDs
    detailed_file = Path("failed_tasks_detailed.json")
    if detailed_file.exists():
        with open(detailed_file, "r") as f:
            data = json.load(f)
            failed_ids = data["failed_task_ids"]
    else:
        # Use some example failed tasks
        failed_ids = [
            "ae3edfdc",  # First failed task
            "d406998b",  # Another failed
            "8403a5d5",  # Has line drawing
            "53b68214",  # Grid partition
        ]

    print("Analyzing failed tasks to understand missing DSL primitives...")
    print(f"Total failed tasks: {len(failed_ids)}")

    # Analyze first few tasks in detail
    for task_id in failed_ids[:5]:
        analyze_task(task_id)

    # Optionally visualize
    visualize_first = input("\nVisualize first task? (y/n): ")
    if visualize_first.lower() == "y":
        visualize_task(failed_ids[0])


if __name__ == "__main__":
    main()
