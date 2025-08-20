#!/usr/bin/env python3
"""Debug why synthesis is failing on specific tasks."""

from utils.imports import setup_project_paths

setup_project_paths()

import json
from pathlib import Path

import numpy as np
from enhanced_compositional_dsl import EnhancedCompositionalDSL


def analyze_task(task_id: str):
    """Analyze a specific task in detail."""
    # Load task
    data_dir = Path("data/arc_agi_official/ARC-AGI/data/evaluation")
    task_file = data_dir / f"{task_id}.json"

    if not task_file.exists():
        data_dir = Path("data/arc_agi_official/ARC-AGI/data/training")
        task_file = data_dir / f"{task_id}.json"

    with open(task_file, "r") as f:
        task = json.load(f)

    print(f"\n{'='*60}")
    print(f"Task: {task_id}")
    print("=" * 60)

    # Analyze examples
    for i, example in enumerate(task["train"]):
        inp = np.array(example["input"])
        out = np.array(example["output"])

        print(f"\nExample {i+1}:")
        print(f"  Input shape: {inp.shape}")
        print(f"  Output shape: {out.shape}")
        print(f"  Input colors: {sorted(np.unique(inp))}")
        print(f"  Output colors: {sorted(np.unique(out))}")

        # Visual comparison
        print("\n  Input:")
        print_grid(inp, indent=4)
        print("\n  Output:")
        print_grid(out, indent=4)

        # Analyze transformation
        if inp.shape == out.shape:
            # Check what changed
            diff_mask = inp != out
            changed = np.sum(diff_mask)
            print(
                f"\n  Changed pixels: {changed}/{inp.size} ({changed/inp.size*100:.1f}%)"
            )

            # Analyze changes
            changes = {}
            for i in range(inp.shape[0]):
                for j in range(inp.shape[1]):
                    if diff_mask[i, j]:
                        key = f"{inp[i,j]}->{out[i,j]}"
                        changes[key] = changes.get(key, 0) + 1

            if changes:
                print("  Color changes:")
                for change, count in sorted(changes.items()):
                    print(f"    {change}: {count} pixels")


def print_grid(grid, indent=0):
    """Print a grid with colors."""
    color_map = {
        0: ".",  # Background
        1: "1",
        2: "2",
        3: "3",
        4: "4",
        5: "5",
        6: "6",
        7: "7",
        8: "8",
        9: "9",
    }

    for row in grid:
        print(" " * indent + " ".join(color_map.get(c, str(c)) for c in row))


def test_specific_primitives(task_id: str):
    """Test specific primitives on a task."""
    # Load task
    data_dir = Path("data/arc_agi_official/ARC-AGI/data/evaluation")
    task_file = data_dir / f"{task_id}.json"

    if not task_file.exists():
        data_dir = Path("data/arc_agi_official/ARC-AGI/data/training")
        task_file = data_dir / f"{task_id}.json"

    with open(task_file, "r") as f:
        task = json.load(f)

    # Get examples
    train_examples = [
        (np.array(ex["input"]), np.array(ex["output"])) for ex in task["train"]
    ]

    # Create DSL
    EnhancedCompositionalDSL()

    # Test new primitives
    print(f"\nTesting primitives on {task_id}:")

    from missing_dsl_primitives import (
        ConditionalFill,
        CountObjects,
        DrawLine,
        PropagatePattern,
        SortBySize,
    )

    test_primitives = [
        ConditionalFill("adjacent_to", 3),
        ConditionalFill("between_colors", 7),
        DrawLine(direction="horizontal"),
        CountObjects(action="mark_nth"),
        SortBySize("horizontal"),
        PropagatePattern("all"),
    ]

    from compositional_dsl import ExecutionContext

    for primitive in test_primitives:
        print(f"\n  Testing {primitive}:")

        for i, (inp, expected) in enumerate(train_examples[:1]):
            context = ExecutionContext(input_grid=inp.copy(), current_grid=inp.copy())

            try:
                result_context = primitive.execute(context)
                result = result_context.current_grid

                # Check accuracy
                accuracy = np.mean(result == expected)
                print(f"    Example {i+1}: {accuracy:.1%} accuracy")

                if accuracy > 0.5:
                    print("      Promising primitive!")

            except Exception as e:
                print(f"    Example {i+1}: Error - {e}")


def main():
    """Main debugging function."""
    # Test on first failed task
    failed_task = "ae3edfdc"

    print("Analyzing failed synthesis task")
    analyze_task(failed_task)

    print("\n" + "=" * 60)
    print("Testing specific primitives")
    print("=" * 60)
    test_specific_primitives(failed_task)

    # Try manual solutions
    print("\n" + "=" * 60)
    print("Attempting manual synthesis")
    print("=" * 60)

    # Based on the analysis, suggest what primitive might work
    print("\nBased on analysis, this task seems to involve:")
    print("- Conditional color changes based on neighbors")
    print("- Possibly connecting or grouping objects")
    print("- May need a combination of primitives")


if __name__ == "__main__":
    main()
