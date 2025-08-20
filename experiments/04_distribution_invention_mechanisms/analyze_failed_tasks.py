"""Analyze failed ARC tasks to identify missing pattern types."""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def load_task(task_id: str) -> Dict:
    """Load an ARC task from file."""
    data_dir = Path("data/arc_agi_official/ARC-AGI/data/training")
    with open(data_dir / f"{task_id}.json", "r") as f:
        return json.load(f)


def analyze_transformation(inp: np.ndarray, out: np.ndarray) -> Dict[str, Any]:
    """Analyze the transformation between input and output."""
    analysis = {
        "size_change": inp.shape != out.shape,
        "input_shape": inp.shape,
        "output_shape": out.shape,
        "unique_input_colors": np.unique(inp).tolist(),
        "unique_output_colors": np.unique(out).tolist(),
        "new_colors": [],
        "removed_colors": [],
    }

    # Check for new/removed colors
    inp_colors = set(np.unique(inp))
    out_colors = set(np.unique(out))
    analysis["new_colors"] = list(out_colors - inp_colors)
    analysis["removed_colors"] = list(inp_colors - out_colors)

    # Check for specific patterns
    if inp.shape == out.shape:
        # Same size - likely in-place transformation
        analysis["type_hints"] = []

        # Check if it's a color remapping
        if len(analysis["new_colors"]) > 0 or len(analysis["removed_colors"]) > 0:
            analysis["type_hints"].append("color_remapping")

        # Check for movement/rotation
        if not np.array_equal(inp, out):
            # Check if pixels moved
            inp_nonzero = np.argwhere(inp != 0)
            out_nonzero = np.argwhere(out != 0)
            if len(inp_nonzero) == len(out_nonzero):
                analysis["type_hints"].append("movement_or_rotation")

            # Check for pattern filling
            if np.sum(out != 0) > np.sum(inp != 0):
                analysis["type_hints"].append("pattern_filling")
    else:
        # Different size - extraction, scaling, or generation
        analysis["type_hints"] = []
        if out.shape[0] < inp.shape[0] or out.shape[1] < inp.shape[1]:
            analysis["type_hints"].append("extraction_or_crop")
        elif out.shape[0] > inp.shape[0] or out.shape[1] > inp.shape[1]:
            analysis["type_hints"].append("expansion_or_tiling")

        # Check if output is a multiple of input
        if inp.shape[0] > 0 and out.shape[0] % inp.shape[0] == 0:
            analysis["type_hints"].append("possible_tiling")

    return analysis


def find_pattern_type(examples: List[Tuple[np.ndarray, np.ndarray]]) -> str:
    """Try to identify the pattern type from examples."""
    analyses = [analyze_transformation(inp, out) for inp, out in examples]

    # Check consistency across examples
    size_changes = [a["size_change"] for a in analyses]
    if all(size_changes):
        # Check if all outputs have same size
        out_shapes = [a["output_shape"] for a in analyses]
        if len(set(map(tuple, out_shapes))) == 1:
            return "fixed_size_output"
        else:
            return "variable_size_transformation"
    elif not any(size_changes):
        # All same size transformations
        hints = []
        for a in analyses:
            hints.extend(a.get("type_hints", []))

        if "color_remapping" in hints:
            return "color_based_transformation"
        elif "pattern_filling" in hints:
            return "pattern_completion"
        elif "movement_or_rotation" in hints:
            return "spatial_transformation"
        else:
            return "in_place_modification"
    else:
        return "mixed_size_transformation"


def visualize_task(task_id: str, task: Dict, max_examples: int = 3):
    """Visualize a task's examples."""
    train_examples = task["train"][:max_examples]

    fig, axes = plt.subplots(
        len(train_examples), 2, figsize=(8, 4 * len(train_examples))
    )
    if len(train_examples) == 1:
        axes = axes.reshape(1, -1)

    for i, example in enumerate(train_examples):
        inp = np.array(example["input"])
        out = np.array(example["output"])

        # Input
        axes[i, 0].imshow(inp, cmap="tab10", vmin=0, vmax=9)
        axes[i, 0].set_title(f"Input {i+1}")
        axes[i, 0].grid(True, alpha=0.3)
        axes[i, 0].set_xticks(range(inp.shape[1]))
        axes[i, 0].set_yticks(range(inp.shape[0]))

        # Output
        axes[i, 1].imshow(out, cmap="tab10", vmin=0, vmax=9)
        axes[i, 1].set_title(f"Output {i+1}")
        axes[i, 1].grid(True, alpha=0.3)
        axes[i, 1].set_xticks(range(out.shape[1]))
        axes[i, 1].set_yticks(range(out.shape[0]))

    plt.suptitle(f"Task {task_id}")
    plt.tight_layout()
    plt.savefig(f"failed_task_{task_id}.png", dpi=150, bbox_inches="tight")
    plt.close()


def main():
    """Analyze all failed tasks."""
    failed_tasks = ["1cf80156", "25ff71a9", "3aa6fb7a", "a416b8f3", "007bbfb7"]

    print("=" * 60)
    print("FAILED TASK ANALYSIS")
    print("=" * 60)

    pattern_types = {}

    for task_id in failed_tasks:
        print(f"\n--- Task {task_id} ---")

        try:
            task = load_task(task_id)
            examples = [
                (np.array(e["input"]), np.array(e["output"])) for e in task["train"]
            ]

            # Analyze first example in detail
            inp, out = examples[0]
            analysis = analyze_transformation(inp, out)

            print(f"Input shape: {analysis['input_shape']}")
            print(f"Output shape: {analysis['output_shape']}")
            print(f"Size change: {analysis['size_change']}")
            print(f"Input colors: {analysis['unique_input_colors']}")
            print(f"Output colors: {analysis['unique_output_colors']}")

            if analysis["new_colors"]:
                print(f"New colors added: {analysis['new_colors']}")
            if analysis["removed_colors"]:
                print(f"Colors removed: {analysis['removed_colors']}")

            if analysis.get("type_hints"):
                print(f"Type hints: {analysis['type_hints']}")

            # Identify pattern type
            pattern = find_pattern_type(examples)
            print(f"Pattern type: {pattern}")
            pattern_types[task_id] = pattern

            # Visualize the task
            visualize_task(task_id, task)

            # Detailed pattern analysis
            print("\nDetailed observations:")

            # Check for specific patterns we might be missing
            for i, (inp, out) in enumerate(examples[:3]):
                print(f"  Example {i+1}:")

                # Check for object counting
                inp_objects = np.sum(inp != 0)
                out_objects = np.sum(out != 0)
                if out_objects != inp_objects:
                    print(f"    - Object count change: {inp_objects} -> {out_objects}")

                # Check for specific structures
                if inp.shape != out.shape:
                    size_ratio = (
                        out.shape[0] / inp.shape[0],
                        out.shape[1] / inp.shape[1],
                    )
                    print(f"    - Size ratio: {size_ratio}")

                # Check for repetition patterns
                if out.shape[0] > inp.shape[0] and out.shape[0] % inp.shape[0] == 0:
                    repeat_factor = out.shape[0] // inp.shape[0]
                    print(f"    - Possible repetition factor: {repeat_factor}")

        except Exception as e:
            print(f"Error analyzing task: {e}")

    print("\n" + "=" * 60)
    print("SUMMARY OF MISSING PATTERNS")
    print("=" * 60)

    for task_id, pattern in pattern_types.items():
        print(f"{task_id}: {pattern}")

    print("\nPattern distribution:")
    from collections import Counter

    pattern_counts = Counter(pattern_types.values())
    for pattern, count in pattern_counts.most_common():
        print(f"  {pattern}: {count} tasks")

    print("\nRECOMMENDATIONS:")
    print("1. Need better handling of variable size outputs")
    print("2. Missing extraction/cropping patterns")
    print("3. Need object counting and replication")
    print("4. Missing complex spatial transformations")


if __name__ == "__main__":
    main()
