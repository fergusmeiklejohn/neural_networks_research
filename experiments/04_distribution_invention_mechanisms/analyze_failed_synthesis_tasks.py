#!/usr/bin/env python3
"""Analyze failed synthesis tasks to identify missing DSL patterns.

This script loads the synthesis evaluation results and analyzes each failed task
to identify common patterns that our DSL is missing.
"""

from utils.imports import setup_project_paths

setup_project_paths()

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def load_evaluation_results() -> Dict:
    """Load the synthesis evaluation results."""
    # Check parent directory first
    results_file = Path("../../synthesis_evaluation_results.json")
    if not results_file.exists():
        # Try current directory
        results_file = Path("synthesis_evaluation_results.json")

    if not results_file.exists():
        print(f"❌ Results file not found: {results_file}")
        return None

    with open(results_file, "r") as f:
        return json.load(f)


def load_arc_task(task_id: str) -> Dict:
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


def analyze_transformation(
    examples: List[Tuple[np.ndarray, np.ndarray]]
) -> Dict[str, any]:
    """Analyze the transformation pattern in examples."""
    analysis = {
        "size_changes": [],
        "color_mappings": set(),
        "object_counts": [],
        "spatial_patterns": [],
        "structural_changes": [],
    }

    for inp, out in examples:
        # Size changes
        size_change = (inp.shape, out.shape)
        analysis["size_changes"].append(size_change)

        # Color analysis
        in_colors = set(np.unique(inp))
        out_colors = set(np.unique(out))
        new_colors = out_colors - in_colors
        removed_colors = in_colors - out_colors

        if new_colors:
            analysis["color_mappings"].add(f"new_colors: {new_colors}")
        if removed_colors:
            analysis["color_mappings"].add(f"removed: {removed_colors}")

        # Object counting (connected components)
        from scipy import ndimage

        # Count objects for each non-zero color
        in_objects = {}
        out_objects = {}

        for color in in_colors:
            if color != 0:
                mask = (inp == color).astype(int)
                labeled, count = ndimage.label(mask)
                in_objects[color] = count

        for color in out_colors:
            if color != 0:
                mask = (out == color).astype(int)
                labeled, count = ndimage.label(mask)
                out_objects[color] = count

        analysis["object_counts"].append({"input": in_objects, "output": out_objects})

        # Detect structural patterns
        if inp.shape != out.shape:
            if out.shape[0] > inp.shape[0] or out.shape[1] > inp.shape[1]:
                analysis["structural_changes"].append("expansion")
            elif out.shape[0] < inp.shape[0] or out.shape[1] < inp.shape[1]:
                analysis["structural_changes"].append("cropping")

            # Check for tiling
            if out.shape[0] % inp.shape[0] == 0 and out.shape[1] % inp.shape[1] == 0:
                analysis["structural_changes"].append("possible_tiling")

        # Check for patterns
        if np.array_equal(inp, np.rot90(out)):
            analysis["spatial_patterns"].append("rotation")
        elif np.array_equal(inp, np.fliplr(out)):
            analysis["spatial_patterns"].append("horizontal_flip")
        elif np.array_equal(inp, np.flipud(out)):
            analysis["spatial_patterns"].append("vertical_flip")

    return analysis


def identify_missing_primitives(failed_tasks: List[Dict]) -> Dict[str, List]:
    """Identify common patterns in failed tasks that suggest missing primitives."""
    missing_patterns = defaultdict(list)

    for task_info in failed_tasks:
        task_id = task_info["task_id"]
        task = load_arc_task(task_id)

        if not task:
            continue

        # Convert to numpy arrays
        examples = [
            (np.array(ex["input"]), np.array(ex["output"])) for ex in task["train"]
        ]

        # Analyze the transformation
        analysis = analyze_transformation(examples)

        # Identify what primitives might be needed
        inp, out = examples[0]

        # Check for line drawing
        if len(analysis["structural_changes"]) == 0 and inp.shape == out.shape:
            # Same size transformation
            # Check if output has lines not in input
            diff = out - inp
            if np.any(diff != 0):
                # Check for horizontal/vertical lines
                for row in diff:
                    if np.sum(row != 0) > 2 and len(np.unique(row[row != 0])) == 1:
                        missing_patterns["line_drawing"].append(task_id)
                        break

                for col in diff.T:
                    if np.sum(col != 0) > 2 and len(np.unique(col[col != 0])) == 1:
                        missing_patterns["line_drawing"].append(task_id)
                        break

        # Check for counting operations
        if any(oc["input"] != oc["output"] for oc in analysis["object_counts"]):
            # Object count changes - might need counting
            missing_patterns["counting"].append(task_id)

        # Check for sorting/ordering
        unique_sizes = set()
        for inp, out in examples:
            # Get object sizes
            for color in np.unique(inp):
                if color != 0:
                    mask = inp == color
                    size = np.sum(mask)
                    unique_sizes.add(size)

        if len(unique_sizes) > 2:
            missing_patterns["sorting_by_size"].append(task_id)

        # Check for grid partitioning
        if "expansion" in analysis["structural_changes"]:
            # Check if output is divided into regions
            h, w = out.shape
            if h % 2 == 0 and w % 2 == 0:
                # Could be grid partitioning
                quadrants = [
                    out[: h // 2, : w // 2],
                    out[: h // 2, w // 2 :],
                    out[h // 2 :, : w // 2],
                    out[h // 2 :, w // 2 :],
                ]
                if any(not np.array_equal(q, quadrants[0]) for q in quadrants[1:]):
                    missing_patterns["grid_partition"].append(task_id)

        # Check for pattern propagation
        if "possible_tiling" not in analysis["structural_changes"]:
            # Check if there's a repeating pattern
            h, w = inp.shape
            if h >= 3 and w >= 3:
                # Check 3x3 patterns
                pattern = inp[:3, :3]
                repeats = True
                for i in range(0, h - 2, 3):
                    for j in range(0, w - 2, 3):
                        if not np.array_equal(inp[i : i + 3, j : j + 3], pattern):
                            repeats = False
                            break

                if repeats:
                    missing_patterns["pattern_propagation"].append(task_id)

        # Check for conditional fills
        # If colors change based on neighbors
        for inp, out in examples:
            # Only check if shapes match
            if inp.shape == out.shape:
                changed_pixels = inp != out
                if np.any(changed_pixels):
                    # Check if changes depend on neighbors
                    for i in range(1, inp.shape[0] - 1):
                        for j in range(1, inp.shape[1] - 1):
                            if changed_pixels[i, j]:
                                neighbors = [
                                    inp[i - 1, j],
                                    inp[i + 1, j],
                                    inp[i, j - 1],
                                    inp[i, j + 1],
                                ]
                                if len(set(neighbors)) > 1:
                                    missing_patterns["conditional_fill"].append(task_id)
                                    break

        # Check for edge/boundary detection
        if any("boundary" in str(cm) for cm in analysis["color_mappings"]):
            missing_patterns["edge_detection"].append(task_id)

    return dict(missing_patterns)


def summarize_findings(results: Dict, missing_patterns: Dict[str, List]) -> str:
    """Create a summary of findings."""
    summary = []
    summary.append("# Failed Task Analysis Summary\n")
    summary.append(f"Total tasks evaluated: {len(results['random_sample'])}\n")

    # Count solved vs failed
    solved = sum(1 for t in results["random_sample"] if t["solved"])
    failed = len(results["random_sample"]) - solved

    summary.append(
        f"Tasks solved: {solved} ({solved/len(results['random_sample'])*100:.1f}%)"
    )
    summary.append(
        f"Tasks failed: {failed} ({failed/len(results['random_sample'])*100:.1f}%)\n"
    )

    summary.append("## Missing DSL Primitives\n")
    summary.append("Based on analysis of failed tasks, we need:\n")

    # Sort by frequency
    pattern_counts = [
        (pattern, len(tasks)) for pattern, tasks in missing_patterns.items()
    ]
    pattern_counts.sort(key=lambda x: x[1], reverse=True)

    for pattern, count in pattern_counts:
        percentage = count / failed * 100 if failed > 0 else 0
        summary.append(f"\n### {pattern.replace('_', ' ').title()}")
        summary.append(f"- Found in {count} failed tasks ({percentage:.1f}%)")
        summary.append(f"- Example tasks: {', '.join(missing_patterns[pattern][:5])}")

    summary.append("\n## Recommendations\n")
    summary.append("Priority primitives to implement:\n")

    recommendations = [
        ("Line Drawing", "DrawLine, ConnectPoints with straight lines"),
        ("Counting", "CountObjects, CountByColor, CountBySize"),
        ("Grid Partition", "PartitionGrid, ExtractQuadrant, MergeRegions"),
        ("Conditional Fill", "FillIfNeighbor, PropagateColor, FloodFillConditional"),
        ("Edge Detection", "ExtractBoundaries, TraceBorder, GetPerimeter"),
        ("Pattern Propagation", "ExtendPattern, RepeatUntilEdge, FillWithPattern"),
        ("Sorting", "SortBySize, SortByPosition, ArrangeInOrder"),
    ]

    for i, (name, desc) in enumerate(recommendations, 1):
        summary.append(f"{i}. **{name}**: {desc}")

    return "\n".join(summary)


def main():
    """Main analysis function."""
    print("Loading evaluation results...")
    results = load_evaluation_results()

    if not results:
        return

    # Get failed tasks
    failed_tasks = [task for task in results["random_sample"] if not task["solved"]]

    print(f"Analyzing {len(failed_tasks)} failed tasks...")

    # Identify missing primitives
    missing_patterns = identify_missing_primitives(failed_tasks)

    # Create summary
    summary = summarize_findings(results, missing_patterns)

    # Save to markdown file
    output_file = Path("failed_tasks_patterns.md")
    with open(output_file, "w") as f:
        f.write(summary)

    print(f"\n✅ Analysis complete! Results saved to {output_file}")
    print("\nKey findings:")
    for pattern, tasks in missing_patterns.items():
        print(f"  - {pattern}: {len(tasks)} tasks")

    # Also save detailed analysis
    detailed_file = Path("failed_tasks_detailed.json")
    with open(detailed_file, "w") as f:
        json.dump(
            {
                "missing_patterns": missing_patterns,
                "failed_task_ids": [t["task_id"] for t in failed_tasks],
                "pattern_counts": {p: len(t) for p, t in missing_patterns.items()},
            },
            f,
            indent=2,
        )

    print(f"\nDetailed analysis saved to {detailed_file}")


if __name__ == "__main__":
    main()
