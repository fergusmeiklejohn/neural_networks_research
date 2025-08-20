"""Analyze ARC tasks to identify hierarchical pattern structures.

Hierarchical patterns are "patterns of patterns" - e.g.:
- A pattern repeated in a grid
- A transformation applied recursively
- Patterns that operate at multiple scales
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def detect_grid_structure(grid: np.ndarray) -> Dict[str, Any]:
    """Detect if a grid has repeating substructures."""
    h, w = grid.shape

    # Check for common tile sizes
    tile_sizes = [(2, 2), (3, 3), (4, 4), (2, 3), (3, 2)]

    for tile_h, tile_w in tile_sizes:
        if h % tile_h == 0 and w % tile_w == 0:
            # Check if grid is made of repeating tiles
            num_tiles_y = h // tile_h
            num_tiles_x = w // tile_w

            # Extract all tiles
            tiles = []
            for i in range(num_tiles_y):
                for j in range(num_tiles_x):
                    tile = grid[
                        i * tile_h : (i + 1) * tile_h, j * tile_w : (j + 1) * tile_w
                    ]
                    tiles.append(tile)

            # Check for patterns in tiles
            unique_tiles = []
            for tile in tiles:
                is_unique = True
                for unique in unique_tiles:
                    if np.array_equal(tile, unique):
                        is_unique = False
                        break
                if is_unique:
                    unique_tiles.append(tile)

            if len(unique_tiles) < len(tiles):
                return {
                    "has_grid": True,
                    "tile_size": (tile_h, tile_w),
                    "num_unique_tiles": len(unique_tiles),
                    "total_tiles": len(tiles),
                    "unique_tiles": unique_tiles,
                }

    return {"has_grid": False}


def detect_recursive_structure(inp: np.ndarray, out: np.ndarray) -> Dict[str, Any]:
    """Detect if transformation has recursive/fractal properties."""

    # Check if output is a scaled version with internal structure
    if out.shape[0] > inp.shape[0] and out.shape[1] > inp.shape[1]:
        scale_y = out.shape[0] // inp.shape[0]
        scale_x = out.shape[1] // inp.shape[1]

        if scale_y == scale_x:
            # Check if each input cell maps to a structured output region
            scale = scale_y

            # Analyze mapping patterns
            has_self_similarity = False

            for y in range(min(3, inp.shape[0])):  # Check first few cells
                for x in range(min(3, inp.shape[1])):
                    if inp[y, x] != 0:
                        # Get the output region
                        region = out[
                            y * scale : (y + 1) * scale, x * scale : (x + 1) * scale
                        ]

                        # Check if region has structure similar to input
                        if np.sum(region != 0) > 1:
                            # Check for self-similarity
                            region_pattern = (region != 0).astype(int)
                            input_pattern = (inp != 0).astype(int)

                            # Simplified check - does region have similar structure?
                            if region.shape == inp.shape:
                                if (
                                    np.sum(np.abs(region_pattern - input_pattern))
                                    < region.size // 2
                                ):
                                    has_self_similarity = True

            return {
                "has_recursion": has_self_similarity,
                "scale_factor": scale,
                "type": "self_similar"
                if has_self_similarity
                else "structured_expansion",
            }

    return {"has_recursion": False}


def detect_multi_scale_patterns(
    examples: List[Tuple[np.ndarray, np.ndarray]]
) -> Dict[str, Any]:
    """Detect patterns that operate at multiple scales."""

    patterns = {
        "local": [],  # Single cell or small neighborhood
        "regional": [],  # Groups of cells
        "global": [],  # Entire grid transformations
    }

    for inp, out in examples:
        # Local patterns - single cell transformations
        if inp.shape == out.shape:
            # Check if transformation is local (cell-by-cell)
            diff_count = np.sum(inp != out)
            if diff_count > 0 and diff_count < inp.size // 4:
                patterns["local"].append("cell_modification")

        # Regional patterns - operate on connected components
        from scipy import ndimage

        labeled_inp, num_inp = ndimage.label(inp != 0)
        labeled_out, num_out = ndimage.label(out != 0)

        if num_inp > 0 and num_out > 0:
            if num_inp != num_out:
                patterns["regional"].append("component_manipulation")

        # Global patterns - entire grid transformations
        if inp.shape != out.shape:
            patterns["global"].append("size_transformation")
        elif np.array_equal(out, np.rot90(inp)) or np.array_equal(out, np.fliplr(inp)):
            patterns["global"].append("geometric_transformation")

    # Determine if multi-scale
    active_scales = sum(1 for scale in patterns.values() if len(scale) > 0)

    return {
        "is_multi_scale": active_scales > 1,
        "active_scales": active_scales,
        "patterns": patterns,
    }


def analyze_hierarchical_candidates():
    """Find ARC tasks that likely need hierarchical pattern detection."""

    # Tasks that might have hierarchical structure
    candidate_tasks = [
        "68b16354",  # Failed in our v12 test
        "25ff71a9",  # Also failed - might need hierarchical
        # Add more complex tasks
        "3c9b0459",
        "6d0aefbc",
        "8d510a79",
        "a79310a0",
        "b1948b0a",
        "c8f0f002",
        "d631b094",
        "e8593010",
    ]

    hierarchical_tasks = []

    print("=" * 60)
    print("HIERARCHICAL PATTERN ANALYSIS")
    print("=" * 60)

    data_dir = Path("data/arc_agi_official/ARC-AGI/data/training")

    for task_id in candidate_tasks:
        try:
            with open(data_dir / f"{task_id}.json", "r") as f:
                task = json.load(f)

            examples = [
                (np.array(e["input"]), np.array(e["output"])) for e in task["train"]
            ]

            print(f"\n--- Task {task_id} ---")

            # Check for hierarchical structures
            hierarchical_indicators = []

            for i, (inp, out) in enumerate(examples[:2]):
                # Check grid structure
                inp_grid = detect_grid_structure(inp)
                out_grid = detect_grid_structure(out)

                if inp_grid["has_grid"] or out_grid["has_grid"]:
                    hierarchical_indicators.append("grid_structure")
                    print(f"  Example {i+1}: Grid structure detected")
                    if inp_grid["has_grid"]:
                        print(
                            f"    Input: {inp_grid['tile_size']} tiles, {inp_grid['num_unique_tiles']} unique"
                        )
                    if out_grid["has_grid"]:
                        print(
                            f"    Output: {out_grid['tile_size']} tiles, {out_grid['num_unique_tiles']} unique"
                        )

                # Check recursive structure
                recursive = detect_recursive_structure(inp, out)
                if recursive["has_recursion"]:
                    hierarchical_indicators.append("recursive")
                    print(
                        f"  Example {i+1}: Recursive structure (scale {recursive['scale_factor']})"
                    )

            # Check multi-scale
            multi_scale = detect_multi_scale_patterns(examples)
            if multi_scale["is_multi_scale"]:
                hierarchical_indicators.append("multi_scale")
                print(
                    f"  Multi-scale patterns detected ({multi_scale['active_scales']} scales)"
                )
                for scale, patterns in multi_scale["patterns"].items():
                    if patterns:
                        print(f"    {scale}: {patterns}")

            if hierarchical_indicators:
                hierarchical_tasks.append(task_id)
                print(f"  ✓ HIERARCHICAL CANDIDATE: {set(hierarchical_indicators)}")

        except FileNotFoundError:
            print(f"  Task {task_id} not found")
        except Exception as e:
            print(f"  Error analyzing {task_id}: {e}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Hierarchical candidates: {len(hierarchical_tasks)}/{len(candidate_tasks)}")
    print(f"Tasks: {hierarchical_tasks}")

    return hierarchical_tasks


def visualize_hierarchical_task(task_id: str):
    """Visualize a task to understand its hierarchical structure."""
    data_dir = Path("data/arc_agi_official/ARC-AGI/data/training")

    with open(data_dir / f"{task_id}.json", "r") as f:
        task = json.load(f)

    # Visualize first 2 examples
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    for i in range(min(2, len(task["train"]))):
        inp = np.array(task["train"][i]["input"])
        out = np.array(task["train"][i]["output"])

        axes[i, 0].imshow(inp, cmap="tab10", vmin=0, vmax=9)
        axes[i, 0].set_title(f"Input {i+1} ({inp.shape})")
        axes[i, 0].grid(True, alpha=0.3)

        axes[i, 1].imshow(out, cmap="tab10", vmin=0, vmax=9)
        axes[i, 1].set_title(f"Output {i+1} ({out.shape})")
        axes[i, 1].grid(True, alpha=0.3)

        # Add grid lines to see structure
        for ax in [axes[i, 0], axes[i, 1]]:
            ax.set_xticks(
                np.arange(-0.5, max(inp.shape[1], out.shape[1]), 1), minor=True
            )
            ax.set_yticks(
                np.arange(-0.5, max(inp.shape[0], out.shape[0]), 1), minor=True
            )
            ax.grid(which="minor", color="gray", linestyle="-", linewidth=0.5)

    plt.suptitle(f"Hierarchical Task: {task_id}")
    plt.tight_layout()
    plt.savefig(f"hierarchical_task_{task_id}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved visualization: hierarchical_task_{task_id}.png")


if __name__ == "__main__":
    # Find hierarchical candidates
    hierarchical_tasks = analyze_hierarchical_candidates()

    # Visualize the failed tasks to understand them better
    print("\nVisualizing failed tasks...")
    for task_id in ["68b16354", "25ff71a9"]:
        try:
            visualize_hierarchical_task(task_id)
        except Exception as e:
            print(f"Could not visualize {task_id}: {e}")
