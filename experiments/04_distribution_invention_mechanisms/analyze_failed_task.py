#!/usr/bin/env python3
"""Analyze why a specific task failed pattern discovery."""

from utils.imports import setup_project_paths

setup_project_paths()

import json
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from automated_primitive_discovery_v2 import PrimitiveDiscovererV2


def visualize_task(task_id, example_idx=0):
    """Visualize input and output for a task."""
    data_dir = Path("data/arc_agi_official/ARC-AGI/data/training")
    task_file = data_dir / f"{task_id}.json"

    with open(task_file, "r") as f:
        task = json.load(f)

    inp = np.array(task["train"][example_idx]["input"])
    out = np.array(task["train"][example_idx]["output"])

    # Create color map
    cmap = mcolors.ListedColormap(
        [
            "black",
            "blue",
            "red",
            "green",
            "yellow",
            "grey",
            "pink",
            "orange",
            "teal",
            "brown",
        ]
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.imshow(inp, cmap=cmap, vmin=0, vmax=9)
    ax1.set_title(f"Input {example_idx+1}")
    ax1.grid(True, alpha=0.3)

    ax2.imshow(out, cmap=cmap, vmin=0, vmax=9)
    ax2.set_title(f"Output {example_idx+1}")
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f"Task {task_id}")
    plt.tight_layout()
    plt.savefig(f"task_{task_id}_ex{example_idx}.png")
    print(f"Saved visualization to task_{task_id}_ex{example_idx}.png")


def analyze_pattern_consistency(task_id):
    """Analyze why patterns aren't consistent across examples."""

    print(f"\n{'='*60}")
    print(f"Analyzing Task: {task_id}")
    print("=" * 60)

    data_dir = Path("data/arc_agi_official/ARC-AGI/data/training")
    task_file = data_dir / f"{task_id}.json"

    with open(task_file, "r") as f:
        task = json.load(f)

    train_examples = [
        (np.array(ex["input"]), np.array(ex["output"])) for ex in task["train"]
    ]

    print(f"Number of examples: {len(train_examples)}")

    # Create discoverer
    discoverer = PrimitiveDiscovererV2(verbose=False)

    # Analyze patterns for each example individually
    print("\nPatterns per example:")
    example_patterns = []

    for i, (inp, out) in enumerate(train_examples):
        patterns = discoverer._extract_patterns([(inp, out)])
        example_patterns.append(patterns)

        print(f"\nExample {i+1}:")
        print(f"  Input shape: {inp.shape}, Output shape: {out.shape}")
        print(f"  Patterns found: {len(patterns)}")

        for p in patterns:
            print(f"    - {p['type']}", end="")
            if p["type"] == "spatial" and p["data"]:
                print(f" ({p['data'].get('pattern', 'unknown')})", end="")
            print()

            # Show details for spatial patterns
            if p["type"] == "spatial" and p["data"]:
                if p["data"].get("pattern") == "cross":
                    details = p["data"].get("details", [])
                    if details:
                        print(f"      Cross centers: {[d['center'] for d in details]}")
                elif p["data"].get("pattern") == "line":
                    details = p["data"].get("details", [])
                    if details:
                        print(f"      Lines: {details[:3]}")  # Show first 3

    # Check pattern consistency
    print("\n" + "=" * 60)
    print("Pattern Consistency Analysis:")

    # Find common pattern types
    all_types = set()
    for patterns in example_patterns:
        for p in patterns:
            all_types.add(p["type"])

    print(f"\nPattern types across examples:")
    for ptype in all_types:
        count = sum(
            1
            for patterns in example_patterns
            if any(p["type"] == ptype for p in patterns)
        )
        consistency = count / len(train_examples) * 100
        print(f"  {ptype}: {count}/{len(train_examples)} examples ({consistency:.0f}%)")

    # Test full discovery
    print("\n" + "=" * 60)
    print("Full Discovery Attempt:")

    discovered = discoverer.discover_primitive(task_id, train_examples)

    if discovered:
        print("✅ Discovery succeeded!")
    else:
        print("❌ Discovery failed")

        # Try to understand why
        all_patterns = discoverer._extract_patterns(train_examples)

        if all_patterns:
            print(f"\nExtracted {len(all_patterns)} total patterns")

            # Test pattern matching
            for p in all_patterns[:3]:  # Test first 3 patterns
                matches = 0
                for inp, out in train_examples:
                    if discoverer._pattern_matches(p, inp, out):
                        matches += 1

                print(
                    f"  {p['type']}: matches {matches}/{len(train_examples)} examples"
                )
        else:
            print("No patterns extracted from combined examples")

    # Analyze transformations
    print("\n" + "=" * 60)
    print("Transformation Analysis:")

    for i, (inp, out) in enumerate(train_examples[:2]):  # First 2 examples
        print(f"\nExample {i+1}:")

        # Size change?
        if inp.shape != out.shape:
            print(f"  Size change: {inp.shape} -> {out.shape}")

        # Color changes?
        inp_colors = set(inp.flatten())
        out_colors = set(out.flatten())

        new_colors = out_colors - inp_colors
        lost_colors = inp_colors - out_colors

        if new_colors:
            print(f"  New colors: {new_colors}")
        if lost_colors:
            print(f"  Lost colors: {lost_colors}")

        # Pixel changes
        changed = np.sum(inp != out)
        total = inp.size
        print(f"  Pixels changed: {changed}/{total} ({changed/total*100:.1f}%)")


def main():
    """Analyze failed tasks."""

    # Tasks that failed discovery
    failed_tasks = [
        "00d62c1b",  # Line drawing - patterns found but inconsistent
        "05269061",  # Object manipulation - patterns found but inconsistent
        "05f2a901",  # Color mapping - patterns found but inconsistent
    ]

    for task_id in failed_tasks[:1]:  # Analyze first one in detail
        analyze_pattern_consistency(task_id)

        # Visualize first example
        try:
            visualize_task(task_id, 0)
        except Exception as e:
            print(f"Visualization failed: {e}")


if __name__ == "__main__":
    main()
