#!/usr/bin/env python3
"""Detailed analysis of V7 failures to identify missing patterns."""

from utils.imports import setup_project_paths

setup_project_paths()

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from enhanced_arc_solver_v7 import EnhancedARCSolverV7


def load_arc_task(task_path: Path) -> Dict:
    """Load an ARC task from JSON file."""
    with open(task_path) as f:
        return json.load(f)


def analyze_task_pattern(examples: List[Tuple[np.ndarray, np.ndarray]]) -> Dict:
    """Analyze the pattern in a task's examples."""
    analysis = {
        "size_change": False,
        "color_mapping": {},
        "spatial_pattern": None,
        "rule_type": None,
        "complexity": None,
    }

    if not examples:
        return analysis

    # Check size changes
    for inp, out in examples:
        if inp.shape != out.shape:
            analysis["size_change"] = True
            analysis["size_ratio"] = (
                out.shape[0] / inp.shape[0],
                out.shape[1] / inp.shape[1],
            )
            break

    # Analyze color mappings
    for idx, (inp, out) in enumerate(examples):
        unique_in = set(inp.flatten())
        unique_out = set(out.flatten())
        analysis["color_mapping"][str(idx)] = {
            "input_colors": sorted(list(unique_in)),
            "output_colors": sorted(list(unique_out)),
            "new_colors": sorted(list(unique_out - unique_in)),
        }

    # Detect pattern types
    first_in, first_out = examples[0]

    # Check for repetition/tiling
    if analysis["size_change"] and "size_ratio" in analysis:
        ratio = analysis["size_ratio"]
        if ratio[0] == int(ratio[0]) and ratio[1] == int(ratio[1]):
            # Integer scaling - possible tiling
            scale = int(ratio[0])
            if scale > 1:
                # Check if output is tiled version of input
                is_tiling = True
                for i in range(scale):
                    for j in range(scale):
                        tile = first_out[
                            i * first_in.shape[0] : (i + 1) * first_in.shape[0],
                            j * first_in.shape[1] : (j + 1) * first_in.shape[1],
                        ]
                        if not np.array_equal(tile, first_in):
                            is_tiling = False
                            break
                if is_tiling:
                    analysis["spatial_pattern"] = "perfect_tiling"
                else:
                    analysis["spatial_pattern"] = "modified_tiling"

    # Check for object manipulation
    if not analysis["size_change"]:
        # Look for connected components
        def find_objects(grid):
            visited = np.zeros_like(grid, dtype=bool)
            objects = []

            def dfs(i, j, color):
                if (
                    i < 0
                    or i >= grid.shape[0]
                    or j < 0
                    or j >= grid.shape[1]
                    or visited[i, j]
                    or grid[i, j] != color
                ):
                    return []
                visited[i, j] = True
                cells = [(i, j)]
                for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    cells.extend(dfs(i + di, j + dj, color))
                return cells

            for i in range(grid.shape[0]):
                for j in range(grid.shape[1]):
                    if not visited[i, j] and grid[i, j] != 0:
                        obj = dfs(i, j, grid[i, j])
                        if obj:
                            objects.append((grid[i, j], obj))
            return objects

        in_objects = find_objects(first_in)
        out_objects = find_objects(first_out)

        if len(in_objects) > 0 and len(out_objects) > 0:
            analysis["spatial_pattern"] = "object_manipulation"
            analysis["num_input_objects"] = len(in_objects)
            analysis["num_output_objects"] = len(out_objects)

    # Detect rule complexity
    if analysis["spatial_pattern"] == "modified_tiling":
        analysis["complexity"] = "high"
        analysis["rule_type"] = "position_dependent"
    elif analysis["spatial_pattern"] == "object_manipulation":
        analysis["complexity"] = "medium"
        analysis["rule_type"] = "object_based"
    elif analysis["size_change"]:
        analysis["complexity"] = "medium"
        analysis["rule_type"] = "transformation"
    else:
        analysis["complexity"] = "low"
        analysis["rule_type"] = "simple"

    return analysis


def analyze_solver_behavior(solver, examples, test_input):
    """Analyze how the solver attempts to solve a task."""
    # Get solver's internal analysis
    solution = solver.solve(examples, test_input)

    behavior = {
        "method_used": solution.method_used,
        "confidence": solution.confidence,
        "tried_synthesis": "synthesis" in solution.method_used,
        "fell_back_to_tta": "tta" in solution.method_used,
    }

    # Analyze perception
    if hasattr(solver, "perception_analyzer"):
        perception = solver.perception_analyzer.analyze(examples)
        behavior["perception"] = {
            "has_spatial": perception.has_spatial_patterns,
            "has_objects": perception.has_object_patterns,
            "has_arithmetic": perception.has_arithmetic_patterns,
            "has_logical": perception.has_logical_patterns,
        }

    return behavior


def main():
    """Analyze V7 failures in detail."""
    print("=" * 80)
    print("V7 FAILURE ANALYSIS")
    print("=" * 80)

    DATA_DIR = (
        Path(__file__).parent.parent.parent
        / "data"
        / "arc_agi_official"
        / "ARC-AGI"
        / "data"
        / "training"
    )

    # Test tasks that V7 failed on
    failed_tasks = [
        "007bbfb7.json",  # 70.4% - Complex tiling with modifications
        "00d62c1b.json",  # 91.8% - Close but not perfect
        "05f2a901.json",  # 94.5% - Very close
        "0a938d79.json",  # 44.8% - Poor performance
        "0d3d703e.json",  # 66.7% - Moderate performance
        "0ca9ddb6.json",  # 80.2% - Good but not perfect
        "05269061.json",  # 22.4% - Very poor
    ]

    solver = EnhancedARCSolverV7(
        use_synthesis=True,
        synthesis_timeout=8.0,
        use_position_learning=True,
    )

    failure_patterns = []

    for task_name in failed_tasks:
        task_path = DATA_DIR / task_name
        if not task_path.exists():
            continue

        print(f"\n{'='*60}")
        print(f"Task: {task_name.replace('.json', '')}")
        print("=" * 60)

        task = load_arc_task(task_path)

        # Extract examples
        train_examples = [
            (np.array(ex["input"]), np.array(ex["output"])) for ex in task["train"]
        ]

        test_example = task["test"][0]
        test_input = np.array(test_example["input"])
        expected_output = np.array(test_example["output"])

        # Analyze the pattern
        pattern = analyze_task_pattern(train_examples)
        print("\nPattern Analysis:")
        print(f"  Size change: {pattern['size_change']}")
        if "size_ratio" in pattern:
            print(f"  Size ratio: {pattern['size_ratio']}")
        print(f"  Spatial pattern: {pattern['spatial_pattern']}")
        print(f"  Rule type: {pattern['rule_type']}")
        print(f"  Complexity: {pattern['complexity']}")

        # Show color analysis
        print("\nColor Analysis:")
        for ex_idx, colors in pattern["color_mapping"].items():
            print(f"  Example {ex_idx}:")
            print(f"    Input colors: {colors['input_colors']}")
            print(f"    Output colors: {colors['output_colors']}")
            if colors["new_colors"]:
                print(f"    New colors: {colors['new_colors']}")

        # Analyze solver behavior
        print("\nSolver Behavior:")
        behavior = analyze_solver_behavior(solver, train_examples, test_input)
        print(f"  Method used: {behavior['method_used']}")
        print(f"  Confidence: {behavior['confidence']:.2f}")
        print(f"  Tried synthesis: {behavior['tried_synthesis']}")
        print(f"  Fell back to TTA: {behavior['fell_back_to_tta']}")

        if "perception" in behavior:
            print("\nPerception Analysis:")
            for key, value in behavior["perception"].items():
                print(f"    {key}: {value}")

        # Get actual solution
        solution = solver.solve(train_examples, test_input)

        # Calculate accuracy
        if solution.output_grid.shape == expected_output.shape:
            accuracy = (
                np.sum(solution.output_grid == expected_output) / expected_output.size
            )
            print(f"\nAccuracy: {accuracy:.1%}")

            # Show where errors occur
            if accuracy < 1.0:
                errors = solution.output_grid != expected_output
                error_positions = np.argwhere(errors)
                if len(error_positions) > 0:
                    print(f"  Errors at {len(error_positions)} positions")
                    # Sample a few error positions
                    for i, (r, c) in enumerate(error_positions[:5]):
                        print(
                            f"    ({r},{c}): expected {expected_output[r,c]}, "
                            f"got {solution.output_grid[r,c]}"
                        )
                    if len(error_positions) > 5:
                        print(f"    ... and {len(error_positions)-5} more")
        else:
            print(f"\nShape mismatch:")
            print(f"  Expected: {expected_output.shape}")
            print(f"  Got: {solution.output_grid.shape}")

        # Record failure pattern
        failure_patterns.append(
            {
                "task": task_name.replace(".json", ""),
                "pattern_type": pattern["spatial_pattern"],
                "rule_type": pattern["rule_type"],
                "complexity": pattern["complexity"],
                "accuracy": accuracy
                if solution.output_grid.shape == expected_output.shape
                else 0.0,
                "method_tried": behavior["method_used"],
                "size_change": pattern["size_change"],
            }
        )

    # Summary of failure patterns
    print("\n" + "=" * 80)
    print("FAILURE PATTERN SUMMARY")
    print("=" * 80)

    # Group by pattern type
    pattern_groups = {}
    for fp in failure_patterns:
        pt = fp["pattern_type"] or "unknown"
        if pt not in pattern_groups:
            pattern_groups[pt] = []
        pattern_groups[pt].append(fp)

    print("\nFailures by Pattern Type:")
    for pattern_type, tasks in pattern_groups.items():
        avg_acc = sum(t["accuracy"] for t in tasks) / len(tasks)
        print(f"\n{pattern_type}:")
        print(f"  Tasks: {len(tasks)}")
        print(f"  Average accuracy: {avg_acc:.1%}")
        for t in tasks:
            print(f"    - {t['task']}: {t['accuracy']:.1%} ({t['method_tried']})")

    # Group by complexity
    complexity_groups = {}
    for fp in failure_patterns:
        c = fp["complexity"] or "unknown"
        if c not in complexity_groups:
            complexity_groups[c] = []
        complexity_groups[c].append(fp)

    print("\nFailures by Complexity:")
    for complexity, tasks in complexity_groups.items():
        avg_acc = sum(t["accuracy"] for t in tasks) / len(tasks)
        print(f"\n{complexity}:")
        print(f"  Tasks: {len(tasks)}")
        print(f"  Average accuracy: {avg_acc:.1%}")

    # Key insights
    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)

    print("\n1. MODIFIED TILING PATTERNS:")
    print("   - Task 007bbfb7 has position-dependent modifications")
    print("   - V7 achieves 70.4% but can't capture all tile variations")
    print("   - Need: Better position-rule learning or enumeration")

    print("\n2. HIGH-ACCURACY FAILURES (>90%):")
    print("   - Tasks 00d62c1b (91.8%) and 05f2a901 (94.5%)")
    print("   - Very close but missing subtle patterns")
    print("   - Need: Fine-grained pattern detection or correction")

    print("\n3. POOR PERFORMANCE TASKS (<50%):")
    print("   - Tasks 0a938d79 (44.8%) and 05269061 (22.4%)")
    print("   - Fundamental pattern not captured")
    print("   - Need: New pattern types or better perception")

    print("\n4. SYNTHESIS EFFECTIVENESS:")
    print("   - Only succeeded on 1/8 tasks (08ed6ac7)")
    print("   - Early synthesis attempts often fail (confidence 0.00)")
    print("   - Need: Better perception-guided synthesis or longer search")

    print("\n5. TTA FALLBACK:")
    print("   - Still used as last resort in multiple tasks")
    print("   - Indicates gaps in deterministic methods")
    print("   - Need: Expand pattern library or improve learning")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
